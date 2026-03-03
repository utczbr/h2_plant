#!/usr/bin/env python3
"""
Backfill OPEX feed columns in chunked Parquet history.

Adds/recomputes:
  - biogas_feed_kg_step
  - water_makeup_kg_step

Water reconstruction basis implemented:
  - source_equivalent:
      Reconstruct Water_Source intake from UltraPure_Tank control zones with
      one-step lag continuity across chunk boundaries.

Usage:
  python3 tools/reconstruct_opex_feed_columns.py \
    --input-history-dir scenarios/20_years/history_chunks \
    --output-history-dir scenarios/20_years_derived/history_chunks \
    --topology-yaml scenarios/plant_topology.yaml \
    --water-basis source_equivalent
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


LOGGER = logging.getLogger(__name__)
CHUNK_RE = re.compile(r"^chunk_(\d+)\.parquet$")
DEFAULT_DT_HOURS = 1.0 / 60.0


@dataclass(frozen=True)
class WaterModelParams:
    nominal_production_kg_h: float
    source_max_flow_kg_h: float


def _safe_float(value: Any, default: float) -> float:
    try:
        out = float(value)
        if np.isfinite(out):
            return out
    except (TypeError, ValueError):
        pass
    return default


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill biogas_feed_kg_step and water_makeup_kg_step in parquet chunks."
    )
    parser.add_argument(
        "--input-history-dir",
        required=True,
        help="Directory containing chunk_*.parquet input files.",
    )
    parser.add_argument(
        "--output-history-dir",
        required=True,
        help="Directory where reconstructed chunk_*.parquet files will be written.",
    )
    parser.add_argument(
        "--topology-yaml",
        required=True,
        help="Path to plant_topology.yaml to read Water_Source/UltraPure_Tank parameters.",
    )
    parser.add_argument(
        "--water-basis",
        default="source_equivalent",
        choices=["source_equivalent"],
        help="Water reconstruction basis. Default: source_equivalent.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty output-history-dir (replaces existing chunk_*.parquet).",
    )
    parser.add_argument(
        "--audit-json",
        default=None,
        help="Optional explicit path for reconstruction_audit.json (default: <output-history-dir>/../reconstruction_audit.json).",
    )
    return parser.parse_args()


def _sorted_chunk_files(history_dir: Path) -> List[Path]:
    chunk_files = list(history_dir.glob("chunk_*.parquet"))
    if not chunk_files:
        return []

    def _sort_key(path: Path) -> Tuple[int, int, str]:
        match = CHUNK_RE.match(path.name)
        if match is None:
            return (1, 0, path.name)
        return (0, int(match.group(1)), path.name)

    return sorted(chunk_files, key=_sort_key)


def _resolve_output_dir(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_chunks = list(output_dir.glob("chunk_*.parquet"))
    if existing_chunks and not overwrite:
        raise ValueError(
            f"Output history dir already contains {len(existing_chunks)} chunk files: {output_dir}. "
            "Pass --overwrite to replace."
        )
    if overwrite:
        for chunk in existing_chunks:
            chunk.unlink()


def _find_node(nodes: List[Dict[str, Any]], node_id: str) -> Optional[Dict[str, Any]]:
    for node in nodes:
        if str(node.get("id", "")).strip() == node_id:
            return node
    return None


def _load_water_model_params(topology_yaml: Path) -> Tuple[WaterModelParams, Dict[str, Any]]:
    if not topology_yaml.exists():
        raise FileNotFoundError(f"Topology YAML not found: {topology_yaml}")

    with open(topology_yaml, "r", encoding="utf-8") as handle:
        topology = yaml.safe_load(handle) or {}

    nodes = topology.get("nodes") or []
    if not isinstance(nodes, list):
        nodes = []

    tank_node = _find_node(nodes, "UltraPure_Tank")
    water_source_node = _find_node(nodes, "Water_Source")

    tank_params = (tank_node or {}).get("params") or {}
    source_params = (water_source_node or {}).get("params") or {}

    nominal_production_kg_h = _safe_float(
        tank_params.get("nominal_production_kg_h"),
        default=10000.0,
    )
    source_max_flow_kg_h = _safe_float(
        source_params.get("flow_rate_kg_h"),
        default=12000.0,
    )

    if nominal_production_kg_h <= 0:
        nominal_production_kg_h = 10000.0
    if source_max_flow_kg_h <= 0:
        source_max_flow_kg_h = 12000.0

    params = WaterModelParams(
        nominal_production_kg_h=float(nominal_production_kg_h),
        source_max_flow_kg_h=float(source_max_flow_kg_h),
    )

    provenance = {
        "tank_node_found": tank_node is not None,
        "water_source_node_found": water_source_node is not None,
        "tank_nominal_from_yaml": tank_params.get("nominal_production_kg_h", None),
        "water_source_max_from_yaml": source_params.get("flow_rate_kg_h", None),
    }
    return params, provenance


def _infer_dt_hours(minute_series: pd.Series) -> Tuple[float, bool]:
    minute_values = pd.to_numeric(minute_series, errors="coerce").dropna().to_numpy(dtype=float)
    if minute_values.size < 2:
        return DEFAULT_DT_HOURS, True
    diffs = np.diff(minute_values)
    positive = diffs[diffs > 0]
    if positive.size == 0:
        return DEFAULT_DT_HOURS, True
    dt_h = float(np.median(positive) / 60.0)
    if dt_h <= 0 or not np.isfinite(dt_h):
        return DEFAULT_DT_HOURS, True
    return dt_h, False


def _map_zone_to_requested_rate(
    zones: np.ndarray,
    nominal_kg_h: float,
) -> Tuple[np.ndarray, int]:
    requested = np.full(zones.shape, nominal_kg_h, dtype=float)
    requested[zones == 1] = 1.2 * nominal_kg_h
    requested[zones == 2] = 1.0 * nominal_kg_h
    requested[zones == 3] = 0.0

    known_mask = (zones == 1) | (zones == 2) | (zones == 3)
    unknown_count = int((~known_mask).sum())
    return requested, unknown_count


def _lag_requested_rate(
    requested: np.ndarray,
    prev_requested_last: Optional[float],
) -> Tuple[np.ndarray, Optional[float]]:
    if requested.size == 0:
        return requested, prev_requested_last

    lagged = np.empty_like(requested, dtype=float)
    lagged[0] = requested[0] if prev_requested_last is None else prev_requested_last
    if requested.size > 1:
        lagged[1:] = requested[:-1]
    return lagged, float(requested[-1])


def reconstruct_history_chunks(
    *,
    input_history_dir: Path,
    output_history_dir: Path,
    topology_yaml: Path,
    water_basis: str,
    overwrite: bool,
    audit_json_path: Optional[Path] = None,
) -> Dict[str, Any]:
    if water_basis != "source_equivalent":
        raise ValueError(f"Unsupported water basis: {water_basis}")

    if not input_history_dir.exists() or not input_history_dir.is_dir():
        raise FileNotFoundError(f"Input history dir not found: {input_history_dir}")

    chunk_files = _sorted_chunk_files(input_history_dir)
    if not chunk_files:
        raise FileNotFoundError(f"No chunk_*.parquet files found in {input_history_dir}")

    _resolve_output_dir(output_history_dir, overwrite=overwrite)

    water_params, water_param_provenance = _load_water_model_params(topology_yaml)
    zone_map = {
        "1": 1.2 * water_params.nominal_production_kg_h,
        "2": 1.0 * water_params.nominal_production_kg_h,
        "3": 0.0,
    }

    summary: Dict[str, Any] = {
        "chunk_count": 0,
        "row_count_total": 0,
        "fallback_dt_chunk_count": 0,
        "missing_minute_chunk_count": 0,
        "unknown_zone_count_total": 0,
        "nan_count_total": {"biogas_feed_kg_step": 0, "water_makeup_kg_step": 0},
        "negative_count_total": {"biogas_feed_kg_step": 0, "water_makeup_kg_step": 0},
        "total_kg": {"biogas_feed_kg_step": 0.0, "water_makeup_kg_step": 0.0},
        "overwritten_preexisting_columns": {
            "biogas_feed_kg_step": {"rows_compared": 0, "mae": 0.0, "max_abs_err": 0.0},
            "water_makeup_kg_step": {"rows_compared": 0, "mae": 0.0, "max_abs_err": 0.0},
        },
        "compatibility_backfills": {
            "electricity_consumption_kwh_step": 0,
            "pem_electricity_consumption_kwh_step": 0,
            "soec_electricity_consumption_kwh_step": 0,
            "bop_electricity_consumption_kwh_step": 0,
            "sold_energy_mwh_step": 0,
            "cooling_duty_kwh_th_step": 0,
        },
        "chunk_stats": [],
    }

    prev_requested_rate: Optional[float] = None

    for chunk_path in chunk_files:
        table = pq.read_table(chunk_path)
        df = table.to_pandas()
        rows_in = int(len(df))

        original_biogas = (
            pd.to_numeric(df["biogas_feed_kg_step"], errors="coerce")
            if "biogas_feed_kg_step" in df.columns
            else None
        )
        original_water = (
            pd.to_numeric(df["water_makeup_kg_step"], errors="coerce")
            if "water_makeup_kg_step" in df.columns
            else None
        )

        if "minute" in df.columns:
            dt_h, used_fallback_dt = _infer_dt_hours(df["minute"])
        else:
            dt_h, used_fallback_dt = DEFAULT_DT_HOURS, True
            summary["missing_minute_chunk_count"] += 1
        if used_fallback_dt:
            summary["fallback_dt_chunk_count"] += 1

        if "Biogas_Source_outlet_mass_flow_kg_h" not in df.columns:
            raise ValueError(
                f"Missing required column 'Biogas_Source_outlet_mass_flow_kg_h' in {chunk_path.name}"
            )
        biogas_flow_kg_h = (
            pd.to_numeric(df["Biogas_Source_outlet_mass_flow_kg_h"], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        biogas_step_kg = np.clip(biogas_flow_kg_h * dt_h, a_min=0.0, a_max=None)

        if "UltraPure_Tank_control_zone_int" not in df.columns:
            raise ValueError(
                f"Missing required column 'UltraPure_Tank_control_zone_int' in {chunk_path.name}"
            )
        zone_values = (
            pd.to_numeric(df["UltraPure_Tank_control_zone_int"], errors="coerce")
            .fillna(2.0)
            .astype(int)
            .to_numpy()
        )
        requested_rate_kg_h, unknown_zone_count = _map_zone_to_requested_rate(
            zone_values,
            nominal_kg_h=water_params.nominal_production_kg_h,
        )
        lagged_rate_kg_h, prev_requested_rate = _lag_requested_rate(
            requested=requested_rate_kg_h,
            prev_requested_last=prev_requested_rate,
        )
        lagged_rate_kg_h = np.clip(
            lagged_rate_kg_h,
            a_min=0.0,
            a_max=water_params.source_max_flow_kg_h,
        )
        water_step_kg = lagged_rate_kg_h * dt_h

        df["biogas_feed_kg_step"] = biogas_step_kg.astype(np.float64)
        df["water_makeup_kg_step"] = water_step_kg.astype(np.float64)

        # Legacy schema compatibility for OPEX config:
        # If canonical step columns are missing, derive them from existing power/duty tags.
        pem_kwh_step = (
            pd.to_numeric(df["P_pem_grid_mw"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            * 1000.0
            * dt_h
            if "P_pem_grid_mw" in df.columns
            else np.zeros(rows_in, dtype=float)
        )
        soec_kwh_step = (
            pd.to_numeric(df["P_soec_grid_mw"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            * 1000.0
            * dt_h
            if "P_soec_grid_mw" in df.columns
            else np.zeros(rows_in, dtype=float)
        )
        bop_kwh_step = (
            pd.to_numeric(df["P_bop_grid_usage_mw"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            * 1000.0
            * dt_h
            if "P_bop_grid_usage_mw" in df.columns
            else np.zeros(rows_in, dtype=float)
        )
        total_kwh_step = pem_kwh_step + soec_kwh_step + bop_kwh_step

        if "pem_electricity_consumption_kwh_step" not in df.columns:
            df["pem_electricity_consumption_kwh_step"] = np.clip(pem_kwh_step, a_min=0.0, a_max=None).astype(np.float64)
            summary["compatibility_backfills"]["pem_electricity_consumption_kwh_step"] += 1
        if "soec_electricity_consumption_kwh_step" not in df.columns:
            df["soec_electricity_consumption_kwh_step"] = np.clip(soec_kwh_step, a_min=0.0, a_max=None).astype(np.float64)
            summary["compatibility_backfills"]["soec_electricity_consumption_kwh_step"] += 1
        if "bop_electricity_consumption_kwh_step" not in df.columns:
            df["bop_electricity_consumption_kwh_step"] = np.clip(bop_kwh_step, a_min=0.0, a_max=None).astype(np.float64)
            summary["compatibility_backfills"]["bop_electricity_consumption_kwh_step"] += 1
        if "electricity_consumption_kwh_step" not in df.columns:
            df["electricity_consumption_kwh_step"] = np.clip(total_kwh_step, a_min=0.0, a_max=None).astype(np.float64)
            summary["compatibility_backfills"]["electricity_consumption_kwh_step"] += 1

        if "sold_energy_mwh_step" not in df.columns:
            sold_mw = (
                pd.to_numeric(df["P_sold"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                if "P_sold" in df.columns
                else np.zeros(rows_in, dtype=float)
            )
            df["sold_energy_mwh_step"] = np.clip(sold_mw, a_min=0.0, a_max=None).astype(np.float64) * dt_h
            summary["compatibility_backfills"]["sold_energy_mwh_step"] += 1

        if "cooling_duty_kwh_th_step" not in df.columns:
            if "total_cooling_duty_kw" in df.columns:
                cooling_kw = pd.to_numeric(df["total_cooling_duty_kw"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            else:
                glycol = (
                    pd.to_numeric(df["cooling_manager_glycol_duty_kw"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                    if "cooling_manager_glycol_duty_kw" in df.columns
                    else np.zeros(rows_in, dtype=float)
                )
                cw = (
                    pd.to_numeric(df["cooling_manager_cw_duty_kw"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                    if "cooling_manager_cw_duty_kw" in df.columns
                    else np.zeros(rows_in, dtype=float)
                )
                cooling_kw = glycol + cw
            df["cooling_duty_kwh_th_step"] = np.clip(cooling_kw, a_min=0.0, a_max=None).astype(np.float64) * dt_h
            summary["compatibility_backfills"]["cooling_duty_kwh_th_step"] += 1

        if len(df) != rows_in:
            raise RuntimeError(
                f"Row count changed unexpectedly for {chunk_path.name}: {rows_in} -> {len(df)}"
            )

        out_chunk_path = output_history_dir / chunk_path.name
        out_table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(out_table, out_chunk_path)

        biogas_series = pd.to_numeric(df["biogas_feed_kg_step"], errors="coerce")
        water_series = pd.to_numeric(df["water_makeup_kg_step"], errors="coerce")

        biogas_nan_count = int(biogas_series.isna().sum())
        water_nan_count = int(water_series.isna().sum())
        biogas_negative_count = int((biogas_series < 0).sum())
        water_negative_count = int((water_series < 0).sum())

        summary["chunk_count"] += 1
        summary["row_count_total"] += rows_in
        summary["unknown_zone_count_total"] += unknown_zone_count
        summary["nan_count_total"]["biogas_feed_kg_step"] += biogas_nan_count
        summary["nan_count_total"]["water_makeup_kg_step"] += water_nan_count
        summary["negative_count_total"]["biogas_feed_kg_step"] += biogas_negative_count
        summary["negative_count_total"]["water_makeup_kg_step"] += water_negative_count
        summary["total_kg"]["biogas_feed_kg_step"] += float(np.nansum(biogas_series.to_numpy(dtype=float)))
        summary["total_kg"]["water_makeup_kg_step"] += float(np.nansum(water_series.to_numpy(dtype=float)))

        if original_biogas is not None:
            comp = np.abs(
                biogas_series.fillna(0.0).to_numpy(dtype=float)
                - original_biogas.fillna(0.0).to_numpy(dtype=float)
            )
            rows_comp = int(comp.size)
            if rows_comp > 0:
                weighted = summary["overwritten_preexisting_columns"]["biogas_feed_kg_step"]
                prev_rows = weighted["rows_compared"]
                weighted["rows_compared"] = prev_rows + rows_comp
                weighted["mae"] = (
                    (weighted["mae"] * prev_rows + float(comp.mean()) * rows_comp)
                    / weighted["rows_compared"]
                )
                weighted["max_abs_err"] = max(weighted["max_abs_err"], float(comp.max()))

        if original_water is not None:
            comp = np.abs(
                water_series.fillna(0.0).to_numpy(dtype=float)
                - original_water.fillna(0.0).to_numpy(dtype=float)
            )
            rows_comp = int(comp.size)
            if rows_comp > 0:
                weighted = summary["overwritten_preexisting_columns"]["water_makeup_kg_step"]
                prev_rows = weighted["rows_compared"]
                weighted["rows_compared"] = prev_rows + rows_comp
                weighted["mae"] = (
                    (weighted["mae"] * prev_rows + float(comp.mean()) * rows_comp)
                    / weighted["rows_compared"]
                )
                weighted["max_abs_err"] = max(weighted["max_abs_err"], float(comp.max()))

        summary["chunk_stats"].append(
            {
                "chunk": chunk_path.name,
                "rows": rows_in,
                "dt_h": dt_h,
                "used_fallback_dt": used_fallback_dt,
                "biogas_sum_kg": float(np.nansum(biogas_series.to_numpy(dtype=float))),
                "water_sum_kg": float(np.nansum(water_series.to_numpy(dtype=float))),
                "unknown_zone_rows": unknown_zone_count,
            }
        )

    if audit_json_path is None:
        audit_json_path = output_history_dir.parent / "reconstruction_audit.json"
    audit_json_path.parent.mkdir(parents=True, exist_ok=True)

    audit = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "tool": "reconstruct_opex_feed_columns.py",
        "inputs": {
            "input_history_dir": str(input_history_dir.resolve()),
            "output_history_dir": str(output_history_dir.resolve()),
            "topology_yaml": str(topology_yaml.resolve()),
            "water_basis": water_basis,
            "overwrite": overwrite,
        },
        "model_parameters": {
            "water": {
                "nominal_production_kg_h": water_params.nominal_production_kg_h,
                "source_max_flow_kg_h": water_params.source_max_flow_kg_h,
                "zone_map_requested_rate_kg_h": zone_map,
                "lag_policy": "one_step_lag_with_cross_chunk_continuity",
            },
            "biogas": {
                "formula": "biogas_feed_kg_step = max(0, Biogas_Source_outlet_mass_flow_kg_h * dt_h)"
            },
            "dt_policy": "median(positive diff(minute))/60; fallback 1/60",
            "water_parameter_provenance": water_param_provenance,
        },
        "summary": summary,
    }

    with open(audit_json_path, "w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)

    LOGGER.info("Reconstruction complete.")
    LOGGER.info("  Chunks processed: %d", summary["chunk_count"])
    LOGGER.info("  Rows processed:   %d", summary["row_count_total"])
    LOGGER.info(
        "  Totals (kg):      biogas=%0.3f, water=%0.3f",
        summary["total_kg"]["biogas_feed_kg_step"],
        summary["total_kg"]["water_makeup_kg_step"],
    )
    LOGGER.info("  Audit JSON:       %s", audit_json_path)

    return audit


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args()

    input_history_dir = Path(args.input_history_dir).resolve()
    output_history_dir = Path(args.output_history_dir).resolve()
    topology_yaml = Path(args.topology_yaml).resolve()
    audit_json_path = Path(args.audit_json).resolve() if args.audit_json else None

    try:
        reconstruct_history_chunks(
            input_history_dir=input_history_dir,
            output_history_dir=output_history_dir,
            topology_yaml=topology_yaml,
            water_basis=args.water_basis,
            overwrite=args.overwrite,
            audit_json_path=audit_json_path,
        )
    except Exception as exc:
        LOGGER.error("Backfill failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
