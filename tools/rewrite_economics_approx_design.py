#!/usr/bin/env python3
"""
Rewrite historical CAPEX/OPEX reports into an approximate design-mode format.

This bridge tool is optimized for speed:
1) Scans Parquet metadata across all chunks (cheap).
2) Reads only sampled chunks per year to estimate degradation ramp.
3) Rewrites CAPEX/OPEX reports with yearly OPEX arrays for graph tooling.

Outputs:
  - <output-dir>/capex_report.json
  - <output-dir>/opex_report.json
  - <output-dir>/approximation_audit.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml


LOGGER = logging.getLogger(__name__)
HOURS_PER_YEAR = 8760.0
MINUTES_PER_YEAR = 525600.0
DEFAULT_DT_MIN = 1.0
EPS = 1e-12
CHUNK_RE = re.compile(r"^chunk_(\d+)\.parquet$")

PRIMARY_ELEC_COLS = (
    "pem_electricity_consumption_kwh_step",
    "soec_electricity_consumption_kwh_step",
)
FALLBACK_ELEC_COL = "electricity_consumption_kwh_step"
H2_COLS = ("H2_pem_kg", "H2_soec_kg")


@dataclass
class ChunkMeta:
    path: Path
    index: int
    rows: int
    minute_min: Optional[float]
    minute_max: Optional[float]
    duration_h: float
    year_idx: Optional[int] = None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if np.isfinite(out):
            return out
    except (TypeError, ValueError):
        pass
    return default


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fast bridge script to rewrite historical CAPEX/OPEX reports into "
            "approximate design-mode format with yearly OPEX arrays."
        )
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing source capex_report.json and opex_report.json.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where rewritten report JSON files will be written.",
    )
    parser.add_argument(
        "--history-dir",
        default=None,
        help="Optional history source directory (history_chunks or its parent).",
    )
    parser.add_argument(
        "--samples-per-year",
        type=int,
        default=5,
        help="Number of sampled chunks per year for degradation ramp (default: 5).",
    )
    parser.add_argument(
        "--design-capex-reference",
        default=None,
        help="Optional reference capex_report.json (design-mode) for scale derivation.",
    )
    parser.add_argument(
        "--design-capex-factor",
        type=float,
        default=1.0,
        help="Fallback global CAPEX scale when no reference is provided (default: 1.0).",
    )
    parser.add_argument(
        "--opex-config",
        default=str(Path("scenarios") / "Economics" / "opex_config.yaml"),
        help="Path to opex_config.yaml to identify FCI-based items.",
    )
    parser.add_argument(
        "--project-years",
        type=int,
        default=None,
        help="Optional explicit year count for synthesized yearly series.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output capex/opex/audit JSON files.",
    )
    parser.add_argument(
        "--audit-json",
        default=None,
        help="Optional explicit path for approximation_audit.json.",
    )
    return parser.parse_args()


def _sorted_chunk_files(history_chunks_dir: Path) -> List[Path]:
    chunk_files = list(history_chunks_dir.glob("chunk_*.parquet"))
    if not chunk_files:
        return []

    def _key(path: Path) -> Tuple[int, int, str]:
        match = CHUNK_RE.match(path.name)
        if match is None:
            return (1, 0, path.name)
        return (0, int(match.group(1)), path.name)

    return sorted(chunk_files, key=_key)


def _resolve_history_chunks(history_dir: Optional[str]) -> Optional[Path]:
    if not history_dir:
        return None
    base = Path(history_dir).resolve()
    if base.is_dir() and base.name.lower() == "history_chunks":
        if list(base.glob("chunk_*.parquet")):
            return base
    if base.is_dir():
        candidate = base / "history_chunks"
        if candidate.is_dir() and list(candidate.glob("chunk_*.parquet")):
            return candidate
    return None


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(payload)}")
    return payload


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _ensure_required_keys(payload: Dict[str, Any], required: Sequence[str], label: str) -> None:
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Missing required {label} keys: {', '.join(missing)}")


def _load_fci_factor_item_names(opex_config_path: Path) -> Tuple[set[str], Dict[str, Any]]:
    if not opex_config_path.exists():
        return set(), {"config_found": False, "fci_item_names": []}

    with open(opex_config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}

    item_names: set[str] = set()
    for item in (cfg.get("opex_items") or []):
        if not isinstance(item, dict):
            continue
        strategy = str(item.get("strategy", "")).strip().lower()
        base_reference = str(item.get("base_reference", "")).strip().upper()
        name = str(item.get("name", "")).strip()
        if strategy == "factor" and base_reference == "FCI" and name:
            item_names.add(name.lower())

    provenance = {
        "config_found": True,
        "config_path": str(opex_config_path),
        "fci_item_names": sorted(item_names),
    }
    return item_names, provenance


def _find_minute_col_index(parquet_file: pq.ParquetFile) -> Optional[int]:
    schema = parquet_file.metadata.schema
    for idx in range(len(schema)):
        if schema.column(idx).name == "minute":
            return idx
    return None


def _collect_chunk_metadata(chunk_files: List[Path]) -> Dict[str, Any]:
    chunks: List[ChunkMeta] = []
    minute_global_min: Optional[float] = None
    minute_global_max: Optional[float] = None
    dt_estimates_min: List[float] = []
    minute_stats_missing = 0
    col_names: List[str] = []

    for order_idx, chunk_path in enumerate(chunk_files):
        parquet_file = pq.ParquetFile(chunk_path)
        metadata = parquet_file.metadata
        row_count = int(metadata.num_rows)
        if order_idx == 0:
            try:
                col_names = pq.read_schema(chunk_path).names
            except Exception:
                col_names = []

        minute_idx = _find_minute_col_index(parquet_file)
        chunk_min: Optional[float] = None
        chunk_max: Optional[float] = None
        if minute_idx is not None:
            for rg_idx in range(metadata.num_row_groups):
                stats = metadata.row_group(rg_idx).column(minute_idx).statistics
                if stats is None or stats.min is None or stats.max is None:
                    continue
                rg_min = _safe_float(stats.min, default=np.nan)
                rg_max = _safe_float(stats.max, default=np.nan)
                if not np.isfinite(rg_min) or not np.isfinite(rg_max):
                    continue
                chunk_min = rg_min if chunk_min is None else min(chunk_min, rg_min)
                chunk_max = rg_max if chunk_max is None else max(chunk_max, rg_max)

        dt_est_min: Optional[float] = None
        if (
            chunk_min is not None
            and chunk_max is not None
            and row_count > 1
            and chunk_max > chunk_min
        ):
            dt_est_min = (chunk_max - chunk_min) / float(max(row_count - 1, 1))
            if dt_est_min > 0 and np.isfinite(dt_est_min):
                dt_estimates_min.append(float(dt_est_min))

        if chunk_min is None or chunk_max is None:
            minute_stats_missing += 1

        if chunk_min is not None:
            minute_global_min = chunk_min if minute_global_min is None else min(minute_global_min, chunk_min)
        if chunk_max is not None:
            minute_global_max = chunk_max if minute_global_max is None else max(minute_global_max, chunk_max)

        dt_for_duration = dt_est_min if dt_est_min is not None else DEFAULT_DT_MIN
        if chunk_min is not None and chunk_max is not None and chunk_max >= chunk_min:
            duration_h = (chunk_max - chunk_min + dt_for_duration) / 60.0
        else:
            duration_h = row_count * (DEFAULT_DT_MIN / 60.0)

        chunks.append(
            ChunkMeta(
                path=chunk_path,
                index=order_idx,
                rows=row_count,
                minute_min=chunk_min,
                minute_max=chunk_max,
                duration_h=float(max(duration_h, 0.0)),
            )
        )

    dt_global_min = float(np.median(dt_estimates_min)) if dt_estimates_min else DEFAULT_DT_MIN
    if not np.isfinite(dt_global_min) or dt_global_min <= 0:
        dt_global_min = DEFAULT_DT_MIN

    year_count = 0
    year_hours: List[float] = []
    if minute_global_min is not None and minute_global_max is not None:
        sim_end_min = float(minute_global_max) + dt_global_min
        span_min = max(0.0, sim_end_min - float(minute_global_min))
        year_count = int(math.ceil(span_min / MINUTES_PER_YEAR)) if span_min > 0 else 1
        for y in range(year_count):
            y_start = float(minute_global_min) + y * MINUTES_PER_YEAR
            y_end = y_start + MINUTES_PER_YEAR
            overlap = max(0.0, min(sim_end_min, y_end) - max(float(minute_global_min), y_start))
            year_hours.append(overlap / 60.0)

        for chunk in chunks:
            if chunk.minute_min is None or chunk.minute_max is None:
                continue
            minute_mid = 0.5 * (chunk.minute_min + chunk.minute_max)
            y_idx = int(math.floor((minute_mid - float(minute_global_min)) / MINUTES_PER_YEAR))
            chunk.year_idx = max(0, y_idx)

    return {
        "chunks": chunks,
        "column_names": col_names,
        "minute_origin": minute_global_min,
        "minute_max": minute_global_max,
        "dt_global_min": dt_global_min,
        "year_count": year_count,
        "year_hours": year_hours,
        "minute_stats_missing_count": minute_stats_missing,
    }


def _determine_year_count(
    src_opex: Dict[str, Any],
    history_meta: Optional[Dict[str, Any]],
    project_years: Optional[int],
) -> int:
    if project_years is not None and project_years > 0:
        return int(project_years)

    sim_hours = _safe_float(src_opex.get("simulation_hours"), default=0.0)
    if sim_hours > 0:
        return max(1, int(math.ceil(sim_hours / HOURS_PER_YEAR)))

    if history_meta:
        hist_years = int(history_meta.get("year_count") or 0)
        if hist_years > 0:
            return hist_years

    return 1


def _build_year_hours(
    *,
    n_years: int,
    src_opex: Dict[str, Any],
    history_meta: Optional[Dict[str, Any]],
) -> List[float]:
    if n_years <= 0:
        return [HOURS_PER_YEAR]

    if history_meta:
        hist_hours = [float(v) for v in (history_meta.get("year_hours") or [])]
        if hist_hours:
            if len(hist_hours) >= n_years:
                out = hist_hours[:n_years]
            else:
                out = hist_hours + [HOURS_PER_YEAR] * (n_years - len(hist_hours))
            if np.nansum(out) > 0:
                return out

    sim_hours = _safe_float(src_opex.get("simulation_hours"), default=0.0)
    if sim_hours > 0:
        out: List[float] = []
        remaining = sim_hours
        for _ in range(n_years):
            chunk_h = max(0.0, min(HOURS_PER_YEAR, remaining))
            out.append(chunk_h)
            remaining -= chunk_h
        if np.nansum(out) > 0:
            return out

    return [HOURS_PER_YEAR] * n_years


def _assign_chunk_years_if_missing(chunks: List[ChunkMeta], n_years: int) -> None:
    if not chunks or n_years <= 0:
        return
    missing = [chunk for chunk in chunks if chunk.year_idx is None]
    if not missing:
        return
    total = len(chunks)
    for chunk in missing:
        y_idx = int(math.floor((chunk.index / max(total, 1)) * n_years))
        chunk.year_idx = min(max(y_idx, 0), n_years - 1)


def _sample_positions(n: int, k: int) -> List[int]:
    if n <= 0:
        return []
    if k >= n:
        return list(range(n))
    raw = np.linspace(0, n - 1, num=k)
    idx = sorted({int(round(v)) for v in raw})
    if len(idx) < k:
        for candidate in range(n):
            if candidate not in idx:
                idx.append(candidate)
            if len(idx) >= k:
                break
        idx = sorted(idx)
    return idx[:k]


def _allocate_equivalent_annual(
    annual_value: float,
    year_hours: np.ndarray,
    profile_weights: np.ndarray,
) -> np.ndarray:
    annual = float(annual_value)
    n = len(year_hours)
    if n == 0:
        return np.zeros(0, dtype=float)

    hours = np.array(year_hours, dtype=float)
    hours[~np.isfinite(hours)] = 0.0
    hours = np.clip(hours, a_min=0.0, a_max=None)
    total_hours = float(np.nansum(hours))
    target_sum = annual * total_hours / HOURS_PER_YEAR if total_hours > 0 else annual

    weights = np.array(profile_weights, dtype=float)
    weights[~np.isfinite(weights)] = 0.0
    weights = np.clip(weights, a_min=0.0, a_max=None)
    if float(np.nansum(weights)) <= 0:
        weights = hours.copy()
    if float(np.nansum(weights)) <= 0:
        out = np.zeros(n, dtype=float)
        out[0] = target_sum
        return out
    return target_sum * (weights / float(np.nansum(weights)))


def _equivalent_annual(series: np.ndarray, year_hours: np.ndarray) -> float:
    total_hours = float(np.nansum(year_hours))
    total_val = float(np.nansum(series))
    if total_hours > 0:
        return total_val / total_hours * HOURS_PER_YEAR
    return total_val


def _derive_capex_scales(
    src_capex: Dict[str, Any],
    ref_capex: Optional[Dict[str, Any]],
    fallback_factor: float,
) -> Dict[str, float]:
    fallback = _safe_float(fallback_factor, default=1.0)
    if fallback <= 0:
        fallback = 1.0

    keys = {
        "base": "total_installed_cost",
        "low": "total_installed_cost_low",
        "high": "total_installed_cost_high",
    }
    scales: Dict[str, float] = {}
    for variant, key in keys.items():
        if ref_capex is not None:
            src_val = _safe_float(src_capex.get(key), default=0.0)
            ref_val = _safe_float(ref_capex.get(key), default=0.0)
            if src_val > 0 and ref_val > 0:
                scales[variant] = ref_val / src_val
                continue
        scales[variant] = fallback
    return scales


def _rewrite_capex_report(
    src_capex: Dict[str, Any],
    ref_capex: Optional[Dict[str, Any]],
    global_scales: Dict[str, float],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    ref_entry_by_tag: Dict[str, Dict[str, Any]] = {}
    if ref_capex is not None:
        for entry in ref_capex.get("entries", []):
            if not isinstance(entry, dict):
                continue
            tag = str(entry.get("tag", "")).strip()
            if tag:
                ref_entry_by_tag[tag] = entry

    entry_scales: Dict[str, Dict[str, float]] = {}
    new_entries: List[Dict[str, Any]] = []
    for src_entry in (src_capex.get("entries") or []):
        if not isinstance(src_entry, dict):
            continue
        tag = str(src_entry.get("tag", "")).strip()
        ref_entry = ref_entry_by_tag.get(tag)

        base_scale = global_scales["base"]
        low_scale = global_scales["low"]
        high_scale = global_scales["high"]

        src_cbm = _safe_float(src_entry.get("C_BM"), default=0.0)
        if ref_entry is not None:
            ref_cbm = _safe_float(ref_entry.get("C_BM"), default=0.0)
            if src_cbm > 0 and ref_cbm > 0:
                base_scale = ref_cbm / src_cbm
            src_low = _safe_float(src_entry.get("C_BM_low"), default=0.0)
            ref_low = _safe_float(ref_entry.get("C_BM_low"), default=0.0)
            if src_low > 0 and ref_low > 0:
                low_scale = ref_low / src_low
            src_high = _safe_float(src_entry.get("C_BM_high"), default=0.0)
            ref_high = _safe_float(ref_entry.get("C_BM_high"), default=0.0)
            if src_high > 0 and ref_high > 0:
                high_scale = ref_high / src_high

        entry_scales[tag or f"entry_{len(new_entries)}"] = {
            "base": float(base_scale),
            "low": float(low_scale),
            "high": float(high_scale),
        }

        new_entry = dict(src_entry)
        cp0 = src_entry.get("C_p0")
        cbm = src_entry.get("C_BM")
        cbm_low = src_entry.get("C_BM_low")
        cbm_high = src_entry.get("C_BM_high")
        if cp0 is not None:
            new_entry["C_p0"] = _safe_float(cp0) * base_scale
        if cbm is not None:
            new_entry["C_BM"] = _safe_float(cbm) * base_scale
        if cbm_low is not None:
            new_entry["C_BM_low"] = _safe_float(cbm_low) * low_scale
        if cbm_high is not None:
            new_entry["C_BM_high"] = _safe_float(cbm_high) * high_scale

        if ref_entry is not None and ref_entry.get("design_capacity") is not None:
            new_entry["design_capacity"] = ref_entry.get("design_capacity")

        src_cap_source = str(src_entry.get("capacity_source", "")).strip()
        new_entry["capacity_source"] = (
            f"{src_cap_source} [approx_design_bridge]"
            if src_cap_source
            else "approx_design_bridge"
        )
        new_entries.append(new_entry)

    entry_by_tag = {str(entry.get("tag", "")): entry for entry in new_entries}
    src_blocks = src_capex.get("block_summaries") or []
    new_blocks: List[Dict[str, Any]] = []
    for src_block in src_blocks:
        if not isinstance(src_block, dict):
            continue
        block_name = str(src_block.get("block_name", ""))
        tags = list(src_block.get("equipment_tags") or [])

        equip_base = sum(_safe_float(entry_by_tag.get(tag, {}).get("C_BM"), 0.0) for tag in tags)
        equip_low = sum(_safe_float(entry_by_tag.get(tag, {}).get("C_BM_low"), 0.0) for tag in tags)
        equip_high = sum(_safe_float(entry_by_tag.get(tag, {}).get("C_BM_high"), 0.0) for tag in tags)

        src_equip_base = _safe_float(src_block.get("equipment_total"), 0.0)
        src_equip_low = _safe_float(src_block.get("equipment_total_low"), 0.0)
        src_equip_high = _safe_float(src_block.get("equipment_total_high"), 0.0)
        src_inst_base = _safe_float(src_block.get("installation_total"), 0.0)
        src_inst_low = _safe_float(src_block.get("installation_total_low"), 0.0)
        src_inst_high = _safe_float(src_block.get("installation_total_high"), 0.0)

        ratio_inst_base = (src_inst_base / src_equip_base) if src_equip_base > 0 else 0.0
        ratio_inst_low = (src_inst_low / src_equip_low) if src_equip_low > 0 else ratio_inst_base
        ratio_inst_high = (src_inst_high / src_equip_high) if src_equip_high > 0 else ratio_inst_base

        inst_base = equip_base * ratio_inst_base
        inst_low = equip_low * ratio_inst_low
        inst_high = equip_high * ratio_inst_high

        src_installation_costs = src_block.get("installation_costs") or {}
        new_installation_costs: Dict[str, float] = {}
        src_total_install_costs = _safe_float(src_block.get("installation_total"), 0.0)
        for cat, src_cat_cost in src_installation_costs.items():
            share = (_safe_float(src_cat_cost, 0.0) / src_total_install_costs) if src_total_install_costs > 0 else 0.0
            new_installation_costs[str(cat)] = inst_base * share

        new_blocks.append(
            {
                "block_name": block_name,
                "equipment_tags": tags,
                "equipment_total": equip_base,
                "equipment_total_low": equip_low,
                "equipment_total_high": equip_high,
                "installation_costs": new_installation_costs,
                "installation_total": inst_base,
                "installation_total_low": inst_low,
                "installation_total_high": inst_high,
                "total_installed_cost": equip_base + inst_base,
                "total_installed_cost_low": equip_low + inst_low,
                "total_installed_cost_high": equip_high + inst_high,
            }
        )

    total_cbm = float(sum(_safe_float(entry.get("C_BM"), 0.0) for entry in new_entries))
    total_cbm_low = float(sum(_safe_float(entry.get("C_BM_low"), 0.0) for entry in new_entries))
    total_cbm_high = float(sum(_safe_float(entry.get("C_BM_high"), 0.0) for entry in new_entries))
    total_inst = float(sum(_safe_float(block.get("installation_total"), 0.0) for block in new_blocks))
    total_inst_low = float(sum(_safe_float(block.get("installation_total_low"), 0.0) for block in new_blocks))
    total_inst_high = float(sum(_safe_float(block.get("installation_total_high"), 0.0) for block in new_blocks))
    total_installed = total_cbm + total_inst
    total_installed_low = total_cbm_low + total_inst_low
    total_installed_high = total_cbm_high + total_inst_high

    rewritten = dict(src_capex)
    rewritten["generated_at"] = datetime.now(timezone.utc).isoformat()
    rewritten["entries"] = new_entries
    rewritten["block_summaries"] = new_blocks
    rewritten["total_C_BM"] = total_cbm
    rewritten["total_C_BM_low"] = total_cbm_low
    rewritten["total_C_BM_high"] = total_cbm_high
    rewritten["total_installation"] = total_inst
    rewritten["total_installation_low"] = total_inst_low
    rewritten["total_installation_high"] = total_inst_high
    rewritten["total_installed_cost"] = total_installed
    rewritten["total_installed_cost_low"] = total_installed_low
    rewritten["total_installed_cost_high"] = total_installed_high
    rewritten["entries_with_cost"] = int(sum(1 for entry in new_entries if _safe_float(entry.get("C_BM"), 0.0) > 0))
    rewritten["entries_without_cost"] = int(len(new_entries) - rewritten["entries_with_cost"])

    audit = {
        "global_scales": global_scales,
        "entry_scales": entry_scales,
        "source_total_installed_cost": _safe_float(src_capex.get("total_installed_cost"), 0.0),
        "rewritten_total_installed_cost": total_installed,
    }
    return rewritten, audit


def _build_degradation_ramp(
    *,
    history_meta: Optional[Dict[str, Any]],
    n_years: int,
    samples_per_year: int,
) -> Dict[str, Any]:
    if not history_meta or n_years <= 0:
        return {
            "ramp": np.ones(max(n_years, 1), dtype=float),
            "raw_ratio_by_year": [None] * max(n_years, 1),
            "filled_ratio_by_year": [1.0] * max(n_years, 1),
            "samples_by_year": {},
            "sampled_chunk_count": 0,
            "sampled_fraction": 0.0,
            "fallback_reason": "history_not_available",
        }

    chunks: List[ChunkMeta] = list(history_meta["chunks"])
    if not chunks:
        return {
            "ramp": np.ones(n_years, dtype=float),
            "raw_ratio_by_year": [None] * n_years,
            "filled_ratio_by_year": [1.0] * n_years,
            "samples_by_year": {},
            "sampled_chunk_count": 0,
            "sampled_fraction": 0.0,
            "fallback_reason": "no_chunks",
        }

    _assign_chunk_years_if_missing(chunks, n_years=n_years)
    chunks_by_year: Dict[int, List[ChunkMeta]] = {y: [] for y in range(n_years)}
    for chunk in chunks:
        y = int(chunk.year_idx or 0)
        y = min(max(y, 0), n_years - 1)
        chunks_by_year[y].append(chunk)

    samples_by_year: Dict[int, List[ChunkMeta]] = {}
    for y in range(n_years):
        year_chunks = sorted(chunks_by_year.get(y, []), key=lambda rec: rec.index)
        if not year_chunks:
            samples_by_year[y] = []
            continue
        positions = _sample_positions(len(year_chunks), max(1, samples_per_year))
        samples_by_year[y] = [year_chunks[pos] for pos in positions]

    # Assume schema is stable across chunks; take first chunk schema names.
    try:
        col_names = set(history_meta.get("column_names") or [])
    except Exception:
        col_names = set()

    available_primary_elec = [col for col in PRIMARY_ELEC_COLS if col in col_names]
    use_fallback_elec = (not available_primary_elec) and (FALLBACK_ELEC_COL in col_names)
    available_h2 = [col for col in H2_COLS if col in col_names]

    raw_ratio = np.full(n_years, np.nan, dtype=float)
    sampled_chunk_count = 0
    samples_manifest: Dict[str, List[str]] = {}
    for y in range(n_years):
        weighted_numer = 0.0
        weighted_denom = 0.0
        samples_manifest[str(y + 1)] = [sample.path.name for sample in samples_by_year.get(y, [])]
        for sample in samples_by_year.get(y, []):
            sampled_chunk_count += 1
            read_cols: List[str] = list(available_h2)
            if available_primary_elec:
                read_cols.extend(available_primary_elec)
            elif use_fallback_elec:
                read_cols.append(FALLBACK_ELEC_COL)
            read_cols = list(dict.fromkeys(read_cols))
            if not read_cols:
                continue
            try:
                frame = pd.read_parquet(sample.path, columns=read_cols)
            except Exception:
                continue

            h2_sum = 0.0
            for col in available_h2:
                if col in frame.columns:
                    h2_sum += float(pd.to_numeric(frame[col], errors="coerce").fillna(0.0).sum())
            if h2_sum <= EPS:
                continue

            elec_sum = 0.0
            if available_primary_elec:
                for col in available_primary_elec:
                    if col in frame.columns:
                        elec_sum += float(pd.to_numeric(frame[col], errors="coerce").fillna(0.0).sum())
            elif use_fallback_elec and FALLBACK_ELEC_COL in frame.columns:
                elec_sum = float(pd.to_numeric(frame[FALLBACK_ELEC_COL], errors="coerce").fillna(0.0).sum())
            if elec_sum <= EPS:
                continue

            ratio = elec_sum / h2_sum
            weight = max(float(sample.duration_h), EPS)
            weighted_numer += ratio * weight
            weighted_denom += weight

        if weighted_denom > 0:
            raw_ratio[y] = weighted_numer / weighted_denom

    if np.isfinite(raw_ratio).sum() == 0:
        return {
            "ramp": np.ones(n_years, dtype=float),
            "raw_ratio_by_year": [None] * n_years,
            "filled_ratio_by_year": [1.0] * n_years,
            "samples_by_year": samples_manifest,
            "sampled_chunk_count": sampled_chunk_count,
            "sampled_fraction": sampled_chunk_count / float(max(len(chunks), 1)),
            "fallback_reason": "no_valid_sample_ratios",
        }

    idx = np.arange(n_years, dtype=float)
    valid = np.isfinite(raw_ratio) & (raw_ratio > 0)
    valid_idx = idx[valid]
    valid_vals = raw_ratio[valid]
    filled = np.interp(idx, valid_idx, valid_vals)

    # Normalize against weighted mean and clip to conservative range.
    year_hours = np.array(history_meta.get("year_hours") or [HOURS_PER_YEAR] * n_years, dtype=float)
    if len(year_hours) < n_years:
        year_hours = np.concatenate([year_hours, np.full(n_years - len(year_hours), HOURS_PER_YEAR)])
    elif len(year_hours) > n_years:
        year_hours = year_hours[:n_years]
    weight_vec = np.clip(year_hours, a_min=0.0, a_max=None)
    if float(np.nansum(weight_vec)) <= 0:
        weight_vec = np.ones(n_years, dtype=float)

    weighted_mean = float(np.average(filled, weights=weight_vec))
    if weighted_mean <= 0 or not np.isfinite(weighted_mean):
        ramp = np.ones(n_years, dtype=float)
    else:
        ramp = filled / weighted_mean
        ramp = np.clip(ramp, a_min=0.85, a_max=1.25)

    return {
        "ramp": ramp,
        "raw_ratio_by_year": [None if not np.isfinite(v) else float(v) for v in raw_ratio],
        "filled_ratio_by_year": [float(v) for v in filled],
        "samples_by_year": samples_manifest,
        "sampled_chunk_count": sampled_chunk_count,
        "sampled_fraction": sampled_chunk_count / float(max(len(chunks), 1)),
        "fallback_reason": None,
        "primary_electricity_columns": available_primary_elec,
        "fallback_electricity_column_used": bool(use_fallback_elec and not available_primary_elec),
        "h2_columns": available_h2,
    }


def _rewrite_opex_report(
    *,
    src_opex: Dict[str, Any],
    rewritten_capex: Dict[str, Any],
    fci_factor_item_names: set[str],
    year_hours: np.ndarray,
    ramp: np.ndarray,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    n_years = len(year_hours)
    if n_years <= 0:
        year_hours = np.array([HOURS_PER_YEAR], dtype=float)
        ramp = np.array([1.0], dtype=float)
        n_years = 1
    if len(ramp) != n_years:
        if len(ramp) < n_years:
            ramp = np.concatenate([ramp, np.ones(n_years - len(ramp), dtype=float)])
        else:
            ramp = ramp[:n_years]

    old_fci = _safe_float(src_opex.get("fci"), default=0.0)
    if old_fci <= 0:
        old_fci = _safe_float(src_opex.get("total_installed_cost"), default=0.0)
    if old_fci <= 0:
        old_fci = _safe_float(src_opex.get("total_opex"), default=1.0)
    if old_fci <= 0:
        old_fci = 1.0

    new_fci_base = _safe_float(rewritten_capex.get("total_installed_cost"), default=old_fci)
    new_fci_low = _safe_float(rewritten_capex.get("total_installed_cost_low"), default=new_fci_base)
    new_fci_high = _safe_float(rewritten_capex.get("total_installed_cost_high"), default=new_fci_base)
    if new_fci_base <= 0:
        new_fci_base = old_fci
    if new_fci_low <= 0:
        new_fci_low = new_fci_base
    if new_fci_high <= 0:
        new_fci_high = new_fci_base

    fci_scales = {
        "base": new_fci_base / old_fci,
        "low": new_fci_low / old_fci,
        "high": new_fci_high / old_fci,
    }

    src_items = src_opex.get("items") or []
    new_items: List[Dict[str, Any]] = []
    item_annual_cost_by_year: Dict[str, List[float]] = {}
    name_counter: Dict[str, int] = {}

    totals = {
        "base": {"variable": 0.0, "fixed": 0.0, "maintenance": 0.0, "credit": 0.0},
        "low": {"variable": 0.0, "fixed": 0.0, "maintenance": 0.0, "credit": 0.0},
        "high": {"variable": 0.0, "fixed": 0.0, "maintenance": 0.0, "credit": 0.0},
    }
    scaled_item_names: List[str] = []

    weights_var = np.clip(ramp * np.clip(year_hours, 0.0, None), a_min=0.0, a_max=None)
    weights_flat = np.clip(year_hours, a_min=0.0, a_max=None)

    for raw_item in src_items:
        if not isinstance(raw_item, dict):
            continue
        item = dict(raw_item)
        name = str(item.get("name", "")).strip()
        name_key = name.lower()
        category = str(item.get("category", "")).strip().lower()
        is_credit = bool(item.get("is_credit", False))
        src_cost = _safe_float(item.get("annual_cost"), default=0.0)

        is_fci_factor_item = name_key in fci_factor_item_names
        if is_fci_factor_item:
            scaled_item_names.append(name)

        cost_by_variant = {
            variant: (src_cost * fci_scales[variant] if is_fci_factor_item else src_cost)
            for variant in ("base", "low", "high")
        }

        # Base item list stays in classic OPEX item schema.
        item["annual_cost"] = float(cost_by_variant["base"])
        if is_fci_factor_item:
            base_formula = str(item.get("formula", "")).strip()
            suffix = f" [approx FCI scale={fci_scales['base']:.6g}]"
            item["formula"] = f"{base_formula}{suffix}" if base_formula else suffix.strip()
        new_items.append(item)

        bucket = "fixed"
        if category == "variable":
            bucket = "variable"
        elif category == "maintenance":
            bucket = "maintenance"
        for variant in ("base", "low", "high"):
            totals[variant][bucket] += cost_by_variant[variant]
            if is_credit:
                totals[variant]["credit"] += cost_by_variant[variant]

        base_name = name or f"item_{len(new_items)}"
        seen = name_counter.get(base_name, 0) + 1
        name_counter[base_name] = seen
        key = base_name if seen == 1 else f"{base_name} [{seen}]"

        profile = weights_var if (category == "variable" and not is_credit) else weights_flat
        yearly_item = _allocate_equivalent_annual(
            annual_value=cost_by_variant["base"],
            year_hours=year_hours,
            profile_weights=profile,
        )
        item_annual_cost_by_year[key] = [float(v) for v in yearly_item]

    def _series_from_scalars(variant: str) -> Dict[str, np.ndarray]:
        var_series = _allocate_equivalent_annual(
            annual_value=totals[variant]["variable"],
            year_hours=year_hours,
            profile_weights=weights_var,
        )
        fix_series = _allocate_equivalent_annual(
            annual_value=totals[variant]["fixed"],
            year_hours=year_hours,
            profile_weights=weights_flat,
        )
        maint_series = _allocate_equivalent_annual(
            annual_value=totals[variant]["maintenance"],
            year_hours=year_hours,
            profile_weights=weights_flat,
        )
        credit_series = _allocate_equivalent_annual(
            annual_value=totals[variant]["credit"],
            year_hours=year_hours,
            profile_weights=weights_flat,
        )
        total_series = var_series + fix_series + maint_series
        cash_series = total_series - credit_series
        return {
            "variable": var_series,
            "fixed": fix_series,
            "maintenance": maint_series,
            "credit": credit_series,
            "total": total_series,
            "cash": cash_series,
        }

    base_series = _series_from_scalars("base")
    low_series = _series_from_scalars("low")
    high_series = _series_from_scalars("high")

    annual_h2_scalar = _safe_float(src_opex.get("annual_h2_production_kg"), default=0.0)
    h2_yearly = np.zeros(n_years, dtype=float)
    if annual_h2_scalar > 0:
        h2_weights = np.clip(year_hours / np.clip(ramp, a_min=0.05, a_max=None), a_min=0.0, a_max=None)
        h2_yearly = _allocate_equivalent_annual(
            annual_value=annual_h2_scalar,
            year_hours=year_hours,
            profile_weights=h2_weights,
        )
    annual_h2_eq = _equivalent_annual(h2_yearly, year_hours) if annual_h2_scalar > 0 else annual_h2_scalar

    def _totals_from_variant(variant: str) -> Dict[str, float]:
        data = totals[variant]
        total_opex = float(data["variable"] + data["fixed"] + data["maintenance"])
        total_credit = float(data["credit"])
        return {
            "total_variable_cost": float(data["variable"]),
            "total_fixed_cost": float(data["fixed"]),
            "total_maintenance_cost": float(data["maintenance"]),
            "total_opex": total_opex,
            "total_credit_cost": total_credit,
            "total_opex_cashflow": float(total_opex - total_credit),
        }

    scalar_base = _totals_from_variant("base")
    scalar_low = _totals_from_variant("low")
    scalar_high = _totals_from_variant("high")

    rewritten = dict(src_opex)
    rewritten["fci"] = float(new_fci_base)
    rewritten["items"] = new_items

    rewritten["total_variable_cost"] = scalar_base["total_variable_cost"]
    rewritten["total_fixed_cost"] = scalar_base["total_fixed_cost"]
    rewritten["total_maintenance_cost"] = scalar_base["total_maintenance_cost"]
    rewritten["total_opex"] = scalar_base["total_opex"]
    rewritten["total_credit_cost"] = scalar_base["total_credit_cost"]
    rewritten["total_opex_cashflow"] = scalar_base["total_opex_cashflow"]

    rewritten["total_opex_low"] = scalar_low["total_opex"]
    rewritten["total_opex_high"] = scalar_high["total_opex"]
    rewritten["total_opex_cashflow_low"] = scalar_low["total_opex_cashflow"]
    rewritten["total_opex_cashflow_high"] = scalar_high["total_opex_cashflow"]

    rewritten["year_index"] = [idx + 1 for idx in range(n_years)]
    rewritten["year_hours"] = [float(v) for v in year_hours]
    rewritten["item_annual_cost_by_year"] = item_annual_cost_by_year

    rewritten["total_variable_cost_by_year"] = [float(v) for v in base_series["variable"]]
    rewritten["total_fixed_cost_by_year"] = [float(v) for v in base_series["fixed"]]
    rewritten["total_maintenance_cost_by_year"] = [float(v) for v in base_series["maintenance"]]
    rewritten["total_opex_by_year"] = [float(v) for v in base_series["total"]]
    rewritten["total_opex_cashflow_by_year"] = [float(v) for v in base_series["cash"]]

    rewritten["total_opex_low_by_year"] = [float(v) for v in low_series["total"]]
    rewritten["total_opex_high_by_year"] = [float(v) for v in high_series["total"]]
    rewritten["total_opex_cashflow_low_by_year"] = [float(v) for v in low_series["cash"]]
    rewritten["total_opex_cashflow_high_by_year"] = [float(v) for v in high_series["cash"]]

    rewritten["annual_h2_production_kg_by_year"] = [float(v) for v in h2_yearly]
    rewritten["annual_h2_production_kg"] = float(annual_h2_eq)
    rewritten["opex_per_kg_h2"] = (
        float(scalar_base["total_opex"] / annual_h2_eq) if annual_h2_eq > 0 else 0.0
    )

    simulation_hours = _safe_float(rewritten.get("simulation_hours"), default=0.0)
    if simulation_hours > 0:
        rewritten["annualization_factor"] = HOURS_PER_YEAR / simulation_hours

    audit = {
        "old_fci": old_fci,
        "new_fci": {"base": new_fci_base, "low": new_fci_low, "high": new_fci_high},
        "fci_scales": fci_scales,
        "fci_scaled_items": sorted(set(scaled_item_names)),
        "equivalent_annual_checks": {
            "base_total_opex": _equivalent_annual(np.array(base_series["total"]), year_hours),
            "base_total_opex_cashflow": _equivalent_annual(np.array(base_series["cash"]), year_hours),
            "low_total_opex": _equivalent_annual(np.array(low_series["total"]), year_hours),
            "high_total_opex": _equivalent_annual(np.array(high_series["total"]), year_hours),
        },
    }
    return rewritten, audit


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    opex_config_path = Path(args.opex_config).resolve()
    audit_path = Path(args.audit_json).resolve() if args.audit_json else (output_dir / "approximation_audit.json")
    design_ref_path = Path(args.design_capex_reference).resolve() if args.design_capex_reference else None

    capex_in = input_dir / "capex_report.json"
    opex_in = input_dir / "opex_report.json"
    if not capex_in.exists():
        raise FileNotFoundError(f"Input CAPEX report not found: {capex_in}")
    if not opex_in.exists():
        raise FileNotFoundError(f"Input OPEX report not found: {opex_in}")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_capex = output_dir / "capex_report.json"
    out_opex = output_dir / "opex_report.json"
    existing = [path for path in (out_capex, out_opex, audit_path) if path.exists()]
    if existing and not args.overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise ValueError(f"Output files already exist. Pass --overwrite to replace: {joined}")

    src_capex = _load_json(capex_in)
    src_opex = _load_json(opex_in)
    _ensure_required_keys(
        src_capex,
        required=("entries", "block_summaries", "total_installed_cost"),
        label="CAPEX",
    )
    _ensure_required_keys(
        src_opex,
        required=("items", "total_opex", "total_variable_cost", "total_fixed_cost", "total_maintenance_cost"),
        label="OPEX",
    )

    ref_capex = None
    if design_ref_path:
        if not design_ref_path.exists():
            raise FileNotFoundError(f"Design CAPEX reference not found: {design_ref_path}")
        ref_capex = _load_json(design_ref_path)

    fci_factor_item_names, opex_cfg_provenance = _load_fci_factor_item_names(opex_config_path)

    chunks_dir = _resolve_history_chunks(args.history_dir)
    history_meta = None
    if chunks_dir:
        chunk_files = _sorted_chunk_files(chunks_dir)
        if chunk_files:
            history_meta = _collect_chunk_metadata(chunk_files)
            LOGGER.info(
                "History metadata scanned: %d chunks (minute stats missing in %d chunks).",
                len(history_meta["chunks"]),
                history_meta["minute_stats_missing_count"],
            )
        else:
            LOGGER.warning("No chunk_*.parquet files found in %s. Falling back to flat yearly profile.", chunks_dir)
    elif args.history_dir:
        LOGGER.warning("History path does not contain history chunks: %s", args.history_dir)

    n_years = _determine_year_count(src_opex=src_opex, history_meta=history_meta, project_years=args.project_years)
    year_hours = np.array(
        _build_year_hours(n_years=n_years, src_opex=src_opex, history_meta=history_meta),
        dtype=float,
    )
    if len(year_hours) != n_years:
        if len(year_hours) < n_years:
            year_hours = np.concatenate([year_hours, np.full(n_years - len(year_hours), HOURS_PER_YEAR)])
        else:
            year_hours = year_hours[:n_years]

    ramp_info = _build_degradation_ramp(
        history_meta=history_meta,
        n_years=n_years,
        samples_per_year=max(1, int(args.samples_per_year)),
    )
    ramp = np.array(ramp_info["ramp"], dtype=float)

    capex_scales = _derive_capex_scales(
        src_capex=src_capex,
        ref_capex=ref_capex,
        fallback_factor=args.design_capex_factor,
    )
    rewritten_capex, capex_audit = _rewrite_capex_report(
        src_capex=src_capex,
        ref_capex=ref_capex,
        global_scales=capex_scales,
    )
    rewritten_opex, opex_audit = _rewrite_opex_report(
        src_opex=src_opex,
        rewritten_capex=rewritten_capex,
        fci_factor_item_names=fci_factor_item_names,
        year_hours=year_hours,
        ramp=ramp,
    )

    _write_json(out_capex, rewritten_capex)
    _write_json(out_opex, rewritten_opex)

    audit_payload: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "input_dir": str(input_dir),
            "capex_report": str(capex_in),
            "opex_report": str(opex_in),
            "history_chunks": str(chunks_dir) if chunks_dir else None,
            "design_capex_reference": str(design_ref_path) if design_ref_path else None,
            "opex_config": str(opex_config_path),
        },
        "settings": {
            "samples_per_year": int(args.samples_per_year),
            "design_capex_factor": float(args.design_capex_factor),
            "project_years": int(args.project_years) if args.project_years else None,
        },
        "history_metadata": {
            "chunk_count": int(len(history_meta["chunks"])) if history_meta else 0,
            "minute_origin": history_meta.get("minute_origin") if history_meta else None,
            "minute_max": history_meta.get("minute_max") if history_meta else None,
            "year_count_from_history": int(history_meta.get("year_count") or 0) if history_meta else 0,
            "year_hours_from_history": history_meta.get("year_hours") if history_meta else [],
            "minute_stats_missing_count": int(history_meta.get("minute_stats_missing_count", 0)) if history_meta else 0,
        },
        "capex_approximation": capex_audit,
        "opex_approximation": opex_audit,
        "degradation_ramp": {
            "values": [float(v) for v in ramp],
            "raw_ratio_by_year": ramp_info.get("raw_ratio_by_year"),
            "filled_ratio_by_year": ramp_info.get("filled_ratio_by_year"),
            "samples_by_year": ramp_info.get("samples_by_year"),
            "sampled_chunk_count": ramp_info.get("sampled_chunk_count"),
            "sampled_fraction": ramp_info.get("sampled_fraction"),
            "fallback_reason": ramp_info.get("fallback_reason"),
            "primary_electricity_columns": ramp_info.get("primary_electricity_columns"),
            "fallback_electricity_column_used": ramp_info.get("fallback_electricity_column_used"),
            "h2_columns": ramp_info.get("h2_columns"),
        },
        "year_model": {
            "n_years": int(n_years),
            "year_hours": [float(v) for v in year_hours],
        },
        "opex_config_provenance": opex_cfg_provenance,
        "outputs": {
            "capex_report": str(out_capex),
            "opex_report": str(out_opex),
            "audit_json": str(audit_path),
        },
    }
    _write_json(audit_path, audit_payload)

    LOGGER.info("Approximate rewrite completed.")
    LOGGER.info("  CAPEX output: %s", out_capex)
    LOGGER.info("  OPEX output:  %s", out_opex)
    LOGGER.info("  Audit output: %s", audit_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        LOGGER.error("Approximate rewrite failed: %s", exc)
        sys.exit(1)
