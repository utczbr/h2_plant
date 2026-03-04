"""
Discounted LCOH Calculator.

Computes discounted Levelized Cost of Hydrogen (LCOH) for plant and pathways.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from h2_plant.economics.models import CapexReport, BlockCostSummary
from h2_plant.economics.opex_models import OpexReport
from h2_plant.economics.lcoh_models import LcohReport, LcohVariantsReport

logger = logging.getLogger(__name__)
MINUTES_PER_YEAR = 525600.0
HOURS_PER_YEAR = 8760.0
AUTO_HISTORY_SCAN_WORKERS = 2
HISTORY_SCAN_MB_PER_WORKER = 128


@dataclass
class LcohInputs:
    capex_report: CapexReport
    opex_report: OpexReport
    history_chunks_dir: Path
    discount_rate: float
    project_years: int
    history_scan_workers: int = 0
    history_scan_max_memory_mb: Optional[int] = None
    history_scan_mode: str = "auto"


class LcohCalculator:
    """Calculate discounted LCOH for plant and pathways."""

    VARIANT_ORDER = ("low", "base", "high")
    CAPEX_VARIANT_TOTAL_FIELDS = {
        "low": "total_installed_cost_low",
        "base": "total_installed_cost",
        "high": "total_installed_cost_high",
    }
    CAPEX_VARIANT_BLOCK_FIELDS = {
        "low": "total_installed_cost_low",
        "base": "total_installed_cost",
        "high": "total_installed_cost_high",
    }
    OPEX_VARIANT_TOTAL_FIELDS = {
        "low": "total_opex_low",
        "base": "total_opex",
        "high": "total_opex_high",
    }

    @staticmethod
    def _raise_disconnected_mount(path: Path, exc: OSError) -> None:
        raise ValueError(
            "History chunks are on a disconnected mount (Errno 107). "
            f"Failed while reading: {path}. "
            "Remount the drive or copy history_chunks locally and rerun with "
            "--history-dir /path/to/history_chunks."
        ) from exc

    def _discount_factor_sum(self, r: float, n: int) -> float:
        if n <= 0:
            return 0.0
        if r == 0:
            return float(n)
        years = np.arange(1, n + 1, dtype=float)
        return float(np.sum(1.0 / ((1.0 + r) ** years)))

    @staticmethod
    def _safe_div(num: float, den: float) -> float:
        if den <= 0:
            return 0.0
        return num / den

    @staticmethod
    def _to_positive_float(value, field_name: str, variant: str) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"Missing or invalid {field_name} for LCOH variant '{variant}'."
            ) from None
        if parsed <= 0:
            raise ValueError(
                f"Missing or non-positive {field_name} for LCOH variant '{variant}'."
            )
        return parsed

    def _resolve_history_scan_workers(
        self,
        *,
        requested_workers: int,
        max_memory_mb: Optional[int],
        n_chunks: int,
        scan_mode: str,
    ) -> int:
        mode = str(scan_mode or "auto").strip().lower()
        if mode not in {"auto", "serial", "parallel"}:
            raise ValueError(f"Unknown history_scan_mode: {scan_mode}")
        if mode == "serial" or n_chunks <= 1:
            return 1

        if requested_workers > 0:
            effective = int(requested_workers)
        else:
            effective = AUTO_HISTORY_SCAN_WORKERS

        effective = min(effective, n_chunks, max(1, int(os.cpu_count() or 1)))

        if max_memory_mb is not None and max_memory_mb > 0:
            max_by_memory = int(max_memory_mb) // HISTORY_SCAN_MB_PER_WORKER
            if max_by_memory <= 0:
                effective = 1
            else:
                effective = min(effective, max_by_memory)

        return max(1, effective)

    def _find_minute_origin(self, chunk_files: list[Path]) -> Optional[float]:
        for chunk_file in chunk_files:
            try:
                df_minute = pd.read_parquet(chunk_file, columns=["minute"])
            except OSError as e:
                if getattr(e, "errno", None) == 107:
                    self._raise_disconnected_mount(chunk_file, e)
                raise
            if df_minute.empty:
                continue
            minute_values = pd.to_numeric(df_minute["minute"], errors="coerce").to_numpy(dtype=float)
            valid = minute_values[np.isfinite(minute_values)]
            if valid.size > 0:
                return float(valid[0])
        return None

    def _scan_h2_chunk(
        self,
        *,
        chunk_file: Path,
        required_cols: list[str],
        minute_origin: Optional[float],
    ) -> Dict[str, object]:
        try:
            df = pd.read_parquet(chunk_file, columns=required_cols)
        except OSError as e:
            if getattr(e, "errno", None) == 107:
                self._raise_disconnected_mount(chunk_file, e)
            raise

        if df.empty:
            return {
                "totals": {"pem": 0.0, "soec": 0.0, "atr": 0.0},
                "totals_by_year": {"pem": {}, "soec": {}, "atr": {}},
                "minutes_min": None,
                "minutes_max": None,
                "median_diff": None,
                "year_indices": [],
            }

        minute_series = pd.to_numeric(df["minute"], errors="coerce")
        minute_values = minute_series.to_numpy(dtype=float)
        minute_valid = minute_values[np.isfinite(minute_values)]

        pem_vals = pd.to_numeric(df["H2_pem_kg"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        soec_vals = pd.to_numeric(df["H2_soec_kg"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        atr_vals = pd.to_numeric(df["H2_atr_kg"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

        totals = {
            "pem": float(np.sum(pem_vals)),
            "soec": float(np.sum(soec_vals)),
            "atr": float(np.sum(atr_vals)),
        }
        totals_by_year = {"pem": {}, "soec": {}, "atr": {}}
        year_indices: set[int] = set()

        if minute_origin is not None:
            year_idx = np.full(len(df), -1, dtype=int)
            valid_mask = np.isfinite(minute_values)
            year_idx[valid_mask] = np.floor(
                (minute_values[valid_mask] - minute_origin) / MINUTES_PER_YEAR
            ).astype(int)
            year_idx[year_idx < 0] = 0
            valid_years = np.unique(year_idx[year_idx >= 0])
            for y in valid_years:
                year_mask = year_idx == y
                year_key = int(y)
                totals_by_year["pem"][year_key] = float(np.sum(pem_vals[year_mask]))
                totals_by_year["soec"][year_key] = float(np.sum(soec_vals[year_mask]))
                totals_by_year["atr"][year_key] = float(np.sum(atr_vals[year_mask]))
                year_indices.add(year_key)

        minutes_min = None
        min_val = minute_series.min()
        if pd.notna(min_val):
            minutes_min = float(min_val)
        minutes_max = None
        max_val = minute_series.max()
        if pd.notna(max_val):
            minutes_max = float(max_val)

        median_diff = None
        if len(minute_valid) > 1:
            diffs = np.diff(minute_valid)
            if diffs.size > 0:
                median_diff = float(np.median(diffs))

        return {
            "totals": totals,
            "totals_by_year": totals_by_year,
            "minutes_min": minutes_min,
            "minutes_max": minutes_max,
            "median_diff": median_diff,
            "year_indices": sorted(year_indices),
        }

    @staticmethod
    def _merge_year_totals(
        totals_by_year: Dict[str, Dict[int, float]],
        partial_totals_by_year: Dict[str, Dict[int, float]],
    ) -> None:
        for key in ("pem", "soec", "atr"):
            for year_idx, value in partial_totals_by_year.get(key, {}).items():
                totals_by_year[key][int(year_idx)] = (
                    float(totals_by_year[key].get(int(year_idx), 0.0)) + float(value)
                )

    def _load_h2_totals(
        self,
        chunks_dir: Path,
        *,
        workers: int = 0,
        max_memory_mb: Optional[int] = None,
        scan_mode: str = "auto",
    ) -> Dict[str, object]:
        try:
            chunk_files = sorted(
                chunks_dir.glob("chunk_*.parquet"),
                key=lambda p: int(p.stem.split("_")[-1])
            )
        except Exception:
            chunk_files = sorted(chunks_dir.glob("chunk_*.parquet"))

        if not chunk_files:
            raise ValueError(f"No chunk files found in {chunks_dir}")

        required_cols = ["minute", "H2_pem_kg", "H2_soec_kg", "H2_atr_kg"]

        def _schema_columns(first_chunk: Path):
            try:
                import pyarrow.parquet as pq
                return pq.read_schema(first_chunk).names
            except OSError as e:
                if getattr(e, "errno", None) == 107:
                    self._raise_disconnected_mount(first_chunk, e)
            except Exception:
                pass

            try:
                df_preview = pd.read_parquet(first_chunk, nrows=1)
            except OSError as e:
                if getattr(e, "errno", None) == 107:
                    self._raise_disconnected_mount(first_chunk, e)
                raise

            schema_cols_local = list(df_preview.columns)
            del df_preview
            return schema_cols_local

        schema_cols = _schema_columns(chunk_files[0])

        missing = [c for c in required_cols if c not in schema_cols]
        if missing:
            h2_cols = [c for c in schema_cols if "h2" in c.lower() and "kg" in c.lower()]
            raise ValueError(
                f"Missing required H2 columns: {missing}. "
                f"Available H2 columns: {h2_cols[:20]}"
            )

        minute_origin = self._find_minute_origin(chunk_files)
        resolved_workers = self._resolve_history_scan_workers(
            requested_workers=int(workers or 0),
            max_memory_mb=max_memory_mb,
            n_chunks=len(chunk_files),
            scan_mode=scan_mode,
        )

        totals = {"pem": 0.0, "soec": 0.0, "atr": 0.0}
        totals_by_year = {
            "pem": {},
            "soec": {},
            "atr": {},
        }
        minutes_min = None
        minutes_max = None
        diff_samples = []
        year_indices_seen: set[int] = set()

        if resolved_workers <= 1:
            partials = (
                self._scan_h2_chunk(
                    chunk_file=chunk_file,
                    required_cols=required_cols,
                    minute_origin=minute_origin,
                )
                for chunk_file in chunk_files
            )
        else:
            def _scan_chunk_file(chunk_file: Path) -> Dict[str, object]:
                return self._scan_h2_chunk(
                    chunk_file=chunk_file,
                    required_cols=required_cols,
                    minute_origin=minute_origin,
                )

            with ThreadPoolExecutor(max_workers=resolved_workers) as executor:
                partials = list(executor.map(_scan_chunk_file, chunk_files))
        for partial in partials:
            totals["pem"] += float(partial["totals"]["pem"])
            totals["soec"] += float(partial["totals"]["soec"])
            totals["atr"] += float(partial["totals"]["atr"])
            self._merge_year_totals(totals_by_year, partial["totals_by_year"])

            partial_min = partial["minutes_min"]
            partial_max = partial["minutes_max"]
            if partial_min is not None and (minutes_min is None or partial_min < minutes_min):
                minutes_min = float(partial_min)
            if partial_max is not None and (minutes_max is None or partial_max > minutes_max):
                minutes_max = float(partial_max)

            partial_diff = partial["median_diff"]
            if partial_diff is not None:
                diff_samples.append(float(partial_diff))
            year_indices_seen.update(int(v) for v in partial.get("year_indices", []))

        # Estimate simulation hours
        sim_hours = 0.0
        dt_h = None
        if diff_samples:
            dt_min = float(np.median(diff_samples))
            if dt_min > 0:
                dt_h = dt_min / 60.0
        if minutes_min is not None and minutes_max is not None and minutes_max >= minutes_min:
            sim_hours = (minutes_max - minutes_min) / 60.0
            if dt_h is not None:
                sim_hours += dt_h

        # Build year axis/hours from minute span using the same relative-year policy
        if minute_origin is not None and minutes_max is not None:
            dt_min = float(np.median(diff_samples)) if diff_samples else 1.0
            if not np.isfinite(dt_min) or dt_min <= 0:
                dt_min = 1.0
            sim_end_min = float(minutes_max) + dt_min
            span_min = max(0.0, sim_end_min - float(minute_origin))
            year_count = int(np.ceil(span_min / MINUTES_PER_YEAR)) if span_min > 0 else 1
            year_indices = list(range(year_count))
            year_hours: List[float] = []
            for year_idx in year_indices:
                year_start = float(minute_origin) + (year_idx * MINUTES_PER_YEAR)
                year_end = year_start + MINUTES_PER_YEAR
                overlap_min = max(0.0, min(sim_end_min, year_end) - max(float(minute_origin), year_start))
                year_hours.append(overlap_min / 60.0)
        else:
            year_indices = sorted(year_indices_seen) if year_indices_seen else [0]
            year_hours = [0.0 for _ in year_indices]

        return {
            "totals": totals,
            "totals_by_year": totals_by_year,
            "year_indices": year_indices,
            "year_hours": year_hours,
            "simulation_hours": sim_hours,
        }

    def _resolve_annual_production(
        self,
        history_chunks_dir: Path,
        opex_report: OpexReport,
        warnings: list[str],
        *,
        history_scan_workers: int = 0,
        history_scan_max_memory_mb: Optional[int] = None,
        history_scan_mode: str = "auto",
    ) -> tuple[Dict[str, float], float, Dict[str, float], Dict[str, object]]:
        loaded = self._load_h2_totals(
            history_chunks_dir,
            workers=history_scan_workers,
            max_memory_mb=history_scan_max_memory_mb,
            scan_mode=history_scan_mode,
        )
        totals = dict(loaded.get("totals", {}))
        sim_hours = loaded.get("simulation_hours", 0.0)

        annual_factor = 0.0
        if sim_hours and sim_hours > 0:
            annual_factor = HOURS_PER_YEAR / float(sim_hours)
        else:
            fallback_hours = getattr(opex_report, "simulation_hours", 0.0)
            if fallback_hours and fallback_hours > 0:
                annual_factor = HOURS_PER_YEAR / float(fallback_hours)
                warnings.append("Simulation hours inferred from OPEX report.")
            else:
                warnings.append("Simulation hours could not be determined; annualization set to 0.")
                logger.warning("Simulation hours could not be determined; annualization set to 0.")

        annual_by = {k: v * annual_factor for k, v in totals.items()}
        annual_total = sum(annual_by.values())
        if annual_total <= 0:
            warnings.append("Annual H2 production is zero; LCOH will be zeroed.")

        prod_shares = {
            k: (annual_by[k] / annual_total) if annual_total > 0 else 0.0
            for k in annual_by
        }
        return annual_by, annual_total, prod_shares, loaded

    def _extract_capex_variant_totals(
        self,
        capex_report: CapexReport,
        strict: bool,
    ) -> Dict[str, float]:
        if not strict:
            base_total = float(capex_report.total_installed_cost or capex_report.total_C_BM or 0.0)
            return {"base": base_total}

        totals: Dict[str, float] = {}
        for variant in self.VARIANT_ORDER:
            field = self.CAPEX_VARIANT_TOTAL_FIELDS[variant]
            totals[variant] = self._to_positive_float(
                getattr(capex_report, field, None),
                field_name=field,
                variant=variant,
            )
        return totals

    def _extract_opex_variant_totals(
        self,
        opex_report: OpexReport,
        strict: bool,
    ) -> Dict[str, float]:
        if not strict:
            return {"base": float(opex_report.total_opex or 0.0)}

        totals: Dict[str, float] = {}
        for variant in self.VARIANT_ORDER:
            field = self.OPEX_VARIANT_TOTAL_FIELDS[variant]
            totals[variant] = self._to_positive_float(
                getattr(opex_report, field, None),
                field_name=field,
                variant=variant,
            )
        return totals

    @staticmethod
    def _build_h2_yearly_totals(
        loaded_h2: Dict[str, object],
    ) -> Tuple[List[int], np.ndarray, Dict[str, np.ndarray]]:
        year_indices = [int(v) for v in loaded_h2.get("year_indices", [])]
        if not year_indices:
            year_indices = [0]

        totals_by_year = loaded_h2.get("totals_by_year", {}) or {}
        by_path: Dict[str, np.ndarray] = {}
        for key in ("pem", "soec", "atr"):
            key_map = totals_by_year.get(key, {}) or {}
            by_path[key] = np.array(
                [float(key_map.get(int(y), 0.0) or 0.0) for y in year_indices],
                dtype=float,
            )

        total = by_path["pem"] + by_path["soec"] + by_path["atr"]
        return year_indices, total, by_path

    def _extract_opex_yearly_stream(
        self,
        *,
        opex_report: OpexReport,
        variant: str,
        fallback_annual_opex: float,
        year_count: int,
    ) -> np.ndarray:
        variant_field_map = {
            "base": "total_opex_by_year",
            "low": "total_opex_low_by_year",
            "high": "total_opex_high_by_year",
        }
        candidate_field = variant_field_map.get(variant, "total_opex_by_year")
        values = getattr(opex_report, candidate_field, None)
        if values is None and variant != "base":
            # In mixed reports, low/high yearly may be missing while base exists.
            values = getattr(opex_report, "total_opex_by_year", None)
            if values is not None and fallback_annual_opex > 0 and float(opex_report.total_opex or 0.0) > 0:
                scale = fallback_annual_opex / float(opex_report.total_opex)
                values = [float(v) * scale for v in values]

        if isinstance(values, list) and len(values) > 0:
            arr = np.array([float(v or 0.0) for v in values], dtype=float)
            if len(arr) >= year_count:
                return arr[:year_count]
            out = np.zeros(year_count, dtype=float)
            out[: len(arr)] = arr
            return out

        # Backward-compatible fallback: constant annual OPEX over available horizon.
        return np.full(year_count, float(fallback_annual_opex), dtype=float)

    @staticmethod
    def _discount_series(values: np.ndarray, discount_rate: float) -> np.ndarray:
        if values.size == 0:
            return values
        years = np.arange(1, len(values) + 1, dtype=float)
        if discount_rate == 0:
            return values
        factors = 1.0 / np.power(1.0 + discount_rate, years)
        return values * factors

    def _allocate_capex_by_block(
        self,
        *,
        block_summaries: list[BlockCostSummary],
        fallback_shares: Dict[str, float],
        warnings: list[str],
        variant: str,
    ) -> tuple[Dict[str, float], Dict[str, bool]]:
        capex_by = {"pem": 0.0, "soec": 0.0, "atr": 0.0}
        block_map = {
            "pem": ["pem"],
            "soec": ["soec"],
            "atr": ["atr"],
        }
        matched = {"pem": False, "soec": False, "atr": False}
        block_total_field = self.CAPEX_VARIANT_BLOCK_FIELDS[variant]

        for block in block_summaries:
            block_name = block.block_name.lower()
            for key, tokens in block_map.items():
                if any(tok in block_name for tok in tokens):
                    capex_by[key] += float(getattr(block, block_total_field, 0.0) or 0.0)
                    matched[key] = True

        for key, ok in matched.items():
            if not ok:
                share = fallback_shares.get(key, 0.0)
                warnings.append(
                    f"CAPEX block for {key.upper()} missing in {variant.upper()} variant; "
                    f"allocating by production share ({share:.2%})."
                )

        return capex_by, matched

    def _allocate_opex_components(
        self,
        *,
        opex_report: OpexReport,
        target_opex_total: float,
    ) -> Dict[str, float]:
        """
        Build annual OPEX component totals from tagged OPEX items.

        The raw item totals are scaled to match the requested OPEX variant total,
        preserving compatibility between sub-breakdown and top-level OPEX.
        """
        components = {
            "energy": 0.0,
            "water": 0.0,
            "compression": 0.0,
            "other_opex": 0.0,
        }

        for item in opex_report.items:
            annual_cost = float(getattr(item, "annual_cost", 0.0) or 0.0)
            raw_tag = str(getattr(item, "lcoh_component", "") or "").strip().lower()
            if raw_tag in ("energy", "water", "compression"):
                components[raw_tag] += annual_cost
            else:
                components["other_opex"] += annual_cost

        raw_total = sum(components.values())
        if raw_total <= 0:
            if target_opex_total > 0:
                components["other_opex"] = float(target_opex_total)
            return components

        scale = float(target_opex_total) / raw_total
        for key in components:
            components[key] *= scale
        return components

    @staticmethod
    def _normalize_shares(shares: Dict[str, float], valid_keys: list[str]) -> Dict[str, float]:
        cleaned = {
            key: max(0.0, float(shares.get(key, 0.0) or 0.0))
            for key in valid_keys
        }
        total = sum(cleaned.values())
        if total <= 0:
            return {key: 0.0 for key in valid_keys}
        return {key: value / total for key, value in cleaned.items()}

    def _allocate_opex_by_pathway(
        self,
        *,
        opex_report: OpexReport,
        target_opex_total: float,
        fallback_shares: Dict[str, float],
    ) -> Dict[str, float]:
        pathway_keys = list(fallback_shares.keys())
        normalized_fallback = self._normalize_shares(fallback_shares, pathway_keys)
        opex_by = {key: 0.0 for key in pathway_keys}

        for item in opex_report.items:
            annual_cost = float(getattr(item, "annual_cost", 0.0) or 0.0)
            if annual_cost == 0:
                continue

            shares = normalized_fallback
            if (
                item.category.value == "Variable"
                and getattr(item, "pathway_shares", None) is not None
            ):
                item_shares = self._normalize_shares(
                    getattr(item, "pathway_shares", {}) or {},
                    pathway_keys,
                )
                if sum(item_shares.values()) <= 0:
                    raise ValueError(
                        "Invalid causal pathway allocation: non-zero variable OPEX item "
                        f"'{item.name}' has pathway_shares that sum to zero."
                    )
                shares = item_shares

            for key in pathway_keys:
                opex_by[key] += annual_cost * shares.get(key, 0.0)

        raw_total = sum(opex_by.values())
        if raw_total > 0:
            scale = float(target_opex_total) / raw_total
            for key in pathway_keys:
                opex_by[key] *= scale
            return opex_by

        if target_opex_total <= 0:
            return opex_by

        for key in pathway_keys:
            opex_by[key] = float(target_opex_total) * normalized_fallback.get(key, 0.0)
        return opex_by

    def _build_variant_report(
        self,
        *,
        variant: str,
        capex_total: float,
        opex_total: float,
        opex_report: OpexReport,
        annual_by: Dict[str, float],
        annual_total: float,
        prod_shares: Dict[str, float],
        discount_rate: float,
        project_years: int,
        loaded_h2: Dict[str, object],
        capex_report: CapexReport,
        shared_warnings: list[str],
        generated_at: str,
    ) -> LcohReport:
        warnings = list(shared_warnings)

        capex_by, matched = self._allocate_capex_by_block(
            block_summaries=capex_report.block_summaries,
            fallback_shares=prod_shares,
            warnings=warnings,
            variant=variant,
        )
        for key, ok in matched.items():
            if not ok:
                capex_by[key] = capex_total * prod_shares.get(key, 0.0)

        opex_by = self._allocate_opex_by_pathway(
            opex_report=opex_report,
            target_opex_total=opex_total,
            fallback_shares=prod_shares,
        )

        year_indices, h2_total_by_year, h2_by_pathway_year = self._build_h2_yearly_totals(loaded_h2)
        available_years = len(year_indices)
        analysis_years = min(project_years, available_years) if available_years > 0 else 0
        if analysis_years <= 0:
            analysis_years = project_years
            h2_total_series = np.full(analysis_years, annual_total, dtype=float)
            for key in ("pem", "soec", "atr"):
                h2_by_pathway_year[key] = np.full(analysis_years, annual_by.get(key, 0.0), dtype=float)
        else:
            h2_total_series = h2_total_by_year[:analysis_years]
            for key in ("pem", "soec", "atr"):
                h2_by_pathway_year[key] = h2_by_pathway_year[key][:analysis_years]
            if project_years > available_years:
                warnings.append(
                    f"Project lifetime ({project_years} years) exceeds available history horizon "
                    f"({available_years} years); discounted LCOH computed on available horizon."
                )

        opex_series = self._extract_opex_yearly_stream(
            opex_report=opex_report,
            variant=variant,
            fallback_annual_opex=opex_total,
            year_count=len(h2_total_series),
        )
        discounted_opex_series = self._discount_series(opex_series, discount_rate)
        discounted_h2_series = self._discount_series(h2_total_series, discount_rate)

        pv_opex = float(np.sum(discounted_opex_series))
        pv_h2_total = float(np.sum(discounted_h2_series))
        df_sum = self._discount_factor_sum(discount_rate, len(h2_total_series))
        lcoh_total = self._safe_div(capex_total + pv_opex, pv_h2_total)

        lcoh_by = {}
        opex_total_alloc = sum(opex_by.values())
        opex_share_by_path = {
            key: (opex_by.get(key, 0.0) / opex_total_alloc) if opex_total_alloc > 0 else prod_shares.get(key, 0.0)
            for key in ("pem", "soec", "atr")
        }
        for key in ("pem", "soec", "atr"):
            pv_h2 = float(np.sum(self._discount_series(h2_by_pathway_year.get(key, np.zeros(len(h2_total_series))), discount_rate)))
            lcoh_by[key] = self._safe_div(
                capex_by.get(key, 0.0) + (pv_opex * opex_share_by_path.get(key, 0.0)),
                pv_h2,
            )

        weighted = 0.0
        if annual_total > 0:
            weighted = sum(lcoh_by[k] * annual_by.get(k, 0.0) for k in lcoh_by) / annual_total

        opex_component_annual = self._allocate_opex_components(
            opex_report=opex_report,
            target_opex_total=opex_total,
        )
        component_share = {
            key: (opex_component_annual.get(key, 0.0) / opex_total) if opex_total > 0 else 0.0
            for key in ("energy", "water", "compression", "other_opex")
        }
        breakdown = {
            "capex": self._safe_div(capex_total, pv_h2_total),
            "opex": self._safe_div(pv_opex, pv_h2_total),
            "energy": self._safe_div(pv_opex * component_share["energy"], pv_h2_total),
            "water": self._safe_div(pv_opex * component_share["water"], pv_h2_total),
            "compression": self._safe_div(pv_opex * component_share["compression"], pv_h2_total),
            "other_opex": self._safe_div(pv_opex * component_share["other_opex"], pv_h2_total),
        }

        opex_by_year = {str(idx + 1): float(val) for idx, val in enumerate(opex_series)}
        h2_by_year = {str(idx + 1): float(val) for idx, val in enumerate(h2_total_series)}

        return LcohReport(
            generated_at=generated_at,
            discount_rate=discount_rate,
            project_lifetime_years=project_years,
            discount_factor_sum=df_sum,
            capex_total=capex_total,
            opex_annual_total=opex_total,
            opex_by_year=opex_by_year,
            annual_h2_total_kg=annual_total,
            annual_h2_by_pathway=annual_by,
            annual_h2_total_kg_by_year=h2_by_year,
            capex_by_pathway=capex_by,
            opex_by_pathway=opex_by,
            lcoh_total=lcoh_total,
            lcoh_by_pathway=lcoh_by,
            lcoh_weighted_plant=weighted,
            discounted_opex_pv=pv_opex,
            discounted_h2_pv=pv_h2_total,
            lcoh_breakdown=breakdown,
            warnings=warnings,
        )

    def generate(self, inputs: LcohInputs) -> LcohReport:
        warnings: list[str] = []
        annual_by, annual_total, prod_shares, loaded_h2 = self._resolve_annual_production(
            history_chunks_dir=inputs.history_chunks_dir,
            opex_report=inputs.opex_report,
            warnings=warnings,
            history_scan_workers=inputs.history_scan_workers,
            history_scan_max_memory_mb=inputs.history_scan_max_memory_mb,
            history_scan_mode=inputs.history_scan_mode,
        )
        capex_total = self._extract_capex_variant_totals(inputs.capex_report, strict=False)["base"]
        opex_total = self._extract_opex_variant_totals(inputs.opex_report, strict=False)["base"]
        generated_at = datetime.now().isoformat()
        return self._build_variant_report(
            variant="base",
            capex_total=capex_total,
            opex_total=opex_total,
            opex_report=inputs.opex_report,
            annual_by=annual_by,
            annual_total=annual_total,
            prod_shares=prod_shares,
            discount_rate=inputs.discount_rate,
            project_years=inputs.project_years,
            loaded_h2=loaded_h2,
            capex_report=inputs.capex_report,
            shared_warnings=warnings,
            generated_at=generated_at,
        )

    def generate_variants(self, inputs: LcohInputs) -> LcohVariantsReport:
        shared_warnings: list[str] = []
        annual_by, annual_total, prod_shares, loaded_h2 = self._resolve_annual_production(
            history_chunks_dir=inputs.history_chunks_dir,
            opex_report=inputs.opex_report,
            warnings=shared_warnings,
            history_scan_workers=inputs.history_scan_workers,
            history_scan_max_memory_mb=inputs.history_scan_max_memory_mb,
            history_scan_mode=inputs.history_scan_mode,
        )

        capex_totals = self._extract_capex_variant_totals(inputs.capex_report, strict=True)
        opex_totals = self._extract_opex_variant_totals(inputs.opex_report, strict=True)

        generated_at = datetime.now().isoformat()
        variants: Dict[str, LcohReport] = {}
        for variant in self.VARIANT_ORDER:
            variants[variant] = self._build_variant_report(
                variant=variant,
                capex_total=capex_totals[variant],
                opex_total=opex_totals[variant],
                opex_report=inputs.opex_report,
                annual_by=annual_by,
                annual_total=annual_total,
                prod_shares=prod_shares,
                discount_rate=inputs.discount_rate,
                project_years=inputs.project_years,
                loaded_h2=loaded_h2,
                capex_report=inputs.capex_report,
                shared_warnings=shared_warnings,
                generated_at=generated_at,
            )

        combined_warnings: list[str] = []
        for warning in shared_warnings:
            if warning not in combined_warnings:
                combined_warnings.append(warning)
        for variant in self.VARIANT_ORDER:
            for warning in variants[variant].warnings:
                if warning not in combined_warnings:
                    combined_warnings.append(warning)

        base_report = variants["base"]
        return LcohVariantsReport(
            generated_at=generated_at,
            discount_rate=inputs.discount_rate,
            project_lifetime_years=inputs.project_years,
            variant_order=list(self.VARIANT_ORDER),
            variants=variants,
            warnings=combined_warnings,
            discount_factor_sum=base_report.discount_factor_sum,
            capex_total=base_report.capex_total,
            opex_annual_total=base_report.opex_annual_total,
            annual_h2_total_kg=base_report.annual_h2_total_kg,
            annual_h2_by_pathway=base_report.annual_h2_by_pathway,
            capex_by_pathway=base_report.capex_by_pathway,
            opex_by_pathway=base_report.opex_by_pathway,
            lcoh_total=base_report.lcoh_total,
            lcoh_by_pathway=base_report.lcoh_by_pathway,
            lcoh_weighted_plant=base_report.lcoh_weighted_plant,
            lcoh_breakdown=base_report.lcoh_breakdown,
        )
