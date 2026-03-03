"""
Discounted LCOH Calculator.

Computes discounted Levelized Cost of Hydrogen (LCOH) for plant and pathways.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from h2_plant.economics.models import CapexReport, BlockCostSummary
from h2_plant.economics.opex_models import OpexReport
from h2_plant.economics.lcoh_models import LcohReport, LcohVariantsReport

logger = logging.getLogger(__name__)


@dataclass
class LcohInputs:
    capex_report: CapexReport
    opex_report: OpexReport
    history_chunks_dir: Path
    discount_rate: float
    project_years: int


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

    def _load_h2_totals(self, chunks_dir: Path) -> Dict[str, float]:
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
        totals = {"pem": 0.0, "soec": 0.0, "atr": 0.0}
        minutes_min = None
        minutes_max = None
        diff_samples = []

        for chunk_file in chunk_files:
            try:
                df = pd.read_parquet(chunk_file, columns=required_cols)
            except OSError as e:
                if getattr(e, "errno", None) == 107:
                    self._raise_disconnected_mount(chunk_file, e)
                raise
            if df.empty:
                continue
            totals["pem"] += float(pd.to_numeric(df["H2_pem_kg"], errors="coerce").fillna(0).sum())
            totals["soec"] += float(pd.to_numeric(df["H2_soec_kg"], errors="coerce").fillna(0).sum())
            totals["atr"] += float(pd.to_numeric(df["H2_atr_kg"], errors="coerce").fillna(0).sum())

            min_val = pd.to_numeric(df["minute"], errors="coerce").min()
            max_val = pd.to_numeric(df["minute"], errors="coerce").max()
            if minutes_min is None or (pd.notna(min_val) and min_val < minutes_min):
                minutes_min = float(min_val)
            if minutes_max is None or (pd.notna(max_val) and max_val > minutes_max):
                minutes_max = float(max_val)
            minute_vals = pd.to_numeric(df["minute"], errors="coerce").dropna().values
            if len(minute_vals) > 1:
                diffs = np.diff(minute_vals)
                if diffs.size:
                    diff_samples.append(np.median(diffs))

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
        return {**totals, "simulation_hours": sim_hours}

    def _resolve_annual_production(
        self,
        history_chunks_dir: Path,
        opex_report: OpexReport,
        warnings: list[str],
    ) -> tuple[Dict[str, float], float, Dict[str, float]]:
        totals = self._load_h2_totals(history_chunks_dir)
        sim_hours = totals.pop("simulation_hours")

        annual_factor = 0.0
        if sim_hours and sim_hours > 0:
            annual_factor = 8760.0 / sim_hours
        else:
            fallback_hours = getattr(opex_report, "simulation_hours", 0.0)
            if fallback_hours and fallback_hours > 0:
                annual_factor = 8760.0 / float(fallback_hours)
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
        return annual_by, annual_total, prod_shares

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
        df_sum = self._discount_factor_sum(discount_rate, project_years)

        pv_h2_total = annual_total * df_sum
        lcoh_total = self._safe_div(capex_total + (opex_total * df_sum), pv_h2_total)

        lcoh_by = {}
        for key in ("pem", "soec", "atr"):
            pv_h2 = annual_by.get(key, 0.0) * df_sum
            lcoh_by[key] = self._safe_div(
                capex_by.get(key, 0.0) + (opex_by.get(key, 0.0) * df_sum),
                pv_h2,
            )

        weighted = 0.0
        if annual_total > 0:
            weighted = sum(lcoh_by[k] * annual_by.get(k, 0.0) for k in lcoh_by) / annual_total

        opex_component_annual = self._allocate_opex_components(
            opex_report=opex_report,
            target_opex_total=opex_total,
        )
        breakdown = {
            "capex": self._safe_div(capex_total, pv_h2_total),
            "opex": self._safe_div(opex_total * df_sum, pv_h2_total),
            "energy": self._safe_div(opex_component_annual["energy"] * df_sum, pv_h2_total),
            "water": self._safe_div(opex_component_annual["water"] * df_sum, pv_h2_total),
            "compression": self._safe_div(opex_component_annual["compression"] * df_sum, pv_h2_total),
            "other_opex": self._safe_div(opex_component_annual["other_opex"] * df_sum, pv_h2_total),
        }

        return LcohReport(
            generated_at=generated_at,
            discount_rate=discount_rate,
            project_lifetime_years=project_years,
            discount_factor_sum=df_sum,
            capex_total=capex_total,
            opex_annual_total=opex_total,
            annual_h2_total_kg=annual_total,
            annual_h2_by_pathway=annual_by,
            capex_by_pathway=capex_by,
            opex_by_pathway=opex_by,
            lcoh_total=lcoh_total,
            lcoh_by_pathway=lcoh_by,
            lcoh_weighted_plant=weighted,
            lcoh_breakdown=breakdown,
            warnings=warnings,
        )

    def generate(self, inputs: LcohInputs) -> LcohReport:
        warnings: list[str] = []
        annual_by, annual_total, prod_shares = self._resolve_annual_production(
            history_chunks_dir=inputs.history_chunks_dir,
            opex_report=inputs.opex_report,
            warnings=warnings,
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
            capex_report=inputs.capex_report,
            shared_warnings=warnings,
            generated_at=generated_at,
        )

    def generate_variants(self, inputs: LcohInputs) -> LcohVariantsReport:
        shared_warnings: list[str] = []
        annual_by, annual_total, prod_shares = self._resolve_annual_production(
            history_chunks_dir=inputs.history_chunks_dir,
            opex_report=inputs.opex_report,
            warnings=shared_warnings,
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
