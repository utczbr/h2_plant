"""
OPEX Generator

Orchestrates OPEX calculation by:
1. Loading configuration from YAML
2. Extracting quantities from simulation history
3. Calculating costs using appropriate strategies
4. Aggregating by category
5. Exporting to JSON/CSV
"""

import csv
import difflib
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Any, List, Callable, Tuple

import numpy as np
import pandas as pd
import yaml

from h2_plant.economics.opex_models import (
    OpexCategory,
    OpexItemConfig,
    OpexResult,
    OpexReport,
)
from h2_plant.economics.opex_strategies import get_opex_strategy
from h2_plant.economics.models import CapexReport

logger = logging.getLogger(__name__)
MINUTES_PER_YEAR = 525600.0
HOURS_PER_YEAR = 8760.0


class OpexGenerator:
    """
    Generate OPEX reports combining simulation data with manual configuration.
    
    Usage:
        generator = OpexGenerator()
        report = generator.generate(
            config_path="opex_config.yaml",
            capex_report=capex_report,
            history_df=history_df,
            output_dir="output/"
        )
    """
    
    def __init__(self):
        self.config: Dict[str, Any] = {}
        self.items: list[OpexItemConfig] = []
        
    def load_config(self, config_path: str) -> None:
        """Load OPEX configuration from YAML file."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"OPEX config not found: {config_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Parse items
        self.items = [
            OpexItemConfig(**item) 
            for item in self.config.get('opex_items', [])
        ]
        
        logger.info(f"Loaded {len(self.items)} OPEX items from {config_path}")

    def _is_primary_labor_item(self, item: OpexItemConfig) -> bool:
        """
        Identify the direct labor source used as base for labor-derived factors.

        This intentionally excludes derived items such as "Laboratory & QC".
        """
        if item.strategy == "turton_labor":
            return True
        if item.strategy != "fixed":
            return False

        name = item.name.strip().lower()
        if "laboratory" in name:
            return False
        if "operating labor" in name:
            return True
        if "direct labor" in name:
            return True
        if "total staffing" in name:
            return True
        return name == "labor"

    def _sorted_items(self) -> List[OpexItemConfig]:
        """Sort items so primary labor appears before labor-dependent factors."""
        return sorted(self.items, key=lambda item: 0 if self._is_primary_labor_item(item) else 1)

    def _evaluate_items_for_fci(
        self,
        *,
        sorted_items: List[OpexItemConfig],
        item_calculator: Callable[[OpexItemConfig, Dict[str, float]], OpexResult],
        fci: float,
    ) -> Tuple[List[OpexResult], float]:
        """
        Evaluate all OPEX items for a given FCI with consistent labor-base handling.
        """
        base_costs = {
            "FCI": float(fci),
            "Labor": 0.0,
            "C_OL": 0.0,
        }
        results: List[OpexResult] = []
        labor_cost = 0.0

        for item in sorted_items:
            result = item_calculator(item, base_costs)
            if self._is_primary_labor_item(item):
                labor_cost = float(result.annual_cost)
                base_costs["Labor"] = labor_cost
                base_costs["C_OL"] = labor_cost
            results.append(result)

        return results, labor_cost

    def _calculate_total_opex_for_fci(
        self,
        *,
        sorted_items: List[OpexItemConfig],
        item_calculator: Callable[[OpexItemConfig, Dict[str, float]], OpexResult],
        fci: float,
    ) -> float:
        """Recompute total OPEX for a supplied FCI using the active config items."""
        results, _ = self._evaluate_items_for_fci(
            sorted_items=sorted_items,
            item_calculator=item_calculator,
            fci=fci,
        )
        return float(sum(float(item.annual_cost) for item in results))

    @staticmethod
    def _to_positive_float(value: Any) -> Optional[float]:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        if parsed <= 0:
            return None
        return parsed

    def _required_variable_items(self) -> List[OpexItemConfig]:
        return [
            item for item in self.items
            if item.strategy == "variable"
            and (
                item.resource_id
                or item.price_resource_id
                or item.pathway_driver_resource_ids
            )
        ]

    @staticmethod
    def _dynamic_pair_key(item: OpexItemConfig) -> Optional[str]:
        if not item.resource_id or not item.price_resource_id:
            return None
        return f"__dynamic_pair__{item.resource_id}__{item.price_resource_id}"

    @staticmethod
    def _annualize_metric(raw_qty: float, metric: str, annualization_factor: float) -> float:
        if metric == "avg":
            return float(raw_qty) * 8760.0
        if metric == "max":
            return float(raw_qty)
        return float(raw_qty) * annualization_factor

    @staticmethod
    def _extract_metric_from_quantities(
        quantities: Dict[str, Dict[str, float]],
        signal_id: str,
        metric: str,
    ) -> float:
        raw_entry = quantities.get(signal_id, {})
        if isinstance(raw_entry, dict):
            return float(raw_entry.get(metric, 0.0) or 0.0)
        return float(raw_entry or 0.0)

    @staticmethod
    def _equivalent_annual_from_yearly(yearly_costs: np.ndarray, year_hours: np.ndarray) -> float:
        total_hours = float(np.nansum(year_hours))
        total_cost = float(np.nansum(yearly_costs))
        if total_hours > 0:
            return total_cost / total_hours * HOURS_PER_YEAR
        return total_cost

    @staticmethod
    def _period_quantity_from_metric(raw_qty: float, metric: str, period_hours: float) -> float:
        if metric == "avg":
            return float(raw_qty) * float(max(period_hours, 0.0))
        if metric == "max":
            return float(raw_qty)
        return float(raw_qty)

    @staticmethod
    def _build_year_hours_from_minutes(
        minute_origin: Optional[float],
        minute_max: Optional[float],
        minute_diffs: List[float],
    ) -> Tuple[List[int], List[float]]:
        if minute_origin is None or minute_max is None:
            return [0], [0.0]

        dt_min = float(np.median(minute_diffs)) if minute_diffs else 1.0
        if not np.isfinite(dt_min) or dt_min <= 0:
            dt_min = 1.0

        sim_end = float(minute_max) + dt_min
        total_span = max(0.0, sim_end - float(minute_origin))
        year_count = int(np.ceil(total_span / MINUTES_PER_YEAR)) if total_span > 0 else 1
        year_indices = list(range(year_count))

        year_hours: List[float] = []
        for year_idx in year_indices:
            year_start = float(minute_origin) + (year_idx * MINUTES_PER_YEAR)
            year_end = year_start + MINUTES_PER_YEAR
            overlap_min = max(0.0, min(sim_end, year_end) - max(float(minute_origin), year_start))
            year_hours.append(overlap_min / 60.0)
        return year_indices, year_hours

    @staticmethod
    def _zero_metric_entry(is_dynamic_pair: bool) -> Dict[str, float]:
        if is_dynamic_pair:
            return {"sum_product": 0.0}
        return {"sum": 0.0, "max": 0.0, "avg": 0.0}

    @staticmethod
    def _resolve_column_name(resource_id: str, all_cols: List[str]) -> Optional[str]:
        """Resolve a configured resource_id to a concrete history column."""
        target = resource_id.lower()
        exact_matches = [col for col in all_cols if col.lower() == target]
        if exact_matches:
            return exact_matches[0]

        contains_matches = [col for col in all_cols if target in col.lower()]
        if contains_matches:
            return contains_matches[0]
        return None

    @staticmethod
    def _closest_column_matches(resource_id: str, all_cols: List[str], limit: int = 5) -> List[str]:
        if not all_cols:
            return []
        lowered_map = {col.lower(): col for col in all_cols}
        close = difflib.get_close_matches(
            resource_id.lower(),
            list(lowered_map.keys()),
            n=limit,
            cutoff=0.4,
        )
        if close:
            return [lowered_map[c] for c in close]

        tokens = [tok for tok in resource_id.lower().replace("_", " ").split() if tok]
        partial = [col for col in all_cols if any(tok in col.lower() for tok in tokens)]
        return partial[:limit]

    def _resolve_required_columns(
        self,
        *,
        all_cols: List[str],
        required_items: List[OpexItemConfig],
        source_label: str,
        strict: bool,
    ) -> Dict[str, str]:
        col_map: Dict[str, str] = {}
        missing: Dict[str, List[str]] = {}

        required_signals: set[str] = set()
        for item in required_items:
            if item.resource_id:
                required_signals.add(item.resource_id)
            if item.price_resource_id:
                required_signals.add(item.price_resource_id)
            if item.pathway_driver_resource_ids:
                for signal in item.pathway_driver_resource_ids.values():
                    if signal:
                        required_signals.add(signal)

        for signal_id in sorted(required_signals):
            resolved = self._resolve_column_name(signal_id, all_cols)
            if resolved is None:
                missing[signal_id] = self._closest_column_matches(signal_id, all_cols)
                continue
            col_map[signal_id] = resolved

        if strict and missing:
            missing_descriptions = []
            for resource_id, suggestions in missing.items():
                if suggestions:
                    missing_descriptions.append(
                        f"{resource_id} (closest: {', '.join(suggestions)})"
                    )
                else:
                    missing_descriptions.append(f"{resource_id} (closest: none)")
            raise ValueError(
                "Missing configured OPEX variable signals in "
                f"{source_label}: {', '.join(missing_descriptions)}"
            )

        return col_map

    @staticmethod
    def _infer_simulation_hours_from_minutes(minute_series: pd.Series) -> Optional[float]:
        minute_values = pd.to_numeric(minute_series, errors="coerce").dropna()
        if minute_values.empty:
            return None

        minutes_min = float(minute_values.min())
        minutes_max = float(minute_values.max())
        if minutes_max < minutes_min:
            return None

        simulation_hours = (minutes_max - minutes_min) / 60.0

        if len(minute_values) > 1:
            diffs = minute_values.diff().dropna()
            positive_diffs = diffs[diffs > 0]
            if not positive_diffs.empty:
                simulation_hours += float(positive_diffs.median()) / 60.0

        if simulation_hours <= 0:
            return None
        return simulation_hours

    def _resolve_simulation_hours_for_annualization(
        self,
        *,
        configured_hours: float,
        inferred_hours: Optional[float],
        source_label: str,
    ) -> float:
        configured = float(configured_hours) if configured_hours is not None else 0.0
        inferred = float(inferred_hours) if inferred_hours is not None else 0.0

        if inferred > 0:
            if configured <= 0:
                logger.warning(
                    "Simulation hours for OPEX annualization not configured or non-positive "
                    "(configured=%s). Using inferred span from %s: %.6f h.",
                    configured_hours,
                    source_label,
                    inferred,
                )
                return inferred
            if abs(inferred - configured) > 1e-6:
                logger.warning(
                    "Simulation hours mismatch for OPEX annualization from %s: "
                    "configured=%.6f h, inferred=%.6f h. Using inferred span.",
                    source_label,
                    configured,
                    inferred,
                )
                return inferred
            return configured

        return configured if configured > 0 else 8760.0

    def _apply_uncertainty_bands(
        self,
        report: OpexReport,
        capex_report: Optional[CapexReport],
    ) -> bool:
        """
        Apply low/high OPEX uncertainty bands using CAPEX AACE class factors.

        If CAPEX context is missing, low/high are intentionally omitted.
        """
        if capex_report is None:
            logger.warning(
                "CAPEX report not provided; OPEX uncertainty bands (low/high) will be omitted."
            )
            return False

        cost_class = getattr(capex_report, "overall_cost_class", None)
        if cost_class is None:
            logger.warning(
                "CAPEX cost class not available; OPEX uncertainty bands (low/high) will be omitted."
            )
            return False

        try:
            low_factor, high_factor = cost_class.accuracy_range
        except Exception as e:
            logger.warning(f"Could not derive AACE accuracy range for OPEX uncertainty: {e}")
            return False

        try:
            report.total_opex_low = float(report.total_opex) * float(low_factor)
            report.total_opex_high = float(report.total_opex) * float(high_factor)
        except (TypeError, ValueError) as e:
            logger.warning(f"Could not compute OPEX uncertainty bands: {e}")
            return False
        return True

    def _populate_opex_variants(
        self,
        *,
        report: OpexReport,
        capex_report: Optional[CapexReport],
        sorted_items: List[OpexItemConfig],
        item_calculator: Callable[[OpexItemConfig, Dict[str, float]], OpexResult],
    ) -> None:
        """
        Populate OPEX low/high using CAPEX low/high recomputation, with AACE fallback.
        """
        report.total_opex_low = None
        report.total_opex_high = None

        if capex_report is None:
            self._apply_uncertainty_bands(report, capex_report)
            return

        capex_low = self._to_positive_float(getattr(capex_report, "total_installed_cost_low", None))
        capex_high = self._to_positive_float(getattr(capex_report, "total_installed_cost_high", None))

        if capex_low is not None and capex_high is not None:
            report.total_opex_low = self._calculate_total_opex_for_fci(
                sorted_items=sorted_items,
                item_calculator=item_calculator,
                fci=capex_low,
            )
            report.total_opex_high = self._calculate_total_opex_for_fci(
                sorted_items=sorted_items,
                item_calculator=item_calculator,
                fci=capex_high,
            )
            logger.info(
                "Computed OPEX low/high by recomputing config with CAPEX low/high "
                "(low=%s, high=%s).",
                f"{capex_low:,.0f}",
                f"{capex_high:,.0f}",
            )
            return

        logger.warning(
            "CAPEX low/high are missing or non-positive; falling back to AACE-based "
            "OPEX uncertainty bands."
        )
        self._apply_uncertainty_bands(report, capex_report)

    @staticmethod
    def _populate_cashflow_fields(report: OpexReport) -> None:
        report.total_opex_cashflow = float(report.total_opex) - float(report.total_credit_cost)
        if report.total_opex_low is not None:
            report.total_opex_cashflow_low = float(report.total_opex_low) - float(report.total_credit_cost)
        else:
            report.total_opex_cashflow_low = None
        if report.total_opex_high is not None:
            report.total_opex_cashflow_high = float(report.total_opex_high) - float(report.total_credit_cost)
        else:
            report.total_opex_cashflow_high = None
    
    def generate(
        self,
        config_path: str,
        capex_report: Optional[CapexReport] = None,
        history_df: Optional[pd.DataFrame] = None,
        output_dir: Optional[str] = None,
        simulation_hours: float = 8760.0,
    ) -> OpexReport:
        """
        Generate OPEX report.
        
        Args:
            config_path: Path to opex_config.yaml
            capex_report: Previous CAPEX report (for FCI reference)
            history_df: Simulation history DataFrame
            output_dir: Directory for output files (JSON, CSV)
            simulation_hours: Hours of simulation data (for annualization)
            
        Returns:
            OpexReport with calculated costs
        """
        # Load configuration
        self.load_config(config_path)

        required_items = self._required_variable_items()
        resolved_columns: Dict[str, str] = {}
        history_available = history_df is not None
        if history_available:
            resolved_columns = self._resolve_required_columns(
                all_cols=list(history_df.columns),
                required_items=required_items,
                source_label="history dataframe",
                strict=True,
            )
        
        # Initialize report
        report = OpexReport(
            scenario_name=self.config.get('scenario_name', 'default'),
            simulation_hours=simulation_hours,
            annualization_factor=8760.0 / simulation_hours if simulation_hours > 0 else 1.0,
        )
        
        # Get FCI from CAPEX report
        if capex_report:
            report.fci = capex_report.total_installed_cost or capex_report.total_C_BM or 0.0
            logger.info(f"Using FCI from CAPEX: ${report.fci:,.0f}")
        
        sorted_items = self._sorted_items()

        def item_calculator(item: OpexItemConfig, base_costs: Dict[str, float]) -> OpexResult:
            return self._calculate_item(
                item=item,
                history_df=history_df,
                resolved_columns=resolved_columns,
                history_available=history_available,
                base_costs=base_costs,
                annualization_factor=report.annualization_factor,
            )

        base_results, labor_cost = self._evaluate_items_for_fci(
            sorted_items=sorted_items,
            item_calculator=item_calculator,
            fci=report.fci,
        )
        report.items.extend(base_results)
        report.labor_cost = labor_cost
        
        # Calculate H2 production from history
        if history_df is not None:
            report.annual_h2_production_kg = self._extract_h2_production(
                history_df, report.annualization_factor
            )
        
        # Calculate totals
        report.calculate_totals()
        self._populate_opex_variants(
            report=report,
            capex_report=capex_report,
            sorted_items=sorted_items,
            item_calculator=item_calculator,
        )
        self._populate_cashflow_fields(report)
        
        # Export if output_dir provided
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            self._export_json(report, output_path / "opex_report.json")
            self._export_csv(report, output_path / "opex_report.csv")
            
            logger.info(f"OPEX report generated: {output_dir}")
            logger.info(f"  Total OPEX: ${report.total_opex:,.0f}/year")
            if report.total_opex_low is not None and report.total_opex_high is not None:
                logger.info(
                    f"  OPEX Range: ${report.total_opex_low:,.0f} - ${report.total_opex_high:,.0f}"
                )
            logger.info(f"  Variable: ${report.total_variable_cost:,.0f}")
            logger.info(f"  Fixed: ${report.total_fixed_cost:,.0f}")
            logger.info(f"  Maintenance: ${report.total_maintenance_cost:,.0f}")
        
        return report
    
    def generate_streaming(
        self,
        config_path: str,
        csv_path: Path,
        capex_report: Optional[CapexReport] = None,
        output_dir: Optional[str] = None,
        simulation_hours: float = 8760.0,
        chunk_size: int = 50_000,
    ) -> OpexReport:
        """
        Generate OPEX report using streaming CSV processing.
        
        This method reads the simulation history in chunks with column filtering,
        using only ~100 MB memory instead of loading the full file (can be 7+ GB).
        
        Args:
            config_path: Path to opex_config.yaml
            csv_path: Path to simulation_history.csv
            capex_report: Previous CAPEX report (for FCI reference)
            output_dir: Directory for output files (JSON, CSV)
            simulation_hours: Hours of simulation data (for annualization)
            chunk_size: Rows per chunk (default 50,000)
            
        Returns:
            OpexReport with calculated costs
        """
        import time
        start_time = time.time()
        
        # Load configuration
        self.load_config(config_path)

        # Extract quantities from CSV using streaming
        csv_path = Path(csv_path)
        history_available = csv_path.exists()
        if csv_path.exists():
            (
                quantities,
                inferred_hours,
                quantities_by_year,
                year_indices,
                year_hours,
            ) = self._extract_quantities_streaming(
                csv_path=csv_path,
                required_items=self._required_variable_items(),
                chunk_size=chunk_size,
            )
            logger.info(f"Extracted {len(quantities)} quantities via streaming")
        else:
            quantities = {}
            inferred_hours = None
            quantities_by_year = {}
            year_indices = [0]
            year_hours = [0.0]
            logger.warning(f"CSV not found: {csv_path}")

        effective_simulation_hours = self._resolve_simulation_hours_for_annualization(
            configured_hours=simulation_hours,
            inferred_hours=inferred_hours,
            source_label=f"CSV minute column ({csv_path})",
        )
        annualization_factor = (
            8760.0 / effective_simulation_hours if effective_simulation_hours > 0 else 1.0
        )

        # Initialize report
        report = OpexReport(
            scenario_name=self.config.get('scenario_name', 'default'),
            simulation_hours=effective_simulation_hours,
            annualization_factor=annualization_factor,
        )

        if not year_indices:
            year_indices = [0]
        if len(year_hours) != len(year_indices):
            year_hours = [0.0 for _ in year_indices]
        year_hours_arr = np.array(year_hours, dtype=float)
        if float(np.nansum(year_hours_arr)) <= 0:
            year_hours_arr = np.zeros(len(year_indices), dtype=float)
            year_hours_arr[0] = float(effective_simulation_hours) if effective_simulation_hours > 0 else HOURS_PER_YEAR

        # Get FCI from CAPEX report
        if capex_report:
            report.fci = capex_report.total_installed_cost or capex_report.total_C_BM or 0.0
            logger.info(f"Using FCI from CAPEX: ${report.fci:,.0f}")
        
        sorted_items = self._sorted_items()
        base_eval = self._evaluate_streaming_yearly_for_fci(
            sorted_items=sorted_items,
            fci=report.fci,
            quantities=quantities,
            quantities_by_year=quantities_by_year,
            history_available=history_available,
            annualization_factor=annualization_factor,
            year_indices=year_indices,
            year_hours=year_hours_arr,
            keep_scalar_results=True,
        )
        report.items.extend(base_eval["scalar_results"])
        report.labor_cost = float(base_eval["labor_cost"])

        # Yearly report fields
        report.year_index = [int(y) + 1 for y in year_indices]
        report.year_hours = [float(v) for v in year_hours_arr]
        report.item_annual_cost_by_year = base_eval["item_costs_by_year"]
        report.total_variable_cost_by_year = [float(v) for v in base_eval["variable_by_year"]]
        report.total_fixed_cost_by_year = [float(v) for v in base_eval["fixed_by_year"]]
        report.total_maintenance_cost_by_year = [float(v) for v in base_eval["maintenance_by_year"]]
        report.total_opex_by_year = [float(v) for v in base_eval["total_by_year"]]
        report.total_opex_cashflow_by_year = [float(v) for v in base_eval["cashflow_by_year"]]

        # H2 yearly production from cumulative increments
        h2_year_map = quantities_by_year.get("cumulative_h2_kg", {})
        h2_by_year = np.array(
            [
                float((h2_year_map.get(int(y), {}) or {}).get("sum", 0.0) or 0.0)
                for y in year_indices
            ],
            dtype=float,
        )
        report.annual_h2_production_kg_by_year = [float(v) for v in h2_by_year]
        report.annual_h2_production_kg = self._equivalent_annual_from_yearly(h2_by_year, year_hours_arr)
        if report.annual_h2_production_kg <= 0:
            # Fallback for legacy datasets missing cumulative_h2_kg
            h2_entry = quantities.get('cumulative_h2_kg', 0.0)
            if isinstance(h2_entry, dict):
                report.annual_h2_production_kg = float(h2_entry.get('sum', 0.0) or 0.0) * annualization_factor
            else:
                report.annual_h2_production_kg = float(h2_entry or 0.0) * annualization_factor

        # Backward-compatible scalar totals from yearly series (equivalent annual)
        report.total_variable_cost = self._equivalent_annual_from_yearly(base_eval["variable_by_year"], year_hours_arr)
        report.total_fixed_cost = self._equivalent_annual_from_yearly(base_eval["fixed_by_year"], year_hours_arr)
        report.total_maintenance_cost = self._equivalent_annual_from_yearly(base_eval["maintenance_by_year"], year_hours_arr)
        report.total_opex = self._equivalent_annual_from_yearly(base_eval["total_by_year"], year_hours_arr)
        report.total_credit_cost = self._equivalent_annual_from_yearly(base_eval["credit_by_year"], year_hours_arr)
        report.total_opex_cashflow = self._equivalent_annual_from_yearly(base_eval["cashflow_by_year"], year_hours_arr)
        report.opex_per_kg_h2 = (
            float(report.total_opex) / float(report.annual_h2_production_kg)
            if float(report.annual_h2_production_kg) > 0
            else 0.0
        )

        # Low/high variants with yearly series
        report.total_opex_low = None
        report.total_opex_high = None
        report.total_opex_cashflow_low = None
        report.total_opex_cashflow_high = None
        report.total_opex_low_by_year = None
        report.total_opex_high_by_year = None
        report.total_opex_cashflow_low_by_year = None
        report.total_opex_cashflow_high_by_year = None

        capex_low = self._to_positive_float(getattr(capex_report, "total_installed_cost_low", None)) if capex_report else None
        capex_high = self._to_positive_float(getattr(capex_report, "total_installed_cost_high", None)) if capex_report else None

        if capex_low is not None and capex_high is not None:
            low_eval = self._evaluate_streaming_yearly_for_fci(
                sorted_items=sorted_items,
                fci=capex_low,
                quantities=quantities,
                quantities_by_year=quantities_by_year,
                history_available=history_available,
                annualization_factor=annualization_factor,
                year_indices=year_indices,
                year_hours=year_hours_arr,
                keep_scalar_results=False,
            )
            high_eval = self._evaluate_streaming_yearly_for_fci(
                sorted_items=sorted_items,
                fci=capex_high,
                quantities=quantities,
                quantities_by_year=quantities_by_year,
                history_available=history_available,
                annualization_factor=annualization_factor,
                year_indices=year_indices,
                year_hours=year_hours_arr,
                keep_scalar_results=False,
            )

            report.total_opex_low_by_year = [float(v) for v in low_eval["total_by_year"]]
            report.total_opex_high_by_year = [float(v) for v in high_eval["total_by_year"]]
            report.total_opex_cashflow_low_by_year = [float(v) for v in low_eval["cashflow_by_year"]]
            report.total_opex_cashflow_high_by_year = [float(v) for v in high_eval["cashflow_by_year"]]
            report.total_opex_low = self._equivalent_annual_from_yearly(low_eval["total_by_year"], year_hours_arr)
            report.total_opex_high = self._equivalent_annual_from_yearly(high_eval["total_by_year"], year_hours_arr)
            report.total_opex_cashflow_low = self._equivalent_annual_from_yearly(low_eval["cashflow_by_year"], year_hours_arr)
            report.total_opex_cashflow_high = self._equivalent_annual_from_yearly(high_eval["cashflow_by_year"], year_hours_arr)
            logger.info(
                "Computed OPEX low/high by recomputing config with CAPEX low/high "
                "(low=%s, high=%s).",
                f"{capex_low:,.0f}",
                f"{capex_high:,.0f}",
            )
        else:
            self._apply_uncertainty_bands(report, capex_report)
            if report.total_opex > 0:
                low_scale = (float(report.total_opex_low) / float(report.total_opex)) if report.total_opex_low is not None else 0.0
                high_scale = (float(report.total_opex_high) / float(report.total_opex)) if report.total_opex_high is not None else 0.0
                base_total = np.array(base_eval["total_by_year"], dtype=float)
                base_cash = np.array(base_eval["cashflow_by_year"], dtype=float)
                if report.total_opex_low is not None:
                    low_total = base_total * low_scale
                    low_cash = base_cash * low_scale
                    report.total_opex_low_by_year = [float(v) for v in low_total]
                    report.total_opex_cashflow_low_by_year = [float(v) for v in low_cash]
                    report.total_opex_cashflow_low = self._equivalent_annual_from_yearly(low_cash, year_hours_arr)
                if report.total_opex_high is not None:
                    high_total = base_total * high_scale
                    high_cash = base_cash * high_scale
                    report.total_opex_high_by_year = [float(v) for v in high_total]
                    report.total_opex_cashflow_high_by_year = [float(v) for v in high_cash]
                    report.total_opex_cashflow_high = self._equivalent_annual_from_yearly(high_cash, year_hours_arr)
        
        # Export if output_dir provided
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            self._export_json(report, output_path / "opex_report.json")
            self._export_csv(report, output_path / "opex_report.csv")
            
            elapsed = time.time() - start_time
            logger.info(f"OPEX report generated (streaming) in {elapsed:.1f}s")
            logger.info(f"  Total OPEX: ${report.total_opex:,.0f}/year")
            if report.total_opex_low is not None and report.total_opex_high is not None:
                logger.info(
                    f"  OPEX Range: ${report.total_opex_low:,.0f} - ${report.total_opex_high:,.0f}"
                )
            logger.info(f"  Variable: ${report.total_variable_cost:,.0f}")
            logger.info(f"  Fixed: ${report.total_fixed_cost:,.0f}")
            logger.info(f"  Maintenance: ${report.total_maintenance_cost:,.0f}")
        
        return report

    def generate_streaming_parquet(
        self,
        config_path: str,
        chunks_dir: Path,
        capex_report: Optional[CapexReport] = None,
        output_dir: Optional[str] = None,
        simulation_hours: float = 8760.0,
    ) -> OpexReport:
        """
        Generate OPEX report using streaming Parquet history (history_chunks).
        
        Args:
            config_path: Path to opex_config.yaml
            chunks_dir: Path to history_chunks/ with chunk_*.parquet
            capex_report: Previous CAPEX report (for FCI reference)
            output_dir: Directory for output files (JSON, CSV)
            simulation_hours: Hours of simulation data (for annualization)
        """
        import time
        start_time = time.time()

        # Load configuration
        self.load_config(config_path)

        chunks_path = Path(chunks_dir)
        history_available = chunks_path.exists() and any(chunks_path.glob("chunk_*.parquet"))

        (
            quantities,
            inferred_hours,
            quantities_by_year,
            year_indices,
            year_hours,
        ) = self._extract_quantities_parquet_chunks(
            chunks_dir=chunks_dir,
            required_items=self._required_variable_items(),
        )
        logger.info(f"Extracted {len(quantities)} quantities from Parquet chunks")

        effective_simulation_hours = self._resolve_simulation_hours_for_annualization(
            configured_hours=simulation_hours,
            inferred_hours=inferred_hours,
            source_label=f"Parquet minute column ({chunks_dir})",
        )
        annualization_factor = (
            8760.0 / effective_simulation_hours if effective_simulation_hours > 0 else 1.0
        )

        # Initialize report
        report = OpexReport(
            scenario_name=self.config.get('scenario_name', 'default'),
            simulation_hours=effective_simulation_hours,
            annualization_factor=annualization_factor,
        )

        if not year_indices:
            year_indices = [0]
        if len(year_hours) != len(year_indices):
            year_hours = [0.0 for _ in year_indices]
        year_hours_arr = np.array(year_hours, dtype=float)
        if float(np.nansum(year_hours_arr)) <= 0:
            year_hours_arr = np.zeros(len(year_indices), dtype=float)
            year_hours_arr[0] = float(effective_simulation_hours) if effective_simulation_hours > 0 else HOURS_PER_YEAR

        # Get FCI from CAPEX report
        if capex_report:
            report.fci = capex_report.total_installed_cost or capex_report.total_C_BM or 0.0
            logger.info(f"Using FCI from CAPEX: ${report.fci:,.0f}")

        sorted_items = self._sorted_items()
        base_eval = self._evaluate_streaming_yearly_for_fci(
            sorted_items=sorted_items,
            fci=report.fci,
            quantities=quantities,
            quantities_by_year=quantities_by_year,
            history_available=history_available,
            annualization_factor=annualization_factor,
            year_indices=year_indices,
            year_hours=year_hours_arr,
            keep_scalar_results=True,
        )
        report.items.extend(base_eval["scalar_results"])
        report.labor_cost = float(base_eval["labor_cost"])

        report.year_index = [int(y) + 1 for y in year_indices]
        report.year_hours = [float(v) for v in year_hours_arr]
        report.item_annual_cost_by_year = base_eval["item_costs_by_year"]
        report.total_variable_cost_by_year = [float(v) for v in base_eval["variable_by_year"]]
        report.total_fixed_cost_by_year = [float(v) for v in base_eval["fixed_by_year"]]
        report.total_maintenance_cost_by_year = [float(v) for v in base_eval["maintenance_by_year"]]
        report.total_opex_by_year = [float(v) for v in base_eval["total_by_year"]]
        report.total_opex_cashflow_by_year = [float(v) for v in base_eval["cashflow_by_year"]]

        h2_year_map = quantities_by_year.get("cumulative_h2_kg", {})
        h2_by_year = np.array(
            [
                float((h2_year_map.get(int(y), {}) or {}).get("sum", 0.0) or 0.0)
                for y in year_indices
            ],
            dtype=float,
        )
        report.annual_h2_production_kg_by_year = [float(v) for v in h2_by_year]
        report.annual_h2_production_kg = self._equivalent_annual_from_yearly(h2_by_year, year_hours_arr)
        if report.annual_h2_production_kg <= 0:
            h2_entry = quantities.get('cumulative_h2_kg', 0.0)
            if isinstance(h2_entry, dict):
                report.annual_h2_production_kg = float(h2_entry.get('sum', 0.0) or 0.0) * annualization_factor
            else:
                report.annual_h2_production_kg = float(h2_entry or 0.0) * annualization_factor

        report.total_variable_cost = self._equivalent_annual_from_yearly(base_eval["variable_by_year"], year_hours_arr)
        report.total_fixed_cost = self._equivalent_annual_from_yearly(base_eval["fixed_by_year"], year_hours_arr)
        report.total_maintenance_cost = self._equivalent_annual_from_yearly(base_eval["maintenance_by_year"], year_hours_arr)
        report.total_opex = self._equivalent_annual_from_yearly(base_eval["total_by_year"], year_hours_arr)
        report.total_credit_cost = self._equivalent_annual_from_yearly(base_eval["credit_by_year"], year_hours_arr)
        report.total_opex_cashflow = self._equivalent_annual_from_yearly(base_eval["cashflow_by_year"], year_hours_arr)
        report.opex_per_kg_h2 = (
            float(report.total_opex) / float(report.annual_h2_production_kg)
            if float(report.annual_h2_production_kg) > 0
            else 0.0
        )

        report.total_opex_low = None
        report.total_opex_high = None
        report.total_opex_cashflow_low = None
        report.total_opex_cashflow_high = None
        report.total_opex_low_by_year = None
        report.total_opex_high_by_year = None
        report.total_opex_cashflow_low_by_year = None
        report.total_opex_cashflow_high_by_year = None

        capex_low = self._to_positive_float(getattr(capex_report, "total_installed_cost_low", None)) if capex_report else None
        capex_high = self._to_positive_float(getattr(capex_report, "total_installed_cost_high", None)) if capex_report else None

        if capex_low is not None and capex_high is not None:
            low_eval = self._evaluate_streaming_yearly_for_fci(
                sorted_items=sorted_items,
                fci=capex_low,
                quantities=quantities,
                quantities_by_year=quantities_by_year,
                history_available=history_available,
                annualization_factor=annualization_factor,
                year_indices=year_indices,
                year_hours=year_hours_arr,
                keep_scalar_results=False,
            )
            high_eval = self._evaluate_streaming_yearly_for_fci(
                sorted_items=sorted_items,
                fci=capex_high,
                quantities=quantities,
                quantities_by_year=quantities_by_year,
                history_available=history_available,
                annualization_factor=annualization_factor,
                year_indices=year_indices,
                year_hours=year_hours_arr,
                keep_scalar_results=False,
            )

            report.total_opex_low_by_year = [float(v) for v in low_eval["total_by_year"]]
            report.total_opex_high_by_year = [float(v) for v in high_eval["total_by_year"]]
            report.total_opex_cashflow_low_by_year = [float(v) for v in low_eval["cashflow_by_year"]]
            report.total_opex_cashflow_high_by_year = [float(v) for v in high_eval["cashflow_by_year"]]
            report.total_opex_low = self._equivalent_annual_from_yearly(low_eval["total_by_year"], year_hours_arr)
            report.total_opex_high = self._equivalent_annual_from_yearly(high_eval["total_by_year"], year_hours_arr)
            report.total_opex_cashflow_low = self._equivalent_annual_from_yearly(low_eval["cashflow_by_year"], year_hours_arr)
            report.total_opex_cashflow_high = self._equivalent_annual_from_yearly(high_eval["cashflow_by_year"], year_hours_arr)
            logger.info(
                "Computed OPEX low/high by recomputing config with CAPEX low/high "
                "(low=%s, high=%s).",
                f"{capex_low:,.0f}",
                f"{capex_high:,.0f}",
            )
        else:
            self._apply_uncertainty_bands(report, capex_report)
            if report.total_opex > 0:
                low_scale = (float(report.total_opex_low) / float(report.total_opex)) if report.total_opex_low is not None else 0.0
                high_scale = (float(report.total_opex_high) / float(report.total_opex)) if report.total_opex_high is not None else 0.0
                base_total = np.array(base_eval["total_by_year"], dtype=float)
                base_cash = np.array(base_eval["cashflow_by_year"], dtype=float)
                if report.total_opex_low is not None:
                    low_total = base_total * low_scale
                    low_cash = base_cash * low_scale
                    report.total_opex_low_by_year = [float(v) for v in low_total]
                    report.total_opex_cashflow_low_by_year = [float(v) for v in low_cash]
                    report.total_opex_cashflow_low = self._equivalent_annual_from_yearly(low_cash, year_hours_arr)
                if report.total_opex_high is not None:
                    high_total = base_total * high_scale
                    high_cash = base_cash * high_scale
                    report.total_opex_high_by_year = [float(v) for v in high_total]
                    report.total_opex_cashflow_high_by_year = [float(v) for v in high_cash]
                    report.total_opex_cashflow_high = self._equivalent_annual_from_yearly(high_cash, year_hours_arr)

        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            self._export_json(report, output_path / "opex_report.json")
            self._export_csv(report, output_path / "opex_report.csv")

            elapsed = time.time() - start_time
            logger.info(f"OPEX report generated (parquet streaming) in {elapsed:.1f}s")
            logger.info(f"  Total OPEX: ${report.total_opex:,.0f}/year")
            if report.total_opex_low is not None and report.total_opex_high is not None:
                logger.info(
                    f"  OPEX Range: ${report.total_opex_low:,.0f} - ${report.total_opex_high:,.0f}"
                )
            logger.info(f"  Variable: ${report.total_variable_cost:,.0f}")
            logger.info(f"  Fixed: ${report.total_fixed_cost:,.0f}")
            logger.info(f"  Maintenance: ${report.total_maintenance_cost:,.0f}")

        return report
    
    def _extract_quantities_streaming(
        self,
        csv_path: Path,
        required_items: List[OpexItemConfig],
        chunk_size: int = 50_000,
    ) -> Tuple[
        Dict[str, Dict[str, float]],
        Optional[float],
        Dict[str, Dict[int, Dict[str, float]]],
        List[int],
        List[float],
    ]:
        """
        Extract quantities using chunked streaming with column filtering.
        
        Memory-efficient: reads only needed columns, processes in chunks.
        
        Args:
            csv_path: Path to simulation_history.csv
            required_items: List of OPEX items requiring simulation data
            chunk_size: Rows per chunk
            
        Returns:
            Tuple:
              (quantities_by_resource_id,
               inferred_simulation_hours,
               quantities_by_resource_id_by_year,
               year_indices_zero_based,
               year_hours)
        """
        import gc
        
        # 1. Scan header to find matching columns
        header_df = pd.read_csv(csv_path, nrows=0)
        all_cols = list(header_df.columns)

        # Map configured resource_id -> actual column name with strict policy
        col_map = self._resolve_required_columns(
            all_cols=all_cols,
            required_items=required_items,
            source_label=f"CSV history header ({csv_path})",
            strict=True,
        )

        # Always include H2 production column for per-kg OPEX metric
        h2_col = self._resolve_column_name("cumulative_h2_kg", all_cols)
        if h2_col is not None:
            col_map['cumulative_h2_kg'] = h2_col
        minute_col = self._resolve_column_name("minute", all_cols)
        if minute_col is not None:
            col_map['minute'] = minute_col

        dynamic_pairs: Dict[str, Tuple[str, str]] = {}
        for item in required_items:
            pair_key = self._dynamic_pair_key(item)
            if pair_key and item.resource_id and item.price_resource_id:
                dynamic_pairs[pair_key] = (item.resource_id, item.price_resource_id)
        
        # 2. Select only needed columns
        needed_cols = list(set(col_map.values()))
        if not needed_cols:
            logger.warning("No matching columns found for OPEX extraction")
            return {}, None, {}, [0], [0.0]
        
        logger.info(f"Streaming OPEX: reading {len(needed_cols)} columns in chunks of {chunk_size}")
        
        # 3. Initialize accumulators
        accumulators: Dict[str, Dict[str, float]] = {}
        for res_id, col in col_map.items():
            is_cumulative = 'cumulative' in col.lower()
            accumulators[res_id] = {
                "sum": 0.0,
                "max": float("-inf"),
                "count": 0.0,
                "last": 0.0,
                "is_cumulative": True if is_cumulative else False,
            }

        yearly_accumulators: Dict[str, Dict[int, Dict[str, float]]] = {
            res_id: {} for res_id in accumulators if res_id != "minute"
        }

        rows_processed = 0
        minutes_min: Optional[float] = None
        minutes_max: Optional[float] = None
        minute_diffs: List[float] = []
        last_minute: Optional[float] = None
        dynamic_pair_products = {pair_key: 0.0 for pair_key in dynamic_pairs}
        dynamic_pair_products_by_year: Dict[str, Dict[int, float]] = {
            pair_key: {} for pair_key in dynamic_pairs
        }
        minute_origin: Optional[float] = None
        year_indices_seen: set[int] = set()
        
        # 4. Process chunks
        try:
            for chunk in pd.read_csv(csv_path, usecols=needed_cols, chunksize=chunk_size):
                rows_processed += len(chunk)
                minute_values: Optional[np.ndarray] = None
                year_idx_values: Optional[np.ndarray] = None

                minute_signal_col = col_map.get("minute")
                if minute_signal_col and minute_signal_col in chunk.columns:
                    minute_series = pd.to_numeric(chunk[minute_signal_col], errors="coerce")
                    minute_values = minute_series.to_numpy(dtype=float)
                    minute_valid = minute_values[np.isfinite(minute_values)]
                    minute_series_clean = pd.Series(minute_valid)
                    minute_series = minute_series_clean
                    if not minute_series.empty:
                        if minute_origin is None:
                            minute_origin = float(minute_series.iloc[0])
                        chunk_minutes = minute_series.values.astype(float)
                        chunk_min = float(np.min(chunk_minutes))
                        chunk_max = float(np.max(chunk_minutes))
                        if minutes_min is None or chunk_min < minutes_min:
                            minutes_min = chunk_min
                        if minutes_max is None or chunk_max > minutes_max:
                            minutes_max = chunk_max

                        if last_minute is not None:
                            cross_diff = chunk_minutes[0] - last_minute
                            if cross_diff > 0:
                                minute_diffs.append(float(cross_diff))
                        if chunk_minutes.size > 1:
                            diffs = np.diff(chunk_minutes)
                            positive = diffs[diffs > 0]
                            if positive.size > 0:
                                minute_diffs.append(float(np.median(positive)))
                        last_minute = float(chunk_minutes[-1])
                    if minute_origin is not None:
                        year_idx_values = np.full(len(minute_values), -1, dtype=int)
                        valid_mask = np.isfinite(minute_values)
                        year_idx_values[valid_mask] = np.floor(
                            (minute_values[valid_mask] - minute_origin) / MINUTES_PER_YEAR
                        ).astype(int)
                        year_idx_values[year_idx_values < 0] = 0
                        if valid_mask.any():
                            year_indices_seen.update(int(v) for v in np.unique(year_idx_values[valid_mask]))
                
                for res_id, col in col_map.items():
                    if res_id == "minute":
                        continue
                    if col in chunk.columns:
                        series = pd.to_numeric(chunk[col], errors="coerce")
                        is_cumulative = bool(accumulators[res_id]["is_cumulative"])
                        if is_cumulative:
                            if len(series) > 0:
                                accumulators[res_id]["last"] = series.iloc[-1]
                                series_max = series.max()
                                if pd.notna(series_max):
                                    accumulators[res_id]["max"] = max(accumulators[res_id]["max"], series_max)
                        else:
                            series_sum = series.sum()
                            series_max = series.max()
                            series_count = series.count()
                            if pd.notna(series_sum):
                                accumulators[res_id]["sum"] += series_sum
                            if pd.notna(series_max):
                                accumulators[res_id]["max"] = max(accumulators[res_id]["max"], series_max)
                            if pd.notna(series_count):
                                accumulators[res_id]["count"] += series_count

                        # Per-year accumulation
                        if year_idx_values is None:
                            local_year_indices = np.zeros(len(series), dtype=int)
                            year_indices_seen.add(0)
                        else:
                            local_year_indices = year_idx_values
                        values = series.to_numpy(dtype=float)
                        finite_mask = np.isfinite(values) & (local_year_indices >= 0)
                        if not np.any(finite_mask):
                            continue
                        valid_years = local_year_indices[finite_mask]
                        valid_values = values[finite_mask]
                        for year_idx in np.unique(valid_years):
                            year_mask = valid_years == year_idx
                            year_values = valid_values[year_mask]
                            year_acc = yearly_accumulators[res_id].setdefault(
                                int(year_idx),
                                {
                                    "sum": 0.0,
                                    "max": float("-inf"),
                                    "count": 0.0,
                                    "last": 0.0,
                                    "is_cumulative": True if is_cumulative else False,
                                },
                            )
                            if is_cumulative:
                                year_acc["last"] = float(year_values[-1])
                                year_max = float(np.max(year_values))
                                if np.isfinite(year_max):
                                    year_acc["max"] = max(year_acc["max"], year_max)
                            else:
                                year_sum = float(np.sum(year_values))
                                year_max = float(np.max(year_values))
                                year_count = float(year_values.size)
                                if np.isfinite(year_sum):
                                    year_acc["sum"] += year_sum
                                if np.isfinite(year_max):
                                    year_acc["max"] = max(year_acc["max"], year_max)
                                if np.isfinite(year_count):
                                    year_acc["count"] += year_count

                for pair_key, (quantity_signal, price_signal) in dynamic_pairs.items():
                    quantity_col = col_map.get(quantity_signal)
                    price_col = col_map.get(price_signal)
                    if not quantity_col or not price_col:
                        continue
                    if quantity_col not in chunk.columns or price_col not in chunk.columns:
                        continue
                    qty_series = pd.to_numeric(chunk[quantity_col], errors="coerce").fillna(0.0)
                    price_series = pd.to_numeric(chunk[price_col], errors="coerce").fillna(0.0)
                    step_products = (qty_series * price_series).to_numpy(dtype=float)
                    dynamic_pair_products[pair_key] += float(np.sum(step_products))

                    if year_idx_values is None:
                        dynamic_pair_products_by_year[pair_key][0] = (
                            dynamic_pair_products_by_year[pair_key].get(0, 0.0)
                            + float(np.sum(step_products))
                        )
                        year_indices_seen.add(0)
                    else:
                        valid_mask = np.isfinite(step_products) & (year_idx_values >= 0)
                        if not np.any(valid_mask):
                            continue
                        valid_years = year_idx_values[valid_mask]
                        valid_products = step_products[valid_mask]
                        for year_idx in np.unique(valid_years):
                            year_mask = valid_years == year_idx
                            dynamic_pair_products_by_year[pair_key][int(year_idx)] = (
                                dynamic_pair_products_by_year[pair_key].get(int(year_idx), 0.0)
                                + float(np.sum(valid_products[year_mask]))
                            )
                
                # Periodic cleanup
                if rows_processed % (chunk_size * 5) == 0:
                    gc.collect()
        
        except Exception as e:
            logger.error(f"Error in streaming extraction: {e}")
            return {}, None, {}, [0], [0.0]
        
        # Finalize metrics
        quantities: Dict[str, Dict[str, float]] = {}
        for res_id, acc in accumulators.items():
            if res_id == "minute":
                continue
            if bool(acc["is_cumulative"]):
                last_val = acc["last"] if acc["last"] is not None else 0.0
                quantities[res_id] = {
                    "sum": last_val,
                    "max": last_val,
                    "avg": last_val,
                }
            else:
                max_val = acc["max"] if acc["max"] != float("-inf") else 0.0
                avg_val = (acc["sum"] / acc["count"]) if acc["count"] > 0 else 0.0
                quantities[res_id] = {
                    "sum": acc["sum"],
                    "max": max_val,
                    "avg": avg_val,
                }

        for pair_key, product_sum in dynamic_pair_products.items():
            quantities[pair_key] = {"sum_product": float(product_sum)}
        
        logger.info(f"Streaming complete: {rows_processed:,} rows processed")
        inferred_hours = None
        if (
            minutes_min is not None
            and minutes_max is not None
            and minutes_max >= minutes_min
        ):
            inferred_hours = (minutes_max - minutes_min) / 60.0
            if minute_diffs:
                inferred_hours += float(np.median(minute_diffs)) / 60.0
            if inferred_hours <= 0:
                inferred_hours = None

        year_indices_axis, year_hours = self._build_year_hours_from_minutes(
            minute_origin=minute_origin,
            minute_max=minutes_max,
            minute_diffs=minute_diffs,
        )
        if minute_origin is None and year_indices_seen:
            year_indices_axis = sorted(int(v) for v in year_indices_seen)
            year_hours = [0.0] * len(year_indices_axis)

        yearly_quantities: Dict[str, Dict[int, Dict[str, float]]] = {}
        for res_id, by_year in yearly_accumulators.items():
            res_year: Dict[int, Dict[str, float]] = {}
            if not by_year:
                yearly_quantities[res_id] = res_year
                continue
            is_cumulative = bool(next(iter(by_year.values())).get("is_cumulative", False))
            if is_cumulative:
                prev_last = 0.0
                for year_idx in sorted(by_year):
                    last_val = float(by_year[year_idx].get("last", 0.0) or 0.0)
                    delta = max(0.0, last_val - prev_last)
                    prev_last = last_val
                    res_year[int(year_idx)] = {
                        "sum": delta,
                        "max": last_val,
                        "avg": delta,
                    }
            else:
                for year_idx in sorted(by_year):
                    acc = by_year[year_idx]
                    max_val = acc["max"] if acc["max"] != float("-inf") else 0.0
                    avg_val = (acc["sum"] / acc["count"]) if acc["count"] > 0 else 0.0
                    res_year[int(year_idx)] = {
                        "sum": float(acc["sum"]),
                        "max": float(max_val),
                        "avg": float(avg_val),
                    }
            yearly_quantities[res_id] = res_year

        for pair_key, by_year in dynamic_pair_products_by_year.items():
            yearly_quantities[pair_key] = {
                int(year_idx): {"sum_product": float(value)}
                for year_idx, value in by_year.items()
            }

        for res_id, by_year in yearly_quantities.items():
            is_pair = res_id.startswith("__dynamic_pair__")
            for year_idx in year_indices_axis:
                by_year.setdefault(int(year_idx), self._zero_metric_entry(is_pair))

        return quantities, inferred_hours, yearly_quantities, year_indices_axis, year_hours

    def _extract_quantities_parquet_chunks(
        self,
        chunks_dir: Path,
        required_items: List[OpexItemConfig],
    ) -> Tuple[
        Dict[str, Dict[str, float]],
        Optional[float],
        Dict[str, Dict[int, Dict[str, float]]],
        List[int],
        List[float],
    ]:
        """
        Extract quantities from Parquet history chunks with metric-aware aggregation.
        
        Args:
            chunks_dir: Path to history_chunks with chunk_*.parquet files
            required_items: OPEX items requiring simulation data
            
        Returns:
            Tuple:
              (quantities_by_resource_id,
               inferred_simulation_hours,
               quantities_by_resource_id_by_year,
               year_indices_zero_based,
               year_hours)
        """
        try:
            chunk_files = sorted(
                chunks_dir.glob("chunk_*.parquet"),
                key=lambda p: int(p.stem.split('_')[-1])
            )
        except Exception:
            chunk_files = sorted(chunks_dir.glob("chunk_*.parquet"))

        if not chunk_files:
            logger.warning(f"No chunk files found in {chunks_dir}")
            return {}, None, {}, [0], [0.0]

        # Resolve available columns
        all_cols: List[str] = []
        try:
            import pyarrow.parquet as pq
            all_cols = pq.read_schema(chunk_files[0]).names
        except Exception:
            try:
                df_preview = pd.read_parquet(chunk_files[0])
                all_cols = list(df_preview.columns)
                del df_preview
            except Exception as e:
                logger.warning(f"Failed to read parquet schema: {e}")
                return {}, None, {}, [0], [0.0]

        col_map = self._resolve_required_columns(
            all_cols=all_cols,
            required_items=required_items,
            source_label=f"Parquet history schema ({chunks_dir})",
            strict=True,
        )

        # Always include H2 production column
        h2_col = self._resolve_column_name("cumulative_h2_kg", all_cols)
        if h2_col is not None:
            col_map['cumulative_h2_kg'] = h2_col
        minute_col = self._resolve_column_name("minute", all_cols)
        if minute_col is not None:
            col_map['minute'] = minute_col

        dynamic_pairs: Dict[str, Tuple[str, str]] = {}
        for item in required_items:
            pair_key = self._dynamic_pair_key(item)
            if pair_key and item.resource_id and item.price_resource_id:
                dynamic_pairs[pair_key] = (item.resource_id, item.price_resource_id)

        needed_cols = list(set(col_map.values()))
        if not needed_cols:
            logger.warning("No matching columns found for Parquet OPEX extraction")
            return {}, None, {}, [0], [0.0]

        accumulators: Dict[str, Dict[str, float]] = {}
        for res_id, col in col_map.items():
            is_cumulative = 'cumulative' in col.lower()
            accumulators[res_id] = {
                "sum": 0.0,
                "max": float("-inf"),
                "count": 0.0,
                "last": 0.0,
                "is_cumulative": True if is_cumulative else False,
            }

        yearly_accumulators: Dict[str, Dict[int, Dict[str, float]]] = {
            res_id: {} for res_id in accumulators if res_id != "minute"
        }

        minutes_min: Optional[float] = None
        minutes_max: Optional[float] = None
        minute_diffs: List[float] = []
        last_minute: Optional[float] = None
        dynamic_pair_products = {pair_key: 0.0 for pair_key in dynamic_pairs}
        dynamic_pair_products_by_year: Dict[str, Dict[int, float]] = {
            pair_key: {} for pair_key in dynamic_pairs
        }
        minute_origin: Optional[float] = None
        year_indices_seen: set[int] = set()

        for chunk_file in chunk_files:
            try:
                df = pd.read_parquet(chunk_file, columns=needed_cols)
            except Exception:
                try:
                    import pyarrow.parquet as pq
                    schema_cols = pq.read_schema(chunk_file).names
                    valid_cols = [c for c in needed_cols if c in schema_cols]
                    if not valid_cols:
                        continue
                    df = pd.read_parquet(chunk_file, columns=valid_cols)
                except Exception as e:
                    logger.warning(f"Skipping {chunk_file.name}: {e}")
                    continue

            if df.empty:
                continue
            year_idx_values: Optional[np.ndarray] = None

            minute_signal_col = col_map.get("minute")
            if minute_signal_col and minute_signal_col in df.columns:
                minute_series = pd.to_numeric(df[minute_signal_col], errors="coerce")
                minute_values = minute_series.to_numpy(dtype=float)
                minute_valid = minute_values[np.isfinite(minute_values)]
                minute_series = pd.Series(minute_valid)
                if not minute_series.empty:
                    if minute_origin is None:
                        minute_origin = float(minute_series.iloc[0])
                    chunk_minutes = minute_series.values.astype(float)
                    chunk_min = float(np.min(chunk_minutes))
                    chunk_max = float(np.max(chunk_minutes))
                    if minutes_min is None or chunk_min < minutes_min:
                        minutes_min = chunk_min
                    if minutes_max is None or chunk_max > minutes_max:
                        minutes_max = chunk_max
                    if last_minute is not None:
                        cross_diff = chunk_minutes[0] - last_minute
                        if cross_diff > 0:
                            minute_diffs.append(float(cross_diff))
                    if chunk_minutes.size > 1:
                        diffs = np.diff(chunk_minutes)
                        positive = diffs[diffs > 0]
                        if positive.size > 0:
                            minute_diffs.append(float(np.median(positive)))
                    last_minute = float(chunk_minutes[-1])
                if minute_origin is not None:
                    year_idx_values = np.full(len(df), -1, dtype=int)
                    valid_mask = np.isfinite(minute_values)
                    year_idx_values[valid_mask] = np.floor(
                        (minute_values[valid_mask] - minute_origin) / MINUTES_PER_YEAR
                    ).astype(int)
                    year_idx_values[year_idx_values < 0] = 0
                    if valid_mask.any():
                        year_indices_seen.update(int(v) for v in np.unique(year_idx_values[valid_mask]))

            for res_id, col in col_map.items():
                if res_id == "minute":
                    continue
                if col not in df.columns:
                    continue
                series = pd.to_numeric(df[col], errors="coerce")
                is_cumulative = bool(accumulators[res_id]["is_cumulative"])
                if is_cumulative:
                    if len(series) > 0:
                        accumulators[res_id]["last"] = series.iloc[-1]
                    series_max = series.max()
                    if pd.notna(series_max):
                        accumulators[res_id]["max"] = max(accumulators[res_id]["max"], series_max)
                else:
                    series_sum = series.sum()
                    series_max = series.max()
                    series_count = series.count()
                    if pd.notna(series_sum):
                        accumulators[res_id]["sum"] += series_sum
                    if pd.notna(series_max):
                        accumulators[res_id]["max"] = max(accumulators[res_id]["max"], series_max)
                    if pd.notna(series_count):
                        accumulators[res_id]["count"] += series_count

                # Per-year accumulation
                if year_idx_values is None:
                    local_year_indices = np.zeros(len(series), dtype=int)
                    year_indices_seen.add(0)
                else:
                    local_year_indices = year_idx_values
                values = series.to_numpy(dtype=float)
                finite_mask = np.isfinite(values) & (local_year_indices >= 0)
                if not np.any(finite_mask):
                    continue
                valid_years = local_year_indices[finite_mask]
                valid_values = values[finite_mask]
                for year_idx in np.unique(valid_years):
                    year_mask = valid_years == year_idx
                    year_values = valid_values[year_mask]
                    year_acc = yearly_accumulators[res_id].setdefault(
                        int(year_idx),
                        {
                            "sum": 0.0,
                            "max": float("-inf"),
                            "count": 0.0,
                            "last": 0.0,
                            "is_cumulative": True if is_cumulative else False,
                        },
                    )
                    if is_cumulative:
                        year_acc["last"] = float(year_values[-1])
                        year_max = float(np.max(year_values))
                        if np.isfinite(year_max):
                            year_acc["max"] = max(year_acc["max"], year_max)
                    else:
                        year_sum = float(np.sum(year_values))
                        year_max = float(np.max(year_values))
                        year_count = float(year_values.size)
                        if np.isfinite(year_sum):
                            year_acc["sum"] += year_sum
                        if np.isfinite(year_max):
                            year_acc["max"] = max(year_acc["max"], year_max)
                        if np.isfinite(year_count):
                            year_acc["count"] += year_count

            for pair_key, (quantity_signal, price_signal) in dynamic_pairs.items():
                quantity_col = col_map.get(quantity_signal)
                price_col = col_map.get(price_signal)
                if not quantity_col or not price_col:
                    continue
                if quantity_col not in df.columns or price_col not in df.columns:
                    continue
                qty_series = pd.to_numeric(df[quantity_col], errors="coerce").fillna(0.0)
                price_series = pd.to_numeric(df[price_col], errors="coerce").fillna(0.0)
                step_products = (qty_series * price_series).to_numpy(dtype=float)
                dynamic_pair_products[pair_key] += float(np.sum(step_products))

                if year_idx_values is None:
                    dynamic_pair_products_by_year[pair_key][0] = (
                        dynamic_pair_products_by_year[pair_key].get(0, 0.0)
                        + float(np.sum(step_products))
                    )
                    year_indices_seen.add(0)
                else:
                    valid_mask = np.isfinite(step_products) & (year_idx_values >= 0)
                    if not np.any(valid_mask):
                        continue
                    valid_years = year_idx_values[valid_mask]
                    valid_products = step_products[valid_mask]
                    for year_idx in np.unique(valid_years):
                        year_mask = valid_years == year_idx
                        dynamic_pair_products_by_year[pair_key][int(year_idx)] = (
                            dynamic_pair_products_by_year[pair_key].get(int(year_idx), 0.0)
                            + float(np.sum(valid_products[year_mask]))
                        )

        quantities: Dict[str, Dict[str, float]] = {}
        for res_id, acc in accumulators.items():
            if res_id == "minute":
                continue
            if bool(acc["is_cumulative"]):
                last_val = acc["last"] if acc["last"] is not None else 0.0
                quantities[res_id] = {
                    "sum": last_val,
                    "max": last_val,
                    "avg": last_val,
                }
            else:
                max_val = acc["max"] if acc["max"] != float("-inf") else 0.0
                avg_val = (acc["sum"] / acc["count"]) if acc["count"] > 0 else 0.0
                quantities[res_id] = {
                    "sum": acc["sum"],
                    "max": max_val,
                    "avg": avg_val,
                }

        for pair_key, product_sum in dynamic_pair_products.items():
            quantities[pair_key] = {"sum_product": float(product_sum)}

        inferred_hours = None
        if (
            minutes_min is not None
            and minutes_max is not None
            and minutes_max >= minutes_min
        ):
            inferred_hours = (minutes_max - minutes_min) / 60.0
            if minute_diffs:
                inferred_hours += float(np.median(minute_diffs)) / 60.0
            if inferred_hours <= 0:
                inferred_hours = None

        year_indices_axis, year_hours = self._build_year_hours_from_minutes(
            minute_origin=minute_origin,
            minute_max=minutes_max,
            minute_diffs=minute_diffs,
        )
        if minute_origin is None and year_indices_seen:
            year_indices_axis = sorted(int(v) for v in year_indices_seen)
            year_hours = [0.0] * len(year_indices_axis)

        yearly_quantities: Dict[str, Dict[int, Dict[str, float]]] = {}
        for res_id, by_year in yearly_accumulators.items():
            res_year: Dict[int, Dict[str, float]] = {}
            if not by_year:
                yearly_quantities[res_id] = res_year
                continue
            is_cumulative = bool(next(iter(by_year.values())).get("is_cumulative", False))
            if is_cumulative:
                prev_last = 0.0
                for year_idx in sorted(by_year):
                    last_val = float(by_year[year_idx].get("last", 0.0) or 0.0)
                    delta = max(0.0, last_val - prev_last)
                    prev_last = last_val
                    res_year[int(year_idx)] = {
                        "sum": delta,
                        "max": last_val,
                        "avg": delta,
                    }
            else:
                for year_idx in sorted(by_year):
                    acc = by_year[year_idx]
                    max_val = acc["max"] if acc["max"] != float("-inf") else 0.0
                    avg_val = (acc["sum"] / acc["count"]) if acc["count"] > 0 else 0.0
                    res_year[int(year_idx)] = {
                        "sum": float(acc["sum"]),
                        "max": float(max_val),
                        "avg": float(avg_val),
                    }
            yearly_quantities[res_id] = res_year

        for pair_key, by_year in dynamic_pair_products_by_year.items():
            yearly_quantities[pair_key] = {
                int(year_idx): {"sum_product": float(value)}
                for year_idx, value in by_year.items()
            }

        for res_id, by_year in yearly_quantities.items():
            is_pair = res_id.startswith("__dynamic_pair__")
            for year_idx in year_indices_axis:
                by_year.setdefault(int(year_idx), self._zero_metric_entry(is_pair))

        return quantities, inferred_hours, yearly_quantities, year_indices_axis, year_hours

    @staticmethod
    def _normalize_pathway_shares(pathway_totals: Dict[str, float]) -> Dict[str, float]:
        cleaned = {k: max(0.0, float(v or 0.0)) for k, v in pathway_totals.items()}
        total = sum(cleaned.values())
        if total <= 0:
            return {k: 0.0 for k in cleaned}
        return {k: v / total for k, v in cleaned.items()}

    def _build_pathway_shares_from_quantities(
        self,
        *,
        item: OpexItemConfig,
        quantities: Dict[str, Dict[str, float]],
        annualization_factor: float,
        history_available: bool,
    ) -> Optional[Dict[str, float]]:
        if not item.pathway_driver_resource_ids:
            return None
        if not history_available:
            return {key: 0.0 for key in item.pathway_driver_resource_ids}

        pathway_totals: Dict[str, float] = {}
        for pathway, signal_id in item.pathway_driver_resource_ids.items():
            raw = self._extract_metric_from_quantities(quantities, signal_id, item.metric)
            pathway_totals[pathway] = self._annualize_metric(raw, item.metric, annualization_factor)
        return self._normalize_pathway_shares(pathway_totals)

    def _build_pathway_shares_from_dataframe(
        self,
        *,
        item: OpexItemConfig,
        history_df: Optional[pd.DataFrame],
        resolved_columns: Dict[str, str],
        annualization_factor: float,
    ) -> Optional[Dict[str, float]]:
        if not item.pathway_driver_resource_ids:
            return None
        if history_df is None:
            return {key: 0.0 for key in item.pathway_driver_resource_ids}

        pathway_totals: Dict[str, float] = {}
        for pathway, signal_id in item.pathway_driver_resource_ids.items():
            qty, _ = self._extract_quantity(
                history_df,
                signal_id,
                item.metric,
                annualization_factor,
                resolved_column=resolved_columns.get(signal_id),
            )
            pathway_totals[pathway] = float(qty)
        return self._normalize_pathway_shares(pathway_totals)

    def _calculate_dynamic_annual_cost_from_quantities(
        self,
        *,
        item: OpexItemConfig,
        quantities: Dict[str, Dict[str, float]],
        annualization_factor: float,
    ) -> Tuple[float, str]:
        pair_key = self._dynamic_pair_key(item)
        if pair_key is None:
            return 0.0, "Dynamic pricing unavailable"

        pair_entry = quantities.get(pair_key, {})
        if isinstance(pair_entry, dict):
            period_sum = float(pair_entry.get("sum_product", 0.0) or 0.0)
        else:
            period_sum = float(pair_entry or 0.0)

        annual_cost = (
            period_sum
            * annualization_factor
            * float(item.cost_multiplier)
            * float(item.price)
        )
        formula = (
            "annualized(sum(quantity_step × price_step)) × "
            f"{item.cost_multiplier:.6g} × {item.price:.6g}"
        )
        return annual_cost, formula

    def _calculate_dynamic_yearly_cost_from_quantities(
        self,
        *,
        item: OpexItemConfig,
        quantities_by_year: Dict[str, Dict[int, Dict[str, float]]],
        year_indices: List[int],
    ) -> np.ndarray:
        pair_key = self._dynamic_pair_key(item)
        if pair_key is None:
            return np.zeros(len(year_indices), dtype=float)

        by_year = quantities_by_year.get(pair_key, {})
        out = np.zeros(len(year_indices), dtype=float)
        for idx, year_idx in enumerate(year_indices):
            entry = by_year.get(int(year_idx), {})
            period_sum = float(entry.get("sum_product", 0.0) or 0.0)
            out[idx] = period_sum * float(item.cost_multiplier) * float(item.price)
        return out

    @staticmethod
    def _metric_to_yearly_quantity(raw_qty: float, metric: str, period_hours: float) -> float:
        return OpexGenerator._period_quantity_from_metric(raw_qty, metric, period_hours)

    def _calculate_item_streaming_yearly(
        self,
        *,
        item: OpexItemConfig,
        quantities: Dict[str, Dict[str, float]],
        quantities_by_year: Dict[str, Dict[int, Dict[str, float]]],
        history_available: bool,
        base_costs: Dict[str, float],
        annualization_factor: float,
        year_indices: List[int],
        year_hours: np.ndarray,
    ) -> Tuple[OpexResult, np.ndarray]:
        scalar_result = self._calculate_item_streaming(
            item=item,
            quantities=quantities,
            history_available=history_available,
            base_costs=base_costs,
            annualization_factor=annualization_factor,
        )

        year_count = len(year_indices)
        if year_count == 0:
            return scalar_result, np.zeros(0, dtype=float)

        year_scale = np.clip(year_hours, a_min=0.0, a_max=None) / HOURS_PER_YEAR

        if item.strategy == "variable" and item.resource_id:
            if not history_available:
                return scalar_result, np.zeros(year_count, dtype=float)

            if item.price_resource_id:
                yearly_costs = self._calculate_dynamic_yearly_cost_from_quantities(
                    item=item,
                    quantities_by_year=quantities_by_year,
                    year_indices=year_indices,
                )
                return scalar_result, yearly_costs

            signal_by_year = quantities_by_year.get(item.resource_id, {})
            yearly_costs = np.zeros(year_count, dtype=float)
            for idx, year_idx in enumerate(year_indices):
                raw_entry = signal_by_year.get(int(year_idx), {})
                raw_qty = float(raw_entry.get(item.metric, 0.0) or 0.0)
                period_qty = self._metric_to_yearly_quantity(
                    raw_qty=raw_qty,
                    metric=item.metric,
                    period_hours=float(year_hours[idx]),
                )
                yearly_costs[idx] = (
                    period_qty
                    * float(item.price)
                    * float(item.cost_multiplier)
                )
            return scalar_result, yearly_costs

        yearly_costs = np.full(year_count, float(scalar_result.annual_cost), dtype=float) * year_scale
        return scalar_result, yearly_costs

    def _evaluate_streaming_yearly_for_fci(
        self,
        *,
        sorted_items: List[OpexItemConfig],
        fci: float,
        quantities: Dict[str, Dict[str, float]],
        quantities_by_year: Dict[str, Dict[int, Dict[str, float]]],
        history_available: bool,
        annualization_factor: float,
        year_indices: List[int],
        year_hours: np.ndarray,
        keep_scalar_results: bool = False,
    ) -> Dict[str, Any]:
        base_costs = {
            "FCI": float(fci),
            "Labor": 0.0,
            "C_OL": 0.0,
        }
        year_count = len(year_indices)
        variable_by_year = np.zeros(year_count, dtype=float)
        fixed_by_year = np.zeros(year_count, dtype=float)
        maintenance_by_year = np.zeros(year_count, dtype=float)
        credit_by_year = np.zeros(year_count, dtype=float)
        scalar_results: List[OpexResult] = []
        labor_cost = 0.0

        item_costs_by_year: Dict[str, List[float]] = {}
        name_count: Dict[str, int] = {}

        for item in sorted_items:
            scalar_result, yearly_costs = self._calculate_item_streaming_yearly(
                item=item,
                quantities=quantities,
                quantities_by_year=quantities_by_year,
                history_available=history_available,
                base_costs=base_costs,
                annualization_factor=annualization_factor,
                year_indices=year_indices,
                year_hours=year_hours,
            )

            if self._is_primary_labor_item(item):
                labor_cost = float(scalar_result.annual_cost)
                base_costs["Labor"] = labor_cost
                base_costs["C_OL"] = labor_cost

            if item.category == OpexCategory.VARIABLE:
                variable_by_year += yearly_costs
            elif item.category == OpexCategory.FIXED:
                fixed_by_year += yearly_costs
            elif item.category == OpexCategory.MAINTENANCE:
                maintenance_by_year += yearly_costs

            if bool(getattr(item, "is_credit", False)):
                credit_by_year += yearly_costs

            base_name = str(item.name)
            n = name_count.get(base_name, 0) + 1
            name_count[base_name] = n
            key_name = base_name if n == 1 else f"{base_name} [{n}]"
            item_costs_by_year[key_name] = [float(v) for v in yearly_costs]

            if keep_scalar_results:
                scalar_results.append(scalar_result)

        total_by_year = variable_by_year + fixed_by_year + maintenance_by_year
        cashflow_by_year = total_by_year - credit_by_year

        return {
            "scalar_results": scalar_results,
            "labor_cost": float(labor_cost),
            "item_costs_by_year": item_costs_by_year,
            "variable_by_year": variable_by_year,
            "fixed_by_year": fixed_by_year,
            "maintenance_by_year": maintenance_by_year,
            "credit_by_year": credit_by_year,
            "total_by_year": total_by_year,
            "cashflow_by_year": cashflow_by_year,
        }

    def _calculate_dynamic_annual_cost_from_dataframe(
        self,
        *,
        item: OpexItemConfig,
        history_df: Optional[pd.DataFrame],
        resolved_columns: Dict[str, str],
        annualization_factor: float,
    ) -> Tuple[float, str]:
        if history_df is None or not item.resource_id or not item.price_resource_id:
            return 0.0, "Dynamic pricing unavailable"

        quantity_col = resolved_columns.get(item.resource_id)
        if quantity_col is None or quantity_col not in history_df.columns:
            quantity_col = self._resolve_column_name(item.resource_id, list(history_df.columns))
        price_col = resolved_columns.get(item.price_resource_id)
        if price_col is None or price_col not in history_df.columns:
            price_col = self._resolve_column_name(item.price_resource_id, list(history_df.columns))
        if quantity_col is None or price_col is None:
            return 0.0, "Dynamic pricing signals not found"

        quantity_series = pd.to_numeric(history_df[quantity_col], errors="coerce").fillna(0.0)
        price_series = pd.to_numeric(history_df[price_col], errors="coerce").fillna(0.0)
        period_sum = float((quantity_series * price_series).sum())
        annual_cost = (
            period_sum
            * annualization_factor
            * float(item.cost_multiplier)
            * float(item.price)
        )
        formula = (
            "annualized(sum(quantity_step × price_step)) × "
            f"{item.cost_multiplier:.6g} × {item.price:.6g}"
        )
        return annual_cost, formula
    
    def _calculate_item_streaming(
        self,
        item: OpexItemConfig,
        quantities: Dict[str, Dict[str, float]],
        history_available: bool,
        base_costs: Dict[str, float],
        annualization_factor: float,
    ) -> OpexResult:
        """Calculate cost for a single OPEX item using pre-extracted quantities."""
        
        result = OpexResult(
            name=item.name,
            category=item.category,
            unit_price=item.price,
            source="config",
            is_credit=item.is_credit,
            lcoh_component=item.lcoh_component,
        )
        
        # Extract quantity from pre-computed values
        quantity = 1.0
        if item.strategy == "variable" and item.resource_id:
            if not history_available:
                raw_qty = 0.0
                result.source = "no_history"
            else:
                raw_entry = quantities.get(item.resource_id, {})
                if isinstance(raw_entry, dict):
                    raw_qty = raw_entry.get(item.metric, 0.0)
                else:
                    raw_qty = raw_entry
                result.source = f"simulation:streaming:{item.resource_id}"

            quantity = self._annualize_metric(raw_qty, item.metric, annualization_factor)
        elif item.strategy == "turton_labor":
            quantity = item.hours_per_year
        
        result.annual_quantity = quantity
        result.pathway_shares = self._build_pathway_shares_from_quantities(
            item=item,
            quantities=quantities,
            annualization_factor=annualization_factor,
            history_available=history_available,
        )

        if item.strategy == "variable" and item.resource_id and item.price_resource_id:
            annual_cost, formula = self._calculate_dynamic_annual_cost_from_quantities(
                item=item,
                quantities=quantities,
                annualization_factor=annualization_factor,
            )
            if history_available:
                result.source = f"simulation:streaming:{item.resource_id}*{item.price_resource_id}"
            result.annual_cost = round(annual_cost, 2)
            result.formula = formula
            return result
        
        # Get strategy and calculate
        try:
            strategy = get_opex_strategy(item.strategy)
            
            kwargs = {
                "unit": item.unit,
                "base_reference": item.base_reference,
                "P": item.turton_P,
                "Nnp": item.turton_Nnp,
                "shifts": item.shifts,
                "hours_per_year": item.hours_per_year,
                "ref_production": item.ref_production,
                "scaling_exponent": item.scaling_exponent,
            }
            
            base_val = 0.0
            if item.strategy == "factor" and item.base_reference:
                base_val = base_costs.get(item.base_reference, 0.0)
                if base_val == 0.0:
                    result.warnings.append(f"Base reference '{item.base_reference}' not found or zero")
            
            cost, formula = strategy.calculate(
                quantity=quantity,
                price=item.price,
                base_cost=base_val,
                **kwargs
            )
            if item.strategy == "variable":
                cost *= float(item.cost_multiplier)
                if float(item.cost_multiplier) != 1.0:
                    formula = f"{formula} × {item.cost_multiplier:.6g}"
            
            result.annual_cost = cost
            result.formula = formula
            
        except Exception as e:
            logger.warning(f"Error calculating {item.name}: {e}")
            result.warnings.append(str(e))
            result.formula = f"Error: {e}"
        
        return result
    
    def _calculate_item(
        self,
        item: OpexItemConfig,
        history_df: Optional[pd.DataFrame],
        resolved_columns: Dict[str, str],
        history_available: bool,
        base_costs: Dict[str, float],
        annualization_factor: float,
    ) -> OpexResult:
        """Calculate cost for a single OPEX item."""
        
        result = OpexResult(
            name=item.name,
            category=item.category,
            unit_price=item.price,
            source="config",
            is_credit=item.is_credit,
            lcoh_component=item.lcoh_component,
        )
        
        # Extract quantity from simulation if applicable
        quantity = 1.0
        if item.strategy == "variable" and item.resource_id:
            if history_df is not None:
                quantity, source = self._extract_quantity(
                    history_df,
                    item.resource_id,
                    item.metric,
                    annualization_factor,
                    resolved_column=resolved_columns.get(item.resource_id),
                )
                result.source = source
            elif history_available:
                # Defensive guard for malformed callers that pass an unavailable frame.
                raise ValueError(
                    "History source was declared available but no dataframe was provided "
                    f"for variable OPEX item '{item.name}' ({item.resource_id})."
                )
            else:
                quantity = 0.0
                result.source = "no_history"
        elif item.strategy == "turton_labor":
            quantity = item.hours_per_year
        
        result.annual_quantity = quantity
        result.pathway_shares = self._build_pathway_shares_from_dataframe(
            item=item,
            history_df=history_df,
            resolved_columns=resolved_columns,
            annualization_factor=annualization_factor,
        )

        if item.strategy == "variable" and item.resource_id and item.price_resource_id:
            annual_cost, formula = self._calculate_dynamic_annual_cost_from_dataframe(
                item=item,
                history_df=history_df,
                resolved_columns=resolved_columns,
                annualization_factor=annualization_factor,
            )
            if history_df is not None:
                result.source = f"simulation:{item.resource_id}*{item.price_resource_id}"
            result.annual_cost = round(annual_cost, 2)
            result.formula = formula
            return result
        
        # Get strategy and calculate
        try:
            strategy = get_opex_strategy(item.strategy)
            
            # Prepare kwargs
            kwargs = {
                "unit": item.unit,
                "base_reference": item.base_reference,
                "P": item.turton_P,
                "Nnp": item.turton_Nnp,
                "shifts": item.shifts,
                "hours_per_year": item.hours_per_year,
                "ref_production": item.ref_production,
                "scaling_exponent": item.scaling_exponent,
            }
            
            # Get base cost for factor strategies
            base_val = 0.0
            if item.strategy == "factor" and item.base_reference:
                base_val = base_costs.get(item.base_reference, 0.0)
                if base_val == 0.0:
                    result.warnings.append(f"Base reference '{item.base_reference}' not found or zero")
            
            cost, formula = strategy.calculate(
                quantity=quantity,
                price=item.price,
                base_cost=base_val,
                **kwargs
            )
            if item.strategy == "variable":
                cost *= float(item.cost_multiplier)
                if float(item.cost_multiplier) != 1.0:
                    formula = f"{formula} × {item.cost_multiplier:.6g}"
            
            result.annual_cost = cost
            result.formula = formula
            
        except Exception as e:
            logger.warning(f"Error calculating {item.name}: {e}")
            result.warnings.append(str(e))
            result.formula = f"Error: {e}"
        
        return result
    
    def _extract_quantity(
        self,
        df: pd.DataFrame,
        resource_id: str,
        metric: str,
        annualization_factor: float,
        resolved_column: Optional[str] = None,
    ) -> tuple[float, str]:
        """
        Extract quantity from simulation history.
        
        Returns:
            Tuple of (quantity, source_description)
        """
        if resolved_column is not None and resolved_column in df.columns:
            col = resolved_column
        else:
            col = self._resolve_column_name(resource_id, list(df.columns))

        if col is None:
            logger.warning("No columns matching '%s' in history", resource_id)
            return 0.0, "not_found"
        
        # Aggregate based on metric
        if metric == "sum":
            # Sum represents total over simulation period, annualize
            raw_sum = df[col].sum()
            quantity = raw_sum * annualization_factor
        elif metric == "max":
            quantity = df[col].max()
        elif metric == "avg":
            quantity = df[col].mean() * 8760  # Avg rate × hours/year
        else:
            quantity = df[col].sum() * annualization_factor
        
        return quantity, f"simulation:{col}"
    
    def _extract_h2_production(
        self,
        df: pd.DataFrame,
        annualization_factor: float,
    ) -> float:
        """Extract total H2 production from simulation history."""
        
        # Look for cumulative H2 columns
        h2_cols = [c for c in df.columns if 'cumulative_h2_kg' in c.lower() or 'h2_total_kg' in c.lower()]
        
        if h2_cols:
            # Use final value of cumulative column
            final_production = df[h2_cols[0]].iloc[-1] if len(df) > 0 else 0.0
            return final_production * annualization_factor
        
        # Alternative: sum mass flows from electrolyzers
        pem_cols = [c for c in df.columns if 'pem' in c.lower() and 'h2' in c.lower() and 'kg' in c.lower()]
        soec_cols = [c for c in df.columns if 'soec' in c.lower() and 'h2' in c.lower() and 'kg' in c.lower()]
        
        total = 0.0
        for col in pem_cols + soec_cols:
            if 'flow' in col.lower() or 'rate' in col.lower():
                total += df[col].sum()
        
        return total * annualization_factor
    
    def _export_json(self, report: OpexReport, path: Path) -> None:
        """Export report to JSON."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(report.model_dump(), f, indent=2, default=str)
        logger.info(f"✓ JSON export: {path}")
    
    def _export_csv(self, report: OpexReport, path: Path) -> None:
        """Export report to CSV."""
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                "Name", "Category", "Annual Quantity", "Unit Price",
                "Annual Cost (USD)", "Formula", "Source", "Warnings"
            ])
            
            # Data rows
            for item in report.items:
                writer.writerow([
                    item.name,
                    item.category.value,
                    round(item.annual_quantity, 2),
                    item.unit_price,
                    round(item.annual_cost, 2),
                    item.formula,
                    item.source,
                    "; ".join(item.warnings),
                ])
            
            # Summary section
            writer.writerow([])
            writer.writerow(["SUMMARY", "", "", "", "", "", "", ""])
            writer.writerow(["Variable Costs", "", "", "", round(report.total_variable_cost, 2), "", "", ""])
            writer.writerow(["Fixed Costs", "", "", "", round(report.total_fixed_cost, 2), "", "", ""])
            writer.writerow(["Maintenance Costs", "", "", "", round(report.total_maintenance_cost, 2), "", "", ""])
            writer.writerow(["TOTAL OPEX", "", "", "", round(report.total_opex, 2), "", "", ""])
            writer.writerow(["TOTAL CREDIT COST", "", "", "", round(report.total_credit_cost, 2), "", "", ""])
            writer.writerow(["TOTAL OPEX CASHFLOW", "", "", "", round(report.total_opex_cashflow, 2), "", "", ""])
            if report.total_opex_low is not None:
                writer.writerow(["TOTAL OPEX LOW", "", "", "", round(report.total_opex_low, 2), "", "", ""])
            if report.total_opex_cashflow_low is not None:
                writer.writerow(["TOTAL OPEX CASHFLOW LOW", "", "", "", round(report.total_opex_cashflow_low, 2), "", "", ""])
            if report.total_opex_high is not None:
                writer.writerow(["TOTAL OPEX HIGH", "", "", "", round(report.total_opex_high, 2), "", "", ""])
            if report.total_opex_cashflow_high is not None:
                writer.writerow(["TOTAL OPEX CASHFLOW HIGH", "", "", "", round(report.total_opex_cashflow_high, 2), "", "", ""])
            
            # Production metrics
            if report.annual_h2_production_kg > 0:
                writer.writerow([])
                writer.writerow(["METRICS", "", "", "", "", "", "", ""])
                writer.writerow(["Annual H2 Production (kg)", "", "", "", round(report.annual_h2_production_kg, 0), "", "", ""])
                writer.writerow(["OPEX per kg H2", "", "", "", round(report.opex_per_kg_h2, 4), "", "", ""])
        
        logger.info(f"✓ CSV export: {path}")
