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
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Any, List

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

    def _apply_uncertainty_bands(
        self,
        report: OpexReport,
        capex_report: Optional[CapexReport],
    ) -> None:
        """
        Apply low/high OPEX uncertainty bands using CAPEX AACE class factors.

        If CAPEX context is missing, low/high are intentionally omitted.
        """
        if capex_report is None:
            logger.warning(
                "CAPEX report not provided; OPEX uncertainty bands (low/high) will be omitted."
            )
            return

        cost_class = getattr(capex_report, "overall_cost_class", None)
        if cost_class is None:
            logger.warning(
                "CAPEX cost class not available; OPEX uncertainty bands (low/high) will be omitted."
            )
            return

        try:
            low_factor, high_factor = cost_class.accuracy_range
        except Exception as e:
            logger.warning(f"Could not derive AACE accuracy range for OPEX uncertainty: {e}")
            return

        try:
            report.total_opex_low = float(report.total_opex) * float(low_factor)
            report.total_opex_high = float(report.total_opex) * float(high_factor)
        except (TypeError, ValueError) as e:
            logger.warning(f"Could not compute OPEX uncertainty bands: {e}")
            return
    
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
        
        # Base costs for factor calculations
        base_costs = {
            "FCI": report.fci,
            "Labor": 0.0,
            "C_OL": 0.0,
        }
        
        # Sort items to ensure Labor is calculated before items dependent on it
        sorted_items = sorted(
            self.items,
            key=lambda x: 0 if x.strategy == "turton_labor" else (1 if x.strategy == "fixed" and "Labor" in x.name else 2)
        )
        
        # Calculate each item
        for item in sorted_items:
            result = self._calculate_item(
                item=item,
                history_df=history_df,
                base_costs=base_costs,
                annualization_factor=report.annualization_factor,
            )
            
            # Update base costs if this is a labor item
            if "Labor" in item.name or item.strategy == "turton_labor":
                base_costs["Labor"] = result.annual_cost
                base_costs["C_OL"] = result.annual_cost
                report.labor_cost = result.annual_cost
            
            report.items.append(result)
        
        # Calculate H2 production from history
        if history_df is not None:
            report.annual_h2_production_kg = self._extract_h2_production(
                history_df, report.annualization_factor
            )
        
        # Calculate totals
        report.calculate_totals()
        self._apply_uncertainty_bands(report, capex_report)
        
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
        
        # Initialize report
        annualization_factor = 8760.0 / simulation_hours if simulation_hours > 0 else 1.0
        report = OpexReport(
            scenario_name=self.config.get('scenario_name', 'default'),
            simulation_hours=simulation_hours,
            annualization_factor=annualization_factor,
        )
        
        # Get FCI from CAPEX report
        if capex_report:
            report.fci = capex_report.total_installed_cost or capex_report.total_C_BM or 0.0
            logger.info(f"Using FCI from CAPEX: ${report.fci:,.0f}")
        
        # Base costs for factor calculations
        base_costs = {
            "FCI": report.fci,
            "Labor": 0.0,
            "C_OL": 0.0,
        }
        
        # Extract quantities from CSV using streaming
        csv_path = Path(csv_path)
        if csv_path.exists():
            # Identify required columns from config
            required_items = [
                item for item in self.items
                if item.strategy == "variable" and item.resource_id
            ]
            
            quantities = self._extract_quantities_streaming(
                csv_path, required_items, chunk_size
            )
            logger.info(f"Extracted {len(quantities)} quantities via streaming")
        else:
            quantities = {}
            logger.warning(f"CSV not found: {csv_path}")
        
        # Sort items to ensure Labor is calculated before dependent items
        sorted_items = sorted(
            self.items,
            key=lambda x: 0 if x.strategy == "turton_labor" else (1 if x.strategy == "fixed" and "Labor" in x.name else 2)
        )
        
        # Calculate each item
        for item in sorted_items:
            result = self._calculate_item_streaming(
                item=item,
                quantities=quantities,
                base_costs=base_costs,
                annualization_factor=annualization_factor,
            )
            
            # Update base costs if this is a labor item
            if "Labor" in item.name or item.strategy == "turton_labor":
                base_costs["Labor"] = result.annual_cost
                base_costs["C_OL"] = result.annual_cost
                report.labor_cost = result.annual_cost
            
            report.items.append(result)
        
        # Set H2 production from streaming extraction
        h2_entry = quantities.get('cumulative_h2_kg', 0.0)
        if isinstance(h2_entry, dict):
            report.annual_h2_production_kg = h2_entry.get('sum', 0.0) * annualization_factor
        else:
            report.annual_h2_production_kg = h2_entry * annualization_factor
        
        # Calculate totals
        report.calculate_totals()
        self._apply_uncertainty_bands(report, capex_report)
        
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

        # Initialize report
        annualization_factor = 8760.0 / simulation_hours if simulation_hours > 0 else 1.0
        report = OpexReport(
            scenario_name=self.config.get('scenario_name', 'default'),
            simulation_hours=simulation_hours,
            annualization_factor=annualization_factor,
        )

        # Get FCI from CAPEX report
        if capex_report:
            report.fci = capex_report.total_installed_cost or capex_report.total_C_BM or 0.0
            logger.info(f"Using FCI from CAPEX: ${report.fci:,.0f}")

        base_costs = {
            "FCI": report.fci,
            "Labor": 0.0,
            "C_OL": 0.0,
        }

        required_items = [
            item for item in self.items
            if item.strategy == "variable" and item.resource_id
        ]

        quantities = self._extract_quantities_parquet_chunks(chunks_dir, required_items)
        logger.info(f"Extracted {len(quantities)} quantities from Parquet chunks")

        sorted_items = sorted(
            self.items,
            key=lambda x: 0 if x.strategy == "turton_labor" else (1 if x.strategy == "fixed" and "Labor" in x.name else 2)
        )

        for item in sorted_items:
            result = self._calculate_item_streaming(
                item=item,
                quantities=quantities,
                base_costs=base_costs,
                annualization_factor=annualization_factor,
            )

            if "Labor" in item.name or item.strategy == "turton_labor":
                base_costs["Labor"] = result.annual_cost
                base_costs["C_OL"] = result.annual_cost
                report.labor_cost = result.annual_cost

            report.items.append(result)

        h2_entry = quantities.get('cumulative_h2_kg', 0.0)
        if isinstance(h2_entry, dict):
            report.annual_h2_production_kg = h2_entry.get('sum', 0.0) * annualization_factor
        else:
            report.annual_h2_production_kg = h2_entry * annualization_factor

        report.calculate_totals()
        self._apply_uncertainty_bands(report, capex_report)

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
    ) -> Dict[str, Dict[str, float]]:
        """
        Extract quantities using chunked streaming with column filtering.
        
        Memory-efficient: reads only needed columns, processes in chunks.
        
        Args:
            csv_path: Path to simulation_history.csv
            required_items: List of OPEX items requiring simulation data
            chunk_size: Rows per chunk
            
        Returns:
            Dict of resource_id -> metric quantities (sum/max/avg)
        """
        import gc
        
        # 1. Scan header to find matching columns
        header_df = pd.read_csv(csv_path, nrows=0)
        all_cols = list(header_df.columns)
        
        # Map resource_id -> actual column name
        col_map = {}
        required_resources = []
        for item in required_items:
            if item.resource_id:
                required_resources.append(item.resource_id)

        for res_id in required_resources:
            for col in all_cols:
                if res_id.lower() in col.lower():
                    col_map[res_id] = col
                    break
        
        # Always include H2 production column
        h2_cols = [c for c in all_cols if 'cumulative_h2_kg' in c.lower()]
        if h2_cols:
            col_map['cumulative_h2_kg'] = h2_cols[0]
        
        # 2. Select only needed columns
        needed_cols = list(set(col_map.values()))
        if not needed_cols:
            logger.warning("No matching columns found for OPEX extraction")
            return {}
        
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

        rows_processed = 0
        
        # 4. Process chunks
        try:
            for chunk in pd.read_csv(csv_path, usecols=needed_cols, chunksize=chunk_size):
                rows_processed += len(chunk)
                
                for res_id, col in col_map.items():
                    if col in chunk.columns:
                        series = chunk[col]
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
                
                # Periodic cleanup
                if rows_processed % (chunk_size * 5) == 0:
                    gc.collect()
        
        except Exception as e:
            logger.error(f"Error in streaming extraction: {e}")
            return {}
        
        # Finalize metrics
        quantities: Dict[str, Dict[str, float]] = {}
        for res_id, acc in accumulators.items():
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
        
        logger.info(f"Streaming complete: {rows_processed:,} rows processed")
        return quantities

    def _extract_quantities_parquet_chunks(
        self,
        chunks_dir: Path,
        required_items: List[OpexItemConfig],
    ) -> Dict[str, Dict[str, float]]:
        """
        Extract quantities from Parquet history chunks with metric-aware aggregation.
        
        Args:
            chunks_dir: Path to history_chunks with chunk_*.parquet files
            required_items: OPEX items requiring simulation data
            
        Returns:
            Dict of resource_id -> metric quantities (sum/max/avg)
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
            return {}

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
                return {}

        # Map resource_id -> actual column name
        col_map: Dict[str, str] = {}
        for item in required_items:
            if not item.resource_id:
                continue
            res_id = item.resource_id
            for col in all_cols:
                if res_id.lower() in col.lower():
                    col_map[res_id] = col
                    break

        # Always include H2 production column
        h2_cols = [c for c in all_cols if 'cumulative_h2_kg' in c.lower()]
        if h2_cols:
            col_map['cumulative_h2_kg'] = h2_cols[0]

        needed_cols = list(set(col_map.values()))
        if not needed_cols:
            logger.warning("No matching columns found for Parquet OPEX extraction")
            return {}

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

            for res_id, col in col_map.items():
                if col not in df.columns:
                    continue
                series = df[col]
                is_cumulative = bool(accumulators[res_id]["is_cumulative"])
                if is_cumulative:
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

        quantities: Dict[str, Dict[str, float]] = {}
        for res_id, acc in accumulators.items():
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

        return quantities
    
    def _calculate_item_streaming(
        self,
        item: OpexItemConfig,
        quantities: Dict[str, Dict[str, float]],
        base_costs: Dict[str, float],
        annualization_factor: float,
    ) -> OpexResult:
        """Calculate cost for a single OPEX item using pre-extracted quantities."""
        
        result = OpexResult(
            name=item.name,
            category=item.category,
            unit_price=item.price,
            source="config",
        )
        
        # Extract quantity from pre-computed values
        quantity = 1.0
        if item.strategy == "variable" and item.resource_id:
            raw_entry = quantities.get(item.resource_id, {})
            if isinstance(raw_entry, dict):
                raw_qty = raw_entry.get(item.metric, 0.0)
            else:
                raw_qty = raw_entry

            if item.metric == "avg":
                quantity = raw_qty * 8760.0
            elif item.metric == "max":
                quantity = raw_qty
            else:
                quantity = raw_qty * annualization_factor

            result.source = f"simulation:streaming:{item.resource_id}"
        elif item.strategy == "turton_labor":
            quantity = item.hours_per_year
        
        result.annual_quantity = quantity
        
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
        base_costs: Dict[str, float],
        annualization_factor: float,
    ) -> OpexResult:
        """Calculate cost for a single OPEX item."""
        
        result = OpexResult(
            name=item.name,
            category=item.category,
            unit_price=item.price,
            source="config",
        )
        
        # Extract quantity from simulation if applicable
        quantity = 1.0
        if item.strategy == "variable" and item.resource_id:
            if history_df is not None:
                quantity, source = self._extract_quantity(
                    history_df, 
                    item.resource_id, 
                    item.metric,
                    annualization_factor
                )
                result.source = source
            else:
                quantity = 0.0
                result.source = "no_history"
        elif item.strategy == "turton_labor":
            quantity = item.hours_per_year
        
        result.annual_quantity = quantity
        
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
    ) -> tuple[float, str]:
        """
        Extract quantity from simulation history.
        
        Returns:
            Tuple of (quantity, source_description)
        """
        # Find matching columns
        matching_cols = [c for c in df.columns if resource_id.lower() in c.lower()]
        
        if not matching_cols:
            logger.warning(f"No columns matching '{resource_id}' in history")
            return 0.0, "not_found"
        
        col = matching_cols[0]
        if len(matching_cols) > 1:
            logger.debug(f"Multiple columns match '{resource_id}', using '{col}'")
        
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
            if report.total_opex_low is not None:
                writer.writerow(["TOTAL OPEX LOW", "", "", "", round(report.total_opex_low, 2), "", "", ""])
            if report.total_opex_high is not None:
                writer.writerow(["TOTAL OPEX HIGH", "", "", "", round(report.total_opex_high, 2), "", "", ""])
            
            # Production metrics
            if report.annual_h2_production_kg > 0:
                writer.writerow([])
                writer.writerow(["METRICS", "", "", "", "", "", "", ""])
                writer.writerow(["Annual H2 Production (kg)", "", "", "", round(report.annual_h2_production_kg, 0), "", "", ""])
                writer.writerow(["OPEX per kg H2", "", "", "", round(report.opex_per_kg_h2, 4), "", "", ""])
        
        logger.info(f"✓ CSV export: {path}")
