"""
CAPEX Configuration Generator

Professional-grade CAPEX generator that:
1. Extracts design capacities from component parameters and simulation history
2. Maps topology IDs to equipment tags
3. Calculates C_BM using configurable cost strategies
4. Generates detailed JSON/CSV outputs with formulas, notes, and uncertainty bands
"""

import json
import csv
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Set

import yaml
import numpy as np
import pandas as pd

from h2_plant.economics.models import (
    EquipmentMapping,
    CostCoefficients,
    CapexEntry,
    CapexReport,
    CEPCIData,
    AACECostClass,
    BlockCostSummary,
)
from h2_plant.core.constants import DryCoolerIndirectConstants as DCC
from h2_plant.economics.cost_strategies import get_strategy, CostStrategy

logger = logging.getLogger(__name__)


# Default component type coefficients (Turton 2018)
DEFAULT_COEFFICIENTS: Dict[str, Dict[str, Any]] = {
    "Centrifugal Compressor": {
        "K1": 2.2891, "K2": 1.3604, "K3": -0.1027,
        "F_m": 2.5, "F_m_note": "Stainless Steel 316 for steam/H2",
        "F_BM": 2.15, "F_BM_note": "Package installation",
        "capacity_min": 450, "capacity_max": 3000, "capacity_unit": "kW"
    },
    "Reciprocating Compressor": {
        "K1": 2.0309, "K2": 1.2524, "K3": -0.0638,
        "F_m": 2.1, "F_m_note": "Alloy Steel for H2 service",
        "F_BM": 2.15,
        "capacity_min": 10, "capacity_max": 10000, "capacity_unit": "kW"
    },
    "Air-Cooled Heat Exchanger": {
        "K1": 4.3247, "K2": -0.303, "K3": 0.1634,
        "F_m": 2.7, "F_m_note": "Stainless Steel 316SS tubes",
        "F_p": 1.092, "B1": 1.63, "B2": 1.66,
        "capacity_min": 10, "capacity_max": 1000, "capacity_unit": "m²"
    },
    "Shell and Tube Heat Exchanger": {
        "K1": 4.8306, "K2": -0.8509, "K3": 0.3187,
        "F_m": 2.7, "B1": 1.63, "B2": 1.66, "F_p": 1.0,
        "capacity_min": 10, "capacity_max": 1000, "capacity_unit": "m²"
    },
    "Electric Heater": {
        "K1": 2.858, "K2": 0.8209, "K3": 0.0075,
        "F_BM": 1.3, "F_m": 1.0,
        "capacity_min": 10, "capacity_max": 1000, "capacity_unit": "kW"
    },
    "Vertical Pressure Vessel": {
        "K1": 3.4974, "K2": 0.4485, "K3": 0.1074,
        "F_m": 3.1, "F_m_note": "SS316 for H2/process",
        "F_p": 1.0, "B1": 2.25, "B2": 1.82,
        "capacity_min": 0.3, "capacity_max": 520, "capacity_unit": "m³"
    },
    "Horizontal Pressure Vessel": {
        "K1": 3.5565, "K2": 0.3776, "K3": 0.0905,
        "F_m": 3.1, "F_p": 1.0, "B1": 1.49, "B2": 1.52,
        "capacity_min": 0.1, "capacity_max": 628, "capacity_unit": "m³"
    },
    "Centrifugal Pump": {
        "K1": 3.3892, "K2": 0.0536, "K3": 0.1538,
        "F_m": 2.0, "F_BM": 3.30,
        "capacity_min": 1, "capacity_max": 300, "capacity_unit": "kW"
    },
    "PSA Unit": {
        "K1": 4.0, "K2": 0.7, "K3": 0.0,
        "F_BM": 2.5, "F_m": 1.5,
        "capacity_min": 10, "capacity_max": 5000, "capacity_unit": "Nm³/h"
    },
    "Electrolyzer SOEC": {
        "K1": 5.5, "K2": 0.65, "K3": 0.0,
        "F_BM": 1.2, "F_m": 1.0,
        "F_BM_note": "Modular factory installation",
        "capacity_min": 100, "capacity_max": 100000, "capacity_unit": "kW"
    },
    "Electrolyzer PEM": {
        "K1": 5.3, "K2": 0.70, "K3": 0.0,
        "F_BM": 1.2, "F_m": 1.0,
        "capacity_min": 100, "capacity_max": 50000, "capacity_unit": "kW"
    },
    "Electrical Chiller": {
        "K1": 4.2523, "K2": 0.7615, "K3": -0.0031,
        "F_BM": 1.3, "F_m": 1.0,  # Assumed standard material based on F_BM usage
        "capacity_min": 10, "capacity_max": 1000, "capacity_unit": "kW"
    },
}

# Electric Motor Coefficients (Turton 2018)
# Used as driver for Centrifugal Pumps
MOTOR_COEFFS_SMALL = {"K1": 3.3432, "K2": 0.2761, "K3": 0.0543} # <= 75 kW
MOTOR_COEFFS_LARGE = {"K1": 2.9508, "K2": 1.0688, "K3": -0.1315} # > 75 kW to 2600 kW


class CapexGenerator:
    """
    Professional CAPEX configuration generator.
    
    Usage:
        generator = CapexGenerator()
        generator.load_config("equipment_mappings.yaml")
        report = generator.generate(registry, monitoring, output_dir)
    """

    PARAM_MAPPINGS: Dict[str, List[str]] = {
        "cross_sectional_area_m2": [
            "cross_sectional_area_m2", "area_cross_section_m2",
        ],
        "power_kw": [
            "max_power_kw", "rated_power_kw", "power_kw", "P_max",
            "rated_power_mw", "max_power_mw", "max_power_nominal_mw",
        ],
        "area_m2": [
            "area_m2", "heat_transfer_area_m2", "A_hx", "exchange_area_m2",
        ],
        "volume_m3": [
            "volume_m3", "V_tank", "total_volume_m3", "capacity_m3",
        ],
        "flow_kg_h": [
            "max_flow_kg_h", "design_flow_kg_h", "capacity_kg_h", "rated_flow_kg_h",
        ],
        "flow_nm3_h": [
            "capacity_nm3_h", "design_capacity_nm3_h", "rated_capacity_nm3_h",
        ],
        "flow_m3_s": [
            "flow_m3_s", "volumetric_flow_m3_s",
        ],
    }

    HISTORY_MAPPINGS: Dict[str, List[str]] = {
        "power_kw": [
            "power_kw", "power_input_kw", "P_consumed_kw", "electrical_power_kw",
            "timestep_power_kw", "power_shaft_kw", "power_fluid_kw",
            "energy_consumed_kwh",
        ],
        "flow_kg_h": [
            "mass_flow_kg_h", "outlet_mass_flow_kg_h", "actual_mass_transferred_kg",
        ],
        "area_m2": [
            "heat_duty_kw",
            "q_transferred_kw",
            "heat_rejected_kw",
            "cooling_load_kw",
            "tqc_duty_kw",
        ],
    }
    
    def __init__(
        self,
        cepci: Optional[CEPCIData] = None,
        default_strategy: str = "turton",
        capacity_mode: str = "design"
    ):
        """
        Initialize generator.
        
        Args:
            cepci: CEPCI data for inflation adjustment
            default_strategy: Default cost estimation strategy
            capacity_mode: Global capacity extraction mode ('design' or 'history')
                - 'design': Use design parameters from topology (max_power_kw, etc.)
                - 'history': Use maximum observed values from simulation history
        """
        self.cepci = cepci or CEPCIData()
        self.default_strategy = default_strategy
        self.capacity_mode = capacity_mode
        self.mappings: List[EquipmentMapping] = []
        self.type_coefficients = DEFAULT_COEFFICIENTS.copy()
        self._history_maxima: Dict[str, float] = {}  # Cache for CSV/Parquet history maxima
        self._history_maxima_loaded: bool = False
        self._history_scan_output_dir: Optional[Path] = None
        self._history_scan_workers: int = 0
        self._history_scan_max_memory_mb: Optional[int] = None
        self._history_scan_mode: str = "auto"
        self.installation_factors: Dict[str, Dict[str, float]] = {}  # Block -> Category -> %
    
    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> "CapexGenerator":
        """
        Create generator from YAML configuration file.
        
        Args:
            config_path: Path to equipment_mappings.yaml
            
        Returns:
            Configured CapexGenerator instance
        """
        generator = cls()
        generator.load_config(config_path)
        return generator
    
    def load_config(self, config_path: Union[str, Path]) -> None:
        """
        Load equipment mappings and coefficients from YAML.
        
        Args:
            config_path: Path to configuration file
        """
        path = Path(config_path)
        if not path.exists():
            logger.warning(f"Config file not found: {path}. Using defaults.")
            return
        
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Load CEPCI data
        if 'cepci' in config:
            self.cepci = CEPCIData(**config['cepci'])
        
        # Load global capacity mode
        if 'capacity_mode' in config:
            mode = config['capacity_mode'].lower()
            if mode in ('design', 'history'):
                self.capacity_mode = mode
                logger.info(f"Capacity mode set to: {mode}")
            else:
                logger.warning(f"Invalid capacity_mode '{mode}', using 'design'")
        
        # Load installation factors
        if 'installation_factors' in config:
            self.installation_factors = config['installation_factors']
            logger.info(f"Loaded installation factors for {len(self.installation_factors)} blocks")
        
        # Load type coefficients
        if 'coefficients' in config:
            for comp_type, coeffs in config['coefficients'].items():
                self.type_coefficients[comp_type] = coeffs
        
        def _normalize_topology_ids(raw_ids: Any) -> List[str]:
            if raw_ids is None:
                return []
            if isinstance(raw_ids, str):
                return [part.strip() for part in raw_ids.split(",") if part.strip()]
            if isinstance(raw_ids, list):
                normalized: List[str] = []
                for item in raw_ids:
                    if item is None:
                        continue
                    if isinstance(item, str):
                        if "," in item:
                            normalized.extend([part.strip() for part in item.split(",") if part.strip()])
                        else:
                            normalized.append(item.strip())
                    else:
                        normalized.append(str(item))
                return normalized
            return [str(raw_ids)]

        # Load equipment mappings
        if 'equipment' in config:
            for mapping_data in config['equipment']:
                # Normalize topology_ids to avoid comma-separated strings
                if 'topology_ids' in mapping_data:
                    mapping_data['topology_ids'] = _normalize_topology_ids(mapping_data.get('topology_ids'))

                # Get coefficients from type or inline
                comp_type = mapping_data.get('component_type', '')
                if 'coefficients' not in mapping_data and comp_type in self.type_coefficients:
                    mapping_data['coefficients'] = CostCoefficients(
                        **self.type_coefficients[comp_type]
                    )
                elif 'coefficients' in mapping_data:
                    coeffs = mapping_data['coefficients']
                    if coeffs.get('cost_method') in ('power_law_scaling',):
                        # Non-Turton cost method — store raw dict, skip CostCoefficients
                        logger.info(f"Skipping CostCoefficients for {mapping_data.get('tag')}: "
                                    f"cost_method={coeffs['cost_method']}")
                        mapping_data['coefficients'] = None
                    else:
                        mapping_data['coefficients'] = CostCoefficients(**coeffs)
                
                self.mappings.append(EquipmentMapping(**mapping_data))
        
        logger.info(f"Loaded {len(self.mappings)} equipment mappings from {path}")

    def _derive_required_history_columns(self) -> Set[str]:
        """Build required history column names from mappings/topology IDs."""
        required_columns: Set[str] = set()
        for mapping in self.mappings:
            history_keys = self.HISTORY_MAPPINGS.get(
                mapping.capacity_variable,
                [mapping.capacity_variable],
            )
            for topology_id in mapping.topology_ids:
                if not topology_id:
                    continue
                for key in history_keys:
                    required_columns.add(f"{topology_id}_{key}")
        return required_columns

    @staticmethod
    def _merge_maxima(target: Dict[str, float], source: Dict[str, float]) -> None:
        """In-place max-reduction of column maxima."""
        for col_name, value in source.items():
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(parsed):
                continue
            if col_name in target:
                target[col_name] = max(target[col_name], parsed)
            else:
                target[col_name] = parsed

    @staticmethod
    def _compute_numeric_column_maxima(df: pd.DataFrame) -> Dict[str, float]:
        maxima: Dict[str, float] = {}
        for col_name in df.columns:
            numeric = pd.to_numeric(df[col_name], errors="coerce")
            if numeric.notna().any():
                maxima[col_name] = float(numeric.max())
        return maxima

    def _resolve_history_scan_workers(
        self,
        *,
        requested_workers: int,
        file_count: int,
        max_memory_mb: Optional[int],
        needed_columns_count: int,
    ) -> int:
        if file_count <= 1:
            return 1

        workers = requested_workers if requested_workers > 0 else 2
        workers = max(1, min(workers, file_count))

        if max_memory_mb and max_memory_mb > 0:
            # Conservative bound per worker; scales with selected column count.
            estimated_worker_mb = max(64.0, min(512.0, max(1, needed_columns_count) * 0.25))
            max_workers_by_memory = max(1, int(max_memory_mb // estimated_worker_mb))
            workers = max(1, min(workers, max_workers_by_memory, file_count))

        return workers

    def _scan_parquet_file_maxima(
        self,
        pq_file: Path,
        required_columns: Optional[Set[str]],
        scan_mode: str,
    ) -> Dict[str, float]:
        """
        Compute maxima for one parquet file with optional stats-first mode.

        scan_mode:
            - auto: row-group stats first; fallback to column read for missing stats.
            - stats: row-group stats only.
            - read: direct column read only.
        """
        mode = scan_mode.lower()
        if mode not in {"auto", "stats", "read"}:
            mode = "auto"

        file_maxima: Dict[str, float] = {}
        try:
            import pyarrow.parquet as pq

            parquet_file = pq.ParquetFile(pq_file)
            schema_names = parquet_file.schema.names
            if required_columns:
                selected_columns = [c for c in schema_names if c in required_columns]
            else:
                selected_columns = list(schema_names)

            if not selected_columns:
                return {}

            if mode == "read":
                df = pd.read_parquet(pq_file, columns=selected_columns)
                return self._compute_numeric_column_maxima(df)

            # Stats-based path
            column_indices = {name: idx for idx, name in enumerate(schema_names)}
            missing_stats: Set[str] = set()
            for rg_idx in range(parquet_file.metadata.num_row_groups):
                row_group = parquet_file.metadata.row_group(rg_idx)
                for col_name in selected_columns:
                    col_idx = column_indices[col_name]
                    stats = row_group.column(col_idx).statistics
                    if stats is None or stats.max is None:
                        missing_stats.add(col_name)
                        continue
                    try:
                        value = float(stats.max)
                    except (TypeError, ValueError):
                        missing_stats.add(col_name)
                        continue
                    if not np.isfinite(value):
                        missing_stats.add(col_name)
                        continue
                    if col_name in file_maxima:
                        file_maxima[col_name] = max(file_maxima[col_name], value)
                    else:
                        file_maxima[col_name] = value

            if mode == "stats":
                return file_maxima

            fallback_columns = [
                c for c in selected_columns
                if c not in file_maxima or c in missing_stats
            ]
            if fallback_columns:
                fallback_df = pd.read_parquet(pq_file, columns=fallback_columns)
                fallback_maxima = self._compute_numeric_column_maxima(fallback_df)
                self._merge_maxima(file_maxima, fallback_maxima)
        except Exception as exc:
            logger.warning(f"Failed to scan parquet maxima from {pq_file}: {exc}")
            return {}

        return file_maxima

    def _ensure_history_maxima_loaded(self) -> None:
        """Load history maxima on first use, only if needed."""
        if self._history_maxima_loaded:
            return

        if self._history_scan_output_dir is None:
            self._history_maxima = {}
            self._history_maxima_loaded = True
            return

        required_columns = self._derive_required_history_columns()
        if not required_columns:
            self._history_maxima = {}
            self._history_maxima_loaded = True
            return

        self._history_maxima = self._load_history_maxima(
            output_dir=self._history_scan_output_dir,
            required_columns=required_columns,
            workers=self._history_scan_workers,
            max_memory_mb=self._history_scan_max_memory_mb,
            scan_mode=self._history_scan_mode,
        )
        self._history_maxima_loaded = True
    
    def _extract_capacity(
        self,
        topology_ids: List[str],
        capacity_variable: str,
        aggregation: str,
        registry: Any,
        monitoring: Any,
        capacity_mode: Optional[str] = None
    ) -> tuple[float, int, str, List[str]]:
        """
        Extract design capacity from components.
        
        Extraction priority depends on capacity_mode:
        
        MODE 'design' (default):
            1. Direct component parameters (max_power_kw, volume_m3, etc.)
            2. Calculated from component attributes (volume_per_tank * n_tanks, etc.)
            3. Fallback to monitoring history if design params not found
            
        MODE 'history':
            1. Monitoring history (max observed value over simulation)
            2. Fallback to component parameters if no history found
        
        Args:
            topology_ids: Component IDs to check
            capacity_variable: Variable to extract (power_kw, area_m2, volume_m3, flow_kg_h)
            aggregation: How to combine multiple values (sum, max, avg)
            registry: ComponentRegistry
            monitoring: MonitoringSystem (optional)
            capacity_mode: Override mode ('design' or 'history'), defaults to self.capacity_mode
            
        Returns:
            Tuple of (capacity, source_description, notes)
        """
        # Use provided mode or fall back to global setting
        mode = (capacity_mode or self.capacity_mode).lower()
        values = []
        notes = []
        source = "unknown"
        total_num_units = 1  # Default unit count multiplier

        for comp_id in topology_ids:
            comp_value = None
            comp_source = None
            comp = None
            
            # Get component from registry (needed for design mode)
            if registry is not None and hasattr(registry, 'has') and registry.has(comp_id):
                comp = registry.get(comp_id)
            
            # =====================================================================
            # MODE-BASED EXTRACTION
            # =====================================================================
            
            if mode == "history":
                # HISTORY MODE: Try monitoring history FIRST
                # This gives actual operational capacity
                
                # Try monitoring history first
                history_val = self._extract_from_history(
                    comp_id, capacity_variable, monitoring, self.HISTORY_MAPPINGS
                )
                if history_val is not None:
                    comp_value, hist_note = history_val
                    comp_source = hist_note
                    source = "monitoring_history"
                
                # Fallback to design parameters if no history
                if comp_value is None and comp is not None:
                    param_names = self.PARAM_MAPPINGS.get(capacity_variable, [capacity_variable])
                    for param in param_names:
                        if hasattr(comp, param):
                            val = getattr(comp, param)
                            if val is not None and val > 0:
                                if param == "rated_power_mw":
                                    val = val * 1000
                                comp_value = float(val)
                                comp_source = f"{comp_id}: {param} = {val} (fallback from design)"
                                source = "component_parameter"
                                break
                    
                    # Try calculated attributes as final fallback
                    if comp_value is None:
                        calculated = self._calculate_capacity_from_attributes(
                            comp, comp_id, capacity_variable
                        )
                        if calculated is not None:
                            comp_value, calc_note = calculated
                            tried = self.PARAM_MAPPINGS.get(capacity_variable, [capacity_variable])
                            comp_source = calc_note
                            source = f"calculated (fallback; {capacity_variable} not in history or params: {', '.join(tried)})"
            
            else:
                # DESIGN MODE (default): Try component parameters FIRST
                # This gives design/sizing-based capacity
                
                if comp is not None:
                    # ===== TIER 1: Direct component parameters =====
                    param_names = self.PARAM_MAPPINGS.get(capacity_variable, [capacity_variable])
                    
                    for param in param_names:
                        if hasattr(comp, param):
                            val = getattr(comp, param)
                            if val is not None and val > 0:
                                # Handle unit conversions
                                # Handle unit conversions
                                if param in ["rated_power_mw", "max_power_mw", "max_power_nominal_mw"]:
                                    val = val * 1000  # MW to kW
                                
                                comp_value = float(val)
                                comp_source = f"{comp_id}: {param} = {val}"
                                source = "component_parameter"
                                break
                    
                    # ===== TIER 2: Calculated from composite attributes =====
                    if comp_value is None:
                        calculated = self._calculate_capacity_from_attributes(
                            comp, comp_id, capacity_variable
                        )
                        if calculated is not None:
                            comp_value, calc_note = calculated
                            tried = self.PARAM_MAPPINGS.get(capacity_variable, [capacity_variable])
                            comp_source = calc_note
                            source = f"calculated (fallback; {capacity_variable} not in params: {', '.join(tried)})"
                
                # ===== TIER 3: Monitoring history as fallback =====
                # Try this even if component is not in registry (uses CSV cache)
                if comp_value is None:
                    history_val = self._extract_from_history(
                        comp_id, capacity_variable, monitoring, self.HISTORY_MAPPINGS
                    )
                    if history_val is not None:
                        comp_value, hist_note = history_val
                        comp_source = f"{hist_note} (fallback from history)"
                        source = "monitoring_history"
            
            # Record result for this component
            if comp_value is not None:
                values.append(comp_value)
                notes.append(comp_source)
            else:
                if comp is None:
                    notes.append(f"❌ {comp_id} not found in registry or history")
                else:
                    tried = self.PARAM_MAPPINGS.get(capacity_variable, [capacity_variable])
                    notes.append(f"⚠️ {comp_id}: no {capacity_variable} found (tried params: {', '.join(tried)})")
        
        # Aggregate values across all components
        if not values:
            notes.append(f"❌ No capacity data found for {capacity_variable}")
            return 0.0, 1, "not_found", notes
        
        if aggregation == "sum":
            capacity = sum(values)
        elif aggregation == "max":
            capacity = max(values)
        elif aggregation == "avg":
            capacity = sum(values) / len(values)
        else:
            capacity = sum(values)
        
        return capacity, total_num_units, source, notes
    
    def _calculate_capacity_from_attributes(
        self,
        comp: Any,
        comp_id: str,
        capacity_variable: str
    ) -> Optional[tuple[float, str]]:
        """
        Calculate capacity from composite component attributes.
        
        Examples:
        - volume_m3 = volume_per_tank * n_tanks (for tank arrays)
        - total_volume_m3 = length * diameter^2 * π/4 (for vessels)
        
        Returns:
            Tuple of (calculated_value, description_note) or None
        """
        # Tank array: volume = volume_per_tank * n_tanks
        if capacity_variable == "volume_m3":
            if hasattr(comp, 'volume_per_tank') and hasattr(comp, 'n_tanks'):
                vol_per = getattr(comp, 'volume_per_tank', 0)
                n = getattr(comp, 'n_tanks', 0)
                if vol_per > 0 and n > 0:
                    total = vol_per * n
                    return total, f"{comp_id}: volume_per_tank({vol_per}) × n_tanks({n}) = {total}"
            
            # Single tank with volume_per_tank
            if hasattr(comp, 'volume_per_tank'):
                vol = getattr(comp, 'volume_per_tank', 0)
                if vol > 0:
                    return vol, f"{comp_id}: volume_per_tank = {vol}"
            
            # CHECK FOR MODULAR DESIGN (e.g. Coalescers split into small shells)
            # This is a special logic where we define capacity = module_volume
            # and calculate N = total_volume / module_volume
            if hasattr(comp, 'modular_design') and getattr(comp, 'modular_design') is True:
                 # Logic handled in _extract_capacity wrapper? 
                 # Ideally we calculate the *Single Module* capacity here if possible
                 # But we need the Total Volume to find N.
                 # Let's extract Total Volume first.
                 pass
            
            # Geometric calculation for vertical vessels (e.g. KnockOutDrum)
            
            # Geometric calculation for vertical vessels (e.g. KnockOutDrum)
            # Volume = (π * D^2 / 4) * L
            # If Length not specified, assume L/D = 3.0 (typical for vertical separators)
            if hasattr(comp, 'diameter_m'):
                D = getattr(comp, 'diameter_m', 0.0)
                if D > 0:
                    L = getattr(comp, 'length_m', 0.0)
                    L_D_ratio = getattr(comp, 'L_D_ratio', 3.0)
                    
                    if L <= 0:
                        L = D * L_D_ratio
                        note_suffix = f"(Assuming L/D={L_D_ratio})"
                    else:
                        note_suffix = f"(L={L}m)"
                    
                    vol = (3.14159 * (D**2) / 4) * L
                    return vol, f"{comp_id}: Calc from D={D}m {note_suffix} -> V={vol:.2f} m³"
        
        # Compressor: max_flow_kg_h might be the sizing parameter
        if capacity_variable == "power_kw":
            # Some compressors are sized by flow, not power
            # We'll need history for actual power
            if (
                hasattr(comp, 'max_flow_kg_h') and
                hasattr(comp, 'inlet_pressure_bar') and
                hasattr(comp, 'outlet_pressure_bar')
            ):
                max_flow_kg_h = getattr(comp, 'max_flow_kg_h', 0.0) or 0.0
                inlet_p_bar = getattr(comp, 'inlet_pressure_bar', 0.0) or 0.0
                outlet_p_bar = getattr(comp, 'outlet_pressure_bar', 0.0) or 0.0
                inlet_t_c = getattr(comp, 'inlet_temperature_c', 25.0) or 25.0

                if max_flow_kg_h > 0 and inlet_p_bar > 0 and outlet_p_bar > inlet_p_bar:
                    # Infer gas properties from component ID (fallback to air)
                    comp_id_upper = comp_id.upper()
                    if "H2" in comp_id_upper:
                        gamma = 1.41
                        R_spec = 4124.0  # J/(kg·K)
                    elif "O2" in comp_id_upper:
                        gamma = 1.40
                        R_spec = 259.8
                    elif "STEAM" in comp_id_upper or "H2O" in comp_id_upper:
                        gamma = 1.33
                        R_spec = 461.5
                    elif "CO2" in comp_id_upper:
                        gamma = 1.30
                        R_spec = 188.9
                    else:
                        gamma = 1.40
                        R_spec = 287.0

                    eta_is = getattr(comp, 'isentropic_efficiency', 0.7) or 0.7
                    eta_m = getattr(comp, 'mechanical_efficiency', 0.96) or 0.96

                    pr = outlet_p_bar / inlet_p_bar
                    t_in_k = inlet_t_c + 273.15

                    # Isentropic compression work per kg (J/kg)
                    exponent = (gamma - 1.0) / gamma
                    work_j_kg = (gamma / (gamma - 1.0)) * R_spec * t_in_k * (pr ** exponent - 1.0)

                    m_dot = max_flow_kg_h / 3600.0
                    denom = max(eta_is, 1e-3) * max(eta_m, 1e-3)
                    power_kw = (m_dot * work_j_kg / denom) / 1000.0

                    if power_kw > 0:
                        return power_kw, (
                            f"{comp_id}: est power from max_flow_kg_h({max_flow_kg_h:.1f}), "
                            f"PR={pr:.2f}, T={inlet_t_c:.1f}C"
                        )

            # Pump: estimate power from capacity_kg_h and target_pressure
            if hasattr(comp, 'capacity_kg_h') and hasattr(comp, 'target_pressure_pa'):
                cap_kg_h = getattr(comp, 'capacity_kg_h', 0.0) or 0.0
                target_pa = getattr(comp, 'target_pressure_pa', 0.0) or 0.0
                eta_is = getattr(comp, 'eta_is', 0.82) or 0.82
                eta_m = getattr(comp, 'eta_m', 0.96) or 0.96

                # Assume inlet at atmospheric (1 bar = 1e5 Pa) for water pumps
                inlet_pa = 1e5
                if target_pa > inlet_pa and cap_kg_h > 0:
                    delta_p = target_pa - inlet_pa
                    rho_water = 998.0  # kg/m³
                    m_dot = cap_kg_h / 3600.0
                    vol_flow = m_dot / rho_water  # m³/s
                    hydraulic_power = vol_flow * delta_p  # W
                    power_kw = (hydraulic_power / (max(eta_is, 1e-3) * max(eta_m, 1e-3))) / 1000.0
                    if power_kw > 0:
                        return power_kw, (
                            f"{comp_id}: est pump power from capacity_kg_h({cap_kg_h:.1f}), "
                            f"ΔP={delta_p/1e5:.1f} bar"
                        )

        # Heat exchanger area extraction (including DryCooler)
        if capacity_variable == "area_m2":
            # 1. Check for explicit area attribute
            if hasattr(comp, 'area_m2') and getattr(comp, 'area_m2') > 0:
                 A = getattr(comp, 'area_m2')
                 return A, f"{comp_id}: area_m2 = {A}"
            
            # 2. DryCooler: Sum of TQC + DC areas (Conditional)
            # If use_central_utility is True, we ONLY pay for TQC (Interchanger), not the air fan/coil deck.
            use_central = getattr(comp, 'use_central_utility', False)
            
            # Get internal areas (lazily computed or 0.0)
            tqc = getattr(comp, 'tqc_area_m2', 0.0)
            dc = getattr(comp, 'dc_area_m2', 0.0)
            
            # Estimation logic if lazy geometry not yet initialized
            if tqc == 0.0 and dc == 0.0:
                 if hasattr(comp, 'design_capacity_kw'):
                    cap_kw = getattr(comp, 'design_capacity_kw', 100.0)
                    scale = cap_kw / 100.0
                    
                    is_o2 = 'O2' in comp_id or 'Oxygen' in comp_id
                    
                    if is_o2:
                         tqc = DCC.AREA_O2_TQC_M2 * scale
                         dc = DCC.AREA_O2_DC_M2 * scale
                    else:
                         tqc = DCC.AREA_H2_TQC_M2 * scale
                         dc = DCC.AREA_H2_DC_M2 * scale
            
            # Final Summation based on Topology
            if use_central:
                return tqc, f"{comp_id}: TQC Area Only ({tqc:.1f}) [Central Utility]"
            else:
                return tqc + dc, f"{comp_id}: TQC({tqc:.1f}) + DC({dc:.1f}) = {tqc+dc:.1f}"
        
        return None
    
    def _extract_from_history(
        self,
        comp_id: str,
        capacity_variable: str,
        monitoring: Any,
        history_mappings: Dict[str, List[str]]
    ) -> Optional[tuple[float, str]]:
        """
        Extract max value from monitoring history (in-memory or CSV file).
        
        Priority:
        1. In-memory component_metrics (if not in lightweight mode)
        2. CSV/Parquet history file via _history_maxima cache
        
        For compressors: finds max power_kw over simulation
        For heat exchangers: finds max heat_duty_kw
        
        Returns:
            Tuple of (max_value, description_note) or None
        """
        history_keys = history_mappings.get(capacity_variable, [capacity_variable])
        
        # TIER 3a: Try in-memory component_metrics first
        if monitoring is not None and hasattr(monitoring, 'component_metrics'):
            comp_metrics = monitoring.component_metrics.get(comp_id, {})
            if comp_metrics:
                for key in history_keys:
                    if key in comp_metrics:
                        history = comp_metrics[key]
                        if history and len(history) > 0:
                            if isinstance(history, (list, np.ndarray)):
                                max_val = float(np.max(history))
                            else:
                                max_val = float(history)
                            
                            if max_val > 0:
                                return max_val, f"{comp_id}: max({key}) from memory = {max_val:.2f}"
        
        # TIER 3b: Try CSV history cache (loaded from file)
        if not self._history_maxima_loaded:
            self._ensure_history_maxima_loaded()

        if self._history_maxima:
            for key in history_keys:
                # Build column name pattern: comp_id_key
                col_name = f"{comp_id}_{key}"
                if col_name in self._history_maxima:
                    max_val = self._history_maxima[col_name]
                    if max_val > 0:
                        return max_val, f"{comp_id}: max({key}) from history = {max_val:.2f}"
        
        return None
    
    def _load_history_maxima(
        self,
        output_dir: Path,
        required_columns: Optional[Set[str]] = None,
        workers: int = 0,
        max_memory_mb: Optional[int] = None,
        scan_mode: str = "auto",
    ) -> Dict[str, float]:
        """
        Load simulation history and extract max values per column.
        
        This is used as a fallback when monitoring.component_metrics is empty
        (e.g., in lightweight mode). Looks for:
        1. simulation_history.csv
        2. history_chunk_*.parquet files
        
        Args:
            output_dir: Directory containing history files
            required_columns: Optional explicit subset of columns to scan.
            workers: Thread-pool workers for parquet file scans (0 = auto).
            max_memory_mb: Optional soft memory cap for worker resolution.
            scan_mode: auto | stats | read.
            
        Returns:
            Dict mapping column names to their max values
        """
        maxima: Dict[str, float] = {}
        required = set(required_columns) if required_columns else None

        def _chunk_sort_key(path: Path) -> tuple[int, int, str]:
            stem = path.stem
            suffix = stem.split("_")[-1]
            if suffix.isdigit():
                return (0, int(suffix), path.name)
            return (1, 0, path.name)

        def _load_csv_maxima(csv_path: Path) -> Dict[str, float]:
            csv_maxima: Dict[str, float] = {}
            try:
                header_df = pd.read_csv(csv_path, nrows=0)
                available_cols = list(header_df.columns)
            except Exception as exc:
                logger.warning(f"Failed to inspect CSV history {csv_path}: {exc}")
                return {}

            if required is not None:
                usecols = [c for c in available_cols if c in required]
            else:
                usecols = available_cols

            if not usecols:
                return {}

            try:
                for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=10_000):
                    self._merge_maxima(csv_maxima, self._compute_numeric_column_maxima(chunk))
            except Exception as exc:
                logger.warning(f"Failed to load CSV history from {csv_path}: {exc}")
                return {}

            return csv_maxima

        def _load_parquet_maxima(parquet_files: List[Path]) -> Dict[str, float]:
            if not parquet_files:
                return {}

            needed_columns_count = len(required) if required is not None else 1
            resolved_workers = self._resolve_history_scan_workers(
                requested_workers=workers,
                file_count=len(parquet_files),
                max_memory_mb=max_memory_mb,
                needed_columns_count=needed_columns_count,
            )
            logger.info(
                "History maxima scan: %d parquet files, workers=%d, mode=%s",
                len(parquet_files),
                resolved_workers,
                scan_mode,
            )

            parquet_maxima: Dict[str, float] = {}
            ordered_files = sorted(parquet_files, key=_chunk_sort_key)

            def _scan(path: Path) -> Dict[str, float]:
                return self._scan_parquet_file_maxima(path, required, scan_mode)

            if resolved_workers <= 1:
                for pq_file in ordered_files:
                    self._merge_maxima(parquet_maxima, _scan(pq_file))
                return parquet_maxima

            with ThreadPoolExecutor(max_workers=resolved_workers) as executor:
                # executor.map preserves input ordering for deterministic reduction
                scanned = executor.map(_scan, ordered_files)
                for file_maxima in scanned:
                    self._merge_maxima(parquet_maxima, file_maxima)
            return parquet_maxima

        def _try_dir(base_dir: Path) -> Dict[str, float]:
            dir_maxima: Dict[str, float] = {}

            csv_path = base_dir / "simulation_history.csv"
            if csv_path.exists():
                self._merge_maxima(dir_maxima, _load_csv_maxima(csv_path))

            chunks_dir = base_dir / "history_chunks"
            parquet_files: List[Path] = []
            if chunks_dir.exists():
                parquet_files = list(chunks_dir.glob("chunk_*.parquet"))
            if not parquet_files:
                parquet_files = list(base_dir.glob("chunk_*.parquet"))
            if parquet_files:
                self._merge_maxima(dir_maxima, _load_parquet_maxima(parquet_files))

            return dir_maxima

        # Try output_dir first, then parent (common when output_dir is /Economics)
        self._merge_maxima(maxima, _try_dir(output_dir))
        if not maxima and output_dir.name.lower() == "economics":
            self._merge_maxima(maxima, _try_dir(output_dir.parent))
        elif not maxima and output_dir.parent != output_dir:
            self._merge_maxima(maxima, _try_dir(output_dir.parent))

        if maxima:
            logger.info(f"Loaded {len(maxima)} history maxima columns")
        else:
            logger.info("No history maxima loaded (no matching columns/files found)")

        return maxima

    
    def _calculate_cost(
        self,
        design_capacity: float,
        mapping: EquipmentMapping,
    ) -> tuple[Optional[float], Optional[float], str, AACECostClass, bool]:
        """
        Calculate cost using appropriate strategy.
        
        Returns:
            Tuple of (C_p0, C_BM, formula, cost_class, within_bounds)
        """
        if mapping.cost_source == "excluded":
            return (0.0, 0.0, "Excluded", AACECostClass.CLASS_5, True)

        if mapping.cost_source == "fixed":
            cost_eur = 0.0
            formula = "Fixed Cost"
            if mapping.vendor_quote_eur:
                cost_eur = mapping.vendor_quote_eur
                formula = f"Fixed: €{mapping.vendor_quote_eur:,.0f}"

            return (cost_eur, cost_eur, formula, AACECostClass.CLASS_1, True)

        if mapping.cost_source == "iea_scaling":
            # Linear scaling without inflation (current cost basis)
            # K1 = Unit Cost (USD/unit), B1 = Installation Factor
            coeffs = mapping.coefficients
            if not coeffs:
                return (None, None, "Missing coefficients for iea_scaling", AACECostClass.CLASS_5, False)
                
            unit_cost = coeffs.K1
            factor = coeffs.B1
            
            cp0 = design_capacity * unit_cost
            c_bm = cp0 * factor
            
            formula = f"IEA Method: {design_capacity:,.1f} {mapping.capacity_unit} * €{unit_cost} * {factor}"
            return (cp0, c_bm, formula, AACECostClass.CLASS_4, True)

        strategy = get_strategy(mapping.cost_source)
        
        # Get coefficients
        coefficients = mapping.coefficients
        if coefficients is None and mapping.component_type in self.type_coefficients:
            coefficients = CostCoefficients(**self.type_coefficients[mapping.component_type])
        
        # Check bounds
        within_bounds = True
        if coefficients:
            if coefficients.capacity_min and design_capacity < coefficients.capacity_min:
                within_bounds = False
            if coefficients.capacity_max and design_capacity > coefficients.capacity_max:
                within_bounds = False
        
        # Calculate cost
        kwargs = {}
        if mapping.vendor_quote_eur:
            kwargs['vendor_quote_eur'] = mapping.vendor_quote_eur
        
        C_p0, C_BM, formula, cost_class = strategy.calculate(
            design_capacity=design_capacity,
            coefficients=coefficients,
            cepci=self.cepci,
            **kwargs
        )
        
        if mapping.component_type == "Centrifugal Pump":
             motor_cost, motor_formula = self._calculate_motor_cost(design_capacity)
             
             # Add motor cost to totals
             if C_p0 is not None:
                 C_p0 += motor_cost
             else:
                 C_p0 = motor_cost
                 
             if C_BM is not None:
                 C_BM += motor_cost
             else:
                 C_BM = motor_cost
                 
             formula += f" || + {motor_formula}"
        
        return C_p0, C_BM, formula, cost_class, within_bounds
    
    def _calculate_motor_cost(self, power_shaft_kw: float) -> tuple[float, str]:
        """
        Calculate cost of Electric Motor driver for pumps.
        
        Args:
            power_shaft_kw: Shaft power required by pump
            
        Returns:
            Tuple of (inflated_cost_usd, formula_string)
        """
        # Assumptions
        motor_efficiency = 0.90
        
        # Calculate required motor power rating
        motor_power_kw = power_shaft_kw / motor_efficiency
        
        # Select coefficients based on power
        if motor_power_kw <= 75:
            coeffs = MOTOR_COEFFS_SMALL
            range_desc = "<=75kW"
        else:
            coeffs = MOTOR_COEFFS_LARGE
            range_desc = ">75kW"
            if motor_power_kw > 2600:
                logger.warning(f"Motor power {motor_power_kw:.1f} kW exceeds correlation max (2600 kW)")

        # Calculate Base Cost (Cp0) 
        # log10(Cp0) = K1 + K2*log10(P) + K3*(log10(P)^2)
        if motor_power_kw <= 0:
            return 0.0, "Motor(P_elec=0kW): skipped"

        log_P = np.log10(motor_power_kw)
        log_Cp0 = coeffs["K1"] + coeffs["K2"] * log_P + coeffs["K3"] * (log_P ** 2)
        cp0_usd = 10 ** log_Cp0
        
        # Inflate to current year
        # Motors are "off-the-shelf" components, essentially C_BM = Cp0 * inflation
        # (Installation is typically covered in the Pump's Bare Module factor, 
        #  but the purchase cost of the motor itself must be added)
        inflated_cost = cp0_usd * self.cepci.inflation_factor
        
        formula = (
            f"Motor(P_elec={motor_power_kw:.1f}kW): €{inflated_cost:,.0f} "
            f"[{range_desc} K1={coeffs['K1']}]"
        )
        
        return inflated_cost, formula
    
    def generate(
        self,
        registry: Any = None,
        monitoring: Any = None,
        output_dir: Optional[Path] = None,
        simulation_name: Optional[str] = None,
        simulation_hours: Optional[int] = None,
        history_scan_workers: int = 0,
        history_scan_max_memory_mb: Optional[int] = None,
        history_scan_mode: str = "auto",
    ) -> CapexReport:
        """
        Generate CAPEX report.
        
        Args:
            registry: ComponentRegistry for capacity extraction
            monitoring: MonitoringSystem for history-based extraction
            output_dir: Directory for output files
            simulation_name: Name for report metadata
            simulation_hours: Simulation duration for metadata
            history_scan_workers: Worker count for history scans (0 = auto).
            history_scan_max_memory_mb: Optional soft cap for history scan workers.
            history_scan_mode: History scan mode: auto | stats | read.
            
        Returns:
            CapexReport with all entries and totals
        """
        # Reset history cache and defer loading until actually needed by fallback extraction.
        self._history_maxima = {}
        self._history_maxima_loaded = False
        self._history_scan_output_dir = (
            Path(output_dir) if output_dir and not isinstance(output_dir, Path) else output_dir
        )
        self._history_scan_workers = max(0, int(history_scan_workers or 0))
        self._history_scan_max_memory_mb = (
            int(history_scan_max_memory_mb)
            if history_scan_max_memory_mb is not None
            else None
        )
        if self._history_scan_max_memory_mb is not None and self._history_scan_max_memory_mb <= 0:
            self._history_scan_max_memory_mb = None
        mode = (history_scan_mode or "auto").lower()
        if mode not in {"auto", "stats", "read"}:
            logger.warning(
                "Invalid history_scan_mode '%s'; falling back to 'auto'.",
                history_scan_mode,
            )
            mode = "auto"
        self._history_scan_mode = mode
        
        report = CapexReport(
            generated_at=datetime.now().isoformat(),
            simulation_name=simulation_name,
            total_simulation_hours=simulation_hours,
            cepci=self.cepci,
        )
        
        for mapping in self.mappings:
            # Extract capacity (use per-equipment mode if specified, else global)
            capacity, num_units, capacity_source, notes = self._extract_capacity(
                topology_ids=mapping.topology_ids,
                capacity_variable=mapping.capacity_variable,
                aggregation=mapping.capacity_aggregation,
                registry=registry,
                monitoring=monitoring,
                capacity_mode=mapping.capacity_mode  # Per-equipment override
            )
            
            # SPECIAL: Modular Design Check (Attribute-based override from Mapping)
            # If mapping says 'modular_design': true, we recalculate N and Capacity
            if getattr(mapping, 'modular_design', False):
                # General modular split (applies to area/power/flow/volume)
                module_capacity = getattr(mapping, 'module_capacity', None)
                module_count = getattr(mapping, 'module_count', None)

                if module_capacity and module_capacity > 0:
                    if module_count and module_count > 0:
                        num_units = int(module_count)
                        capacity = float(module_capacity)
                        notes.append(
                            f"MODULAR: Using fixed {num_units} x {capacity:.3f} {mapping.capacity_unit} units"
                        )
                    else:
                        required_units = int(np.ceil(capacity / module_capacity)) if capacity > 0 else 0
                        if required_units > 0:
                            num_units = required_units
                            capacity = float(module_capacity)
                            notes.append(
                                f"MODULAR: Split {capacity * num_units:.3f} {mapping.capacity_unit} "
                                f"into {num_units} x {module_capacity:.3f} {mapping.capacity_unit} units"
                            )
                    capacity_source += " [Modular Split]"
                else:
                    # Volume-specific modular split (legacy behavior)
                    module_d = getattr(mapping, 'module_d_shell', 0.3)
                    # Recalculate module capacity (Volume of cylinder D=0.3, L=5D?)
                    # L/D = 5 assumed for standard vertical vessels
                    L_mod = 5.0 * module_d
                    module_vol = (3.14159 * (module_d**2) / 4) * L_mod
                    
                    total_vol_required = capacity  # The 'capacity' returned is the Total Volume
                    
                    if module_vol > 0:
                         required_units = int(np.ceil(total_vol_required / module_vol))
                         # Update for cost calculation
                         capacity = module_vol
                         num_units = required_units
                         notes.append(f"MODULAR: Split {total_vol_required:.2f} m3 into {num_units} x {module_vol:.3f} m3 units (D={module_d}m)")
                         capacity_source += " [Modular Split]"
            
            # Calculate cost (Unit Cost)
            C_p0, C_BM, formula, cost_class, within_bounds = self._calculate_cost(
                design_capacity=capacity,
                mapping=mapping
            )
            
            # Apply Modular Multiplier
            if num_units > 1 and C_BM is not None:
                C_p0 = C_p0 * num_units
                C_BM = C_BM * num_units
                formula = f"{num_units} x ({formula})"
            
            
            # Calculate uncertainty bands
            if C_BM:
                low_factor, high_factor = cost_class.accuracy_range
                C_BM_low = C_BM * low_factor
                C_BM_high = C_BM * high_factor
            else:
                C_BM_low = None
                C_BM_high = None
            
            # Build entry
            warnings = []
            errors = []
            
            # Suppress capacity error if we have a valid cost from a direct source (Vendor Quote)
            # This respects the user's "source of truth" in equipment_mappings.yaml
            is_direct_cost = mapping.cost_source in ["vendor_quote", "fixed_cost", "manual", "excluded"]
            
            if not within_bounds and not is_direct_cost:
                warnings.append(f"Capacity {capacity} outside correlation bounds")
            
            if capacity == 0:
                if C_BM is not None and is_direct_cost:
                    # Logic: If we have a price, we don't strictly need the capacity for the report to be valid
                    pass 
                else:
                    errors.append("Failed to extract design capacity")

            if C_BM is None:
                errors.append("Cost calculation failed")
            
            # Get coefficient dict for audit
            coeffs_dict = None
            if mapping.coefficients:
                coeffs_dict = mapping.coefficients.model_dump(exclude_none=True)
            elif mapping.component_type in self.type_coefficients:
                coeffs_dict = self.type_coefficients[mapping.component_type]
            
            entry = CapexEntry(
                tag=mapping.tag,
                name=mapping.name,
                block=mapping.block,
                topology_ids=mapping.topology_ids,
                component_type=mapping.component_type,
                design_capacity=round(capacity, 2),
                capacity_unit=mapping.capacity_unit,
                capacity_source=capacity_source,
                capacity_within_bounds=within_bounds,
                C_p0=C_p0,
                C_BM=C_BM,
                C_BM_low=round(C_BM_low, 2) if C_BM_low else None,
                C_BM_high=round(C_BM_high, 2) if C_BM_high else None,
                cost_formula=formula,
                cost_source=mapping.cost_source,
                coefficients=coeffs_dict,
                cost_class=cost_class,
                notes=notes + mapping.notes,
                warnings=warnings,
                errors=errors,
            )
            
            report.entries.append(entry)
        
        # Calculate equipment totals
        report.calculate_totals()
        
        # Calculate block costs with installation factors
        self._calculate_block_costs(report)
        
        # Export if output_dir provided
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            self._export_json(report, output_dir / "capex_report.json")
            self._export_csv(report, output_dir / "capex_report.csv")
            
            logger.info(f"CAPEX report generated: {output_dir}")
            logger.info(f"  Equipment Total C_BM: €{report.total_C_BM:,.0f}")
            logger.info(f"  Installation Total: €{report.total_installation:,.0f}")
            logger.info(f"  Total Installed Cost: €{report.total_installed_cost:,.0f}")
            logger.info(f"  Entries: {report.entries_with_cost}/{len(report.entries)} with valid cost")
        
        return report
    
    def _export_json(self, report: CapexReport, path: Path) -> None:
        """Export report to JSON."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(report.model_dump(), f, indent=2, default=str)
        logger.info(f"✓ JSON export: {path}")
    
    def _export_csv(self, report: CapexReport, path: Path) -> None:
        """Export report to CSV."""
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                "Tag", "Name", "Block", "Component Type", "Topology IDs",
                "Design Capacity", "Unit", "Capacity Source", "Capacity Method",
                "C_BM (EUR)", "C_BM Low", "C_BM High", "Cost Class",
                "Formula", "Within Bounds", "Warnings", "Errors"
            ])

            # Data rows
            for entry in report.entries:
                writer.writerow([
                    entry.tag,
                    entry.name,
                    entry.block,
                    entry.component_type,
                    ", ".join(entry.topology_ids),
                    entry.design_capacity,
                    entry.capacity_unit,
                    entry.capacity_source,
                    entry.cost_source,
                    entry.C_BM or "",
                    entry.C_BM_low or "",
                    entry.C_BM_high or "",
                    entry.cost_class.value,
                    entry.cost_formula or "",
                    "Yes" if entry.capacity_within_bounds else "No",
                    "; ".join(entry.warnings),
                    "; ".join(entry.errors),
                ])
            
            # Summary row
            writer.writerow([])
            writer.writerow(["TOTAL", "", "", "", "", "", "", "", "",
                           report.total_C_BM, report.total_C_BM_low, report.total_C_BM_high,
                           report.overall_cost_class.value, "", "", "", ""])

            # Append Block Summary Section
            writer.writerow([])
            writer.writerow([])
            writer.writerow(["BLOCK SUMMARY"] + [""] * 16)
            writer.writerow([
                "Block",
                "Equipment Count",
                "Equipment Total (EUR)",
                "Equipment Low (EUR)",
                "Equipment High (EUR)",
                "Installation Categories",
                "Installation Total (EUR)",
                "Installation Low (EUR)",
                "Installation High (EUR)",
                "Total Installed Cost (EUR)",
                "Total Installed Low (EUR)",
                "Total Installed High (EUR)",
            ] + [""] * 5)

            for summary in report.block_summaries:
                install_cats = "; ".join([f"{k}: €{v:,.0f}" for k, v in summary.installation_costs.items()])
                writer.writerow([
                    summary.block_name,
                    len(summary.equipment_tags),
                    round(summary.equipment_total, 0),
                    round(summary.equipment_total_low, 0),
                    round(summary.equipment_total_high, 0),
                    install_cats,
                    round(summary.installation_total, 0),
                    round(summary.installation_total_low, 0),
                    round(summary.installation_total_high, 0),
                    round(summary.total_installed_cost, 0),
                    round(summary.total_installed_cost_low, 0),
                    round(summary.total_installed_cost_high, 0),
                ] + [""] * 5)

            # Overall Totals with Installation
            writer.writerow([])
            writer.writerow([
                "OVERALL TOTAL",
                len(report.entries),
                round(report.total_C_BM, 0),
                round(report.total_C_BM_low, 0),
                round(report.total_C_BM_high, 0),
                "",
                round(report.total_installation, 0),
                round(report.total_installation_low, 0),
                round(report.total_installation_high, 0),
                round(report.total_installed_cost, 0),
                round(report.total_installed_cost_low, 0),
                round(report.total_installed_cost_high, 0),
            ] + [""] * 5)

        logger.info(f"✓ CSV export: {path}")
    
    def _calculate_block_costs(self, report: CapexReport) -> None:
        """
        Calculate block costs and apply installation factors.
        
        Groups equipment by block, sums C_BM per block, applies installation
        percentages, and calculates total installed cost.
        """
        # Group equipment by block
        block_equipment: Dict[str, List[str]] = {}
        block_costs: Dict[str, Dict[str, float]] = {}
        
        for mapping in self.mappings:
            block = mapping.block
            if block not in block_equipment:
                block_equipment[block] = []
                block_costs[block] = {
                    "base": 0.0,
                    "low": 0.0,
                    "high": 0.0,
                }
            block_equipment[block].append(mapping.tag)
        
        # Sum equipment costs per block
        for entry in report.entries:
            # Find which block this entry belongs to
            for block, tags in block_equipment.items():
                if entry.tag in tags:
                    block_costs[block]["base"] += entry.C_BM or 0.0
                    block_costs[block]["low"] += entry.C_BM_low or 0.0
                    block_costs[block]["high"] += entry.C_BM_high or 0.0
                    break
        
        # Create block summaries with installation factors
        total_installation = 0.0
        total_installation_low = 0.0
        total_installation_high = 0.0
        report.block_summaries = []
        for block_name, equipment_tags in block_equipment.items():
            factors = self.installation_factors.get(block_name, {})
            
            summary = BlockCostSummary(
                block_name=block_name,
                equipment_tags=equipment_tags,
                equipment_total=block_costs.get(block_name, {}).get("base", 0.0),
                equipment_total_low=block_costs.get(block_name, {}).get("low", 0.0),
                equipment_total_high=block_costs.get(block_name, {}).get("high", 0.0),
            )
            
            # Apply installation factors
            factor_sum = 0.0
            for category, pct in factors.items():
                cost = summary.equipment_total * pct
                summary.installation_costs[category] = round(cost, 2)
                factor_sum += pct
            
            summary.installation_total = sum(summary.installation_costs.values())
            summary.installation_total_low = round(summary.equipment_total_low * factor_sum, 2)
            summary.installation_total_high = round(summary.equipment_total_high * factor_sum, 2)
            summary.total_installed_cost = summary.equipment_total + summary.installation_total
            summary.total_installed_cost_low = summary.equipment_total_low + summary.installation_total_low
            summary.total_installed_cost_high = summary.equipment_total_high + summary.installation_total_high
            total_installation += summary.installation_total
            total_installation_low += summary.installation_total_low
            total_installation_high += summary.installation_total_high
            
            report.block_summaries.append(summary)
        
        # Update report totals
        report.total_installation = total_installation
        report.total_installation_low = total_installation_low
        report.total_installation_high = total_installation_high
        report.total_installed_cost = report.total_C_BM + total_installation
        report.total_installed_cost_low = report.total_C_BM_low + total_installation_low
        report.total_installed_cost_high = report.total_C_BM_high + total_installation_high
        
        logger.info(f"Calculated costs for {len(report.block_summaries)} blocks")
    
