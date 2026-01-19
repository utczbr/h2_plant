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
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

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


class CapexGenerator:
    """
    Professional CAPEX configuration generator.
    
    Usage:
        generator = CapexGenerator()
        generator.load_config("equipment_mappings.yaml")
        report = generator.generate(registry, monitoring, output_dir)
    """
    
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
        
        # Load equipment mappings
        if 'equipment' in config:
            for mapping_data in config['equipment']:
                # Get coefficients from type or inline
                comp_type = mapping_data.get('component_type', '')
                if 'coefficients' not in mapping_data and comp_type in self.type_coefficients:
                    mapping_data['coefficients'] = CostCoefficients(
                        **self.type_coefficients[comp_type]
                    )
                elif 'coefficients' in mapping_data:
                    mapping_data['coefficients'] = CostCoefficients(
                        **mapping_data['coefficients']
                    )
                
                self.mappings.append(EquipmentMapping(**mapping_data))
        
        logger.info(f"Loaded {len(self.mappings)} equipment mappings from {path}")
    
    def _extract_capacity(
        self,
        topology_ids: List[str],
        capacity_variable: str,
        aggregation: str,
        registry: Any,
        monitoring: Any,
        capacity_mode: Optional[str] = None
    ) -> tuple[float, str, List[str]]:
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
        
        # Mapping of capacity_variable to potential attribute names
        param_mappings = {
            "power_kw": [
                "max_power_kw", "rated_power_kw", "power_kw", "P_max",
                "rated_power_mw", "max_power_mw", "max_power_nominal_mw",  # Need to convert MW to kW
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
        
        # History-based mappings for monitoring data
        history_mappings = {
            "power_kw": [
                "power_kw", "P_consumed_kw", "electrical_power_kw",
                "timestep_power_kw", "energy_consumed_kwh",  # Need to derive power
            ],
            "flow_kg_h": [
                "mass_flow_kg_h", "outlet_mass_flow_kg_h", "actual_mass_transferred_kg",
            ],
            "area_m2": [
                "heat_duty_kw",  # Might need to derive from duty
            ],
        }
        
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
                    comp_id, capacity_variable, monitoring, history_mappings
                )
                if history_val is not None:
                    comp_value, hist_note = history_val
                    comp_source = hist_note
                    source = "monitoring_history"
                
                # Fallback to design parameters if no history
                if comp_value is None and comp is not None:
                    param_names = param_mappings.get(capacity_variable, [capacity_variable])
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
                            comp_source = f"{calc_note} (fallback from design)"
                            source = "calculated"
            
            else:
                # DESIGN MODE (default): Try component parameters FIRST
                # This gives design/sizing-based capacity
                
                if comp is not None:
                    # ===== TIER 1: Direct component parameters =====
                    param_names = param_mappings.get(capacity_variable, [capacity_variable])
                    
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
                            comp_source = calc_note
                            source = "calculated"
                
                # ===== TIER 3: Monitoring history as fallback =====
                # Try this even if component is not in registry (uses CSV cache)
                if comp_value is None:
                    history_val = self._extract_from_history(
                        comp_id, capacity_variable, monitoring, history_mappings
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
                    notes.append(f"⚠️ {comp_id}: no {capacity_variable} data found")
        
        # Aggregate values across all components
        if not values:
            notes.append(f"❌ No capacity data found for {capacity_variable}")
            return 0.0, "not_found", notes
        
        if aggregation == "sum":
            capacity = sum(values)
        elif aggregation == "max":
            capacity = max(values)
        elif aggregation == "avg":
            capacity = sum(values) / len(values)
        else:
            capacity = sum(values)
        
        return capacity, source, notes
    
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
            if hasattr(comp, 'max_flow_kg_h'):
                # Can't directly convert flow to power - need history
                pass
        
        # Heat exchanger area extraction (including DryCooler)
        if capacity_variable == "area_m2":
            # 1. Check for explicit area attribute
            if hasattr(comp, 'area_m2') and getattr(comp, 'area_m2') > 0:
                 A = getattr(comp, 'area_m2')
                 return A, f"{comp_id}: area_m2 = {A}"
            
            # 2. DryCooler: Sum of TQC + DC areas
            # Note: DryCooler lazily configures geometry on first flow.
            # If 0, we estimate based on design_capacity_kw and type hint.
            tqc = getattr(comp, 'tqc_area_m2', 0.0)
            dc = getattr(comp, 'dc_area_m2', 0.0)
            
            if tqc + dc > 0.0:
                 return tqc + dc, f"{comp_id}: TQC({tqc:.1f}) + DC({dc:.1f}) = {tqc+dc:.1f}"
            
            # Fallback for DryCooler if not initialized (Laziness)
            if hasattr(comp, 'design_capacity_kw'):
                cap_kw = getattr(comp, 'design_capacity_kw', 100.0)
                scale = cap_kw / 100.0
                
                # Check component ID for service type hint
                is_o2 = 'O2' in comp_id or 'Oxygen' in comp_id
                
                # Constants from h2_plant.core.constants.DryCoolerIndirectConstants
                if is_o2:
                     # O2 Service (Base 100kW)
                     base_area = DCC.AREA_O2_TQC_M2 + DCC.AREA_O2_DC_M2
                     A_est = base_area * scale
                     return A_est, f"{comp_id}: Est. O2 Area ({base_area:.1f}*{scale:.1f}) = {A_est:.1f} (Default)"
                else:
                     # H2 Service (Base 100kW)
                     base_area = DCC.AREA_H2_TQC_M2 + DCC.AREA_H2_DC_M2
                     A_est = base_area * scale
                     return A_est, f"{comp_id}: Est. H2 Area ({base_area:.1f}*{scale:.1f}) = {A_est:.1f} (Default)"
        
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
        if self._history_maxima:
            for key in history_keys:
                # Build column name pattern: comp_id_key
                col_name = f"{comp_id}_{key}"
                if col_name in self._history_maxima:
                    max_val = self._history_maxima[col_name]
                    if max_val > 0:
                        return max_val, f"{comp_id}: max({key}) from CSV = {max_val:.2f}"
        
        return None
    
    def _load_history_maxima(self, output_dir: Path) -> Dict[str, float]:
        """
        Load simulation history and extract max values per column.
        
        This is used as a fallback when monitoring.component_metrics is empty
        (e.g., in lightweight mode). Looks for:
        1. simulation_history.csv
        2. history_chunk_*.parquet files
        
        Args:
            output_dir: Directory containing history files
            
        Returns:
            Dict mapping column names to their max values
        """
        maxima: Dict[str, float] = {}
        
        # Try CSV first
        csv_path = output_dir / "simulation_history.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path, nrows=0)  # Get columns only
                columns = df.columns.tolist()
                
                # Read in chunks to handle large files
                chunk_size = 10000
                for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
                    for col in chunk.columns:
                        if chunk[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                            chunk_max = chunk[col].max()
                            if col in maxima:
                                maxima[col] = max(maxima[col], chunk_max)
                            else:
                                maxima[col] = chunk_max
                
                logger.info(f"Loaded {len(maxima)} column maxima from {csv_path}")
                return maxima
            except Exception as e:
                logger.warning(f"Failed to load CSV history: {e}")
        
        # Try Parquet chunks
        parquet_pattern = list(output_dir.glob("history_chunk_*.parquet"))
        if parquet_pattern:
            try:
                for pq_file in sorted(parquet_pattern):
                    df = pd.read_parquet(pq_file)
                    for col in df.columns:
                        if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                            chunk_max = df[col].max()
                            if col in maxima:
                                maxima[col] = max(maxima[col], chunk_max)
                            else:
                                maxima[col] = chunk_max
                
                logger.info(f"Loaded {len(maxima)} column maxima from {len(parquet_pattern)} Parquet files")
                return maxima
            except Exception as e:
                logger.warning(f"Failed to load Parquet history: {e}")
        
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
            cost_usd = 0.0
            formula = "Fixed Cost"
            if mapping.fixed_cost_eur:
                cost_usd = mapping.fixed_cost_eur  # Approx EUR -> USD
                formula = f"Fixed: €{mapping.fixed_cost_eur:,.0f}"
            elif mapping.vendor_quote_usd:
                cost_usd = mapping.vendor_quote_usd
                formula = f"Fixed: ${mapping.vendor_quote_usd:,.0f}"
            
            return (cost_usd, cost_usd, formula, AACECostClass.CLASS_1, True)

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
            
            formula = f"IEA Method: {design_capacity:,.1f} {mapping.capacity_unit} * ${unit_cost} * {factor}"
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
        if mapping.vendor_quote_usd:
            kwargs['vendor_quote_usd'] = mapping.vendor_quote_usd
        
        C_p0, C_BM, formula, cost_class = strategy.calculate(
            design_capacity=design_capacity,
            coefficients=coefficients,
            cepci=self.cepci,
            **kwargs
        )
        
        return C_p0, C_BM, formula, cost_class, within_bounds
    
    def generate(
        self,
        registry: Any = None,
        monitoring: Any = None,
        output_dir: Optional[Path] = None,
        simulation_name: Optional[str] = None,
        simulation_hours: Optional[int] = None
    ) -> CapexReport:
        """
        Generate CAPEX report.
        
        Args:
            registry: ComponentRegistry for capacity extraction
            monitoring: MonitoringSystem for history-based extraction
            output_dir: Directory for output files
            simulation_name: Name for report metadata
            simulation_hours: Simulation duration for metadata
            
        Returns:
            CapexReport with all entries and totals
        """
        # Load history maxima from CSV/Parquet (for TIER 3b extraction)
        if output_dir:
            output_path = Path(output_dir) if not isinstance(output_dir, Path) else output_dir
            self._history_maxima = self._load_history_maxima(output_path)
        
        report = CapexReport(
            generated_at=datetime.now().isoformat(),
            simulation_name=simulation_name,
            total_simulation_hours=simulation_hours,
            cepci=self.cepci,
        )
        
        for mapping in self.mappings:
            # Extract capacity (use per-equipment mode if specified, else global)
            capacity, capacity_source, notes = self._extract_capacity(
                topology_ids=mapping.topology_ids,
                capacity_variable=mapping.capacity_variable,
                aggregation=mapping.capacity_aggregation,
                registry=registry,
                monitoring=monitoring,
                capacity_mode=mapping.capacity_mode  # Per-equipment override
            )
            
            # Calculate cost
            C_p0, C_BM, formula, cost_class, within_bounds = self._calculate_cost(
                design_capacity=capacity,
                mapping=mapping
            )
            
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
            
            if not within_bounds:
                warnings.append(f"Capacity {capacity} outside correlation bounds")
            
            # Suppress capacity error if we have a valid cost from a direct source (Vendor Quote)
            # This respects the user's "source of truth" in equipment_mappings.yaml
            is_direct_cost = mapping.cost_source in ["vendor_quote", "fixed_cost", "manual"]
            
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
            logger.info(f"  Equipment Total C_BM: ${report.total_C_BM:,.0f}")
            logger.info(f"  Installation Total: ${report.total_installation:,.0f}")
            logger.info(f"  Total Installed Cost: ${report.total_installed_cost:,.0f}")
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
                "Tag", "Name", "Component Type", "Topology IDs",
                "Design Capacity", "Unit", "Capacity Source",
                "C_BM (USD)", "C_BM Low", "C_BM High", "Cost Class",
                "Formula", "Within Bounds", "Warnings", "Errors"
            ])
            
            # Data rows
            for entry in report.entries:
                writer.writerow([
                    entry.tag,
                    entry.name,
                    entry.component_type,
                    ", ".join(entry.topology_ids),
                    entry.design_capacity,
                    entry.capacity_unit,
                    entry.capacity_source,
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
            writer.writerow(["TOTAL", "", "", "", "", "", "",
                           report.total_C_BM, report.total_C_BM_low, report.total_C_BM_high,
                           report.overall_cost_class.value, "", "", "", ""])
        
        
            # Append Block Summary Section
            writer.writerow([])
            writer.writerow([])
            writer.writerow(["BLOCK SUMMARY", "", "", "", "", "", "", "", "", "", "", "", "", "", ""])
            writer.writerow([
                "Block", "Equipment Count", "Equipment Total (USD)",
                "Installation Categories", "Installation Total (USD)",
                "Total Installed Cost (USD)", "", "", "", "", "", "", "", "", ""
            ])
            
            for summary in report.block_summaries:
                install_cats = "; ".join([f"{k}: ${v:,.0f}" for k, v in summary.installation_costs.items()])
                writer.writerow([
                    summary.block_name,
                    len(summary.equipment_tags),
                    round(summary.equipment_total, 0),
                    install_cats,
                    round(summary.installation_total, 0),
                    round(summary.total_installed_cost, 0),
                    "", "", "", "", "", "", "", "", ""
                ])
                
            # Overall Totals with Installation
            writer.writerow([])
            writer.writerow([
                "OVERALL TOTAL", len(report.entries),
                round(report.total_C_BM, 0), "",
                round(report.total_installation, 0),
                round(report.total_installed_cost, 0),
                "", "", "", "", "", "", "", "", ""
            ])

        logger.info(f"✓ CSV export: {path}")
    
    def _calculate_block_costs(self, report: CapexReport) -> None:
        """
        Calculate block costs and apply installation factors.
        
        Groups equipment by block, sums C_BM per block, applies installation
        percentages, and calculates total installed cost.
        """
        # Group equipment by block
        block_equipment: Dict[str, List[str]] = {}
        block_costs: Dict[str, float] = {}
        
        for mapping in self.mappings:
            block = mapping.block
            if block not in block_equipment:
                block_equipment[block] = []
                block_costs[block] = 0.0
            block_equipment[block].append(mapping.tag)
        
        # Sum equipment costs per block
        for entry in report.entries:
            # Find which block this entry belongs to
            for block, tags in block_equipment.items():
                if entry.tag in tags:
                    block_costs[block] += entry.C_BM or 0.0
                    break
        
        # Create block summaries with installation factors
        total_installation = 0.0
        for block_name, equipment_tags in block_equipment.items():
            factors = self.installation_factors.get(block_name, {})
            
            summary = BlockCostSummary(
                block_name=block_name,
                equipment_tags=equipment_tags,
                equipment_total=block_costs.get(block_name, 0.0),
            )
            
            # Apply installation factors
            for category, pct in factors.items():
                cost = summary.equipment_total * pct
                summary.installation_costs[category] = round(cost, 2)
            
            summary.installation_total = sum(summary.installation_costs.values())
            summary.total_installed_cost = summary.equipment_total + summary.installation_total
            total_installation += summary.installation_total
            
            report.block_summaries.append(summary)
        
        # Update report totals
        report.total_installation = total_installation
        report.total_installed_cost = report.total_C_BM + total_installation
        
        logger.info(f"Calculated costs for {len(report.block_summaries)} blocks")
    

