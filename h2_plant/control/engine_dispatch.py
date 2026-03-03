"""
Integrated Dispatch Strategy for SimulationEngine.
OPTIMIZED: Pre-bound array access and zero-flow guarding.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, TYPE_CHECKING, List, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import logging

from h2_plant.control.dispatch import (
    DispatchInput,
    DispatchState,
    DispatchResult,
    DispatchStrategy as BaseDispatchStrategy,
    ReferenceHybridStrategy,
    SoecOnlyStrategy,
    EconomicSpotDispatchStrategy
)
from h2_plant.core.enums import DispatchStrategyEnum

if TYPE_CHECKING:
    from h2_plant.core.component_registry import ComponentRegistry
    from h2_plant.config.plant_config import SimulationContext
from h2_plant.components.electrolysis.soec_operator import SOECOperator
from h2_plant.components.storage.detailed_tank import DetailedTankArray

# Import specific component types for type checking
from h2_plant.components.thermal.chiller import Chiller
from h2_plant.components.separation.coalescer import Coalescer
from h2_plant.components.purification.deoxo_reactor import DeoxoReactor
from h2_plant.components.separation.psa import PSA
from h2_plant.components.separation.psa_syngas import SyngasPSA
from h2_plant.components.separation.knock_out_drum import KnockOutDrum
from h2_plant.components.separation.hydrogen_cyclone import HydrogenMultiCyclone
from h2_plant.components.thermal.interchanger import Interchanger
from h2_plant.components.compression.compressor_single import CompressorSingle
from h2_plant.components.cooling.dry_cooler import DryCooler
from h2_plant.components.thermal.heat_exchanger import HeatExchanger
from h2_plant.components.thermal.heat_exchanger import HeatExchanger
from h2_plant.components.thermal.electric_boiler import ElectricBoiler
from h2_plant.components.reforming.integrated_atr_plant import IntegratedATRPlant
from h2_plant.components.water.drain_recorder_mixer import DrainRecorderMixer
from h2_plant.components.water.water_pump import WaterPumpThermodynamic
from h2_plant.components.storage.h2_tank import TankArray
from h2_plant.components.storage.h2_storage_enhanced import H2StorageTankEnhanced
from h2_plant.components.storage.detailed_tank import DetailedTankArray
from h2_plant.components.water.ultrapure_water_tank import UltraPureWaterTank
from h2_plant.components.water.ultrapure_water_tank import UltraPureWaterTank
from h2_plant.components.delivery.discharge_station import DischargeStation
from h2_plant.components.mixing.multicomponent_mixer import MultiComponentMixer
from h2_plant.components.mixing.water_mixer import WaterMixer
from h2_plant.components.external.biogas_source import BiogasSource
from h2_plant.components.external.water_source import ExternalWaterSource
from h2_plant.optimization.numba_ops import calculate_storage_mpc_factor

logger = logging.getLogger(__name__)


@dataclass
class IntegratedDispatchState:
    P_soec_prev: float = 0.0
    force_sell: bool = False
    step_idx: int = 0
    cumulative_h2_kg: float = 0.0


@dataclass
class StreamRecorder:
    """
    Helper struct to hold pre-resolved array references for fast recording.
    Eliminates dict lookups and string formatting in the hot loop.
    """
    component: Any
    stream_attr: str  # e.g., 'outlet_stream'
    temp_arr: np.ndarray
    press_arr: np.ndarray
    flow_arr: np.ndarray
    h2o_frac_arr: np.ndarray
    h2o_frac_arr: np.ndarray
    h2o_vapor_arr: np.ndarray  # Water vapor mass flow (kg/h)
    mole_arrs: List[np.ndarray] # (H2, O2, N2, H2O, CH4, CO2) - MUTABLE LIST
    
    # Specific component metric arrays (optional)
    # Changed from List[Tuple] to List[List] to allow re-binding inner array reference
    extra_metric_arrs: List[List[Any]] = field(default_factory=list)

    # Column names for chunked history re-binding
    temp_col_name: str = ""
    press_col_name: str = ""
    flow_col_name: str = ""
    h2o_frac_col_name: str = ""
    mole_cols: List[Tuple[str, str]] = field(default_factory=list) # List of (species, col_name)
    extra_metric_cols: List[Tuple[str, str]] = field(default_factory=list) # List of (attr, col_name)
    
    # Optimization: Pre-bound accessor for stream retrieval
    stream_getter: Any = None  # Callable[[], Optional[Stream]]

    def bind_accessor(self):
        """
        Determine the optimal way to access the stream and bind a callable.
        This removes dynamic checks from the hot loop.
        """
        # 1. Known Port Method (e.g. get_output('h2_out'))
        if hasattr(self.component, 'get_output') and self.stream_attr in ('outlet', 'purified_gas_out', 'h2_out'):
            # Bind directly to the method call with fixed argument
            # Lambda captures self.component and self.stream_attr
            self.stream_getter = lambda: self.component.get_output(self.stream_attr)
            return

        # 2. Direct Attribute (e.g. comp.outlet_stream)
        if hasattr(self.component, self.stream_attr):
            # Check if it's really an attribute or a method
            val = getattr(self.component, self.stream_attr)
            if not callable(val):
                self.stream_getter = lambda: getattr(self.component, self.stream_attr)
                return
        
        # 3. Fallback: Try get_output for any attribute name if direct access fails
        if hasattr(self.component, 'get_output'):
             self.stream_getter = lambda: self._safe_get_output()
             return

        # 4. Dead End: Always return None
        self.stream_getter = lambda: None

    def _safe_get_output(self):
        try:
            return self.component.get_output(self.stream_attr)
        except (ValueError, KeyError, AttributeError):
            return None




class EngineDispatchStrategy(ABC):
    @abstractmethod
    def initialize(self, registry: 'ComponentRegistry', context: 'SimulationContext', total_steps: int) -> None:
        pass

    @abstractmethod
    def decide_and_apply(self, t: float, prices: np.ndarray, wind: np.ndarray) -> None:
        pass

    @abstractmethod
    def get_history(self) -> Dict[str, np.ndarray]:
        pass


class HybridArbitrageEngineStrategy(ReferenceHybridStrategy):
    def __init__(self):
        self._registry: Optional['ComponentRegistry'] = None
        self._context: Optional['SimulationContext'] = None
        self._inner_strategy: Optional[BaseDispatchStrategy] = None
        self._state = IntegratedDispatchState()
        
        # Strategy override (set by runner for CLI/config selection)
        self._strategy_override: Optional[str] = None

        # Component references
        self._soec = None
        self._pem = None
        self._atr = None
        
        # Performance: Pre-bound recorders
        self._recorders: List[StreamRecorder] = []

        # Capacity cache
        self._soec_capacity: float = 0.0
        self._pem_max: float = 0.0

        # Pre-allocated history arrays
        self._history: Dict[str, np.ndarray] = {}
        self._total_steps: int = 0
        self._cached_history: Optional[Dict[str, np.ndarray]] = None
        self._biogas_flow_key: Optional[str] = None
        self._water_flow_key: Optional[str] = None

    def initialize(
        self,
        registry: 'ComponentRegistry',
        context: 'SimulationContext',
        total_steps: int,
        output_dir: 'Path' = None,
        use_chunked_history: bool = False,
        resume: bool = False
    ) -> None:
        self._registry = registry
        self._context = context
        self._total_steps = total_steps

        # Detect topology
        self._soec = self._find_soec(registry)
        self._pem = self._find_pem(registry)
        self._atr = self._find_atr(registry)
        
        # Lookup Power Transformers for efficiency gross-up
        self._soec_trafo = registry.get('SOEC_Transformer') if registry.has('SOEC_Transformer') else None
        self._pem_trafo = registry.get('PEM_Transformer') if registry.has('PEM_Transformer') else None
        self._bop_trafo = registry.get('BOP_Transformer') if registry.has('BOP_Transformer') else None
        
        # Cache transformer efficiencies (default 1.0 if no transformer)
        self._η_soec_trafo = self._soec_trafo.efficiency if self._soec_trafo else 1.0
        self._η_pem_trafo = self._pem_trafo.efficiency if self._pem_trafo else 1.0
        self._η_bop_trafo = self._bop_trafo.efficiency if self._bop_trafo else 1.0
        
        if self._soec_trafo:
            logger.info(f"SOEC Transformer found: η={self._η_soec_trafo:.3f}")
        if self._pem_trafo:
            logger.info(f"PEM Transformer found: η={self._η_pem_trafo:.3f}")
        if self._bop_trafo:
            logger.info(f"BOP Transformer found: η={self._η_bop_trafo:.3f}")
        
        # Strategy selection: CLI/config override > topology auto-detection
        if self._strategy_override:
            strategy_name = self._strategy_override.upper()
            if strategy_name == "SOEC_ONLY":
                logger.info(f"Using strategy (config/CLI): SoecOnlyStrategy")
                self._inner_strategy = SoecOnlyStrategy()
            elif strategy_name == "ECONOMIC_SPOT":
                logger.info(f"Using strategy (config/CLI): EconomicSpotDispatchStrategy")
                self._inner_strategy = EconomicSpotDispatchStrategy()
            else:  # REFERENCE_HYBRID or default
                logger.info(f"Using strategy (config/CLI): ReferenceHybridStrategy")
                self._inner_strategy = ReferenceHybridStrategy()
        else:
            # Auto-detect based on topology
            if self._soec and not self._pem:
                logger.info("Topology detected: SOEC Only. Using SoecOnlyStrategy.")
                self._inner_strategy = SoecOnlyStrategy()
            else:
                logger.info("Topology detected: Hybrid. Using ReferenceHybridStrategy.")
                self._inner_strategy = ReferenceHybridStrategy()

        # Cache component capacities
        if self._soec:
            spec = context.physics.soec_cluster
            self._soec_capacity = spec.num_modules * spec.max_power_nominal_mw * spec.optimal_limit
        if self._pem:
            self._pem_max = context.physics.pem_system.max_power_mw

        # =====================================================================
        # HISTORY STORAGE: Chunked (memory-efficient) or In-Memory (fast)
        # =====================================================================
        self._use_chunked_history = use_chunked_history
        self._history_manager = None
        
        if use_chunked_history and output_dir:
            # Use chunked storage for long simulations (constant ~100MB memory)
            from h2_plant.storage.history_manager import ChunkedHistoryManager
            
            self._history_manager = ChunkedHistoryManager(
                output_dir=output_dir,
                total_steps=total_steps,
                chunk_size=10_000,  # ~7 simulated days
                resume=resume
            )
            
            # Define ALL columns (Base + Extra) in one place for consistency
            full_columns = {
                'minute': np.int32,
                'P_offer': np.float64,
                'P_soec_actual': np.float64,
                'P_pem': np.float64,
                'P_sold': np.float64,
                'spot_price': np.float64,
                'h2_kg': np.float64,
                'H2_soec_kg': np.float64,
                'H2_pem_kg': np.float64,
                'H2_atr_kg': np.float64,
                'cumulative_h2_kg': np.float64,
                'steam_soec_kg': np.float64,
                'H2O_soec_out_kg': np.float64,
                'soec_active_modules': np.int32,
                'H2O_pem_kg': np.float64,
                'O2_pem_kg': np.float64,
                'pem_V_cell': np.float64,
                'pem_current_density': np.float64,
                'pem_efficiency': np.float64,
                'P_bop_mw': np.float64,
                'tank_level_kg': np.float64,
                'tank_pressure_bar': np.float64,
                'compressor_power_kw': np.float64,
                'sell_decision': np.int8,
                'PEM_o2_impurity_ppm_mol': np.float64,
                'storage_soc': np.float64,
                'storage_dsoc_per_h': np.float64,
                'storage_zone': np.int8,
                'storage_action_factor': np.float64,
                'storage_time_to_full_h': np.float64,
                'h2_rfnbo_kg': np.float64,
                'h2_non_rfnbo_kg': np.float64,
                'cumulative_h2_rfnbo_kg': np.float64,
                'cumulative_h2_non_rfnbo_kg': np.float64,
                'spot_purchased_mw': np.float64,
                'spot_threshold_eur_mwh': np.float64,
                'bop_grid_import_mw': np.float64,
                'bop_price_eur_mwh': np.float64,
                'bop_cost_eur': np.float64,
                'cumulative_bop_cost_eur': np.float64,
                'ppa_price_effective_eur_mwh': np.float64,
                'cooling_manager_glycol_supply_temp_c': np.float64,
                'cooling_manager_glycol_duty_kw': np.float64,
                'cooling_manager_cw_supply_temp_c': np.float64,
                'cooling_manager_cw_duty_kw': np.float64,
                'cooling_manager_tower_fan_power_kw': np.float64,
                'cooling_manager_glycol_fan_power_kw': np.float64,
                'cooling_manager_power_kw': np.float64,
                'integrated_global_efficiency': np.float64,
                'P_soec_grid_mw': np.float64,
                'P_pem_grid_mw': np.float64,
                'P_bop_grid_usage_mw': np.float64,
                'sold_energy_mwh_step': np.float64,
                'pem_electricity_consumption_kwh_step': np.float64,
                'soec_electricity_consumption_kwh_step': np.float64,
                'bop_electricity_consumption_kwh_step': np.float64,
                'total_electric_load_mw': np.float64,
                'electricity_consumption_kwh_step': np.float64,
                'total_cooling_duty_kw': np.float64,
                'cooling_duty_kwh_th_step': np.float64,
                'biogas_feed_kg_step': np.float64,
                'water_makeup_kg_step': np.float64,
            }
            
            self._history_manager.register_columns(full_columns)
            
            # OPTIMIZATION: Use direct buffer access instead of Proxy
            self._history = self._history_manager.buffers
            
            logger.info(f"Using CHUNKED history storage: {total_steps} steps, "
                       f"chunk_size=10,000, output={output_dir}")
        else:
            # In-Memory Storage (legacy/short runs)
            self._history_manager = None
            
            # Use the same exact schema for in-memory
            full_columns = {
                'minute': np.int32,
                'P_offer': np.float64,
                'P_soec_actual': np.float64,
                'P_pem': np.float64,
                'P_sold': np.float64,
                'spot_price': np.float64,
                'h2_kg': np.float64,
                'H2_soec_kg': np.float64,
                'H2_pem_kg': np.float64,
                'H2_atr_kg': np.float64,
                'cumulative_h2_kg': np.float64,
                'steam_soec_kg': np.float64,
                'H2O_soec_out_kg': np.float64,
                'soec_active_modules': np.int32,
                'H2O_pem_kg': np.float64,
                'O2_pem_kg': np.float64,
                'pem_V_cell': np.float64,
                'pem_current_density': np.float64,
                'pem_efficiency': np.float64,
                'P_bop_mw': np.float64,
                'tank_level_kg': np.float64,
                'tank_pressure_bar': np.float64,
                'compressor_power_kw': np.float64,
                'sell_decision': np.int8,
                'PEM_o2_impurity_ppm_mol': np.float64,
                'storage_soc': np.float64,
                'storage_dsoc_per_h': np.float64,
                'storage_zone': np.int8,
                'storage_action_factor': np.float64,
                'storage_time_to_full_h': np.float64,
                'h2_rfnbo_kg': np.float64,
                'h2_non_rfnbo_kg': np.float64,
                'cumulative_h2_rfnbo_kg': np.float64,
                'cumulative_h2_non_rfnbo_kg': np.float64,
                'spot_purchased_mw': np.float64,
                'spot_threshold_eur_mwh': np.float64,
                'bop_grid_import_mw': np.float64,
                'bop_price_eur_mwh': np.float64,
                'bop_cost_eur': np.float64,
                'cumulative_bop_cost_eur': np.float64,
                'ppa_price_effective_eur_mwh': np.float64,
                'cooling_manager_glycol_supply_temp_c': np.float64,
                'cooling_manager_glycol_duty_kw': np.float64,
                'cooling_manager_cw_supply_temp_c': np.float64,
                'cooling_manager_cw_duty_kw': np.float64,
                'cooling_manager_tower_fan_power_kw': np.float64,
                'cooling_manager_glycol_fan_power_kw': np.float64,
                'cooling_manager_power_kw': np.float64,
                'integrated_global_efficiency': np.float64,
                'P_soec_grid_mw': np.float64,
                'P_pem_grid_mw': np.float64,
                'P_bop_grid_usage_mw': np.float64,
                'sold_energy_mwh_step': np.float64,
                'pem_electricity_consumption_kwh_step': np.float64,
                'soec_electricity_consumption_kwh_step': np.float64,
                'bop_electricity_consumption_kwh_step': np.float64,
                'total_electric_load_mw': np.float64,
                'electricity_consumption_kwh_step': np.float64,
                'total_cooling_duty_kw': np.float64,
                'cooling_duty_kwh_th_step': np.float64,
                'biogas_feed_kg_step': np.float64,
                'water_makeup_kg_step': np.float64,
            }
            # Allocate arrays
            self._history = {k: np.zeros(total_steps, dtype=dt) for k, dt in full_columns.items()}
        
        # Accumulators for cumulative metrics (needed for chunked continuity)
        self._accum_h2_kg = 0.0
        self._accum_h2_rfnbo = 0.0
        self._accum_h2_non_rfnbo = 0.0
        self._accum_spot_purchased = 0.0

        # 2. Identify Components & Pre-Bind Arrays
        self._recorders = []
        self._prebind_recorders(registry, total_steps)

        def _resolve_mass_flow_key(default_key: str, token: str) -> Optional[str]:
            if default_key in self._history:
                return default_key
            for key in self._history.keys():
                if key.endswith("_outlet_mass_flow_kg_h") and token in key.lower():
                    return key
            return None

        self._biogas_flow_key = _resolve_mass_flow_key(
            "Biogas_Source_outlet_mass_flow_kg_h",
            "biogas_source",
        )
        self._water_flow_key = _resolve_mass_flow_key(
            "Water_Source_outlet_mass_flow_kg_h",
            "water_source",
        )
        
        # SOEC Specific Modules - Per-Module Recording
        if self._soec:
            num_modules = getattr(self._soec, 'num_modules', 0)
            for i in range(num_modules):
                self._history[f"soec_module_powers_{i+1}"] = np.zeros(total_steps, dtype=np.float64)
                # NEW: Per-module degradation tracking
                self._history[f"soec_module_hours_{i+1}"] = np.zeros(total_steps, dtype=np.float64)
                self._history[f"soec_module_eff_{i+1}"] = np.zeros(total_steps, dtype=np.float64)


        # 3. Storage Controller Setup (APC)
        self._setup_storage_controller(registry)
        
        # Record storage capacity to history for visualization (constant value)
        if self._storage_total_capacity_kg > 0:
            self._history['storage_capacity_kg'] = np.full(total_steps, self._storage_total_capacity_kg, dtype=np.float64)

        # 4. OPTIMIZATION: Pre-cache BOP power consumers with resolved getter
        self._bop_power_getters = []  # List of (component, getter_func)
        for cid, comp in registry.list_components():
            if hasattr(comp, 'power_kw'):
                self._bop_power_getters.append(lambda c=comp: c.power_kw)
            elif hasattr(comp, 'electrical_power_kw'):
                self._bop_power_getters.append(lambda c=comp: c.electrical_power_kw)
            elif hasattr(comp, 'fan_power_kw'):
                self._bop_power_getters.append(lambda c=comp: c.fan_power_kw)
            elif hasattr(comp, 'current_power_w'):
                self._bop_power_getters.append(lambda c=comp: c.current_power_w / 1000.0)
        logger.info(f"Pre-cached {len(self._bop_power_getters)} BOP power consumers")

        # 5. Pre-resolve compressors and tanks (avoids hasattr check every step)
        self._compressors = [comp for _, comp in registry.list_components() if isinstance(comp, CompressorSingle)]
        from h2_plant.components.storage.h2_tank import TankArray
        self._tanks = [comp for _, comp in registry.list_components() if isinstance(comp, TankArray)]

        # 6. Pre-resolve SOEC attribute getters
        if self._soec:
            self._soec_has_real_powers = hasattr(self._soec, 'real_powers')
            if hasattr(self._soec, 'last_step_h2_kg'):
                self._get_soec_h2 = lambda: self._soec.last_step_h2_kg
            elif hasattr(self._soec, 'last_h2_output_kg'):
                self._get_soec_h2 = lambda: self._soec.last_h2_output_kg
            elif hasattr(self._soec, 'h2_output_kg'):
                self._get_soec_h2 = lambda: self._soec.h2_output_kg
            else:
                self._get_soec_h2 = lambda: 0.0
            self._soec_cid = getattr(self._soec, 'component_id', 'SOEC_Cluster')
        else:
            self._soec_has_real_powers = False
            self._get_soec_h2 = lambda: 0.0
            self._soec_cid = None

        # 7. Pre-resolve cooling manager
        self._cooling_manager = registry.get('cooling_manager') if registry and registry.has('cooling_manager') else None

        # 8. Pre-resolve economics config
        self._bop_pricing_mode = getattr(context.economics, 'bop_pricing_mode', 'fixed')
        self._bop_fixed_price = getattr(context.economics, 'bop_fixed_price_eur_mwh', 80.0)

        # 9. Initialize accumulators
        self._accum_bop_cost = 0.0

        self._state = IntegratedDispatchState()
        logger.info(f"Initialized HybridArbitrageEngineStrategy with {total_steps} steps and Storage APC")

    def _prebind_recorders(self, registry: 'ComponentRegistry', total_steps: int) -> None:
        """
        Scan registry, allocate arrays, and bind them to StreamRecorder objects.
        This enables O(1) access during the simulation loop.
        """
        def _alloc_stream_history_with_prefix(prefix: str, steps: int) -> None:
            keys = [
                f"{prefix}_outlet_temp_c",
                f"{prefix}_outlet_pressure_bar",
                f"{prefix}_outlet_mass_flow_kg_h",
                f"{prefix}_outlet_h2o_frac",
                f"{prefix}_h2o_vapor_kg_h"
            ]
            species = ['H2', 'O2', 'N2', 'H2O', 'CH4', 'CO2']
            keys.extend([f"{prefix}_outlet_{sp}_molf" for sp in species])

            if self._use_chunked_history:
                for k in keys:
                    self._history_manager.register_column(k, np.float64)

            for k in keys:
                self._history[k] = np.zeros(steps, dtype=np.float64)

        # Mapping: Class Type -> (Stream Attribute Name, List of (Metric Name, Metric Attribute))
        CONFIG_MAP = {
            Chiller: ('outlet_stream', [('cooling_load_kw', 'cooling_load_kw'), ('electrical_power_kw', 'electrical_power_kw'), ('latent_heat_kw', 'latent_heat_kw'), ('sensible_heat_kw', 'sensible_heat_kw')]),
            Coalescer: ('output_stream', [('delta_p_bar', 'delta_p_bar'), ('drain_flow_kg_h', 'drain_flow_kg_h')]),
            DeoxoReactor: ('output_stream', [('outlet_o2_ppm_mol', 'outlet_o2_ppm_mol'), ('peak_temp_c', 'peak_temp_c'), ('inlet_temp_c', 'inlet_temp_c'), ('o2_in_kg_h', 'o2_in_kg_h'), ('mass_flow_kg_h', 'mass_flow_kg_h')]),
            PSA: ('product_outlet', [('outlet_o2_ppm_mol', 'outlet_o2_ppm_mol')]), 
            SyngasPSA: ('product_outlet', []),  # ATR Syngas PSA 
            KnockOutDrum: ('_gas_outlet_stream', [('water_removed_kg_h', 'water_removed_kg_h'), ('m_dot_H2O_liq_accomp_kg_s', 'm_dot_H2O_liq_accomp_kg_s')]),
            HydrogenMultiCyclone: ('_outlet_stream', [('pressure_drop_mbar', 'pressure_drop_mbar')]),
            CompressorSingle: ('outlet', [('power_kw', 'power_kw')]),
            DryCooler: ('outlet_stream', [('heat_rejected_kw', 'dc_duty_kw'), ('fan_power_kw', 'fan_power_kw'), ('latent_heat_kw', 'latent_heat_kw')]),
            ElectricBoiler: ('_output_stream', [('power_input_kw', 'power_kw')]),
            Interchanger: ('hot_out', [('q_transferred_kw', 'q_transferred_kw')]),
            DetailedTankArray: ('h2_out', [('inventory_kg', 'total_mass_kg'), ('avg_pressure_bar', 'avg_pressure_bar')]),
            UltraPureWaterTank: ('consumer_out', [('mass_kg', 'mass_kg'), ('control_zone_int', 'control_zone_int')]),
            DrainRecorderMixer: ('outlet_stream', [('outlet_mass_flow_kg_h', 'outlet_mass_flow_kg_h'), ('dissolved_gas_ppm', 'dissolved_gas_ppm')]),
            WaterPumpThermodynamic: ('water_out', [('power_kw', 'power_kw')]),
            IntegratedATRPlant: ('syngas_out', [
                ('atr_efficiency_chemical', 'atr_efficiency_chemical'), 
                ('atr_efficiency_global', 'atr_efficiency_global'), 
                ('atr_q_useful_kw', 'q_useful_kw'),
                ('atr_q_useful_kw', 'q_useful_kw'),
                ('atr_heat_duty_kw', 'heat_duty_kw')
            ]),
            MultiComponentMixer: ('outlet_stream', [('temperature_k', 'temperature_k'), ('pressure_pa', 'pressure_pa')]),
            WaterMixer: ('outlet_stream', [('outlet_temperature_c', 'outlet_temperature_c'), ('outlet_mass_flow_kg_h', 'outlet_mass_flow_kg_h')]),
            BiogasSource: ('out', []),
            ExternalWaterSource: ('water_out', []),
            DischargeStation: ('h2_out', [
                ('truck_demand_kg_h', 'total_demand_signal_kg_h'),
                ('trucks_filled', 'trucks_filled_total'),
                ('truck_power_kw', 'power_consumption_kw')
            ])
        }

        # Also add SOEC Cluster if it has a stream
        soec = self._soec
        if soec:
             # Manually add SOEC - now properly binds a recorder to 'h2_out' port
             cid = soec.component_id if hasattr(soec, 'component_id') else 'SOEC_Cluster'
             self._alloc_stream_history(cid, total_steps)
             
             # Allocate extra metric array for O2 PPM (fetched from get_state)
             self._history[f"{cid}_outlet_o2_ppm_mol"] = np.zeros(total_steps, dtype=np.float64)
             
             # Create recorder bound to 'h2_out' port
             soec_recorder = StreamRecorder(
                 component=soec,
                 stream_attr='h2_out',  # This is the OUTPUT PORT name
                 temp_arr=self._history[f"{cid}_outlet_temp_c"],
                 press_arr=self._history[f"{cid}_outlet_pressure_bar"],
                 flow_arr=self._history[f"{cid}_outlet_mass_flow_kg_h"],
                 h2o_frac_arr=self._history[f"{cid}_outlet_h2o_frac"],
                 h2o_vapor_arr=self._history[f"{cid}_h2o_vapor_kg_h"],
                 mole_arrs=[
                     self._history[f"{cid}_outlet_H2_molf"],
                     self._history[f"{cid}_outlet_O2_molf"],
                     self._history[f"{cid}_outlet_N2_molf"],
                     self._history[f"{cid}_outlet_H2O_molf"],
                     self._history[f"{cid}_outlet_CH4_molf"],
                     self._history[f"{cid}_outlet_CO2_molf"]
                 ],
                 extra_metric_arrs=[['outlet_o2_ppm_mol', self._history[f"{cid}_outlet_o2_ppm_mol"]]],
                 
                 # Column names for re-binding
                 temp_col_name=f"{cid}_outlet_temp_c",
                 press_col_name=f"{cid}_outlet_pressure_bar",
                 flow_col_name=f"{cid}_outlet_mass_flow_kg_h",
                 h2o_frac_col_name=f"{cid}_outlet_h2o_frac",
                 mole_cols=[
                     ('H2', f"{cid}_outlet_H2_molf"),
                     ('O2', f"{cid}_outlet_O2_molf"),
                     ('N2', f"{cid}_outlet_N2_molf"),
                     ('H2O', f"{cid}_outlet_H2O_molf"),
                     ('CH4', f"{cid}_outlet_CH4_molf"),
                     ('CO2', f"{cid}_outlet_CO2_molf")
                 ],
                 extra_metric_cols=[('outlet_o2_ppm_mol', f"{cid}_outlet_o2_ppm_mol")]
             )

             soec_recorder.bind_accessor()
             self._recorders.append(soec_recorder)
        
        # Separate dict for 2D matrix arrays (not compatible with HistoryDictProxy)
        # These are always in-memory numpy arrays regardless of chunked mode
        self._matrix_history: Dict[str, np.ndarray] = {}
        self._detailed_tank_recorders = []  # List of (component, pressure_matrix, mass_matrix)

        for cid, comp in registry.list_components():
            ctype = type(comp)
            
            # Special Matrix Recording for DetailedTankArray -> Flattened for Chunked Compatibility
            if isinstance(comp, DetailedTankArray):
                n_tanks = comp.n_tanks
                # Allocate flattened columns: tank_0_p_bar, tank_0_m_kg, etc.
                p_keys = [f"{cid}_tank_{i}_p_bar" for i in range(n_tanks)]
                m_keys = [f"{cid}_tank_{i}_m_kg" for i in range(n_tanks)]
                
                # Register all columns if chunked
                if self._use_chunked_history:
                    for k in p_keys + m_keys:
                        self._history_manager.register_column(k, np.float64)
                        
                # Create buffer pointers (lists of arrays)
                p_arrs = []
                m_arrs = []
                
                for k in p_keys:
                    self._history[k] = np.zeros(total_steps, dtype=np.float64)
                    p_arrs.append(self._history[k])
                    
                for k in m_keys:
                    self._history[k] = np.zeros(total_steps, dtype=np.float64)
                    m_arrs.append(self._history[k])
                
                self._detailed_tank_recorders.append((
                    comp, 
                    p_keys, m_keys,
                    p_arrs, m_arrs
                ))
            
            # Special handling for inheritance or if exact type in map
            config = None
            for base_cls, conf in CONFIG_MAP.items():
                if isinstance(comp, base_cls):
                    config = conf
                    break
            
            if config:
                stream_attr, metrics = config
                
                # 1. Allocate Stream Arrays
                self._alloc_stream_history(cid, total_steps)
                
                # 2. Allocate Metric Arrays
                metric_recorders = []
                for hist_suffix, obj_attr in metrics:
                    hist_key = f"{cid}_{hist_suffix}"
                    
                    if self._use_chunked_history:
                        self._history_manager.register_column(hist_key, np.float64)
                    
                    # Safe to assign because buffers (self._history) points to current chunk
                    self._history[hist_key] = np.zeros(total_steps, dtype=np.float64)
                    metric_recorders.append([obj_attr, self._history[hist_key]])
                
                # 3. Create Recorder
                # Bind the arrays we just created
                recorder = StreamRecorder(
                    component=comp,
                    stream_attr=stream_attr,
                    temp_arr=self._history[f"{cid}_outlet_temp_c"],
                    press_arr=self._history[f"{cid}_outlet_pressure_bar"],
                    flow_arr=self._history[f"{cid}_outlet_mass_flow_kg_h"],
                    h2o_frac_arr=self._history[f"{cid}_outlet_h2o_frac"],
                    h2o_vapor_arr=self._history[f"{cid}_h2o_vapor_kg_h"],
                    mole_arrs=[
                        self._history[f"{cid}_outlet_H2_molf"],
                        self._history[f"{cid}_outlet_O2_molf"],
                        self._history[f"{cid}_outlet_N2_molf"],
                        self._history[f"{cid}_outlet_H2O_molf"],
                    self._history[f"{cid}_outlet_CH4_molf"],
                    self._history[f"{cid}_outlet_CO2_molf"]
                ],
                # Convert tuples to lists for mutability
                extra_metric_arrs=[[attr, arr] for attr, arr in metric_recorders],
                    
                    # Column names for re-binding
                    temp_col_name=f"{cid}_outlet_temp_c",
                    press_col_name=f"{cid}_outlet_pressure_bar",
                    flow_col_name=f"{cid}_outlet_mass_flow_kg_h",
                    h2o_frac_col_name=f"{cid}_outlet_h2o_frac",
                    mole_cols=[
                        ('H2', f"{cid}_outlet_H2_molf"),
                        ('O2', f"{cid}_outlet_O2_molf"),
                        ('N2', f"{cid}_outlet_N2_molf"),
                        ('H2O', f"{cid}_outlet_H2O_molf"),
                        ('CH4', f"{cid}_outlet_CH4_molf"),
                        ('CO2', f"{cid}_outlet_CO2_molf")
                    ],
                    extra_metric_cols=[(obj_attr, f"{cid}_{hist_suffix}") for hist_suffix, obj_attr in metrics]
                )

                recorder.bind_accessor()
                self._recorders.append(recorder)

                # Syngas PSA Tail Gas Recorder (separate stream)
                if isinstance(comp, SyngasPSA):
                    tail_prefix = f"{cid}_tail_gas"
                    _alloc_stream_history_with_prefix(tail_prefix, total_steps)

                    tail_recorder = StreamRecorder(
                        component=comp,
                        stream_attr='tail_gas_out',
                        temp_arr=self._history[f"{tail_prefix}_outlet_temp_c"],
                        press_arr=self._history[f"{tail_prefix}_outlet_pressure_bar"],
                        flow_arr=self._history[f"{tail_prefix}_outlet_mass_flow_kg_h"],
                        h2o_frac_arr=self._history[f"{tail_prefix}_outlet_h2o_frac"],
                        h2o_vapor_arr=self._history[f"{tail_prefix}_h2o_vapor_kg_h"],
                        mole_arrs=[
                            self._history[f"{tail_prefix}_outlet_H2_molf"],
                            self._history[f"{tail_prefix}_outlet_O2_molf"],
                            self._history[f"{tail_prefix}_outlet_N2_molf"],
                            self._history[f"{tail_prefix}_outlet_H2O_molf"],
                            self._history[f"{tail_prefix}_outlet_CH4_molf"],
                            self._history[f"{tail_prefix}_outlet_CO2_molf"]
                        ],
                        extra_metric_arrs=[],
                        temp_col_name=f"{tail_prefix}_outlet_temp_c",
                        press_col_name=f"{tail_prefix}_outlet_pressure_bar",
                        flow_col_name=f"{tail_prefix}_outlet_mass_flow_kg_h",
                        h2o_frac_col_name=f"{tail_prefix}_outlet_h2o_frac",
                        mole_cols=[
                            ('H2', f"{tail_prefix}_outlet_H2_molf"),
                            ('O2', f"{tail_prefix}_outlet_O2_molf"),
                            ('N2', f"{tail_prefix}_outlet_N2_molf"),
                            ('H2O', f"{tail_prefix}_outlet_H2O_molf"),
                            ('CH4', f"{tail_prefix}_outlet_CH4_molf"),
                            ('CO2', f"{tail_prefix}_outlet_CO2_molf")
                        ],
                        extra_metric_cols=[]
                    )

                    tail_recorder.bind_accessor()
                    self._recorders.append(tail_recorder)

    def _alloc_stream_history(self, cid: str, total_steps: int) -> None:
        """Allocate standard outlet stream history arrays."""
        keys = [
            f"{cid}_outlet_temp_c",
            f"{cid}_outlet_pressure_bar",
            f"{cid}_outlet_mass_flow_kg_h",
            f"{cid}_outlet_h2o_frac",
            f"{cid}_h2o_vapor_kg_h"
        ]
        species = ['H2', 'O2', 'N2', 'H2O', 'CH4', 'CO2']
        keys.extend([f"{cid}_outlet_{sp}_molf" for sp in species])

        if self._use_chunked_history:
            for k in keys:
                self._history_manager.register_column(k, np.float64)
        
        for k in keys:
            self._history[k] = np.zeros(total_steps, dtype=np.float64)

    # =========================================================================
    # STORAGE FEEDBACK CONTROL (APC)
    # =========================================================================

    def _setup_storage_controller(self, registry: 'ComponentRegistry') -> None:
        """Initialize storage references and control parameters."""
        self._storage_components = []
        
        # Scan registry for storage components
        for cid, comp in registry.list_components():
            logger.debug(f"Storage APC Scan: {cid} type={type(comp).__name__}")
            if isinstance(comp, (TankArray, H2StorageTankEnhanced, DetailedTankArray)):
                logger.info(f"Storage APC: Found storage component {cid} (type={type(comp).__name__})")
                self._storage_components.append(comp)

        # Calculate Total System Capacity (Max Mass in kg)
        total_cap = 0.0
        self._storage_info = []  # List of (component, max_capacity_kg)
        
        for comp in self._storage_components:
            cap = 0.0
            if isinstance(comp, TankArray):
                cap = comp.n_tanks * comp.capacity_kg
            elif isinstance(comp, H2StorageTankEnhanced):
                # Calculate max mass via Ideal Gas Law at max pressure: m = PV/RT
                try:
                    V = comp.volume_m3
                    P_max = comp.max_pressure_bar * 1e5
                    R = getattr(comp.accumulator, 'R', 4124.0)  # H2 specific gas constant
                    T = getattr(comp.accumulator, 'T', 298.15)
                    cap = (P_max * V) / (R * T)
                except Exception:
                    cap = 0.0
            elif isinstance(comp, DetailedTankArray):
                # Use tank's own Real Gas EOS (LUT-backed Helmholtz) for accurate capacity
                try:
                    max_mass_per_tank = comp._max_mass_at_pressure(
                        comp.max_pressure_pa, comp.volume_per_tank, comp.ambient_temp_k
                    )
                    cap = comp.n_tanks * max_mass_per_tank
                except Exception:
                    cap = 0.0
            
            if cap > 0:
                self._storage_info.append((comp, cap))
                total_cap += cap
        
        self._storage_total_capacity_kg = total_cap
        
        # Control Parameters (Tuning)
        self._ctrl_params = {
            'SOC_LOW': 0.60,         # < 60%: Normal Operation
            'SOC_HIGH': 0.80,        # 60-80%: Attention (Start linear reduction)
            'SOC_CRITICAL': 0.95,    # > 95%: Critical (Hard Stop)
            'HYSTERESIS': 0.02,      # 2% deadband to prevent chatter
            'MAX_RATE_H': 0.20,      # If filling > 20%/hour, trigger alert early
            'MIN_ACTION_FACTOR': 0.1 # Minimum turndown before shutdown (10%)
        }
        
        # Runtime State
        self._ctrl_state = {
            'prev_soc': 0.0,
            'current_zone': 0,  # 0: Normal, 1: Attention, 2: Alert, 3: Critical
            'time_to_full_h': 999.0
        }
        
        if self._storage_components:
            logger.info(f"Storage APC: Found {len(self._storage_components)} tanks, "
                       f"total capacity = {total_cap:.1f} kg")
        else:
            logger.warning("Storage APC: No storage components found in registry")
        
        # --- MPC: Locate Discharge Station for Demand Forecasting ---
        self._discharge_station = None
        self._ds_params = None
        for cid, comp in registry.list_components():
            if isinstance(comp, DischargeStation):
                self._discharge_station = comp
                logger.info(f"Storage MPC: Found DischargeStation {cid} for demand forecasting")
                break
        
        # Pre-calculate station params to avoid lookup in loop
        if self._discharge_station:
            ds = self._discharge_station
            self._ds_params = {
                'max_rate_kg_h': ds.max_fill_rate * 60.0 * ds.n_stations,
                'min_rate_kg_h': ds.min_fill_rate * 60.0 * ds.n_stations,
                'day_limit_hour': ds.h_in_day_max if ds.h_in_day_max else 24.0,
                'is_scheduled': ds.h_in_day_max is not None
            }
            logger.info(f"Storage MPC: Discharge params - max={self._ds_params['max_rate_kg_h']:.1f} kg/h, "
                       f"schedule={'YES' if self._ds_params['is_scheduled'] else 'NO'}")

    def _get_aggregate_soc(self) -> Tuple[float, float]:
        """
        Calculate Plant-Wide State of Charge (0.0 to 1.0) and current mass.

        Includes pipeline inventory (compressor buffers + tank pending input)
        as virtual mass so the APC triggers curtailment early enough to account
        for the 2+ timestep latency in the processing chain.
        """
        if self._storage_total_capacity_kg <= 0:
            return 0.0, 0.0

        current_mass = 0.0
        for comp, _ in self._storage_info:
            # Use unified interface or direct access
            if hasattr(comp, 'get_inventory_kg'):
                current_mass += comp.get_inventory_kg()
            elif hasattr(comp, 'get_total_mass'): # DetailedTankArray
                current_mass += comp.get_total_mass()
            elif hasattr(comp, 'masses'):  # TankArray direct
                current_mass += np.sum(comp.masses)
            elif hasattr(comp, 'mass_kg'):  # Enhanced direct
                current_mass += comp.mass_kg

        # Add pipeline inventory: H2 in-flight that will arrive at tank regardless
        pipeline_kg = 0.0
        for comp in self._compressors:
            pipeline_kg += getattr(comp, 'transfer_mass_kg', 0.0)
        for comp, _ in self._storage_info:
            pending_rate = getattr(comp, '_h2_in_rate', 0.0)
            if pending_rate > 0:
                dt_h = self._context.simulation.timestep_hours if self._context else 0.0167
                pipeline_kg += pending_rate * dt_h
        current_mass += pipeline_kg

        soc = current_mass / self._storage_total_capacity_kg
        return min(max(soc, 0.0), 1.0), current_mass

    def _determine_zone(self, soc: float, dsoc_dt: float = 0.0) -> int:
        """
        Determine control zone with hysteresis (Schmitt Trigger).
        Zones: 0 (Normal), 1 (Attention), 2 (Alert), 3 (Critical)
        
        IMPORTANT: When dsoc_dt < 0 (tank is draining because demand > production),
        the APC stays in Normal zone to avoid reducing production when it's needed.
        """
        p = self._ctrl_params
        current_zone = self._ctrl_state['current_zone']
        
        # CRITICAL FIX: If tank is draining (demand > production), stay in Normal zone
        # This prevents production reduction when the system is supply-limited
        if dsoc_dt < -0.001 and soc < p['SOC_HIGH']:  # Tank is draining at > 0.1%/hour AND not in high SOC
            return 0  # Normal zone - no production reduction
        
        # Thresholds
        z1_thresh = p['SOC_LOW']
        z2_thresh = p['SOC_HIGH']
        z3_thresh = p['SOC_CRITICAL']
        hyst = p['HYSTERESIS']

        new_zone = current_zone

        # Transition Logic (Upward is instant, Downward requires hysteresis)
        if soc >= z3_thresh:
            new_zone = 3
        elif soc >= z2_thresh:
            if current_zone == 3 and soc > (z3_thresh - hyst):
                new_zone = 3  # Stick to 3
            else:
                new_zone = 2
        elif soc >= z1_thresh:
            if current_zone == 2 and soc > (z2_thresh - hyst):
                new_zone = 2  # Stick to 2
            else:
                new_zone = 1
        else:
            if current_zone == 1 and soc > (z1_thresh - hyst):
                new_zone = 1  # Stick to 1
            else:
                new_zone = 0
            
        return new_zone

    def _calculate_action_factor(self, zone: int, soc: float, dsoc_dt: float) -> float:
        """
        Calculate power scaling factor (0.0 to 1.0).
        Includes derivative action for fast filling.
        """
        p = self._ctrl_params
        
        # 1. Base Factor based on Zone
        factor = 1.0
        
        if zone == 0:  # Normal
            factor = 1.0
        elif zone == 1:  # Attention (Linear reduction 1.0 -> 0.7)
            # Normalize soc within the zone
            norm = (soc - p['SOC_LOW']) / (p['SOC_HIGH'] - p['SOC_LOW'])
            factor = 1.0 - (0.3 * norm)
        elif zone == 2:  # Alert (Aggressive reduction 0.7 -> 0.0)
            norm = (soc - p['SOC_HIGH']) / (p['SOC_CRITICAL'] - p['SOC_HIGH'])
            factor = 0.7 * (1.0 - norm)
        elif zone == 3:  # Critical
            factor = 0.0

        # 2. Derivative Action (Fast fill protection)
        # If filling very fast, artificially reduce factor to slow down
        if dsoc_dt > p['MAX_RATE_H']:
            rate_penalty = (dsoc_dt - p['MAX_RATE_H']) * 2.0  # Tuning scalar
            factor = max(0.0, factor - rate_penalty)

        return max(0.0, min(1.0, factor))

    def decide_and_apply(self, t: float, prices: np.ndarray, wind: np.ndarray) -> None:
        """
        Make dispatch decision and apply setpoints.
        """
        dt = self._context.simulation.timestep_hours
        step_idx = self._state.step_idx

        if step_idx >= self._total_steps:
            return

        minute = int(round(t * 60))
        
        # Grid Firming: Ensure minimum guaranteed power
        # MODIFIED: Variable power is ADDED to guaranteed power (Base Load + Variable Peaking)
        guaranteed_mw = getattr(self._context.economics, 'guaranteed_power_mw', 0.0)
        wind_mw = wind[step_idx]
        
        # Old Logic: P_offer = max(wind_mw, guaranteed_mw)
        # New Logic: Guaranteed is constant base load, Wind is added on top
        P_offer = guaranteed_mw + wind_mw
        
        current_price = prices[step_idx]
        
        # Future offer also respects firming (Base + Forecasted Wind)
        wind_fut = wind[min(step_idx + 60, len(wind) - 1)]
        # Old Logic: P_future = max(wind_fut, guaranteed_mw)
        P_future = guaranteed_mw + wind_fut

        # =====================================================================
        # DUAL PPA PRICING: Weighted Average Calculation
        # =====================================================================
        # Contract block (up to guaranteed_mw): ppa_contract_price_eur_mwh
        # Variable excess (above guaranteed_mw): ppa_variable_price_eur_mwh
        price_contract = getattr(self._context.economics, 'ppa_contract_price_eur_mwh', 80.0)
        price_variable = getattr(self._context.economics, 'ppa_variable_price_eur_mwh', 55.0)
        
        if P_offer <= 1e-6:
            current_ppa_price = price_contract
        elif P_offer <= guaranteed_mw:
            # Entire power is within the guaranteed contract block
            current_ppa_price = price_contract
        else:
            # Power exceeds guaranteed block: Blend the prices
            # Effective = (Contract_MW × Contract_Price + Excess_MW × Variable_Price) / Total_MW
            # With New Logic: Excess_MW is exactly equal to the Wind_MW component
            excess_mw = P_offer - guaranteed_mw
            total_cost_per_hour = (guaranteed_mw * price_contract) + (excess_mw * price_variable)
            current_ppa_price = total_cost_per_hour / P_offer

        soec_kwh_kg = getattr(self._context.physics.soec_cluster, 'kwh_per_kg', 37.5)
        pem_kwh_kg = getattr(self._context.physics.pem_system, 'kwh_per_kg', 50.0)

        d_input = DispatchInput(
            minute=minute,
            P_offer=P_offer,
            P_future_offer=P_future,
            current_price=current_price,
            soec_capacity_mw=self._soec_capacity,
            pem_max_power_mw=self._pem_max,
            soec_h2_kwh_kg=soec_kwh_kg,
            pem_h2_kwh_kg=pem_kwh_kg,
            ppa_price_eur_mwh=current_ppa_price,  # Use calculated weighted average
            h2_price_eur_kg=getattr(self._context.economics, 'h2_price_eur_kg', 9.6),
            arbitrage_threshold_eur_mwh=getattr(self._context.economics, 'arbitrage_threshold_eur_mwh', None),
            # RFNBO / Economic Spot parameters
            h2_non_rfnbo_price_eur_kg=getattr(self._context.economics, 'h2_non_rfnbo_price_eur_kg', 2.0),
            p_grid_max_mw=getattr(self._context.economics, 'p_grid_max_mw', 30.0)
        )

        d_state = DispatchState(
            P_soec_prev=self._state.P_soec_prev,
            force_sell=self._state.force_sell
        )

        result = self._inner_strategy.decide(d_input, d_state)
        self._state.force_sell = result.state_update.get('force_sell', False)

        # =====================================================================
        # STORAGE FEEDBACK CONTROL (APC) - Closed Loop
        # =====================================================================
        
        # A. Calculate System State
        soc, current_mass = self._get_aggregate_soc()
        prev_soc = self._ctrl_state['prev_soc']
        
        # B. Calculate Derivative (dSOC/dt in 1/hour)
        dsoc_dt = (soc - prev_soc) / dt if dt > 0 else 0.0
        
        # C. Calculate Time to Full (hours)
        # Only meaningful when tank is actually filling (dsoc_dt > 0)
        if dsoc_dt > 0.001:
            self._ctrl_state['time_to_full_h'] = (1.0 - soc) / dsoc_dt
        else:
            # Tank is draining or stable - no overflow risk
            self._ctrl_state['time_to_full_h'] = 999.0

        # D. Storage Control Mode Selection
        # =====================================================================
        # Read control mode from config (defaults to SCHMITT_TRIGGER)
        storage_control_mode = getattr(self._context.simulation, 'storage_control_mode', 'SCHMITT_TRIGGER')
        if storage_control_mode is None:
            storage_control_mode = 'SCHMITT_TRIGGER'
        storage_control_mode = storage_control_mode.upper()
        
        if storage_control_mode == 'MPC':
            # -----------------------------------------------------------------
            # MPC-based Action Factor Calculation (Predictive)
            # -----------------------------------------------------------------
            # Build production forecast from wind profile (next ~60 min)
            HORIZON = 60  # minutes
            remaining_steps = len(wind) - step_idx
            
            if remaining_steps >= HORIZON:
                wind_forecast = wind[step_idx : step_idx + HORIZON]
            else:
                # Pad end with last value
                wind_slice = wind[step_idx:]
                if len(wind_slice) > 0:
                    pad = np.full(HORIZON - len(wind_slice), wind_slice[-1])
                    wind_forecast = np.concatenate([wind_slice, pad])
                else:
                    wind_forecast = np.zeros(HORIZON)
            
            # Convert MW Wind -> kg/h H2 Potential
            # Safe assumption: best-case efficiency 37.5 kWh/kg (conservative for overflow prevention)
            prod_kwh_kg = 37.5
            production_forecast_kg_h = (wind_forecast * 1000.0) / prod_kwh_kg
            
            # Build demand forecast from DischargeStation schedule
            demand_forecast_kg_h = np.zeros(HORIZON, dtype=np.float64)
            
            if self._ds_params and self._ds_params['is_scheduled']:
                # Vectorized schedule: hour-of-day based
                time_offsets = np.arange(HORIZON, dtype=np.float64) / 60.0
                future_times = t + time_offsets
                hours_of_day = future_times % 24.0
                
                limit = self._ds_params['day_limit_hour']
                max_r = self._ds_params['max_rate_kg_h']
                min_r = self._ds_params['min_rate_kg_h']
                
                demand_forecast_kg_h = np.where(hours_of_day < limit, max_r, min_r)
            elif self._ds_params:
                # Stochastic mode: assume average 30% utilization
                avg_rate = self._ds_params['max_rate_kg_h'] * 0.3
                demand_forecast_kg_h[:] = avg_rate
            
            # Execute MPC Solver
            action_factor = calculate_storage_mpc_factor(
                current_soc=soc,
                total_capacity_kg=self._storage_total_capacity_kg,
                production_profile_kg_h=production_forecast_kg_h.astype(np.float64),
                demand_profile_kg_h=demand_forecast_kg_h,
                dt_hours=dt,
                soc_limit_high=0.95,
                horizon_steps=HORIZON
            )
            
            # Fallback: Determine zone for critical safety (hard stop at 95%)
            zone = self._determine_zone(soc, dsoc_dt)
        else:
            # -----------------------------------------------------------------
            # SCHMITT_TRIGGER (Default) - Reactive Zone-Based Control
            # -----------------------------------------------------------------
            zone = self._determine_zone(soc, dsoc_dt)
            action_factor = self._calculate_action_factor(zone, soc, dsoc_dt)
        
        # Update control state
        self._ctrl_state['current_zone'] = zone
        self._ctrl_state['prev_soc'] = soc

        # E. Modulate Power (Apply action factor to reduce production)
        P_soec_final = result.P_soec * action_factor
        P_pem_final = result.P_pem * action_factor

        # F. Pipeline Latency Safety Clamp
        # remaining_kg already accounts for pipeline inventory (included in current_mass
        # via _get_aggregate_soc). Check if current-step production would overshoot.
        remaining_kg = self._storage_total_capacity_kg - current_mass
        est_production_kg = (P_soec_final / 37.5 + P_pem_final / 50.0) * dt * 1000.0
        if remaining_kg < est_production_kg * 1.5:
            P_soec_final = 0.0
            P_pem_final = 0.0
            zone = 3

        # If in Critical Zone (3), force_sell to True (Safety Sell)
        if zone == 3:
            self._state.force_sell = True

        # =====================================================================
        # APPLY FINAL SETPOINTS (After APC modulation)
        # Gross-up power to account for transformer losses: P_grid = P_stack / η
        # =====================================================================

        if self._soec:
            # Calculate grid draw accounting for transformer losses
            P_soec_grid = P_soec_final / self._η_soec_trafo if self._η_soec_trafo > 0 else P_soec_final
            
            # Send power to transformer (or directly to SOEC if no transformer)
            if self._soec_trafo:
                self._soec_trafo.receive_input('power_in', P_soec_grid, 'electricity')
                # SOEC receives P_soec_final after transformer step (η * P_grid)
            
            # Still send power command to SOEC for compatibility
            self._soec.receive_input('power_in', P_soec_final, 'electricity')

        if self._pem:
            # Calculate grid draw accounting for transformer losses
            P_pem_grid = P_pem_final / self._η_pem_trafo if self._η_pem_trafo > 0 else P_pem_final
            
            if self._pem_trafo:
                self._pem_trafo.receive_input('power_in', P_pem_grid, 'electricity')
            
            # NOTE: Water supply is handled by the topology (PEM_Water_Pump).
            # DO NOT add hardcoded water here - it causes double delivery!
            self._pem.set_power_input_mw(P_pem_final)

        if self._use_chunked_history:
            history_store = self._history_manager.buffers
            local_idx = step_idx % self._history_manager.chunk_size
            
            # CRITICAL: Re-bind optimized recorders at the start of each new chunk
            # because the underlying arrays in history_store are re-allocated.
            if local_idx == 0:
                self._rebind_recorders(history_store)
        else:
            history_store = self._history
            local_idx = step_idx

        # Record dispatch data
        history_store['minute'][local_idx] = minute
        history_store['P_offer'][local_idx] = P_offer

        # Record Storage APC data
        history_store['storage_soc'][local_idx] = soc
        history_store['storage_dsoc_per_h'][local_idx] = dsoc_dt
        history_store['storage_zone'][local_idx] = zone
        history_store['storage_action_factor'][local_idx] = action_factor
        history_store['storage_time_to_full_h'][local_idx] = self._ctrl_state['time_to_full_h']
        history_store['spot_price'][local_idx] = current_price
        history_store['ppa_price_effective_eur_mwh'][local_idx] = current_ppa_price
        
        # Record RFNBO classification metrics
        if 'h2_rfnbo_kg' in result.state_update:
            spot_purchased = result.state_update.get('spot_purchased_mw', 0.0)
            spot_threshold = result.state_update.get('spot_threshold_eur_mwh', 0.0)
            h2_non_rfnbo = result.state_update.get('h2_non_rfnbo_kg', 0.0)
        else:
            spot_purchased = 0.0
            spot_threshold = 0.0
            h2_non_rfnbo = 0.0
        
        # Initialize with 0 - actual values are set in record_post_step using actual H2 production
        history_store['h2_rfnbo_kg'][local_idx] = 0.0
        history_store['h2_non_rfnbo_kg'][local_idx] = h2_non_rfnbo
        history_store['spot_purchased_mw'][local_idx] = spot_purchased
        history_store['spot_threshold_eur_mwh'][local_idx] = spot_threshold
        
        # Cumulative RFNBO - Initialize with current accumulator value
        # (will be updated in record_post_step after production)
        history_store['cumulative_h2_rfnbo_kg'][local_idx] = self._accum_h2_rfnbo
        
        # Update and record non-RFNBO immediately (as it comes from grid decision, not physics)
        self._accum_h2_non_rfnbo += h2_non_rfnbo
        history_store['cumulative_h2_non_rfnbo_kg'][local_idx] = self._accum_h2_non_rfnbo
        
        self._state.step_idx = step_idx
        
        # ... (rest of the method continues)

    def _rebind_recorders(self, history_store: Dict[str, np.ndarray]) -> None:
        """
        Update optimized recorders to point to the current chunk's arrays.
        Called when a new chunk is allocated.
        """
        # 1. Stream Recorders
        for rec in self._recorders:
             # Re-bind main arrays
             if rec.temp_col_name in history_store: rec.temp_arr = history_store[rec.temp_col_name]
             if rec.press_col_name in history_store: rec.press_arr = history_store[rec.press_col_name]
             if rec.flow_col_name in history_store: rec.flow_arr = history_store[rec.flow_col_name]
             if rec.h2o_frac_col_name in history_store: rec.h2o_frac_arr = history_store[rec.h2o_frac_col_name]
             
             # Re-bind mole fraction arrays
             for i, (species, col_name) in enumerate(rec.mole_cols):
                 if col_name in history_store:
                     rec.mole_arrs[i] = history_store[col_name]
            
             # Re-bind extra metrics
             for i, (attr, col_name) in enumerate(rec.extra_metric_cols):
                 if col_name in history_store:
                     rec.extra_metric_arrs[i][1] = history_store[col_name]

        # 2. Detailed Tank Recorders (Flattened Strategy)
        for comp, p_keys, m_keys, p_arrs, m_arrs in self._detailed_tank_recorders:
            for i, key in enumerate(p_keys):
                if key in history_store:
                    p_arrs[i] = history_store[key]
            
            for i, key in enumerate(m_keys):
                if key in history_store:
                    m_arrs[i] = history_store[key]

        


    def _get_buffer(self, name: str, history_store: Dict[str, np.ndarray]) -> np.ndarray:
        """Helper to get or register a buffer array."""
        if name not in history_store:
            if self._use_chunked_history:
                 self._history_manager.register_column(name)
            else:
                 history_store[name] = np.zeros(self._total_steps)
        return history_store[name]

    def record_post_step(self) -> None:
        """
        Record component outputs.
        OPTIMIZED: Uses local buffering and accumulator variables to minimize overhead.
        """
        step_idx = self._state.step_idx
        if step_idx >= self._total_steps:
            return

        # Resolve Storage and Index
        if self._use_chunked_history:
            history_store = self._history_manager.buffers
            local_idx = step_idx % self._history_manager.chunk_size
        else:
            history_store = self._history
            local_idx = step_idx

        # 1. Specialized Recording (SOEC/PEM Main metrics)

        # SOEC Logic
        P_soec_actual = 0.0
        h2_soec = 0.0
        steam_soec = 0.0
        if self._soec:
            if self._soec_has_real_powers:
                P_soec_actual = float(np.sum(self._soec.real_powers))
            h2_soec = self._get_soec_h2()
            steam_soec = getattr(self._soec, 'last_step_steam_input_kg', 0.0)

            history_store['H2O_soec_out_kg'][local_idx] = getattr(self._soec, 'last_water_output_kg', 0.0)

            # Dynamic SOEC Stream
            cid = self._soec_cid
            if cid:
                try:
                    out_stream = self._soec.get_output('h2_out')
                    if out_stream and out_stream.mass_flow_kg_h > 1e-6:
                         self._get_buffer(f"{cid}_outlet_mass_flow_kg_h", history_store)[local_idx] = out_stream.mass_flow_kg_h
                         self._get_buffer(f"{cid}_outlet_temp_c", history_store)[local_idx] = out_stream.temperature_k - 273.15
                         self._get_buffer(f"{cid}_outlet_pressure_bar", history_store)[local_idx] = out_stream.pressure_pa / 1e5
                         self._get_buffer(f"{cid}_outlet_h2o_frac", history_store)[local_idx] = out_stream.composition.get('H2O', 0.0)
                except Exception:
                    pass

        self._state.P_soec_prev = P_soec_actual

        # PEM Logic
        h2_pem = 0.0
        P_pem_actual = 0.0
        if self._pem:
            h2_pem = getattr(self._pem, 'h2_output_kg', 0.0)
            if hasattr(self._pem, 'P_consumed_W'): P_pem_actual = self._pem.P_consumed_W / 1e6
            
            history_store['PEM_o2_impurity_ppm_mol'][local_idx] = getattr(self._pem, 'o2_impurity_ppm_mol', 0.0)
            history_store['H2O_pem_kg'][local_idx] = getattr(self._pem, 'water_consumption_kg', 0.0)
            history_store['O2_pem_kg'][local_idx] = getattr(self._pem, 'o2_output_kg', 0.0)
            history_store['pem_V_cell'][local_idx] = getattr(self._pem, 'V_cell', 0.0)

        # ATR Logic
        h2_atr = 0.0
        if self._atr:
            prod_kmol_h = getattr(self._atr, 'h2_production_kmol_h', 0.0)
            dt = self._context.simulation.timestep_hours
            h2_atr = prod_kmol_h * 2.016 * dt

        # Global Power and Component Logic
        P_bop_kw = 0.0

        # Compressor Total Power (pre-resolved in initialize)
        total_comp_power = 0.0
        for comp in self._compressors:
            total_comp_power += comp.power_kw
        if total_comp_power < 0.0:
            total_comp_power = 0.0
        history_store['compressor_power_kw'][local_idx] = total_comp_power

        # Tank Levels
        total_tank_mass = 0.0
        avg_tank_pressure = 0.0
        if self._tanks:
            for tank in self._tanks:
                total_tank_mass += tank.get_total_mass()
            if len(self._tanks[0].pressures) > 0:
                avg_tank_pressure = np.mean(self._tanks[0].pressures) / 1e5
        history_store['tank_level_kg'][local_idx] = total_tank_mass
        history_store['tank_pressure_bar'][local_idx] = avg_tank_pressure

        # BOP Calculation - pre-resolved getters from initialize()
        for getter in self._bop_power_getters:
            P_bop_kw += getter()

        P_bop_mw = P_bop_kw / 1000.0
        
        P_consumed_from_wind = P_soec_actual + P_pem_actual
        P_offer = history_store['P_offer'][local_idx]
        P_sold_corrected = max(0.0, P_offer - P_consumed_from_wind)
        
        # BOP Grid Import Cost (pre-resolved pricing config from initialize)
        dt = self._context.simulation.timestep_hours
        spot_price = history_store['spot_price'][local_idx]
        bop_price = spot_price if self._bop_pricing_mode == 'spot' else self._bop_fixed_price
        
        P_soec_grid_mw = P_soec_actual / self._η_soec_trafo if self._η_soec_trafo > 0 else P_soec_actual
        P_pem_grid_mw = P_pem_actual / self._η_pem_trafo if self._η_pem_trafo > 0 else P_pem_actual
        P_bop_grid_mw = P_bop_mw / self._η_bop_trafo if self._η_bop_trafo > 0 else P_bop_mw
        
        bop_cost_eur = P_bop_grid_mw * dt * bop_price
        
        total_h2 = h2_soec + h2_pem + h2_atr
        self._state.cumulative_h2_kg += total_h2 # State variable handles its own accumulation across steps? 
        # Wait, self._state might be transient? No, StateManager persists it.

        # Fast Array Writes
        history_store['P_soec_actual'][local_idx] = P_soec_actual
        history_store['P_pem'][local_idx] = P_pem_actual
        history_store['P_sold'][local_idx] = P_sold_corrected
        
        history_store['P_soec_grid_mw'][local_idx] = P_soec_grid_mw
        history_store['P_pem_grid_mw'][local_idx] = P_pem_grid_mw
        history_store['P_bop_grid_usage_mw'][local_idx] = P_bop_grid_mw
        history_store['sold_energy_mwh_step'][local_idx] = P_sold_corrected * dt
        history_store['pem_electricity_consumption_kwh_step'][local_idx] = P_pem_grid_mw * 1000.0 * dt
        history_store['soec_electricity_consumption_kwh_step'][local_idx] = P_soec_grid_mw * 1000.0 * dt
        history_store['bop_electricity_consumption_kwh_step'][local_idx] = P_bop_grid_mw * 1000.0 * dt
        total_electric_load_mw = max(0.0, P_soec_grid_mw + P_pem_grid_mw + P_bop_grid_mw)
        history_store['total_electric_load_mw'][local_idx] = total_electric_load_mw
        history_store['electricity_consumption_kwh_step'][local_idx] = total_electric_load_mw * 1000.0 * dt
        history_store['h2_kg'][local_idx] = total_h2
        history_store['H2_soec_kg'][local_idx] = h2_soec
        history_store['H2_pem_kg'][local_idx] = h2_pem
        history_store['H2_atr_kg'][local_idx] = h2_atr
        
        # Accumulator Updates (Replaces reading [step-1])
        history_store['cumulative_h2_kg'][local_idx] = self._state.cumulative_h2_kg
        
        h2_rfnbo_actual = h2_soec + h2_pem + h2_atr
        history_store['h2_rfnbo_kg'][local_idx] = h2_rfnbo_actual
        
        self._accum_h2_rfnbo += h2_rfnbo_actual
        history_store['cumulative_h2_rfnbo_kg'][local_idx] = self._accum_h2_rfnbo
        
        history_store['steam_soec_kg'][local_idx] = steam_soec
        history_store['P_bop_mw'][local_idx] = P_bop_mw
        history_store['sell_decision'][local_idx] = 1 if P_sold_corrected > 0 else 0
        
        history_store['bop_grid_import_mw'][local_idx] = P_bop_grid_mw
        history_store['bop_price_eur_mwh'][local_idx] = bop_price
        history_store['bop_cost_eur'][local_idx] = bop_cost_eur
        
        self._accum_spot_purchased += bop_cost_eur # Reusing variable name appropriately? Wait, this is bop cost.
        # Check what _accum_spot_purchased was for. It was initialized but logic below uses a NEW cumulative metric for BOP.
        # Let's use a specialized one or reuse.
        
        # Cumulative BOP cost (_accum_bop_cost initialized in initialize())
        self._accum_bop_cost += bop_cost_eur
        history_store['cumulative_bop_cost_eur'][local_idx] = self._accum_bop_cost
        
        # SOEC Modules - Record Powers, Hours, and Efficiencies
        if self._soec and self._soec_has_real_powers:
            history_store['soec_active_modules'][local_idx] = int(np.sum(self._soec.real_powers > 0.01))
            if self._total_steps < 1000000:
                # Get per-module degradation metrics
                module_hours = getattr(self._soec, 'accumulated_hours', None)
                module_effs = getattr(self._soec, 'module_efficiencies', None)
                
                for i, power_mw in enumerate(self._soec.real_powers):
                    # Power
                    key_power = f"soec_module_powers_{i+1}"
                    self._get_buffer(key_power, history_store)[local_idx] = power_mw
                    
                    # Hours (accumulated operating hours)
                    if module_hours is not None and i < len(module_hours):
                        key_hours = f"soec_module_hours_{i+1}"
                        self._get_buffer(key_hours, history_store)[local_idx] = module_hours[i]
                    
                    # Efficiency (SEC kWh/kg)
                    if module_effs is not None and i < len(module_effs):
                        key_eff = f"soec_module_eff_{i+1}"
                        self._get_buffer(key_eff, history_store)[local_idx] = module_effs[i]


        # 2. CoolingManager (pre-resolved in initialize)
        cooling_manager = self._cooling_manager
        glycol_duty_kw = 0.0
        cw_duty_kw = 0.0
        if cooling_manager:
            history_store['cooling_manager_glycol_supply_temp_c'][local_idx] = getattr(cooling_manager, 'glycol_supply_temp_c', 0.0)
            glycol_duty_kw = getattr(cooling_manager, 'glycol_duty_kw', 0.0)
            history_store['cooling_manager_glycol_duty_kw'][local_idx] = glycol_duty_kw
            history_store['cooling_manager_cw_supply_temp_c'][local_idx] = getattr(cooling_manager, 'cw_supply_temp_c', 0.0)
            cw_duty_kw = getattr(cooling_manager, 'cw_duty_kw', 0.0)
            history_store['cooling_manager_cw_duty_kw'][local_idx] = cw_duty_kw
            tower_fan_power_kw = getattr(cooling_manager, 'tower_fan_power_kw', 0.0)
            glycol_fan_power_kw = getattr(cooling_manager, 'glycol_fan_power_kw', 0.0)
            total_cooling_power_kw = getattr(cooling_manager, 'power_kw', None)
            if total_cooling_power_kw is None:
                total_cooling_power_kw = tower_fan_power_kw + glycol_fan_power_kw
            history_store['cooling_manager_tower_fan_power_kw'][local_idx] = tower_fan_power_kw
            history_store['cooling_manager_glycol_fan_power_kw'][local_idx] = glycol_fan_power_kw
            history_store['cooling_manager_power_kw'][local_idx] = total_cooling_power_kw
        total_cooling_duty_kw = max(0.0, glycol_duty_kw + cw_duty_kw)
        history_store['total_cooling_duty_kw'][local_idx] = total_cooling_duty_kw
        history_store['cooling_duty_kwh_th_step'][local_idx] = total_cooling_duty_kw * dt

        # 3. Optimized Component Recording Loop
        for rec in self._recorders:
            stream = rec.stream_getter()
            if stream is not None:
                rec.temp_arr[local_idx] = stream.temperature_k - 273.15
                rec.press_arr[local_idx] = stream.pressure_pa / 1e5
                rec.flow_arr[local_idx] = stream.mass_flow_kg_h
                rec.h2o_frac_arr[local_idx] = stream.composition.get('H2O', 0.0) + stream.composition.get('H2O_liq', 0.0)
                
                # OPTIMIZATION: Skip expensive composition processing for zero-flow streams
                if stream.mass_flow_kg_h <= 1e-9:
                    # No flow = no meaningful composition, write zeros directly
                    rec.mole_arrs[0][local_idx] = 0.0
                    rec.mole_arrs[1][local_idx] = 0.0
                    rec.mole_arrs[2][local_idx] = 0.0
                    rec.mole_arrs[3][local_idx] = 0.0
                    rec.mole_arrs[4][local_idx] = 0.0
                    rec.mole_arrs[5][local_idx] = 0.0
                elif stream.extra and stream.extra.get('m_dot_H2O_liq_accomp_kg_s', 0.0) > 0:
                    # Complex path: Entrained liquid requires full calculation
                    rec.mole_arrs[0][local_idx] = stream.get_total_mole_frac('H2')
                    rec.mole_arrs[1][local_idx] = stream.get_total_mole_frac('O2')
                    rec.mole_arrs[2][local_idx] = stream.get_total_mole_frac('N2')
                    rec.mole_arrs[3][local_idx] = stream.get_total_mole_frac('H2O')
                    rec.mole_arrs[4][local_idx] = stream.get_total_mole_frac('CH4')
                    rec.mole_arrs[5][local_idx] = stream.get_total_mole_frac('CO2')
                else:
                    # Fast Path: Direct copy from cached array (mole_arr order matches SPECIES_INDICES)
                    # Indices: H2=0, O2=1, N2=2, H2O=3, CH4=4, CO2=5
                    mass_arr, mole_arr, M_mix, _ = stream.get_composition_arrays()
                    rec.mole_arrs[0][local_idx] = mole_arr[0]
                    rec.mole_arrs[1][local_idx] = mole_arr[1]
                    rec.mole_arrs[2][local_idx] = mole_arr[2]
                    rec.mole_arrs[3][local_idx] = mole_arr[3]
                    rec.mole_arrs[4][local_idx] = mole_arr[4]
                    rec.mole_arrs[5][local_idx] = mole_arr[5]
            
            for attr_name, metric_arr in rec.extra_metric_arrs:
                val = getattr(rec.component, attr_name, None)
                if val is None:
                    state = rec.component.get_state() if hasattr(rec.component, 'get_state') else {}
                    val = state.get(attr_name, 0.0)
                metric_arr[local_idx] = val if val is not None else 0.0

        # Canonical feed quantities (kg/step), using recorder outputs when available.
        biogas_flow_kg_h = 0.0
        water_flow_kg_h = 0.0

        if self._biogas_flow_key and self._biogas_flow_key in history_store:
            biogas_flow_kg_h = history_store[self._biogas_flow_key][local_idx]

        if self._water_flow_key and self._water_flow_key in history_store:
            water_flow_kg_h = history_store[self._water_flow_key][local_idx]

        history_store['biogas_feed_kg_step'][local_idx] = max(0.0, float(biogas_flow_kg_h) * dt)
        history_store['water_makeup_kg_step'][local_idx] = max(0.0, float(water_flow_kg_h) * dt)

        # 4. DetailedTankArray (Flattened)
        for comp, p_keys, m_keys, p_arrs, m_arrs in self._detailed_tank_recorders:
            for i, tank in enumerate(comp.tanks):
                 p_arrs[i][local_idx] = tank.pressure_pa / 1e5
                 m_arrs[i][local_idx] = tank.mass_kg

        h2_net_kg_h = 0.0
        has_net_h2_source = False

        mixer = self._safe_registry_get('H2_Production_Mixer')
        if mixer:
            has_net_h2_source = True
            out_stream = mixer.get_output('outlet_stream')
            if out_stream:
                h2_net_kg_h = out_stream.mass_flow_kg_h
        else:
            # Fallback: Sum PSA outputs
            for psa_id in ['PEM_H2_PSA_1', 'SOEC_H2_PSA_1', 'ATR_PSA_1']:
                psa = self._safe_registry_get(psa_id)
                if psa:
                    has_net_h2_source = True
                    stream = psa.get_output('product_outlet')
                    if stream:
                        h2_net_kg_h += stream.mass_flow_kg_h

        # If no mixer/PSA references exist in registry, fallback to gross-H2 efficiency
        # path by keeping h2_net_kg as None.
        h2_net_kg = (h2_net_kg_h * dt) if has_net_h2_source else None
            
        # 5. Integrated Efficiency (Inlined)
        self._calculate_integrated_efficiency_inline(history_store, local_idx, dt, h2_net_kg)

        self._state.step_idx += 1
        
        # 6. Chunked Storage: Trigger chunk flush if needed
        if self._use_chunked_history and self._history_manager:
            self._history_manager.step_complete(step_idx)

    def _calculate_integrated_efficiency_inline(self, history_store, idx, dt_hours, h2_net_kg=None):
            """
            Inlined calculation to avoid method call overhead and handle local indexing.
            """
            if dt_hours <= 0: return
            LHV_H2_KWH_KG = 33.33 
            
            # Use Net H2 (Purified) if available, otherwise fallback to Gross (History)
            if h2_net_kg is not None:
                mass_for_eff = h2_net_kg
            else:
                mass_for_eff = history_store['h2_kg'][idx] # Fallback to gross
                
            energy_h2_kw = (mass_for_eff / dt_hours) * LHV_H2_KWH_KG
            
            # Denominator: Grid Power for Stacks + BOP (which includes compression)
            # Note: compressor_power_kw is already inside P_bop_grid_usage_mw via P_bop_mw
            p_el_total_kw = (history_store['P_soec_grid_mw'][idx] + 
                            history_store['P_pem_grid_mw'][idx] + 
                            history_store['P_bop_grid_usage_mw'][idx]) * 1000.0
                            
            energy_biogas_kw = 0.0
            if self._atr:
                energy_biogas_kw = getattr(self._atr, 'biogas_energy_input_kw', 0.0)
            
            denom = p_el_total_kw + energy_biogas_kw
            eff = energy_h2_kw / denom if denom > 1e-6 else 0.0
            history_store['integrated_global_efficiency'][idx] = eff
       


    def get_history(self) -> Dict[str, np.ndarray]:
        """
        Get the recorded history.

        Returns dict of column -> array for in-memory mode.
        For chunked mode, finalizes and returns DataFrame-based dict.
        Results are cached after first call to avoid repeated disk I/O.
        """
        if self._cached_history is not None:
            return self._cached_history

        actual_steps = self._state.step_idx

        if self._use_chunked_history and self._history_manager:
            # Finalize chunks and load from disk
            self._history_manager.finalize()
            df = self._history_manager.get_dataframe()
            result = {col: df[col].values for col in df.columns}
        else:
            # Traditional in-memory mode
            result = {k: v[:actual_steps] for k, v in self._history.items()}

        # Merge matrix history (2D arrays stored separately)
        for k, v in self._matrix_history.items():
            result[k] = v[:actual_steps]

        self._cached_history = result
        return result

    def export_history_to_csv(self, output_path: Path) -> bool:
        """
        Stream history to CSV if using chunked storage.
        Returns True if streamed, False if not (caller should use get_history).
        """
        if self._use_chunked_history and self._history_manager:
            self._history_manager.export_to_csv(output_path)
            return True
        return False
        
    def get_matrix_history(self):
        """Return the matrix history (non-scalar data) separately."""
        # Ensure we return a copy or the dict itself.
        # Check if we need to slice it to actual steps
        actual_steps = self._state.step_idx
        return {k: v[:actual_steps] for k, v in self._matrix_history.items()}

    def _find_soec(self, registry):
        for _, comp in registry.list_components():
            if hasattr(comp, 'soec_state') or comp.__class__.__name__ == 'SOECOperator': return comp
        return None

    def _find_pem(self, registry):
        for _, comp in registry.list_components():
            if hasattr(comp, 'V_cell') or comp.__class__.__name__ == 'DetailedPEMElectrolyzer': return comp
        return None

    def _find_atr(self, registry):
        for _, comp in registry.list_components():
            if 'ATR' in comp.__class__.__name__: return comp
        return None

    def _safe_registry_get(self, component_id: str):
        """Return component if present, otherwise None."""
        if not self._registry:
            return None
        if self._registry.has(component_id):
            return self._registry.get(component_id)
        return None

    def print_summary(self):
        if self._registry: components = {cid: comp for cid, comp in self._registry.list_components()}
        else: components = {}
        from h2_plant.reporting.stream_table import print_stream_summary_table
        print_stream_summary_table(components, list(components.keys()))
