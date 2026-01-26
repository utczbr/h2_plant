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
from h2_plant.components.delivery.discharge_station import DischargeStation
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
    h2o_vapor_arr: np.ndarray  # Water vapor mass flow (kg/h)
    mole_arrs: Tuple[np.ndarray, ...] # (H2, O2, N2, H2O, CH4, CO2)
    
    # Specific component metric arrays (optional)
    extra_metric_arrs: List[Tuple[str, np.ndarray]] = field(default_factory=list)
    
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

    def initialize(
        self,
        registry: 'ComponentRegistry',
        context: 'SimulationContext',
        total_steps: int,
        output_dir: 'Path' = None,
        use_chunked_history: bool = False
    ) -> None:
        self._registry = registry
        self._context = context
        self._total_steps = total_steps

        # Detect topology
        self._soec = self._find_soec(registry)
        self._pem = self._find_pem(registry)
        self._atr = self._find_atr(registry)
        
        # Lookup Power Transformers for efficiency gross-up
        self._soec_trafo = registry.get('SOEC_Transformer')
        self._pem_trafo = registry.get('PEM_Transformer')
        self._bop_trafo = registry.get('BOP_Transformer')
        
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
            from h2_plant.storage.history_manager import ChunkedHistoryManager, HistoryDictProxy
            
            self._history_manager = ChunkedHistoryManager(
                output_dir=output_dir,
                total_steps=total_steps,
                chunk_size=10_000  # ~7 simulated days
            )
            
            # Register all base columns
            base_columns = [
                'minute', 'P_offer', 'P_soec_actual', 'P_pem', 'P_sold',
                'spot_price', 'h2_kg', 'H2_soec_kg', 'H2_pem_kg', 'H2_atr_kg',
                'cumulative_h2_kg', 'steam_soec_kg', 'H2O_soec_out_kg',
                'soec_active_modules', 'H2O_pem_kg', 'O2_pem_kg', 'pem_V_cell',
                'P_bop_mw', 'tank_level_kg', 'tank_pressure_bar', 'compressor_power_kw',
                'sell_decision', 'PEM_o2_impurity_ppm_mol',
                'storage_soc', 'storage_dsoc_per_h', 'storage_zone',
                'storage_action_factor', 'storage_time_to_full_h',
                'h2_rfnbo_kg', 'h2_non_rfnbo_kg', 'cumulative_h2_rfnbo_kg',
                'cumulative_h2_non_rfnbo_kg', 'spot_purchased_mw', 'spot_threshold_eur_mwh',
                'bop_grid_import_mw', 'bop_price_eur_mwh', 'bop_cost_eur',
                'cumulative_bop_cost_eur', 'ppa_price_effective_eur_mwh',
                'integrated_global_efficiency',
                # Efficiency / Grid Side Power (Includes Transformer Losses)
                'P_soec_grid_mw', 'P_pem_grid_mw', 'P_bop_grid_usage_mw'
            ]
            for col in base_columns:
                self._history_manager.register_column(col)
            
            # Allocate first chunk
            self._history_manager.allocate_chunk()
            
            # Use HistoryDictProxy as drop-in replacement for _history
            self._history = HistoryDictProxy(self._history_manager)
            
            logger.info(f"Using CHUNKED history storage: {total_steps} steps, "
                       f"chunk_size=10,000, output={output_dir}")
        else:
            # Traditional in-memory storage (faster, but uses more RAM)
            # 1. Allocate Base History Arrays
            self._history = {
                'minute': np.zeros(total_steps, dtype=np.int32),
                'P_offer': np.zeros(total_steps, dtype=np.float64),
                'P_soec_actual': np.zeros(total_steps, dtype=np.float64),
                'P_pem': np.zeros(total_steps, dtype=np.float64),
                'P_sold': np.zeros(total_steps, dtype=np.float64),
                'spot_price': np.zeros(total_steps, dtype=np.float64),
                'h2_kg': np.zeros(total_steps, dtype=np.float64),
                'H2_soec_kg': np.zeros(total_steps, dtype=np.float64),
                'H2_pem_kg': np.zeros(total_steps, dtype=np.float64),
                'H2_atr_kg': np.zeros(total_steps, dtype=np.float64),
                'cumulative_h2_kg': np.zeros(total_steps, dtype=np.float64),
                'steam_soec_kg': np.zeros(total_steps, dtype=np.float64),
                'H2O_soec_out_kg': np.zeros(total_steps, dtype=np.float64),
                'soec_active_modules': np.zeros(total_steps, dtype=np.int32),
                'H2O_pem_kg': np.zeros(total_steps, dtype=np.float64),
                'O2_pem_kg': np.zeros(total_steps, dtype=np.float64),
                'pem_V_cell': np.zeros(total_steps, dtype=np.float64),
                'P_bop_mw': np.zeros(total_steps, dtype=np.float64),
                'tank_level_kg': np.zeros(total_steps, dtype=np.float64),
                'tank_pressure_bar': np.zeros(total_steps, dtype=np.float64),
                'compressor_power_kw': np.zeros(total_steps, dtype=np.float64),
                'sell_decision': np.zeros(total_steps, dtype=np.int8),
                # Specific PEM Impurity
                'PEM_o2_impurity_ppm_mol': np.zeros(total_steps, dtype=np.float64),
                
                # --- STORAGE CONTROL HISTORY (APC) ---
                'storage_soc': np.zeros(total_steps, dtype=np.float64),
                'storage_dsoc_per_h': np.zeros(total_steps, dtype=np.float64),
                'storage_zone': np.zeros(total_steps, dtype=np.int8),
                'storage_action_factor': np.zeros(total_steps, dtype=np.float64),
                'storage_time_to_full_h': np.zeros(total_steps, dtype=np.float64),
                
                # --- RFNBO CLASSIFICATION (Economic Spot Dispatch) ---
                'h2_rfnbo_kg': np.zeros(total_steps, dtype=np.float64),
                'h2_non_rfnbo_kg': np.zeros(total_steps, dtype=np.float64),
                'cumulative_h2_rfnbo_kg': np.zeros(total_steps, dtype=np.float64),
                'cumulative_h2_non_rfnbo_kg': np.zeros(total_steps, dtype=np.float64),
                'spot_purchased_mw': np.zeros(total_steps, dtype=np.float64),
                'spot_threshold_eur_mwh': np.zeros(total_steps, dtype=np.float64),
                
                # --- BOP GRID IMPORT ---
                'bop_grid_import_mw': np.zeros(total_steps, dtype=np.float64),
                'bop_price_eur_mwh': np.zeros(total_steps, dtype=np.float64),
                'bop_cost_eur': np.zeros(total_steps, dtype=np.float64),
                'cumulative_bop_cost_eur': np.zeros(total_steps, dtype=np.float64),
                
                # --- DUAL PPA PRICING ---
                'ppa_price_effective_eur_mwh': np.zeros(total_steps, dtype=np.float64),
                
                # --- CENTRALIZED COOLING UTILITY ---
                'cooling_manager_glycol_supply_temp_c': np.zeros(total_steps, dtype=np.float64),
                'cooling_manager_glycol_duty_kw': np.zeros(total_steps, dtype=np.float64),
                'cooling_manager_cw_supply_temp_c': np.zeros(total_steps, dtype=np.float64),
                'cooling_manager_cw_supply_temp_c': np.zeros(total_steps, dtype=np.float64),
                'cooling_manager_cw_duty_kw': np.zeros(total_steps, dtype=np.float64),
                
                # --- INTEGRATED PLANT EFFICIENCY (Eq 5.42) ---
                # --- INTEGRATED PLANT EFFICIENCY (Eq 5.42) ---
                'integrated_global_efficiency': np.zeros(total_steps, dtype=np.float64),
                
                # --- GRID SIDE POWER (AC = DC / Eff) ---
                'P_soec_grid_mw': np.zeros(total_steps, dtype=np.float64),
                'P_pem_grid_mw': np.zeros(total_steps, dtype=np.float64),
                'P_bop_grid_usage_mw': np.zeros(total_steps, dtype=np.float64)
            }

        # 2. Identify Components & Pre-Bind Arrays
        self._recorders = []
        self._prebind_recorders(registry, total_steps)
        
        # SOEC Specific Modules
        if self._soec:
            num_modules = getattr(self._soec, 'num_modules', 0)
            for i in range(num_modules):
                self._history[f"soec_module_powers_{i+1}"] = np.zeros(total_steps, dtype=np.float64)

        # 3. Storage Controller Setup (APC)
        self._setup_storage_controller(registry)
        
        # Record storage capacity to history for visualization (constant value)
        if self._storage_total_capacity_kg > 0:
            self._history['storage_capacity_kg'] = np.full(total_steps, self._storage_total_capacity_kg, dtype=np.float64)

        self._state = IntegratedDispatchState()
        logger.info(f"Initialized HybridArbitrageEngineStrategy with {total_steps} steps and Storage APC")

    def _prebind_recorders(self, registry: 'ComponentRegistry', total_steps: int) -> None:
        """
        Scan registry, allocate arrays, and bind them to StreamRecorder objects.
        This enables O(1) access during the simulation loop.
        """
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
                ('atr_heat_duty_kw', 'heat_duty_kw')
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
                 mole_arrs=(
                     self._history[f"{cid}_outlet_H2_molf"],
                     self._history[f"{cid}_outlet_O2_molf"],
                     self._history[f"{cid}_outlet_N2_molf"],
                     self._history[f"{cid}_outlet_H2O_molf"],
                     self._history[f"{cid}_outlet_CH4_molf"],
                     self._history[f"{cid}_outlet_CO2_molf"]
                 ),
                 extra_metric_arrs=[('outlet_o2_ppm_mol', self._history[f"{cid}_outlet_o2_ppm_mol"])]
             )

             soec_recorder.bind_accessor()
             self._recorders.append(soec_recorder)
        
        # Separate dict for 2D matrix arrays (not compatible with HistoryDictProxy)
        # These are always in-memory numpy arrays regardless of chunked mode
        self._matrix_history: Dict[str, np.ndarray] = {}
        self._detailed_tank_recorders = []  # List of (component, pressure_matrix, mass_matrix)

        for cid, comp in registry.list_components():
            ctype = type(comp)
            
            # Special Matrix Recording for DetailedTankArray
            if isinstance(comp, DetailedTankArray):
                n_tanks = comp.n_tanks
                # Allocate (Time, Tank) matrices - use separate dict for 2D arrays
                p_matrix_key = f"{cid}_tank_pressures_bar"
                m_matrix_key = f"{cid}_tank_masses_kg"
                
                self._matrix_history[p_matrix_key] = np.zeros((total_steps, n_tanks), dtype=np.float32)
                self._matrix_history[m_matrix_key] = np.zeros((total_steps, n_tanks), dtype=np.float32)
                
                self._detailed_tank_recorders.append((
                    comp, 
                    self._matrix_history[p_matrix_key],
                    self._matrix_history[m_matrix_key]
                ))
            
            # Special handling for inheritance or if exact type in map
            
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
                    self._history[hist_key] = np.zeros(total_steps, dtype=np.float64)
                    metric_recorders.append((obj_attr, self._history[hist_key]))
                
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
                    mole_arrs=(
                        self._history[f"{cid}_outlet_H2_molf"],
                        self._history[f"{cid}_outlet_O2_molf"],
                        self._history[f"{cid}_outlet_N2_molf"],
                        self._history[f"{cid}_outlet_H2O_molf"],
                        self._history[f"{cid}_outlet_CH4_molf"],
                        self._history[f"{cid}_outlet_CO2_molf"]
                    ),
                    extra_metric_arrs=metric_recorders
                )

                recorder.bind_accessor()
                self._recorders.append(recorder)

    def _alloc_stream_history(self, cid: str, total_steps: int) -> None:
        """Allocate standard outlet stream history arrays."""
        self._history[f"{cid}_outlet_temp_c"] = np.zeros(total_steps, dtype=np.float64)
        self._history[f"{cid}_outlet_pressure_bar"] = np.zeros(total_steps, dtype=np.float64)
        self._history[f"{cid}_outlet_mass_flow_kg_h"] = np.zeros(total_steps, dtype=np.float64)
        self._history[f"{cid}_outlet_h2o_frac"] = np.zeros(total_steps, dtype=np.float64)
        self._history[f"{cid}_h2o_vapor_kg_h"] = np.zeros(total_steps, dtype=np.float64)  # Water vapor mass flow
        for sp in ['H2', 'O2', 'N2', 'H2O', 'CH4', 'CO2']:
            self._history[f"{cid}_outlet_{sp}_molf"] = np.zeros(total_steps, dtype=np.float64)

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
                # Calculate max mass via Ideal Gas Law at max pressure
                try:
                    V = comp.volume_per_tank
                    n = comp.n_tanks
                    P_max = comp.max_pressure_pa  # Already in Pa
                    R = 4124.0  # H2 specific gas constant (J/(kg·K))
                    T = comp.ambient_temp_k  # Use configured ambient temperature
                    cap = n * (P_max * V) / (R * T)
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
        """Calculate Plant-Wide State of Charge (0.0 to 1.0) and current mass."""
        if self._storage_total_capacity_kg <= 0:
            return 0.0, 0.0

        current_mass = 0.0
        for comp, _ in self._storage_info:
            # Use unified interface or direct access
            if hasattr(comp, 'get_inventory_kg'):
                current_mass += comp.get_inventory_kg()
            elif hasattr(comp, 'get_total_mass'): # DetailedTankArray
                val = comp.get_total_mass()
                # logger.info(f"APC Mass Check: {comp.component_id} via get_total_mass() = {val}")
                current_mass += val
            elif hasattr(comp, 'masses'):  # TankArray direct
                current_mass += np.sum(comp.masses)
            elif hasattr(comp, 'mass_kg'):  # Enhanced direct
                val = comp.mass_kg
                logger.info(f"APC Mass Check: {comp.component_id} via mass_kg = {val} (Typo/Wrong Branch?)")
                current_mass += comp.mass_kg

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
                soc_limit_high=0.98,
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

        # Record dispatch data
        self._history['minute'][step_idx] = minute
        self._history['P_offer'][step_idx] = P_offer
        
        # Record Storage APC data
        self._history['storage_soc'][step_idx] = soc
        self._history['storage_dsoc_per_h'][step_idx] = dsoc_dt
        self._history['storage_zone'][step_idx] = zone
        self._history['storage_action_factor'][step_idx] = action_factor
        self._history['storage_time_to_full_h'][step_idx] = self._ctrl_state['time_to_full_h']
        self._history['spot_price'][step_idx] = current_price
        self._history['ppa_price_effective_eur_mwh'][step_idx] = current_ppa_price
        # Record RFNBO classification metrics
        # NOTE: RFNBO is now calculated from ACTUAL production in record_post_step
        # This section only sets dispatch-related metrics (spot_purchased, threshold)
        
        if 'h2_rfnbo_kg' in result.state_update:
            # EconomicSpotDispatchStrategy: extract spot metrics
            spot_purchased = result.state_update.get('spot_purchased_mw', 0.0)
            spot_threshold = result.state_update.get('spot_threshold_eur_mwh', 0.0)
            # h2_non_rfnbo comes from spot purchase (PEM powered by grid)
            h2_non_rfnbo = result.state_update.get('h2_non_rfnbo_kg', 0.0)
        else:
            # Non-ECONOMIC_SPOT: no spot purchase, all renewable
            spot_purchased = 0.0
            spot_threshold = 0.0
            h2_non_rfnbo = 0.0
        
        # Initialize with 0 - actual values are set in record_post_step using actual H2 production
        self._history['h2_rfnbo_kg'][step_idx] = 0.0  # Will be overwritten
        self._history['h2_non_rfnbo_kg'][step_idx] = h2_non_rfnbo
        self._history['spot_purchased_mw'][step_idx] = spot_purchased
        self._history['spot_threshold_eur_mwh'][step_idx] = spot_threshold
        
        # Cumulative RFNBO will be updated in record_post_step with actual production values
        # Initialize with previous value (will be incremented in record_post_step)
        if step_idx > 0:
            self._history['cumulative_h2_rfnbo_kg'][step_idx] = self._history['cumulative_h2_rfnbo_kg'][step_idx - 1]
            self._history['cumulative_h2_non_rfnbo_kg'][step_idx] = (
                self._history['cumulative_h2_non_rfnbo_kg'][step_idx - 1] + h2_non_rfnbo
            )
        else:
            self._history['cumulative_h2_rfnbo_kg'][step_idx] = 0.0
            self._history['cumulative_h2_non_rfnbo_kg'][step_idx] = h2_non_rfnbo
        
        self._state.step_idx = step_idx

    def record_post_step(self) -> None:
        """
        Record component outputs.
        OPTIMIZED: Uses pre-bound recorders to avoid dict lookups.
        """
        step_idx = self._state.step_idx
        if step_idx >= self._total_steps:
            return

        # 1. Specialized Recording (SOEC/PEM Main metrics)
        # (These are kept specific because they involve logic, not just stream dumping)
        
        # SOEC Logic
        P_soec_actual = 0.0
        h2_soec = 0.0
        steam_soec = 0.0
        if self._soec:
            if hasattr(self._soec, 'real_powers'):
                P_soec_actual = float(np.sum(self._soec.real_powers))
            if hasattr(self._soec, 'last_step_h2_kg'): h2_soec = self._soec.last_step_h2_kg
            elif hasattr(self._soec, 'last_h2_output_kg'): h2_soec = self._soec.last_h2_output_kg
            elif hasattr(self._soec, 'h2_output_kg'): h2_soec = self._soec.h2_output_kg
            steam_soec = getattr(self._soec, 'last_step_steam_input_kg', 0.0)
            
            # Record unreacted steam out
            self._history['H2O_soec_out_kg'][step_idx] = getattr(self._soec, 'last_water_output_kg', 0.0)
            
            # Manually handle SOEC stream recording (dynamic stream)
            cid = self._soec.component_id if hasattr(self._soec, 'component_id') else 'SOEC_Cluster'
            if cid:
                try:
                    out_stream = self._soec.get_output('h2_out')
                    if out_stream and out_stream.mass_flow_kg_h > 1e-6:
                         # Using standard access here as SOEC stream is special/rarely zero
                         self._history[f"{cid}_outlet_mass_flow_kg_h"][step_idx] = out_stream.mass_flow_kg_h
                         self._history[f"{cid}_outlet_temp_c"][step_idx] = out_stream.temperature_k - 273.15
                         self._history[f"{cid}_outlet_pressure_bar"][step_idx] = out_stream.pressure_pa / 1e5
                         self._history[f"{cid}_outlet_h2o_frac"][step_idx] = out_stream.composition.get('H2O', 0.0)
                except Exception:
                    pass

        self._state.P_soec_prev = P_soec_actual

        # PEM Logic
        h2_pem = 0.0
        P_pem_actual = 0.0
        if self._pem:
            h2_pem = getattr(self._pem, 'h2_output_kg', 0.0)
            if hasattr(self._pem, 'P_consumed_W'): P_pem_actual = self._pem.P_consumed_W / 1e6
            
            # Record impurity if available
            imp = getattr(self._pem, 'o2_impurity_ppm_mol', 0.0)
            self._history['PEM_o2_impurity_ppm_mol'][step_idx] = imp
            
            # Record additional PEM metrics
            self._history['H2O_pem_kg'][step_idx] = getattr(self._pem, 'water_consumption_kg', 0.0)
            self._history['O2_pem_kg'][step_idx] = getattr(self._pem, 'o2_output_kg', 0.0)
            self._history['pem_V_cell'][step_idx] = getattr(self._pem, 'V_cell', 0.0)

        # ATR Logic
        h2_atr = 0.0
        if self._atr:
            # h2_production_kmol_h -> kg/step
            prod_kmol_h = getattr(self._atr, 'h2_production_kmol_h', 0.0)
            dt = self._context.simulation.timestep_hours
            h2_atr = prod_kmol_h * 2.016 * dt

        # Global Power and Component Logic
        P_bop_kw = 0.0
        # Check compressors and tanks if not already cached
        if not hasattr(self, '_compressors'):
            self._compressors = [comp for _, comp in self._registry.list_components() if isinstance(comp, CompressorSingle)]
        if not hasattr(self, '_tanks'):
            # Handling for TankArray not explicitly imported at top level scope in some contexts
            from h2_plant.components.storage.h2_tank import TankArray
            self._tanks = [comp for _, comp in self._registry.list_components() if isinstance(comp, TankArray)]

        # Compressor Total Power
        total_comp_power = 0.0
        for comp in self._compressors:
            if hasattr(comp, 'power_kw'):
                total_comp_power += comp.power_kw
        if total_comp_power < 0.0:
            total_comp_power = 0.0
        self._history['compressor_power_kw'][step_idx] = total_comp_power

        # Tank Levels
        total_tank_mass = 0.0
        avg_tank_pressure = 0.0
        if self._tanks:
            for tank in self._tanks:
                total_tank_mass += tank.get_total_mass()
            # Use pressure of first tank array as representative of storage pressure
            if len(self._tanks[0].pressures) > 0:
                avg_tank_pressure = np.mean(self._tanks[0].pressures) / 1e5
        self._history['tank_level_kg'][step_idx] = total_tank_mass
        self._history['tank_pressure_bar'][step_idx] = avg_tank_pressure

        # BOP Calculation (iterate all components)
        for _, comp in self._registry.list_components():
             if hasattr(comp, 'power_kw'): P_bop_kw += comp.power_kw
             elif hasattr(comp, 'electrical_power_kw'): P_bop_kw += comp.electrical_power_kw
             elif hasattr(comp, 'fan_power_kw'): P_bop_kw += comp.fan_power_kw
             elif hasattr(comp, 'current_power_w'): P_bop_kw += (comp.current_power_w / 1000.0)

        P_bop_mw = P_bop_kw / 1000.0
        
        # === NEW: BOP is imported from grid separately (not from wind) ===
        # Wind power goes 100% to electrolyzers (RFNBO-compliant)
        P_consumed_from_wind = P_soec_actual + P_pem_actual  # BOP excluded
        P_offer = self._history['P_offer'][step_idx]
        P_sold_corrected = max(0.0, P_offer - P_consumed_from_wind)
        
        # BOP Grid Import Cost Calculation
        dt = self._context.simulation.timestep_hours
        bop_pricing_mode = getattr(self._context.economics, 'bop_pricing_mode', 'fixed')
        spot_price = self._history['spot_price'][step_idx]
        
        if bop_pricing_mode == 'spot':
            bop_price = spot_price
        else:
            bop_price = getattr(self._context.economics, 'bop_fixed_price_eur_mwh', 80.0)
        
        # Calculate Grid Side Power (AC)
        # =====================================================================
        # Apply transformer losses to get actual Grid Draw
        # P_grid = P_dc / efficiency
        
        P_soec_grid_mw = P_soec_actual / self._η_soec_trafo if self._η_soec_trafo > 0 else P_soec_actual
        P_pem_grid_mw = P_pem_actual / self._η_pem_trafo if self._η_pem_trafo > 0 else P_pem_actual
        # For BoP, P_bop_mw is the Load. Grid Draw = Load / Eff
        P_bop_grid_mw = P_bop_mw / self._η_bop_trafo if self._η_bop_trafo > 0 else P_bop_mw
        
        # BOP Cost should use the GRID SIDE power (AC), not load
        bop_cost_eur = P_bop_grid_mw * dt * bop_price
        
        total_h2 = h2_soec + h2_pem + h2_atr
        self._state.cumulative_h2_kg += total_h2

        # Fast Array Writes
        self._history['P_soec_actual'][step_idx] = P_soec_actual
        self._history['P_pem'][step_idx] = P_pem_actual
        self._history['P_sold'][step_idx] = P_sold_corrected
        
        # Record Grid Side Power
        self._history['P_soec_grid_mw'][step_idx] = P_soec_grid_mw
        self._history['P_pem_grid_mw'][step_idx] = P_pem_grid_mw
        self._history['P_bop_grid_usage_mw'][step_idx] = P_bop_grid_mw
        self._history['h2_kg'][step_idx] = total_h2
        self._history['H2_soec_kg'][step_idx] = h2_soec
        self._history['H2_pem_kg'][step_idx] = h2_pem
        self._history['H2_atr_kg'][step_idx] = h2_atr
        self._history['cumulative_h2_kg'][step_idx] = self._state.cumulative_h2_kg
        # === RFNBO CALCULATION FROM ACTUAL PRODUCTION ===
        # Use actual H2 production values instead of dispatch estimates
        # This ensures cumulative_h2_rfnbo_kg == cumulative_h2_kg when Non-RFNBO is 0
        h2_rfnbo_actual = h2_soec + h2_pem + h2_atr
        
        # Set h2_rfnbo_kg (was initialized to 0 in _record_dispatch)
        self._history['h2_rfnbo_kg'][step_idx] = h2_rfnbo_actual
        
        # Update cumulative RFNBO with actual production
        self._history['cumulative_h2_rfnbo_kg'][step_idx] += h2_rfnbo_actual
        
        self._history['steam_soec_kg'][step_idx] = steam_soec
        self._history['P_bop_mw'][step_idx] = P_bop_mw
        self._history['sell_decision'][step_idx] = 1 if P_sold_corrected > 0 else 0
        
        # BOP Grid Import Recording
        # BOP Grid Import Recording
        # Note: 'bop_grid_import_mw' tracks the commercial import (if distinct from P_bop_mw logic)
        # But for consistency, let's record the physical grid usage here too?
        # The original code just recorded P_bop_mw. Now we record P_bop_grid_mw (AC).
        self._history['bop_grid_import_mw'][step_idx] = P_bop_grid_mw
        self._history['bop_price_eur_mwh'][step_idx] = bop_price
        self._history['bop_cost_eur'][step_idx] = bop_cost_eur
        if step_idx > 0:
            self._history['cumulative_bop_cost_eur'][step_idx] = (
                self._history['cumulative_bop_cost_eur'][step_idx - 1] + bop_cost_eur
            )
        else:
            self._history['cumulative_bop_cost_eur'][step_idx] = bop_cost_eur
        
        # SOEC Modules
        if self._soec and hasattr(self._soec, 'real_powers'):
            self._history['soec_active_modules'][step_idx] = int(np.sum(self._soec.real_powers > 0.01))
            # Only record if we need to
            if self._total_steps < 1000000: # Skip module detail for massive runs?
                for i, power_mw in enumerate(self._soec.real_powers):
                    key = f"soec_module_powers_{i+1}"
                    if key in self._history: self._history[key][step_idx] = power_mw

        # 2. CoolingManager State Recording
        cooling_manager = self._registry.get('cooling_manager') if self._registry else None
        if cooling_manager:
            self._history['cooling_manager_glycol_supply_temp_c'][step_idx] = getattr(cooling_manager, 'glycol_supply_temp_c', 0.0)
            self._history['cooling_manager_glycol_duty_kw'][step_idx] = getattr(cooling_manager, 'glycol_duty_kw', 0.0)
            self._history['cooling_manager_cw_supply_temp_c'][step_idx] = getattr(cooling_manager, 'cw_supply_temp_c', 0.0)
            self._history['cooling_manager_cw_duty_kw'][step_idx] = getattr(cooling_manager, 'cw_duty_kw', 0.0)

        # 2. Optimized Component Recording Loop
        # Iterate over pre-bound recorders (O(N) where N is component count, no string hashing)
        for rec in self._recorders:
            # 2a. Get Stream (Optimized Access)
            # Use pre-bound accessor (lambda) to avoid dynamic getattr/if checks in hot loop
            stream = rec.stream_getter()

            # 2b. Zero-Flow Optimization (REMOVED: Need to record T/P even at zero flow)
            if stream is not None:
                # Direct Array Write
                rec.temp_arr[step_idx] = stream.temperature_k - 273.15
                rec.press_arr[step_idx] = stream.pressure_pa / 1e5
                rec.flow_arr[step_idx] = stream.mass_flow_kg_h
                
                h2o_val = stream.composition.get('H2O', 0.0) + stream.composition.get('H2O_liq', 0.0)
                rec.h2o_frac_arr[step_idx] = h2o_val

                # Direct Mole Fraction Write (Total Phase: Gas + Liquid)
                # FIX: Use get_total_mole_frac to include liquid water in the denominator.
                # This matches the summary table and avoids inflating O2 PPM in "wet" streams.
                
                rec.mole_arrs[0][step_idx] = stream.get_total_mole_frac('H2')
                rec.mole_arrs[1][step_idx] = stream.get_total_mole_frac('O2')
                rec.mole_arrs[2][step_idx] = stream.get_total_mole_frac('N2')
                rec.mole_arrs[3][step_idx] = stream.get_total_mole_frac('H2O') # Includes Liq + Vap
                rec.mole_arrs[4][step_idx] = stream.get_total_mole_frac('CH4')
                rec.mole_arrs[5][step_idx] = stream.get_total_mole_frac('CO2')
            
            # 2c. Record Extra Metrics (Independent of flow)
            for attr_name, metric_arr in rec.extra_metric_arrs:
                # Try direct attribute first, fallback to get_state() for state-only metrics
                val = getattr(rec.component, attr_name, None)
                if val is None:
                    # Fallback: Check get_state() dict
                    state = rec.component.get_state() if hasattr(rec.component, 'get_state') else {}
                    val = state.get(attr_name, 0.0)
                metric_arr[step_idx] = val if val is not None else 0.0

        # 3. DetailedTankArray Matrix Recording
        for comp, p_matrix, m_matrix in self._detailed_tank_recorders:
            # Efficiently grab data from tanks
            p_matrix[step_idx, :] = [t.pressure_pa / 1e5 for t in comp.tanks]
            m_matrix[step_idx, :] = [t.mass_kg for t in comp.tanks]
            
        # 4. Integrated Plant Efficiency
        self._calculate_integrated_efficiency(step_idx)

        self._state.step_idx += 1
        
        # 5. Chunked Storage: Trigger chunk flush if needed
        if self._use_chunked_history and self._history_manager:
            self._history_manager.step_complete(step_idx)

    def _calculate_integrated_efficiency(self, step_idx: int) -> None:
        """
        Calculate Integrated Plant Efficiency (Eq 5.42).
        
        Formula:
            eta_global = (m_H2_total * LHV_H2) / (P_el_total + Energy_Biogas)
            
        Where:
            m_H2_total = H2_PEM + H2_SOEC + H2_ATR (kg/s)
            P_el_total = Sum of all auxiliary loads (kW)
            Energy_Biogas = Biogas flow * LHV (kW)
        """
        # 1. H2 Production (Total)
        dt_hours = self._context.simulation.timestep_hours
        if dt_hours <= 0: return

        # Load H2 LHV (kWh/kg -> kW via mass flow)
        # Using LHV_H2 = 33.33 kWh/kg = 120 MJ/kg
        LHV_H2_KWH_KG = 33.33 

        # Sum H2 produced in this timestep (kg)
        try:
            h2_soec_kg = self._history['H2_soec_kg'][step_idx]
            h2_pem_kg = self._history['H2_pem_kg'][step_idx]
            h2_atr_kg = self._history['H2_atr_kg'][step_idx]
        except (KeyError, TypeError):
             h2_soec_kg = h2_pem_kg = h2_atr_kg = 0.0

        total_h2_kg = h2_soec_kg + h2_pem_kg + h2_atr_kg
        
        # Energy Output (kW) = (kg / h) * kWh/kg = kW
        energy_h2_kw = (total_h2_kg / dt_hours) * LHV_H2_KWH_KG

        # 2. Total Electrical Consumption (P_el_total)
        p_el_total_kw = 0.0
        
        # A. Electrolyzers (Use Grid Side AC Power)
        p_el_total_kw += (self._history['P_soec_grid_mw'][step_idx] * 1000.0)
        p_el_total_kw += (self._history['P_pem_grid_mw'][step_idx] * 1000.0)
        
        # B. Auxiliaries (Registry Scan)
        # Robust aggregation of all known power consumers
        if self._registry:
            for cid, comp in self._registry.list_components():
                 # Skip if it's the main electrolyzer stacks (already added from history)
                 # Note: Checking identity for safety
                 if (self._soec and comp == self._soec) or (self._pem and comp == self._pem):
                     continue
                     
                 # Also skip if it's a known non-consuming component class to speed up?
                 # No, better to check for power attributes generically.

                 state = comp.get_state()
                 
                 # Try common power metrics
                 p_comp = 0.0
                 
                 # Check various keys used across codebase
                 # Prioritize specific ones to avoid double counting if multiple keys exist for same value
                 if 'electrical_power_kw' in state:
                      p_comp = state['electrical_power_kw']
                 elif 'power_consumed_kw' in state:
                      p_comp = state['power_consumed_kw']
                 elif 'fan_power_kw' in state:
                      p_comp = state['fan_power_kw'] # Dry Cooler
                 elif 'power_kw' in state:
                      p_comp = state['power_kw'] # Compressors, Pumps
                 
                 # Accumulate positive power consumption only
                 if p_comp > 0:
                     p_el_total_kw += p_comp

        # C. Add centralized BoP (Use Grid Side AC Power)
        p_el_total_kw += (self._history['P_bop_grid_usage_mw'][step_idx] * 1000.0)

        # 3. Biogas Energy Input
        energy_biogas_kw = 0.0
        if self._atr:
             state = self._atr.get_state()
             energy_biogas_kw = state.get('biogas_energy_input_kw', 0.0)
        
        # 4. Calculation
        numerator = energy_h2_kw
        denominator = p_el_total_kw + energy_biogas_kw
        
        if denominator > 1.0:
            eff = numerator / denominator
        else:
            eff = 0.0
            
        # 5. Store in History
        if 'integrated_global_efficiency' in self._history:
             self._history['integrated_global_efficiency'][step_idx] = eff

    def get_history(self) -> Dict[str, np.ndarray]:
        """
        Get the recorded history.
        
        Returns dict of column -> array for in-memory mode.
        For chunked mode, finalizes and returns DataFrame-based dict.
        """
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

    def print_summary(self):
        if self._registry: components = {cid: comp for cid, comp in self._registry.list_components()}
        else: components = {}
        from h2_plant.reporting.stream_table import print_stream_summary_table
        print_stream_summary_table(components, list(components.keys()))

