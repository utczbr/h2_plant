"""
Integrated Dispatch Strategy for SimulationEngine.

This module provides dispatch strategy classes that integrate with the
SimulationEngine, replacing the standalone Orchestrator execution path.

Design Rationale:
    The strategy encapsulates:
    - **Dispatch Decision Logic**: Which electrolyzer, how much to sell.
    - **Power Setpoint Application**: Sets component inputs before step().
    - **History Recording**: NumPy pre-allocated arrays for performance.

Performance Optimization:
    Uses NumPy pre-allocated arrays instead of list.append() for history
    recording. This provides 10-50x speedup for year-long simulations
    by avoiding Python list reallocation overhead.

Integration Pattern:
    1. Engine calls `decide_and_apply()` BEFORE component step().
    2. Engine calls component.step() for physics execution.
    3. Engine calls `record_post_step()` AFTER step() to capture results.

    This separation ensures dispatch sets intentions while physics
    determines actual outcomes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, TYPE_CHECKING
import numpy as np
import logging

from h2_plant.control.dispatch import (
    DispatchInput,
    DispatchState,
    DispatchResult,
    DispatchStrategy as BaseDispatchStrategy,
    ReferenceHybridStrategy,
    SoecOnlyStrategy
)

if TYPE_CHECKING:
    from h2_plant.core.component_registry import ComponentRegistry
    from h2_plant.config.plant_config import SimulationContext

from h2_plant.components.thermal.chiller import Chiller
from h2_plant.components.separation.coalescer import Coalescer
from h2_plant.components.purification.deoxo_reactor import DeoxoReactor
from h2_plant.components.separation.psa import PSA
from h2_plant.components.separation.knock_out_drum import KnockOutDrum
from h2_plant.components.separation.hydrogen_cyclone import HydrogenMultiCyclone
from h2_plant.components.thermal.interchanger import Interchanger
from h2_plant.components.compression.compressor_single import CompressorSingle
from h2_plant.components.cooling.dry_cooler import DryCooler
from h2_plant.components.thermal.heat_exchanger import HeatExchanger
from h2_plant.components.thermal.electric_boiler import ElectricBoiler
from h2_plant.components.water.drain_recorder_mixer import DrainRecorderMixer

logger = logging.getLogger(__name__)




@dataclass
class IntegratedDispatchState:
    """
    Extended dispatch state with history tracking.

    Attributes:
        P_soec_prev (float): Previous SOEC power setpoint (MW).
        force_sell (bool): Arbitrage sell mode active flag.
        step_idx (int): Current step index for array indexing.
        cumulative_h2_kg (float): Running total H₂ production (kg).
    """
    P_soec_prev: float = 0.0
    force_sell: bool = False
    step_idx: int = 0
    cumulative_h2_kg: float = 0.0


class EngineDispatchStrategy(ABC):
    """
    Abstract base for dispatch strategies integrated with SimulationEngine.

    Unlike standalone DispatchStrategy, this class:
    - Has access to ComponentRegistry for direct component manipulation.
    - Manages its own state and history with NumPy arrays.
    - Applies dispatch decisions to components via receive_input().
    - Records actual physics results after step() execution.
    """

    @abstractmethod
    def initialize(
        self,
        registry: 'ComponentRegistry',
        context: 'SimulationContext',
        total_steps: int
    ) -> None:
        """
        Initialize strategy with system references and pre-allocate arrays.

        Args:
            registry (ComponentRegistry): Component registry for access.
            context (SimulationContext): Configuration with physics parameters.
            total_steps (int): Total timesteps for array pre-allocation.
        """
        pass

    @abstractmethod
    def decide_and_apply(self, t: float, prices: np.ndarray, wind: np.ndarray) -> None:
        """
        Make dispatch decision and apply setpoints to components.

        Called by SimulationEngine at each timestep BEFORE stepping components.
        This sets power inputs; actual physics executes in component.step().

        Args:
            t (float): Current simulation time in hours.
            prices (np.ndarray): Energy price array for full simulation (EUR/MWh).
            wind (np.ndarray): Wind power offer array for full simulation (MW).
        """
        pass

    @abstractmethod
    def get_history(self) -> Dict[str, np.ndarray]:
        """
        Return recorded history as NumPy arrays.

        Returns:
            Dict[str, np.ndarray]: History arrays keyed by metric name.
        """
        pass


class HybridArbitrageEngineStrategy(ReferenceHybridStrategy):
    """
    Hybrid SOEC/PEM arbitrage strategy integrated with SimulationEngine.

    Migrated from Orchestrator's run_simulation() dispatch logic with
    NumPy pre-allocated arrays for HPC-compatible performance.

    Features:
        - Automatic topology detection (SOEC-only vs Hybrid).
        - Pre-allocated NumPy arrays avoid list.append() overhead.
        - Separation of setpoint application and result recording.

    History Arrays:
        The strategy pre-allocates arrays for all metrics at initialization,
        providing 10-50x speedup compared to list.append() for year-long runs.

    Example:
        >>> strategy = HybridArbitrageEngineStrategy()
        >>> strategy.initialize(registry, context, total_steps=525600)
        >>> # ... engine calls component.step() ...
        >>> strategy.record_post_step()
    """

    def __init__(self):
        """
        Initialize the hybrid arbitrage strategy.
        """
        self._registry: Optional['ComponentRegistry'] = None
        self._context: Optional['SimulationContext'] = None
        self._inner_strategy: Optional[BaseDispatchStrategy] = None
        self._state = IntegratedDispatchState()

        # Component references
        self._soec = None
        self._pem = None
        self._chillers: list['Chiller'] = []

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
        total_steps: int
    ) -> None:
        """
        Initialize strategy with pre-allocated NumPy arrays.

        Detects topology (SOEC-only vs Hybrid), caches capacities,
        and pre-allocates all history arrays for the full simulation.

        Args:
            registry (ComponentRegistry): Component registry.
            context (SimulationContext): Physics configuration.
            total_steps (int): Total timesteps for pre-allocation.
        """
        self._registry = registry
        self._context = context
        self._total_steps = total_steps

        # Detect topology and select inner strategy
        self._soec = self._find_soec(registry)
        self._pem = self._find_pem(registry)
        
        # Identify auxiliary components for tracking
        self._chillers = [
            c for _, c in registry.list_components() 
            if isinstance(c, Chiller)
        ]
        self._coalescers = [
            c for _, c in registry.list_components() 
            if isinstance(c, Coalescer)
        ]
        self._deoxos = [
            c for _, c in registry.list_components() 
            if isinstance(c, DeoxoReactor)
        ]
        self._psas = [
            c for _, c in registry.list_components() 
            if isinstance(c, PSA)
        ]
        self._kods = [
            c for _, c in registry.list_components() 
            if isinstance(c, KnockOutDrum)
        ]
        self._cyclones = [
            c for _, c in registry.list_components() 
            if isinstance(c, HydrogenMultiCyclone)
        ]
        self._compressors = [
            c for _, c in registry.list_components() 
            if isinstance(c, CompressorSingle)
        ]
        self._dry_coolers = [
            c for _, c in registry.list_components() 
            if isinstance(c, DryCooler)
        ]
        self._heat_exchangers = [
            c for _, c in registry.list_components() 
            if isinstance(c, HeatExchanger)
        ]
        self._drain_mixers = [
            c for _, c in registry.list_components() 
            if isinstance(c, DrainRecorderMixer)
        ]
        self._interchangers = [
            c for _, c in registry.list_components() 
            if isinstance(c, Interchanger)
        ]
        self._boilers = [
            c for _, c in registry.list_components() 
            if isinstance(c, ElectricBoiler)
        ]

        if self._soec and not self._pem:
            logger.info("Topology detected: SOEC Only. Using SoecOnlyStrategy.")
            self._inner_strategy = SoecOnlyStrategy()
        else:
            logger.info("Topology detected: Hybrid (or default). Using ReferenceHybridStrategy.")
            self._inner_strategy = ReferenceHybridStrategy()

        # Cache component capacities
        if self._soec:
            spec = context.physics.soec_cluster
            # Calculate SOEC capacity: Modules * Nominal Power * Limit
            self._soec_capacity = spec.num_modules * spec.max_power_nominal_mw * spec.optimal_limit

        if self._pem:
            self._pem_max = context.physics.pem_system.max_power_mw

        # Pre-allocate history arrays for performance
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
            'cumulative_h2_kg': np.zeros(total_steps, dtype=np.float64),
            'steam_soec_kg': np.zeros(total_steps, dtype=np.float64),
            'H2O_soec_out_kg': np.zeros(total_steps, dtype=np.float64),
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
        }
        
        # Add history tracks for each chiller
        for chiller in self._chillers:
            cid = chiller.component_id
            self._history[f"{cid}_cooling_load_kw"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_electrical_power_kw"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_outlet_mass_flow_kg_h"] = np.zeros(total_steps, dtype=np.float64)

        for coal in self._coalescers:
            cid = coal.component_id
            self._history[f"{cid}_delta_p_bar"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_drain_flow_kg_h"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_outlet_o2_ppm_mol"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_dissolved_gas_ppm"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_dissolved_gas_in_kg_h"] = np.zeros(total_steps, dtype=np.float64)  # IN tracking
            self._history[f"{cid}_dissolved_gas_out_kg_h"] = np.zeros(total_steps, dtype=np.float64) # OUT tracking
            # Outlet Stream Properties (for Stacked Properties Graph)
            self._history[f"{cid}_outlet_temp_c"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_outlet_pressure_bar"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_outlet_h2o_frac"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_outlet_enthalpy_kj_kg"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_outlet_mass_flow_kg_h"] = np.zeros(total_steps, dtype=np.float64)

        # Add history tracks for each deoxo
        for deoxo in self._deoxos:
            cid = deoxo.component_id
            self._history[f"{cid}_outlet_o2_ppm_mol"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_inlet_temp_c"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_inlet_pressure_bar"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_o2_in_kg_h"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_peak_temp_c"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_conversion_percent"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_mass_flow_kg_h"] = np.zeros(total_steps, dtype=np.float64)
            # Outlet Stream Properties (for Stacked Properties Graph)
            self._history[f"{cid}_outlet_temp_c"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_outlet_pressure_bar"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_outlet_h2o_frac"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_outlet_h2o_frac"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_outlet_enthalpy_kj_kg"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_outlet_mass_flow_kg_h"] = np.zeros(total_steps, dtype=np.float64)

        # Add history tracks for PSA
        for psa in self._psas:
            cid = psa.component_id
            self._history[f"{cid}_outlet_o2_ppm_mol"] = np.zeros(total_steps, dtype=np.float64)
            # Outlet Stream Properties (for Stacked Properties Graph)
            self._history[f"{cid}_outlet_temp_c"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_outlet_pressure_bar"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_outlet_h2o_frac"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_outlet_enthalpy_kj_kg"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_outlet_mass_flow_kg_h"] = np.zeros(total_steps, dtype=np.float64)

        # Add history tracks for KOD
        for kod in self._kods:
            cid = kod.component_id
            self._history[f"{cid}_outlet_o2_ppm_mol"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_water_removed_kg_h"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_drain_temp_k"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_drain_pressure_bar"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_dissolved_gas_ppm"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_m_dot_H2O_liq_accomp_kg_s"] = np.zeros(total_steps, dtype=np.float64)
            # Dissolved Gas IN/OUT tracking
            self._history[f"{cid}_dissolved_gas_in_kg_h"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_dissolved_gas_out_kg_h"] = np.zeros(total_steps, dtype=np.float64)
            # Outlet Stream Properties (for Stacked Properties Graph)
            self._history[f"{cid}_outlet_temp_c"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_outlet_pressure_bar"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_outlet_h2o_frac"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_outlet_h2o_frac"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_outlet_enthalpy_kj_kg"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_outlet_mass_flow_kg_h"] = np.zeros(total_steps, dtype=np.float64)

        # Add history tracks for Cyclone
        for cyc in self._cyclones:
            cid = cyc.component_id
            self._history[f"{cid}_outlet_o2_ppm_mol"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_water_removed_kg_h"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_drain_temp_k"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_drain_pressure_bar"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_dissolved_gas_ppm"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_dissolved_gas_in_kg_h"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_dissolved_gas_out_kg_h"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_pressure_drop_mbar"] = np.zeros(total_steps, dtype=np.float64)
            # Outlet Stream Properties (for Stacked Properties Graph)
            self._history[f"{cid}_outlet_temp_c"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_outlet_pressure_bar"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_outlet_h2o_frac"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_outlet_enthalpy_kj_kg"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_outlet_mass_flow_kg_h"] = np.zeros(total_steps, dtype=np.float64)

        # Add history tracks for Compressor
        for comp in self._compressors:
            cid = comp.component_id
            self._history[f"{cid}_outlet_o2_ppm_mol"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_power_kw"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_outlet_temp_c"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_outlet_pressure_bar"] = np.zeros(total_steps, dtype=np.float64)
            # Outlet Stream Properties (for Stacked Properties Graph)
            self._history[f"{cid}_outlet_h2o_frac"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_outlet_enthalpy_kj_kg"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_outlet_mass_flow_kg_h"] = np.zeros(total_steps, dtype=np.float64)

        # Add history tracks for DryCooler
        for dc in self._dry_coolers:
            cid = dc.component_id
            self._history[f"{cid}_outlet_o2_ppm_mol"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_heat_rejected_kw"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_tqc_duty_kw"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_dc_duty_kw"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_fan_power_kw"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_outlet_temp_c"] = np.zeros(total_steps, dtype=np.float64)
            # Outlet Stream Properties (for Stacked Properties Graph)
            self._history[f"{cid}_outlet_pressure_bar"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_outlet_h2o_frac"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_outlet_enthalpy_kj_kg"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_outlet_mass_flow_kg_h"] = np.zeros(total_steps, dtype=np.float64)

        # Add history tracks for HeatExchanger
        for hx in self._heat_exchangers:
            cid = hx.component_id
            self._history[f"{cid}_outlet_o2_ppm_mol"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_heat_removed_kw"] = np.zeros(total_steps, dtype=np.float64)
            # Outlet Stream Properties (for Stacked Properties Graph)
            self._history[f"{cid}_outlet_temp_c"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_outlet_pressure_bar"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_outlet_h2o_frac"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_outlet_enthalpy_kj_kg"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_outlet_mass_flow_kg_h"] = np.zeros(total_steps, dtype=np.float64)

        # Add history tracks for DrainRecorderMixer
        for mixer in self._drain_mixers:
            cid = mixer.component_id
            self._history[f"{cid}_dissolved_gas_ppm"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_outlet_mass_flow_kg_h"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_outlet_temperature_c"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_outlet_pressure_kpa"] = np.zeros(total_steps, dtype=np.float64)

        # Add history tracks for Interchanger
        for ic in self._interchangers:
            cid = ic.component_id
            self._history[f"{cid}_q_transferred_kw"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_hot_out_temp_k"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_cold_out_temp_k"] = np.zeros(total_steps, dtype=np.float64)
            # Outlet Stream Properties (for Stacked Properties Graph - Hot Side)
            self._history[f"{cid}_outlet_temp_c"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_outlet_pressure_bar"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_outlet_h2o_frac"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_outlet_enthalpy_kj_kg"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_outlet_mass_flow_kg_h"] = np.zeros(total_steps, dtype=np.float64)

        # Add history tracks for Chiller impurity (already tracked for cooling load/power)
        for chiller in self._chillers:
            cid = chiller.component_id
            self._history[f"{cid}_outlet_o2_ppm_mol"] = np.zeros(total_steps, dtype=np.float64)
            # Outlet Stream Properties (for Stacked Properties Graph)
            self._history[f"{cid}_outlet_temp_c"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_outlet_pressure_bar"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_outlet_h2o_frac"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_outlet_enthalpy_kj_kg"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_outlet_mass_flow_kg_h"] = np.zeros(total_steps, dtype=np.float64)

        # Add history tracks for ElectricBoiler
        for boiler in self._boilers:
            cid = boiler.component_id
            self._history[f"{cid}_power_input_kw"] = np.zeros(total_steps, dtype=np.float64)
            # Outlet Stream Properties (for Stacked Properties Graph)
            self._history[f"{cid}_outlet_temp_c"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_outlet_pressure_bar"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_outlet_h2o_frac"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_outlet_enthalpy_kj_kg"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_outlet_mass_flow_kg_h"] = np.zeros(total_steps, dtype=np.float64)

        # Add history track for PEM impurity if present
        if self._pem:
            # Using generic name 'PEM' + suffix, or component id
            # Graph plotter looks for suffix '_o2_impurity_ppm_mol'
            cid = self._pem.component_id if hasattr(self._pem, 'component_id') else 'PEM'
            cid = self._pem.component_id if hasattr(self._pem, 'component_id') else 'PEM'
            self._history[f"{cid}_o2_impurity_ppm_mol"] = np.zeros(total_steps, dtype=np.float64)

        # Monitor SOEC individual modules
        if self._soec:
            num_modules = getattr(self._soec, 'num_modules', 0)
            cid = self._soec.component_id if hasattr(self._soec, 'component_id') else 'SOEC_Cluster'
            if cid:
                self._history[f"{cid}_outlet_mass_flow_kg_h"] = np.zeros(total_steps, dtype=np.float64)
                self._history[f"{cid}_outlet_temp_c"] = np.zeros(total_steps, dtype=np.float64)
                self._history[f"{cid}_outlet_pressure_bar"] = np.zeros(total_steps, dtype=np.float64)
                self._history[f"{cid}_outlet_h2o_frac"] = np.zeros(total_steps, dtype=np.float64)
                self._history[f"{cid}_outlet_enthalpy_kj_kg"] = np.zeros(total_steps, dtype=np.float64)
            
            for i in range(num_modules):
                self._history[f"soec_module_powers_{i+1}"] = np.zeros(total_steps, dtype=np.float64)

        self._state = IntegratedDispatchState()

        logger.info(f"Initialized HybridArbitrageEngineStrategy with {total_steps} pre-allocated steps")

    def decide_and_apply(self, t: float, prices: np.ndarray, wind: np.ndarray) -> None:
        """
        Make dispatch decision and apply setpoints to components.

        Sets power inputs on electrolyzers via receive_input(). Does NOT
        call step() - that is the engine's responsibility.

        Args:
            t (float): Current simulation time in hours.
            prices (np.ndarray): Full simulation price array (EUR/MWh).
            wind (np.ndarray): Full simulation wind power array (MW).
        """
        dt = self._context.simulation.timestep_hours
        step_idx = self._state.step_idx

        if step_idx >= self._total_steps:
            logger.warning(f"Step index {step_idx} exceeds pre-allocated size {self._total_steps}")
            return

        # Extract current values
        minute = int(round(t * 60))
        P_offer = wind[step_idx]
        current_price = prices[step_idx]
        P_future = wind[min(step_idx + 60, len(wind) - 1)]

        # Build dispatch input
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
            ppa_price_eur_mwh=getattr(self._context.economics, 'ppa_price_eur_mwh', 50.0),
            h2_price_eur_kg=getattr(self._context.economics, 'h2_price_eur_kg', 9.6),
            arbitrage_threshold_eur_mwh=getattr(self._context.economics, 'arbitrage_threshold_eur_mwh', None)
        )

        d_state = DispatchState(
            P_soec_prev=self._state.P_soec_prev,
            force_sell=self._state.force_sell
        )

        # Get dispatch decision
        result = self._inner_strategy.decide(d_input, d_state)

        # Update internal state
        self._state.force_sell = result.state_update.get('force_sell', False)

        # Apply setpoints to components (no step() calls)
        if self._soec:
            self._soec.receive_input('power_in', result.P_soec, 'electricity')

        if self._pem:
            # CRITICAL: Supply water to PEM before stepping
            # PEM requires water in its buffer for electrolysis (water starves otherwise)
            # This is a workaround until topology files include WaterSupply nodes
            from h2_plant.core.stream import Stream
            water_stream = Stream(
                mass_flow_kg_h=10000.0,  # Ample water supply (infinite source)
                temperature_k=298.15,
                pressure_pa=5e5,
                composition={'H2O': 1.0},
                phase='liquid'
            )
            self._pem.receive_input('water_in', water_stream, 'water')
            self._pem.set_power_input_mw(result.P_pem)

        # Record pre-step data
        self._history['minute'][step_idx] = minute
        self._history['P_offer'][step_idx] = P_offer
        self._history['spot_price'][step_idx] = current_price

        self._state.step_idx = step_idx

    def record_post_step(self) -> None:
        """
        Record component outputs after step() execution.

        Called by SimulationEngine after all components have stepped.
        Captures actual physics results from component state.
        """
        step_idx = self._state.step_idx
        dt = self._context.simulation.timestep_hours

        if step_idx >= self._total_steps:
            return

        # Extract SOEC results
        P_soec_actual = 0.0
        h2_soec = 0.0
        steam_soec = 0.0

        if self._soec:
            if hasattr(self._soec, 'real_powers'):
                P_soec_actual = float(np.sum(self._soec.real_powers))

            # Check multiple possible attribute names for H2 output
            if hasattr(self._soec, 'last_step_h2_kg'):
                h2_soec = self._soec.last_step_h2_kg
            elif hasattr(self._soec, 'last_h2_output_kg'):
                h2_soec = self._soec.last_h2_output_kg
            elif hasattr(self._soec, 'h2_output_kg'):
                h2_soec = self._soec.h2_output_kg

            steam_soec = getattr(self._soec, 'last_step_steam_input_kg', 0.0)

            # Record SOEC outlet flow (constructed on demand)
            cid = self._soec.component_id if hasattr(self._soec, 'component_id') else 'SOEC_Cluster'
            if cid:
                try:
                    out_stream = self._soec.get_output('h2_out')
                    if out_stream:
                        self._history[f"{cid}_outlet_mass_flow_kg_h"][step_idx] = out_stream.mass_flow_kg_h
                        self._history[f"{cid}_outlet_temp_c"][step_idx] = out_stream.temperature_k - 273.15
                        self._history[f"{cid}_outlet_pressure_bar"][step_idx] = out_stream.pressure_pa / 1e5
                        self._history[f"{cid}_outlet_h2o_frac"][step_idx] = out_stream.composition.get('H2O', 0.0) + out_stream.composition.get('H2O_liq', 0.0)
                        self._history[f"{cid}_outlet_enthalpy_kj_kg"][step_idx] = out_stream.specific_enthalpy_j_kg / 1000.0
                except Exception:
                    pass

        self._state.P_soec_prev = P_soec_actual

        # Extract PEM results
        h2_pem = 0.0
        P_pem_actual = 0.0

        if self._pem:
            h2_pem = getattr(self._pem, 'h2_output_kg', 0.0)
            if hasattr(self._pem, 'P_consumed_W'):
                P_pem_actual = self._pem.P_consumed_W / 1e6

        # Aggregate BoP power consumption
        P_bop_kw = 0.0
        for comp_id, comp in self._registry.list_components():
            if hasattr(comp, 'power_kw'):
                P_bop_kw += comp.power_kw
            if hasattr(comp, 'electrical_power_kw'):
                P_bop_kw += comp.electrical_power_kw

        P_bop_mw = P_bop_kw / 1000.0

        # Calculate corrected sold power
        P_total_consumed = P_soec_actual + P_pem_actual + P_bop_mw
        P_offer = self._history['P_offer'][step_idx]
        P_sold_corrected = max(0.0, P_offer - P_total_consumed)

        total_h2 = h2_soec + h2_pem
        self._state.cumulative_h2_kg += total_h2

        # Record to pre-allocated arrays
        self._history['P_soec_actual'][step_idx] = P_soec_actual
        self._history['P_pem'][step_idx] = P_pem_actual
        self._history['P_sold'][step_idx] = P_sold_corrected
        self._history['h2_kg'][step_idx] = total_h2
        self._history['H2_soec_kg'][step_idx] = h2_soec
        self._history['H2_pem_kg'][step_idx] = h2_pem
        self._history['cumulative_h2_kg'][step_idx] = self._state.cumulative_h2_kg
        self._history['steam_soec_kg'][step_idx] = steam_soec
        self._history['P_bop_mw'][step_idx] = P_bop_mw
        self._history['sell_decision'][step_idx] = 1 if P_sold_corrected > 0 else 0

        # SOEC active modules
        if self._soec and hasattr(self._soec, 'real_powers'):
            self._history['soec_active_modules'][step_idx] = int(np.sum(self._soec.real_powers > 0.01))
            
            # Record individual module powers for Wear Graph
            for i, power_mw in enumerate(self._soec.real_powers):
                # We use 1-based indexing for user-facing labels
                key = f"soec_module_powers_{i+1}"
                if key in self._history:
                    self._history[key][step_idx] = power_mw

        h2o_soec_out = getattr(self._soec, 'last_water_output_kg', 0.0) if self._soec else 0.0
        self._history['H2O_soec_out_kg'][step_idx] = h2o_soec_out

        # PEM details
        h2o_pem = getattr(self._pem, 'water_consumption_kg', h2_pem * 9.0 * 1.02) if self._pem else h2_pem * 9.0 * 1.02
        self._history['H2O_pem_kg'][step_idx] = h2o_pem
        self._history['O2_pem_kg'][step_idx] = h2_pem * 8.0
        self._history['pem_V_cell'][step_idx] = getattr(self._pem, 'V_cell', 0.0) if self._pem else 0.0

        # Legacy tank/compressor compatibility
        tank_main = self._registry.get("H2_Tank") if self._registry.has("H2_Tank") else None
        comp_main = self._registry.get("H2_Compressor") if self._registry.has("H2_Compressor") else None

        if tank_main:
            self._history['tank_level_kg'][step_idx] = getattr(tank_main, 'current_level_kg', 0.0)
            self._history['tank_pressure_bar'][step_idx] = getattr(tank_main, 'pressure_bar', 0.0)
        else:
             self._history['tank_level_kg'][step_idx] = 0.0
             self._history['tank_pressure_bar'][step_idx] = 0.0
        
        # Aggregate compressor power
        total_comp_power = 0.0
        if comp_main:
            total_comp_power = getattr(comp_main, 'power_kw', 0.0)
        else:
            for comp in self._compressors:
                 total_comp_power += getattr(comp, 'power_kw', 0.0)

        self._history['compressor_power_kw'][step_idx] = total_comp_power
        
        # Record Chiller states
        for chiller in self._chillers:
            cid = chiller.component_id
            state = chiller.get_state()
            self._history[f"{cid}_cooling_load_kw"][step_idx] = state.get('cooling_load_kw', 0.0)
            self._history[f"{cid}_electrical_power_kw"][step_idx] = state.get('electrical_power_kw', 0.0)
            # Stacked properties
            out_stream = getattr(chiller, 'outlet_stream', None)
            if out_stream:
                self._history[f"{cid}_outlet_temp_c"][step_idx] = out_stream.temperature_k - 273.15
                self._history[f"{cid}_outlet_pressure_bar"][step_idx] = out_stream.pressure_pa / 1e5
                self._history[f"{cid}_outlet_h2o_frac"][step_idx] = out_stream.composition.get('H2O', 0.0) + out_stream.composition.get('H2O_liq', 0.0)
                self._history[f"{cid}_outlet_enthalpy_kj_kg"][step_idx] = out_stream.specific_enthalpy_j_kg / 1000.0
                self._history[f"{cid}_outlet_mass_flow_kg_h"][step_idx] = out_stream.mass_flow_kg_h

        # Record Coalescer states
        for coal in self._coalescers:
            cid = coal.component_id
            state = coal.get_state()
            self._history[f"{cid}_delta_p_bar"][step_idx] = state.get('delta_p_bar', 0.0)
            self._history[f"{cid}_drain_flow_kg_h"][step_idx] = state.get('drain_flow_kg_h', 0.0)
            self._history[f"{cid}_dissolved_gas_ppm"][step_idx] = state.get('dissolved_gas_ppm', 0.0)
            self._history[f"{cid}_dissolved_gas_in_kg_h"][step_idx] = state.get('dissolved_gas_in_kg_h', 0.0)
            self._history[f"{cid}_dissolved_gas_out_kg_h"][step_idx] = state.get('dissolved_gas_out_kg_h', 0.0)
            
            # Stacked properties (Coalescer usually has output_stream)
            out_stream = getattr(coal, 'output_stream', None)
            if out_stream:
                self._history[f"{cid}_outlet_temp_c"][step_idx] = out_stream.temperature_k - 273.15
                self._history[f"{cid}_outlet_pressure_bar"][step_idx] = out_stream.pressure_pa / 1e5
                self._history[f"{cid}_outlet_h2o_frac"][step_idx] = out_stream.composition.get('H2O', 0.0) + out_stream.composition.get('H2O_liq', 0.0)
                self._history[f"{cid}_outlet_enthalpy_kj_kg"][step_idx] = out_stream.specific_enthalpy_j_kg / 1000.0
                self._history[f"{cid}_outlet_mass_flow_kg_h"][step_idx] = out_stream.mass_flow_kg_h

        # Record Deoxo states
        for deoxo in self._deoxos:
            cid = deoxo.component_id
            state = deoxo.get_state()
            self._history[f"{cid}_outlet_o2_ppm_mol"][step_idx] = state.get('outlet_o2_ppm_mol', 0.0)
            # Deoxo specific
            self._history[f"{cid}_inlet_temp_c"][step_idx] = state.get('inlet_temp_c', 0.0)
            self._history[f"{cid}_inlet_pressure_bar"][step_idx] = state.get('inlet_pressure_bar', 0.0)
            self._history[f"{cid}_o2_in_kg_h"][step_idx] = state.get('o2_in_kg_h', 0.0)
            self._history[f"{cid}_peak_temp_c"][step_idx] = state.get('peak_temp_c', 0.0)
            self._history[f"{cid}_conversion_percent"][step_idx] = state.get('conversion_percent', 0.0)
            self._history[f"{cid}_mass_flow_kg_h"][step_idx] = state.get('mass_flow_kg_h', 0.0)
            
            # Stacked properties (Deoxo uses output_stream)
            out_stream = getattr(deoxo, 'output_stream', None)
            if out_stream:
                self._history[f"{cid}_outlet_temp_c"][step_idx] = out_stream.temperature_k - 273.15
                self._history[f"{cid}_outlet_pressure_bar"][step_idx] = out_stream.pressure_pa / 1e5
                self._history[f"{cid}_outlet_h2o_frac"][step_idx] = out_stream.composition.get('H2O', 0.0) + out_stream.composition.get('H2O_liq', 0.0)
                self._history[f"{cid}_outlet_enthalpy_kj_kg"][step_idx] = out_stream.specific_enthalpy_j_kg / 1000.0
                self._history[f"{cid}_outlet_mass_flow_kg_h"][step_idx] = out_stream.mass_flow_kg_h

        # Record PSA states
        for psa in self._psas:
            cid = psa.component_id
            state = psa.get_state()
            self._history[f"{cid}_outlet_o2_ppm_mol"][step_idx] = state.get('outlet_o2_ppm_mol', 0.0)
            
            # Stacked properties (PSA uses 'purified_gas_out' port)
            out_stream = psa.get_output('purified_gas_out')
            if out_stream:
                self._history[f"{cid}_outlet_temp_c"][step_idx] = out_stream.temperature_k - 273.15
                self._history[f"{cid}_outlet_pressure_bar"][step_idx] = out_stream.pressure_pa / 1e5
                self._history[f"{cid}_outlet_h2o_frac"][step_idx] = out_stream.composition.get('H2O', 0.0) + out_stream.composition.get('H2O_liq', 0.0)
                self._history[f"{cid}_outlet_enthalpy_kj_kg"][step_idx] = out_stream.specific_enthalpy_j_kg / 1000.0
                self._history[f"{cid}_outlet_mass_flow_kg_h"][step_idx] = out_stream.mass_flow_kg_h
        # Record KOD states
        for kod in self._kods:
            cid = kod.component_id
            state = kod.get_state()
            self._history[f"{cid}_outlet_o2_ppm_mol"][step_idx] = state.get('outlet_o2_ppm_mol', 0.0)
            # KOD specific
            self._history[f"{cid}_water_removed_kg_h"][step_idx] = state.get('water_removed_kg_h', 0.0)
            self._history[f"{cid}_drain_temp_k"][step_idx] = state.get('drain_temp_k', 0.0)
            self._history[f"{cid}_drain_pressure_bar"][step_idx] = state.get('drain_pressure_bar', 0.0)
            self._history[f"{cid}_dissolved_gas_ppm"][step_idx] = state.get('dissolved_gas_ppm', 0.0)
            self._history[f"{cid}_m_dot_H2O_liq_accomp_kg_s"][step_idx] = state.get('m_dot_H2O_liq_accomp_kg_s', 0.0)
            self._history[f"{cid}_dissolved_gas_in_kg_h"][step_idx] = state.get('dissolved_gas_in_kg_h', 0.0)
            self._history[f"{cid}_dissolved_gas_out_kg_h"][step_idx] = state.get('dissolved_gas_out_kg_h', 0.0)

            # Stacked properties (KOD usually has outlet_stream)
            out_stream = getattr(kod, '_gas_outlet_stream', None) # Note: KOD uses specific internal name
            if not out_stream: out_stream = kod._gas_outlet_stream
            
            # KOD might use different internal stream name? 
            # In KOD code: self._gas_outlet_stream
            if out_stream:
                self._history[f"{cid}_outlet_temp_c"][step_idx] = out_stream.temperature_k - 273.15
                self._history[f"{cid}_outlet_pressure_bar"][step_idx] = out_stream.pressure_pa / 1e5
                self._history[f"{cid}_outlet_h2o_frac"][step_idx] = out_stream.composition.get('H2O', 0.0) + out_stream.composition.get('H2O_liq', 0.0)
                self._history[f"{cid}_outlet_enthalpy_kj_kg"][step_idx] = out_stream.specific_enthalpy_j_kg / 1000.0
                self._history[f"{cid}_outlet_mass_flow_kg_h"][step_idx] = out_stream.mass_flow_kg_h

        # Record Cyclone states
        for cyc in self._cyclones:
            cid = cyc.component_id
            state = cyc.get_state()
            self._history[f"{cid}_outlet_o2_ppm_mol"][step_idx] = state.get('outlet_o2_ppm_mol', 0.0)
            # Cyclone specific
            self._history[f"{cid}_water_removed_kg_h"][step_idx] = state.get('water_removed_kg_h', 0.0)
            self._history[f"{cid}_drain_temp_k"][step_idx] = state.get('drain_temp_k', 0.0)
            self._history[f"{cid}_drain_pressure_bar"][step_idx] = state.get('drain_pressure_bar', 0.0)
            self._history[f"{cid}_dissolved_gas_ppm"][step_idx] = state.get('dissolved_gas_ppm', 0.0)
            self._history[f"{cid}_dissolved_gas_in_kg_h"][step_idx] = state.get('dissolved_gas_in_kg_h', 0.0)
            self._history[f"{cid}_dissolved_gas_out_kg_h"][step_idx] = state.get('dissolved_gas_out_kg_h', 0.0)
            self._history[f"{cid}_pressure_drop_mbar"][step_idx] = state.get('pressure_drop_mbar', 0.0)
            
            # Stacked properties (Cyclone usually has outlet_stream or _outlet_stream)
            out_stream = getattr(cyc, '_outlet_stream', None) 
            # In Cyclone code: self._outlet_stream
            if out_stream:
                self._history[f"{cid}_outlet_temp_c"][step_idx] = out_stream.temperature_k - 273.15
                self._history[f"{cid}_outlet_pressure_bar"][step_idx] = out_stream.pressure_pa / 1e5
                self._history[f"{cid}_outlet_h2o_frac"][step_idx] = out_stream.composition.get('H2O', 0.0) + out_stream.composition.get('H2O_liq', 0.0)
                self._history[f"{cid}_outlet_enthalpy_kj_kg"][step_idx] = out_stream.specific_enthalpy_j_kg / 1000.0
                self._history[f"{cid}_outlet_mass_flow_kg_h"][step_idx] = out_stream.mass_flow_kg_h

        # Record DrainRecorderMixer states
        for mixer in self._drain_mixers:
            cid = mixer.component_id
            state = mixer.get_state()
            self._history[f"{cid}_dissolved_gas_ppm"][step_idx] = state.get('dissolved_gas_ppm', 0.0)
            self._history[f"{cid}_outlet_mass_flow_kg_h"][step_idx] = state.get('outlet_mass_flow_kg_h', 0.0)
            self._history[f"{cid}_outlet_temperature_c"][step_idx] = state.get('outlet_temperature_c', 0.0)
            self._history[f"{cid}_outlet_pressure_kpa"][step_idx] = state.get('outlet_pressure_kpa', 0.0)

        # Record Compressor states
        for comp in self._compressors:
            cid = comp.component_id
            state = comp.get_state()
            self._history[f"{cid}_outlet_o2_ppm_mol"][step_idx] = state.get('outlet_o2_ppm_mol', 0.0)
            # Compressor specific
            self._history[f"{cid}_power_kw"][step_idx] = state.get('power_kw', 0.0)
            self._history[f"{cid}_outlet_temp_c"][step_idx] = state.get('outlet_temperature_c', 0.0)
            self._history[f"{cid}_outlet_pressure_bar"][step_idx] = state.get('outlet_pressure_bar', 0.0)
            
            # Stacked properties (Compressor constructs output on demand via get_output)
            out_stream = comp.get_output('outlet')
            if out_stream:
                self._history[f"{cid}_outlet_h2o_frac"][step_idx] = out_stream.composition.get('H2O', 0.0) + out_stream.composition.get('H2O_liq', 0.0)
                self._history[f"{cid}_outlet_enthalpy_kj_kg"][step_idx] = out_stream.specific_enthalpy_j_kg / 1000.0
                self._history[f"{cid}_outlet_mass_flow_kg_h"][step_idx] = out_stream.mass_flow_kg_h

        # Record Coalescer impurity (added later so manual update here)
        for coal in self._coalescers:
            cid = coal.component_id
            state = coal.get_state()
            self._history[f"{cid}_outlet_o2_ppm_mol"][step_idx] = state.get('outlet_o2_ppm_mol', 0.0)

        # Record DryCooler impurity
        for dc in self._dry_coolers:
            cid = dc.component_id
            state = dc.get_state()
            self._history[f"{cid}_outlet_o2_ppm_mol"][step_idx] = state.get('outlet_o2_ppm_mol', 0.0)
            # DryCooler specific
            self._history[f"{cid}_heat_rejected_kw"][step_idx] = state.get('heat_rejected_kw', 0.0)
            self._history[f"{cid}_tqc_duty_kw"][step_idx] = state.get('tqc_duty_kw', 0.0)
            self._history[f"{cid}_dc_duty_kw"][step_idx] = state.get('dc_duty_kw', 0.0)
            self._history[f"{cid}_fan_power_kw"][step_idx] = state.get('fan_power_kw', 0.0)
            self._history[f"{cid}_outlet_temp_c"][step_idx] = state.get('outlet_temp_c', 0.0)
            
            # Stacked properties (DryCooler uses outlet_stream)
            out_stream = getattr(dc, 'outlet_stream', None)
            if out_stream:
                self._history[f"{cid}_outlet_pressure_bar"][step_idx] = out_stream.pressure_pa / 1e5
                self._history[f"{cid}_outlet_h2o_frac"][step_idx] = out_stream.composition.get('H2O', 0.0) + out_stream.composition.get('H2O_liq', 0.0)
                self._history[f"{cid}_outlet_enthalpy_kj_kg"][step_idx] = out_stream.specific_enthalpy_j_kg / 1000.0
                self._history[f"{cid}_outlet_mass_flow_kg_h"][step_idx] = out_stream.mass_flow_kg_h

        # Record HeatExchanger impurity
        for hx in self._heat_exchangers:
            cid = hx.component_id
            state = hx.get_state()
            self._history[f"{cid}_outlet_o2_ppm_mol"][step_idx] = state.get('outlet_o2_ppm_mol', 0.0)
            # HX specific
            self._history[f"{cid}_heat_removed_kw"][step_idx] = state.get('heat_removed_kw', 0.0)
            
            # Stacked properties
            out_stream = getattr(hx, 'output_stream', None)
            if out_stream:
                self._history[f"{cid}_outlet_temp_c"][step_idx] = out_stream.temperature_k - 273.15
                self._history[f"{cid}_outlet_pressure_bar"][step_idx] = out_stream.pressure_pa / 1e5
                self._history[f"{cid}_outlet_h2o_frac"][step_idx] = out_stream.composition.get('H2O', 0.0) + out_stream.composition.get('H2O_liq', 0.0)
                self._history[f"{cid}_outlet_enthalpy_kj_kg"][step_idx] = out_stream.specific_enthalpy_j_kg / 1000.0
                self._history[f"{cid}_outlet_mass_flow_kg_h"][step_idx] = out_stream.mass_flow_kg_h

        # Record Chiller impurity
        for chiller in self._chillers:
            cid = chiller.component_id
            state = chiller.get_state()
            self._history[f"{cid}_outlet_o2_ppm_mol"][step_idx] = state.get('outlet_o2_ppm_mol', 0.0)

            # Stacked properties (Chiller uses output_stream usually, or fluid_out but stored as out_stream attribute?)
            # Chiller component uses self.output_stream in get_state? No, it exposes keys.
            out_stream = getattr(chiller, 'output_stream', getattr(chiller, '_output_stream', None))
            if out_stream:
                self._history[f"{cid}_outlet_temp_c"][step_idx] = out_stream.temperature_k - 273.15
                self._history[f"{cid}_outlet_pressure_bar"][step_idx] = out_stream.pressure_pa / 1e5
                self._history[f"{cid}_outlet_h2o_frac"][step_idx] = out_stream.composition.get('H2O', 0.0) + out_stream.composition.get('H2O_liq', 0.0)
                self._history[f"{cid}_outlet_enthalpy_kj_kg"][step_idx] = out_stream.specific_enthalpy_j_kg / 1000.0
                self._history[f"{cid}_outlet_mass_flow_kg_h"][step_idx] = out_stream.mass_flow_kg_h

        # Record Interchanger
        for ic in self._interchangers:
            cid = ic.component_id
            state = ic.get_state()
            self._history[f"{cid}_q_transferred_kw"][step_idx] = state.get('q_transferred_kw', 0.0)
            self._history[f"{cid}_hot_out_temp_k"][step_idx] = state.get('hot_out_temp_k', 0.0)
            self._history[f"{cid}_cold_out_temp_k"][step_idx] = state.get('cold_out_temp_k', 0.0)
            
            # Use hot_out as primary outlet for profile graph
            out_stream = ic.get_output('hot_out')
            if out_stream:
                self._history[f"{cid}_outlet_temp_c"][step_idx] = out_stream.temperature_k - 273.15
                self._history[f"{cid}_outlet_pressure_bar"][step_idx] = out_stream.pressure_pa / 1e5
                self._history[f"{cid}_outlet_h2o_frac"][step_idx] = out_stream.composition.get('H2O', 0.0) + out_stream.composition.get('H2O_liq', 0.0)
                self._history[f"{cid}_outlet_enthalpy_kj_kg"][step_idx] = out_stream.specific_enthalpy_j_kg / 1000.0
                self._history[f"{cid}_outlet_mass_flow_kg_h"][step_idx] = out_stream.mass_flow_kg_h

        # Record ElectricBoiler states
        for boiler in self._boilers:
            cid = boiler.component_id
            state = boiler.get_state()
            self._history[f"{cid}_power_input_kw"][step_idx] = state.get('power_input_kw', 0.0)
            
            # Stacked properties (ElectricBoiler uses _output_stream, accessible via get_output)
            out_stream = boiler.get_output('fluid_out')
            if out_stream:
                self._history[f"{cid}_outlet_temp_c"][step_idx] = out_stream.temperature_k - 273.15
                self._history[f"{cid}_outlet_pressure_bar"][step_idx] = out_stream.pressure_pa / 1e5
                self._history[f"{cid}_outlet_h2o_frac"][step_idx] = out_stream.composition.get('H2O', 0.0) + out_stream.composition.get('H2O_liq', 0.0)
                self._history[f"{cid}_outlet_enthalpy_kj_kg"][step_idx] = out_stream.specific_enthalpy_j_kg / 1000.0
                self._history[f"{cid}_outlet_mass_flow_kg_h"][step_idx] = out_stream.mass_flow_kg_h

        # Record PEM impurity
        if self._pem:
            cid = self._pem.component_id if hasattr(self._pem, 'component_id') else 'PEM'
            state = self._pem.get_state()
            # Default to 0 if not yet exposed (handled by get() returning None/default)
            self._history[f"{cid}_o2_impurity_ppm_mol"][step_idx] = state.get('o2_impurity_ppm_mol', 0.0)

        self._state.step_idx += 1

    def get_history(self) -> Dict[str, np.ndarray]:
        """
        Return recorded history as NumPy arrays.

        Returns:
            Dict[str, np.ndarray]: History arrays trimmed to actual steps.
        """
        actual_steps = self._state.step_idx
        return {k: v[:actual_steps] for k, v in self._history.items()}

    def _find_soec(self, registry: 'ComponentRegistry'):
        """
        Find SOEC component in registry.

        Args:
            registry (ComponentRegistry): Registry to search.

        Returns:
            Component or None: SOEC component if found.
        """
        for comp_id, comp in registry.list_components():
            class_name = comp.__class__.__name__
            if hasattr(comp, 'soec_state') or class_name == 'SOECOperator':
                return comp
        return None

    def _find_pem(self, registry: 'ComponentRegistry'):
        """
        Find PEM component in registry.

        Args:
            registry (ComponentRegistry): Registry to search.

        Returns:
            Component or None: PEM component if found.
        """
        for comp_id, comp in registry.list_components():
            class_name = comp.__class__.__name__
            if hasattr(comp, 'V_cell') or class_name == 'DetailedPEMElectrolyzer':
                return comp
        return None

    def print_summary(self):
        """
        Print a unified stream summary table grouped by topology sections.
        Delegates to reporting.stream_table.print_stream_summary_table.
        """
        # Collect all components from registry directly
        # This avoids AttributeError for missing list collections (e.g. self._electrolyzers)
        if self._registry:
            components = {cid: comp for cid, comp in self._registry.list_components()}
        else:
            components = {}

        # Get topology order
        topo_order = self._get_topology_order()
        
        # New imported logic
        from h2_plant.reporting.stream_table import print_stream_summary_table
        print_stream_summary_table(components, topo_order)

    def _get_topology_order(self) -> list:
        """
        Traverse topology connections to get components in flow order.
        
        Returns list of component IDs in the order they appear in the
        process flow (from source to sink).
        """
        # Get topology from context
        if not hasattr(self._context, 'topology') or self._context.topology is None:
            return []
        
        nodes = self._context.topology.nodes
        if not nodes:
            return []
        
        # Build adjacency map: source_id -> target_id
        adjacency = {}
        all_targets = set()
        
        for node in nodes:
            comp_id = node.id
            if node.connections:
                for conn in node.connections:
                    target_id = conn.target_name
                    adjacency[comp_id] = target_id
                    all_targets.add(target_id)
        
        # Find the root (source) - a node that is not a target of any other
        all_sources = set(node.id for node in nodes)
        roots = all_sources - all_targets
        
        if not roots:
            # Fallback: use first node
            roots = {nodes[0].id}
        
        # Traverse from root following connections
        ordered = []
        visited = set()
        
        for root in roots:
            current = root
            while current and current not in visited:
                ordered.append(current)
                visited.add(current)
                current = adjacency.get(current)
        
        # Add any remaining nodes not in the chain
        for node in nodes:
            if node.id not in visited:
                ordered.append(node.id)
        
        return ordered



