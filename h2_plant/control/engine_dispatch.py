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
from h2_plant.components.compression.compressor_single import CompressorSingle
from h2_plant.components.cooling.dry_cooler import DryCooler
from h2_plant.components.thermal.heat_exchanger import HeatExchanger

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

        for coal in self._coalescers:
            cid = coal.component_id
            self._history[f"{cid}_delta_p_bar"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_drain_flow_kg_h"] = np.zeros(total_steps, dtype=np.float64)
            self._history[f"{cid}_outlet_o2_ppm_mol"] = np.zeros(total_steps, dtype=np.float64)

        # Add history tracks for each deoxo
        for deoxo in self._deoxos:
            cid = deoxo.component_id
            self._history[f"{cid}_outlet_o2_ppm_mol"] = np.zeros(total_steps, dtype=np.float64)

        # Add history tracks for PSA
        for psa in self._psas:
            cid = psa.component_id
            self._history[f"{cid}_outlet_o2_ppm_mol"] = np.zeros(total_steps, dtype=np.float64)

        # Add history tracks for KOD
        for kod in self._kods:
            cid = kod.component_id
            self._history[f"{cid}_outlet_o2_ppm_mol"] = np.zeros(total_steps, dtype=np.float64)

        # Add history tracks for Compressor
        for comp in self._compressors:
            cid = comp.component_id
            self._history[f"{cid}_outlet_o2_ppm_mol"] = np.zeros(total_steps, dtype=np.float64)

        # Add history tracks for DryCooler
        for dc in self._dry_coolers:
            cid = dc.component_id
            self._history[f"{cid}_outlet_o2_ppm_mol"] = np.zeros(total_steps, dtype=np.float64)

        # Add history tracks for HeatExchanger
        for hx in self._heat_exchangers:
            cid = hx.component_id
            self._history[f"{cid}_outlet_o2_ppm_mol"] = np.zeros(total_steps, dtype=np.float64)

        # Add history tracks for Chiller impurity (already tracked for cooling load/power)
        for chiller in self._chillers:
            cid = chiller.component_id
            self._history[f"{cid}_outlet_o2_ppm_mol"] = np.zeros(total_steps, dtype=np.float64)

        # Add history track for PEM impurity if present
        if self._pem:
            # Using generic name 'PEM' + suffix, or component id
            # Graph plotter looks for suffix '_o2_impurity_ppm_mol'
            cid = self._pem.component_id if hasattr(self._pem, 'component_id') else 'PEM'
            self._history[f"{cid}_o2_impurity_ppm_mol"] = np.zeros(total_steps, dtype=np.float64)

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

            steam_soec = getattr(self._soec, 'last_steam_output_kg', 0.0)

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

        self._history['tank_level_kg'][step_idx] = getattr(tank_main, 'current_level_kg', 0.0) if tank_main else 0.0
        self._history['tank_pressure_bar'][step_idx] = getattr(tank_main, 'pressure_bar', 0.0) if tank_main else 0.0
        self._history['compressor_power_kw'][step_idx] = getattr(comp_main, 'power_kw', 0.0) if comp_main else 0.0
        
        # Record Chiller states
        for chiller in self._chillers:
            cid = chiller.component_id
            state = chiller.get_state()
            self._history[f"{cid}_cooling_load_kw"][step_idx] = state.get('cooling_load_kw', 0.0)
            self._history[f"{cid}_electrical_power_kw"][step_idx] = state.get('electrical_power_kw', 0.0)

        # Record Coalescer states
        for coal in self._coalescers:
            cid = coal.component_id
            state = coal.get_state()
            self._history[f"{cid}_delta_p_bar"][step_idx] = state.get('delta_p_bar', 0.0)
            self._history[f"{cid}_drain_flow_kg_h"][step_idx] = state.get('drain_flow_kg_h', 0.0)

        # Record Deoxo states
        for deoxo in self._deoxos:
            cid = deoxo.component_id
            state = deoxo.get_state()
            self._history[f"{cid}_outlet_o2_ppm_mol"][step_idx] = state.get('outlet_o2_ppm_mol', 0.0)

        # Record PSA states
        for psa in self._psas:
            cid = psa.component_id
            state = psa.get_state()
            self._history[f"{cid}_outlet_o2_ppm_mol"][step_idx] = state.get('outlet_o2_ppm_mol', 0.0)
            
        # Record KOD states
        for kod in self._kods:
            cid = kod.component_id
            state = kod.get_state()
            self._history[f"{cid}_outlet_o2_ppm_mol"][step_idx] = state.get('outlet_o2_ppm_mol', 0.0)

        # Record Compressor states
        for comp in self._compressors:
            cid = comp.component_id
            state = comp.get_state()
            self._history[f"{cid}_outlet_o2_ppm_mol"][step_idx] = state.get('outlet_o2_ppm_mol', 0.0)

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

        # Record HeatExchanger impurity
        for hx in self._heat_exchangers:
            cid = hx.component_id
            state = hx.get_state()
            self._history[f"{cid}_outlet_o2_ppm_mol"][step_idx] = state.get('outlet_o2_ppm_mol', 0.0)

        # Record Chiller impurity
        for chiller in self._chillers:
            cid = chiller.component_id
            state = chiller.get_state()
            self._history[f"{cid}_outlet_o2_ppm_mol"][step_idx] = state.get('outlet_o2_ppm_mol', 0.0)

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

    def print_summary(self) -> None:
        """
        Print topology-aware simulation summary.
        
        Dynamically detects which components are present in the registry
        and displays only relevant metrics for the current topology.
        """
        dt = self._context.simulation.timestep_hours
        hist = self.get_history()
        actual_steps = self._state.step_idx
        duration_hours = actual_steps * dt
        
        # Component detection by class name patterns
        components_by_type = self._categorize_components()
        
        print("\n" + "=" * 70)
        print("## Simulation Summary")
        print(f"   Duration: {duration_hours:.1f} hours ({actual_steps} steps)")
        print("=" * 70)
        
        # === H2 Source Section ===
        if components_by_type.get('h2_source'):
            print("\n### H2 Source")
            for comp_id, comp in components_by_type['h2_source']:
                state = comp.get_state() if hasattr(comp, 'get_state') else {}
                flow_kg_h = state.get('mass_flow_kg_h', 0.0)
                total_h2 = state.get('cumulative_h2_kg', flow_kg_h * duration_hours)
                temp_k = state.get('temperature_k', 0.0)
                temp_c = temp_k - 273.15 if temp_k > 0 else 0.0
                pressure_bar = state.get('pressure_bar', state.get('pressure_pa', 0.0) / 1e5)
                h2_purity = state.get('h2_purity', 0.0) * 100
                h2o_impurity = state.get('h2o_impurity', 0.0) * 100
                
                print(f"   [{comp_id}]")
                print(f"   * Total H2 Supplied: {total_h2:,.2f} kg")
                print(f"   * Flow Rate: {flow_kg_h:.2f} kg/h")
                print(f"   * Conditions: {pressure_bar:.1f} bar, {temp_c:.1f}°C")
                print(f"   * H2 Purity: {h2_purity:.2f}% | H2O: {h2o_impurity:.2f}%")
                
                # Get output stream
                if hasattr(comp, 'get_output'):
                    try:
                        h2_out = comp.get_output('h2_out')
                        print(f"   ─── Output Stream ───")
                        # self._format_stream_properties(h2_out, "H2 Out", "   ")
                    except Exception:
                        pass
        
        # === Knock-Out Drum Section ===
        if components_by_type.get('knock_out_drum'):
            print("\n### Knock-Out Drums (Water Separation)")
            for comp_id, comp in components_by_type['knock_out_drum']:
                state = comp.get_state() if hasattr(comp, 'get_state') else {}
                water_removed = state.get('water_removed_kg_h', 0.0) * duration_hours
                status = state.get('separation_status', 'N/A')
                v_real = state.get('v_real', 0.0)
                v_max = state.get('v_max', 1.0)
                velocity_margin = ((v_max - v_real) / v_max * 100) if v_max > 0 else 0
                dissolved_gas = state.get('dissolved_gas_kg_h', 0.0) * duration_hours
                
                print(f"   [{comp_id}]")
                print(f"   * Water Removed (Total): {water_removed:,.4f} kg")
                print(f"   * Separation Status: {status}")
                print(f"   * Velocity Margin: {velocity_margin:.1f}% (V_real={v_real:.3f} < V_max={v_max:.3f} m/s)")
                print(f"   * Dissolved Gas Loss: {dissolved_gas:.6f} kg")
                
                # Get inlet/outlet streams if available
                inlet_stream = getattr(comp, '_input_stream', None)
                gas_outlet = getattr(comp, '_gas_outlet_stream', None)
                liquid_drain = getattr(comp, '_liquid_drain_stream', None)
                
                # Also try get_output methods
                if gas_outlet is None and hasattr(comp, 'get_output'):
                    try:
                        gas_outlet = comp.get_output('gas_outlet')
                    except Exception:
                        pass
                if liquid_drain is None and hasattr(comp, 'get_output'):
                    try:
                        liquid_drain = comp.get_output('liquid_drain')
                    except Exception:
                        pass
                
                # print(f"   ─── Stream Properties ───")
                # self._format_stream_properties(gas_outlet, "Gas Outlet", "   ")
                # self._format_stream_properties(liquid_drain, "Liquid Drain", "   ")
        
        # === Coalescer Section ===
        if components_by_type.get('coalescer'):
            print("\n### Coalescers (Aerosol Removal)")
            for comp_id, comp in components_by_type['coalescer']:
                state = comp.get_state() if hasattr(comp, 'get_state') else {}
                liquid_removed = state.get('liquid_removed_kg_h', state.get('drain_flow_kg_h', 0.0)) * duration_hours
                print(f"   [{comp_id}] Liquid Removed: {liquid_removed:,.4f} kg")
                
                # Get output stream (Coalescer uses 'outlet' port)
                gas_outlet = None
                if hasattr(comp, 'get_output'):
                    try:
                        gas_outlet = comp.get_output('outlet')
                    except Exception:
                        pass
                if gas_outlet:
                    # print(f"   ─── Gas Outlet ───")
                    # self._format_stream_properties(gas_outlet, "Gas Out", "   ")
                    pass
        
        # === Chiller Section ===
        if components_by_type.get('chiller'):
            print("\n### Chillers (Active Cooling)")
            for comp_id, comp in components_by_type['chiller']:
                state = comp.get_state() if hasattr(comp, 'get_state') else {}
                cooling_kw = state.get('cooling_load_kw', 0.0)
                power_kw = state.get('electrical_power_kw', 0.0)
                # Get outlet temp from state OR from outlet stream
                temp_out_c = state.get('outlet_temp_k', 0.0) - 273.15 if state.get('outlet_temp_k', 0.0) > 0 else 0.0
                if temp_out_c <= -273:
                    # Try 'outlet_temp_c' or get from stream
                    outlet_stream = getattr(comp, 'outlet_stream', None)
                    if outlet_stream and hasattr(outlet_stream, 'temperature_k'):
                        temp_out_c = outlet_stream.temperature_k - 273.15
                # Water condensed is stored directly in the component
                water_condensed_kg_h = getattr(comp, 'water_condensed_kg_h', 0.0)
                water_condensed = water_condensed_kg_h * duration_hours
                
                print(f"   [{comp_id}]")
                print(f"   * Cooling: {cooling_kw:.2f} kW | Elec. Power: {power_kw:.2f} kW")
                print(f"   * Outlet Temp: {temp_out_c:.1f}°C | Water Condensed: {water_condensed:.4f} kg")
                
                # Get output stream
                gas_outlet = None
                if hasattr(comp, 'get_output'):
                    try:
                        gas_outlet = comp.get_output('fluid_out')
                    except Exception:
                        try:
                            gas_outlet = comp.get_output('gas_outlet')
                        except Exception:
                            pass
                if gas_outlet:
                    # print(f"   ─── Gas Outlet ───")
                    # self._format_stream_properties(gas_outlet, "Gas Out", "   ")
                    pass
        
        # === Dry Cooler Section ===
        if components_by_type.get('dry_cooler'):
            print("\n### Dry Coolers (Passive Cooling)")
            for comp_id, comp in components_by_type['dry_cooler']:
                state = comp.get_state() if hasattr(comp, 'get_state') else {}
                # DryCooler returns 'tqc_duty_kw' and 'dc_duty_kw' for heat duties
                tqc_duty_kw = state.get('tqc_duty_kw', 0.0)
                dc_duty_kw = state.get('dc_duty_kw', 0.0)
                fan_power_kw = state.get('fan_power_kw', 0.0)
                # DryCooler returns 'outlet_temp_c' directly (not in Kelvin!)
                temp_out_c = state.get('outlet_temp_c', 0.0)
                
                print(f"   [{comp_id}] Heat Duty: {tqc_duty_kw:.2f} kW | Fan: {fan_power_kw:.2f} kW | T_out: {temp_out_c:.1f}°C")
                
                # Get output stream
                gas_outlet = None
                if hasattr(comp, 'get_output'):
                    try:
                        gas_outlet = comp.get_output('fluid_out')
                    except Exception:
                        try:
                            gas_outlet = comp.get_output('gas_outlet')
                        except Exception:
                            pass
                if gas_outlet:
                    # print(f"   ─── Gas Outlet ───")
                    # self._format_stream_properties(gas_outlet, "Gas Out", "   ")
                    pass
        
        # === Tank Section ===
        if components_by_type.get('tank'):
            print("\n### Storage Tanks")
            for comp_id, comp in components_by_type['tank']:
                state = comp.get_state() if hasattr(comp, 'get_state') else {}
                level_kg = state.get('current_level_kg', state.get('level_kg', 0.0))
                capacity_kg = state.get('capacity_kg', state.get('max_capacity_kg', 1.0))
                pressure_bar = state.get('pressure_bar', 0.0)
                fill_pct = (level_kg / capacity_kg * 100) if capacity_kg > 0 else 0
                
                print(f"   [{comp_id}] Level: {level_kg:,.2f}/{capacity_kg:,.0f} kg ({fill_pct:.1f}%) | P: {pressure_bar:.1f} bar")
        
        # === Compressor Section ===
        if components_by_type.get('compressor'):
            print("\n### Compressors")
            for comp_id, comp in components_by_type['compressor']:
                state = comp.get_state() if hasattr(comp, 'get_state') else {}
                # Compressor stores energy in timestep_energy_kwh, cumulative in cumulative_energy_kwh
                total_energy_kwh = state.get('cumulative_energy_kwh', 0.0)
                # Power = energy / time
                power_kw = total_energy_kwh / duration_hours if duration_hours > 0 else 0.0
                eta_isen = state.get('isentropic_efficiency', 0.65) * 100
                p_out_bar = state.get('outlet_pressure_bar', 0.0)
                
                print(f"   [{comp_id}] Power: {power_kw:.2f} kW | Energy: {total_energy_kwh:.2f} kWh | η_isen: {eta_isen:.1f}%")
        
        # === Deoxo Section ===
        if components_by_type.get('deoxo'):
            print("\n### Deoxidizer Reactors")
            for comp_id, comp in components_by_type['deoxo']:
                state = comp.get_state() if hasattr(comp, 'get_state') else {}
                # DeoxoReactor returns 'conversion_o2_percent' (already in %), 'peak_temperature_c', 'outlet_o2_ppm_mol'
                o2_conversion = state.get('conversion_o2_percent', 0.0)  # Already in %
                temp_max = state.get('peak_temperature_c', state.get('t_peak_total_c', 0.0))
                o2_out_ppm = state.get('outlet_o2_ppm_mol', 0.0)
                
                print(f"   [{comp_id}] O2 Conversion: {o2_conversion:.1f}% | T_max: {temp_max:.1f}°C | O2 out: {o2_out_ppm:.1f} ppm")
                
                # Get output stream
                gas_outlet = None
                if hasattr(comp, 'get_output'):
                    try:
                        gas_outlet = comp.get_output('h2_out')
                    except Exception:
                        try:
                            gas_outlet = comp.get_output('gas_outlet')
                        except Exception:
                            pass
                if gas_outlet:
                    print(f"   ─── Gas Outlet ───")
                    self._format_stream_properties(gas_outlet, "Gas Out", "   ")
        
        # === PSA/VSA Section ===
        if components_by_type.get('adsorption'):
            print("\n### Adsorption Units (PSA/VSA)")
            for comp_id, comp in components_by_type['adsorption']:
                state = comp.get_state() if hasattr(comp, 'get_state') else {}
                
                # PSA returns: product_flow_kg_h, tail_gas_flow_kg_h, power_consumption_kw
                product_flow = state.get('product_flow_kg_h', 0.0)
                tail_gas_flow = state.get('tail_gas_flow_kg_h', 0.0)  # This IS the H2 loss!
                power_kw = state.get('power_consumption_kw', 0.0)
                
                # Calculate totals over duration
                h2_loss_total = tail_gas_flow * duration_hours
                
                # Get inlet stream to calculate water removed
                inlet_stream = getattr(comp, 'inlet_stream', None)
                product_stream = getattr(comp, 'product_outlet', None)
                water_removed = 0.0
                if inlet_stream and product_stream:
                    inlet_h2o = inlet_stream.composition.get('H2O', 0.0) * inlet_stream.mass_flow_kg_h
                    outlet_h2o = product_stream.composition.get('H2O', 0.0) * product_stream.mass_flow_kg_h
                    water_removed = (inlet_h2o - outlet_h2o) * duration_hours
                
                print(f"   [{comp_id}]")
                print(f"   * Tail Gas Loss: {tail_gas_flow:.3f} kg/h (Total: {h2_loss_total:.3f} kg)")
                print(f"   * Water Removed: {water_removed:.4f} kg | Power: {power_kw:.2f} kW")
                
                # Get output stream (PSA uses 'purified_gas_out' port)
                gas_outlet = None
                if hasattr(comp, 'get_output'):
                    try:
                        gas_outlet = comp.get_output('purified_gas_out')
                    except Exception:
                        try:
                            gas_outlet = comp.get_output('gas_outlet')
                        except Exception:
                            pass
                if gas_outlet:
                    print(f"   ─── Purified Gas ───")
                    self._format_stream_properties(gas_outlet, "Product", "   ")
        
        # === SOEC Section (only if present) ===
        if components_by_type.get('soec'):
            print("\n### SOEC Electrolyzers")
            E_soec = np.sum(hist['P_soec_actual']) * dt
            H2_soec_total = np.sum(hist['H2_soec_kg'])
            print(f"   * Energy Consumed: {E_soec:.2f} MWh")
            print(f"   * H2 Produced: {H2_soec_total:.2f} kg")
        
        # === PEM Section (only if present) ===
        if components_by_type.get('pem'):
            print("\n### PEM Electrolyzers")
            E_pem = np.sum(hist['P_pem']) * dt
            H2_pem_total = np.sum(hist['H2_pem_kg'])
            print(f"   * Energy Consumed: {E_pem:.2f} MWh")
            print(f"   * H2 Produced: {H2_pem_total:.2f} kg")
        
        # === System Totals ===
        print("\n### System Totals")
        E_bop = np.sum(hist['P_bop_mw']) * dt
        E_total_offer = np.sum(hist['P_offer']) * dt
        E_sold = np.sum(hist['P_sold']) * dt
        
        # Only show energy offered/sold if there's actual power data
        if E_total_offer > 0.01:
            print(f"   * Total Offered Energy: {E_total_offer:.2f} MWh")
            print(f"   * Energy Sold to Market: {E_sold:.2f} MWh")
        
        print(f"   * BoP Energy Consumption: {E_bop:.2f} MWh")
        
        # Show H2 production totals only if electrolyzers present
        if components_by_type.get('soec') or components_by_type.get('pem'):
            H2_total = np.sum(hist['H2_soec_kg']) + np.sum(hist['H2_pem_kg'])
            print(f"   * Total H2 Production: {H2_total:.2f} kg")
        
        # === Stream Summary Table ===
        self._print_stream_summary_table(components_by_type)
        
        print("=" * 70)

    def _categorize_components(self) -> Dict[str, list]:
        """
        Categorize registered components by type for summary generation.
        
        Returns:
            Dict mapping component category to list of (comp_id, component) tuples.
        """
        categories = {
            'h2_source': [],
            'knock_out_drum': [],
            'coalescer': [],
            'chiller': [],
            'dry_cooler': [],
            'tank': [],
            'compressor': [],
            'deoxo': [],
            'adsorption': [],
            'soec': [],
            'pem': [],
            'valve': [],
            'mixer': [],
        }
        
        class_patterns = {
            'h2_source': ['ExternalH2Source', 'H2Source', 'O2Source'],
            'knock_out_drum': ['KnockOutDrum', 'KOD'],
            'coalescer': ['Coalescer'],
            'chiller': ['Chiller', 'ActiveCooler'],
            'dry_cooler': ['DryCooler', 'AirCooler'],
            'tank': ['Tank', 'H2Tank', 'StorageTank', 'FlashDrum'],
            'compressor': ['Compressor', 'H2Compressor'],
            'deoxo': ['Deoxo', 'Deoxidizer', 'DeoxoReactor'],
            'adsorption': ['PSA', 'VSA', 'TSA', 'AdsorptionDryer'],
            'soec': ['SOEC', 'SOECOperator', 'SOECCluster'],
            'pem': ['PEM', 'DetailedPEMElectrolyzer', 'PEMElectrolyzer'],
            'valve': ['Valve', 'ThrottlingValve'],
            'mixer': ['Mixer', 'WaterMixer'],
        }
        
        for comp_id, comp in self._registry.list_components():
            class_name = comp.__class__.__name__
            
            for category, patterns in class_patterns.items():
                if any(pattern in class_name for pattern in patterns):
                    categories[category].append((comp_id, comp))
                    break
        
        return categories

    def _format_stream_properties(self, stream, label: str = "Stream", indent: str = "      ") -> None:
        """
        Format and print stream thermodynamic properties.
        
        Displays: Composition, Phase, T, P, m_dot, H, ρ for a complete
        thermodynamic state characterization.
        
        Args:
            stream: Stream object with thermodynamic properties
            label: Description label for the stream
            indent: Whitespace prefix for formatting
        """
        if stream is None:
            print(f"{indent}{label}: No flow")
            return
        
        # Extract properties (handle both Stream objects and dicts)
        if hasattr(stream, 'mass_flow_kg_h'):
            m_dot = stream.mass_flow_kg_h
            T_k = stream.temperature_k
            P_pa = stream.pressure_pa
            composition = stream.composition
            phase = stream.phase
            try:
                h_j_kg = stream.specific_enthalpy_j_kg
                rho = stream.density_kg_m3
            except Exception:
                h_j_kg = 0.0
                rho = 0.0
        else:
            # Dict-like access
            m_dot = stream.get('mass_flow_kg_h', 0.0)
            T_k = stream.get('temperature_k', 273.15)
            P_pa = stream.get('pressure_pa', 101325)
            composition = stream.get('composition', {})
            phase = stream.get('phase', 'unknown')
            h_j_kg = stream.get('specific_enthalpy_j_kg', 0.0)
            rho = stream.get('density_kg_m3', 0.0)
        
        if m_dot <= 0:
            print(f"{indent}{label}: No flow")
            return
        
        # Convert units
        T_c = T_k - 273.15
        P_bar = P_pa / 1e5
        h_kj_kg = h_j_kg / 1000.0
        m_dot_kg_s = m_dot / 3600.0
        
        print(f"{indent}{label}:")
        print(f"{indent}  ├─ Phase: {phase}")
        print(f"{indent}  ├─ T: {T_c:.1f}°C ({T_k:.1f} K)")
        print(f"{indent}  ├─ P: {P_bar:.2f} bar ({P_pa/1000:.1f} kPa)")
        print(f"{indent}  ├─ ṁ: {m_dot:.2f} kg/h ({m_dot_kg_s:.4f} kg/s)")
        
        if h_kj_kg != 0:
            print(f"{indent}  ├─ H_mix: {h_kj_kg:.2f} kJ/kg ({h_j_kg:.0f} J/kg)")
        if rho > 0:
            print(f"{indent}  ├─ ρ: {rho:.4f} kg/m³")
        
        # Composition breakdown
        if composition:
            comp_parts = []
            for species, frac in sorted(composition.items(), key=lambda x: -x[1]):
                if frac > 0.0001:  # Show species > 0.01%
                    if frac >= 0.01:
                        comp_parts.append(f"{species}:{frac*100:.2f}%")
                    else:
                        comp_parts.append(f"{species}:{frac*1e6:.0f}ppm")
            print(f"{indent}  └─ Composition: {', '.join(comp_parts)}")

    def _print_stream_summary_table(self, components_by_type: Dict[str, list]) -> None:
        """
        Print a consolidated summary table of all component output streams.
        
        Shows Component | T_out | P_out | H2 Purity | H2O for each component
        in topology order (following the connection flow).
        """
        # Build ordered list by traversing topology connections
        ordered_comp_ids = self._get_topology_order()
        
        if not ordered_comp_ids:
            # Fallback: just list all components from registry
            ordered_comp_ids = [comp_id for comp_id, _ in self._registry.list_components()]
        
        # Port preferences for each component type
        port_preferences = {
            'ExternalH2Source': ['h2_out'],
            'KnockOutDrum': ['gas_outlet'],
            'DryCooler': ['fluid_out'],
            'Chiller': ['fluid_out'],
            'Coalescer': ['outlet'],
            'DeoxoReactor': ['outlet'],
            'ElectricBoiler': ['fluid_out'],  # Added for thermal heater
            'HeatExchanger': ['fluid_out'],   # Added for heat exchanger
            'PSA': ['purified_gas_out'],
            'Compressor': ['outlet', 'h2_out'],
            'Tank': ['h2_out', 'gas_out'],
        }
        
        # Collect stream data in topology order
        rows = []
        for comp_id in ordered_comp_ids:
            comp = self._registry.get(comp_id)
            if comp is None:
                continue
            
            class_name = comp.__class__.__name__
            
            # Get preferred ports for this component type
            ports_to_try = port_preferences.get(class_name, [])
            # Add generic fallbacks
            ports_to_try += ['fluid_out', 'gas_outlet', 'outlet', 'h2_out', 'purified_gas_out']
            
            stream = None
            if hasattr(comp, 'get_output'):
                for port in ports_to_try:
                    try:
                        stream = comp.get_output(port)
                        if stream and hasattr(stream, 'mass_flow_kg_h') and stream.mass_flow_kg_h > 0:
                            break
                        stream = None
                    except Exception:
                        continue
            
            if stream and hasattr(stream, 'mass_flow_kg_h') and stream.mass_flow_kg_h > 0:
                T_c = stream.temperature_k - 273.15
                P_bar = (stream.pressure_pa / 1e5) if hasattr(stream, 'pressure_pa') else 1.01325
                
                # Molecular weights (g/mol)
                MW_H2 = 2.016    # g/mol
                MW_H2O = 18.015  # g/mol
                MW_O2 = 32.0     # g/mol
                
                # --- Calculate Total Species Mass (including 'extra' liquid) ---
                m_dot_main = stream.mass_flow_kg_h
                
                # 1. Check for accompanying liquid in 'extra' (e.g. from PEM or Separators)
                m_dot_extra_liq = 0.0
                if hasattr(stream, 'extra') and stream.extra:
                    # Convert kg/s from extra to kg/h
                    m_dot_extra_liq = stream.extra.get('m_dot_H2O_liq_accomp_kg_s', 0.0) * 3600.0
                
                # 2. Calculate mass of each species
                # H2 mass
                m_h2 = stream.composition.get('H2', 0.0) * m_dot_main
                
                # Total Water Mass = Vapor (comp) + Liquid (comp) + Liquid (extra)
                m_h2o_vapor = stream.composition.get('H2O', 0.0) * m_dot_main
                m_h2o_liq_comp = stream.composition.get('H2O_liq', 0.0) * m_dot_main
                
                # Total water: ALWAYS sum all sources. 'extra' contains entrained liquid 
                # (e.g., from PEM mist) which is separate from composition-tracked water.
                m_h2o_total = m_h2o_vapor + m_h2o_liq_comp + m_dot_extra_liq
                
                # O2 mass
                m_o2 = stream.composition.get('O2', 0.0) * m_dot_main
                
                # 3. Convert to Moles
                n_h2 = m_h2 / MW_H2
                n_h2o = m_h2o_total / MW_H2O
                n_o2 = m_o2 / MW_O2
                n_total = n_h2 + n_h2o + n_o2
                
                # 4. Calculate Molar fractions (TOTAL including all water)
                y_h2 = n_h2 / n_total if n_total > 0 else 0
                y_h2o = n_h2o / n_total if n_total > 0 else 0
                y_o2 = n_o2 / n_total if n_total > 0 else 0
                
                # Format H2O display (TOTAL MOLAR - vapor + liquid)
                if y_h2o >= 0.01:
                    h2o_str = f"{y_h2o*100:.2f}%"
                elif y_h2o > 0:
                    h2o_str = f"{y_h2o*1e6:.0f} ppm"
                else:
                    h2o_str = "0 ppm"
                
                # Format O2 display (MOLAR ppm)
                if y_o2 >= 0.01:
                    o2_str = f"{y_o2*100:.2f}%"
                elif y_o2 > 0:
                    o2_str = f"{y_o2*1e6:.0f} ppm"
                else:
                    o2_str = "0 ppm"
                
                # Calculate liquid vs vapor percentage of TOTAL STREAM (not just water)
                # Liquid mass = liquid water only (vapor H2O is NOT liquid)
                # Total mass = H2 + H2O (all) + O2
                m_total_stream = m_h2 + m_h2o_total + m_o2
                
                if m_total_stream > 1e-9:
                    # Liquid in stream = H2O_liq (composition) + entrained liquid (extra)
                    m_liq_only = m_h2o_liq_comp + m_dot_extra_liq
                    
                    pct_liq = (m_liq_only / m_total_stream) * 100.0
                    pct_vap = 100.0 - pct_liq
                else:
                    pct_liq = 0.0
                    pct_vap = 0.0
                
                # Calculate total mass (sum of all species)
                m_total = m_h2 + m_h2o_total + m_o2
                
                rows.append({
                    'id': comp_id,
                    'T_c': T_c,
                    'P_bar': P_bar,
                    'H2_purity': y_h2 * 100,  # Molar purity
                    'H2_kg_h': m_h2,           # H2 mass flow (kg/h)
                    'H2O': h2o_str,
                    'H2O_kg_h': m_h2o_total,   # Total H2O mass flow (kg/h)
                    'O2': o2_str,
                    'O2_kg_h': m_o2,           # O2 mass flow (kg/h)
                    'Total_kg_h': m_total,     # Total mass flow (kg/h)
                    'pct_liq': pct_liq,
                    'pct_vap': pct_vap
                })
        
        if not rows:
            return
        
        # Print table (TOTAL Molar - includes entrained liquid)
        print("\n### Stream Summary Table (Topology Order) - TOTAL MOLAR (Vapor + Liquid)")
        print("-" * 180)
        print(f"{'Component':<18} | {'T_out':>7} | {'P_out':>8} | {'H2%':>10} | {'H2 kg/h':>8} | {'H2O':>9} | {'H2O kg/h':>9} | {'O2':>9} | {'O2 kg/h':>8} | {'Total':>10} | {'%Liq':>5} | {'%Vap':>5}")
        print("-" * 180)
        
        for row in rows:
            print(f"{row['id']:<18} | {row['T_c']:>5.1f}°C | {row['P_bar']:>6.2f} bar | {row['H2_purity']:>9.4f}% | {row['H2_kg_h']:>8.3f} | {row['H2O']:>9} | {row['H2O_kg_h']:>9.4f} | {row['O2']:>9} | {row['O2_kg_h']:>8.5f} | {row['Total_kg_h']:>10.2f} | {row['pct_liq']:>4.1f}% | {row['pct_vap']:>4.1f}%")
        
        print("-" * 180)

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



