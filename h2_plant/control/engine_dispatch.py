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


class HybridArbitrageEngineStrategy(EngineDispatchStrategy):
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
        >>> strategy.decide_and_apply(t=0.0, prices=prices, wind=wind)
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

        if self._soec and not self._pem:
            logger.info("Topology detected: SOEC Only. Using SoecOnlyStrategy.")
            self._inner_strategy = SoecOnlyStrategy()
        else:
            logger.info("Topology detected: Hybrid (or default). Using ReferenceHybridStrategy.")
            self._inner_strategy = ReferenceHybridStrategy()

        # Cache component capacities
        if self._soec:
            spec = context.physics.soec_cluster
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
                        self._format_stream_properties(h2_out, "H2 Out", "   ")
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
                
                print(f"   ─── Stream Properties ───")
                self._format_stream_properties(gas_outlet, "Gas Outlet", "   ")
                self._format_stream_properties(liquid_drain, "Liquid Drain", "   ")
        
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
                    print(f"   ─── Gas Outlet ───")
                    self._format_stream_properties(gas_outlet, "Gas Out", "   ")
        
        # === Chiller Section ===
        if components_by_type.get('chiller'):
            print("\n### Chillers (Active Cooling)")
            for comp_id, comp in components_by_type['chiller']:
                state = comp.get_state() if hasattr(comp, 'get_state') else {}
                heat_duty_w = state.get('heat_duty_w', state.get('Q_dot_fluxo_W', 0.0))
                heat_duty_kw = heat_duty_w / 1000.0
                power_kw = state.get('power_kw', state.get('electrical_power_kw', 0.0))
                temp_out_k = state.get('outlet_temperature_k', state.get('T_out_k', 0.0))
                temp_out_c = temp_out_k - 273.15 if temp_out_k > 0 else 0.0
                water_condensed = state.get('water_condensed_kg_h', 0.0) * duration_hours
                
                print(f"   [{comp_id}]")
                print(f"   * Heat Duty: {heat_duty_kw:.2f} kW | Elec. Power: {power_kw:.2f} kW")
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
                    print(f"   ─── Gas Outlet ───")
                    self._format_stream_properties(gas_outlet, "Gas Out", "   ")
        
        # === Dry Cooler Section ===
        if components_by_type.get('dry_cooler'):
            print("\n### Dry Coolers (Passive Cooling)")
            for comp_id, comp in components_by_type['dry_cooler']:
                state = comp.get_state() if hasattr(comp, 'get_state') else {}
                heat_duty_w = state.get('heat_duty_w', state.get('Q_dot_fluxo_W', 0.0))
                heat_duty_kw = heat_duty_w / 1000.0
                fan_power_kw = state.get('fan_power_kw', state.get('power_kw', 0.0))
                temp_out_k = state.get('outlet_temperature_k', 0.0)
                temp_out_c = temp_out_k - 273.15 if temp_out_k > 0 else 0.0
                
                print(f"   [{comp_id}] Heat Duty: {heat_duty_kw:.2f} kW | Fan: {fan_power_kw:.2f} kW | T_out: {temp_out_c:.1f}°C")
                
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
                    print(f"   ─── Gas Outlet ───")
                    self._format_stream_properties(gas_outlet, "Gas Out", "   ")
        
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
                power_kw = state.get('power_kw', state.get('electrical_power_kw', 0.0))
                total_energy_kwh = power_kw * duration_hours
                eta_isen = state.get('isentropic_efficiency', state.get('eta_isentropic', 0.0)) * 100
                p_out_bar = state.get('outlet_pressure_bar', state.get('P_out_bar', 0.0))
                
                print(f"   [{comp_id}] Power: {power_kw:.2f} kW | Energy: {total_energy_kwh:.2f} kWh | η_isen: {eta_isen:.1f}%")
        
        # === Deoxo Section ===
        if components_by_type.get('deoxo'):
            print("\n### Deoxidizer Reactors")
            for comp_id, comp in components_by_type['deoxo']:
                state = comp.get_state() if hasattr(comp, 'get_state') else {}
                o2_conversion = state.get('conversion', state.get('X_O2', 0.0)) * 100
                temp_max = state.get('T_max_C', state.get('max_temperature_c', 0.0))
                o2_out_ppm = state.get('o2_outlet_ppm', 0.0)
                
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
                h2_loss = state.get('h2_purge_loss_kg_h', 0.0) * duration_hours
                water_removed = state.get('water_removed_kg_h', 0.0) * duration_hours
                power_kw = state.get('power_kw', 0.0)
                
                print(f"   [{comp_id}] Water Removed: {water_removed:.4f} kg | H2 Loss: {h2_loss:.4f} kg | Power: {power_kw:.2f} kW")
                
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
                P_bar = stream.pressure_pa / 1e5
                
                # Get H2, H2O, and O2 fractions
                h2_frac = stream.composition.get('H2', 0.0)
                h2o_frac = stream.composition.get('H2O', 0.0) + stream.composition.get('H2O_liq', 0.0)
                o2_frac = stream.composition.get('O2', 0.0)
                
                # Format H2O display
                if h2o_frac >= 0.01:
                    h2o_str = f"{h2o_frac*100:.2f}%"
                elif h2o_frac > 0:
                    h2o_str = f"{h2o_frac*1e6:.0f} ppm"
                else:
                    h2o_str = "0 ppm"
                
                # Format O2 display
                if o2_frac >= 0.01:
                    o2_str = f"{o2_frac*100:.2f}%"
                elif o2_frac > 0:
                    o2_str = f"{o2_frac*1e6:.0f} ppm"
                else:
                    o2_str = "0 ppm"
                
                rows.append({
                    'id': comp_id,
                    'T_c': T_c,
                    'P_bar': P_bar,
                    'H2_purity': h2_frac * 100,
                    'H2O': h2o_str,
                    'O2': o2_str
                })
        
        if not rows:
            return
        
        # Print table
        print("\n### Stream Summary Table (Topology Order)")
        print("-" * 88)
        print(f"{'Component':<18} | {'T_out':>8} | {'P_out':>9} | {'H2 Purity':>10} | {'H2O':>10} | {'O2':>10}")
        print("-" * 88)
        
        for row in rows:
            print(f"{row['id']:<18} | {row['T_c']:>6.1f}°C | {row['P_bar']:>7.2f} bar | {row['H2_purity']:>8.2f}% | {row['H2O']:>10} | {row['O2']:>10}")
        
        print("-" * 88)

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



