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
            pem_h2_kwh_kg=pem_kwh_kg
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

            if hasattr(self._soec, 'last_h2_output_kg'):
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
        Print simulation summary with energy and production totals.
        """
        dt = self._context.simulation.timestep_hours
        hist = self.get_history()

        E_total_offer = np.sum(hist['P_offer']) * dt
        E_soec = np.sum(hist['P_soec_actual']) * dt
        E_pem = np.sum(hist['P_pem']) * dt
        E_sold = np.sum(hist['P_sold']) * dt
        E_bop = np.sum(hist['P_bop_mw']) * dt

        H2_soec_total = np.sum(hist['H2_soec_kg'])
        H2_pem_total = np.sum(hist['H2_pem_kg'])
        H2_total = H2_soec_total + H2_pem_total

        print("\n## Simulation Summary (Total/Average Values)")
        print(f"* Total Offered Energy: {E_total_offer:.2f} MWh")
        print(f"* Energy Supplied to SOEC: {E_soec:.2f} MWh")
        print(f"* Energy Supplied to PEM: {E_pem:.2f} MWh")
        print(f"* BoP Energy Consumption: {E_bop:.2f} MWh")
        print(f"* **Total System Hydrogen Production**: {H2_total:.2f} kg")
        print(f"  * SOEC Production: {H2_soec_total:.2f} kg")
        print(f"  * PEM Production: {H2_pem_total:.2f} kg")
        print(f"* Energy Sold to the Market: {E_sold:.2f} MWh")
        print("-------------------------------------------------------------------")
