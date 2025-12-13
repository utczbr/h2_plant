"""
Integrated Dispatch Strategy for SimulationEngine.

This module provides dispatch strategy classes that integrate with the
SimulationEngine, replacing the standalone Orchestrator execution path.

The strategy encapsulates:
- Dispatch decision logic (which electrolyzer to use, how much to sell)
- Power setpoint application to components
- History recording with NumPy pre-allocation

This is Phase B1 of the architectural remediation.
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
    """Extended dispatch state with history tracking."""
    P_soec_prev: float = 0.0
    force_sell: bool = False
    step_idx: int = 0
    cumulative_h2_kg: float = 0.0


class EngineDispatchStrategy(ABC):
    """
    Abstract base for dispatch strategies integrated with SimulationEngine.
    
    Unlike the original DispatchStrategy, this class:
    - Has access to the ComponentRegistry
    - Manages its own state and history
    - Applies dispatch decisions to components
    - Uses NumPy pre-allocated arrays for history
    """
    
    @abstractmethod
    def initialize(
        self, 
        registry: 'ComponentRegistry', 
        context: 'SimulationContext',
        total_steps: int
    ) -> None:
        """
        Initialize strategy with system references.
        
        Args:
            registry: Component registry for accessing components
            context: Simulation context with configuration
            total_steps: Total number of timesteps (for pre-allocation)
        """
        pass
    
    @abstractmethod
    def decide_and_apply(self, t: float, prices: np.ndarray, wind: np.ndarray) -> None:
        """
        Make dispatch decision and apply to components.
        
        Called by SimulationEngine at each timestep BEFORE stepping components.
        This sets power setpoints; the actual physics happens in component.step().
        
        Args:
            t: Current simulation time (hours)
            prices: Energy price array (full simulation)
            wind: Wind power offer array (full simulation)
        """
        pass
    
    @abstractmethod
    def get_history(self) -> Dict[str, np.ndarray]:
        """Return recorded history as NumPy arrays."""
        pass


class HybridArbitrageEngineStrategy(EngineDispatchStrategy):
    """
    Hybrid SOEC/PEM arbitrage strategy integrated with SimulationEngine.
    
    Migrated from the Orchestrator's run_simulation() dispatch logic.
    Uses NumPy pre-allocated arrays for history (10-50x faster than list.append).
    """
    
    def __init__(self):
        self._registry: Optional['ComponentRegistry'] = None
        self._context: Optional['SimulationContext'] = None
        self._inner_strategy: Optional[BaseDispatchStrategy] = None
        self._state = IntegratedDispatchState()
        
        # Component references (resolved in initialize)
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
        """Initialize strategy with pre-allocated NumPy arrays for history."""
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
        
        # Cache capacities
        if self._soec:
            spec = context.physics.soec_cluster
            self._soec_capacity = spec.num_modules * spec.max_power_nominal_mw * spec.optimal_limit
        
        if self._pem:
            self._pem_max = context.physics.pem_system.max_power_mw
        
        # === A3: Pre-allocate history arrays (HPC FIX) ===
        self._history = {
            # Core metrics
            'minute': np.zeros(total_steps, dtype=np.int32),
            'P_offer': np.zeros(total_steps, dtype=np.float64),
            'P_soec_actual': np.zeros(total_steps, dtype=np.float64),
            'P_pem': np.zeros(total_steps, dtype=np.float64),
            'P_sold': np.zeros(total_steps, dtype=np.float64),
            'spot_price': np.zeros(total_steps, dtype=np.float64),
            'h2_kg': np.zeros(total_steps, dtype=np.float64),
            
            # Detailed production
            'H2_soec_kg': np.zeros(total_steps, dtype=np.float64),
            'H2_pem_kg': np.zeros(total_steps, dtype=np.float64),
            'cumulative_h2_kg': np.zeros(total_steps, dtype=np.float64),
            
            # SOEC details
            'steam_soec_kg': np.zeros(total_steps, dtype=np.float64),
            'H2O_soec_out_kg': np.zeros(total_steps, dtype=np.float64),
            'soec_active_modules': np.zeros(total_steps, dtype=np.int32),
            
            # PEM details
            'H2O_pem_kg': np.zeros(total_steps, dtype=np.float64),
            'O2_pem_kg': np.zeros(total_steps, dtype=np.float64),
            'pem_V_cell': np.zeros(total_steps, dtype=np.float64),
            
            # Energy balance
            'P_bop_mw': np.zeros(total_steps, dtype=np.float64),
            
            # Legacy compatibility
            'tank_level_kg': np.zeros(total_steps, dtype=np.float64),
            'tank_pressure_bar': np.zeros(total_steps, dtype=np.float64),
            'compressor_power_kw': np.zeros(total_steps, dtype=np.float64),
            'sell_decision': np.zeros(total_steps, dtype=np.int8),
        }
        
        # Reset state
        self._state = IntegratedDispatchState()
        
        logger.info(f"Initialized HybridArbitrageEngineStrategy with {total_steps} pre-allocated steps")
    
    def decide_and_apply(self, t: float, prices: np.ndarray, wind: np.ndarray) -> None:
        """
        Make dispatch decision and apply setpoints to components.
        
        IMPORTANT: This method ONLY sets power inputs. It does NOT call step().
        The SimulationEngine is responsible for calling step() on all components.
        """
        dt = self._context.simulation.timestep_hours
        step_idx = self._state.step_idx
        
        if step_idx >= self._total_steps:
            logger.warning(f"Step index {step_idx} exceeds pre-allocated size {self._total_steps}")
            return
        
        # Get current inputs
        minute = int(round(t * 60))
        P_offer = wind[step_idx]
        current_price = prices[step_idx]
        
        # Future offer (lookahead 60 minutes)
        P_future = wind[min(step_idx + 60, len(wind) - 1)]
        
        # Build dispatch input
        # Use safe attribute access with defaults from constants
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
        
        # Build dispatch state
        d_state = DispatchState(
            P_soec_prev=self._state.P_soec_prev,
            force_sell=self._state.force_sell
        )
        
        # Get dispatch decision
        result = self._inner_strategy.decide(d_input, d_state)
        
        # Update internal state
        self._state.force_sell = result.state_update.get('force_sell', False)
        
        # === Apply setpoints to components (NO step() calls!) ===
        P_soec_actual = 0.0
        h2_soec = 0.0
        steam_soec = 0.0
        
        if self._soec:
            # Set power setpoint via receive_input
            self._soec.receive_input('power_in', result.P_soec, 'electricity')
            # Component physics will execute when engine calls step()
        
        P_pem_actual = result.P_pem
        
        if self._pem:
            # Set power setpoint
            self._pem.set_power_input_mw(result.P_pem)
            # Component physics will execute when engine calls step()
        
        # Record pre-step data (setpoints and offers)
        self._history['minute'][step_idx] = minute
        self._history['P_offer'][step_idx] = P_offer
        self._history['spot_price'][step_idx] = current_price
        
        # Post-step recording is handled by record_post_step()
        self._state.step_idx = step_idx
    
    def record_post_step(self) -> None:
        """
        Record component outputs AFTER step() has been called.
        
        Called by SimulationEngine after all components have executed step().
        This captures the actual physics results.
        """
        step_idx = self._state.step_idx
        dt = self._context.simulation.timestep_hours
        
        if step_idx >= self._total_steps:
            return
        
        # Get SOEC results
        P_soec_actual = 0.0
        h2_soec = 0.0
        steam_soec = 0.0
        
        if self._soec:
            # Read actual values from component
            if hasattr(self._soec, 'real_powers'):
                P_soec_actual = float(np.sum(self._soec.real_powers))
            
            if hasattr(self._soec, 'last_h2_output_kg'):
                h2_soec = self._soec.last_h2_output_kg
            elif hasattr(self._soec, 'h2_output_kg'):
                h2_soec = self._soec.h2_output_kg
            
            steam_soec = getattr(self._soec, 'last_steam_output_kg', 0.0)
        
        self._state.P_soec_prev = P_soec_actual
        
        # Get PEM results
        h2_pem = 0.0
        P_pem_actual = 0.0
        
        if self._pem:
            h2_pem = getattr(self._pem, 'h2_output_kg', 0.0)
            if hasattr(self._pem, 'P_consumed_W'):
                P_pem_actual = self._pem.P_consumed_W / 1e6
        
        # Get BoP power
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
        
        # Update cumulative
        self._state.cumulative_h2_kg += total_h2
        
        # Record to pre-allocated arrays (A3: HPC FIX)
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
        
        # SOEC details
        if self._soec and hasattr(self._soec, 'real_powers'):
            self._history['soec_active_modules'][step_idx] = int(np.sum(self._soec.real_powers > 0.01))
        
        h2o_soec_out = getattr(self._soec, 'last_water_output_kg', 0.0) if self._soec else 0.0
        self._history['H2O_soec_out_kg'][step_idx] = h2o_soec_out
        
        # PEM details
        h2o_pem = getattr(self._pem, 'water_consumption_kg', h2_pem * 9.0 * 1.02) if self._pem else h2_pem * 9.0 * 1.02
        self._history['H2O_pem_kg'][step_idx] = h2o_pem
        self._history['O2_pem_kg'][step_idx] = h2_pem * 8.0
        self._history['pem_V_cell'][step_idx] = getattr(self._pem, 'V_cell', 0.0) if self._pem else 0.0
        
        # Legacy compatibility (tank/compressor)
        tank_main = self._registry.get("H2_Tank") if self._registry.has("H2_Tank") else None
        comp_main = self._registry.get("H2_Compressor") if self._registry.has("H2_Compressor") else None
        
        self._history['tank_level_kg'][step_idx] = getattr(tank_main, 'current_level_kg', 0.0) if tank_main else 0.0
        self._history['tank_pressure_bar'][step_idx] = getattr(tank_main, 'pressure_bar', 0.0) if tank_main else 0.0
        self._history['compressor_power_kw'][step_idx] = getattr(comp_main, 'power_kw', 0.0) if comp_main else 0.0
        
        # Increment step for next iteration
        self._state.step_idx += 1
    
    def get_history(self) -> Dict[str, np.ndarray]:
        """Return recorded history as NumPy arrays (A3: HPC compliant)."""
        # Trim to actual steps completed
        actual_steps = self._state.step_idx
        return {k: v[:actual_steps] for k, v in self._history.items()}
    
    def _find_soec(self, registry: 'ComponentRegistry'):
        """Find SOEC component in registry."""
        for comp_id, comp in registry.list_components():
            class_name = comp.__class__.__name__
            if hasattr(comp, 'soec_state') or class_name == 'SOECOperator':
                return comp
        return None
    
    def _find_pem(self, registry: 'ComponentRegistry'):
        """Find PEM component in registry."""
        for comp_id, comp in registry.list_components():
            class_name = comp.__class__.__name__
            if hasattr(comp, 'V_cell') or class_name == 'DetailedPEMElectrolyzer':
                return comp
        return None
    
    def print_summary(self) -> None:
        """Print simulation summary (replaces Orchestrator debug output)."""
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
