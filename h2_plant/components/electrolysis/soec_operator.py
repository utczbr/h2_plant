"""
SOEC (Solid Oxide Electrolyzer Cell) Operator Component.

This module manages multi-module SOEC stacks for high-temperature electrolysis
of steam to produce hydrogen. The operator handles module rotation, degradation
tracking, and power distribution across the parallel module fleet.

Electrochemical Principles:
    - **High-Temperature Electrolysis**: SOEC operates at 700-850°C where solid
      oxide electrolytes conduct oxygen ions (O²⁻). Steam splits at the cathode:
      H₂O + 2e⁻ → H₂ + O²⁻, with O²⁻ migrating to the anode.
    - **Thermal Efficiency Advantage**: Operating above the thermoneutral voltage
      (~1.29V at 800°C) allows heat integration from external sources, achieving
      electrical efficiencies exceeding 100% (LHV basis).
    - **Degradation Mechanisms**: Stack voltage increases over time due to
      electrode sintering, interconnect oxidation, and electrolyte thinning.

Architecture:
    Implements the Component Lifecycle Contract (Layer 1):
    - `initialize()`: Prepares module states and connects to component registry.
    - `step()`: Executes power distribution, rotation, and production calculations.
    - `get_state()`: Returns module states and cumulative production metrics.

Operational Modes:
    Each module operates in one of several states:
    - State 0: OFF (cold shutdown)
    - State 1: Hot standby (maintained at temperature, no production)
    - State 2: Ramping up (transitioning to full power)
    - State 3: Operating (producing hydrogen)
    - State 5: Ramping down

Module Rotation:
    Periodic virtual map rotation distributes operational hours evenly across
    modules, extending fleet lifetime by preventing uneven degradation.

References:
    - Graves, C. et al. (2011). Eliminating degradation in solid oxide
      electrochemical cells by reversible operation. Nature Materials, 10(4).
    - Hauch, A. et al. (2020). Recent advances in solid oxide cell technology
      for electrolysis. Science, 370(6513).
"""

import numpy as np
import math
from typing import Dict, Any, Tuple, List, Optional

from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.stream import Stream

# ============================================================================
# Default Configuration Constants
# ============================================================================
DEFAULT_NUM_MODULES = 6
DEFAULT_MAX_POWER_NOMINAL_MW = 2.4
DEFAULT_OPTIMAL_LIMIT = 0.80
DEFAULT_POWER_STANDBY_MW = 0.0
DEFAULT_POWER_FIRST_STEP_MW = 0.12
DEFAULT_RAMP_STEP_MW = 0.24

# ============================================================================
# Degradation Tables
# ============================================================================
# Empirical degradation curves based on accelerated aging tests.
# Efficiency increases (worsens) and capacity decreases over operating years.
DEG_YEARS = np.array([0, 1, 2, 3, 3.5, 4, 5, 5.5, 6, 7])
DEG_EFFICIENCY_KWH_KG = np.array([37.5, 37.5, 37.5, 37.5, 37.5, 38, 39, 40, 41, 42])
DEG_CAPACITY_FACTOR = np.array([100, 100, 100, 100, 100, 100, 100, 90, 85, 75])


class SOECOperator(Component):
    """
    Multi-module SOEC stack operator for high-temperature hydrogen production.

    Manages a fleet of SOEC modules operating in parallel, handling power
    distribution, state transitions, and degradation tracking. The operator
    uses a virtual map abstraction to rotate module priority and balance wear.

    This component fulfills the Component Lifecycle Contract (Layer 1):
        - `initialize()`: Initializes module states and degradation factors.
        - `step()`: Distributes power, executes state machines, computes production.
        - `get_state()`: Returns operational status and cumulative metrics.

    The power distribution algorithm (implemented in Numba JIT) prioritizes
    modules in virtual map order, ramping them through operating states to
    meet the reference power setpoint.

    Attributes:
        num_modules (int): Number of parallel SOEC modules.
        max_nominal_power (float): Rated power per module (MW).
        current_efficiency_kwh_kg (float): Current specific energy (kWh/kg H₂).
        total_h2_produced (float): Cumulative hydrogen production (kg).

    Example:
        >>> soec = SOECOperator({'num_modules': 6, 'max_power_nominal_mw': 2.4})
        >>> soec.initialize(dt=1/60, registry=registry)
        >>> soec.receive_input('power_in', 10.0, 'electricity')
        >>> power, h2_kg, steam_kg = soec.step(t=0.0)
    """

    def __init__(self, config: Any, physics_config: Dict[str, Any] = None):
        """
        Initialize the SOEC operator.

        Supports both Pydantic SOECPhysicsSpec and legacy dictionary configuration
        formats for backward compatibility.

        Args:
            config (Union[SOECPhysicsSpec, Dict]): Configuration containing:
                - num_modules (int): Number of parallel modules.
                - max_power_nominal_mw (float): Rated power per module in MW.
                - optimal_limit (float): Operating point as fraction of rated (0-1).
                - power_first_step_mw (float): Initial ramp step size in MW.
                - ramp_step_mw (float): Subsequent ramp step size in MW.
                - steam_input_ratio_kg_per_kg_h2 (float): Steam requirement ratio.
                - out_pressure_pa (float, optional): H₂ output pressure in Pa.
            physics_config (Dict, optional): Legacy physics configuration dict.
        """
        super().__init__(config)

        # Configuration parsing for Pydantic model vs legacy dict
        if hasattr(config, 'max_power_nominal_mw'):
            self.spec = config
            self.num_modules = config.num_modules
            self.max_nominal_power = config.max_power_nominal_mw
            self.optimal_limit_ratio = config.optimal_limit
            self.rotation_enabled = False
            self.degradation_year = 0.0

            self.power_standby_mw = DEFAULT_POWER_STANDBY_MW
            self.power_first_step_mw = config.power_first_step_mw
            self.ramp_step_mw = config.ramp_step_mw
            # UPDATE: Default steam ratio to 9.0 (Legacy Alignment)
            self.steam_input_ratio = getattr(config, 'steam_input_ratio_kg_per_kg_h2', 9.0)

            # UPDATE: Default pressure to 1.0 bar (Legacy Alignment), override constant
            self.out_pressure_pa = getattr(config, 'out_pressure_pa', 100000.0)

            self.config = config.dict()
            self.physics_config = {}

        else:
            self.config = config
            self.physics_config = physics_config or {}

            self.num_modules = config.get("num_modules", DEFAULT_NUM_MODULES)
            self.max_nominal_power = config.get("max_power_nominal_mw", DEFAULT_MAX_POWER_NOMINAL_MW)
            self.optimal_limit_ratio = config.get("optimal_limit", DEFAULT_OPTIMAL_LIMIT)
            self.rotation_enabled = config.get("rotation_enabled", False)
            self.degradation_year = config.get("degradation_year", 0.0)

            soec_phys = self.physics_config.get("soec", {})
            self.power_standby_mw = DEFAULT_POWER_STANDBY_MW
            self.power_first_step_mw = soec_phys.get("power_first_step_mw", DEFAULT_POWER_FIRST_STEP_MW)
            self.ramp_step_mw = soec_phys.get("ramp_step_mw", DEFAULT_RAMP_STEP_MW)
            self.ramp_step_mw = soec_phys.get("ramp_step_mw", DEFAULT_RAMP_STEP_MW)
            
            # UPDATE: Default steam ratio to 9.0. Check config first (YAML overrides), then physics, then default.
            self.steam_input_ratio = config.get("steam_input_ratio_kg_per_kg_h2", 
                                                soec_phys.get("steam_input_ratio_kg_per_kg_h2", 9.0))

            # UPDATE: Default pressure to 1.0 bar. Check config first.
            self.out_pressure_pa = config.get("out_pressure_pa", 100000.0)

        # Lifecycle parameter for degradation reset (hours)
        if isinstance(config, dict):
             self.lifecycle_h = config.get("lifecycle", 876000.0) # default ~100 years
        else:
             self.lifecycle_h = getattr(config, "lifecycle", 876000.0)

        # Initialize accumulated hours based on starting degradation year
        self.accumulated_hours = self.degradation_year * 8760.0

        # Initialize module states
        self._initialize_state()

        # Water input accumulation for mass balance limiting
        self._input_water_buffer_kg_h: List[float] = []
        self._last_step_time: float = 0.0

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """
        Prepare the operator for simulation execution.

        Fulfills the Component Lifecycle Contract initialization phase.
        Module states should already be initialized by _initialize_state().

        Args:
            dt (float): Simulation timestep in hours.
            registry (ComponentRegistry): Central registry for component access.
        """
        super().initialize(dt, registry)

    def _interpolate(self, x: float, xp: np.ndarray, fp: np.ndarray) -> float:
        """
        Perform linear interpolation for degradation lookup.

        Args:
            x (float): Query point (operating years).
            xp (np.ndarray): Tabulated year values.
            fp (np.ndarray): Tabulated property values.

        Returns:
            float: Interpolated property value.
        """
        return np.interp(x, xp, fp)

    def _initialize_state(self) -> None:
        """
        Initialize internal state variables for all modules.

        Calculates degradation-adjusted efficiency and capacity, then
        initializes module power, state, and limit vectors. Modules marked
        as offline are set to state 0 with zero power.
        """
        # Interpolate degradation factors at current operating age
        self.current_efficiency_kwh_kg = float(
            self._interpolate(self.degradation_year, DEG_YEARS, DEG_EFFICIENCY_KWH_KG)
        )
        cap_percent = float(
            self._interpolate(self.degradation_year, DEG_YEARS, DEG_CAPACITY_FACTOR)
        )
        self.current_capacity_factor = cap_percent / 100.0

        # Effective maximum power accounts for capacity degradation
        self.effective_max_module_power = self.max_nominal_power * self.current_capacity_factor

        # Optimal operating point (e.g., 80% of rated for efficiency)
        self.uniform_module_max_limit = self.effective_max_module_power * self.optimal_limit_ratio

        # Initialize module vectors
        # PERFORMANCE: Use strict Numba-compatible dtypes to avoid copying per-step.
        self.real_limits = np.full(self.num_modules, self.uniform_module_max_limit, dtype=np.float64)
        self.real_powers = np.full(self.num_modules, self.power_standby_mw, dtype=np.float64)
        self.real_states = np.full(self.num_modules, 1, dtype=np.int32)  # State 1: Hot standby

        # Apply offline modules from configuration
        off_modules = self.config.get("real_off_modules", [])
        off_indices = [mid - 1 for mid in off_modules if 1 <= mid <= self.num_modules]
        if off_indices:
            self.real_states[off_indices] = 0
            self.real_powers[off_indices] = 0.0
            self.real_limits[off_indices] = 0.0

        # Virtual map maps priority order to physical module indices
        # PERFORMANCE: Enforce int32 to match Numba kernel signature.
        active_indices = np.where(self.real_states != 0)[0]
        self.virtual_map = active_indices.astype(np.int32)

        # Tracking variables
        self.difference_history = []
        self.previous_total_power = np.sum(self.real_powers)
        self.current_minute = 0
        self.total_h2_produced = 0.0
        self.total_steam_consumed = 0.0

        # Wear tracking for lifetime optimization
        self.accumulated_wear = np.zeros(self.num_modules, dtype=float)
        self.cycle_counts = np.zeros(self.num_modules, dtype=int)
        self.previous_real_states = np.copy(self.real_states)

        # Power setpoint interface
        self._power_setpoint_mw = 0.0
        self.available_water_kg_h = float('inf')
        self.current_water_input_kg = 0.0

    def _update_virtual_map(self) -> None:
        """
        Rotate the virtual priority map by one position.

        Rotation redistributes module usage to balance wear across the fleet.
        Called hourly when rotation_enabled is True.
        """
        self.virtual_map = np.roll(self.virtual_map, -1)

    def _get_total_power(self, powers: np.ndarray) -> float:
        """Calculate total power consumption across all modules."""
        return np.sum(powers)

    def _update_stationary_state(
        self,
        powers: np.ndarray,
        states: np.ndarray,
        limits: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transition ramping modules to stable operating state.

        Modules in ramping states (2 or 5) are transitioned to operating
        state (3) once power adjustment is complete.

        Args:
            powers: Module power array.
            states: Module state array.
            limits: Module limit array.

        Returns:
            Tuple of updated states and limits arrays.
        """
        ramp_indices = np.where((states == 2) | (states == 5))[0]
        if len(ramp_indices) > 0:
            states[ramp_indices] = 3
        return states, limits

    def _simulate_step(self, reference_power: float) -> None:
        """
        Execute core power distribution logic via Numba JIT.

        Distributes the reference power setpoint across modules according
        to the virtual priority map, respecting per-module limits and
        ramp rate constraints.

        Args:
            reference_power (float): Target total power consumption in MW.
        """
        from h2_plant.optimization.numba_ops import simulate_soec_step_jit

        # PERFORMANCE: real_states and virtual_map are now int32 at init.
        # No astype() copy needed per step.
        self.real_powers, self.real_states, self.real_limits = simulate_soec_step_jit(
            reference_power,
            self.real_powers,
            self.real_states,
            self.real_limits,
            self.virtual_map,
            self.uniform_module_max_limit,
            self.power_standby_mw,
            self.power_first_step_mw,
            self.ramp_step_mw,
            0.0
        )

        # Track power mismatch for diagnostics
        current_total = np.sum(self.real_powers)
        diff = reference_power - current_total
        self.difference_history.append(diff)

    def step(self, t: float = 0.0) -> Tuple[float, float, float]:
        """
        Execute one simulation timestep.

        Performs the complete SOEC operator cycle:
        1. Process water input buffer and reset for new timestep.
        2. Apply power and water constraints to reference setpoint.
        3. Check rotation interval and rotate virtual map if due.
        4. Distribute power via JIT-compiled module state machine.
        5. Calculate hydrogen production from consumed energy.
        6. Track wear and cycle counts for lifetime prediction.

        Args:
            t (float): Current simulation time in hours.

        Returns:
            Tuple[float, float, float]: (total_power_mw, h2_produced_kg, steam_consumed_kg)
        """
        super().step(t)

        # Handle water input accumulation for new timesteps
        # Always process buffer if data is present. Remove time-check guard to prevent
        # accumulation bugs (float comparison or cycle issues).
        if self._input_water_buffer_kg_h:
            water_sum = sum(self._input_water_buffer_kg_h)
            self.available_water_kg_h = water_sum
            self.current_water_input_kg = water_sum
            self._input_water_buffer_kg_h.clear()
        elif t != self._last_step_time:
            # Only reset default if time changed and no input (new step start)
            self.available_water_kg_h = float('inf')
            self.current_water_input_kg = 0.0

        self._last_step_time = t

        # Get power setpoint from port interface
        reference_power_mw = self._power_setpoint_mw

        # Apply system capacity constraint
        system_max_capacity = self.num_modules * self.effective_max_module_power
        reference_power_mw = max(0.0, reference_power_mw)
        clamped_reference = min(reference_power_mw, system_max_capacity)

        # Apply water availability constraint via stoichiometry
        if self.dt > 0 and self.available_water_kg_h < float('inf'):
            h2_prod_rate_kg_mwh = 1000.0 / self.current_efficiency_kwh_kg
            max_h2_from_steam = self.available_water_kg_h * self.dt / self.steam_input_ratio
            max_energy_mwh = max_h2_from_steam / h2_prod_rate_kg_mwh
            max_power_water_mw = max_energy_mwh / self.dt

            clamped_reference = min(clamped_reference, max_power_water_mw)

        # Module rotation logic (hourly rotation for wear distribution)
        minutes_passed = self.dt * 60.0
        prev_minute = self.current_minute
        self.current_minute += minutes_passed

        if self.rotation_enabled and (int(prev_minute / 60) < int(self.current_minute / 60)):
            self._update_virtual_map()
        
        # === Dynamic Degradation Update ===
        # Advance fleet absolute age
        self.accumulated_hours += self.dt
        
        # Calculate effective age for degradation (Modulo Lifecycle)
        effective_hours = self.accumulated_hours % self.lifecycle_h
        effective_year = effective_hours / 8760.0
        
        # Re-interpolate performance factors
        self.current_efficiency_kwh_kg = float(
            self._interpolate(effective_year, DEG_YEARS, DEG_EFFICIENCY_KWH_KG)
        )
        cap_percent = float(
             self._interpolate(effective_year, DEG_YEARS, DEG_CAPACITY_FACTOR)
        )
        new_capacity_factor = cap_percent / 100.0
        
        # If capacity changed, update limits
        if abs(new_capacity_factor - self.current_capacity_factor) > 1e-6:
            self.current_capacity_factor = new_capacity_factor
            self.effective_max_module_power = self.max_nominal_power * self.current_capacity_factor
            self.uniform_module_max_limit = self.effective_max_module_power * self.optimal_limit_ratio
            
            # Simple approach: Update all active modules
            active_mask = self.real_states != 0
            self.real_limits[active_mask] = self.uniform_module_max_limit

        # Execute power distribution
        self._simulate_step(clamped_reference)

        current_total_power = self._get_total_power(self.real_powers)

        # Wear tracking for lifetime optimization
        self.accumulated_wear += self.real_powers

        for i in range(self.num_modules):
            prev = self.previous_real_states[i]
            curr = self.real_states[i]
            if prev == 1 and curr == 2:
                self.cycle_counts[i] += 1

        self.previous_real_states = np.copy(self.real_states)

        # Production calculation using energy balance
        energy_consumed_mwh = current_total_power * self.dt
        h2_prod_rate_kg_mwh = 1000.0 / self.current_efficiency_kwh_kg
        h2_produced_kg = energy_consumed_mwh * h2_prod_rate_kg_mwh
        self.total_h2_produced += h2_produced_kg
        self.last_step_h2_kg = h2_produced_kg

        # Steam consumption (Reaction + Excess)
        # Stoichiometry: 9.0 kg H2O per kg H2 (approx)
        reaction_steam_kg = h2_produced_kg * 8.936 # Precise stoichiometry
        self.total_steam_consumed += reaction_steam_kg # Only count reacted water as consumed
        
        # Calculate unreacted steam based on ACTUAL input
        # Fix: Convert Input Rate (kg/h) to Input Mass (kg) for this timestep
        input_mass_kg = self.current_water_input_kg * self.dt
        
        if input_mass_kg > 0:
             self.last_water_output_kg = max(0.0, input_mass_kg - reaction_steam_kg)
             self.last_step_steam_input_kg = input_mass_kg
        else:
             # Fallback for initialization/standalone mode
             theoretical_input_kg = h2_produced_kg * self.steam_input_ratio
             self.last_step_steam_input_kg = theoretical_input_kg
             self.last_water_output_kg = max(0.0, theoretical_input_kg - reaction_steam_kg)

        self.previous_total_power = current_total_power

        return current_total_power, h2_produced_kg, self.last_step_steam_input_kg

    def receive_input(self, port_name: str, value: Any, resource_type: str = None) -> float:
        """
        Accept input at the specified port.

        Handles power setpoint from coordinator and water/steam input from
        upstream components. Water inputs are buffered for mass-balance limiting.

        Args:
            port_name (str): Target port ('power_in' or 'steam_in'/'water_in').
            value (Any): Power setpoint (MW) or Stream object.
            resource_type (str, optional): Resource classification hint.

        Returns:
            float: Value accepted (MW or kg/h), or 0.0 if rejected.
        """
        if port_name == 'power_in':
            if isinstance(value, (int, float)):
                self._power_setpoint_mw = float(value)
                return float(value)
        elif port_name in ('water_in', 'steam_in'):
            if isinstance(value, Stream):
                self._input_water_buffer_kg_h.append(value.mass_flow_kg_h)
                return value.mass_flow_kg_h
        return 0.0

    def get_output(self, port_name: str) -> Any:
        """
        Retrieve output from a specified port.

        The `h2_out` port now returns the combined **Cathode Exhaust** stream,
        containing H2 (produced), H2O (unreacted steam), and trace O2 (crossover).
        This reflects the physical reality of the SOEC cathode.

        Args:
            port_name (str): Port identifier ('h2_out' or 'o2_out').

        Returns:
            Stream: Output stream with mass flow, temperature, and composition.

        Raises:
            ValueError: If port_name is not a valid output port.
        """
        h2_kg_total = getattr(self, 'last_step_h2_kg', 0.0)
        unreacted_steam_kg = getattr(self, 'last_water_output_kg', 0.0)

        # Calculate flow rates (kg/h)
        h2_flow = h2_kg_total * (1.0 / self.dt) if self.dt > 0 else 0.0
        steam_flow = unreacted_steam_kg * (1.0 / self.dt) if self.dt > 0 else 0.0

        if port_name == 'h2_out':
            # === CATHODE EXHAUST (Wet Hydrogen) ===
            # Composition: H2 (Produced) + H2O (Unreacted) + O2 (Crossover leak)

            # 1. Determine Mass of Crossover O2
            # Assumption: Y_O2_IN_H2 (Molar) = 0.0002 (200 ppm) relative to H2
            y_o2_molar = 0.0002
            mw_h2 = 2.016
            mw_o2 = 32.00

            # Molar ratio -> Mass ratio for H2/O2 pair
            mass_ratio_o2_h2 = (y_o2_molar * mw_o2) / ((1.0 - y_o2_molar) * mw_h2)
            o2_flow = h2_flow * mass_ratio_o2_h2

            # 2. Total Mass Flow
            total_mass_flow = h2_flow + steam_flow + o2_flow

            # 3. Calculate Mass Fractions
            if total_mass_flow > 0:
                w_h2 = h2_flow / total_mass_flow
                w_h2o = steam_flow / total_mass_flow
                w_o2 = o2_flow / total_mass_flow
            else:
                # Default safety state (Steam purge)
                w_h2, w_h2o, w_o2 = 0.0, 1.0, 0.0

            return Stream(
                mass_flow_kg_h=total_mass_flow,
                temperature_k=425.15,  # 152 °C (exit temp per legacy design)
                pressure_pa=self.out_pressure_pa,
                composition={'H2': w_h2, 'H2O': w_h2o, 'O2': w_o2},
                phase='gas'
            )

        elif port_name == 'o2_out':
            # === ANODE EXHAUST (Oxygen) ===
            from h2_plant.core.constants import ProductionConstants

            # Theoretical pure O2 production (Stoichiometric)
            o2_pure_flow = h2_flow * ProductionConstants.O2_TO_H2_MASS_RATIO

            # Impurity: H2 leak into O2 (e.g., 4000 ppm molar)
            y_h2_imp_molar = 0.0040
            mw_h2 = 2.016
            mw_o2 = 32.00

            # Calculate Mass of H2 impurity associated with this O2 production
            mass_ratio_h2_o2 = (y_h2_imp_molar * mw_h2) / ((1.0 - y_h2_imp_molar) * mw_o2)

            h2_leak_flow = o2_pure_flow * mass_ratio_h2_o2
            total_anode_flow = o2_pure_flow + h2_leak_flow

            if total_anode_flow > 0:
                w_o2 = o2_pure_flow / total_anode_flow
                w_h2 = h2_leak_flow / total_anode_flow
            else:
                w_o2, w_h2 = 1.0, 0.0

            return Stream(
                mass_flow_kg_h=total_anode_flow,
                temperature_k=425.15,  # 152 °C (exit temp per legacy design)
                pressure_pa=self.out_pressure_pa,
                composition={'O2': w_o2, 'H2': w_h2},
                phase='gas'
            )

        elif port_name == 'steam_out':
            # DEPRECATED: steam_out has been merged into h2_out (Wet Hydrogen).
            # Return a zero-flow stream with a warning for backward compatibility.
            import logging
            logging.getLogger(__name__).warning(
                "Port 'steam_out' is deprecated. Unreacted steam is now included in 'h2_out'."
            )
            return Stream(
                mass_flow_kg_h=0.0,
                temperature_k=425.15,  # 152 °C (exit temp per legacy design)
                pressure_pa=self.out_pressure_pa,
                composition={'H2O': 1.0},
                phase='gas'
            )

        else:
            raise ValueError(f"Unknown output port '{port_name}' on {self.component_id}")

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """
        Define the physical connection ports for this component.

        Returns:
            Dict[str, Dict[str, str]]: Port definitions with keys:
                - power_in: Electrical power from grid/coordinator.
                - steam_in: High-temperature steam feed.
                - h2_out: Cathode exhaust (Wet Hydrogen: H2 + H2O + trace O2).
                - o2_out: Anode exhaust (Oxygen + trace H2).
        """
        return {
            'power_in': {'type': 'input', 'resource_type': 'electricity', 'units': 'MW'},
            'steam_in': {'type': 'input', 'resource_type': 'water', 'units': 'kg/h'},
            'h2_out': {'type': 'output', 'resource_type': 'gas', 'units': 'kg/h'},  # Wet H2
            'o2_out': {'type': 'output', 'resource_type': 'oxygen', 'units': 'kg/h'}
        }

    def get_status(self) -> Dict[str, Any]:
        """
        Retrieve operational status of the SOEC system.

        Returns:
            Dict[str, Any]: Status dictionary with total power, active module
                count, cumulative production, and per-module arrays.
        """
        return {
            "total_power_mw": self.previous_total_power,
            "active_modules": int(np.sum(self.real_powers > 0.01)),
            "total_h2_kg": self.total_h2_produced,
            "module_powers": self.real_powers.tolist(),
            "module_states": self.real_states.tolist()
        }

    def get_state(self) -> Dict[str, Any]:
        """
        Retrieve the component's current operational state.

        Fulfills the Component Lifecycle Contract state access, combining
        base component state with SOEC-specific operational status.

        Returns:
            Dict[str, Any]: Complete state dictionary for monitoring and persistence.
        """
        return {
            **super().get_state(),
            **self.get_status(),
            # Expose unreacted steam for external logging (was previously steam_out)
            'last_water_output_kg': getattr(self, 'last_water_output_kg', 0.0)
        }
