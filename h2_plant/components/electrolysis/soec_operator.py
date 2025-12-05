import numpy as np
import math
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import math
from typing import Dict, Any, Tuple, List, Optional
# from scipy.interpolate import interp1d # Removed dependency

# Constants (Fallback if not provided in config)
DEFAULT_NUM_MODULES = 6
DEFAULT_MAX_POWER_NOMINAL_MW = 2.4
DEFAULT_OPTIMAL_LIMIT = 0.80
DEFAULT_POWER_STANDBY_MW = 0.0
DEFAULT_POWER_FIRST_STEP_MW = 0.12
DEFAULT_RAMP_STEP_MW = 0.24

# Degradation Tables (Hardcoded as per reference)
DEG_YEARS = np.array([0, 1, 2, 3, 3.5, 4, 5, 5.5, 6, 7])
DEG_EFFICIENCY_KWH_KG = np.array([37.5, 37.5, 37.5, 37.5, 37.5, 38, 39, 40, 41, 42])
DEG_CAPACITY_FACTOR = np.array([100, 100, 100, 100, 100, 100, 100, 90, 85, 75])

from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry

class SOECOperator(Component):
    """
    SOEC Operator Component.
    Manages the operation of Solid Oxide Electrolyzer Cell modules, including
    degradation, rotation, and state management.
    """
    def __init__(self, config: Any, physics_config: Dict[str, Any] = None):
        """
        Args:
            config: SOECPhysicsSpec OR legacy config dict.
            physics_config: Legacy physics dict (optional if using Spec).
        """
        super().__init__(config) 
        
        # Handle Pydantic Model vs Legacy Dict
        if hasattr(config, 'max_power_nominal_mw'):
            # It's a SOECPhysicsSpec
            self.spec = config
            self.num_modules = config.num_modules
            self.max_nominal_power = config.max_power_nominal_mw
            self.optimal_limit_ratio = config.optimal_limit
            self.rotation_enabled = False # Not in spec yet, default False
            self.degradation_year = 0.0   # Not in spec yet, default 0.0
            
            self.power_standby_mw = DEFAULT_POWER_STANDBY_MW
            self.power_first_step_mw = config.power_first_step_mw
            self.ramp_step_mw = config.ramp_step_mw
            self.steam_input_ratio = getattr(config, 'steam_input_ratio_kg_per_kg_h2', 10.5)
            
            # Legacy compatibility
            self.config = config.dict()
            self.physics_config = {} 
            
        else:
            # Legacy Mode
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
            self.steam_input_ratio = soec_phys.get("steam_input_ratio_kg_per_kg_h2", 10.5)
        
        # Initialize Interpolators (Using numpy.interp)
        # self.efficiency_interpolator = interp1d(DEG_YEARS, DEG_EFFICIENCY_KWH_KG, kind='linear', fill_value="extrapolate")
        # self.capacity_interpolator = interp1d(DEG_YEARS, DEG_CAPACITY_FACTOR, kind='linear', fill_value="extrapolate")
        
        # Initialize State
        self._initialize_state()

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """Initialize component."""
        super().initialize(dt, registry)
        # Any specific initialization if needed


    def _interpolate(self, x, xp, fp):
        """Simple linear interpolation wrapper using numpy."""
        return np.interp(x, xp, fp)

    def _initialize_state(self):
        """Initializes the internal state of the SOEC system."""
        # 1. Calculate Degradation Factors
        self.current_efficiency_kwh_kg = float(self._interpolate(self.degradation_year, DEG_YEARS, DEG_EFFICIENCY_KWH_KG))
        cap_percent = float(self._interpolate(self.degradation_year, DEG_YEARS, DEG_CAPACITY_FACTOR))
        self.current_capacity_factor = cap_percent / 100.0
        
        # Calculate Effective Max Power
        self.effective_max_module_power = self.max_nominal_power * self.current_capacity_factor
        
        # 2. Set Global Limits
        self.uniform_module_max_limit = self.effective_max_module_power * self.optimal_limit_ratio
        
        # 3. Vector Initialization
        self.real_limits = np.full(self.num_modules, self.uniform_module_max_limit, dtype=float)
        self.real_powers = np.full(self.num_modules, self.power_standby_mw, dtype=float)
        self.real_states = np.full(self.num_modules, 1, dtype=int) # State 1: Hot Stand-by
        
        # 4. Apply Off Modules (if any)
        off_modules = self.config.get("real_off_modules", [])
        off_indices = [mid - 1 for mid in off_modules if 1 <= mid <= self.num_modules]
        if off_indices:
            self.real_states[off_indices] = 0
            self.real_powers[off_indices] = 0.0
            self.real_limits[off_indices] = 0.0
            
        # 5. Virtual Map
        active_indices = np.where(self.real_states != 0)[0]
        self.virtual_map = active_indices
        
        # History & Counters
        self.difference_history = []
        self.previous_total_power = np.sum(self.real_powers)
        self.current_minute = 0
        self.total_h2_produced = 0.0
        self.total_steam_consumed = 0.0
        
        self.accumulated_wear = np.zeros(self.num_modules, dtype=float)
        self.cycle_counts = np.zeros(self.num_modules, dtype=int)
        self.previous_real_states = np.copy(self.real_states)

    def _update_virtual_map(self):
        """Rotates the virtual map."""
        self.virtual_map = np.roll(self.virtual_map, -1)

    def _get_total_power(self, powers):
        return np.sum(powers)

    def _update_stationary_state(self, powers, states, limits):
        ramp_indices = np.where((states == 2) | (states == 5))[0]
        if len(ramp_indices) > 0:
            states[ramp_indices] = 3
        return states, limits

    def _simulate_step(self, reference_power: float):
        """Executes the core simulation logic for one step using JIT."""
        from h2_plant.optimization.numba_ops import simulate_soec_step_jit
        
        # Ensure arrays are correct type for Numba
        # real_powers: float64, real_states: int32, real_limits: float64, virtual_map: int32
        
        # Call JIT function
        self.real_powers, self.real_states, self.real_limits = simulate_soec_step_jit(
            reference_power,
            self.real_powers,
            self.real_states.astype(np.int32), # Ensure int32
            self.real_limits,
            self.virtual_map.astype(np.int32), # Ensure int32
            self.uniform_module_max_limit,
            self.power_standby_mw,
            self.power_first_step_mw,
            self.ramp_step_mw,
            0.0 # minimum_total_power
        )
        
        # Calculate difference for history (optional, but good for parity)
        current_total = np.sum(self.real_powers)
        diff = reference_power - current_total
        self.difference_history.append(diff)

    def step(self, reference_power_mw: float = None, t: float = None) -> Tuple[float, float, float]:
        """
        Executes one simulation step (1 minute).
        
        Args:
            reference_power_mw: Target power setpoint.
            t: Simulation time (optional, for Component compatibility)
            
        Returns:
            Tuple(actual_power_mw, h2_produced_kg, steam_consumed_kg)
        """
        # Compatibility handling
        if reference_power_mw is None:
            # If called as step(t), we need a stored setpoint.
            # For now, assume 0.0 or raise error if not set?
            # Or just default to 0.0
            reference_power_mw = 0.0

        # 0. Safety Clamp
        system_max_capacity = self.num_modules * self.effective_max_module_power
        clamped_reference = min(reference_power_mw, system_max_capacity)
        
        # 1. Rotation Logic
        if self.rotation_enabled and self.current_minute > 0 and self.current_minute % 60 == 0:
            self._update_virtual_map()
            
        # 2. Execute Simulation Step
        self._simulate_step(clamped_reference)
        
        current_total_power = self._get_total_power(self.real_powers)
        
        # 3. Wear Tracking
        self.accumulated_wear += self.real_powers
        
        for i in range(self.num_modules):
            prev = self.previous_real_states[i]
            curr = self.real_states[i]
            if prev == 1 and curr == 2:
                self.cycle_counts[i] += 1
                
        self.previous_real_states = np.copy(self.real_states)
        
        # 4. Production Calculation
        # Use self.dt (in hours) for energy calculation
        if hasattr(self, 'dt'):
            energy_consumed_mwh = current_total_power * self.dt
        else:
            # Fallback for legacy calls without initialize()
            energy_consumed_mwh = current_total_power / 60.0
            
        h2_prod_rate_kg_mwh = 1000.0 / self.current_efficiency_kwh_kg
        
        h2_produced_kg = energy_consumed_mwh * h2_prod_rate_kg_mwh
        self.total_h2_produced += h2_produced_kg
        self.last_step_h2_kg = h2_produced_kg # Store for get_output
        
        # Steam Input (Total Flow)
        steam_input_kg = h2_produced_kg * self.steam_input_ratio
        self.total_steam_consumed += steam_input_kg # Tracking total input
        
        # Water Output (Unreacted Steam)
        # Reaction consumes 9.0 kg H2O per kg H2
        reaction_steam_kg = h2_produced_kg * 9.0
        self.last_water_output_kg = max(0.0, steam_input_kg - reaction_steam_kg)
        
        # 5. Update Step
        self.previous_total_power = current_total_power
        self.current_minute += 1
        
        return current_total_power, h2_produced_kg, steam_input_kg

    def get_output(self, port_name: str) -> Any:
        """Get output from specific port."""
        from h2_plant.core.stream import Stream
        
        if port_name == 'h2_out':
            # Create H2 stream from last step production
            # total_h2_produced is cumulative, we need the rate from the last step.
            # We can store last_h2_kg in step() or calculate it.
            # step() returns h2_produced_kg, let's store it in self.last_h2_kg
            
            # If we haven't stored it, we might return 0 or estimate.
            # Better to update step() to store it.
            h2_kg = getattr(self, 'last_step_h2_kg', 0.0)
            
            return Stream(
                mass_flow_kg_h=h2_kg * (1.0/self.dt) if self.dt > 0 else 0.0, # Convert kg/step to kg/h
                temperature_k=1073.15, # ~800C
                pressure_pa=101325.0,  # Atmospheric or slightly pressurized
                composition={'H2': 1.0},
                phase='gas'
            )
        elif port_name == 'steam_out':
             # Unreacted steam
             water_kg = getattr(self, 'last_water_output_kg', 0.0)
             return Stream(
                mass_flow_kg_h=water_kg * (1.0/self.dt) if self.dt > 0 else 0.0,
                temperature_k=1073.15,
                pressure_pa=101325.0,
                composition={'H2O': 1.0},
                phase='gas'
            )
        else:
            raise ValueError(f"Unknown output port '{port_name}' on {self.component_id}")

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        return {
            'power_in': {'type': 'input', 'resource_type': 'electricity', 'units': 'MW'},
            'h2_out': {'type': 'output', 'resource_type': 'hydrogen', 'units': 'kg/h'},
            'steam_out': {'type': 'output', 'resource_type': 'water', 'units': 'kg/h'}
        }

    def get_status(self) -> Dict[str, Any]:
        """Returns the current status of the SOEC system."""
        return {
            "total_power_mw": self.previous_total_power,
            "active_modules": int(np.sum(self.real_powers > 0.01)),
            "total_h2_kg": self.total_h2_produced,
            "module_powers": self.real_powers.tolist(),
            "module_states": self.real_states.tolist()
        }

    def get_state(self) -> Dict[str, Any]:
        """Component ABC implementation."""
        return {
            **super().get_state(),
            **self.get_status()
        }
