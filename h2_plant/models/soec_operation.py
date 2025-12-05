"""
SOEC Operation Model with Degradation Tracking
Based on soec_operator.py reference implementation
"""
import math
import numpy as np
from typing import Tuple, List
import math
import numpy as np
from typing import Tuple, List
# from scipy.interpolate import interp1d # Removed dependency

from h2_plant.config.constants_physics import SOECConstants

# Instantiate constants
CONST = SOECConstants()

# --- DEGRADATION LOOKUP TABLES ---
DEG_YEARS = np.array([0, 1, 2, 3, 3.5, 4, 5, 5.5, 6, 7])
DEG_EFFICIENCY_KWH_KG = np.array([37.5, 37.5, 37.5, 37.5, 37.5, 38, 39, 40, 41, 42])
DEG_CAPACITY_FACTOR = np.array([100, 100, 100, 100, 100, 100, 100, 90, 85, 75])

# Interpolators (Using numpy.interp)
# efficiency_interpolator = interp1d(DEG_YEARS, DEG_EFFICIENCY_KWH_KG, kind='linear', fill_value="extrapolate")
# capacity_interpolator = interp1d(DEG_YEARS, DEG_CAPACITY_FACTOR, kind='linear', fill_value="extrapolate")

def interpolate(x, xp, fp):
    return np.interp(x, xp, fp)

# Module State Enumeration
STATES = {
    0: "Off", 
    1: "Hot Stand-by",
    2: "Ramp Up",
    3: "Stationary",
    4: "Optimal",
    5: "Ramp Down"
}

# Global SOEC State Enumeration
SOEC_GLOBAL_STATE = {
    0: "Stand-by/Stationary",
    1: "Ramp Up (System)",
    2: "Ramp Down (System)"
}


class SoecState:
    """SOEC state with degradation and wear tracking"""
    def __init__(self, real_powers, real_states, real_limits, virtual_map, 
                 difference_history, previous_total_power, current_minute, 
                 total_h2_produced, total_steam_consumed):
        self.real_powers = real_powers
        self.real_states = real_states
        self.real_limits = real_limits
        self.virtual_map = virtual_map
        self.difference_history = difference_history
        self.previous_total_power = previous_total_power
        self.current_minute = current_minute
        self.total_h2_produced = total_h2_produced 
        self.total_steam_consumed = total_steam_consumed
        
        # Degradation Parameters
        self.current_year = 0.0
        self.current_efficiency_kwh_kg = 37.5
        self.current_capacity_factor = 1.0  # 0.0 to 1.0
        self.effective_max_module_power = CONST.MAX_POWER_NOMINAL_MW
        
        # Wear Tracking
        self.accumulated_wear = np.zeros(len(real_powers), dtype=float)  # MW * min
        self.cycle_counts = np.zeros(len(real_powers), dtype=int)
        self.previous_real_states = np.copy(real_states)  # For cycle detection


def update_virtual_map(virtual_map):
    """Rotates the virtual map: Real ID i -> Virtual ID i+1."""
    return np.roll(virtual_map, -1)


def initialize_soec_simulation(off_real_modules: list = [], rotation_enabled: bool = False, 
                               use_optimal_limit: bool = True, year: float = 0.0, 
                               num_modules: int = None):
    """
    Initializes the SOEC state and configurations based on input parameters and Year of Operation.
    
    Args:
        off_real_modules: List of module IDs (1-indexed) that are offline
        rotation_enabled: Enable rotation logic
        use_optimal_limit: Use 80% optimal limit vs 100% max
        year: Operational age in years (for degradation)
        num_modules: Number of modules (default from CONST)
    """
    if num_modules is None:
        num_modules = CONST.NUM_MODULES
    
    # 1. Calculate Degradation Factors
    eff_kwh_kg = float(interpolate(year, DEG_YEARS, DEG_EFFICIENCY_KWH_KG))
    cap_percent = float(interpolate(year, DEG_YEARS, DEG_CAPACITY_FACTOR))
    capacity_factor = cap_percent / 100.0
    
    # Calculate Effective Max Power based on degradation
    effective_max_nominal = CONST.MAX_POWER_NOMINAL_MW * capacity_factor
    
    # 2. Set Global Limits based on DEGRADED Capacity
    if use_optimal_limit:
        uniform_module_max_limit = effective_max_nominal * CONST.LIMIT_OPTIMAL_RATIO
    else:
        uniform_module_max_limit = effective_max_nominal
    
    # 3. Vector Initialization
    real_limits = np.full(num_modules, uniform_module_max_limit, dtype=float)
    current_real_powers = np.full(num_modules, CONST.POWER_STANDBY_MW, dtype=float)
    current_real_states = np.full(num_modules, 1, dtype=int)  # State 1: Hot Stand-by
    
    # 4. Apply Off Modules
    off_indices = [real_id - 1 for real_id in off_real_modules if 1 <= real_id <= num_modules]
    if off_indices:
        current_real_states[off_indices] = 0
        current_real_powers[off_indices] = 0.0
        real_limits[off_indices] = 0.0
    
    # 5. Dynamic Virtual Map and Active Module Count
    active_indices = np.where(current_real_states != 0)[0]
    virtual_map = active_indices
    
    initial_total_power = np.sum(current_real_powers)
    
    # Create State Object
    state = SoecState(
        current_real_powers, 
        current_real_states, 
        real_limits, 
        virtual_map, 
        [], 
        initial_total_power, 
        0, 
        0.0, 
        0.0
    )
    
    # Store Degradation Context in State
    state.current_year = year
    state.current_efficiency_kwh_kg = eff_kwh_kg
    state.current_capacity_factor = capacity_factor
    state.effective_max_module_power = uniform_module_max_limit
    
    return state


def get_total_power(powers):
    return np.sum(powers)


def update_stationary_state(powers, states, limits):
    ramp_indices = np.where((states == 2) | (states == 5))[0]
    if len(ramp_indices) > 0:
        states[ramp_indices] = 3
    return states, limits


def simulate_step(reference_power, real_powers, real_states, real_limits, virtual_map, 
                 difference_history, uniform_module_max_limit, num_active_modules):
    """
    Executes a simulation step using degraded limits.
    
    Args:
        reference_power: Target power setpoint
        real_powers: Current power per module
        real_states: Current state per module
        real_limits: Current limits per module
        virtual_map: Virtual to real mapping
        difference_history: History of reference-actual differences
        uniform_module_max_limit: Degraded max limit per module
        num_active_modules: Number of active modules
    """
    # --- 0. VIRTUAL -> REAL MAPPING ---
    powers_v = real_powers[virtual_map].copy()
    states_v = real_states[virtual_map].copy()
    limits_v = real_limits[virtual_map].copy()
    
    requested_power = get_total_power(powers_v) 
    difference = reference_power - requested_power
    difference_history.append(difference)
    
    tolerance = 0.005 
    
    # 1. Promote Stationary (3) -> Optimal (4)
    promotion_indices = np.where(
        (states_v == 3) & 
        (np.abs(powers_v - uniform_module_max_limit) < tolerance)
    )[0] 
    if len(promotion_indices) > 0:
        states_v[promotion_indices] = 4
    
    # --- CHECK STATIONARY ---
    if abs(difference) < 0.01:
        states_v, limits_v = update_stationary_state(powers_v, states_v, limits_v)
        real_powers[virtual_map] = powers_v
        real_states[virtual_map] = states_v
        real_limits[virtual_map] = limits_v 
        return real_powers, real_states, real_limits, difference_history
 
    # --- DYNAMIC N CALCULATION ---
    numerator = max(0.0, reference_power - CONST.POWER_STANDBY_MW)
    broken_number = numerator / uniform_module_max_limit
    
    N_ceil = math.ceil(broken_number)
    
    if broken_number > 0.01 and abs(broken_number - N_ceil) < 0.001:
        target_module_id = int(N_ceil) + 1
    else:
        target_module_id = int(N_ceil)
        
    target_module_id = max(1, min(num_active_modules + 1, target_module_id)) 
    target_index = target_module_id - 1 
    
    # Base Limit
    N_floor = math.floor(broken_number)
    base_limit_term = uniform_module_max_limit * N_floor
    standby_term = 0.0 
    new_limit_calc = reference_power - base_limit_term - standby_term
    
    # --- 4. DEFINITION OF DYNAMIC LIMITS ---
    active_limit = limits_v 
    
    if new_limit_calc > CONST.POWER_STANDBY_MW + tolerance and new_limit_calc < CONST.POWER_FIRST_STEP_MW - tolerance:
        new_limit_calc = CONST.POWER_STANDBY_MW 
    
    if difference > 0:  # RAMP UP
        inactive_limit_const = CONST.POWER_STANDBY_MW 
        if target_module_id > num_active_modules:
            active_limit[:] = uniform_module_max_limit
        else:
            active_limit[target_index] = max(inactive_limit_const, new_limit_calc)
            active_limit[:target_index] = uniform_module_max_limit
        active_limit[target_index + 1:] = inactive_limit_const 
    
    else:  # RAMP DOWN
        if abs(reference_power - CONST.POWER_STANDBY_MW) < 0.01:
            active_limit[:] = CONST.POWER_STANDBY_MW
        else:
            active_limit[target_index] = max(CONST.POWER_STANDBY_MW, new_limit_calc)
            active_limit[:target_index] = uniform_module_max_limit
            active_limit[target_index + 1:] = CONST.POWER_STANDBY_MW 
    
    # --- 5. INDIVIDUALIZED DISPATCH ---
    difference_to_limit = active_limit - powers_v 
    movement = np.minimum(np.abs(difference_to_limit), CONST.RAMP_STEP_MW)
    movement *= np.sign(difference_to_limit)
    
    startup_indices = np.where(
        (np.abs(powers_v - CONST.POWER_STANDBY_MW) < tolerance) & 
        (difference_to_limit > tolerance) 
    )[0]
    
    if len(startup_indices) > 0:
        powers_v[startup_indices] = CONST.POWER_FIRST_STEP_MW
        movement[startup_indices] = 0.0 
    
    powers_v += movement 
    
    indices_to_shutdown_by_cutoff = np.where(
        (powers_v > CONST.POWER_STANDBY_MW + tolerance) & 
        (powers_v < CONST.POWER_FIRST_STEP_MW - tolerance)  
    )[0]
    
    if len(indices_to_shutdown_by_cutoff) > 0:
        powers_v[indices_to_shutdown_by_cutoff] = CONST.POWER_STANDBY_MW
    
    final_hot_standby_indices = np.where(powers_v <= CONST.POWER_STANDBY_MW + tolerance)[0]
    powers_v[final_hot_standby_indices] = CONST.POWER_STANDBY_MW
    
    # --- 6. STATE UPDATE ---
    states_v = np.where(
        np.abs(powers_v - CONST.POWER_STANDBY_MW) < tolerance,
        1,  # Hot Stand-by
        np.where(
            np.abs(powers_v - uniform_module_max_limit) < tolerance,
            4,  # Optimal (Now Degraded Optimal)
            np.where(
                np.abs(powers_v - active_limit) < tolerance,
                3, 
                np.where(
                    difference_to_limit > 0,
                    2, 
                    5  
                )
            )
        )
    )
    
    real_powers[virtual_map] = powers_v
    real_states[virtual_map] = states_v
    real_limits[virtual_map] = active_limit
    
    return real_powers, real_states, real_limits, difference_history


def run_soec_step(reference_power, soec_state: SoecState, rotation_period_minutes: int = 60):
    """
    Executes a 1-minute step with:
    1. Degradation Logic (Year-based Efficiency & Capacity)
    2. Sophisticated Rotation
    3. Wear Tracking (Accumulated & Cycles)
    
    Args:
        reference_power: Power setpoint (MW)
        soec_state: Current SOEC state
        rotation_period_minutes: Rotation period (default 60 min)
    
    Returns:
        current_total_power, soec_state, h2_produced_kg, steam_consumed_kg
    """
    num_modules = len(soec_state.real_powers)
    
    # 0. Safety Clamp on Input (Due to Degradation Capacity Fade)
    system_max_capacity = num_modules * soec_state.effective_max_module_power
    clamped_reference = min(reference_power, system_max_capacity)
    
    # 1. ROTATION LOGIC
    if soec_state.current_minute > 0 and soec_state.current_minute % rotation_period_minutes == 0:
        soec_state.virtual_map = update_virtual_map(soec_state.virtual_map)
    
    # 2. EXECUTE SIMULATION STEP
    num_active_modules = len(soec_state.virtual_map)
    
    soec_state.real_powers, soec_state.real_states, soec_state.real_limits, soec_state.difference_history = simulate_step(
        clamped_reference,
        soec_state.real_powers, 
        soec_state.real_states, 
        soec_state.real_limits,
        soec_state.virtual_map,
        soec_state.difference_history,
        soec_state.effective_max_module_power,
        num_active_modules
    )
    
    current_total_power = get_total_power(soec_state.real_powers)
    
    # 3. WEAR TRACKING
    # A. Accumulated Wear (MW * min)
    soec_state.accumulated_wear += soec_state.real_powers
    
    # B. Cycle Counting (Hot Stand-by (1) -> Ramp Up (2))
    for i in range(num_modules):
        prev = soec_state.previous_real_states[i]
        curr = soec_state.real_states[i]
        if prev == 1 and curr == 2:
            soec_state.cycle_counts[i] += 1
            
    # Update previous states for next step
    soec_state.previous_real_states = np.copy(soec_state.real_states)
    
    # 4. DEGRADED PRODUCTION CALCULATION
    # Energy MWh (1 minute = 1/60 hour)
    energy_consumed_mwh = current_total_power / 60.0
    
    # Calculate degraded rates
    eff_kwh_kg = soec_state.current_efficiency_kwh_kg
    h2_prod_rate_kg_mwh = 1000.0 / eff_kwh_kg
    
    # H2 produced (kg)
    h2_produced_kg = energy_consumed_mwh * h2_prod_rate_kg_mwh
    soec_state.total_h2_produced += h2_produced_kg 
    
    # Steam consumed (kg) - Stoichiometric: 9 kg H2O -> 1 kg H2
    steam_consumed_kg = h2_produced_kg * 9.0 
    soec_state.total_steam_consumed += steam_consumed_kg
    
    # 5. STEP UPDATE
    soec_state.previous_total_power = current_total_power
    soec_state.current_minute += 1
    
    return current_total_power, soec_state, h2_produced_kg, steam_consumed_kg


# Backward compatibility wrappers
def atualizar_mapa_virtual(mapa_virtual: np.ndarray) -> np.ndarray:
    """Backward compatibility wrapper"""
    return update_virtual_map(mapa_virtual)


def simular_passo_soec(potencia_referencia: float, potencias_atuais_reais: np.ndarray, 
                      estados_atuais_reais: np.ndarray, limites_reais: np.ndarray, 
                      mapa_virtual: np.ndarray, rotacao_ativada: bool,
                      modulos_desligados_reais: List[int], potencia_limite_eficiente: bool,
                      year: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray,  np.ndarray, float]:
    """
    Backward compatibility wrapper for old API.
    Creates a temporary SoecState and executes one step.
    """
    # Create temporary state
    num_modules = len(potencias_atuais_reais)
    state = SoecState(
        potencias_atuais_reais.copy(),
        estados_atuais_reais.copy(),
        limites_reais.copy(),
        mapa_virtual.copy(),
        [],
        np.sum(potencias_atuais_reais),
        0,
        0.0,
        0.0
    )
    
    # Set degradation
    state.current_year = year
    state.current_efficiency_kwh_kg = float(interpolate(year, DEG_YEARS, DEG_EFFICIENCY_KWH_KG))
    state.current_capacity_factor = float(interpolate(year, DEG_YEARS, DEG_CAPACITY_FACTOR)) / 100.0
    
    if potencia_limite_eficiente:
        state.effective_max_module_power = CONST.MAX_POWER_NOMINAL_MW * state.current_capacity_factor * CONST.LIMIT_OPTIMAL_RATIO
    else:
        state.effective_max_module_power = CONST.MAX_POWER_NOMINAL_MW * state.current_capacity_factor
    
    # Run step
    _, state, _, _ = run_soec_step(potencia_referencia, state, rotation_period_minutes=60)
    
    return state.real_powers, state.real_states, state.real_limits, state.virtual_map, np.sum(state.real_powers)
