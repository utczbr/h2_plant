import math
import numpy as np
import time
from scipy.interpolate import interp1d

# --- 1. SYSTEM CONSTANTS ---
NUM_MODULES = 6
MAX_NOMINAL_POWER = 2.4  # MW (Maximum Nominal Power)
OPTIMAL_LIMIT = 0.80 # 80%
FIRST_STEP_POWER = 0.12  # MW (Minimum Operating Power)
REAL_STAND_BY_POWER = 0.0   # MW (Hot Stand-by Power)
RAMP_STEP = 0.24  # MW (Increment/decrement step)
MINIMUM_TOTAL_POWER = 0.0 

# DEGRADATION LOOKUP TABLES
DEG_YEARS = np.array([0, 1, 2, 3, 3.5, 4, 5, 5.5, 6, 7])
DEG_EFFICIENCY_KWH_KG = np.array([37.5, 37.5, 37.5, 37.5, 37.5, 38, 39, 40, 41, 42])
DEG_CAPACITY_FACTOR = np.array([100, 100, 100, 100, 100, 100, 100, 90, 85, 75])

# Interpolators
efficiency_interpolator = interp1d(DEG_YEARS, DEG_EFFICIENCY_KWH_KG, kind='linear', fill_value="extrapolate")
capacity_interpolator = interp1d(DEG_YEARS, DEG_CAPACITY_FACTOR, kind='linear', fill_value="extrapolate")

# Default Base Consumption for Fallback (Year 0)
BASE_H2_CONSUMPTION_KWH_PER_KG = 37.5
# Steam Consumption Base (Year 0)
BASE_STEAM_CONSUMPTION_KG_PER_MWH = 672.0 / MAX_NOMINAL_POWER 

# Rotation Settings (Matches 'desgaste modulos 1 mes.py')
ROTATION_PERIOD_MINUTES = 60 

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

# --- 2. GLOBAL CONFIGURATIONS and STATE ---
# Global variables for internal function use (Simulating Limits)
global UNIFORM_MODULE_MAX_LIMIT
global FIXED_N_DENOMINATOR 
global NUM_ACTIVE_MODULES
global ROTATION_ENABLED
global GLOBAL_SOEC_STATE 
GLOBAL_SOEC_STATE = 0 

# --- CLASSE ATUALIZADA COM DESGASTE E CONTADORES ---
class SoecState:
    def __init__(self, real_powers, real_states, real_limits, virtual_map, difference_history, previous_total_power, current_minute, total_h2_produced, total_steam_consumed):
        self.real_powers = real_powers
        self.real_states = real_states
        self.real_limits = real_limits
        self.virtual_map = virtual_map
        self.difference_history = difference_history
        self.previous_total_power = previous_total_power
        self.current_minute = current_minute
        self.total_h2_produced = total_h2_produced 
        self.total_steam_consumed = total_steam_consumed
        
        # New: Degradation Parameters
        self.current_year = 0.0
        self.current_efficiency_kwh_kg = 37.0
        self.current_capacity_factor = 1.0 # 0.0 to 1.0
        self.effective_max_module_power = MAX_NOMINAL_POWER

        # New: Wear Tracking (Matches 'desgaste modulos 1 mes.py')
        self.accumulated_wear = np.zeros(NUM_MODULES, dtype=float) # MW * min
        self.cycle_counts = np.zeros(NUM_MODULES, dtype=int)
        self.previous_real_states = np.copy(real_states) # For cycle detection

def update_virtual_map(virtual_map):
    """Rotates the virtual map: Real ID i -> Virtual ID i+1."""
    return np.roll(virtual_map, -1)

# --- FUNÇÃO DE INICIALIZAÇÃO ATUALIZADA (Parâmetro 'year') ---
def initialize_soec_simulation(off_real_modules: list = [], rotation_enabled: bool = False, use_optimal_limit: bool = True, year: float = 0.0):
    """
    Initializes the SOEC state and configurations based on input parameters and Year of Operation.
    """
    global UNIFORM_MODULE_MAX_LIMIT
    global FIXED_N_DENOMINATOR
    global NUM_ACTIVE_MODULES
    global ROTATION_ENABLED
    
    ROTATION_ENABLED = rotation_enabled
    
    # 1. Calculate Degradation Factors
    eff_kwh_kg = float(efficiency_interpolator(year))
    cap_percent = float(capacity_interpolator(year))
    capacity_factor = cap_percent / 100.0
    
    # Calculate Effective Max Power based on degradation
    effective_max_nominal = MAX_NOMINAL_POWER * capacity_factor
    
    # 2. Set Global Limits based on DEGRADED Capacity
    if use_optimal_limit:
        UNIFORM_MODULE_MAX_LIMIT = effective_max_nominal * OPTIMAL_LIMIT 
    else:
        UNIFORM_MODULE_MAX_LIMIT = effective_max_nominal
        
    FIXED_N_DENOMINATOR = UNIFORM_MODULE_MAX_LIMIT

    # 3. Vector Initialization
    real_limits = np.full(NUM_MODULES, UNIFORM_MODULE_MAX_LIMIT, dtype=float)
    current_real_powers = np.full(NUM_MODULES, REAL_STAND_BY_POWER, dtype=float)
    current_real_states = np.full(NUM_MODULES, 1, dtype=int) # State 1: Hot Stand-by

    # 4. Apply Off Modules
    off_indices = [real_id - 1 for real_id in off_real_modules if 1 <= real_id <= NUM_MODULES]
    if off_indices:
        current_real_states[off_indices] = 0
        current_real_powers[off_indices] = 0.0
        real_limits[off_indices] = 0.0 

    # 5. Dynamic Virtual Map and Active Module Count
    active_indices = np.where(current_real_states != 0)[0]
    NUM_ACTIVE_MODULES = len(active_indices) 
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
    state.effective_max_module_power = UNIFORM_MODULE_MAX_LIMIT # Stores the limit used for logic

    return state

# --- 3. VECTOR-BASED UPDATE FUNCTIONS (Unchanged Logic, uses updated Globals) ---

def get_total_power(powers):
    return np.sum(powers)

def update_stationary_state(powers, states, limits):
    ramp_indices = np.where((states == 2) | (states == 5))[0]
    if len(ramp_indices) > 0:
        states[ramp_indices] = 3
    return states, limits

def simulate_step(reference_power, real_powers, real_states, real_limits, virtual_map, difference_history):
    """
    Executes a simulation step. Uses UNIFORM_MODULE_MAX_LIMIT which is now 
    degraded based on the year initialized.
    """
    global FIXED_N_DENOMINATOR 
    global UNIFORM_MODULE_MAX_LIMIT
    global NUM_ACTIVE_MODULES 

    # --- 0. VIRTUAL -> REAL MAPPING ---
    powers_v = real_powers[virtual_map].copy()
    states_v = real_states[virtual_map].copy()
    limits_v = real_limits[virtual_map].copy()
    
    requested_power = get_total_power(powers_v) 
    difference = reference_power - requested_power
    difference_history.append(difference)

    tolerance = 0.005 

    # 1. Promote Stationary (3) -> Optimal (4)
    # UNIFORM_MODULE_MAX_LIMIT is now the degraded limit
    promotion_indices = np.where(
        (states_v == 3) & 
        (np.abs(powers_v - UNIFORM_MODULE_MAX_LIMIT) < tolerance)
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
    numerator = max(0.0, reference_power - MINIMUM_TOTAL_POWER)
    broken_number = numerator / FIXED_N_DENOMINATOR
    
    N_ceil = math.ceil(broken_number)
    
    if broken_number > 0.01 and abs(broken_number - N_ceil) < 0.001:
        target_module_id = int(N_ceil) + 1
    else:
        target_module_id = int(N_ceil)
        
    target_module_id = max(1, min(NUM_ACTIVE_MODULES + 1, target_module_id)) 
    target_index = target_module_id - 1 

    # Base Limit
    N_floor = math.floor(broken_number)
    base_limit_term = UNIFORM_MODULE_MAX_LIMIT * N_floor
    standby_term = 0.0 
    new_limit_calc = reference_power - base_limit_term - standby_term

    # --- 4. DEFINITION OF DYNAMIC LIMITS ---
    active_limit = limits_v 
    
    if new_limit_calc > REAL_STAND_BY_POWER + tolerance and new_limit_calc < FIRST_STEP_POWER - tolerance:
        new_limit_calc = REAL_STAND_BY_POWER 
    
    if difference > 0: # RAMP UP
        inactive_limit_const = REAL_STAND_BY_POWER 
        if target_module_id > NUM_ACTIVE_MODULES:
            active_limit[:] = UNIFORM_MODULE_MAX_LIMIT
        else:
            active_limit[target_index] = max(inactive_limit_const, new_limit_calc)
            active_limit[:target_index] = UNIFORM_MODULE_MAX_LIMIT
        active_limit[target_index + 1:] = inactive_limit_const 

    else: # RAMP DOWN
        if abs(reference_power - MINIMUM_TOTAL_POWER) < 0.01:
            active_limit[:] = REAL_STAND_BY_POWER
        else:
            active_limit[target_index] = max(REAL_STAND_BY_POWER, new_limit_calc)
            active_limit[:target_index] = UNIFORM_MODULE_MAX_LIMIT
            active_limit[target_index + 1:] = REAL_STAND_BY_POWER 

    # --- 5. INDIVIDUALIZED DISPATCH ---
    difference_to_limit = active_limit - powers_v 
    movement = np.minimum(np.abs(difference_to_limit), RAMP_STEP)
    movement *= np.sign(difference_to_limit)

    startup_indices = np.where(
        (np.abs(powers_v - REAL_STAND_BY_POWER) < tolerance) & 
        (difference_to_limit > tolerance) 
    )[0]
    
    if len(startup_indices) > 0:
        powers_v[startup_indices] = FIRST_STEP_POWER
        movement[startup_indices] = 0.0 

    powers_v += movement 
    
    indices_to_shutdown_by_cutoff = np.where(
        (powers_v > REAL_STAND_BY_POWER + tolerance) & 
        (powers_v < FIRST_STEP_POWER - tolerance)  
    )[0]
    
    if len(indices_to_shutdown_by_cutoff) > 0:
        powers_v[indices_to_shutdown_by_cutoff] = REAL_STAND_BY_POWER
    
    final_hot_standby_indices = np.where(powers_v <= REAL_STAND_BY_POWER + tolerance)[0]
    powers_v[final_hot_standby_indices] = REAL_STAND_BY_POWER

    # --- 6. STATE UPDATE ---
    states_v = np.where(
        np.abs(powers_v - REAL_STAND_BY_POWER) < tolerance,
        1, # Hot Stand-by
        np.where(
            np.abs(powers_v - UNIFORM_MODULE_MAX_LIMIT) < tolerance,
            4, # Optimal (Now Degraded Optimal)
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


# --- FUNÇÃO PRINCIPAL ATUALIZADA (Desgaste e Rotação Avançada) ---
def run_soec_step(reference_power, soec_state: SoecState):
    """
    Executes a 1-minute step with:
    1. Degradation Logic (Year-based Efficiency & Capacity)
    2. Sophisticated Rotation (60 min)
    3. Wear Tracking (Accumulated & Cycles)
    """
    global GLOBAL_SOEC_STATE
    global ROTATION_ENABLED
    global UNIFORM_MODULE_MAX_LIMIT # Used to clamp input
    
    # 0. Safety Clamp on Input (Due to Degradation Capacity Fade)
    # Limit the setpoint to the maximum current physical capability
    system_max_capacity = NUM_MODULES * soec_state.effective_max_module_power
    clamped_reference = min(reference_power, system_max_capacity)

    # 1. ROTATION LOGIC (Refined: Matches 'desgaste modulos 1 mes.py')
    if ROTATION_ENABLED and soec_state.current_minute > 0 and soec_state.current_minute % ROTATION_PERIOD_MINUTES == 0:
        soec_state.virtual_map = update_virtual_map(soec_state.virtual_map)
    
    # 2. EXECUTE SIMULATION STEP
    soec_state.real_powers, soec_state.real_states, soec_state.real_limits, soec_state.difference_history = simulate_step(
        clamped_reference,
        soec_state.real_powers, 
        soec_state.real_states, 
        soec_state.real_limits,
        soec_state.virtual_map,
        soec_state.difference_history
    )
    
    current_total_power = get_total_power(soec_state.real_powers)
    
    # 3. WEAR TRACKING (New)
    # A. Accumulated Wear (MW * min)
    soec_state.accumulated_wear += soec_state.real_powers
    
    # B. Cycle Counting (Hot Stand-by (1) -> Ramp Up (2))
    for i in range(NUM_MODULES):
        prev = soec_state.previous_real_states[i]
        curr = soec_state.real_states[i]
        if prev == 1 and curr == 2:
            soec_state.cycle_counts[i] += 1
            
    # Update previous states for next step
    soec_state.previous_real_states = np.copy(soec_state.real_states)

    # 4. DEGRADED PRODUCTION CALCULATION
    # Energy MWh
    energy_consumed_mwh = current_total_power / 60.0
    
    # Calculate degraded rates
    eff_kwh_kg = soec_state.current_efficiency_kwh_kg
    h2_prod_rate_kg_mwh = 1000.0 / eff_kwh_kg
    
    # H2 produced (kg)
    h2_produced_kg = energy_consumed_mwh * h2_prod_rate_kg_mwh
    soec_state.total_h2_produced += h2_produced_kg 
    
    # Steam consumed (kg) - Using Base efficiency for steam calculation or scaled?
    # Usually steam consumption is proportional to H2 produced (mass balance).
    # 9 kg H2O -> 1 kg H2. So Steam = H2 * 9.
    steam_consumed_kg = h2_produced_kg * 9.0 
    soec_state.total_steam_consumed += steam_consumed_kg
    
    # 5. GLOBAL STATE CALCULATION
    global_tolerance = 0.005
    previous_real_power = soec_state.previous_total_power 
    
    if current_total_power > previous_real_power + global_tolerance:
        GLOBAL_SOEC_STATE = 1 # Ramp Up
    elif current_total_power < previous_real_power - global_tolerance:
        GLOBAL_SOEC_STATE = 2 # Ramp Down
    else:
        GLOBAL_SOEC_STATE = 0 

    # 6. STEP UPDATE
    soec_state.previous_total_power = current_total_power
    soec_state.current_minute += 1
    
    return current_total_power, soec_state, h2_produced_kg, steam_consumed_kg