"""
Numba JIT-compiled operations for hot path performance.

All functions decorated with @njit compile to native machine code,
achieving near-C performance for numerical operations.
"""

import numpy as np
import numpy.typing as npt
from numba import njit
from typing import Tuple

from h2_plant.core.enums import TankState
from h2_plant.core.constants import GasConstants


@njit
def find_available_tank(
    states: npt.NDArray[np.int32],
    masses: npt.NDArray[np.float64],
    capacities: npt.NDArray[np.float64],
    min_capacity: float = 0.0
) -> int:
    """
    Find first idle tank with sufficient available capacity.
    
    Args:
        states: Array of TankState values (IntEnum)
        masses: Array of current masses (kg)
        capacities: Array of tank capacities (kg)
        min_capacity: Minimum required available capacity (kg)
        
    Returns:
        Index of suitable tank, or -1 if none found
        
    Example:
        states = np.array([TankState.FULL, TankState.IDLE, TankState.IDLE], dtype=np.int32)
        masses = np.array([200.0, 50.0, 0.0], dtype=np.float64)
        capacities = np.array([200.0, 200.0, 200.0], dtype=np.float64)
        
        idx = find_available_tank(states, masses, capacities, min_capacity=100.0)
        # Returns 1 (has 150 kg available) or 2 (has 200 kg available)
    """
    for i in range(len(states)):
        available_capacity = capacities[i] - masses[i]
        if states[i] == TankState.IDLE and available_capacity >= min_capacity:
            return i
    return -1


@njit
def find_fullest_tank(
    states: npt.NDArray[np.int32],
    masses: npt.NDArray[np.float64],
    min_mass: float = 0.0
) -> int:
    """
    Find fullest tank available for discharge.
    
    Args:
        states: Array of TankState values
        masses: Array of current masses (kg)
        min_mass: Minimum mass required (kg)
        
    Returns:
        Index of fullest tank meeting criteria, or -1 if none found
    """
    max_mass = -1.0
    best_idx = -1
    
    for i in range(len(states)):
        if (states[i] == TankState.IDLE or states[i] == TankState.FULL) and masses[i] >= min_mass:
            if masses[i] > max_mass:
                max_mass = masses[i]
                best_idx = i
    
    return best_idx


@njit
def batch_pressure_update(
    masses: np.ndarray,
    volumes: np.ndarray,
    pressures: np.ndarray,
    temperature: float,
    gas_constant: float = GasConstants.R_H2
) -> None:
    """
    Update pressures for all tanks using ideal gas law, modifying the pressures array in place.
    
    P = (m/V) * R * T  where m=mass, V=volume, R=gas constant, T=temperature
    """
    for i in range(len(masses)):
        if volumes[i] > 0:
            density = masses[i] / volumes[i]
            pressures[i] = density * gas_constant * temperature
        else:
            pressures[i] = 0.0


@njit
def calculate_compression_work(
    p1: float,
    p2: float,
    mass: float,
    temperature: float,
    efficiency: float = 0.75,
    gamma: float = GasConstants.GAMMA_H2,
    gas_constant: float = GasConstants.R_H2
) -> float:
    """
    Calculate compression work using polytropic process model.
    
    W = (γ/(γ-1)) * (m*R*T/η) * [(P2/P1)^((γ-1)/γ) - 1]
    
    Args:
        p1: Inlet pressure (Pa)
        p2: Outlet pressure (Pa)
        mass: Mass of gas compressed (kg)
        temperature: Inlet temperature (K)
        efficiency: Isentropic efficiency (0-1)
        gamma: Specific heat ratio (Cp/Cv)
        gas_constant: Specific gas constant (J/kg·K)
        
    Returns:
        Compression work (J)
        
    Example:
        # Compress 50 kg H2 from 30 bar to 350 bar
        work_j = calculate_compression_work(30e5, 350e5, 50.0, 298.15)
        work_kwh = work_j / 3.6e6
    """
    if p1 <= 0:
        return 0.0
    pressure_ratio = p2 / p1
    exponent = (gamma - 1.0) / gamma
    
    work = (
        (gamma / (gamma - 1.0)) *
        (mass * gas_constant * temperature / efficiency) *
        (pressure_ratio**exponent - 1.0)
    )
    
    return work


@njit
def distribute_mass_to_tanks(
    total_mass: float,
    states: npt.NDArray[np.int32],
    masses: npt.NDArray[np.float64],
    capacities: npt.NDArray[np.float64]
) -> Tuple[npt.NDArray[np.float64], float]:
    """
    Distribute mass across available tanks, filling sequentially.
    
    Args:
        total_mass: Total mass to distribute (kg)
        states: Array of TankState values
        masses: Array of current masses (kg) - MODIFIED IN PLACE
        capacities: Array of tank capacities (kg)
        
    Returns:
        Tuple of (updated masses array, remaining undistributed mass)
        
    Example:
        masses = np.array([0.0, 50.0, 0.0])
        capacities = np.array([100.0, 100.0, 100.0])
        states = np.array([TankState.IDLE, TankState.IDLE, TankState.IDLE])
        
        updated_masses, overflow = distribute_mass_to_tanks(180.0, states, masses, capacities)
        # Tank 0: 100 kg (filled), Tank 1: 100 kg (topped off), Tank 2: 30 kg, overflow: 0 kg
    """
    remaining = total_mass
    
    for i in range(len(masses)):
        if remaining <= 0:
            break
        
        if not (states[i] == TankState.IDLE or states[i] == TankState.EMPTY):
            continue
        
        available_capacity = capacities[i] - masses[i]
        mass_to_add = min(remaining, available_capacity)
        
        masses[i] += mass_to_add
        remaining -= mass_to_add
        
        # Update state if full
        if masses[i] >= capacities[i] * 0.99:
            states[i] = TankState.FULL
    
    return masses, remaining


@njit
def calculate_total_mass_by_state(
    states: npt.NDArray[np.int32],
    masses: npt.NDArray[np.float64],
    target_state: int
) -> float:
    """
    Calculate total mass in tanks matching a specific state.
    
    Args:
        states: Array of TankState values
        masses: Array of masses (kg)
        target_state: State to filter by (IntEnum value)
        
    Returns:
        Total mass in matching tanks (kg)
        
    Example:
        total_full = calculate_total_mass_by_state(states, masses, TankState.FULL)
    """
    total = 0.0
    
    for i in range(len(states)):
        if states[i] == target_state:
            total += masses[i]
    
    return total


@njit
def simulate_filling_timestep(
    production_rate: float,
    dt: float,
    tank_states: npt.NDArray[np.int32],
    tank_masses: npt.NDArray[np.float64],
    tank_capacities: npt.NDArray[np.float64]
) -> Tuple[float, float]:
    """
    Simulate one timestep of production filling tanks.
    
    Args:
        production_rate: H2 production rate (kg/h)
        dt: Timestep (hours)
        tank_states: Array of TankState values - MODIFIED IN PLACE
        tank_masses: Array of masses (kg) - MODIFIED IN PLACE
        tank_capacities: Array of capacities (kg)
        
    Returns:
        Tuple of (mass stored, mass overflow)
    """
    production = production_rate * dt
    
    _, overflow = distribute_mass_to_tanks(
        production,
        tank_states,
        tank_masses,
        tank_capacities
    )
    
    stored = production - overflow
    
    return stored, overflow


@njit
def solve_rachford_rice_single_condensable(
    z_condensable: float,
    K_value: float
) -> float:
    """
    Analytical solution for Rachford-Rice with single condensable component.
    """
    if K_value >= 1.0:
        return 1.0
    
    if z_condensable < 1e-12:
        return 1.0
    
    beta = (z_condensable - 1.0) / (z_condensable * (K_value - 1.0))
    
    if beta < 0.0:
        beta = 0.0
    elif beta > 1.0:
        beta = 1.0
    
    return beta


@njit
def calculate_mixture_enthalpy(
    temperature: float,
    mole_fractions: np.ndarray,
    h_formations: np.ndarray,
    cp_coeffs_matrix: np.ndarray,
    T_ref: float = 298.15
) -> float:
    """
    Calculate mixture molar enthalpy with Cp integration.
    """
    h_mix = 0.0
    
    for i in range(len(mole_fractions)):
        if mole_fractions[i] < 1e-12:
            continue
        
        h_form = h_formations[i]
        
        A, B, C, D, E = cp_coeffs_matrix[i, 0], cp_coeffs_matrix[i, 1], cp_coeffs_matrix[i, 2], cp_coeffs_matrix[i, 3], cp_coeffs_matrix[i, 4]
        
        delta_h = (
            A * (temperature - T_ref) +
            0.5 * B * (temperature**2 - T_ref**2) +
            (1.0/3.0) * C * (temperature**3 - T_ref**3) +
            0.25 * D * (temperature**4 - T_ref**4) -
            E * (1.0/temperature - 1.0/T_ref) if temperature > 0 and T_ref > 0 else 0.0
        )
        
        h_species = h_form + delta_h
        h_mix += mole_fractions[i] * h_species
    
    return h_mix


# ============================================================================
# PEM ELECTROLYZER OPTIMIZATIONS
# ============================================================================

@njit
def calculate_pem_voltage_jit(
    j: float,
    T: float,
    P_op: float,
    # Constants passed as args to avoid global scope issues
    R: float,
    F: float,
    z: int,
    alpha: float,
    j0: float,
    j_lim: float,
    delta_mem: float,
    sigma_base: float,
    P_ref: float
) -> float:
    """
    Calculate PEM cell voltage (JIT compiled).
    Matches h2_plant.models.pem_physics.calculate_Vcell_base
    """
    # 1. Reversible Voltage (Nernst)
    U_rev_T = 1.229 - 0.9e-3 * (T - 298.15)
    pressure_ratio = P_op / P_ref
    Nernst_correction = (R * T) / (z * F) * np.log(pressure_ratio**1.5)
    U_rev = U_rev_T + Nernst_correction
    
    # 2. Activation Overpotential
    # Avoid log(0)
    j_safe = max(j, 1e-10)
    eta_act = (R * T) / (alpha * z * F) * np.log(j_safe / j0)
    
    # 3. Ohmic Overpotential
    eta_ohm = j * (delta_mem / sigma_base)
    
    # 4. Concentration Overpotential
    if j >= j_lim:
        eta_conc = 100.0 # High penalty
    else:
        eta_conc = (R * T) / (z * F) * np.log(j_lim / (j_lim - j_safe))
        
    return U_rev + eta_act + eta_ohm + eta_conc

@njit
def solve_pem_j_jit(
    target_power_W: float,
    T: float,
    P_op: float,
    Area_Total: float,
    P_bop_fixo: float,
    k_bop_var: float,
    j_guess: float,
    # Physics Constants
    R: float,
    F: float,
    z: int,
    alpha: float,
    j0: float,
    j_lim: float,
    delta_mem: float,
    sigma_base: float,
    P_ref: float,
    # Solver Params
    max_iter: int = 50,
    tol: float = 1e-4
) -> float:
    """
    Solve for current density j given target power using Newton-Raphson.
    P_total = (j * Area * V(j)) * (1 + k_bop) + P_bop_fixo
    Target: P_total - P_target = 0
    """
    x = j_guess
    
    for _ in range(max_iter):
        # Calculate f(x)
        V_c = calculate_pem_voltage_jit(x, T, P_op, R, F, z, alpha, j0, j_lim, delta_mem, sigma_base, P_ref)
        I_t = x * Area_Total
        P_stack = I_t * V_c
        P_total = P_stack * (1.0 + k_bop_var) + P_bop_fixo
        fx = P_total - target_power_W
        
        if abs(fx) < tol:
            return x
            
        # Calculate f'(x) numerically
        delta = 1e-5
        x_delta = x + delta
        V_c_d = calculate_pem_voltage_jit(x_delta, T, P_op, R, F, z, alpha, j0, j_lim, delta_mem, sigma_base, P_ref)
        I_t_d = x_delta * Area_Total
        P_stack_d = I_t_d * V_c_d
        P_total_d = P_stack_d * (1.0 + k_bop_var) + P_bop_fixo
        fx_delta = P_total_d - target_power_W
        
        dfx = (fx_delta - fx) / delta
        
        if dfx == 0.0:
            break
            
        x = x - fx / dfx
        
        # Clamp x to valid range
        if x < 1e-6:
            x = 1e-6
        if x > j_lim - 0.01:
            x = j_lim - 0.01
            
    return x


@njit
def simulate_soec_step_jit(
    reference_power: float,
    real_powers: npt.NDArray[np.float64],
    real_states: npt.NDArray[np.int32],
    real_limits: npt.NDArray[np.float64],
    virtual_map: npt.NDArray[np.int32],
    uniform_module_max_limit: float,
    power_standby_mw: float,
    power_first_step_mw: float,
    ramp_step_mw: float,
    minimum_total_power: float = 0.0
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.int32], npt.NDArray[np.float64]]:
    """
    JIT-compiled core logic for SOEC simulation step.
    Replaces SOECOperator._simulate_step
    """
    # --- 0. VIRTUAL -> REAL MAPPING ---
    # Numba supports advanced indexing
    powers_v = real_powers[virtual_map].copy()
    states_v = real_states[virtual_map].copy()
    limits_v = real_limits[virtual_map].copy()
    
    requested_power = np.sum(powers_v)
    difference = reference_power - requested_power
    
    tolerance = 0.005
    
    # 1. Promote Stationary (3) -> Optimal (4)
    for i in range(len(states_v)):
        if states_v[i] == 3 and abs(powers_v[i] - uniform_module_max_limit) < tolerance:
            states_v[i] = 4
            
    # --- CHECK STATIONARY ---
    if abs(difference) < 0.01:
        # Update stationary state
        for i in range(len(states_v)):
            if states_v[i] == 2 or states_v[i] == 5:
                states_v[i] = 3
                
        # Update real arrays
        for i in range(len(virtual_map)):
            idx = virtual_map[i]
            real_powers[idx] = powers_v[i]
            real_states[idx] = states_v[i]
            real_limits[idx] = limits_v[i]
            
        return real_powers, real_states, real_limits

    # --- DYNAMIC N CALCULATION ---
    num_active_modules = len(virtual_map)
    numerator = max(0.0, reference_power - minimum_total_power)
    broken_number = numerator / uniform_module_max_limit
    
    # N_ceil = math.ceil(broken_number) -> use numpy or manual
    if broken_number == int(broken_number):
        N_ceil = broken_number
    else:
        N_ceil = int(broken_number) + 1
    
    if broken_number > 0.01 and abs(broken_number - N_ceil) < 0.001:
        target_module_id = int(N_ceil) + 1
    else:
        target_module_id = int(N_ceil)
        
    target_module_id = max(1, min(num_active_modules + 1, target_module_id))
    target_index = target_module_id - 1
    
    # Base Limit
    N_floor = int(broken_number) # floor
    base_limit_term = uniform_module_max_limit * N_floor
    standby_term = 0.0
    new_limit_calc = reference_power - base_limit_term - standby_term
    
    # --- 4. DEFINITION OF DYNAMIC LIMITS ---
    active_limit = limits_v
    
    if new_limit_calc > power_standby_mw + tolerance and new_limit_calc < power_first_step_mw - tolerance:
        new_limit_calc = power_standby_mw
        
    if difference > 0: # RAMP UP
        inactive_limit_const = power_standby_mw
        if target_module_id > num_active_modules:
            active_limit[:] = uniform_module_max_limit
        else:
            active_limit[target_index] = max(inactive_limit_const, new_limit_calc)
            active_limit[:target_index] = uniform_module_max_limit
        active_limit[target_index + 1:] = inactive_limit_const
        
    else: # RAMP DOWN
        if abs(reference_power - minimum_total_power) < 0.01:
            active_limit[:] = power_standby_mw
        else:
            active_limit[target_index] = max(power_standby_mw, new_limit_calc)
            active_limit[:target_index] = uniform_module_max_limit
            active_limit[target_index + 1:] = power_standby_mw
            
    # --- 5. INDIVIDUALIZED DISPATCH ---
    difference_to_limit = active_limit - powers_v
    
    # movement = np.minimum(np.abs(difference_to_limit), ramp_step_mw) * np.sign(difference_to_limit)
    movement = np.empty_like(difference_to_limit)
    for i in range(len(difference_to_limit)):
        abs_diff = abs(difference_to_limit[i])
        sign_diff = 1.0 if difference_to_limit[i] >= 0 else -1.0
        movement[i] = min(abs_diff, ramp_step_mw) * sign_diff
    
    # Startup logic
    for i in range(len(powers_v)):
        if abs(powers_v[i] - power_standby_mw) < tolerance and difference_to_limit[i] > tolerance:
            powers_v[i] = power_first_step_mw
            movement[i] = 0.0
            
    powers_v += movement
    
    # Shutdown logic
    for i in range(len(powers_v)):
        if (powers_v[i] > power_standby_mw + tolerance) and (powers_v[i] < power_first_step_mw - tolerance):
            powers_v[i] = power_standby_mw
            
    # Final Hot Standby check
    for i in range(len(powers_v)):
        if powers_v[i] <= power_standby_mw + tolerance:
            powers_v[i] = power_standby_mw
            
    # --- 6. STATE UPDATE ---
    for i in range(len(powers_v)):
        if abs(powers_v[i] - power_standby_mw) < tolerance:
            states_v[i] = 1 # Hot Stand-by
        elif abs(powers_v[i] - uniform_module_max_limit) < tolerance:
            states_v[i] = 4 # Optimal
        elif abs(powers_v[i] - active_limit[i]) < tolerance:
            states_v[i] = 3 # Stationary
        elif difference_to_limit[i] > 0:
            states_v[i] = 2 # Ramp Up
        else:
            states_v[i] = 5 # Ramp Down
            
    # Update real arrays
    for i in range(len(virtual_map)):
        idx = virtual_map[i]
        real_powers[idx] = powers_v[i]
        real_states[idx] = states_v[i]
        real_limits[idx] = active_limit[i]
        
    return real_powers, real_states, real_limits
