"""
Numba JIT-Compiled Operations for Simulation Hot Paths.

This module contains performance-critical numerical operations compiled to
native machine code using Numba's JIT compiler, achieving near-C performance.

Performance Characteristics:
    - @njit decorated functions compile on first call (~100-500ms).
    - Subsequent calls execute at native speed (~10-100x faster than Python).
    - @njit(parallel=True) enables multi-core parallelization.

Usage Guidelines:
    - All inputs must be NumPy arrays or Python primitives.
    - No Python objects (lists, dicts) allowed inside JIT functions.
    - Constants must be passed as arguments (no global scope access).

Numerical Methods:
    - Bilinear interpolation for LUT lookups.
    - Newton-Raphson for implicit equations (PEM voltage, UV flash).
    - RK4 adaptive stepping for PFR reactor integration.
    - Rachford-Rice for flash equilibrium.
"""

import numpy as np
import numpy.typing as npt
from numba import njit
from typing import Tuple

from h2_plant.core.enums import TankState
from h2_plant.core.constants import GasConstants


# =============================================================================
# TANK OPERATIONS
# =============================================================================

@njit
def find_available_tank(
    states: npt.NDArray[np.int32],
    masses: npt.NDArray[np.float64],
    capacities: npt.NDArray[np.float64],
    min_capacity: float = 0.0
) -> int:
    """
    Find first idle tank with sufficient available capacity.

    Scans tank array sequentially, returning index of first tank that
    is idle and has at least min_capacity available.

    Args:
        states (np.ndarray): Array of TankState enum values (int32).
        masses (np.ndarray): Current mass in each tank (kg).
        capacities (np.ndarray): Maximum capacity of each tank (kg).
        min_capacity (float): Minimum required available capacity (kg).

    Returns:
        int: Index of suitable tank, or -1 if none found.
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

    Scans all tanks to find the one with maximum mass that is either
    idle or full and meets minimum mass requirement.

    Args:
        states (np.ndarray): Array of TankState values.
        masses (np.ndarray): Current mass in each tank (kg).
        min_mass (float): Minimum required mass (kg).

    Returns:
        int: Index of fullest suitable tank, or -1 if none found.
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
    Update pressures for all tanks using ideal gas law (in-place).

    Ideal Gas Law: **P = ρ × R × T**

    Args:
        masses (np.ndarray): Mass in each tank (kg). [in]
        volumes (np.ndarray): Tank volumes (m³). [in]
        pressures (np.ndarray): Pressure array (Pa). [out, modified in-place]
        temperature (float): Gas temperature (K).
        gas_constant (float): Specific gas constant (J/(kg·K)).
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
    Calculate polytropic compression work.

    **W = (γ/(γ-1)) × (m×R×T/η) × [(P₂/P₁)^((γ-1)/γ) - 1]**

    This formula assumes ideal gas behavior with constant specific heats.

    Args:
        p1 (float): Inlet pressure (Pa).
        p2 (float): Outlet pressure (Pa).
        mass (float): Mass of gas compressed (kg).
        temperature (float): Inlet temperature (K).
        efficiency (float): Isentropic efficiency (0-1). Default: 0.75.
        gamma (float): Specific heat ratio Cp/Cv. Default: 1.41 for H₂.
        gas_constant (float): Specific gas constant (J/(kg·K)).

    Returns:
        float: Compression work (J).
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

    Fills tanks in index order until all mass is distributed or
    all tanks are full.

    Args:
        total_mass (float): Total mass to distribute (kg).
        states (np.ndarray): Tank states (modified in-place).
        masses (np.ndarray): Tank masses (modified in-place).
        capacities (np.ndarray): Tank capacities (kg).

    Returns:
        Tuple[np.ndarray, float]: (updated masses, undistributed overflow).
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
        states (np.ndarray): Array of TankState values.
        masses (np.ndarray): Array of masses (kg).
        target_state (int): State to filter by (IntEnum value).

    Returns:
        float: Total mass in matching tanks (kg).
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
        production_rate (float): H₂ production rate (kg/h).
        dt (float): Timestep (hours).
        tank_states (np.ndarray): Modified in-place.
        tank_masses (np.ndarray): Modified in-place.
        tank_capacities (np.ndarray): Tank capacities (kg).

    Returns:
        Tuple[float, float]: (mass stored, mass overflow).
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


# =============================================================================
# FLASH EQUILIBRIUM
# =============================================================================

@njit
def solve_rachford_rice_single_condensable(
    z_condensable: float,
    K_value: float
) -> float:
    """
    Analytical Rachford-Rice solution for single condensable component.

    For binary gas-liquid with insoluble gas (x_gas = 0):
    **V/F = (z - 1) / (K - 1)**

    Args:
        z_condensable (float): Mole fraction of condensable (0-1).
        K_value (float): Vapor-liquid equilibrium ratio K = y/x.

    Returns:
        float: Vapor fraction β = V/F (0-1).
    """
    if K_value >= 1.0:
        return 1.0

    if z_condensable < 1e-12:
        return 1.0

    beta = (z_condensable - 1.0) / (K_value - 1.0)

    if beta < 0.0:
        beta = 0.0
    elif beta > 1.0:
        beta = 1.0

    return beta


# =============================================================================
# THERMODYNAMIC PROPERTIES
# =============================================================================

@njit
def calculate_mixture_enthalpy(
    temperature: float,
    mole_fractions: np.ndarray,
    h_formations: np.ndarray,
    cp_coeffs_matrix: np.ndarray,
    T_ref: float = 298.15
) -> float:
    """
    Calculate mixture molar enthalpy with Cp polynomial integration.

    **H_mix = Σ yᵢ × [H_f,i + ∫Cp,i dT]**

    Cp polynomial: Cp = A + BT + CT² + DT³ + E/T²

    Args:
        temperature (float): Temperature (K).
        mole_fractions (np.ndarray): Component mole fractions.
        h_formations (np.ndarray): Formation enthalpies (J/mol).
        cp_coeffs_matrix (np.ndarray): Cp coefficients [n_species × 5].
        T_ref (float): Reference temperature (K). Default: 298.15.

    Returns:
        float: Molar enthalpy (J/mol).
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


@njit
def calculate_mixture_cp(
    temperature: float,
    mole_fractions: np.ndarray,
    cp_coeffs_matrix: np.ndarray,
    T_ref: float = 298.15
) -> float:
    """
    Calculate mixture molar heat capacity at constant pressure.

    **Cp_mix = Σ yᵢ × Cp,i(T)**

    Cp polynomial: Cp = A + BT + CT² + DT³ + E/T²

    Args:
        temperature (float): Temperature (K).
        mole_fractions (np.ndarray): Component mole fractions.
        cp_coeffs_matrix (np.ndarray): Cp coefficients [n_species × 5].
        T_ref (float): Reference temperature (unused, kept for API consistency).

    Returns:
        float: Molar heat capacity (J/(mol·K)).
    """
    cp_mix = 0.0

    for i in range(len(mole_fractions)):
        if mole_fractions[i] < 1e-12:
            continue

        A, B, C, D, E = cp_coeffs_matrix[i, 0], cp_coeffs_matrix[i, 1], cp_coeffs_matrix[i, 2], cp_coeffs_matrix[i, 3], cp_coeffs_matrix[i, 4]

        cp_species = (
            A +
            B * temperature +
            C * temperature**2 +
            D * temperature**3 +
            (E / (temperature**2) if temperature > 0 else 0.0)
        )

        cp_mix += mole_fractions[i] * cp_species

    return cp_mix


# =============================================================================
# PEM ELECTROLYZER
# =============================================================================

@njit
def calculate_pem_voltage_jit(
    j: float,
    T: float,
    P_op: float,
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
    Calculate PEM electrolyzer cell voltage (JIT compiled).

    Voltage decomposition:
    **V = U_rev + η_act + η_ohm + η_conc**

    Components:
    - U_rev: Reversible (Nernst) potential with temperature/pressure correction.
    - η_act: Activation overpotential (Butler-Volmer, Tafel approximation).
    - η_ohm: Ohmic losses in membrane.
    - η_conc: Concentration overpotential at limiting current.

    Args:
        j (float): Current density (A/cm²).
        T (float): Temperature (K).
        P_op (float): Operating pressure (Pa).
        R (float): Universal gas constant (J/(mol·K)).
        F (float): Faraday constant (C/mol).
        z (int): Electrons transferred per reaction (2 for water splitting).
        alpha (float): Charge transfer coefficient.
        j0 (float): Exchange current density (A/cm²).
        j_lim (float): Limiting current density (A/cm²).
        delta_mem (float): Membrane thickness (cm).
        sigma_base (float): Membrane conductivity (S/cm).
        P_ref (float): Reference pressure (Pa).

    Returns:
        float: Cell voltage (V).
    """
    # Reversible voltage with Nernst correction
    U_rev_T = 1.229 - 0.9e-3 * (T - 298.15)
    pressure_ratio = P_op / P_ref
    Nernst_correction = (R * T) / (z * F) * np.log(pressure_ratio**1.5)
    U_rev = U_rev_T + Nernst_correction

    # Activation overpotential
    j_safe = max(j, 1e-10)
    eta_act = (R * T) / (alpha * z * F) * np.log(j_safe / j0)

    # Ohmic overpotential
    eta_ohm = j * (delta_mem / sigma_base)

    # Concentration overpotential
    if j >= j_lim:
        eta_conc = 100.0
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
    R: float,
    F: float,
    z: int,
    alpha: float,
    j0: float,
    j_lim: float,
    delta_mem: float,
    sigma_base: float,
    P_ref: float,
    max_iter: int = 50,
    tol: float = 1e-4
) -> float:
    """
    Solve for current density given target power using Newton-Raphson.

    **P_total = (j × A × V(j)) × (1 + k_bop) + P_bop_fixed**

    Args:
        target_power_W (float): Target total power (W).
        T (float): Temperature (K).
        P_op (float): Operating pressure (Pa).
        Area_Total (float): Total active area (cm²).
        P_bop_fixo (float): Fixed balance-of-plant power (W).
        k_bop_var (float): Variable BoP power fraction.
        j_guess (float): Initial current density guess (A/cm²).
        R, F, z, alpha, j0, j_lim, delta_mem, sigma_base, P_ref:
            Electrochemical parameters (see calculate_pem_voltage_jit).
        max_iter (int): Maximum iterations. Default: 50.
        tol (float): Convergence tolerance (W). Default: 1e-4.

    Returns:
        float: Converged current density (A/cm²).
    """
    x = j_guess

    for _ in range(max_iter):
        V_c = calculate_pem_voltage_jit(x, T, P_op, R, F, z, alpha, j0, j_lim, delta_mem, sigma_base, P_ref)
        I_t = x * Area_Total
        P_stack = I_t * V_c
        P_total = P_stack * (1.0 + k_bop_var) + P_bop_fixo
        fx = P_total - target_power_W

        if abs(fx) < tol:
            return x

        # Numerical derivative
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

        if x < 1e-6:
            x = 1e-6
        if x > j_lim - 0.01:
            x = j_lim - 0.01

    return x


# =============================================================================
# SOEC ELECTROLYZER
# =============================================================================

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
    JIT-compiled SOEC multi-module dispatch and ramping logic.

    Manages power distribution across modules with:
    - Smooth ramping between operating points.
    - Hot standby mode for rapid response.
    - Dynamic limit calculation based on target power.

    Args:
        reference_power (float): Target total power (MW).
        real_powers (np.ndarray): Module power states (MW), modified.
        real_states (np.ndarray): Module operating states, modified.
        real_limits (np.ndarray): Dynamic power limits (MW), modified.
        virtual_map (np.ndarray): Mapping from virtual to real indices.
        uniform_module_max_limit (float): Maximum per-module power (MW).
        power_standby_mw (float): Hot standby power level (MW).
        power_first_step_mw (float): First ramp step from standby (MW).
        ramp_step_mw (float): Ramp rate per timestep (MW).
        minimum_total_power (float): Minimum aggregate power (MW).

    Returns:
        Tuple: (real_powers, real_states, real_limits) arrays.
    """
    powers_v = real_powers[virtual_map].copy()
    states_v = real_states[virtual_map].copy()
    limits_v = real_limits[virtual_map].copy()

    requested_power = np.sum(powers_v)
    difference = reference_power - requested_power

    tolerance = 0.005

    # Promote Stationary → Optimal
    for i in range(len(states_v)):
        if states_v[i] == 3 and abs(powers_v[i] - uniform_module_max_limit) < tolerance:
            states_v[i] = 4

    # Check stationary condition
    if abs(difference) < 0.01:
        for i in range(len(states_v)):
            if states_v[i] == 2 or states_v[i] == 5:
                states_v[i] = 3

        for i in range(len(virtual_map)):
            idx = virtual_map[i]
            real_powers[idx] = powers_v[i]
            real_states[idx] = states_v[i]
            real_limits[idx] = limits_v[i]

        return real_powers, real_states, real_limits

    # Dynamic N calculation
    num_active_modules = len(virtual_map)
    numerator = max(0.0, reference_power - minimum_total_power)
    broken_number = numerator / uniform_module_max_limit

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

    N_floor = int(broken_number)
    base_limit_term = uniform_module_max_limit * N_floor
    standby_term = 0.0
    new_limit_calc = reference_power - base_limit_term - standby_term

    active_limit = limits_v

    if new_limit_calc > power_standby_mw + tolerance and new_limit_calc < power_first_step_mw - tolerance:
        new_limit_calc = power_standby_mw

    if difference > 0:  # Ramp up
        inactive_limit_const = power_standby_mw
        if target_module_id > num_active_modules:
            active_limit[:] = uniform_module_max_limit
        else:
            active_limit[target_index] = max(inactive_limit_const, new_limit_calc)
            active_limit[:target_index] = uniform_module_max_limit
        active_limit[target_index + 1:] = inactive_limit_const

    else:  # Ramp down
        if abs(reference_power - minimum_total_power) < 0.01:
            active_limit[:] = power_standby_mw
        else:
            active_limit[target_index] = max(power_standby_mw, new_limit_calc)
            active_limit[:target_index] = uniform_module_max_limit
            active_limit[target_index + 1:] = power_standby_mw

    # Individual dispatch
    difference_to_limit = active_limit - powers_v

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

    # Hot standby floor
    for i in range(len(powers_v)):
        if powers_v[i] <= power_standby_mw + tolerance:
            powers_v[i] = power_standby_mw

    # State update
    for i in range(len(powers_v)):
        if abs(powers_v[i] - power_standby_mw) < tolerance:
            states_v[i] = 1  # Hot Stand-by
        elif abs(powers_v[i] - uniform_module_max_limit) < tolerance:
            states_v[i] = 4  # Optimal
        elif abs(powers_v[i] - active_limit[i]) < tolerance:
            states_v[i] = 3  # Stationary
        elif difference_to_limit[i] > 0:
            states_v[i] = 2  # Ramp Up
        else:
            states_v[i] = 5  # Ramp Down

    # Update real arrays
    for i in range(len(virtual_map)):
        idx = virtual_map[i]
        real_powers[idx] = powers_v[i]
        real_states[idx] = states_v[i]
        real_limits[idx] = active_limit[i]

    return real_powers, real_states, real_limits


# =============================================================================
# INTERPOLATION
# =============================================================================

@njit
def bilinear_interp_jit(
    grid_x: npt.NDArray[np.float64],
    grid_y: npt.NDArray[np.float64],
    data: npt.NDArray[np.float64],
    x: float,
    y: float
) -> float:
    """
    Perform 2D bilinear interpolation (JIT compiled).

    **f(x, y) = (1-wx)(1-wy)f₀₀ + wx(1-wy)f₁₀ + (1-wx)wy·f₀₁ + wx·wy·f₁₁**

    Args:
        grid_x (np.ndarray): Sorted x coordinates (e.g., pressure).
        grid_y (np.ndarray): Sorted y coordinates (e.g., temperature).
        data (np.ndarray): 2D array [len(grid_x), len(grid_y)].
        x (float): Query x coordinate.
        y (float): Query y coordinate.

    Returns:
        float: Interpolated value.
    """
    # Clamp to bounds
    if x <= grid_x[0]:
        ix = 1
    elif x >= grid_x[-1]:
        ix = len(grid_x) - 1
    else:
        ix = np.searchsorted(grid_x, x)
        if ix == 0:
            ix = 1

    if y <= grid_y[0]:
        iy = 1
    elif y >= grid_y[-1]:

        iy = len(grid_y) - 1
    else:
        iy = np.searchsorted(grid_y, y)
        if iy == 0:
            iy = 1

    x0 = grid_x[ix-1]
    x1 = grid_x[ix]
    y0 = grid_y[iy-1]
    y1 = grid_y[iy]

    q00 = data[ix-1, iy-1]
    q01 = data[ix-1, iy]
    q10 = data[ix, iy-1]
    q11 = data[ix, iy]

    dx = x1 - x0
    dy = y1 - y0

    if dx == 0:
        wx = 0.0
    else:
        wx = (x - x0) / dx

    if dy == 0:
        wy = 0.0
    else:
        wy = (y - y0) / dy

    val = (
        q00 * (1 - wx) * (1 - wy) +
        q10 * wx * (1 - wy) +
        q01 * (1 - wx) * wy +
        q11 * wx * wy
    )

    return val


@njit(parallel=True)
def batch_bilinear_interp_jit(
    grid_x: npt.NDArray[np.float64],
    grid_y: npt.NDArray[np.float64],
    data: npt.NDArray[np.float64],
    x_arr: npt.NDArray[np.float64],
    y_arr: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Vectorized 2D bilinear interpolation with parallelization.

    Uses Numba prange for multi-core execution, achieving 10-50x speedup
    over Python loop implementation.

    Args:
        grid_x (np.ndarray): Sorted x coordinates.
        grid_y (np.ndarray): Sorted y coordinates.
        data (np.ndarray): 2D lookup table.
        x_arr (np.ndarray): Query x array.
        y_arr (np.ndarray): Query y array (same length as x_arr).

    Returns:
        np.ndarray: Interpolated values.
    """
    from numba import prange

    n = len(x_arr)
    results = np.zeros(n, dtype=np.float64)

    for i in prange(n):
        results[i] = bilinear_interp_jit(grid_x, grid_y, data, x_arr[i], y_arr[i])

    return results


@njit
def bilinear_interp_liquid(
    grid_p: npt.NDArray[np.float64],
    grid_t: npt.NDArray[np.float64],
    data: npt.NDArray[np.float64],
    p: float,
    t: float
) -> float:
    """
    Bilinear interpolation with liquid water bounds clamping.

    Clamps inputs to typical liquid water operating range:
    - Pressure: 1-20 bar
    - Temperature: 0-100°C

    Args:
        grid_p (np.ndarray): Pressure grid (Pa).
        grid_t (np.ndarray): Temperature grid (K).
        data (np.ndarray): 2D property table.
        p (float): Query pressure (Pa).
        t (float): Query temperature (K).

    Returns:
        float: Interpolated property value.
    """
    p_safe = p
    if p_safe < 1e5:
        p_safe = 1e5
    if p_safe > 20e5:
        p_safe = 20e5

    t_safe = t
    if t_safe < 273.15:
        t_safe = 273.15
    if t_safe > 373.15:
        t_safe = 373.15

    return bilinear_interp_jit(grid_p, grid_t, data, p_safe, t_safe)


# =============================================================================
# REACTOR MODELS
# =============================================================================

@njit
def solve_deoxo_pfr_step(
    L_total: float,
    steps: int,
    T_in: float,
    P_in_pa: float,
    molar_flow_total: float,
    y_o2_in: float,
    k0: float,
    Ea: float,
    R: float,
    delta_H: float,
    U_a: float,
    T_jacket: float,
    Area: float,
    Cp_mix: float,
    y_o2_target: float = 0.0
) -> Tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve DeOxo PFR mass/energy balance using adaptive RK4.

    Integrates along reactor length with adaptive step sizing
    based on conversion rate and temperature gradients.

    Reaction: 2H₂ + O₂ → 2H₂O (exothermic, ΔH < 0)
    
    Legacy Parity:
    - Stops if y_O2 drops below y_o2_target (5 ppm).

    Args:
        L_total (float): Total reactor length (m).
        steps (int): Ignored (adaptive stepping used).
        T_in (float): Inlet temperature (K).
        P_in_pa (float): Inlet pressure (Pa).
        molar_flow_total (float): Total molar flow (mol/s).
        y_o2_in (float): Inlet O₂ mole fraction.
        k0 (float): Pre-exponential factor (m³/(mol·s)).
        Ea (float): Activation energy (J/mol).
        R (float): Universal gas constant (J/(mol·K)).
        delta_H (float): Reaction enthalpy (J/mol O₂).
        U_a (float): Heat transfer coefficient (W/(m³·K)).
        T_jacket (float): Jacket temperature (K).
        Area (float): Cross-sectional area (m²).
        Cp_mix (float): Mixture heat capacity (J/(mol·K)).
        y_o2_target (float): Target O2 fraction to stop reaction (Legacy parity).

    Returns:
        Tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray]: 
            (conversion X, outlet T, max T, L_profile, T_profile, X_profile).
    """
    L_curr = 0.0
    dL = L_total / 100.0

    X = 0.0
    T = T_in
    T_max = T_in

    F_o2_in = molar_flow_total * y_o2_in
    max_iter = 10000
    
    # Pre-allocate arrays for history
    L_hist = np.zeros(max_iter + 1)
    T_hist = np.zeros(max_iter + 1)
    X_hist = np.zeros(max_iter + 1)
    
    # Initial point
    L_hist[0] = 0.0
    T_hist[0] = T_in
    X_hist[0] = 0.0
    step_count = 1

    if F_o2_in <= 1e-12:
        # Trim arrays to 1 point
        return 0.0, T_in, T_in, L_hist[:1], T_hist[:1], X_hist[:1]

    for _ in range(max_iter):
        if L_curr >= L_total:
            break
            
        # Check target condition (Legacy Parity)
        current_y_o2 = y_o2_in * (1.0 - X)
        if y_o2_target > 0 and current_y_o2 <= y_o2_target:
            # Reaction stops (Rate becomes 0)
            # We continue strictly for Temperature profile if active cooling exists?
            # Legacy says: if X >= target, r_O2 = 0.
            # We enforce that by setting dX=0, dT=cooling only.
            dx_est = 0.0
            dt_est = 0.0 
            if U_a > 0: # If cooling
                dt_est = (Area / (molar_flow_total * Cp_mix)) * (-U_a * (T - T_jacket))
            
            # Step forward
            dL = min(L_total - L_curr, L_total/100.0) # Simple step
            X_next = X
            T_next = T + dt_est * dL
            
            # Update
            X = X_next
            T = T_next
            L_curr += dL
             # Record history
            if step_count < max_iter:
                L_hist[step_count] = L_curr
                T_hist[step_count] = T
                X_hist[step_count] = X
                step_count += 1
            continue

        def get_grads_local(x_c, t_c):
            if x_c >= 1.0:
                dt_val = (Area / (molar_flow_total * Cp_mix)) * (-U_a * (t_c - T_jacket))
                return 0.0, dt_val

            k_eff = k0 * np.exp(-Ea / (R * t_c))
            y_loc = max(0.0, y_o2_in * (1.0 - x_c))
            C_o2 = (P_in_pa * y_loc) / (R * t_c)
            r = k_eff * C_o2

            dx = (Area / F_o2_in) * r

            gen = -delta_H * r
            rem = U_a * (t_c - T_jacket)
            dt = (Area / (molar_flow_total * Cp_mix)) * (gen - rem)
            return dx, dt

        dx_est, dt_est = get_grads_local(X, T)

        # Adaptive step sizing
        if dx_est > 1e-9:
            dL_target = 0.005 / dx_est
        else:
            dL_target = L_total * 0.1

        if abs(dt_est) > 1e-9:
            dL_temp = 5.0 / abs(dt_est)
            dL_target = min(dL_target, dL_temp)

        dL = min(dL_target, L_total - L_curr)
        dL = max(dL, 1e-6)

        # RK4 step
        k1_X, k1_T = get_grads_local(X, T)
        k2_X, k2_T = get_grads_local(X + 0.5*dL*k1_X, T + 0.5*dL*k1_T)
        k3_X, k3_T = get_grads_local(X + 0.5*dL*k2_X, T + 0.5*dL*k2_T)
        k4_X, k4_T = get_grads_local(X + dL*k3_X, T + dL*k3_T)

        X_next = X + (dL / 6.0) * (k1_X + 2*k2_X + 2*k3_X + k4_X)
        T_next = T + (dL / 6.0) * (k1_T + 2*k2_T + 2*k3_T + k4_T)

        X = min(1.0, X_next)
        T = T_next
        if T > T_max:
            T_max = T
        L_curr += dL
        
        # Record history
        if step_count < max_iter:
            L_hist[step_count] = L_curr
            T_hist[step_count] = T
            X_hist[step_count] = X
            step_count += 1

    return X, T, T_max, L_hist[:step_count], T_hist[:step_count], X_hist[:step_count]


@njit
def solve_uv_flash(
    target_u_molar: float,
    volume_m3: float,
    total_moles: float,
    mole_fractions: np.ndarray,
    h_formations: np.ndarray,
    cp_coeffs_matrix: np.ndarray,
    T_guess: float,
    R_gas: float = GasConstants.R_UNIVERSAL_J_PER_MOL_K,
    tol: float = 1e-4,
    max_iter: int = 50
) -> float:
    """
    Solve for temperature given internal energy and volume (UV flash).

    For ideal gas: U = H - RT, so:
    **f(T) = H(T) - R·T - U_target = 0**

    Uses Newton-Raphson with f'(T) = Cp(T) - R.

    Args:
        target_u_molar (float): Target molar internal energy (J/mol).
        volume_m3 (float): System volume (m³).
        total_moles (float): Total moles in system.
        mole_fractions (np.ndarray): Component mole fractions.
        h_formations (np.ndarray): Formation enthalpies (J/mol).
        cp_coeffs_matrix (np.ndarray): Cp coefficients.
        T_guess (float): Initial temperature guess (K).
        R_gas (float): Universal gas constant (J/(mol·K)).
        tol (float): Convergence tolerance. Default: 1e-4.
        max_iter (int): Maximum iterations. Default: 50.

    Returns:
        float: Converged temperature (K).
    """
    if total_moles <= 0 or volume_m3 <= 0:
        return T_guess

    T = T_guess

    for _ in range(max_iter):
        h_mix = calculate_mixture_enthalpy(T, mole_fractions, h_formations, cp_coeffs_matrix)

        u_calc = h_mix - R_gas * T
        f = u_calc - target_u_molar

        if abs(f) < tol:
            return T

        cp_mix = calculate_mixture_cp(T, mole_fractions, cp_coeffs_matrix)
        df = cp_mix - R_gas

        if df == 0.0:
            break

        T_new = T - f / df

        if T_new < 10.0:
            T_new = 10.0
        if T_new > 5000.0:
            T_new = 5000.0

        if abs(T_new - T) < tol:
            return T_new

        T = T_new

    return T


@njit
def dry_cooler_ntu_effectiveness(ntu: float, r: float) -> float:
    """
    Calculate effectiveness for unmixed-mixed crossflow heat exchanger.

    **ε = (1 - exp(-NTU(1-R))) / (1 - R·exp(-NTU(1-R)))**

    Special case R=1: ε = NTU / (1 + NTU)

    Args:
        ntu (float): Number of Transfer Units.
        r (float): Capacity ratio Cmin/Cmax.

    Returns:
        float: Heat exchanger effectiveness (0-1).
    """
    if ntu <= 0:
        return 0.0

    if abs(r - 1.0) < 1e-6:
        return ntu / (1.0 + ntu)
        
    term = np.exp(-ntu * (1.0 + r))
    return (1.0 - term) / (1.0 + r * term)

@njit
def counter_flow_ntu_effectiveness(ntu: float, r: float) -> float:
    """
    Calculate effectiveness for counter-flow heat exchanger.

    **ε = (1 - exp(-NTU(1-R))) / (1 - R·exp(-NTU(1-R)))**

    Special case R=1: ε = NTU / (1 + NTU)

    Args:
        ntu (float): Number of Transfer Units.
        r (float): Capacity ratio Cmin/Cmax.

    Returns:
        float: Heat exchanger effectiveness (0-1).
    """
    if ntu <= 0:
        return 0.0

    if abs(r - 1.0) < 1e-6:
        return ntu / (1.0 + ntu)

    term = np.exp(-ntu * (1.0 - r))
    return (1.0 - term) / (1.0 - r * term)

    arg = -ntu * (1.0 - r)
    if arg < -50.0:
        exp_term = 0.0
    else:
        exp_term = np.exp(arg)

    return (1.0 - exp_term) / (1.0 - r * exp_term)
