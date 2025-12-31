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
# Stream Enthalpy Constants (Hardcoded for JIT performance)
# Order: H2, O2, N2, H2O, CH4, CO2 (Matches CANONICAL_FLUID_ORDER)
GAS_CP_COEFFS = np.array([
    [29.11, -0.1916e-2, 0.4003e-5, -0.8704e-9, 0.0],  # H2
    [29.96, 4.18e-3, -1.67e-6, 0.0, 0.0],             # O2
    [28.98, -0.1571e-2, 0.8081e-5, -2.873e-9, 0.0],   # N2
    [32.24, 1.92e-3, 1.06e-5, -3.60e-9, 0.0],         # H2O (vap)
    [19.89, 5.02e-2, 1.27e-5, -1.10e-8, 0.0],         # CH4
    [22.26, 5.98e-2, -3.50e-5, 7.47e-9, 0.0]          # CO2
], dtype=np.float64)

GAS_MW = np.array([2.016, 32.0, 28.014, 18.015, 16.04, 44.01], dtype=np.float64)

# Pre-computed MW in kg/mol for stream.py compatibility
GAS_MW_KG_MOL = GAS_MW * 1e-3

# H2O Liquid Constants (Cp ~ 75.3 J/molK)
LIQ_CP_COEFFS = np.array([75.3, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
LIQ_MW = 18.015


@njit(cache=True)
def fast_composition_properties(mass_fracs: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Compute mole fractions, molar mass, and entropy mixing term in one JIT pass.
    
    Avoids Python-side NumPy allocations and vector operations by computing
    everything in a single loop with pre-allocated arrays.
    
    Args:
        mass_fracs: Array of mass fractions [H2, O2, N2, H2O, CH4, CO2]
    
    Returns:
        Tuple of (mole_fracs, M_mix, sum_ylny)
    """
    # MW in kg/mol for this calculation
    mw_arr = np.array([0.002016, 0.032, 0.028014, 0.018015, 0.01604, 0.04401])
    n = 6
    
    # Single pass: compute moles and total
    total_moles = 0.0
    moles = np.empty(n, dtype=np.float64)
    
    for i in range(n):
        if mass_fracs[i] > 0.0:
            m = mass_fracs[i] / mw_arr[i]
            moles[i] = m
            total_moles += m
        else:
            moles[i] = 0.0
    
    # Compute mole fractions and properties
    M_mix = 0.0
    sum_ylny = 0.0
    mole_fracs = np.zeros(n, dtype=np.float64)
    
    if total_moles > 1e-15:
        inv_total = 1.0 / total_moles
        for i in range(n):
            if moles[i] > 0.0:
                y = moles[i] * inv_total
                mole_fracs[i] = y
                M_mix += y * mw_arr[i]
                sum_ylny += y * np.log(y)
    else:
        # Fallback (N2 approx)
        M_mix = 0.028
    
    return mole_fracs, M_mix, sum_ylny

@njit(cache=True)
def _integral_cp(t: float, coeffs: np.ndarray) -> float:
    """Helper: Integral of Cp(T) from 0 to T."""
    A, B, C, D, E = coeffs
    # int(A + BT + CT^2 + DT^3 + E/T^2)
    return (A * t + 
            B * t**2 / 2 + 
            C * t**3 / 3 + 
            D * t**4 / 4 - 
            E / t)

@njit(cache=True)
def calculate_stream_enthalpy_jit(
    T_k: float,
    mass_fracs: np.ndarray, # (6,)
    h2o_liq_frac: float
) -> float:
    """
    Calculate specific enthalpy (J/kg) using JIT.
    Replicates Stream._compute_specific_enthalpy (sensible heat vs 298.15K).
    """
    t_ref = 298.15
    h_total = 0.0
    
    # 1. Gas Species
    for i in range(6):
        w_i = mass_fracs[i]
        if w_i > 1e-12:
            dh_mol = _integral_cp(T_k, GAS_CP_COEFFS[i]) - _integral_cp(t_ref, GAS_CP_COEFFS[i])
            h_spec = dh_mol * 1000.0 / GAS_MW[i] # J/mol -> J/kg
            h_total += w_i * h_spec
            
    # 2. Liquid Water
    if h2o_liq_frac > 1e-12:
        dh_mol_liq = _integral_cp(T_k, LIQ_CP_COEFFS) - _integral_cp(t_ref, LIQ_CP_COEFFS)
        h_liq = dh_mol_liq * 1000.0 / LIQ_MW
        h_total += h2o_liq_frac * h_liq
        
    return h_total

# =============================================================================
# TANK OPERATIONS
# =============================================================================

@njit(cache=True)
def find_available_tank(
    states: npt.NDArray[np.int32],
    masses: npt.NDArray[np.float64],
    capacities: npt.NDArray[np.float64],
    min_capacity: float = 0.0
) -> int:
    """
    Identifies the first storage unit capable of accepting mass.

    This function implements a "first-available" allocation strategy, iterating 
    sequentially effectively prioritizing low-index tanks. This deterministic 
    ordering helps maintain stable pressure gradients across the storage bank 
    and simplifies compressor control logic.

    Args:
        states (np.ndarray): Array of TankState enum values (int32).
        masses (np.ndarray): Current fluid mass in each vessel (kg).
        capacities (np.ndarray): Maximum mass rating of each vessel (kg).
        min_capacity (float): Minimum ullage required for selection (kg).

    Returns:
        int: Index of the primary suitable tank, or -1 if the bank is saturated.
    """
    for i in range(len(states)):
        available_capacity = capacities[i] - masses[i]
        if (states[i] == TankState.IDLE or states[i] == TankState.EMPTY) and available_capacity >= min_capacity:
            return i
    return -1


@njit(cache=True)
def find_fullest_tank(
    states: npt.NDArray[np.int32],
    masses: npt.NDArray[np.float64],
    min_mass: float = 0.0
) -> int:
    """
    Selects the optimal tank for discharge based on mass inventory.

    Prioritizes the vessel with the highest current mass to maximize discharge 
    duration and pressure potential. This greedy selection strategy minimizes 
    switching frequency during high-demand operation.

    Args:
        states (np.ndarray): Array of TankState operational codes.
        masses (np.ndarray): Current fluid mass inventory (kg).
        min_mass (float): Minimum heel mass required to initiate discharge (kg).

    Returns:
        int: Index of the optimal source tank, or -1 if no tank meets criteria.
    """
    max_mass = -1.0
    best_idx = -1

    for i in range(len(states)):
        if (states[i] == TankState.IDLE or states[i] == TankState.FULL) and masses[i] >= min_mass:
            if masses[i] > max_mass:
                max_mass = masses[i]
                best_idx = i

    return best_idx


@njit(cache=True)
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


@njit(cache=True)
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
    Computes energy requirement for polytropic compression.

    Real Process Approximation:
    **W = (γ/(γ-1)) × (m·R·T₁/η) × [(P₂/P₁)^((γ-1)/γ) - 1]**

    This model assumes a constant polytropic efficiency and constant specific 
    heats. While less accurate than real-gas integration for extreme pressures, 
    it provides a conservative energy estimate (typically within 5% limits) 
    sufficient for high-level plant sizing and techno-economic analysis.

    Args:
        p1 (float): Suction pressure (Pa).
        p2 (float): Discharge pressure (Pa).
        mass (float): Mass throughput (kg).
        temperature (float): Suction temperature (K).
        efficiency (float): Polytropic efficiency factor (0.0-1.0). Default: 0.75.
        gamma (float): Adiabatic index (Cp/Cv). Default: 1.41 (Hydrogen).
        gas_constant (float): Specific gas constant (J/(kg·K)).

    Returns:
        float: Compression work required (Joules).
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


@njit(cache=True)
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


@njit(cache=True)
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


@njit(cache=True)
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

@njit(cache=True)
def solve_rachford_rice_single_condensable(
    z_condensable: float,
    K_value: float
) -> float:
    """
    Computes vapor fraction for a binary system with one condensable component.

    This implementation uses the analytical solution to the Rachford-Rice equation
    optimized for binary mixtures (e.g., H₂O in H₂). Unlike the general iterative
    solver, this closed-form solution is numerically stable near phase boundaries
    (dew point/bubble point) and significantly faster for repeated flash calculations.

    Physics Principle:
    For a single condensable component (z) with equilibrium ratio K = P_sat/P,
    the Vapor Fraction (β) satisfies the material balance:
    **β = (1 - z) / (1 - K)**  (Assuming inert gas K >> 1 is ideal)

    Logic Flow:
    1. Superheated (K >= 1 or z <= K): β = 1.0 (All Vapor)
    2. Saturated/Two-Phase: β calculated directly.
    3. Clamped to [0, 1] for physical consistency.

    Args:
        z_condensable (float): Feed mole fraction of the condensable species (0-1).
        K_value (float): Equilibrium constant K = P_sat / P_system.

    Returns:
        float: Vapor mole fraction β = V/F (0.0 to 1.0).
    """
    # No condensation if K >= 1 (vapor phase can hold unlimited water at this P)
    if K_value >= 1.0:
        return 1.0

    # No condensable species present
    if z_condensable < 1e-12:
        return 1.0
    
    # Check saturation condition: z > K implies supersaturation
    if z_condensable <= K_value:
        # Undersaturated region - single phase vapor
        return 1.0
    
    # Two-phase region: compute vapor fraction analytically
    # Derived from z = β·y + (1-β)·x with x=1 (pure liquid assumption) and y=K·x=K
    beta = (1.0 - z_condensable) / (1.0 - K_value)

    # Numerical safeguard explicitly clamping result
    if beta < 0.0:
        beta = 0.0
    elif beta > 1.0:
        beta = 1.0

    return beta


# =============================================================================
# THERMODYNAMIC PROPERTIES
# =============================================================================

@njit(cache=True)
def calculate_mixture_enthalpy(
    temperature: float,
    mole_fractions: np.ndarray,
    h_formations: np.ndarray,
    cp_coeffs_matrix: np.ndarray,
    T_ref: float = 298.15
) -> float:
    """
    Calculates the molar enthalpy of a real mixture using NASA polynomial integration.

    Standard Enthalpy Calculation:
    **H_mix(T) = Σ y_i × [ H_f,i + ∫(Cp,i dT) from T_ref to T ]**

    This function integrates the temperature-dependent heat capacity polynomials
    (NASA 7-term format adapted to 5-term simulation standard) to capture 
    sensible heat effects accurately over wide temperature ranges (300K - 1200K).

    Args:
        temperature (float): System temperature (K).
        mole_fractions (np.ndarray): Composition array (sum = 1.0).
        h_formations (np.ndarray): Standard enthalpies of formation (J/mol) at T_ref.
        cp_coeffs_matrix (np.ndarray): Polynomial coefficients [n_species × 5].
                                       Format: A + BT + CT² + DT³ + E/T².
        T_ref (float): Reference temperature for integration (K). Default: 298.15.

    Returns:
        float: Specific molar enthalpy of the mixture (J/mol).
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


@njit(cache=True)
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
# ELECTRIC BOILER
# =============================================================================

@njit(cache=True)
def calc_boiler_outlet_enthalpy(
    h_in_j_kg: float,
    mass_flow_kg_h: float,
    power_input_w: float,
    efficiency: float
) -> float:
    """
    Computes outlet enthalpy based on the First Law of Thermodynamics (Steady Flow).

    Energy Balance:
    **h_out = h_in + Q_net / ṁ**
    where Q_net = Power_electrical × Efficiency

    This steady-state approximation assumes negligible kinetic and potential 
    energy changes. It serves as the boundary condition for the subsequent 
    isobaric flash calculation.

    Args:
        h_in_j_kg (float): Specific enthalpy of the inlet stream (J/kg).
        mass_flow_kg_h (float): Mass flow rate (kg/h).
        power_input_w (float): Gross electrical power input (Watts).
        efficiency (float): Thermal conversion efficiency (0.0-1.0), accounting
                            for heat losses to the environment.

    Returns:
        float: Specific outlet enthalpy (J/kg). Returns h_in if flow is negligible.
    """
    # 1. Zero Flow Protection (Critical for Numba/C-level code)
    if mass_flow_kg_h <= 1e-6:
        return h_in_j_kg
        
    # 2. Calculate Net Heat Input (Joules per second / Watts)
    q_net_w = power_input_w * efficiency
    
    # 3. Unit Conversion: Watts (J/s) -> J/h
    # 1 Watt = 1 J/s * 3600 s/h = 3600 J/h
    q_net_j_h = q_net_w * 3600.0
    
    # 4. Calculate Enthalpy Delta (J/kg)
    delta_h = q_net_j_h / mass_flow_kg_h
    
    return h_in_j_kg + delta_h


@njit(cache=True)
def calc_boiler_batch_scenario(
    h_in_array: np.ndarray,
    flow_array: np.ndarray,
    power_array: np.ndarray,
    efficiency: float
) -> np.ndarray:
    """
    Vectorized version for rapid scenario analysis (e.g., 8760 steps at once).
    
    Optimization: 
        Loops are unrolled by LLVM. 100x faster than pandas/python loops for 
        yearly simulations.
    
    Args:
        h_in_array: Array of inlet enthalpies (J/kg) for each timestep.
        flow_array: Array of mass flow rates (kg/h) for each timestep.
        power_array: Array of applied power (W) for each timestep.
        efficiency: Thermal efficiency factor (0.0 to 1.0).
    
    Returns:
        np.ndarray: Array of outlet enthalpies (J/kg).
    """
    n = len(h_in_array)
    h_out_array = np.zeros(n)
    
    for i in range(n):
        h_out_array[i] = calc_boiler_outlet_enthalpy(
            h_in_array[i],
            flow_array[i],
            power_array[i],
            efficiency
        )
        
    return h_out_array


@njit(cache=True)
def solve_temperature_from_enthalpy_jit(
    h_target: float,
    pressure_pa: float,
    T_guess: float,
    P_grid: np.ndarray,
    T_grid: np.ndarray,
    H_lut: np.ndarray,
    C_lut: np.ndarray,
    cp_default: float = 4180.0,
    tol: float = 0.01,
    max_iter: int = 20
) -> float:
    """
    Newton-Raphson solver for T given h_target at constant P (JIT compiled).
    
    Solves: h(T, P) = h_target using bilinear interpolation on LUT.
    
    Newton-Raphson iteration:
        T_new = T_old + (h_target - h(T_old)) / Cp(T_old)
    
    Args:
        h_target: Target enthalpy (J/kg).
        pressure_pa: Operating pressure (Pa).
        T_guess: Initial temperature guess (K).
        P_grid: Pressure grid array (Pa).
        T_grid: Temperature grid array (K).
        H_lut: Enthalpy LUT [n_P, n_T] (J/kg).
        C_lut: Heat capacity LUT [n_P, n_T] (J/kg/K).
        cp_default: Fallback heat capacity (J/kg/K).
        tol: Convergence tolerance (K). Default: 0.01 K.
        max_iter: Maximum iterations. Default: 20.
        
    Returns:
        Solved temperature (K).
    """
    T = T_guess
    
    # Find pressure index
    n_P = len(P_grid)
    n_T = len(T_grid)
    
    # Clamp pressure to grid bounds
    P_clamped = min(max(pressure_pa, P_grid[0]), P_grid[-1])
    
    # Find pressure brackets
    ip = 0
    for idx in range(1, n_P):
        if P_grid[idx] >= P_clamped:
            ip = idx
            break
    if ip == 0:
        ip = 1
    
    # Pressure interpolation weight
    P0, P1 = P_grid[ip-1], P_grid[ip]
    wp = (P_clamped - P0) / (P1 - P0) if P1 != P0 else 0.0
    
    for _ in range(max_iter):
        # Clamp temperature to grid bounds
        T_clamped = min(max(T, T_grid[0]), T_grid[-1])
        
        # Find temperature brackets
        it = 0
        for idx in range(1, n_T):
            if T_grid[idx] >= T_clamped:
                it = idx
                break
        if it == 0:
            it = 1
        
        # Temperature interpolation weight
        T0, T1 = T_grid[it-1], T_grid[it]
        wt = (T_clamped - T0) / (T1 - T0) if T1 != T0 else 0.0
        
        # Bilinear interpolation for H
        h00 = H_lut[ip-1, it-1]
        h01 = H_lut[ip-1, it]
        h10 = H_lut[ip, it-1]
        h11 = H_lut[ip, it]
        
        h_current = (
            h00 * (1 - wp) * (1 - wt) +
            h10 * wp * (1 - wt) +
            h01 * (1 - wp) * wt +
            h11 * wp * wt
        )
        
        # Bilinear interpolation for Cp
        c00 = C_lut[ip-1, it-1]
        c01 = C_lut[ip-1, it]
        c10 = C_lut[ip, it-1]
        c11 = C_lut[ip, it]
        
        cp_current = (
            c00 * (1 - wp) * (1 - wt) +
            c10 * wp * (1 - wt) +
            c01 * (1 - wp) * wt +
            c11 * wp * wt
        )
        
        # Guard against invalid Cp
        if cp_current < 100.0:
            cp_current = cp_default
        
        # Newton-Raphson step
        residual = h_target - h_current
        T_new = T + residual / cp_current
        
        # Clamp to LUT temperature bounds
        T_new = max(T_new, 273.15)
        T_new = min(T_new, 1200.0)
        
        # Check convergence
        if abs(T_new - T) < tol:
            return T_new
        
        T = T_new
    
    return T


@njit(cache=True)
def calc_boiler_flash_jit(
    h_out: float,
    pressure_pa: float,
    T_in: float,
    P_sat_grid: np.ndarray,
    T_sat_grid: np.ndarray,
    H_liq_sat: np.ndarray,
    H_vap_sat: np.ndarray,
    P_grid: np.ndarray,
    T_grid: np.ndarray,
    H_lut: np.ndarray,
    C_lut: np.ndarray
) -> Tuple[float, float, int]:
    """
    Flash calculation for water/steam boiler (JIT compiled).
    
    Determines phase and temperature from outlet enthalpy.
    
    Args:
        h_out: Outlet enthalpy (J/kg).
        pressure_pa: Operating pressure (Pa).
        T_in: Inlet temperature (K) as fallback guess.
        P_sat_grid: Saturation pressure array (Pa).
        T_sat_grid: Saturation temperature array (K).
        H_liq_sat: Saturated liquid enthalpy array (J/kg).
        H_vap_sat: Saturated vapor enthalpy array (J/kg).
        P_grid, T_grid: Main LUT grids.
        H_lut, C_lut: Main LUT data.
        
    Returns:
        Tuple[T_out, vapor_fraction, phase]:
            - T_out (float): Outlet temperature (K).
            - vapor_fraction (float): Mass fraction vapor (0-1).
            - phase (int): 0=liquid, 1=mixed, 2=gas.
    """
    # 1. Interpolate saturation properties at current pressure
    # P_sat_grid is monotonically increasing with T_sat_grid
    n_sat = len(P_sat_grid)
    
    # Clamp pressure to saturation range
    P_min = P_sat_grid[0]
    P_max = P_sat_grid[-1]
    
    if pressure_pa <= P_min:
        t_sat = T_sat_grid[0]
        h_sat_liq = H_liq_sat[0]
        h_sat_vap = H_vap_sat[0]
    elif pressure_pa >= P_max:
        t_sat = T_sat_grid[-1]
        h_sat_liq = H_liq_sat[-1]
        h_sat_vap = H_vap_sat[-1]
    else:
        # Find index using linear search (P_sat increasing with T)
        idx = 0
        for i in range(1, n_sat):
            if P_sat_grid[i] >= pressure_pa:
                idx = i
                break
        
        # Linear interpolation for T_sat
        P0, P1 = P_sat_grid[idx-1], P_sat_grid[idx]
        w = (pressure_pa - P0) / (P1 - P0) if P1 != P0 else 0.0
        
        t_sat = T_sat_grid[idx-1] * (1 - w) + T_sat_grid[idx] * w
        h_sat_liq = H_liq_sat[idx-1] * (1 - w) + H_liq_sat[idx] * w
        h_sat_vap = H_vap_sat[idx-1] * (1 - w) + H_vap_sat[idx] * w
    
    # 2. Flash calculation
    if h_out < h_sat_liq:
        # Subcooled liquid
        T_out = solve_temperature_from_enthalpy_jit(
            h_out, pressure_pa, t_sat - 10.0,
            P_grid, T_grid, H_lut, C_lut, 4180.0
        )
        return T_out, 0.0, 0
        
    elif h_out > h_sat_vap:
        # Superheated vapor
        T_out = solve_temperature_from_enthalpy_jit(
            h_out, pressure_pa, t_sat + 10.0,
            P_grid, T_grid, H_lut, C_lut, 2080.0
        )
        return T_out, 1.0, 2
        
    else:
        # Saturated mixture
        denom = h_sat_vap - h_sat_liq
        if denom > 1e-6:
            vapor_frac = (h_out - h_sat_liq) / denom
        else:
            vapor_frac = 0.0
        return t_sat, vapor_frac, 1


@njit(cache=True, parallel=True)
def calc_boiler_batch_full(
    h_in_array: np.ndarray,
    flow_array: np.ndarray,
    power_array: np.ndarray,
    pressure_array: np.ndarray,
    T_in_array: np.ndarray,
    efficiency: float,
    is_water: bool,
    P_grid: np.ndarray,
    T_grid: np.ndarray,
    H_lut: np.ndarray,
    C_lut: np.ndarray,
    P_sat_grid: np.ndarray,
    T_sat_grid: np.ndarray,
    H_liq_sat: np.ndarray,
    H_vap_sat: np.ndarray,
    cp_gas: float = 14304.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Full batch electric boiler processing with T(h) solving (JIT + parallel).
    
    Processes an entire time series (e.g., 8760 hourly steps for yearly simulation)
    with near-C performance using Numba parallelization.
    
    Args:
        h_in_array: Inlet enthalpies (J/kg) per timestep.
        flow_array: Mass flow rates (kg/h) per timestep.
        power_array: Applied power (W) per timestep.
        pressure_array: Operating pressures (Pa) per timestep.
        T_in_array: Inlet temperatures (K) per timestep.
        efficiency: Thermal efficiency (0-1).
        is_water: True for water/steam (flash), False for gas (simple Cp).
        P_grid, T_grid: LUT pressure/temperature grids.
        H_lut, C_lut: Enthalpy and Cp LUT data.
        P_sat_grid, T_sat_grid: Saturation grids.
        H_liq_sat, H_vap_sat: Saturation enthalpy arrays.
        cp_gas: Heat capacity for gas mode (J/kg/K). Default: 14304 (H2).
        
    Returns:
        Tuple[h_out, T_out, vapor_frac, phase]:
            - h_out (np.ndarray): Outlet enthalpies (J/kg).
            - T_out (np.ndarray): Outlet temperatures (K).
            - vapor_frac (np.ndarray): Vapor fractions (0-1, only for water mode).
            - phase (np.ndarray): Phase codes (0=liq, 1=mixed, 2=gas).
    """
    n = len(h_in_array)
    h_out = np.zeros(n)
    T_out = np.zeros(n)
    vapor_frac = np.zeros(n)
    phase = np.zeros(n, dtype=np.int32)
    
    for i in range(n):  # Numba parallel=True will auto-parallelize
        # 1. Calculate outlet enthalpy
        h_out[i] = calc_boiler_outlet_enthalpy(
            h_in_array[i],
            flow_array[i],
            power_array[i],
            efficiency
        )
        
        # 2. Solve for temperature
        if is_water:
            # Flash calculation for water
            T_out[i], vapor_frac[i], phase[i] = calc_boiler_flash_jit(
                h_out[i], pressure_array[i], T_in_array[i],
                P_sat_grid, T_sat_grid, H_liq_sat, H_vap_sat,
                P_grid, T_grid, H_lut, C_lut
            )
        else:
            # Simple Cp for gas
            delta_h = h_out[i] - h_in_array[i]
            T_out[i] = T_in_array[i] + delta_h / cp_gas
            vapor_frac[i] = 0.0
            phase[i] = 2  # Always gas
    
    return h_out, T_out, vapor_frac, phase


# =============================================================================
# PEM ELECTROLYZER
# =============================================================================

@njit(cache=True)
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


@njit(cache=True)
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
    Determines the operating current density for a requested power setpoint.

    This function solves the non-linear power balance equation:
    **P_target = (V_cell(j) × j × Area) × (1 + k_bop) + P_bop_fixed**

    Since cell voltage V_cell(j) is non-linear (due to logarithmic activation 
    and degradation terms), a Newton-Raphson iterative solver is required to 
    find the precise current density 'j' that matches the total plant power consumption.

    Args:
        target_power_W (float): Total plant power consumption setpoint (W).
        T (float): Stack operating temperature (K).
        P_op (float): Cathode operating pressure (Pa).
        Area_Total (float): Total active membrane area (cm²).
        P_bop_fixo (float): Fixed parasitic power consumption (W).
        k_bop_var (float): Variable parasitic load factor (proportional to stack power).
        j_guess (float): Initial guess for current density (A/cm²).
        R, F, z, alpha, j0, j_lim, delta_mem, sigma_base, P_ref:
            Electrochemical model parameters (see calculate_pem_voltage_jit).
        max_iter (int): Solver iteration limit. Default: 50.
        tol (float): Power convergence tolerance (W). Default: 1e-4.

    Returns:
        float: Operating current density (A/cm²) satisfying the power balance.
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

@njit(cache=True)
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
    Executes the dispatch control logic for a multi-module SOEC plant.

    This function manages the granular power allocation across SOEC modules to
    optimize efficiency and component lifetime. It prioritizes keeping modules
    in "Hot Standby" rather than cold shutdown to minimize thermal cycling stress
    and maximize ramp-up response speed.

    Control Logic:
    1. **Allocation**: Calculates optimal number of active modules (N_ceil).
    2. **Ramping**: Enforces physical ramp rate limits (MW/step).
    3. **Standby**: Maintains idle modules at `power_standby_mw` to preserve temperature.
    4. **State Management**: Updates module states (Ramp Up/Down, Stationary, Optimal).

    Args:
        reference_power (float): Total plant power setpoint (MW).
        real_powers (np.ndarray): Previous timestep power per module (MW).
        real_states (np.ndarray): Previous timestep operational states.
        real_limits (np.ndarray): Dynamic maximum limits per module (MW).
        virtual_map (np.ndarray): Index mapping for load balancing (rotation).
        uniform_module_max_limit (float): Rated maximum power per module (MW).
        power_standby_mw (float): Minimum power to maintain thermal standby (MW).
        power_first_step_mw (float): Minimum active production power (MW).
        ramp_step_mw (float): Maximum power change per calculation step (MW).
        minimum_total_power (float): Plant-wide minimum turndown (MW).

    Returns:
        Tuple: Updated arrays (powers, states, limits) reflecting the new dispatch.
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

@njit(cache=True)
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


@njit(cache=True)
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

@njit(cache=True)
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
    Integrates the Plug Flow Reactor (PFR) equations for catalytic deoxygenation.

    Reaction Model:
    **2H₂ + O₂ → 2H₂O** (Highly Exothermic)

    Numerical Method:
    Fourth-order Runge-Kutta (RK4) with adaptive step sizing. The adaptive stepper
    is critical because the reaction rate is highly sensitive to temperature 
    (exponential Arrhenius term), leading to "stiff" differential equations 
    near the reactor inlet (hot spot formation).

    Conditions:
    - MassBalance: dX/dL = r_O2 * Area / F_O2_in
    - EnergyBalance: dT/dL = (Generation - Removal) / (F_total * Cp)

    Args:
        L_total (float): Total length of the catalytic bed (m).
        steps (int): (Deprecated) Number of fixed steps - overridden by adaptive logic.
        T_in (float): Feed gas temperature (K).
        P_in_pa (float): Feed gas pressure (Pa). Assumed constant (negligible pressure drop).
        molar_flow_total (float): Total molar flow rate (mol/s).
        y_o2_in (float): Inlet oxygen mole fraction.
        k0 (float): Reaction rate pre-exponential factor (m³/(mol·s)).
        Ea (float): Activation energy (J/mol).
        R (float): Universal constant (J/(mol·K)).
        delta_H (float): Enthalpy of reaction (J/mol O₂ consumed).
        U_a (float): Overall heat transfer coefficient per unit volume (W/(m³·K)).
        T_jacket (float): Cooling jacket temperature (K).
        Area (float): Reactor cross-sectional area (m²).
        Cp_mix (float): Molar heat capacity of the mixture (J/(mol·K)).
        y_o2_target (float): Target O2 fraction for simulation cutoff (default 0.0).

    Returns:
        Tuple:
            - conversion X (float): Final fractional conversion of O₂.
            - outlet T (float): Exit temperature (K).
            - max T (float): Peak temperature observed (Hot Spot) (K).
            - L_profile (array): Length coordinate history.
            - T_profile (array): Temperature profile history.
            - X_profile (array): Conversion profile history.
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


@njit(cache=True)
def solve_deoxo_multizone_jit(
    # Arrays for Zone Parameters
    L_zones: npt.NDArray[np.float64],
    k0_zones: npt.NDArray[np.float64],
    U_a_zones: npt.NDArray[np.float64],
    # Scalar Inputs
    T_in: float,
    P_in_pa: float,
    molar_flow_total: float,
    y_o2_in: float,
    Ea: float,
    R: float,
    delta_H: float,
    T_jacket: float,
    Area: float,
    Cp_mix: float,
    y_o2_target: float,
    max_steps_per_zone: int = 50
) -> Tuple[float, float, float, npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Simulates a multi-zone PFR in a single JIT-compiled pass.
    
    Optimized to eliminate Python overhead by handling zone transitions
    and profile aggregation in native code. RK4 gradients are inlined
    to avoid Numba closure limitations.
    
    Args:
        L_zones: Array of lengths for each zone (m).
        k0_zones: Array of kinetic pre-factors for each zone.
        U_a_zones: Array of heat transfer coefficients for each zone (W/m³/K).
        T_in: Inlet temperature (K).
        P_in_pa: Inlet pressure (Pa).
        molar_flow_total: Total molar flow rate (mol/s).
        y_o2_in: Inlet O2 mole fraction.
        Ea: Activation energy (J/mol).
        R: Gas constant (J/mol/K).
        delta_H: Reaction enthalpy (J/mol O2, negative = exothermic).
        T_jacket: Cooling jacket temperature (K).
        Area: Reactor cross-sectional area (m²).
        Cp_mix: Mixture molar heat capacity (J/mol/K).
        y_o2_target: Target O2 fraction (not used in this version).
        max_steps_per_zone: Integration steps per zone.
        
    Returns:
        Tuple: (X_total, T_out, T_peak, L_profile, T_profile, X_profile)
    """
    num_zones = len(L_zones)
    
    # Pre-allocate profile arrays (static sizing for performance)
    total_max_points = num_zones * (max_steps_per_zone + 1) + 1
    L_hist = np.zeros(total_max_points)
    T_hist = np.zeros(total_max_points)
    X_hist = np.zeros(total_max_points)
    
    # State tracking
    current_idx = 0
    L_cumulative_offset = 0.0
    T_curr = T_in
    y_o2_curr = y_o2_in
    T_max_global = T_in
    X_global = 0.0
    
    # Initialize first point
    L_hist[0] = 0.0
    T_hist[0] = T_in
    X_hist[0] = 0.0
    current_idx = 1
    
    # Pre-calc constant term for heat capacity flow
    flow_Cp = molar_flow_total * Cp_mix

    for z in range(num_zones):
        L_zone = L_zones[z]
        if L_zone <= 1e-6:
            continue
            
        k0 = k0_zones[z]
        U_a = U_a_zones[z]
        
        # Reset local zone integration vars
        X_local = 0.0
        T_local = T_curr
        
        # Determine flux for this zone
        F_o2_in_zone = molar_flow_total * y_o2_curr
        
        # Skip reaction if no O2 left, just record continuity point with cooling
        if F_o2_in_zone <= 1e-12:
            L_cumulative_offset += L_zone
            L_hist[current_idx] = L_cumulative_offset
            # Simple analytical cooling if U_a > 0
            if U_a > 0 and flow_Cp > 0:
                NTU = (U_a * Area * L_zone) / flow_Cp
                T_curr = T_jacket + (T_local - T_jacket) * np.exp(-NTU)
            T_hist[current_idx] = T_curr
            X_hist[current_idx] = X_global
            current_idx += 1
            continue

        # RK4 Integration Loop
        dL = L_zone / max_steps_per_zone
        L_local = 0.0
        
        for step in range(max_steps_per_zone):
            # --- Inline RK4 Steps (k1, k2, k3, k4) ---
            # State: (x, t) -> Gradients: (dx/dL, dT/dL)
            
            # K1
            x_k = X_local
            t_k = T_local
            if x_k >= 1.0:
                dx1 = 0.0
                dt1 = (Area / flow_Cp) * (-U_a * (t_k - T_jacket))
            else:
                k_eff = k0 * np.exp(-Ea / (R * t_k))
                y_loc = max(0.0, y_o2_curr * (1.0 - x_k))
                C_o2 = (P_in_pa * y_loc) / (R * t_k)
                r = k_eff * C_o2
                dx1 = (Area / F_o2_in_zone) * r
                dt1 = (Area / flow_Cp) * (-delta_H * r - U_a * (t_k - T_jacket))

            # K2
            x_k = X_local + 0.5 * dL * dx1
            t_k = T_local + 0.5 * dL * dt1
            if x_k >= 1.0:
                dx2 = 0.0
                dt2 = (Area / flow_Cp) * (-U_a * (t_k - T_jacket))
            else:
                k_eff = k0 * np.exp(-Ea / (R * t_k))
                y_loc = max(0.0, y_o2_curr * (1.0 - x_k))
                C_o2 = (P_in_pa * y_loc) / (R * t_k)
                r = k_eff * C_o2
                dx2 = (Area / F_o2_in_zone) * r
                dt2 = (Area / flow_Cp) * (-delta_H * r - U_a * (t_k - T_jacket))

            # K3
            x_k = X_local + 0.5 * dL * dx2
            t_k = T_local + 0.5 * dL * dt2
            if x_k >= 1.0:
                dx3 = 0.0
                dt3 = (Area / flow_Cp) * (-U_a * (t_k - T_jacket))
            else:
                k_eff = k0 * np.exp(-Ea / (R * t_k))
                y_loc = max(0.0, y_o2_curr * (1.0 - x_k))
                C_o2 = (P_in_pa * y_loc) / (R * t_k)
                r = k_eff * C_o2
                dx3 = (Area / F_o2_in_zone) * r
                dt3 = (Area / flow_Cp) * (-delta_H * r - U_a * (t_k - T_jacket))

            # K4
            x_k = X_local + dL * dx3
            t_k = T_local + dL * dt3
            if x_k >= 1.0:
                dx4 = 0.0
                dt4 = (Area / flow_Cp) * (-U_a * (t_k - T_jacket))
            else:
                k_eff = k0 * np.exp(-Ea / (R * t_k))
                y_loc = max(0.0, y_o2_curr * (1.0 - x_k))
                C_o2 = (P_in_pa * y_loc) / (R * t_k)
                r = k_eff * C_o2
                dx4 = (Area / F_o2_in_zone) * r
                dt4 = (Area / flow_Cp) * (-delta_H * r - U_a * (t_k - T_jacket))

            # Update State
            X_next = X_local + (dL / 6.0) * (dx1 + 2*dx2 + 2*dx3 + dx4)
            T_next = T_local + (dL / 6.0) * (dt1 + 2*dt2 + 2*dt3 + dt4)
            
            X_local = min(1.0, max(0.0, X_next))
            T_local = T_next
            L_local += dL
            
            # Global Tracking
            if T_local > T_max_global:
                T_max_global = T_local
                
            L_hist[current_idx] = L_cumulative_offset + L_local
            T_hist[current_idx] = T_local
            
            # Global Conversion: 1 - (1 - X_prev_global) * (1 - X_local)
            current_global_X = 1.0 - (1.0 - X_global) * (1.0 - X_local)
            X_hist[current_idx] = current_global_X
            
            current_idx += 1
            
        # End of Zone Update
        L_cumulative_offset += L_zone
        T_curr = T_local
        X_global = 1.0 - (1.0 - X_global) * (1.0 - X_local)
        y_o2_curr = y_o2_curr * (1.0 - X_local)
        
    return X_global, T_curr, T_max_global, L_hist[:current_idx], T_hist[:current_idx], X_hist[:current_idx]

@njit(cache=True)
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


@njit(cache=True)
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

@njit(cache=True)
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

@njit(cache=True)
def calculate_compression_realgas_jit(
    p_in_pa: float,
    p_out_pa: float,
    T_in_k: float,
    efficiency: float,
    P_grid: np.ndarray,
    T_grid: np.ndarray,
    S_grid: np.ndarray,
    H_lut: np.ndarray,
    S_lut: np.ndarray,
    C_lut: np.ndarray,
    H_from_PS_lut: np.ndarray
) -> Tuple[float, float, float]:
    """
    Calculate real-gas polytropic compression using JIT-compiled LUT lookups.
    
    Performs full isentropic compression calculation:
    1. s_in = S(P_in, T_in)
    2. h_in = H(P_in, T_in)
    3. h_out_isen = H(P_out, s_in)  [Using P-S grid]
    4. w_isen = h_out_isen - h_in
    5. w_actual = w_isen / efficiency
    6. h_out_actual = h_in + w_actual
    7. T_out = T(P_out, h_out_actual) [Solved via Newton-Raphson]
    
    Args:
        p_in_pa: Inlet pressure (Pa).
        p_out_pa: Outlet pressure (Pa).
        T_in_k: Inlet temperature (K).
        efficiency: Isentropic efficiency (0-1).
        P_grid: Pressure grid array (Pas).
        T_grid: Temperature grid array (K).
        S_grid: Entropy grid array (J/kgK) for H_from_PS_lut.
        H_lut: Enthalpy LUT (P, T) -> J/kg.
        S_lut: Entropy LUT (P, T) -> J/kgK.
        C_lut: Heat Capacity LUT (P, T) -> J/kgK.
        H_from_PS_lut: Enthalpy LUT (P, S) -> J/kg.
        
    Returns:
        Tuple[float, float, float]: (specific_work_j_kg, T_out_k, h_out_actual)
    """
    # 1. Inlet State
    s_in = bilinear_interp_jit(P_grid, T_grid, S_lut, p_in_pa, T_in_k)
    h_in = bilinear_interp_jit(P_grid, T_grid, H_lut, p_in_pa, T_in_k)    
    # 2. Isentropic Outlet State (Constant Entropy)
    # H_from_PS_lut uses (P, S) coordinates
    h_out_isen = bilinear_interp_jit(P_grid, S_grid, H_from_PS_lut, p_out_pa, s_in)
    
    # 3. Actual Work
    w_isen = h_out_isen - h_in
    w_actual = w_isen / efficiency
    
    # 4. Actual Outlet Enthalpy
    h_out_actual = h_in + w_actual
    
    # 5. Solve for Outlet Temperature
    # T_guess estimation (Ideal gas relation)
    gamma = 1.41
    exponent = (gamma - 1.0) / gamma
    T_guess = T_in_k * (p_out_pa / p_in_pa)**exponent
    
    # Clamp T_guess to LUT bounds (273.15 K to 1200 K)
    T_guess = max(273.15, min(T_guess, 1200.0))
    
    T_out_k = solve_temperature_from_enthalpy_jit(
        h_out_actual,
        p_out_pa,
        T_guess,
        P_grid,
        T_grid,
        H_lut,
        C_lut
    )
    
    return w_actual, T_out_k, h_out_actual


# =============================================================================
# CYCLONE SEPARATOR MECHANICS
# =============================================================================

@njit(cache=True)
def solve_cyclone_mechanics(
    Q_gas_m3s: float,
    rho_g: float,
    rho_l: float,
    mu_g: float,
    D_element_m: float,
    vane_angle_rad: float,
    N_tubes: int
) -> Tuple[float, float, float, float]:
    """
    Computes cyclone separation performance and hydrodynamics (JIT Compiled).
    
    Implements the Barth/Muschelknautz critical particle cut-size model (d₅₀)
    and Euler number pressure drop correlations for axial multi-cyclone separators.
    
    Physics Model:
        1. **Velocity Decomposition**:
           v_ax = Q / (N × A_annulus)
           v_tan = v_ax × tan(α)
           
        2. **Separation (Stokes Law in Centrifugal Field)**:
           d₅₀ = √[ 18μs_drift / ((ρ_l - ρ_g) × ω² × r × t_res) ]
           
        3. **Pressure Drop (Euler Method)**:
           ΔP = ξ × ½ρv_ax²
           
    Args:
        Q_gas_m3s (float): Actual gas volumetric flow (m³/s).
        rho_g (float): Gas density (kg/m³).
        rho_l (float): Liquid density (kg/m³).
        mu_g (float): Gas dynamic viscosity (Pa·s).
        D_element_m (float): Cyclone tube internal diameter (m).
        vane_angle_rad (float): Inlet vane angle (radians).
        N_tubes (int): Number of active cyclone elements.
        
    Returns:
        Tuple[float, float, float, float]: 
            - d50_microns: Cut-size diameter (μm).
            - delta_P_pa: Pressure drop (Pa).
            - v_axial: Axial velocity (m/s).
            - v_tan: Tangential velocity (m/s).
            
    References:
        Hoffmann, A.C. & Stein, L.E. (2008). Gas Cyclones and Swirl Tubes.
        Coker, A.K. (2007). Ludwig's Applied Process Design. Vol. 1.
    """
    if N_tubes <= 0 or Q_gas_m3s <= 1e-9:
        return 0.0, 0.0, 0.0, 0.0

    # --- GEOMETRY DEFINITION ---
    # Hub obstruction ratio fixed at 0.3 per Hoffmann & Stein (Ref [3])
    D_hub = 0.3 * D_element_m
    Area_annulus = (np.pi / 4.0) * (D_element_m**2 - D_hub**2)
    
    # --- VELOCITY FIELD ---
    v_axial = Q_gas_m3s / (N_tubes * Area_annulus)
    v_tan = v_axial * np.tan(vane_angle_rad)
    
    # --- SEPARATION PHYSICS (Barth/Muschelknautz) ---
    # Geometric mean radius for spin acceleration
    r_mean = (D_element_m + D_hub) / 4.0
    
    # Centrifugal acceleration: a_c = v_tan² / r
    g_spin = (v_tan**2) / r_mean
    
    # Residence time (t_res) and Drift Distance (s_drift)
    L_sep = 3.0 * D_element_m  # Separation length ~3× tube diameter
    t_res = L_sep / v_axial if v_axial > 1e-9 else 1e6
    s_drift = (D_element_m - D_hub) / 2.0
    
    # Stokes' Law application for cut-size diameter
    # d₅₀² = 18μs / ((ρ_l - ρ_g) × g_spin × t_res)
    # Note: 1e6 factor converts m to μm
    density_diff = rho_l - rho_g
    if density_diff > 0 and g_spin > 0 and t_res > 0:
        d50_sq = (18.0 * mu_g * s_drift) / (density_diff * g_spin * t_res)
        d50_microns = np.sqrt(d50_sq) * 1e6
    else:
        d50_microns = 0.0

    # --- FLUID DYNAMICS (PRESSURE LOSS) ---
    # Euler Number approximation: ξ ≈ 4.8 for 45-degree vanes (Coker/Ludwig)
    Xi = 4.8 
    delta_P_pa = Xi * 0.5 * rho_g * (v_axial**2)
    
    return d50_microns, delta_P_pa, v_axial, v_tan


# =============================================================================
# MIXTURE THERMODYNAMICS JIT (OPTIMIZED)
# =============================================================================

@njit(cache=True)
def get_interp_weights_jit(
    grid_x: npt.NDArray[np.float64],
    grid_y: npt.NDArray[np.float64],
    x: float,
    y: float
) -> Tuple[int, int, float, float]:
    """
    Calculate bilinear interpolation weights once for reuse across multiple properties.
    
    Returns:
        ix, iy (indices of top-left corner)
        wx, wy (interpolation weights for x and y)
    """
    # X Grid Search (searchsorted is O(log N))
    if x <= grid_x[0]:
        ix = 0
        wx = 0.0
    elif x >= grid_x[-1]:
        ix = len(grid_x) - 2
        wx = 1.0
    else:
        ix = np.searchsorted(grid_x, x) - 1
        x0 = grid_x[ix]
        x1 = grid_x[ix+1]
        wx = (x - x0) / (x1 - x0)

    # Y Grid Search
    if y <= grid_y[0]:
        iy = 0
        wy = 0.0
    elif y >= grid_y[-1]:
        iy = len(grid_y) - 2
        wy = 1.0
    else:
        iy = np.searchsorted(grid_y, y) - 1
        y0 = grid_y[iy]
        y1 = grid_y[iy+1]
        wy = (y - y0) / (y1 - y0)
        
    return ix, iy, wx, wy

@njit(cache=True)
def interp_from_weights_jit(
    data: npt.NDArray[np.float64], # 2D array
    ix: int,
    iy: int,
    wx: float,
    wy: float
) -> float:
    """Apply pre-calculated weights to a data grid."""
    # f(x, y) = (1-wx)(1-wy)f00 + wx(1-wy)f10 + (1-wx)wy*f01 + wx*wy*f11
    
    # Boundary protection is handled by clamping ix/iy in weight calc
    # But data might be smaller? Assuming aligned.
    f00 = data[ix, iy]
    f10 = data[ix+1, iy]
    f01 = data[ix, iy+1]
    f11 = data[ix+1, iy+1]
    
    return (1.0-wx)*(1.0-wy)*f00 + wx*(1.0-wy)*f10 + (1.0-wx)*wy*f01 + wx*wy*f11

@njit(cache=True)
def get_mix_cp_jit(
    P_grid: np.ndarray,
    T_grid: np.ndarray,
    C_luts: np.ndarray,  # (N_fl, NP, NT)
    weights: np.ndarray, # Mass fractions
    ix: int, iy: int, wx: float, wy: float # Pre-calculated context
) -> float:
    """Calculate mixture Cp (mass weighted) using pre-calc weights."""
    cp_mix = 0.0
    n = len(weights)
    for i in range(n):
        if weights[i] > 1e-9:
            c_val = interp_from_weights_jit(C_luts[i], ix, iy, wx, wy)
            cp_mix += weights[i] * c_val
    return cp_mix

@njit(cache=True)
def get_mix_enthalpy_fast_jit(
    H_luts: np.ndarray,
    weights: np.ndarray,
    ix: int, iy: int, wx: float, wy: float
) -> float:
    """Calculate mixture H (mass weighted) using pre-calc weights."""
    h_mix = 0.0
    for i in range(len(weights)):
        if weights[i] > 1e-9:
            val = interp_from_weights_jit(H_luts[i], ix, iy, wx, wy)
            h_mix += weights[i] * val
    return h_mix

@njit(cache=True)
def get_mix_density_jit(
    D_luts: np.ndarray,
    weights: np.ndarray,
    ix: int, iy: int, wx: float, wy: float
) -> float:
    """
    Calculate mixture density (Amagat's Law/Volume Additivity) using pre-calc weights.
    rho_mix = 1 / Sum(w_i / rho_i)
    """
    sum_vol_spec = 0.0
    for i in range(len(weights)):
        if weights[i] > 1e-9:
            rho_i = interp_from_weights_jit(D_luts[i], ix, iy, wx, wy)
            if rho_i > 1e-6:
                sum_vol_spec += weights[i] / rho_i
    
    if sum_vol_spec > 1e-9:
        return 1.0 / sum_vol_spec
    return 0.0

@njit(cache=True)
def calculate_mixture_density_jit(
    p_pa: float,
    T_k: float,
    P_grid: np.ndarray,
    T_grid: np.ndarray,
    D_luts: np.ndarray,
    weights: np.ndarray
) -> float:
    """
    Calculate mixture density using JIT and stacked LUTs.
    """
    ix, iy, wx, wy = get_interp_weights_jit(P_grid, T_grid, p_pa, T_k)
    return get_mix_density_jit(D_luts, weights, ix, iy, wx, wy)

@njit(cache=True)
def get_mix_entropy_fast_jit(
    S_luts: np.ndarray,
    weights: np.ndarray,
    mole_fracs: np.ndarray,
    M_mix_kg_mol: float,
    sum_ylny: float, # Pre-calculated Sum(y ln y)
    ix: int, iy: int, wx: float, wy: float
) -> float:
    """Calculate mixture S (mass weighted + mixing term) using pre-calc weights."""
    R_UNIVERSAL = 8.314462618
    
    # Base S
    s_base = 0.0
    for i in range(len(weights)):
        if weights[i] > 1e-9:
            val = interp_from_weights_jit(S_luts[i], ix, iy, wx, wy)
            s_base += weights[i] * val
            
    # Mixing Term: s_mix = - (R / M_mix) * Sum(y ln y)
    # Sum(y ln y) is negative. R_mix > 0. s_mixing > 0.
    if M_mix_kg_mol > 1e-9:
        R_mix = R_UNIVERSAL / M_mix_kg_mol
        s_mixing = -R_mix * sum_ylny
    else:
        s_mixing = 0.0
        
    return s_base + s_mixing

@njit(cache=True)
def calculate_mixture_compression_jit(
    p_in_pa: float,
    p_out_pa: float,
    T_in_k: float,
    efficiency: float,
    P_grid: np.ndarray,
    T_grid: np.ndarray,
    H_luts: np.ndarray, # (N, NP, NT)
    S_luts: np.ndarray, 
    C_luts: np.ndarray, # Needed for derivative
    weights: np.ndarray,
    mole_fracs: np.ndarray,
    M_mix_kg_mol: float,
    sum_ylny: float
) -> Tuple[float, float, float]:
    """
    Calculate real-gas mixture compression using JIT and Cp-based derivatives.
    """
    # 1. Inlet State
    ix_in, iy_in, wx_in, wy_in = get_interp_weights_jit(P_grid, T_grid, p_in_pa, T_in_k)
    
    # Verify grid bounds (if P_in is very different, ix relies on P lookup logic)
    # Note: get_interp_weights_jit handles search.
    
    s_in = get_mix_entropy_fast_jit(S_luts, weights, mole_fracs, M_mix_kg_mol, sum_ylny, ix_in, iy_in, wx_in, wy_in)
    h_in = get_mix_enthalpy_fast_jit(H_luts, weights, ix_in, iy_in, wx_in, wy_in)
    
    # 2. Isentropic Step: Find T_out_isen such that S_mix(p_out, T) = s_in
    # Newton-Raphson on T.  dS/dT = Cp/T (Isobaric)
    
    T_guess = T_in_k * (p_out_pa / p_in_pa) ** 0.286
    
    # Safety bounds
    if T_guess < 200.0: T_guess = 200.0
    if T_guess > 1200.0: T_guess = 1200.0
    
    for _ in range(8): # Reduced iterations as requested
        # Calc properties at T_guess (P_out fixed)
        ix_out, iy_out, wx_out, wy_out = get_interp_weights_jit(P_grid, T_grid, p_out_pa, T_guess)
        
        s_guess = get_mix_entropy_fast_jit(S_luts, weights, mole_fracs, M_mix_kg_mol, sum_ylny, ix_out, iy_out, wx_out, wy_out)
        
        diff = s_guess - s_in
        if abs(diff) < 1e-4: 
            break
            
        # Analytic Derivative: dS/dT = Cp_mix / T
        cp_mix = get_mix_cp_jit(P_grid, T_grid, C_luts, weights, ix_out, iy_out, wx_out, wy_out)
        if abs(cp_mix) < 1e-9: cp_mix = 14000.0 # Safety
        
        ds_dt = cp_mix / T_guess
        
        # Update
        delta_T = diff / ds_dt
        
        # Limiter to prevent overshoot
        if delta_T > 50.0: delta_T = 50.0
        if delta_T < -50.0: delta_T = -50.0
        
        T_guess = T_guess - delta_T
        if T_guess < 100.0: T_guess = 100.0
        if T_guess > 2000.0: T_guess = 2000.0
        
    # Isentropic done. Get H_isen at final T
    # Need new weights for final T
    ix_iso, iy_iso, wx_iso, wy_iso = get_interp_weights_jit(P_grid, T_grid, p_out_pa, T_guess)
    h_out_isen = get_mix_enthalpy_fast_jit(H_luts, weights, ix_iso, iy_iso, wx_iso, wy_iso)
    
    # 3. Work
    w_isen = h_out_isen - h_in
    w_actual = w_isen / efficiency
    h_out_actual = h_in + w_actual
    
    # 4. Actual Outlet T: Match H(P_out, T) = h_out_actual
    # Start guess at T_isen (T_guess)
    T_act = T_guess 
    
    for _ in range(8):
        ix_act, iy_act, wx_act, wy_act = get_interp_weights_jit(P_grid, T_grid, p_out_pa, T_act)
        
        h_val = get_mix_enthalpy_fast_jit(H_luts, weights, ix_act, iy_act, wx_act, wy_act)
        cp_val = get_mix_cp_jit(P_grid, T_grid, C_luts, weights, ix_act, iy_act, wx_act, wy_act)
        
        diff = h_val - h_out_actual
        if abs(diff) < 1.0: # J/kg
            break
            
        if abs(cp_val) < 1e-9: cp_val = 14000.0
        
        # dH/dT = Cp
        delta_T = diff / cp_val
        if delta_T > 50.0: delta_T = 50.0
        if delta_T < -50.0: delta_T = -50.0
        
        T_act = T_act - delta_T
        if T_act < 100.0: T_act = 100.0
        if T_act > 2000.0: T_act = 2000.0
        
    return w_actual, T_act, h_out_actual


# =============================================================================
# TEMP-LIMITED PRESSURE SOLVER (Replaces Python bisection loop)
# =============================================================================

@njit(cache=True)
def solve_temp_limited_pressure_jit(
    p_in_pa: float,
    p_max_pa: float,
    T_in_k: float,
    T_max_k: float,
    efficiency: float,
    P_grid: np.ndarray,
    T_grid: np.ndarray,
    H_luts: np.ndarray,
    S_luts: np.ndarray,
    C_luts: np.ndarray,
    weights: np.ndarray,
    mole_fracs: np.ndarray,
    M_mix_kg_mol: float,
    sum_ylny: float
) -> Tuple[float, float, float]:
    """
    Solve for maximum outlet pressure that satisfies temperature constraint.
    
    Entirely JIT-compiled bisection + compression calculation.
    Replaces Python loop with 30 scalar calls to _compute_outlet_temp.
    
    Args:
        p_in_pa: Inlet pressure (Pa)
        p_max_pa: Maximum target pressure (Pa) - upper bound for search
        T_in_k: Inlet temperature (K)
        T_max_k: Maximum allowed outlet temperature (K)
        efficiency: Isentropic efficiency (0-1)
        P_grid, T_grid: LUT grids
        H_luts, S_luts, C_luts: Stacked property LUTs (N_fluids, NP, NT)
        weights: Mass fractions array (canonical order)
        mole_fracs: Mole fractions array
        M_mix_kg_mol: Mixture molar mass (kg/mol)
        sum_ylny: Pre-computed entropy mixing term
        
    Returns:
        Tuple[float, float, float]: (P_out_pa, T_out_k, W_actual_J_kg)
    """
    # 1. Check if target pressure is achievable
    T_at_max = _compute_T_out_for_P_jit(
        p_in_pa, p_max_pa, T_in_k, efficiency,
        P_grid, T_grid, H_luts, S_luts, C_luts,
        weights, mole_fracs, M_mix_kg_mol, sum_ylny
    )
    
    if T_at_max <= T_max_k:
        # Target pressure achievable - compute full result
        w_act, T_out, h_out = calculate_mixture_compression_jit(
            p_in_pa, p_max_pa, T_in_k, efficiency,
            P_grid, T_grid, H_luts, S_luts, C_luts,
            weights, mole_fracs, M_mix_kg_mol, sum_ylny
        )
        return p_max_pa, T_out, w_act
    
    # 2. Bisection search for P_out such that T_out = T_max
    p_low = p_in_pa
    p_high = p_max_pa
    
    for _ in range(25):  # Converges to <0.1% in ~20 iterations
        p_mid = (p_low + p_high) * 0.5
        
        T_mid = _compute_T_out_for_P_jit(
            p_in_pa, p_mid, T_in_k, efficiency,
            P_grid, T_grid, H_luts, S_luts, C_luts,
            weights, mole_fracs, M_mix_kg_mol, sum_ylny
        )
        
        if T_mid <= T_max_k:
            p_low = p_mid
        else:
            p_high = p_mid
        
        # Convergence check
        if abs(p_high - p_low) / p_low < 0.001:
            break
    
    # 3. Calculate final compression at converged pressure
    p_out_final = p_low
    w_act, T_out, h_out = calculate_mixture_compression_jit(
        p_in_pa, p_out_final, T_in_k, efficiency,
        P_grid, T_grid, H_luts, S_luts, C_luts,
        weights, mole_fracs, M_mix_kg_mol, sum_ylny
    )
    
    return p_out_final, T_out, w_act


@njit(cache=True)
def _compute_T_out_for_P_jit(
    p_in_pa: float,
    p_out_pa: float,
    T_in_k: float,
    efficiency: float,
    P_grid: np.ndarray,
    T_grid: np.ndarray,
    H_luts: np.ndarray,
    S_luts: np.ndarray,
    C_luts: np.ndarray,
    weights: np.ndarray,
    mole_fracs: np.ndarray,
    M_mix_kg_mol: float,
    sum_ylny: float
) -> float:
    """
    Compute outlet temperature for given P_out (helper for bisection).
    Lightweight version: doesn't return work, just T_out.
    """
    # Inlet state
    ix_in, iy_in, wx_in, wy_in = get_interp_weights_jit(P_grid, T_grid, p_in_pa, T_in_k)
    s_in = get_mix_entropy_fast_jit(S_luts, weights, mole_fracs, M_mix_kg_mol, sum_ylny, ix_in, iy_in, wx_in, wy_in)
    h_in = get_mix_enthalpy_fast_jit(H_luts, weights, ix_in, iy_in, wx_in, wy_in)
    
    # Initial guess (ideal gas)
    T_guess = T_in_k * (p_out_pa / p_in_pa) ** 0.286
    if T_guess < 200.0: T_guess = 200.0
    if T_guess > 1200.0: T_guess = 1200.0
    
    # Newton for isentropic T
    for _ in range(6):  # Reduced iterations for speed
        ix_out, iy_out, wx_out, wy_out = get_interp_weights_jit(P_grid, T_grid, p_out_pa, T_guess)
        s_guess = get_mix_entropy_fast_jit(S_luts, weights, mole_fracs, M_mix_kg_mol, sum_ylny, ix_out, iy_out, wx_out, wy_out)
        
        diff = s_guess - s_in
        if abs(diff) < 1e-3:
            break
        
        cp_mix = get_mix_cp_jit(P_grid, T_grid, C_luts, weights, ix_out, iy_out, wx_out, wy_out)
        if abs(cp_mix) < 1e-9: cp_mix = 14000.0
        
        ds_dt = cp_mix / T_guess
        delta_T = diff / ds_dt
        if delta_T > 30.0: delta_T = 30.0
        if delta_T < -30.0: delta_T = -30.0
        
        T_guess = T_guess - delta_T
        if T_guess < 100.0: T_guess = 100.0
        if T_guess > 1500.0: T_guess = 1500.0
    
    # Get isentropic enthalpy
    ix_iso, iy_iso, wx_iso, wy_iso = get_interp_weights_jit(P_grid, T_grid, p_out_pa, T_guess)
    h_out_isen = get_mix_enthalpy_fast_jit(H_luts, weights, ix_iso, iy_iso, wx_iso, wy_iso)
    
    # Actual work and outlet enthalpy
    w_isen = h_out_isen - h_in
    w_actual = w_isen / efficiency
    h_out_actual = h_in + w_actual
    
    # Newton for actual T
    T_act = T_guess
    for _ in range(6):
        ix_act, iy_act, wx_act, wy_act = get_interp_weights_jit(P_grid, T_grid, p_out_pa, T_act)
        h_val = get_mix_enthalpy_fast_jit(H_luts, weights, ix_act, iy_act, wx_act, wy_act)
        
        diff = h_val - h_out_actual
        if abs(diff) < 10.0:  # J/kg - looser tolerance for speed
            break
        
        cp_val = get_mix_cp_jit(P_grid, T_grid, C_luts, weights, ix_act, iy_act, wx_act, wy_act)
        if abs(cp_val) < 1e-9: cp_val = 14000.0
        
        delta_T = diff / cp_val
        if delta_T > 30.0: delta_T = 30.0
        if delta_T < -30.0: delta_T = -30.0
        
        T_act = T_act - delta_T
        if T_act < 100.0: T_act = 100.0
        if T_act > 2000.0: T_act = 2000.0
    
    return T_act


# =============================================================================

