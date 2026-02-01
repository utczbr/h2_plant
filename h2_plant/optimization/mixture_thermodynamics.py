"""
Mixture Thermodynamics using Ideal Mixing of Real Gases.

This module provides thermodynamic property calculations for gas mixtures
by combining pure-component LUT lookups with proper mixing rules.

Physical Basis (Ideal Mixing of Real Gases):
    - Individual molecules follow real-gas behavior (via LUTs)
    - Unlike molecules do not interact (k_ij = 0)
    - Significantly more accurate than ideal gas law
    - Avoids computational cost of rigorous EOS mixtures

Key Equations:
    Enthalpy:   h_mix = Σ w_i · h_i(P, T)
    Entropy:    s_mix = Σ w_i · s_i(P, T) - R_mix · Σ y_i · ln(y_i)
    Density:    1/ρ_mix = Σ w_i / ρ_i(P, T)  [Amagat's Law]

References:
    - Smith, Van Ness, Abbott: Introduction to Chemical Engineering Thermodynamics
    - Perry's Chemical Engineers' Handbook, 8th Ed., Chapter 4
"""

import numpy as np
from typing import Dict, Tuple, Optional, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from h2_plant.optimization.lut_manager import LUTManager

logger = logging.getLogger(__name__)

# Universal gas constant
R_UNIVERSAL = 8.314  # J/(mol·K)

# Molar masses (kg/mol) for supported species
MOLAR_MASSES = {
    'H2': 0.002016,
    'O2': 0.032000,
    'N2': 0.028014,
    'CH4': 0.016043,
    'CO2': 0.044010,
    'H2O': 0.018015,
}


def get_mixture_enthalpy(
    composition_mass: Dict[str, float],
    P_pa: float,
    T_k: float,
    lut_manager: 'LUTManager'
) -> float:
    """
    Calculate mixture specific enthalpy using mass-weighted pure component enthalpies.
    
    h_mix = Σ w_i · h_i(P, T)
    
    Args:
        composition_mass: Mass fractions {species: w_i}
        P_pa: System pressure [Pa]
        T_k: System temperature [K]
        lut_manager: LUTManager instance for property lookups
        
    Returns:
        Mixture specific enthalpy [J/kg]
    """
    h_mix = 0.0
    
    for species, w_i in composition_mass.items():
        if w_i < 1e-9:
            continue
        if species in ('H2O_liq',):
            # Liquid water - use H2O LUT at saturation or skip
            continue
            
        try:
            h_i = lut_manager.lookup(species, 'H', P_pa, T_k)
            h_mix += w_i * h_i
        except (ValueError, KeyError):
            logger.debug(f"LUT lookup failed for {species}, skipping")
            continue
            
    return h_mix


def get_mixture_entropy(
    composition_mass: Dict[str, float],
    composition_mole: Dict[str, float],
    P_pa: float,
    T_k: float,
    lut_manager: 'LUTManager',
    M_mix_kg_mol: Optional[float] = None
) -> float:
    """
    Calculate mixture specific entropy including mixing correction.
    
    s_mix = Σ w_i · s_i(P, T) - R_mix · Σ y_i · ln(y_i)
    
    The second term corrects for the irreversibility of mixing gases.
    It is always positive, representing entropy generation.
    
    Args:
        composition_mass: Mass fractions {species: w_i}
        composition_mole: Mole fractions {species: y_i}
        P_pa: System pressure [Pa]
        T_k: System temperature [K]
        lut_manager: LUTManager instance
        M_mix_kg_mol: Optional pre-calculated mixture molar mass
        
    Returns:
        Mixture specific entropy [J/(kg·K)]
    """
    # 1. Base entropy: mass-weighted sum of pure components
    s_base = 0.0
    for species, w_i in composition_mass.items():
        if w_i < 1e-9 or species in ('H2O_liq',):
            continue
        try:
            s_i = lut_manager.lookup(species, 'S', P_pa, T_k)
            s_base += w_i * s_i
        except (ValueError, KeyError):
            continue
    
    # 2. Calculate mixture molar mass if not provided
    if M_mix_kg_mol is None:
        M_mix_kg_mol = 0.0
        for species, y_i in composition_mole.items():
            if y_i > 1e-9 and species in MOLAR_MASSES:
                M_mix_kg_mol += y_i * MOLAR_MASSES[species]
    
    if M_mix_kg_mol < 1e-9:
        return s_base  # Can't compute mixing term
    
    # 3. Entropy of mixing correction
    # Δs_mix = -R_mix · Σ y_i · ln(y_i)
    # R_mix = R / M_mix [J/(kg·K)]
    R_mix = R_UNIVERSAL / M_mix_kg_mol
    
    entropy_gen = 0.0
    for species, y_i in composition_mole.items():
        if y_i > 1e-9 and species not in ('H2O_liq',):
            entropy_gen += y_i * np.log(y_i)  # This is negative (ln of value < 1)
    
    # -R_mix * (negative sum) = positive correction
    s_mixing = -R_mix * entropy_gen
    
    return s_base + s_mixing


def get_mixture_density(
    composition_mass: Dict[str, float],
    P_pa: float,
    T_k: float,
    lut_manager: 'LUTManager'
) -> float:
    """
    Calculate mixture density using Amagat's Law (additive volumes).
    
    1/ρ_mix = Σ w_i / ρ_i(P, T)
    
    Args:
        composition_mass: Mass fractions {species: w_i}
        P_pa: System pressure [Pa]
        T_k: System temperature [K]
        lut_manager: LUTManager instance
        
    Returns:
        Mixture density [kg/m³]
    """
    inv_rho_mix = 0.0
    
    for species, w_i in composition_mass.items():
        if w_i < 1e-9 or species in ('H2O_liq',):
            continue
        try:
            rho_i = lut_manager.lookup(species, 'D', P_pa, T_k)
            if rho_i > 1e-9:
                inv_rho_mix += w_i / rho_i
        except (ValueError, KeyError):
            # Fallback: ideal gas density for this component
            if species in MOLAR_MASSES:
                R_spec = R_UNIVERSAL / MOLAR_MASSES[species]
                rho_ideal = P_pa / (R_spec * T_k)
                inv_rho_mix += w_i / rho_ideal
    
    if inv_rho_mix > 1e-12:
        return 1.0 / inv_rho_mix
    return 0.0


def get_mixture_molar_mass(composition_mole: Dict[str, float]) -> float:
    """
    Calculate mixture molar mass from mole fractions.
    
    M_mix = Σ y_i · M_i
    
    Returns:
        Mixture molar mass [kg/mol]
    """
    M_mix = 0.0
    for species, y_i in composition_mole.items():
        if y_i > 1e-9 and species in MOLAR_MASSES:
            M_mix += y_i * MOLAR_MASSES[species]
    return M_mix


def solve_isentropic_outlet_temperature(
    composition_mass: Dict[str, float],
    composition_mole: Dict[str, float],
    P_in_pa: float,
    T_in_k: float,
    P_out_pa: float,
    lut_manager: 'LUTManager',
    max_iter: int = 10,
    tol_k: float = 0.1
) -> Tuple[float, float, float]:
    """
    Find outlet temperature for isentropic compression using Newton-Raphson.
    
    Solves: s_mix(P_out, T_out) = s_mix(P_in, T_in)
    
    Since we cannot invert s(T) directly for mixtures, we iterate on T
    until entropy matches the inlet value.
    
    Args:
        composition_mass: Mass fractions
        composition_mole: Mole fractions
        P_in_pa: Inlet pressure [Pa]
        T_in_k: Inlet temperature [K]
        P_out_pa: Outlet pressure [Pa]
        lut_manager: LUTManager instance
        max_iter: Maximum iterations
        tol_k: Temperature convergence tolerance [K]
        
    Returns:
        Tuple of (T_out_isentropic [K], h_in [J/kg], h_out_isen [J/kg])
    """
    M_mix = get_mixture_molar_mass(composition_mole)
    
    # Calculate inlet state
    h_in = get_mixture_enthalpy(composition_mass, P_in_pa, T_in_k, lut_manager)
    s_in = get_mixture_entropy(composition_mass, composition_mole, 
                                P_in_pa, T_in_k, lut_manager, M_mix)
    
    # Initial guess: ideal gas approximation
    # T_out_ideal = T_in * (P_out / P_in)^((γ-1)/γ)
    # For H2-rich mixtures, γ ≈ 1.4
    gamma_approx = 1.4
    pressure_ratio = P_out_pa / P_in_pa
    T_guess = T_in_k * (pressure_ratio ** ((gamma_approx - 1) / gamma_approx))
    
    T_out = T_guess
    
    for iteration in range(max_iter):
        # Calculate entropy at current guess
        s_out = get_mixture_entropy(composition_mass, composition_mole,
                                     P_out_pa, T_out, lut_manager, M_mix)
        
        # Error: we want s_out = s_in
        error = s_out - s_in
        
        # Check convergence
        # For ds/dT ≈ Cp/T, so dT ≈ T * error / Cp
        # Approximate Cp from enthalpy gradient
        dT_probe = 1.0  # 1 K probe
        h_plus = get_mixture_enthalpy(composition_mass, P_out_pa, T_out + dT_probe, lut_manager)
        h_minus = get_mixture_enthalpy(composition_mass, P_out_pa, T_out, lut_manager)
        Cp_approx = (h_plus - h_minus) / dT_probe
        
        if Cp_approx < 100:  # Sanity check
            Cp_approx = 1000.0  # Conservative fallback
        
        # Newton step: ds/dT ≈ Cp / T
        dsdt = Cp_approx / T_out
        dT = -error / dsdt if abs(dsdt) > 1e-6 else 0.0
        
        # Limit step size for stability
        dT = np.clip(dT, -50.0, 50.0)
        
        T_out += dT
        T_out = max(T_out, T_in_k)  # Temperature must increase in compression
        
        if abs(dT) < tol_k:
            break
    
    # Calculate final outlet enthalpy
    h_out = get_mixture_enthalpy(composition_mass, P_out_pa, T_out, lut_manager)
    
    return T_out, h_in, h_out


def calculate_compression_work(
    composition_mass: Dict[str, float],
    composition_mole: Dict[str, float],
    P_in_pa: float,
    T_in_k: float,
    P_out_pa: float,
    isentropic_efficiency: float,
    lut_manager: 'LUTManager'
) -> Tuple[float, float, float]:
    """
    Calculate compression work and outlet temperature for a mixture.
    
    Uses the isentropic outlet temperature solver, then applies
    efficiency correction to get actual outlet conditions.
    
    Args:
        composition_mass: Mass fractions
        composition_mole: Mole fractions
        P_in_pa: Inlet pressure [Pa]
        T_in_k: Inlet temperature [K]
        P_out_pa: Outlet pressure [Pa]
        isentropic_efficiency: Compressor efficiency (0-1)
        lut_manager: LUTManager instance
        
    Returns:
        Tuple of:
            - w_specific [J/kg]: Actual specific work
            - T_out_actual [K]: Actual outlet temperature
            - T_out_isen [K]: Isentropic outlet temperature
    """
    # 1. Find isentropic outlet state
    T_out_isen, h_in, h_out_isen = solve_isentropic_outlet_temperature(
        composition_mass, composition_mole,
        P_in_pa, T_in_k, P_out_pa,
        lut_manager
    )
    
    # 2. Calculate isentropic work
    w_isentropic = h_out_isen - h_in
    
    # 3. Apply efficiency
    w_actual = w_isentropic / isentropic_efficiency
    
    # 4. Find actual outlet temperature from h_actual = h_in + w_actual
    h_out_actual = h_in + w_actual
    
    # Newton iteration to find T where h(T) = h_out_actual
    T_out = T_out_isen / isentropic_efficiency  # Initial guess (overestimates)
    
    for _ in range(5):
        h_current = get_mixture_enthalpy(composition_mass, P_out_pa, T_out, lut_manager)
        
        # Approximate Cp
        dT = 1.0
        h_plus = get_mixture_enthalpy(composition_mass, P_out_pa, T_out + dT, lut_manager)
        Cp = (h_plus - h_current) / dT
        
        if Cp > 100:
            delta_T = (h_out_actual - h_current) / Cp
            delta_T = np.clip(delta_T, -50.0, 50.0)
            T_out += delta_T
            
            if abs(delta_T) < 0.1:
                break
    
    return w_actual, T_out, T_out_isen
