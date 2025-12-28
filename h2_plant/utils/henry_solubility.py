"""
Henry's Law Dissolved Gas Solubility Calculator.

This module provides a centralized implementation of Henry's Law for calculating
gas solubility in liquid water. Used by separation components (KOD, Coalescer,
Cyclone, TSA) to compute dissolved gas losses in drain streams.

Physical Basis:
    Henry's Law states that at constant temperature, the amount of gas dissolved
    in a liquid is directly proportional to the partial pressure of that gas:
    
        C = P_gas / H(T)
    
    where:
        C = molar concentration (mol/L)
        P_gas = partial pressure of gas (atm)
        H(T) = temperature-dependent Henry's constant (L·atm/mol)
    
    Temperature dependence follows van't Hoff:
        H(T) = H_298 * exp(C * (1/T - 1/298.15))
    
    where C = ΔH_soln / R (temperature coefficient).

References:
    - Sander, R. (2015). Compilation of Henry's law constants.
    - NIST Chemistry WebBook.
"""

import math
from typing import Literal

from h2_plant.core.constants import HenryConstants


def calculate_dissolved_gas_mg_kg(
    temperature_k: float,
    gas_partial_pressure_pa: float,
    gas_species: Literal['H2', 'O2'] = 'H2'
) -> float:
    """
    Calculate dissolved gas concentration in liquid water using Henry's Law.
    
    Args:
        temperature_k: Liquid temperature in Kelvin.
        gas_partial_pressure_pa: Partial pressure of the gas species (Pa).
            IMPORTANT: This should be the gas partial pressure (P_total * y_gas),
            NOT the total system pressure.
        gas_species: Target gas species ('H2' or 'O2').
    
    Returns:
        Dissolved gas concentration in mg per kg of water.
        Returns 0.0 for invalid inputs or unknown species.
    
    Example:
        >>> # H2 dissolved in water at 25°C, 10 bar partial pressure
        >>> conc = calculate_dissolved_gas_mg_kg(298.15, 10e5, 'H2')
        >>> print(f"{conc:.3f} mg/kg")
        15.876 mg/kg
    """
    if temperature_k <= 0 or gas_partial_pressure_pa <= 0:
        return 0.0
    
    # Get species-specific constants
    if gas_species == 'H2':
        H_298 = HenryConstants.H2_H_298_L_ATM_MOL  # L·atm/mol at 298.15K
        C = HenryConstants.H2_DELTA_H_R_K          # K (temperature coefficient)
        MW_kg_mol = HenryConstants.H2_MOLAR_MASS_KG_MOL
    elif gas_species == 'O2':
        H_298 = HenryConstants.O2_H_298_L_ATM_MOL
        C = HenryConstants.O2_DELTA_H_R_K
        MW_kg_mol = HenryConstants.O2_MOLAR_MASS_KG_MOL
    else:
        return 0.0
    
    T0 = 298.15  # Reference temperature (K)
    
    # Temperature-corrected Henry constant (L·atm/mol)
    H_T = H_298 * math.exp(C * (1.0 / temperature_k - 1.0 / T0))
    
    # Convert pressure to atm
    p_atm = gas_partial_pressure_pa / 101325.0
    
    # Molar concentration (mol/L)
    c_mol_L = p_atm / H_T
    
    # Convert to mass concentration (mg/kg)
    # Assuming water density ~1 kg/L, so mol/L ≈ mol/kg_water
    mw_g_mol = MW_kg_mol * 1000.0  # kg/mol -> g/mol
    c_mg_kg = c_mol_L * mw_g_mol * 1000.0  # mol/L * g/mol * 1000 mg/g
    
    return c_mg_kg


def calculate_dissolved_gas_kg_h(
    liquid_flow_kg_h: float,
    temperature_k: float,
    gas_partial_pressure_pa: float,
    gas_species: Literal['H2', 'O2'] = 'H2'
) -> float:
    """
    Calculate dissolved gas mass flow rate in a liquid stream.
    
    Args:
        liquid_flow_kg_h: Liquid water mass flow rate (kg/h).
        temperature_k: Liquid temperature (K).
        gas_partial_pressure_pa: Partial pressure of gas in contact (Pa).
        gas_species: Target gas species ('H2' or 'O2').
    
    Returns:
        Dissolved gas mass flow rate (kg/h).
    
    Example:
        >>> # 100 kg/h of water at 25°C, 10 bar H2 partial pressure
        >>> loss = calculate_dissolved_gas_kg_h(100.0, 298.15, 10e5, 'H2')
        >>> print(f"{loss:.6f} kg/h")
        0.001588 kg/h
    """
    if liquid_flow_kg_h <= 0:
        return 0.0
    
    concentration_mg_kg = calculate_dissolved_gas_mg_kg(
        temperature_k, gas_partial_pressure_pa, gas_species
    )
    
    # mg/kg * kg/h = mg/h, then convert to kg/h
    dissolved_gas_kg_h = liquid_flow_kg_h * (concentration_mg_kg / 1e6)
    
    return dissolved_gas_kg_h
