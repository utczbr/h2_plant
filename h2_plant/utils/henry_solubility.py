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

Performance Note:
    Core calculations are JIT-compiled via Numba in h2_plant.optimization.numba_ops
    for ~10-100x speedup in simulation hot paths.
"""

from typing import Literal

from h2_plant.optimization.numba_ops import (
    calculate_dissolved_gas_mg_kg_jit,
    HENRY_H2_H298, HENRY_H2_C, HENRY_H2_MW,
    HENRY_O2_H298, HENRY_O2_C, HENRY_O2_MW
)


def calculate_dissolved_gas_mg_kg(
    temperature_k: float,
    gas_partial_pressure_pa: float,
    gas_species: Literal['H2', 'O2'] = 'H2'
) -> float:
    """
    Calculate dissolved gas concentration in liquid water using Henry's Law.
    
    This is a thin wrapper around the JIT-compiled numba implementation.
    
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
    if gas_species == 'H2':
        return calculate_dissolved_gas_mg_kg_jit(
            temperature_k, gas_partial_pressure_pa,
            HENRY_H2_H298, HENRY_H2_C, HENRY_H2_MW
        )
    elif gas_species == 'O2':
        return calculate_dissolved_gas_mg_kg_jit(
            temperature_k, gas_partial_pressure_pa,
            HENRY_O2_H298, HENRY_O2_C, HENRY_O2_MW
        )
    return 0.0


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
