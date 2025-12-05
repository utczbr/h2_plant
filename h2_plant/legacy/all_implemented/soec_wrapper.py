"""
Simplified wrapper for SOEC operation model.

This module provides a clean wrapper around the detailed SOEC physics model
for use in the component-based simulation architecture.
"""

import numpy as np
from typing import Tuple
from h2_plant.models.soec_operation import simular_passo_soec, atualizar_mapa_virtual
from h2_plant.config.constants_physics import SOECConstants

CONST = SOECConstants()

def soec_operation(
    P_setp: float,
    P_max_nom: float,
    n_mod: int,
    current_module: int,
    ramp_rate: float,
    f_rot: bool,
    f_eff: float,
    P_prev: float,
    t_h: float
) -> Tuple[float, int, float, float, float]:
    """
    Simplified wrapper for SOEC operation.
    
    Args:
        P_setp: Power setpoint (MW)
        P_max_nom: Maximum nominal power per module (MW)
        n_mod: Number of modules
        current_module: Currently active module count
        ramp_rate: Ramp rate (MW/min)
        f_rot: Rotation enabled flag
        f_eff: Efficient power limit ratio (e.g., 0.80)
        P_prev: Previous total power (MW)
        t_h: Operating time (hours)
    
    Returns:
        Tuple of (P_total_mw, active_modules, m_H2_kg_h, m_O2_kg_h, m_H2O_kg_h)
    """
    
    # Initialize state arrays for SOEC modules
    potencias = np.zeros(n_mod)
    est ados = np.ones(n_mod, dtype=np.int32)  # All in hot standby
    limites = np.full(n_mod, P_max_nom * f_eff)
    mapa_virtual = np.arange(n_mod, dtype=np.int32)
    
    # Set initial power based on previous state
    if P_prev > 0:
        # Distribute previous power across modules
        power_per_module = P_prev / n_mod
        potencias[:] = min(power_per_module, P_max_nom * f_eff)
    
    # Run one simulation step
    potencias, estados, limites, mapa_virtual, P_total = simular_passo_soec(
        potencia_referencia=P_setp,
        potencias_atuais_reais=potencias,
        estados_atuais_reais=estados,
        limites_reais=limites,
        mapa_virtual=mapa_virtual,
        rotacao_ativada=f_rot,
        modulos_desligados_reais=[],
        potencia_limite_eficiente=True
    )
    
    # Calculate H2 production (kg/h)
    # Using SOEC efficiency: 37.5 kWh/kg H2
    # H2 production rate: 1000 / 37.5 = 26.67 kg/MWh
    h2_rate_kg_per_mwh = 1000.0 / 37.5  # kg/MWh
    m_H2_kg_h = P_total * h2_rate_kg_per_mwh  # MW * (kg/MWh) = kg/h
    
    # Calculate O2 production (stoichiometric: 8 kg O2 per 1 kg H2)
    m_O2_kg_h = m_H2_kg_h * 8.0
    
    # Calculate H2O consumption (stoichiometric: 9 kg H2O per 1 kg H2)
    m_H2O_kg_h = m_H2_kg_h * 9.0
    
    # Count active modules (those with power > 0.01 MW)
    active_modules = np.sum(potencias > 0.01)
    
    return P_total, int(active_modules), m_H2_kg_h, m_O2_kg_h, m_H2O_kg_h
