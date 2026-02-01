import numpy as np
from h2_plant.config.constants_physics import PEMConstants

CONST = PEMConstants()

def calculate_Urev(T: float, P_op: float) -> float:
    """
    Tensão reversível (Nernst).
    Exactly matches logic from pem_operator.py: calculate_Urev
    """
    # 1.229 - 0.9e-3 * (T - 298.15)
    U_rev_T = 1.229 - 0.9e-3 * (T - 298.15)
    
    pressure_ratio = P_op / CONST.P_ref
    
    # (R * T) / (z * F) * np.log(pressure_ratio**1.5)
    Nernst_correction = (CONST.R * T) / (CONST.z * CONST.F) * np.log(pressure_ratio**1.5)
    
    return U_rev_T + Nernst_correction

def calculate_Vcell_base(j: float, T: float, P_op: float) -> float:
    """
    Calcula Vcell BOL (Início da vida).
    Exactly matches logic from pem_operator.py: calculate_Vcell_base
    """
    U_rev = calculate_Urev(T, P_op)
    
    # eta_act = (R * T) / (alpha * z * F) * np.log(np.maximum(j, 1e-10) / j0)
    eta_act = (CONST.R * T) / (CONST.alpha * CONST.z * CONST.F) * np.log(np.maximum(j, 1e-10) / CONST.j0)
    
    # eta_ohm = j * (delta_mem / sigma_base)
    eta_ohm = j * (CONST.delta_mem / CONST.sigma_base)
    
    # eta_conc = np.where(j >= j_lim, 100.0, (R * T) / (z * F) * np.log(j_lim / (j_lim - np.maximum(j, 1e-10))))
    eta_conc = np.where(
        j >= CONST.j_lim, 
        100.0, 
        (CONST.R * T) / (CONST.z * CONST.F) * np.log(CONST.j_lim / (CONST.j_lim - np.maximum(j, 1e-10)))
    )
    
    return U_rev + eta_act + eta_ohm + eta_conc

def calculate_eta_F(j: float) -> float:
    """
    Eficiência de Faraday (perdas por crossover em baixa corrente).
    Exactly matches logic from pem_operator.py: calculate_eta_F
    """
    # np.maximum(j, 1e-6)**2 / (np.maximum(j, 1e-6)**2 + floss)
    j_safe = np.maximum(j, 1e-6)
    return j_safe**2 / (j_safe**2 + CONST.floss)

def calculate_flows(j: float) -> tuple:
    """
    Calculates instantaneous flows (kg/s) based on current density.
    Derived from end of run_pem_step in pem_operator.py
    """
    eta_F = calculate_eta_F(j)
    I_total = j * CONST.Area_Total
    
    m_H2_dot = (I_total * eta_F * CONST.MH2) / (CONST.z * CONST.F)
    m_O2_dot = (I_total * eta_F * (CONST.MO2 / 2.0)) / (CONST.z * CONST.F)
    m_H2O_dot = (I_total * eta_F * CONST.MH2O) / (CONST.z * CONST.F)
    
    return m_H2_dot, m_O2_dot, m_H2O_dot


# ============================================================================
# BACKWARD COMPATIBILITY FUNCTIONS
# ============================================================================

def calculate_Vcell(j: float, T: float, P_op: float, U_deg: float) -> float:
    """
    Calculate total Cell Voltage including degradation.
    Backward compatibility wrapper for tests.
    
    Args:
        j: Current density in A/cm²
        T: Temperature in Kelvin
        P_op: Operating pressure in Pascals
        U_deg: Degradation voltage in Volts
        
    Returns:
        Total cell voltage in Volts
    """
    V_base = calculate_Vcell_base(j, T, P_op)
    return V_base + U_deg

