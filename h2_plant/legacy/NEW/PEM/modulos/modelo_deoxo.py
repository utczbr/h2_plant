# modelo_deoxo_otimizado.py
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import CoolProp.CoolProp as CP 

# =================================================================
# DADOS DE DIMENSIONAMENTO E PARÂMETROS FIXOS (Do modelo anterior)
# =================================================================
V_REACTOR = 0.11        # Volume total do reator (m³) - (Mantido fixo)
D_R_M = 0.324           # Diâmetro do Reator (m) - (Mantido fixo)
# A_R_M2 será calculada dentro da função usando o diâmetro fixo.

# Propriedades do Catalisador e Reação
R = 8.314               # Constante dos Gases (J/mol·K)
k0_vol = 1.0e10         # Fator pré-exponencial Volumétrico (1/s)
Ea = 55000.0            # Energia de Ativação (J/mol)
DELTA_H_RXN = -242000.0 # Entalpia de Reação (J/mol de O2 consumido) - Exotérmica

# PARÂMETROS DE CONTROLE TÉRMICO (CONSTANTES IGNORADAS, APENAS PARA CONSISTÊNCIA)
T_JACKET_C = 50.0       
T_JACKET_K = T_JACKET_C + 273.15 
U_A = 5000.0            

# Massas Molares
M_O2 = 0.031998
M_H2O = 0.018015
M_H2 = 0.002016

# Capacidade Calorífica Molar Média (J/mol·K) - Simplificada
CP_MIX_EST = 29.5 

# Alvo de pureza (5 ppm de O2)
Y_O2_OUT_TARGET = 5.0e-6

# =================================================================
# FUNÇÃO DE MODELAGEM CENTRAL (MODELO ADIABÁTICO)
# =================================================================

def modelar_deoxo(m_dot_g_kg_s: float, P_in_bar: float, T_in_C: float, y_H2O_in: float, y_O2_in: float, L_M_input: float) -> dict:
    """
    Modela o reator Deoxo ADIABATICAMENTE (sem resfriamento), resolvendo as ODEs
    de Massa e Energia.
    L_M_input (m): Comprimento do reator a ser usado na simulação.
    """
    
    L_M = L_M_input # Usa o comprimento fornecido
    A_R_M2 = np.pi * (D_R_M**2) / 4.0 # Área da seção transversal
    
    # 1. Verificação de Condições Inválidas
    if m_dot_g_kg_s == 0 or y_O2_in == 0:
        T_out_C = T_in_C
        P_out_bar = P_in_bar
        y_O2_out = y_O2_in
        return {"T_C": T_out_C, "P_bar": P_out_bar, "y_O2_out": y_O2_out, "Q_dot_fluxo_W": 0.0, "W_dot_comp_W": 0.0, "X_O2": 0.0, "T_profile_C": np.array([T_in_C]), "L_span": np.array([0.0]), "T_max_calc": T_in_C, "Agua_Gerada_kg_s": 0.0, "y_H2O_out": y_H2O_in, "m_dot_gas_out_kg_s": m_dot_g_kg_s}

    T_in_K = T_in_C + 273.15
    P_in_Pa = P_in_bar * 100000.0 

    # 2. Cálculo da Vazão Molar Total
    M_GAS = M_H2 
    F_H2_molar = m_dot_g_kg_s / M_GAS
    y_H2_approx = 1.0 - y_H2O_in - y_O2_in
    F_O2_molar = F_H2_molar * (y_O2_in / y_H2_approx) if y_H2_approx > 0 else 0
    F_H2O_molar_in = F_H2_molar * (y_H2O_in / y_H2_approx) if y_H2_approx > 0 else 0
    dot_n_total = F_H2_molar + F_O2_molar + F_H2O_molar_in
    F_O2_in = F_O2_molar
    conversion_target = 1.0 - (Y_O2_OUT_TARGET / y_O2_in)
    
    # --- DEFINIÇÃO DAS EQUAÇÕES DIFERENCIAIS ---

    def pfr_ode_adiabatico(L, P):
        X = P[0]
        T = P[1]
        current_X = np.clip(X, 0.0, 1.0) 
        
        # 1. Balanço de Massa (dX/dL)
        if current_X >= conversion_target or current_X >= 1.0:
            r_O2 = 0.0
            dX_dL = 0.0
        else:
            k_eff_prime = k0_vol * np.exp(-Ea / (R * T))
            Y_O2 = y_O2_in * (1.0 - current_X)
            C_O2 = P_in_Pa * Y_O2 / (R * T)
            r_O2 = k_eff_prime * C_O2 
            dX_dL = (A_R_M2 / F_O2_in) * r_O2
            
        # 2. Balanço de Energia (dT/dL) - ADIABÁTICO!
        heat_generated = (-DELTA_H_RXN) * r_O2 # W/m³
        heat_removed = 0.0 # <--- TERMO DE RESFRIAMENTO REMOVIDO
        
        dT_dL = (A_R_M2 / (dot_n_total * CP_MIX_EST)) * (heat_generated - heat_removed)
        
        return [dX_dL, dT_dL]

    # --- SOLUÇÃO DO MODELO ---
    X0 = 0.0
    T0 = T_in_K
    L_span = np.linspace(0, L_M, 100) # Otimizado para o novo L_M
    sol = solve_ivp(pfr_ode_adiabatico, [0, L_M], [X0, T0], t_eval=L_span)

    X_profile = np.clip(sol.y[0], 0.0, 1.0)
    T_profile_K = sol.y[1]
    T_profile_C = T_profile_K - 273.15
    
    X_final = X_profile[-1]
    T_out_K = T_profile_K[-1]
    T_out_C = T_profile_C[-1]
    T_max_calc = np.max(T_profile_C) 
    
    # --- CÁLCULO DE PROPRIEDADES DE SAÍDA ---
    
    F_O2_consumed = F_O2_in * X_final 
    F_H2O_generated = F_O2_consumed * 2.0 
    
    m_dot_H2_consumed = F_O2_consumed * 2.0 * M_H2
    m_dot_g_out_kg_s = m_dot_g_kg_s - m_dot_H2_consumed
    m_dot_H2O_generated_kg_s = F_H2O_generated * M_H2O
    
    y_O2_out = y_O2_in * (1.0 - X_final)
    F_total_out = dot_n_total - F_O2_consumed + F_H2O_generated
    y_H2O_out = (F_H2O_molar_in + F_H2O_generated) / F_total_out
    
    DELTA_P_DEOXO_BAR = 0.05
    P_out_bar = P_in_bar - DELTA_P_DEOXO_BAR
    
    # Q_dot_fluxo_W é o calor que o fluxo de gás absorveu/ganhou.
    Q_fluxo_ganho_W = dot_n_total * CP_MIX_EST * (T_profile_K[-1] - T_in_K)
    
    W_dot_comp_W = 0.0 

    results = {
        "T_C": T_out_C,
        "P_bar": P_out_bar,
        "y_O2_out": y_O2_out,
        "y_H2O_out": y_H2O_out,
        "m_dot_gas_out_kg_s": m_dot_g_out_kg_s,
        "Agua_Gerada_kg_s": m_dot_H2O_generated_kg_s, 
        "Q_dot_fluxo_W": Q_fluxo_ganho_W, 
        "W_dot_comp_W": W_dot_comp_W,
        "X_O2": X_final,
        "T_profile_C": T_profile_C,
        "L_span": L_span,
        "T_max_calc": T_max_calc
    }
    
    return results