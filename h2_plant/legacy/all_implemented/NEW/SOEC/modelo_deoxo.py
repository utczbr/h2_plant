# modelo_deoxo.py (Versão Final Adiabática, Focada no Pico de T e Calor Gerado)
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import CoolProp.CoolProp as CP 
import pandas as pd # Adicionado para melhor retorno de histórico

# =================================================================
# DADOS DE DIMENSIONAMENTO E PARÂMETROS FIXOS (Do modelo anterior)
# =================================================================
# 💥 NOVO DIMENSIONAMENTO:
V_REACTOR = 0.26169     # Volume total do reator (m³) - NOVO VALOR
D_R_M = 0.437           # Diâmetro do Reator (m) - NOVO VALOR
# L_M (Comprimento) será usado como L_M_input na função.
R = 8.314               # Constante dos Gases (J/mol·K)
k0_vol = 1.0e10         # Fator pré-exponencial Volumétrico (1/s)
Ea = 55000.0            # Energia de Ativação (J/mol)
DELTA_H_RXN = -242000.0 # Entalpia de Reação (J/mol de O2 consumido) - Exotérmica

# Massas Molares
M_O2 = 0.031998
M_H2O = 0.018015
M_H2 = 0.002016

# Capacidade Calorífica Molar Média (J/mol·K) - Simplificada
CP_MIX_EST = 29.5 

# Alvo de pureza (5 ppm de O2)
Y_O2_OUT_TARGET = 5.0e-6

# 🛑 REMOVIDO: Temperatura de Reação Típica (Não é mais forçada)
# T_REATOR_OPERACAO_C = 150.0 

# =================================================================
# FUNÇÃO DE MODELAGEM CENTRAL (MODELO ADIABÁTICO)
# =================================================================

def modelar_deoxo(m_dot_g_kg_s: float, P_in_bar: float, T_in_C: float, y_H2O_in: float, y_O2_in: float, L_M_input: float) -> dict:
    """
    Modela o reator Deoxo ADIABATICAMENTE, resolvendo as ODEs de Massa e Energia.
    Agora usa a T_in_C real para a cinética.
    """
    
    L_M = L_M_input # Usa o comprimento fornecido (1.747 m)
    A_R_M2 = np.pi * (D_R_M**2) / 4.0 
    
    # 1. Verificação de Condições Inválidas/Bypass
    if m_dot_g_kg_s < 1e-6 or y_O2_in < 1e-12:
        return {"T_C": T_in_C, "P_bar": P_in_bar, "y_O2_out": y_O2_in, "Q_dot_fluxo_W": 0.0, "W_dot_comp_W": 0.0, "X_O2": 0.0, "T_profile_C": np.array([T_in_C]), "L_span": np.array([0.0]), "T_max_calc": T_in_C, "Agua_Gerada_kg_s": 0.0, "y_H2O_out": y_H2O_in, "m_dot_gas_out_kg_s": m_dot_g_kg_s}

    # 🛑 AJUSTE CRÍTICO: Usa a T_in_C real para iniciar a cinética (T_START_REACTION_C)
    T_START_REACTION_C = T_in_C # Agora é 40.00 °C, conforme a simulação.
    T_in_K = T_START_REACTION_C + 273.15
    P_in_Pa = P_in_bar * 100000.0 

    # 2. Cálculo da Vazão Molar Total
    M_GAS = M_H2 
    F_H2_molar = m_dot_g_kg_s / M_GAS
    y_H2_approx = 1.0 - y_H2O_in - y_O2_in
    F_O2_molar_in = F_H2_molar * (y_O2_in / y_H2_approx) if y_H2_approx > 0 else 0
    F_H2O_molar_in = F_H2_molar * (y_H2O_in / y_H2_approx) if y_H2_approx > 0 else 0
    dot_n_total = F_H2_molar + F_O2_molar_in + F_H2O_molar_in
    F_O2_in = F_O2_molar_in
    
    conversion_target = 1.0 - (Y_O2_OUT_TARGET / y_O2_in)
    conversion_target = np.clip(conversion_target, 0.0, 1.0)
    
    # --- DEFINIÇÃO DAS EQUAÇÕES DIFERENCIAIS (ADIABÁTICAS) ---

    def pfr_ode_adiabatico(L, P):
        X = P[0] # Conversão de O2
        T = P[1] # Temperatura em K
        current_X = np.clip(X, 0.0, 1.0) 
        
        # 1. Balanço de Massa (dX/dL)
        if current_X >= conversion_target or current_X >= 1.0:
            r_O2 = 0.0
            dX_dL = 0.0
        else:
            k_eff_prime = k0_vol * np.exp(-Ea / (R * T))
            Y_O2 = y_O2_in * (1.0 - current_X)
            C_O2 = P_in_Pa * Y_O2 / (R * T) # Concentração molar (mol/m³)
            # A lei de taxa é muito sensível. Esta é a versão simplificada do modelo.
            r_O2 = k_eff_prime * C_O2 * (1.0 + Y_O2)**(-1) 
            dX_dL = (A_R_M2 / F_O2_in) * r_O2
            
        # 2. Balanço de Energia (dT/dL) - ADIABÁTICO!
        heat_generated_W_m = A_R_M2 * (-DELTA_H_RXN) * r_O2 # Calor gerado pela reação (W/m)
        heat_removed_W_m = 0.0 # <--- MANTIDO ZERO PARA ADIABÁTICO
        
        # dT/dL = [Calor Gerado - Calor Removido] / [dot_n_total * CP_MIX_EST]
        # O Delta H de reação é a principal fonte do pico de T.
        dT_dL = (heat_generated_W_m - heat_removed_W_m) / (dot_n_total * CP_MIX_EST)
        
        return [dX_dL, dT_dL]

    # --- SOLUÇÃO DO MODELO ---
    X0 = 0.0
    T0 = T_in_K # Usando a T ajustada para a cinética
    L_span = np.linspace(0, L_M, 100) 
    
    sol = solve_ivp(pfr_ode_adiabatico, [0, L_M], [X0, T0], t_eval=L_span, method='RK45')
    
    X_profile = np.clip(sol.y[0], 0.0, 1.0)
    T_profile_K = sol.y[1]
    T_profile_C = T_profile_K - 273.15
    
    X_final = X_profile[-1]
    T_out_K = T_profile_K[-1]
    T_out_C_final = T_out_K - 273.15
    T_max_calc = np.max(T_profile_C) 


    # --- CÁLCULO DE PROPRIEDADES DE SAÍDA ---
    
    F_O2_consumed = F_O2_in * X_final 
    F_H2O_generated = F_O2_consumed * 2.0 
    
    m_dot_H2_consumed = F_O2_consumed * 2.0 * M_H2 # H2 consumido
    m_dot_g_out_kg_s = m_dot_g_kg_s - m_dot_H2_consumed
    m_dot_H2O_generated_kg_s = F_H2O_generated * M_H2O # Água gerada
    
    y_O2_out = y_O2_in * (1.0 - X_final)
    F_total_out = dot_n_total - F_O2_consumed + F_H2O_generated
    y_H2O_out = (F_H2O_molar_in + F_H2O_generated) / F_total_out
    
    DELTA_P_DEOXO_BAR = 0.05
    P_out_bar = P_in_bar - DELTA_P_DEOXO_BAR
    
    # Q_dot_fluxo_W é o calor total LIBERADO pela reação (que DEVE ser removido)
    Q_liberado_W = F_O2_consumed * (-DELTA_H_RXN)
    
    W_dot_comp_W = 0.0 

    results = {
        # Mantendo T_C = T_in_C para simular o resfriamento pós-reator
        "T_C": T_in_C, 
        "T_peak_C": T_max_calc, 
        "P_bar": P_out_bar,
        "y_O2_out": y_O2_out,
        "y_H2O_out": y_H2O_out,
        "m_dot_gas_out_kg_s": m_dot_g_out_kg_s,
        "Agua_Gerada_kg_s": m_dot_H2O_generated_kg_s, 
        "Q_dot_fluxo_W": -Q_liberado_W, 
        "W_dot_comp_W": W_dot_comp_W,
        "X_O2": X_final,
        "T_profile_C": T_profile_C,
        "L_span": L_span,
        "T_max_calc": T_max_calc,
        "m_dot_O2_in_kg_s": F_O2_in * M_O2 
    }
    
    return results