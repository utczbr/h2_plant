import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt # Mantido para compatibilidade, caso decida re-incluir o teste.

# =================================================================
# DADOS DE DIMENSIONAMENTO E PARÂMETROS FIXOS
# (Valores adaptados do seu código original)
# =================================================================
V_REACTOR = 0.11        # Volume total do reator (m³)
L_M = 1.294             # Comprimento do Reator (m)
D_R_M = 0.324           # Diâmetro do Reator (m)
A_R_M2 = np.pi * (D_R_M**2) / 4.0 # Área da seção transversal (m²)

# Propriedades do Catalisador e Reação
R = 8.314               # Constante dos Gases (J/mol·K)
k0_vol = 1.0e10         # Fator pré-exponencial Volumétrico (1/s)
Ea = 55000.0            # Energia de Ativação (J/mol)
DELTA_H_RXN = -242000.0 # Entalpia de Reação (J/mol de O2 consumido) - Exotérmica

# PARÂMETROS DE CONTROLE TÉRMICO (CAMISA DE RESFRIAMENTO)
T_JACKET_C = 120.0      # Temperatura da água de resfriamento na camisa (C)
U_A = 5000.0            # Coeficiente de transferência de calor por volume de reator (W/m³·K)
T_JACKET_K = T_JACKET_C + 273.15

# Massas Molares
M_O2 = 0.031998
M_H2O = 0.018015
M_H2 = 0.002016

# Capacidade Calorífica Molar Média (J/mol·K) - Simplificada
CP_MIX_EST = 29.5 

# Alvo de pureza (5 ppm de O2)
Y_O2_OUT_TARGET = 5.0e-6

# =================================================================
# FUNÇÃO DE MODELAGEM CENTRAL
# =================================================================

def modelar_deoxo(m_dot_g_kg_s: float, P_in_bar: float, T_in_C: float, y_H2O_in: float, y_O2_in: float) -> dict:
    """
    Modela o reator Deoxo (PFR com resfriamento) para remover O2 do fluxo de H2.
    """
    
    if m_dot_g_kg_s == 0:
        # Retorno padrão para quando o fluxo é zero (como no O2, onde é zero)
        return {"T_C": T_in_C, "P_bar": P_in_bar, "y_O2_out": y_O2_in, "Q_dot_fluxo_W": 0.0, "W_dot_comp_W": 0.0, "X_O2": 0.0, "T_profile_C": np.array([T_in_C]), "L_span": np.array([0.0])}
    
    T_in_K = T_in_C + 273.15
    P_in_Pa = P_in_bar * 100000.0 

    # Vazão Molar de Entrada (mol/s)
    # Assumindo que o gás é H2
    M_GAS = M_H2 
    F_gas_molar = m_dot_g_kg_s / M_GAS
    
    # Cálculo da Vazão Molar Total e de O2 na entrada
    F_H2O_molar_in = F_gas_molar * (y_H2O_in / (1.0 - y_H2O_in))
    # dot_n_total é a vazão molar H2 + H2O. O O2 é tratado como impureza
    dot_n_total = F_gas_molar + F_H2O_molar_in
    
    F_O2_in = dot_n_total * y_O2_in

    # Conversão alvo
    conversion_target = 1.0 - (Y_O2_OUT_TARGET / y_O2_in)
    
    # --- DEFINIÇÃO DAS EQUAÇÕES DIFERENCIAIS (Balanco de Massa e Energia PFR) ---

    def pfr_ode(L, P):
        """
        Sistema de Equações Diferenciais Ordinárias (ODEs) para o PFR.
        L: Posição axial (m)
        P[0]: X (Conversão de O2)
        P[1]: T (Temperatura em K)
        """
        X = P[0]
        T = P[1]
        
        current_X = np.clip(X, 0.0, 1.0) 
        
        # Parada de Reação ou Conversão Máxima
        if current_X >= conversion_target or current_X >= 1.0:
            r_O2 = 0.0
            dX_dL = 0.0
        else:
            # Taxa de Reação (r_O2): mol de O2 / m³ de reator / s
            k_eff_prime = k0_vol * np.exp(-Ea / (R * T))
            
            # Concentração de O2
            Y_O2 = y_O2_in * (1.0 - current_X)
            C_O2 = P_in_Pa * Y_O2 / (R * T)
            
            r_O2 = k_eff_prime * C_O2 

            # Balanço de Massa (dX/dL)
            dX_dL = (A_R_M2 / F_O2_in) * r_O2
            
        # Balanço de Energia (dT/dL)
        heat_generated = (-DELTA_H_RXN) * r_O2 # W/m³
        heat_removed = U_A * (T - T_JACKET_K) # W/m³
        
        # Evita que a temperatura caia abaixo de T_jacket se não houver reação
        if T < T_JACKET_K and heat_generated < 1e-6:
            heat_removed = 0.0
        
        # dT/dL = (Area / (n_total * Cp_mix)) * (Geração - Remoção)
        dT_dL = (A_R_M2 / (dot_n_total * CP_MIX_EST)) * (heat_generated - heat_removed)
        
        return [dX_dL, dT_dL]

    # --- SOLUÇÃO DO MODELO ---
    
    # Condições Iniciais: X=0, T=T_in_K
    X0 = 0.0
    T0 = T_in_K
    L_span = np.linspace(0, L_M, 100)
    
    # Solução do sistema de ODEs
    sol = solve_ivp(pfr_ode, [0, L_M], [X0, T0], t_eval=L_span)

    # Variáveis de Perfil 
    X_profile = np.clip(sol.y[0], 0.0, 1.0)
    T_profile_K = sol.y[1]
    T_profile_C = T_profile_K - 273.15
    
    X_final = X_profile[-1]
    T_out_K = T_profile_K[-1]
    T_out_C = T_profile_C[-1]
    
    # --- CÁLCULO DE ENERGIA E PROPRIEDADES DE SAÍDA ---
    
    # 1. Vazão Molar de O2 Consumido
    F_O2_consumed = F_O2_in * X_final 
    
    # 2. Vazão Molar de H2O Gerada (Reação: O2 + 2H2 -> 2H2O)
    F_H2O_generated = F_O2_consumed * 2.0 
    
    # 3. Vazão Mássica de Gás (H2) de Saída
    # Reação consome 2 moles de H2 por 1 mol de O2. (H2 é o gás principal)
    # Vazão Mássica de H2 consumida (kg/s)
    m_dot_H2_consumed = F_O2_consumed * 2.0 * M_H2
    m_dot_g_out_kg_s = m_dot_g_kg_s - m_dot_H2_consumed
    
    # 4. Vazão Mássica de H2O gerada (adicionada ao vapor)
    m_dot_H2O_generated_kg_s = F_H2O_generated * M_H2O
    
    # 5. Nova Fração Molar de O2 de Saída
    y_O2_out = y_O2_in * (1.0 - X_final)
    
    # 6. Nova Fração Molar de H2O de Saída (y_H2O aumenta devido à reação)
    F_total_out = dot_n_total - F_O2_consumed + F_H2O_generated # Total - O2 + H2O
    y_H2O_out = (F_H2O_molar_in + F_H2O_generated) / F_total_out
    
    # 7. Pressão de Saída (Queda de Pressão Estimada)
    DELTA_P_DEOXO_BAR = 0.05
    P_out_bar = P_in_bar - DELTA_P_DEOXO_BAR
    
    # 8. Carga Térmica (Calor Removido do Reator)
    # Q_removed = Fluxo de Energia no Fluxo (J/s)
    # Usamos a diferença entálpica.
    Q_removed_W = (dot_n_total * CP_MIX_EST * (T_in_K - T_out_K)) + (F_O2_consumed * DELTA_H_RXN)
    
    # Energia Elétrica (W) - Consideramos zero para o reator.
    W_dot_comp_W = 0.0 

    results = {
        "T_C": T_out_C,
        "P_bar": P_out_bar,
        "y_O2_out": y_O2_out,
        "y_H2O_out": y_H2O_out,
        "m_dot_gas_out_kg_s": m_dot_g_out_kg_s,
        "Agua_Gerada_kg_s": m_dot_H2O_generated_kg_s, # Água adicionada ao vapor
        "Q_dot_fluxo_W": Q_removed_W, # Q transferido do reator
        "W_dot_comp_W": W_dot_comp_W,
        "X_O2": X_final,
        "T_profile_C": T_profile_C, # Incluído o perfil
        "L_span": L_span # Incluído o span de comprimento
    }
    
    return results