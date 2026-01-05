# modelo_coalescedor.py
import math
import numpy as np
import CoolProp.CoolProp as CP 
from aux_coolprop import calcular_solubilidade_gas_henry, calcular_pressao_sublimacao_gelo # Importação corrigida

# --- CONSTANTES GLOBAIS (Sistema SI) ---
BAR_TO_PA = 1e5    
PA_TO_BAR = 1e-5   
M_H2O = 0.018015   
R_UNIV = 8.31446   

# --- PARÂMETROS DE DIMENSIONAMENTO E EMPÍRICOS ---
DELTA_P_COALESCER_BAR = 0.15 
ETA_LIQ_REMOCAO = 0.98  

# Valores de referência de massa molar (apenas para cálculo de vazão volumétrica)
MM_H2 = 0.002016
MM_O2 = 0.031998

FLUXOS_DE_GAS_MODELO = {
    'H2': {'M_gas': MM_H2, 'D_shell_dim': 0.32},
    'O2': {'M_gas': MM_O2, 'D_shell_dim': 0.32}
}

# --- FUNÇÕES AUXILIARES (mantidas inalteradas) ---
def calcular_pressao_sublimacao_gelo(T_K):
    """
    Calcula a pressão de vapor/sublimação sobre GELO (em Pa) para T < 273.15 K,
    ou Pressão de Vapor (CoolProp) para T >= 273.15 K.
    """
    # 🛑 REMOVIDO FALLBACK COOLPROP
    T_C = T_K - 273.15
    if T_C < 0:
        P_Pa = 611.2 * np.exp( (22.44 * T_C) / (T_C + 272.44) ) 
    else:
        T_K_SAFE = max(T_K, 273.16)
        P_Pa = CP.PropsSI('P', 'T', T_K_SAFE, 'Q', 0, 'Water')
            
    return P_Pa


def modelar_coalescedor(gas_fluido: str, m_dot_g_kg_s: float, P_in_bar: float, T_in_C: float, y_H2O: float, m_dot_H2O_liq_in_kg_s: float) -> dict:
    """
    Modela o coalescedor. O balanço de massa considera o gás dissolvido,
    que é subtraído da vazão mássica principal (m_dot_g_kg_s).
    """
    
    T_K = T_in_C + 273.15
    P_total_Pa = P_in_bar * BAR_TO_PA
    
    # 1. PARÂMETROS DO GÁS
    M_gas_principal = FLUXOS_DE_GAS_MODELO[gas_fluido]['M_gas']
    # 🛑 REMOVIDO FALLBACK: O componente deve falhar se o gás não for mapeado

    # 2. CÁLCULO DA DENSIDADE DA MISTURA
    y_gas_princ_resto = 1.0 - y_H2O
    M_avg = (y_gas_princ_resto * M_gas_principal) + (y_H2O * M_H2O)
    
    T_K_SAFE = max(T_K, 273.16)
    # 🛑 REMOVIDO FALLBACK COOLPROP
    Z_mix = CP.PropsSI('Z', 'T', T_K, 'P', P_total_Pa, gas_fluido)
    
    rho_mix = (P_total_Pa * M_avg) / (Z_mix * R_UNIV * T_K) 
    
    # 3. CÁLCULO ENERGÉTICO
    Q_VOLUMETRICO = m_dot_g_kg_s / rho_mix # m³/s
    Delta_P_bar = DELTA_P_COALESCER_BAR
    Delta_P_Pa = Delta_P_bar * BAR_TO_PA
    W_dot_comp_W = Q_VOLUMETRICO * Delta_P_Pa
    
    # --- MODELAGEM DA REMOÇÃO DE LÍQUIDO E PERDA DE MASSA ---
    
    Agua_Liquida_Drenada_H2O_kg_s = m_dot_H2O_liq_in_kg_s * ETA_LIQ_REMOCAO
    Q_M_liq_out_kg_s = m_dot_H2O_liq_in_kg_s * (1.0 - ETA_LIQ_REMOCAO)
    P_out_bar = P_in_bar - Delta_P_bar
    
    # 4. CÁLCULO DO GÁS DISSOLVIDO (Lei de Henry)
    
    y_gas_dominante_aprox = 1.0 - y_H2O 
    solubilidade_mg_kg = calcular_solubilidade_gas_henry(gas_fluido, T_in_C, P_in_bar, y_gas_dominante_aprox)
    
    # Massa de Gás Dissolvida (H2 ou O2) na Água Drenada (kg/s)
    M_dot_Gas_Dissolvido_kg_s = Agua_Liquida_Drenada_H2O_kg_s * (solubilidade_mg_kg / 1e6)
    
    # Vazão total removida (Água líquida + Gás Dissolvido)
    Agua_Removida_Coalescer_kg_s_total = Agua_Liquida_Drenada_H2O_kg_s + M_dot_Gas_Dissolvido_kg_s
    
    # Vazão de Gás Principal de Saída (CORREÇÃO: Gás principal é subtraído da massa dissolvida)
    m_dot_gas_out_kg_s = m_dot_g_kg_s - M_dot_Gas_Dissolvido_kg_s
    
    # ----------------------------------------------------
    
    results = {
        "T_C": T_in_C, 
        "P_bar": P_out_bar,
        "y_H2O_out_vap": y_H2O, 
        "m_dot_gas_out_kg_s": m_dot_gas_out_kg_s, # Vazão PRINCIPAL corrigida
        "Agua_Removida_Coalescer_kg_s": Agua_Removida_Coalescer_kg_s_total, 
        "Agua_Liquida_Residual_out_kg_s": Q_M_liq_out_kg_s, 
        "Q_dot_fluxo_W": 0.0,
        "W_dot_comp_W": W_dot_comp_W,
        "Gas_Dissolvido_removido_kg_s": M_dot_Gas_Dissolvido_kg_s, # Gás PRINCIPAL removido (para balanço central)
        "Agua_Pura_Removida_H2O_kg_s": Agua_Liquida_Drenada_H2O_kg_s, 
    }

    return results