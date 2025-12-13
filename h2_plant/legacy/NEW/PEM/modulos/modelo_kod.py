# modelo_kod.py
import CoolProp.CoolProp as CP
import numpy as np
from aux_coolprop import calcular_solubilidade_gas_henry, calcular_pressao_sublimacao_gelo # Importação corrigida

# --- CONSTANTES DE PROCESSO E PROJETO ---
R_UNIV = 8.31446    # J/(mol*K)
RHO_L_WATER = 1000.0 # kg/m³
K_SOUDERS_BROWN = 0.08 # m/s
DIAMETRO_VASO_M = 1.0 # m
DELTA_P_BAR = 0.05 # bar
EFICIENCIA_REMOCAO = 0.97 # 97% de eficiência na remoção do condensado

def modelar_knock_out_drum(gas_fluido: str, m_dot_g_kg_s: float, P_in_bar: float, T_in_C: float, y_H2O_in: float, m_dot_H2O_liq_in_kg_s: float) -> dict:
    """
    Modela um Knock-Out Drum (KOD). O gás dissolvido é calculado e subtraído
    da vazão mássica principal (m_dot_g_kg_s).
    """
    
    T_IN_K = T_in_C + 273.15
    P_IN_PA = P_in_bar * 1e5
    
    # 1. CÁLCULO DA SATURAÇÃO
    try:
        M_H2O = CP.PropsSI('M', 'Water')
        M_H2 = CP.PropsSI('M', 'H2')
        M_O2 = CP.PropsSI('M', 'O2')
        
        P_SAT_H2O_PA = calcular_pressao_sublimacao_gelo(T_IN_K) 

        P_out_pa_ideal = P_IN_PA
        y_H2O_sat = P_SAT_H2O_PA / P_out_pa_ideal
        y_gas_sat = 1.0 - y_H2O_sat
        y_H2O_out_vap = y_H2O_sat 
        
        M_GAS_PRINCIPAL = M_H2 if gas_fluido == 'H2' else M_O2
        F_gas_molar = m_dot_g_kg_s / M_GAS_PRINCIPAL
        F_H2O_molar_in_vap = F_gas_molar * (y_H2O_in / (1.0 - y_H2O_in))
        F_H2O_molar_out_vap = F_gas_molar * (y_H2O_out_vap / y_gas_sat)
        F_H2O_molar_condensada_total = max(0.0, F_H2O_molar_in_vap - F_H2O_molar_out_vap)
        
        M_dot_H2O_liq_total_in = (F_H2O_molar_condensada_total * M_H2O) + m_dot_H2O_liq_in_kg_s 
        
        # --- CÁLCULO DE GÁS DISSOLVIDO E PERDA DE MASSA ---
        
        Agua_Liquida_Drenada_H2O_kg_s = M_dot_H2O_liq_total_in * EFICIENCIA_REMOCAO 
        
        # Fração molar do gás dominante na Fase Gás (Aproximação)
        y_gas_dominante_aprox = 1.0 - y_H2O_in 
        solubilidade_mg_kg = calcular_solubilidade_gas_henry(gas_fluido, T_in_C, P_in_bar, y_gas_dominante_aprox)
        
        # Massa de Gás Dissolvida (H2 ou O2) na Água Drenada (kg/s)
        M_dot_Gas_Dissolvido_kg_s = Agua_Liquida_Drenada_H2O_kg_s * (solubilidade_mg_kg / 1e6)
        
        # Vazão Mássica de Gás de Saída (CORREÇÃO: Gás principal é subtraído da massa dissolvida)
        m_dot_gas_out_kg_s = m_dot_g_kg_s - M_dot_Gas_Dissolvido_kg_s 
        
        # Vazão Total Removida
        Agua_Condensada_removida_kg_s = Agua_Liquida_Drenada_H2O_kg_s + M_dot_Gas_Dissolvido_kg_s 
        Agua_Liquida_Residual_kg_s = M_dot_H2O_liq_total_in * (1.0 - EFICIENCIA_REMOCAO)

        # 3. PROPRIEDADES DE SAÍDA
        M_MIX_G_out = y_gas_sat * M_GAS_PRINCIPAL + y_H2O_out_vap * M_H2O
        P_OUT_BAR = P_in_bar - DELTA_P_BAR
        P_OUT_PA = P_OUT_BAR * 1e5
        
        # Cálculo de dimensionamento
        R_UNIV_J = 8.31446 
        Z_gas = CP.PropsSI('Z', 'T', T_IN_K, 'P', P_OUT_PA, gas_fluido)
        rho_G_out = P_OUT_PA * M_MIX_G_out / (Z_gas * R_UNIV_J * T_IN_K)
        vazao_volumetrica_gas_out = m_dot_gas_out_kg_s / rho_G_out 
        
        V_max = K_SOUDERS_BROWN * np.sqrt((RHO_L_WATER - rho_G_out) / rho_G_out)
        A_vaso = np.pi * (DIAMETRO_VASO_M / 2)**2
        V_superficial_real = vazao_volumetrica_gas_out / A_vaso
        status_separacao = ("OK" if V_superficial_real < V_max else "ATENÇÃO: Vaso subdimensionado!")
        W_dot_adicional_W = vazao_volumetrica_gas_out * (DELTA_P_BAR * 1e5)
        
    except Exception as e:
        return {"erro": f"Erro no cálculo de propriedades do CoolProp para {gas_fluido}: {e}"}

    # 4. Dicionário de Saída
    results = {
        "T_C": T_in_C,
        "P_bar": P_OUT_BAR,
        "y_H2O_out_vap": y_H2O_out_vap, 
        "m_dot_gas_out_kg_s": m_dot_gas_out_kg_s, # Vazão PRINCIPAL corrigida
        "Agua_Condensada_removida_kg_s": Agua_Condensada_removida_kg_s, # Total (H2O + Gás Dissolvido)
        "Agua_Liquida_Residual_kg_s": Agua_Liquida_Residual_kg_s, 
        "Q_dot_fluxo_W": 0.0,
        "W_dot_comp_W": W_dot_adicional_W,
        "Status_KOD": status_separacao,
        "Gas_Dissolvido_removido_kg_s": M_dot_Gas_Dissolvido_kg_s, # Gás PRINCIPAL removido (para balanço central)
        "Agua_Pura_Removida_H2O_kg_s": Agua_Liquida_Drenada_H2O_kg_s, 
    }
    
    return results