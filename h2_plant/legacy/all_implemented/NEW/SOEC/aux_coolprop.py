# aux_coolprop.py (Versão Estável e Corrigida)
# Funções Auxiliares de Cálculo de Propriedades Termodinâmicas (CoolProp) e Alertas
# Importado pelo main_simulator.py

import numpy as np
import pandas as pd
import CoolProp.CoolProp as CP
import sys
# Importa as constantes globais do módulo dedicado
from constants_and_config import LIMITES, Y_H2O_LIMIT_MOLAR, P_IN_BAR, T_IN_C 

# --- CONSTANTES GLOBAIS DE SOLUBILIDADE (Lei de Henry) ---
R_UNIV = 8.31446 # J/(mol*K)
R_IDEAL_ATM_L_MOL_K = 0.082057 # L*atm / (mol*K)
BAR_TO_ATM = 1.01325

# Constantes de Henry (H_i(298K) em L*atm/mol) e Relação Entalpica (-Delta_solH/R em K)
CONST_HENRY = {
    'H2': {
        'H_298_L_atm_mol': 1300.0, 
        'Delta_H_R_K': 500.0, 
        'M_kg_mol': 0.002016 
    }, 
    'O2': {
        'H_298_L_atm_mol': 770.0, 
        'Delta_H_R_K': 1700.0, 
        'M_kg_mol': 0.031998 
    }
}
# -------------------------------------------------------------------------


# =================================================================
# === FUNÇÕES AUXILIARES DE CÁLCULO DE PROPRIEDADES (COOLPROP) ===
# =================================================================

def calcular_solubilidade_gas_henry(gas_fluido: str, T_C: float, P_bar: float, y_gas_dominante: float) -> float:
    """
    Estima a solubilidade mássica (mg/kg H2O) de H2 ou O2 em água nas condições T e P.
    """
    T_K = T_C + 273.15
    
    if gas_fluido not in CONST_HENRY:
        return 0.0
        
    consts = CONST_HENRY[gas_fluido]
    
    # 1. Ajuste da Constante de Henry para a Temperatura (Equação van't Hoff simplificada)
    T0 = 298.15 # K (25 °C)
    
    H_298 = consts['H_298_L_atm_mol']
    Delta_H_R = consts['Delta_H_R_K']
    
    H_T = H_298 * np.exp( Delta_H_R * (1.0/T_K - 1.0/T0) )
    
    # 2. Cálculo da Pressão Parcial e Solubilidade (Lei de Henry)
    P_total_atm = P_bar * BAR_TO_ATM
    P_parcial_gas_atm = P_total_atm * y_gas_dominante
    
    c_mol_L = P_parcial_gas_atm / H_T
    
    # 3. Conversão para Solubilidade Mássica (mg/kg H2O)
    M_g_mol = consts['M_kg_mol'] * 1000.0 # g/mol
    c_mg_L = c_mol_L * M_g_mol * 1000.0 # mg/L
    solubilidade_mg_kg = c_mg_L 
    
    return solubilidade_mg_kg

def calcular_pressao_sublimacao_gelo(T_K: float) -> float:
    """Calcula a pressão de vapor/sublimação sobre GELO (em Pa) para T < 273.15 K."""
    T_C = T_K - 273.15
    
    if T_C < 0:
        # P sobre gelo (sublimação)
        P_Pa = 611.2 * np.exp( (22.44 * T_C) / (T_C + 272.44) ) 
    else:
        # P sobre água líquida (CoolProp) para T >= 0 °C
        T_K_SAFE = max(T_K, 273.16)
        P_Pa = CP.PropsSI('P', 'T', T_K_SAFE, 'Q', 0, 'Water') 
            
    return P_Pa

def verificar_limites_operacionais(componente: str, estado_in: dict, T_max_calc: float = None):
    """Verifica e imprime alertas se T ou y_O2/y_H2O excederem os limites."""
    T_C = estado_in['T_C']
    P_bar = estado_in['P_bar']
    
    if componente in LIMITES:
        limites_comp = LIMITES[componente]
        
        # 1. Alerta de Temperatura Máxima
        T_MAX = limites_comp.get('T_MAX_C', np.inf)
        T_CHECK = T_max_calc if T_max_calc is not None else T_C
        if T_CHECK > T_MAX:
             print(f"[ALERTA TÉRMICO - {componente}] Temperatura de {T_CHECK:.2f} °C excede o limite máximo de {T_MAX:.2f} °C.")
        
        # 2. Alerta de Oxigênio (para fluxo H2)
        if 'y_O2_MAX' in limites_comp and estado_in.get('gas_fluido') == 'H2':
            y_O2_MAX = limites_comp['y_O2_MAX']
            y_O2_atual = estado_in.get('y_O2', 0.0)
            if y_O2_atual > y_O2_MAX:
                print(f"[ALERTA SEGURANÇA - {componente}] y_O2 de {y_O2_atual:.2e} excede o limite de segurança de {y_O2_MAX:.2e} (Risco de explosão).")
        
        # 3. Alerta de Umidade (H2O)
        if 'y_H2O_MAX_PPM' in limites_comp:
            y_H2O_MAX = limites_comp['y_H2O_MAX_PPM'] * 1e-6
            y_H2O_atual = estado_in.get('y_H2O', 0.0)
            if y_H2O_atual > y_H2O_MAX:
                 print(f"[ALERTA QUALIDADE - {componente}] y_H2O de {y_H2O_atual*1e6:.2f} ppm excede o limite de qualidade de {limites_comp['y_H2O_MAX_PPM']:.2f} ppm.")


def calcular_estado_termodinamico(gas_fluido: str, T_C: float, P_bar: float, m_dot_gas_princ_kg_s: float, y_H2O_alvo: float, y_O2: float = 0.0, y_H2: float = 0.0) -> dict:
    """
    Calcula propriedades termodinâmicas da mistura (gás principal + impurezas) usando CoolProp.
    (Versão restaurada, baseada apenas em T e P)
    """
    T_K = T_C + 273.15
    P_PA = P_bar * 1e5
    
    # Constantes de Massa Molar
    M_H2O = CP.PropsSI('M', 'Water')
    M_H2 = CP.PropsSI('M', 'H2')
    M_O2 = CP.PropsSI('M', 'O2')
    
    # 💥 SOEC ENTRADA (LÍQUIDO RECIRCULADO)
    if m_dot_gas_princ_kg_s < 1e-10:
        
        from constants_and_config import M_DOT_H2O_RECIRC_TOTAL_KGS
        m_dot_total_in = M_DOT_H2O_RECIRC_TOTAL_KGS
        
        T_K_SAFE = max(T_K, 273.16)
        H_H2O_liq = CP.PropsSI('H', 'T', T_K_SAFE, 'Q', 0, 'Water') 
             
        m_dot_gas_princ_kg_s = 0.0
        m_dot_mix_kg_s = m_dot_total_in
        F_H2O_molar = m_dot_mix_kg_s / M_H2O
        
        H_mix_J_kg = H_H2O_liq 
        w_H2O_final = 1.0 # 100% Água
        
        return {
            "T_C": T_C,
            "P_bar": P_bar,
            "y_H2O": 1.0, 
            "w_H2O": w_H2O_final, 
            "y_O2": 0.0, 
            "y_H2": 0.0, 
            "m_dot_gas_kg_s": m_dot_gas_princ_kg_s,
            "m_dot_mix_kg_s": m_dot_mix_kg_s, 
            "m_dot_H2O_vap_kg_s": 0.0,
            "H_mix_J_kg": H_mix_J_kg,
            "P_H2O_bar": P_bar,
            "y_H2O_sat": 1.0,
            "Estado_H2O": "Água Líquida (SOEC Entrada) - H Simplificado", 
            "F_molar_total": F_H2O_molar,
            "H_in_mix_J_kg": H_mix_J_kg * m_dot_mix_kg_s, # Entalpia total (J/s)
            "gas_fluido": gas_fluido
        }
    # --------------------------------------------------------------
    
    
    # 1. Vazões Molares e CÁLCULO DO VAPOR REAL DISPONÍVEL
    
    # 1.1. Cálculo da Fração Molar MÁXIMA (Saturação)
    y_H2O_sat = calcular_pressao_sublimacao_gelo(T_K) / P_PA # Usando a função auxiliar
    
    # 1.2. Define a fração molar real de H2O (y_H2O_final)
    y_H2O_final = min(y_H2O_sat, y_H2O_alvo)
    
    # Se y_H2O_final for muito próximo de 1.0 (saturação em P muito baixa), forçamos um valor seguro.
    if y_H2O_final >= 1.0:
        y_H2O_final = 1.0 - 1e-6
        
    # 1.3. Frações Molares Finais (Base Total)
    y_gas_principal_e_impureza = 1.0 - y_H2O_final
    
    if gas_fluido == 'H2':
        y_O2_final = y_O2 * y_gas_principal_e_impureza
        y_H2_final = 1.0 - y_H2O_final - y_O2_final
    else: # O2
        y_H2_final = y_H2 * y_gas_principal_e_impureza
        y_O2_final = 1.0 - y_H2O_final - y_H2_final
        
    # 1.4. Vazões Molares (Baseado na Vazão Mássica do Gás Principal)
    
    if gas_fluido == 'H2':
        M_PRINCIPAL = M_H2
        F_H2_molar = m_dot_gas_princ_kg_s / M_PRINCIPAL # Vazão molar principal (H2)
        F_molar_total_calc = F_H2_molar / y_H2_final if y_H2_final > 0 else 0 
    else: # O2
        M_PRINCIPAL = M_O2
        F_O2_molar = m_dot_gas_princ_kg_s / M_PRINCIPAL # Vazão molar principal (O2)
        F_molar_total_calc = F_O2_molar / y_O2_final if y_O2_final > 0 else 0
        
    # Vazões molares finais (mantendo a proporção de y_H2O_final)
    F_molar_total = F_molar_total_calc 
    F_H2O_molar = F_molar_total * y_H2O_final
    F_H2_molar = F_molar_total * y_H2_final
    F_O2_molar = F_molar_total * y_O2_final


    # 1.5. Vazão Mássica Total da Mistura
    m_dot_mix_kg_s = F_H2_molar * M_H2 + F_O2_molar * M_O2 + F_H2O_molar * M_H2O

    
    # 2. Entalpia e Pressões
    
    # 2.1. Cálculo da Entalpia do H2O (Vapor)
    P_H2O_PA_parcial = P_PA * y_H2O_final
    T_K_SAFE = max(T_K, 273.16) 
    
    # ===========================================================
    # BLOCO FINAL CORRIGIDO PARA CÁLCULO DE PROPRIEDADES DE VAPOR
    # ===========================================================

    P_MIN_CLAMP_PA = 1e-3     # pressao mínima aceitável
    REL_TOL_SAT = 1e-4        # tolerância para detectar igualdade com saturação
    H_H2O_vap = None

    # Caso não haja água no gás → y_H2O_alvo == 0
    if y_H2O_alvo <= 0.0 or P_H2O_PA_parcial <= 0.0:
        # Não existe vapor de água na mistura
        m_dot_H2O_vap_kg_s = 0.0
        y_H2O_sat = 0.0
    else:
        # Tenta obter Psat para comparação
        try:
            P_sat_at_T = calcular_pressao_sublimacao_gelo(T_K_SAFE) # Usa a função auxiliar
        except Exception:
            P_sat_at_T = None

        # Se Psat foi calculado, checar se P_H2O_calc ≈ Psat
        if P_sat_at_T is not None:
            if abs(P_H2O_PA_parcial - P_sat_at_T) / max(P_sat_at_T, 1.0) < REL_TOL_SAT:
                # Força valor ligeiramente abaixo da saturação
                P_H2O_PA_parcial = min(P_H2O_PA_parcial, P_sat_at_T * (1.0 - 1e-6))

        # Evitar P=0
        P_used = max(P_H2O_PA_parcial, P_MIN_CLAMP_PA)

        # Tentativa de calcular H do vapor de água
        try:
            H_H2O_vap = CP.PropsSI('H', 'T', T_K_SAFE, 'P', P_used, 'Water')
        except Exception as e:
            print(f"[AVISO] Falha no cálculo de H_vapor com P={P_used:.6e} Pa. Erro: {e}")
            H_H2O_vap = None

        # Fração molar saturada de água
        if P_sat_at_T is not None and P_sat_at_T > 0:
            y_H2O_sat = min(1.0, P_sat_at_T / P_PA)
        else:
            y_H2O_sat = y_H2O_alvo
        
        m_dot_H2O_vap_kg_s = F_H2O_molar * M_H2O
    # ===========================================================


    
    # 2.2. Entalpia dos Gases Principais (H2 e O2)
    P_MIN_PA = 1.0 

    P_H2_PA_parcial = P_PA * y_H2_final
    P_H2_calc = max(P_H2_PA_parcial, P_MIN_PA)
    H_H2 = CP.PropsSI('H', 'T', T_K, 'P', P_H2_calc, 'H2') 

    P_O2_PA_parcial = P_PA * y_O2_final
    P_O2_calc = max(P_O2_PA_parcial, P_MIN_PA)
    H_O2 = CP.PropsSI('H', 'T', T_K, 'P', P_O2_calc, 'O2') 

    # 2.3. Entalpia Mássica da Mistura (J/kg_mix)
    
    H_mix_sum_J_s = F_H2_molar * M_H2 * H_H2 + F_O2_molar * M_O2 * H_O2
    m_dot_mix_gases_kg_s = F_H2_molar * M_H2 + F_O2_molar * M_O2
    
    # 🛑 PROTEÇÃO DE VAPOR: Apenas adiciona se H_H2O_vap foi calculado com sucesso (não é None)
    if H_H2O_vap is not None:
         H_mix_sum_J_s += m_dot_H2O_vap_kg_s * H_H2O_vap
         m_dot_mix_gases_kg_s += m_dot_H2O_vap_kg_s
    else:
         # Se não houver vapor, a vazão da mistura deve ser ajustada para refletir apenas os gases
         m_dot_mix_kg_s = m_dot_mix_gases_kg_s
         m_dot_H2O_vap_kg_s = 0.0 # Reinicializa para zero no retorno

    # Recalcula H_mix_J_kg com base na massa de gases + vapor (se houver)
    if m_dot_mix_gases_kg_s > 0:
        H_mix_J_kg = H_mix_sum_J_s / m_dot_mix_gases_kg_s
    else:
        H_mix_J_kg = 0.0
        
    
    # 3. Pressões Parciais
    P_H2O_parcial_bar = P_PA * y_H2O_final / 1e5
    
    # 4. Estado de Saturação (para exibição)
    estado_saturacao = "Saturado" if y_H2O_final >= y_H2O_sat else "Vapor Superaquecido"
    
    # 5. CÁLCULO DA FRAÇÃO MÁSSICA (w_H2O)
    w_H2O_final = (m_dot_H2O_vap_kg_s) / m_dot_mix_kg_s if m_dot_mix_kg_s > 0 else 0.0
    
    return {
        "T_C": T_C,
        "P_bar": P_bar,
        "y_H2O": y_H2O_final,
        "w_H2O": w_H2O_final, # Fração Mássica
        "y_O2": y_O2_final, 
        "y_H2": y_H2_final, 
        "m_dot_gas_kg_s": m_dot_gas_princ_kg_s, # Vazão mássica principal rastreada
        "m_dot_mix_kg_s": m_dot_mix_kg_s, # Vazão mássica total da mistura
        "m_dot_H2O_vap_kg_s": m_dot_H2O_vap_kg_s, # Vazão mássica de vapor de água
        "H_mix_J_kg": H_mix_J_kg,
        "P_H2O_bar": P_H2O_parcial_bar,
        "y_H2O_sat": y_H2O_sat,
        "Estado_H2O": estado_saturacao,
        "F_molar_total": F_molar_total,
        "H_in_mix_J_kg": H_mix_J_kg * m_dot_mix_kg_s, # Entalpia total (J/s) da mistura
        "gas_fluido": gas_fluido # Gás fluído para alertas
    }

# 🌟 FUNÇÃO REINSERIDA PARA RESOLVER O IMPORTERROR
def calcular_y_H2O_inicial(T_C, P_bar) -> float:
    """Calcula a fração molar inicial de água (y_H2O) assumindo saturação total."""
    T_K = T_C + 273.15
    P_PA = P_bar * 1e5
    P_sat_H2O_PA = calcular_pressao_sublimacao_gelo(T_K)
    return P_sat_H2O_PA / P_PA