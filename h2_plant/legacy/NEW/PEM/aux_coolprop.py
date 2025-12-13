# aux_coolprop.py (Completo e Corrigido)
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
# ... (Função inalterada) ...
    """
    Estima a solubilidade mássica (mg/kg H2O) de H2 ou O2 em água nas condições T e P.
    Assume que 1 kg H2O = 1 L H2O (densidade ~1000 kg/m3).
    
    Args:
        gas_fluido (str): 'H2' ou 'O2'.
        T_C (float): Temperatura (°C).
        P_bar (float): Pressão Total (bar).
        y_gas_dominante (float): Fração molar do gás principal (H2 ou O2) na fase gás.
        
    Returns:
        float: Solubilidade mássica de gás dissolvido (mg gás / kg H2O).
    """
    T_K = T_C + 273.15
    
    if gas_fluido not in CONST_HENRY:
        return 0.0
        
    consts = CONST_HENRY[gas_fluido]
    
    # 1. Ajuste da Constante de Henry para a Temperatura (Equação van't Hoff simplificada)
    # H_i(T) = H_i(T0) * exp[ Delta_H_R_K * (1/T - 1/T0) ]
    T0 = 298.15 # K (25 °C)
    
    H_298 = consts['H_298_L_atm_mol']
    Delta_H_R = consts['Delta_H_R_K']
    
    H_T = H_298 * np.exp( Delta_H_R * (1.0/T_K - 1.0/T0) )
    
    # 2. Cálculo da Pressão Parcial e Solubilidade (Lei de Henry)
    P_total_atm = P_bar * BAR_TO_ATM
    P_parcial_gas_atm = P_total_atm * y_gas_dominante
    
    # c_i (mol/L) = P_i / H_i(T)
    c_mol_L = P_parcial_gas_atm / H_T
    
    # 3. Conversão para Solubilidade Mássica (mg/kg H2O)
    # Vazão molar de gás (mol/L) * Massa Molar (g/mol) * (1000 mg/g) / (1 kg/L)
    # Assumindo rho_H2O = 1 kg/L
    M_g_mol = consts['M_kg_mol'] * 1000.0 # g/mol
    
    c_mg_L = c_mol_L * M_g_mol * 1000.0 # mg/L
    
    # Solubilidade Mássica (mg/kg)
    solubilidade_mg_kg = c_mg_L 
    
    return solubilidade_mg_kg

def calcular_pressao_sublimacao_gelo(T_K: float) -> float:
# ... (Função inalterada) ...
    """Calcula a pressão de vapor/sublimação sobre GELO (em Pa) para T < 273.15 K."""
    T_C = T_K - 273.15
    
    if T_C < 0:
        # P sobre gelo (sublimação)
        P_Pa = 611.2 * np.exp( (22.44 * T_C) / (T_C + 272.44) ) 
    else:
        # P sobre água líquida (CoolProp) para T >= 0 °C
        T_K_SAFE = max(T_K, 273.16)
        try:
            P_Pa = CP.PropsSI('P', 'T', T_K_SAFE, 'Q', 0, 'Water') 
        except ValueError:
            # Fallback
            P_Pa = 611.2 * np.exp( (17.62 * T_C) / (T_C + 243.5) ) 
            
    return P_Pa

def verificar_limites_operacionais(componente: str, estado_in: dict, T_max_calc: float = None):
# ... (Função inalterada) ...
    """
    Verifica se os parâmetros de entrada excedem os limites seguros para o componente
    e dispara alertas operacionais.
    """
    if componente in LIMITES:
        limites = LIMITES[componente]
        T_C = T_max_calc if T_max_calc is not None else estado_in.get('T_C') # Usa T_max_calc (pico do Deoxo) se fornecido
        y_O2 = estado_in.get('y_O2', 0)
        y_H2O = estado_in.get('y_H2O', 0)
        
        gas_fluido = estado_in.get('gas_fluido', 'N/A')
        
        alertas = []
        
        # Alerta de Temperatura
        if T_C > limites.get('T_MAX_C', 999):
            alertas.append(f"T ({T_C:.2f}°C) > LIMITE ({limites['T_MAX_C']:.1f}°C)")
            
        # Alerta de O2 (apenas para Deoxo)
        if componente == 'Deoxo' and y_O2 > limites.get('y_O2_MAX', 999):
            alertas.append(f"y_O2 ({(y_O2*100):.2f}%) > LIMITE ({(limites['y_O2_MAX']*100):.2f}%)")
            
        # Alerta de H2O (para Secadores Adsorventes, PSA e VSA)
        if (componente in ['PSA', 'Secador Adsorvente', 'VSA']) and y_H2O * 1e6 > limites.get('y_H2O_MAX_PPM', 9999):
            alertas.append(f"y_H2O ({(y_H2O*1e6):.1f} ppm) > LIMITE ({(limites['y_H2O_MAX_PPM']):.1f} ppm)")
        
        if alertas:
            print(f"!!! ALERTA OPERACIONAL para {componente} ({gas_fluido}) !!!")
            for alerta in alertas:
                print(f"    - {alerta}")
            print("!!! Continuação da Modelagem... !!!")


# 🌟 FUNÇÃO MODIFICADA PARA AMARRAR O VAPOR AO ESTOQUE LÍQUIDO DISPONÍVEL
def calcular_estado_termodinamico(gas_fluido: str, T_C: float, P_bar: float, m_dot_gas_princ_kg_s: float, y_H2O_alvo: float, y_O2: float = 0.0, y_H2: float = 0.0) -> dict:
    """
    Calcula propriedades termodinâmicas da mistura (gás principal + impurezas) usando CoolProp.
    
    A vazão mássica de entrada é APENAS a do gás principal (H2 ou O2), sem impurezas ou vapor d'água.
    
    Args:
        m_dot_gas_princ_kg_s (float): Vazão Mássica do Gás Principal PURO (H2 ou O2).
        y_H2O_alvo (float): Fração molar ALVO de H2O (usado como limite de saturação).
    """
    T_K = T_C + 273.15
    P_PA = P_bar * 1e5
    
    # Constantes de Massa Molar
    M_H2O = CP.PropsSI('M', 'Water')
    M_H2 = CP.PropsSI('M', 'H2')
    M_O2 = CP.PropsSI('M', 'O2')

    # 1. Vazões Molares e CÁLCULO DO VAPOR REAL DISPONÍVEL
    
    # 1.1. Cálculo da Fração Molar MÁXIMA (Saturação)
    y_H2O_sat = calcular_pressao_sublimacao_gelo(T_K) / P_PA # Usando a função auxiliar
    
    # 1.2. Define a fração molar real de H2O (y_H2O_final)
    # y_H2O_final é o mínimo entre a saturação e o valor alvo (proveniente do cálculo anterior ou input).
    # O valor 'y_H2O_alvo' agora representa o LIMITE IMPOSTO PELO ESTOQUE/ALVO DE PUREZA.
    y_H2O_final = min(y_H2O_sat, y_H2O_alvo)
    
    # Se y_H2O_final for muito próximo de 1.0 (saturação em P muito baixa), forçamos um valor seguro.
    if y_H2O_final >= 1.0:
        y_H2O_final = 1.0 - 1e-6
        
    # 1.3. Frações Molares Finais (Base Total)
    # A fração dos gases principais é reduzida para acomodar o vapor d'água final.
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
    
    try:
        P_sat_H2O_PA = calcular_pressao_sublimacao_gelo(T_K)
        
        # Defesa contra erro de saturação da água
        if abs(P_H2O_PA_parcial - P_sat_H2O_PA) / P_sat_H2O_PA < 1e-6: 
             P_H2O_calc = P_H2O_PA_parcial * (1.0 - 1e-7) 
        else:
             P_H2O_calc = P_H2O_PA_parcial
             
        H_H2O_vap = CP.PropsSI('H', 'T', T_K_SAFE, 'P', P_H2O_calc, 'Water') # J/kg_H2O

    except ValueError:
        # Fallback
        H_H2O_vap = CP.PropsSI('H', 'T', T_K_SAFE, 'P', P_PA / 10.0, 'Water') 
    
    # 2.2. Entalpia dos Gases Principais (H2 e O2)
    P_MIN_PA = 1.0 

    P_H2_PA_parcial = P_PA * y_H2_final
    P_H2_calc = max(P_H2_PA_parcial, P_MIN_PA)
    H_H2 = CP.PropsSI('H', 'T', T_K, 'P', P_H2_calc, 'H2') 

    P_O2_PA_parcial = P_PA * y_O2_final
    P_O2_calc = max(P_O2_PA_parcial, P_MIN_PA)
    H_O2 = CP.PropsSI('H', 'T', T_K, 'P', P_O2_calc, 'O2') 

    # 2.3. Entalpia Mássica da Mistura (J/kg_mix)
    H_mix_J_kg = (F_H2_molar * M_H2 * H_H2 + F_O2_molar * M_O2 * H_O2 + F_H2O_molar * M_H2O * H_H2O_vap) / m_dot_mix_kg_s
    
    # 3. Pressões Parciais
    P_H2O_parcial_bar = P_PA * y_H2O_final / 1e5
    
    # 4. Estado de Saturação (para exibição)
    estado_saturacao = "Saturado" if y_H2O_final >= y_H2O_sat else "Vapor Superaquecido"
    
    # 5. CÁLCULO DA FRAÇÃO MÁSSICA (w_H2O)
    w_H2O_final = (F_H2O_molar * M_H2O) / m_dot_mix_kg_s
    
    return {
        "T_C": T_C,
        "P_bar": P_bar,
        "y_H2O": y_H2O_final,
        "w_H2O": w_H2O_final, # Fração Mássica
        "y_O2": y_O2_final, 
        "y_H2": y_H2_final, 
        "m_dot_gas_kg_s": m_dot_gas_princ_kg_s, # Vazão mássica principal rastreada
        "m_dot_mix_kg_s": m_dot_mix_kg_s, # Vazão mássica total da mistura
        "m_dot_H2O_vap_kg_s": F_H2O_molar * M_H2O, # Vazão mássica de vapor de água
        "H_mix_J_kg": H_mix_J_kg,
        "P_H2O_bar": P_H2O_parcial_bar,
        "y_H2O_sat": y_H2O_sat,
        "Estado_H2O": estado_saturacao,
        "F_molar_total": F_molar_total,
        "H_in_mix_J_kg": H_mix_J_kg * m_dot_mix_kg_s, # Entalpia total (J/s) da mistura
        "gas_fluido": gas_fluido # Gás fluído para alertas
    }

def calcular_y_H2O_inicial(T_C, P_bar) -> float:
    """Calcula a fração molar inicial de água (y_H2O) assumindo saturação total."""
    T_K = T_C + 273.15
    P_PA = P_bar * 1e5
    try:
        # Usa a função que corrige o range
        P_sat_H2O_PA = calcular_pressao_sublimacao_gelo(T_K)
        return P_sat_H2O_PA / P_PA
    except:
        return 0.0 # Retorna 0 em caso de erro