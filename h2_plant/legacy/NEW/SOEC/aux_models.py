# aux_models.py
# Contém as funções analíticas de suporte (compressão, resfriamento simples)
# e as constantes e imports necessários para estas funções.

import CoolProp.CoolProp as CP
import numpy as np
# Importação explícita das funções termodinâmicas de aux_coolprop
from aux_coolprop import calcular_y_H2O_inicial

# Importações de constantes de limites
from constants_and_config import (
    P_MAX_TEORICA_COMPRESSOR_H2_BAR, 
)

# 💥 CONSTANTE DE EFICIÊNCIA USADA NO MODELO COMPRESSOR (PARA AJUSTE DA LÓGICA ANALÍTICA)
ETA_IS_ASSUMIDA = 0.65 # Eficiência isentrópica real (65%)

# ----------------------------------------------------------------------
# NOVO: MODELO DE RESFRIADOR SIMPLES
# ----------------------------------------------------------------------
def modelar_resfriador_simples(estado_in: dict, T_target_C: float, gas_fluido: str, calcular_estado_termodinamico_func):
    """
    Resfria o gás (compressor after/intercooler) para uma T alvo (T_target_C).
    
    Args:
        estado_in (dict): Estado termodinâmico de entrada.
        T_target_C (float): Temperatura alvo de saída.
        gas_fluido (str): Nome do fluido ('H2' ou 'O2').
        calcular_estado_termodinamico_func (function): Função do aux_coolprop para calcular o estado da mistura.
    """
    
    T_out_C = T_target_C; P_out_bar = estado_in['P_bar']
    
    # 1. Calcula o estado na T alvo
    y_H2O_out_sat = calcular_y_H2O_inicial(T_out_C, P_out_bar) 
    y_H2O_out_vap = min(y_H2O_out_sat, estado_in['y_H2O']) # Pode condensar
    
    # Usa a função passada como argumento
    estado_out_calc = calcular_estado_termodinamico_func(gas_fluido, T_out_C, P_out_bar, estado_in['m_dot_gas_kg_s'], y_H2O_out_vap, estado_in['y_O2'], estado_in['y_H2'])

    # 2. Balanço de Energia (para Q)
    Q_dot_fluxo_W = estado_out_calc['H_in_mix_J_kg'] - estado_in['H_in_mix_J_kg'] # Q removido (negativo)

    # 3. Condensação
    m_dot_H2O_vap_out = estado_out_calc['m_dot_H2O_vap_kg_s']
    m_dot_H2O_vap_in = estado_in['m_dot_H2O_vap_kg_s']
    Agua_Condensada = max(0.0, m_dot_H2O_vap_in - m_dot_H2O_vap_out)
    
    m_dot_H2O_liq_accomp_in = estado_in.get('m_dot_H2O_liq_accomp_kg_s', 0.0)
    
    return {
        'estado_termodinamico': estado_out_calc,
        'Q_dot_fluxo_W': Q_dot_fluxo_W,
        'Agua_Condensada_kg_s': Agua_Condensada,
        'm_dot_liq_accomp_out': m_dot_H2O_liq_accomp_in + Agua_Condensada, 
        'T_C': T_out_C, 
        'P_bar': P_out_bar
    }

# =================================================================
# 💥 FUNÇÃO ANALÍTICA: CALCULA PRESSÃO MÁXIMA BASEADA EM T_MAX
# =================================================================

def calcular_pressao_maxima_analitica(estado_in: dict, fluido_nome: str, T_alvo_max_C: float, eta_is: float = ETA_IS_ASSUMIDA) -> float:
    """
    Calcula a pressão máxima de saída P_out que resulta em T_out real <= T_alvo_max_C,
    usando a relação isentrópica ajustada pela eficiência (eta_is).
    """
    T1_C = estado_in['T_C']
    P1_bar = estado_in['P_bar']
    
    T1_K = T1_C + 273.15
    T_max_real_K = T_alvo_max_C + 273.15
    
    if T1_K >= T_max_real_K:
        return P1_bar 
        
    # 1. Calcula a temperatura isentrópica máxima (T2,ideal) que deve ser atingida
    T_ideal_max_K = T1_K + eta_is * (T_max_real_K - T1_K)
    
    if T_ideal_max_K <= T1_K:
         return P1_bar * 1.0001 

    # Tenta obter o índice isentrópico (k = Cp/Cv) do CoolProp
    P_Pa = P1_bar * 1e5
    
    # 🛑 SOLUÇÃO DEFINITIVA: Mapeamento de nome de fluido para evitar o ValueError 'CPLMASS'
    mapa_fluido = {
        'h2': 'H2',
        'hydrogen': 'H2',
        'o2': 'O2',
        'oxygen': 'O2'
    }

    fluido_cp = mapa_fluido.get(fluido_nome.lower(), fluido_nome)

    # Verifica se o fluido_cp do estado termodinâmico é H2O (mistura)
    fluido_cp_estado = estado_in.get('fluido_cp', '').lower()
    if 'h2o' in fluido_cp_estado: 
        k = 1.41 if fluido_nome == 'hydrogen' else 1.40
    else:
        # Usa o nome mapeado ("H2" ou "O2")
        Cp = CP.PropsSI('CPMASS', 'P', P_Pa, 'T', T1_K, fluido_cp)
        Cv = CP.PropsSI('CVMASS', 'P', P_Pa, 'T', T1_K, fluido_cp)
        k = Cp / Cv
        
    # Razão máxima de temperatura isentrópica permitida
    Ratio_T_ideal_max = T_ideal_max_K / T1_K
    
    # Exponente do processo isentrópico (k / (k-1))
    exp = k / (k - 1.0)
    
    # Razão máxima de pressão permitida (P2/P1)
    Ratio_P_max = Ratio_T_ideal_max ** exp
    
    # P_out máxima
    P2_max_bar = P1_bar * Ratio_P_max
    
    P2_max_bar = min(P2_max_bar, P_MAX_TEORICA_COMPRESSOR_H2_BAR)
    
    return P2_max_bar