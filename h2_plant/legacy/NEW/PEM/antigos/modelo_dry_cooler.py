import numpy as np
from aux_coolprop import get_gas_cp_and_liquid_cp
from constants_and_config import (
    AREA_H2_DESIGN, AREA_O2_DESIGN,
    M_DOT_A_H2_DESIGN, M_DOT_A_O2_DESIGN,
    U_VALUE_DESIGN, T_A_IN_OP,
    DP_AIR_DESIGN, P_PERDA_BAR
)
# A importação de 'modelo_ventiladores' foi removida.

# =================================================================
# === FUNÇÃO INCORPORADA: CÁLCULO DA POTÊNCIA DO VENTILADOR ===
# =================================================================

def calcular_potencia_ventilador(m_dot_ar_kg_s: float, DP_ar_Pa: float, eficiencia: float = 0.65) -> float:
    """
    Calcula a potência elétrica do ventilador (W) necessária para mover o ar.
    
    W_dot = (V_dot * DP) / eficiência, onde V_dot = m_dot / rho.
    """
    
    # Propriedades do ar (assumindo ar a 25°C e 1 atm, ρ ≈ 1.18 kg/m³)
    RHO_AR_KGM3 = 1.18 
    
    if m_dot_ar_kg_s <= 0:
        return 0.0
        
    # Vazão Volumétrica (m³/s)
    V_dot_ar_m3_s = m_dot_ar_kg_s / RHO_AR_KGM3
    
    # Potência do ventilador (W)
    W_dot_ventilador_W = (V_dot_ar_m3_s * DP_ar_Pa) / eficiencia
    
    return W_dot_ventilador_W

# =================================================================
# === MODELO PRINCIPAL: DRY COOLER ===
# =================================================================

def modelar_dry_cooler(
    gas_fluido: str,
    m_dot_mix_kg_s: float,
    m_dot_H2O_liq_kg_s: float,
    P_in_bar: float,
    T_g_in_C: float,
    m_dot_a_op: float = None
) -> dict:
    """
    Modelagem do Dry Cooler contabilizando energia do gás/vapor + água líquida acompanhante.
    """
    if gas_fluido == 'H2':
        Area_m2 = AREA_H2_DESIGN
        if m_dot_a_op is None:
            m_dot_a_op = M_DOT_A_H2_DESIGN
    elif gas_fluido == 'O2':
        Area_m2 = AREA_O2_DESIGN
        if m_dot_a_op is None:
            m_dot_a_op = M_DOT_A_O2_DESIGN
    else:
        return {"erro": f"Gás {gas_fluido} não suportado."}

    U_value = U_VALUE_DESIGN
    T_a_in_op = T_A_IN_OP
    
    # Cp gás e Cp água líquida
    c_p_g, c_p_H2O_liq = get_gas_cp_and_liquid_cp(gas_fluido)
    c_p_a = 1005.0  # Ar

    # Mistura física total deve incluir o líquido
    if m_dot_mix_kg_s < m_dot_H2O_liq_kg_s:
        return {"erro": "m_dot_mix_kg_s < m_dot_H2O_liq_kg_s (fluxo inconsistente no Dry Cooler)."}

    # Fase gasosa remanescente
    m_dot_gas_fase = m_dot_mix_kg_s - m_dot_H2O_liq_kg_s

    # Capacidade térmica do fluxo quente total
    C_gas_mix = m_dot_gas_fase * c_p_g
    C_liquido = m_dot_H2O_liq_kg_s * c_p_H2O_liq
    C_quente = C_gas_mix + C_liquido

    C_a = m_dot_a_op * c_p_a

    C_min = min(C_quente, C_a)
    if C_min <= 0:
        return {"erro": "C_min é zero. Vazão mássica ou cp inválido."}

    C_max = max(C_quente, C_a)
    Cr = C_min / C_max
    NTU = U_value * Area_m2 / C_min

    E = 1.0 - np.exp((np.exp(-Cr * NTU) - 1.0) / Cr)

    T_g_in_K = T_g_in_C + 273.15
    T_a_in_K = T_a_in_op + 273.15
    Q_max = C_min * (T_g_in_K - T_a_in_K)

    Q_dot_real_W = E * Q_max

    T_g_out_K = T_g_in_K - Q_dot_real_W / C_quente
    T_g_out_C = T_g_out_K - 273.15
    
    # === CÁLCULO DA TEMPERATURA DE SAÍDA DO LADO FRIO (AR) ===
    # Q_dot_real = C_a * (T_a_out_K - T_a_in_K)
    T_a_out_K = T_a_in_K + Q_dot_real_W / C_a
    T_a_out_C = T_a_out_K - 273.15
    # =========================================================

    # A função calcular_potencia_ventilador agora está definida neste arquivo.
    W_dot_ventilador_W = calcular_potencia_ventilador(m_dot_a_op, DP_AIR_DESIGN) 
    P_out_bar = P_in_bar - P_PERDA_BAR

    return {
        "T_C": T_g_out_C,
        "P_bar": P_out_bar,
        "Q_dot_fluxo_W": -abs(Q_dot_real_W),  # calor removido do fluido
        "W_dot_comp_W": W_dot_ventilador_W,
        # === NOVOS CAMPOS PARA RASTREIO DO LADO FRIO ===
        "T_cold_out_C": T_a_out_C, # Temperatura de saída do AR
        "m_dot_cold_liq_kg_s": m_dot_a_op, # Vazão do AR de resfriamento (m_dot_a_op)
        # ===============================================
    }