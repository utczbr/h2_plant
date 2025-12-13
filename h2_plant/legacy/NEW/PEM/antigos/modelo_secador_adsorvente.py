import numpy as np
from CoolProp.CoolProp import PropsSI

# --- PARÂMETROS GLOBAIS DO SECADOR (Simples) ---
P_AD_PA = 4.0 * 1e5   # P_alta
P_REG_PA = 1.0 * 1e5  # P_baixa
ETA_REMOCAO_H2O = 0.95 # Alvo: Redução de 95% da H2O (diferente da capacidade de trabalho)
DELTA_P_ADS_BAR = 0.5 # Queda de pressão no secador
FW = 1.5
DELTA_Q_H2O = 0.10
RHO_ADS = 700.0
T_CICLO = 120.0

# Parâmetros Termodinâmicos (para O2/N2)
KAPPA_O2 = 1.4
ETA_COMP = 0.75

def calcular_trabalho_comp_purga_o2(dot_m_purga_total, T_K, P_adsorcao_Pa, P_regeneracao_Pa):
    """Estima a potência necessária para compressão/vácuo do gás de purga (W)."""
    
    # Assumimos a densidade do O2 puro para simplificação na P_baixa
    rho_O2_purga = PropsSI('D', 'P', P_regeneracao_Pa, 'T', T_K, 'O2')
    
    if dot_m_purga_total == 0 or rho_O2_purga == 0:
        return 0.0
        
    dot_V_purga = dot_m_purga_total / rho_O2_purga

    # Trabalho isentrópico (W_dot)
    W_dot_isentropico = (dot_V_purga * P_regeneracao_Pa * KAPPA_O2 / (KAPPA_O2 - 1)) * \
                        ((P_adsorcao_Pa / P_regeneracao_Pa)**((KAPPA_O2 - 1) / KAPPA_O2) - 1)

    Potencia_Estimada_W = W_dot_isentropico / ETA_COMP # Potência em W
    return Potencia_Estimada_W

def modelar_secador_ads(m_dot_g_kg_s: float, P_in_bar: float, T_in_C: float, y_H2O_in: float, y_H2_in: float) -> dict:
    """
    Modela o Secador Adsorvente (Dryer) para o fluxo de O2 (remoção de H2O).
    Reduz a umidade em 95%. Não altera a vazão do gás principal (O2 + H2).
    """
    
    T_K = T_in_C + 273.15
    P_in_Pa = P_in_bar * 1e5
    
    MM_O2 = PropsSI('M', 'O2')
    MM_H2O = PropsSI('M', 'H2O')
    MM_H2 = PropsSI('M', 'H2')
    
    # 1. ENTRADA (Fração Molar e Mássica)
    y_O2_in = 1.0 - y_H2O_in - y_H2_in
    MM_mix = y_O2_in * MM_O2 + y_H2_in * MM_H2 + y_H2O_in * MM_H2O
    
    # Vazão molar total de entrada (usando a massa mássica total de gás principal (O2) m_dot_g_kg_s)
    dot_F_mix_in = m_dot_g_kg_s / (y_O2_in * MM_O2) # Aproximação baseada no O2 principal

    dot_m_H2O_in = dot_F_mix_in * y_H2O_in * MM_H2O
    
    # 2. SAÍDA (Remoção de Água)
    
    # Água removida (mássica)
    dot_m_H2O_removida = dot_m_H2O_in * ETA_REMOCAO_H2O
    
    # Água de saída (mássica)
    dot_m_H2O_out = dot_m_H2O_in - dot_m_H2O_removida
    
    # Recalcula y_H2O de saída (molar)
    F_H2O_out = dot_m_H2O_out / MM_H2O
    F_O2_H2_in = dot_F_mix_in - (dot_m_H2O_in / MM_H2O) # Vazão molar do gás não condensável
    
    # Evita divisão por zero/negativo
    if (F_O2_H2_in + F_H2O_out) <= 0:
        y_H2O_out_vap = 0.0
    else:
        y_H2O_out_vap = F_H2O_out / (F_O2_H2_in + F_H2O_out)
    
    # 3. CUSTOS E PERDAS
    
    P_out_bar = P_in_bar - DELTA_P_ADS_BAR
    
    # Potência (Assume que o gás de purga é a água removida e uma pequena fração do O2/H2)
    dot_m_purga_total = dot_m_H2O_removida + (m_dot_g_kg_s * 0.01) # Estimativa
    
    W_dot_comp_W = calcular_trabalho_comp_purga_o2(dot_m_purga_total, T_K, P_in_Pa, P_REG_PA) # W

    # 4. Dimensionamento do Adsorvente (Apenas para referência, mas necessário para manter a estrutura)
    M_H2O_ciclo = dot_m_H2O_in * T_CICLO
    M_ads_total = FW * (M_H2O_ciclo / DELTA_Q_H2O)

    # Saída Final
    results = {
        "T_C": T_in_C, # Isotérmico
        "P_bar": P_out_bar,
        "y_H2O_out": y_H2O_out_vap,
        "m_dot_gas_out_kg_s": m_dot_g_kg_s, # Vazão principal O2 não muda
        "Agua_Removida_kg_s": dot_m_H2O_removida,
        "Q_dot_fluxo_W": 0.0,
        "W_dot_comp_W": W_dot_comp_W,
        "M_ads_total_kg": M_ads_total
    }
    
    return results