# modelo_psa.py
import numpy as np
from CoolProp.CoolProp import PropsSI

# --- CONSTANTES DE ENGENHARIA (Para Ergun e Dimensionamento) ---
RHO_ADS = 700.0            # Densidade aparente (kg/m³)
EPSILON_BED = 0.40         # Porosidade do leito (adimensional)
DP_PARTICLE_M = 0.003      # Diâmetro médio da partícula do adsorvente (m) - Usado no Ergun
L_LEITO_DEFAULT_M = 1.0    # Comprimento de leito assumido para Ergun simplificado (m)
D_LEITO_DEFAULT_M = 0.35   # Diâmetro de leito assumido para Ergun simplificado (m)

# --- 1. DADOS DE ENTRADA DO PROCESSO PSA (H2) ---
ETA_REC_H2 = 0.90   # Recuperação de H2 desejada (90%)
P_AD_PA = 40.0 * 1e5 # P_alta (Pa)
P_REG_PA = 1.0 * 1e5 # P_baixa (Regeneração) (Pa)

# MODIFICAÇÕES SOLICITADAS:
T_CICLO = 250.0     # s (Tempo total de um ciclo)
N_VASOS = 4         # Número de vasos 
# Fator MTZ/Eficiência Dinâmica
DELTA_Q_H2O_ESTATICA = 0.10  # kg H2O / kg Adsorvente (Capacidade Estática)
FATOR_EFICIENCIA_MTZ = 0.70 # Fator de eficiência dinâmica/MTZ
DELTA_Q_H2O_EFETIVA = DELTA_Q_H2O_ESTATICA * FATOR_EFICIENCIA_MTZ 

FW = 1.5            # Fator de Quebra do Leito (Segurança)
Y_H2O_OUT_PPM = 5.0 # Pureza alvo (5 ppmv)

# Parâmetros Empíricos/Termodinâmicos
KAPPA_H2 = 1.4
ETA_COMP = 0.75

# --- REMOVIDO: DELTA_P_PSA_FIXO_BAR (Será usado o valor da Ergun) ---

# --- FUNÇÕES DE CÁLCULO ---

def calcular_trabalho_comp_purga(dot_m_purga_total, T_K, P_adsorcao_Pa, P_regeneracao_Pa):
    """Estima a potência necessária para compressão/vácuo do gás de purga (kW)."""
    
    # Assumimos a densidade do H2 puro para simplificação na P_baixa
    rho_H2_purga = PropsSI('D', 'P', P_regeneracao_Pa, 'T', T_K, 'H2')
    
    if dot_m_purga_total == 0 or rho_H2_purga == 0:
        return 0.0
        
    dot_V_purga = dot_m_purga_total / rho_H2_purga # Vazão volumétrica de purga (m³/s)

    # Trabalho isentrópico (W_dot)
    W_dot_isentropico = (dot_V_purga * P_regeneracao_Pa * KAPPA_H2 / (KAPPA_H2 - 1)) * \
                        ((P_adsorcao_Pa / P_regeneracao_Pa)**((KAPPA_H2 - 1) / KAPPA_H2) - 1)

    Potencia_Estimada_W = W_dot_isentropico / ETA_COMP # Potência em W
    return Potencia_Estimada_W

def calcular_delta_p_ergun(dot_V, A_c, L_leito, eps, dp, mu, rho):
    """
    Calcula a queda de pressão (DeltaP) usando a Equação de Ergun.
    dot_V: vazão volumétrica (m3/s)
    A_c: área de seção transversal do leito (m2)
    Retorna DeltaP (Pa)
    """
    if A_c == 0 or L_leito == 0:
        return 0.0
        
    # velocidade superficial
    u = dot_V / A_c
    
    # Termo Viscoso (Laminar)
    term1 = (150 * (1 - eps)**2 * mu * u) / (eps**3 * dp**2)
    # Termo Cinético (Turbulento)
    term2 = (1.75 * (1 - eps) * rho * u**2) / (eps**3 * dp)
    
    deltaP_por_L = term1 + term2
    return deltaP_por_L * L_leito # Pa

def modelar_psa(m_dot_g_kg_s: float, P_in_bar: float, T_in_C: float, y_H2O_in: float, y_O2_in: float, N_vasos_int: int = N_VASOS) -> dict:
    """
    Modela o PSA para purificação de H2 (removendo H2O para pureza ultra-alta).
    """
    
    T_K = T_in_C + 273.15
    P_in_Pa = P_in_bar * 1e5
    
    # --- CONSTANTES ---
    MM_H2 = PropsSI('M', 'H2')
    MM_H2O = PropsSI('M', 'H2O')
    MM_O2 = PropsSI('M', 'O2')
    
    # --- CÁLCULO DE VAZÕES DE ENTRADA (CORREÇÃO DE VAZÃO) ---
    # m_dot_g_kg_s é a vazão mássica total de mistura (kg/s)
    m_dot_total = m_dot_g_kg_s
    
    # Frações molares para obtenção das massas molares médias e vazões
    y_H2_in = 1.0 - y_H2O_in - y_O2_in
    MM_mix_in = y_H2_in * MM_H2 + y_H2O_in * MM_H2O + y_O2_in * MM_O2
    
    # Vazão Molar Total de entrada
    dot_F_mix_in = m_dot_total / MM_mix_in
    
    # Vazões Mássicas por Componente (kg/s) - Correção da Métrica de Vazão
    dot_m_H2O_in = dot_F_mix_in * y_H2O_in * MM_H2O
    dot_m_O2_in  = dot_F_mix_in * y_O2_in * MM_O2
    dot_m_H2_in  = dot_F_mix_in * y_H2_in * MM_H2

    # Verificação: m_dot_total deve ser igual à soma de dot_m_X_in. 
    # Usaremos dot_m_H2_in no cálculo de recuperação e dot_m_H2O_in no cálculo de dim.
    
    # --- 2. DIMENSIONAMENTO DE ADSORVENTE (CAPACIDADE DE ÁGUA) ---
    M_H2O_ciclo = dot_m_H2O_in * T_CICLO
    
    # USO DA CAPACIDADE EFETIVA PARA SIMULAR MTZ
    M_ads_total = FW * (M_H2O_ciclo / DELTA_Q_H2O_EFETIVA) 
    
    V_ads_total = M_ads_total / RHO_ADS

    # --- 3. SAÍDA (Pós-Adsorção/Purga) ---
    
    # Perda de H2 (CUSTO)
    dot_m_H2_produto = dot_m_H2_in * ETA_REC_H2 
    dot_m_H2_purga = dot_m_H2_in * (1.0 - ETA_REC_H2)
    
    # O2 (Não adsorvido significativamente)
    dot_m_O2_out = dot_m_O2_in # O O2 remanescente sai no produto
    m_dot_gas_out_princ = dot_m_H2_produto + dot_m_O2_out # Vazão mássica de H2 produto + O2
    
    # Pureza (H2O reduzida para nível ppm)
    y_H2O_out_vap = Y_H2O_OUT_PPM / 1e6 
    
    # --- 4. CÁLCULO DE QUEDA DE PRESSÃO (ERGUN) ---
    
    # Cálculo da Densidade e Viscosidade (CoolProp com fallback)
    # CORRIGIDO: Usar H2 PURO para cálculo de rho e mu do leito (melhora a precisão e evita o erro do CoolProp).
    try:
        rho_mix_in = PropsSI('D', 'P', P_in_Pa, 'T', T_K, 'H2')
        mu_mix_in = PropsSI('V', 'P', P_in_Pa, 'T', T_K, 'H2') 
    except ValueError:
        # Fallback: Lei dos Gases Ideais e Viscosidade de H2 puro a baixa pressão (aproximação)
        R_UNIV = 8.31446 
        rho_mix_in = P_in_Pa * MM_H2 / (R_UNIV * T_K) # Usando MM_H2
        mu_mix_in = 1.05e-5 
        print("Aviso: Falha CoolProp (Ergun). Usando Lei dos Gases Ideais/Viscosidade aproximada.")
        
    dot_V_in = m_dot_total / rho_mix_in 

    # Parâmetros físicos do leito (Estimativas para Ergun)
    A_c = np.pi * (D_LEITO_DEFAULT_M**2) / 4.0
    
    # --- NOVO: Cálculo e Aplicação da Lei de Ergun ---
    DELTA_P_PSA_PA = calcular_delta_p_ergun(
        dot_V_in, A_c, L_LEITO_DEFAULT_M, EPSILON_BED, DP_PARTICLE_M, mu_mix_in, rho_mix_in
    )
    
    # Converte o resultado da Ergun para Bar
    DELTA_P_TOTAL_PSA_BAR = DELTA_P_PSA_PA / 1e5
    
    P_out_bar = P_in_bar - DELTA_P_TOTAL_PSA_BAR
    
    # --- 5. TRABALHO DE COMPRESSÃO (CUSTO) ---
    # O fluxo de purga total deve incluir o H2 perdido (dot_m_H2_purga) E a água removida (dot_m_H2O_in)
    dot_m_purga_total = dot_m_H2_purga + dot_m_H2O_in 
    W_dot_comp_W = calcular_trabalho_comp_purga(dot_m_purga_total, T_K, P_in_Pa, P_REG_PA) # W

    # Saída Final
    results = {
        "T_C": T_in_C, 
        "P_in_bar": P_in_bar,
        "P_out_bar": P_out_bar, 
        # MODIFICADO: Retorna o delta P total (calculado pela Ergun)
        "DELTA_P_PSA_bar": DELTA_P_TOTAL_PSA_BAR,
        "y_H2O_out": y_H2O_out_vap,
        "m_dot_gas_out_kg_s": m_dot_gas_out_princ,
        "Agua_Removida_kg_s": dot_m_H2O_in,
        
        "Q_dot_fluxo_W": 0.0,
        "W_dot_comp_W": W_dot_comp_W,
        
        "M_ads_total_kg": M_ads_total,
        "V_ads_total_m3": V_ads_total,
        "H2_Perdido_kg_s": dot_m_H2_purga,
        "N_vasos": N_VASOS
    }
    
    return results