import numpy as np
import CoolProp.CoolProp as CP

# ==============================================================================
#                      PARTE 1: CONSTANTES E PARÂMETROS
# ==============================================================================

# Constantes Fundamentais e Propriedades dos Materiais
R_UNIVERSAL = 8.314        # Constante universal dos gases (J/(mol*K))
MM_H2 = CP.PropsSI('M', 'H2') # Massa Molar do H2 (kg/mol)
MM_H2O = CP.PropsSI('M', 'H2O') # Massa Molar da H2O (kg/mol)
RHO_ADS = 700.0            # Densidade aparente do adsorvente (kg/m³) - Peneira 4A
EPSILON_BED = 0.40         # Porosidade do leito (fração de vazios)
DP_PARTICLE_M = 0.0025     # Diâmetro médio da partícula do adsorvente (2.5 mm)

# Parâmetros de Dimensionamento Importados (Resultados da Etapa 1) - FIXOS NO MODELO TSA
W_CAP = 0.05               # Capacidade de trabalho (kg_H2O/kg_ads)
T_ADS_HR = 6.0             # Tempo de adsorção por leito (horas)
MASS_ADS_PER_BED_KG = 20.04 # kg
T_ADS_SEC = T_ADS_HR * 3600  # Tempo de adsorção em segundos

# Parâmetros de Modelagem de Regeneração
HEAT_ADS_KJ_KG = 2000.0    # Calor de Adsorção/Desorção da H2O (kJ/kg H2O)
CP_ADS_J_KG_K = 900.0      # Calor específico do adsorvente (J/(kg·K))
T_REG_C = 250.0            # Temperatura de Regeneração (target) (°C)
EFFICIENCY_HEAT = 0.85     # Eficiência da transferência de calor


def calcular_propriedades_fluxo(q_mass_mix, mm_h2, mm_h2o, y_h2o_in, p_pa, t_k):
    """Calcula propriedades do fluxo de gás necessárias para a modelagem."""
    MM_MIX = y_h2o_in * mm_h2o + (1.0 - y_h2o_in) * mm_h2
    
    # Usando CoolProp para densidade real (melhor que Gás Ideal)
    try:
        y_h2_in = 1.0 - y_h2o_in
        y_h2o_in_safe = round(y_h2o_in, 8)
        y_h2_in_safe = round(y_h2_in, 8)
        y_sum = y_h2o_in_safe + y_h2_in_safe
        if y_sum != 1.0: y_h2_in_safe += (1.0 - y_sum)
        
        mixture_string = f'H2[{y_h2_in_safe:.8f}]|H2O[{y_h2o_in_safe:.8f}]'
        RHO_GAS = CP.PropsSI('D', 'P', p_pa, 'T', t_k, mixture_string)
    except Exception:
        # Fallback Gás Ideal
        RHO_GAS = p_pa * MM_MIX / (R_UNIVERSAL * t_k)
    
    Q_MOLAR_MIX = q_mass_mix / MM_MIX
    Q_MASS_H2O = Q_MOLAR_MIX * y_h2o_in * mm_h2o
    
    return RHO_GAS, Q_MASS_H2O, MM_MIX

def calcular_energia_regeneracao(
    mass_ads, q_mass_h2o, t_ads_sec, heat_ads, cp_ads, t_reg_k, t_in_k, efficiency
):
    """
    Calcula a energia térmica requerida para regenerar o leito.
    """
    # 1. Calor Sensível (Aquecer o Adsorvente)
    Q_SENSIBLE_J = mass_ads * cp_ads * (t_reg_k - t_in_k)

    # 2. Calor de Adsorção/Desorção (Remover a H2O)
    MASS_H2O_TOTAL = q_mass_h2o * t_ads_sec
    Q_DESORPTION_J = MASS_H2O_TOTAL * heat_ads * 1000 # kJ/kg -> J/kg

    # 3. Energia Térmica Total Mínima Requerida (J)
    Q_TOTAL_MIN_J = Q_SENSIBLE_J + Q_DESORPTION_J

    # 4. Energia com Eficiência (J)
    Q_TOTAL_J = Q_TOTAL_MIN_J / efficiency

    # 5. Potência Térmica Média (W) - Assume-se que a regeneração dura T_ADS_SEC
    POWER_THERMAL_W = Q_TOTAL_J / T_ADS_SEC

    return Q_TOTAL_J, POWER_THERMAL_W, MASS_H2O_TOTAL


def modelar_tsa(m_dot_g_kg_s: float, P_in_bar: float, T_in_C: float, y_H2O_in: float, y_O2_in: float) -> dict:
    """
    Modela o TSA para purificação de H2 (removendo H2O para pureza ultra-alta).
    Assume 2 leitos operando em ciclo de 6 horas.
    """
    
    T_H2_K = T_in_C + 273.15
    P_H2_PA = P_in_bar * 1e5
    T_REG_K = T_REG_C + 273.15
    
    # 1. Calcular Propriedades de Fluxo
    rho_gas, q_mass_h2o, mm_mix = calcular_propriedades_fluxo(
        m_dot_g_kg_s, MM_H2, MM_H2O, y_H2O_in, P_H2_PA, T_H2_K
    )

    # 2. Modelagem Operacional de Regeneração (Energia Térmica)
    q_total_j, power_thermal_w, mass_h2o_total = calcular_energia_regeneracao(
        MASS_ADS_PER_BED_KG, q_mass_h2o, T_ADS_SEC, 
        HEAT_ADS_KJ_KG, CP_ADS_J_KG_K, T_REG_K, T_H2_K, EFFICIENCY_HEAT
    )
    
    # Queda de pressão no TSA - Usamos um valor fixo baixo para simplicidade
    # (A Ergun original requeria mais constantes dimensionais)
    DELTA_P_TSA_BAR = 0.05 
    
    # Pureza (TSA pode atingir pureza ultra-alta)
    Y_H2O_OUT_PPM = 5.0 
    y_H2O_out_vap = Y_H2O_OUT_PPM / 1e6 
    
    results = {
        "T_C": T_in_C, # Assumido isotérmico para TSA
        "P_out_bar": P_in_bar - DELTA_P_TSA_BAR,
        "y_H2O_out": y_H2O_out_vap,
        "m_dot_gas_out_kg_s": m_dot_g_kg_s, # Nenhuma perda de H2
        "Agua_Removida_kg_s": q_mass_h2o,
        
        "Q_dot_thermal_W": power_thermal_w, # Potência térmica (W)
        "W_dot_comp_W": 0.0, # Nenhum consumo elétrico direto para purga/compressão
        
        "M_ads_total_kg": MASS_ADS_PER_BED_KG * 2, # Assumindo 2 leitos (2 * 20.04 kg)
        "H2_Perdido_kg_s": 0.0, # Nenhuma perda de H2
        "DELTA_P_TSA_bar": DELTA_P_TSA_BAR
    }
    
    return results