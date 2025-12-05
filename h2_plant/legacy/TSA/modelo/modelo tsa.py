import numpy as np

# ==============================================================================
#                      PARTE 1: CONSTANTES E PARÂMETROS
# ==============================================================================

# Constantes Fundamentais e Propriedades dos Materiais
R_UNIVERSAL = 8.314        # Constante universal dos gases (J/(mol*K))
MM_H2 = 2.016e-3           # Massa Molar do H2 (kg/mol)
MM_H2O = 18.015e-3         # Massa Molar da H2O (kg/mol)
RHO_ADS = 700.0            # Densidade aparente do adsorvente (kg/m³) - Peneira 4A
EPSILON_BED = 0.40         # Porosidade do leito (fração de vazios)
DP_PARTICLE_M = 0.0025     # Diâmetro médio da partícula do adsorvente (2.5 mm)

# Propriedades do Fluxo (Dados do KOD 2)
T_H2_C = 4.00              # Temperatura de Entrada (°C)
P_H2_BAR = 39.20           # Pressão de Entrada (bar)
Q_MASS_H2 = 0.08527        # Vazão Mássica Total (kg/s)
Y_H2O_IN = 0.000207        # Fração Molar de H2O na entrada
T_H2 = T_H2_C + 273.15     # Temperatura em Kelvin
P_H2 = P_H2_BAR * 1e5      # Pressão em Pascal
# Viscosidade do H2 (aproximada para 4°C e 39 bar)
# A viscosidade do H2 é relativamente insensível à pressão em altas T e baixas P, 
# mas usamos um valor ajustado para a alta pressão
MU_H2_4C = 1.05e-5         # Viscosidade do H2 (Pa·s)

# Parâmetros de Dimensionamento Importados (Resultados da Etapa 1)
W_CAP = 0.05               # Capacidade de trabalho (kg_H2O/kg_ads)
T_ADS_HR = 6.0             # Tempo de adsorção por leito (horas)
L_D_RATIO = 2.5            # Relação Comprimento/Diâmetro
# Resultados do dimensionamento estático (Massa = 20.0 kg, D = 0.320 m, L = 0.800 m)
MASS_ADS_PER_BED_KG = 20.04  
DIAMETER_BED_M = 0.320       
LENGTH_BED_M = 0.800         

# Parâmetros de Modelagem de Regeneração
HEAT_ADS_KJ_KG = 2000.0    # Calor de Adsorção/Desorção da H2O (kJ/kg H2O)
CP_ADS_J_KG_K = 900.0      # Calor específico do adsorvente (J/(kg·K))
T_REG_C = 250.0            # Temperatura de Regeneração (target) (°C)
T_REG_K = T_REG_C + 273.15 # K
EFFICIENCY_HEAT = 0.85     # Eficiência da transferência de calor

# ==============================================================================
#                      PARTE 2: FUNÇÕES DE DIMENSIONAMENTO E MODELAGEM
# ==============================================================================

def calcular_propriedades_fluxo(q_mass_mix, mm_h2, mm_h2o, y_h2o_in, p_pa, t_k):
    """Calcula propriedades do fluxo de gás necessárias para a modelagem."""
    MM_MIX = y_h2o_in * mm_h2o + (1.0 - y_h2o_in) * mm_h2
    RHO_GAS = p_pa * MM_MIX / (R_UNIVERSAL * t_k)
    Q_MOLAR_MIX = q_mass_mix / MM_MIX
    
    # Vazão mássica de H2O (kg/s)
    Q_MASS_H2O = Q_MOLAR_MIX * y_h2o_in * mm_h2o
    
    return RHO_GAS, Q_MASS_H2O, MM_MIX

def calcular_queda_pressao_ergun(
    L, D_p, epsilon, mu, rho_gas, q_mass_mix, D_bed
):
    """
    Calcula a queda de pressão (ΔP) usando a Equação de Ergun.
    Esta é uma modelagem crucial do estágio de adsorção.
    """
    # 1. Massa Mássica (Velocidade Mássica Superficial) G = Q_mass / Area
    AREA_BED = np.pi * (D_bed**2) / 4.0
    G = q_mass_mix / AREA_BED # kg/(m²·s)

    # 2. Velocidade Superficial (u_s)
    u_s = G / rho_gas

    # 3. Equação de Ergun
    # Termo Viscoso (Laminar)
    TERM_VISCOUS = 150.0 * ( (1.0 - epsilon)**2 / (epsilon**3) ) * (mu * u_s / (D_p**2))
    # Termo Cinético (Turbulento)
    TERM_KINETIC = 1.75 * ( (1.0 - epsilon) / (epsilon**3) ) * (rho_gas * u_s**2 / D_p)

    DELTA_P = L * (TERM_VISCOUS + TERM_KINETIC) # Pa
    return DELTA_P / 1e5 # Retorna em bar

def calcular_energia_regeneracao(
    mass_ads, q_mass_h2o, t_ads_hr, heat_ads, cp_ads, t_reg_k, t_in_k, efficiency
):
    """
    Calcula a energia térmica requerida para regenerar o leito.
    Esta é uma modelagem crucial do estágio de regeneração.
    """
    T_ADS_SEC = t_ads_hr * 3600  # Tempo de adsorção em segundos

    # 1. Calor Sensível (Aquecer o Adsorvente)
    # Q_sensivel = Massa_ads * Cp_ads * Delta_T
    Q_SENSIBLE_J = mass_ads * cp_ads * (t_reg_k - t_in_k)

    # 2. Calor de Adsorção/Desorção (Remover a H2O)
    # Massa H2O total removida: Q_mass_h2o * T_ads_sec
    MASS_H2O_TOTAL = q_mass_h2o * T_ADS_SEC
    Q_DESORPTION_J = MASS_H2O_TOTAL * heat_ads * 1000 # kJ/kg -> J/kg

    # 3. Energia Térmica Total Mínima Requerida (J)
    Q_TOTAL_MIN_J = Q_SENSIBLE_J + Q_DESORPTION_J

    # 4. Energia com Eficiência (J)
    Q_TOTAL_J = Q_TOTAL_MIN_J / efficiency

    # 5. Potência Térmica Média (W) - Assume-se que a regeneração dura T_ADS_HR
    POWER_THERMAL_W = Q_TOTAL_J / T_ADS_SEC

    return Q_TOTAL_J, POWER_THERMAL_W, MASS_H2O_TOTAL

# ==============================================================================
#                      PARTE 3: EXECUÇÃO E SAÍDA DO MODELO
# ==============================================================================

# 1. Calcular Propriedades de Fluxo
rho_gas, q_mass_h2o, mm_mix = calcular_propriedades_fluxo(
    Q_MASS_H2, MM_H2, MM_H2O, Y_H2O_IN, P_H2, T_H2
)

# 2. Modelagem Operacional de Adsorção (Queda de Pressão)
delta_p_bar = calcular_queda_pressao_ergun(
    LENGTH_BED_M, DP_PARTICLE_M, EPSILON_BED, MU_H2_4C, rho_gas,
    Q_MASS_H2, DIAMETER_BED_M
)

# 3. Modelagem Operacional de Regeneração (Energia Térmica)
q_total_j, power_thermal_w, mass_h2o_total = calcular_energia_regeneracao(
    MASS_ADS_PER_BED_KG, q_mass_h2o, T_ADS_HR, 
    HEAT_ADS_KJ_KG, CP_ADS_J_KG_K, T_REG_K, T_H2, EFFICIENCY_HEAT
)

# Imprime os resultados da Modelagem
print("================================================================================")
print("             🧪 MODELAGEM OPERACIONAL SIMPLIFICADA DO TSA (H₂)          ")
print("================================================================================")
print(f"Dimensões do Leito: D={DIAMETER_BED_M:.3f} m, L={LENGTH_BED_M:.3f} m")
print(f"Massa de Adsorvente por Leito: {MASS_ADS_PER_BED_KG:.1f} kg")
print("--------------------------------------------------------------------------------")
print("                  MODELAGEM OPERACIONAL (ADSORÇÃO)                             ")
print("--------------------------------------------------------------------------------")
print(f"1. Densidade do Gás (Entrada): {rho_gas:.3f} kg/m³")
print(f"2. Queda de Pressão (ΔP) no Leito: {delta_p_bar * 1000:.2f} mbar")
print("   (Calculado via Equação de Ergun durante a Adsorção)")
print("--------------------------------------------------------------------------------")
print("                  MODELAGEM OPERACIONAL (REGENERAÇÃO)                          ")
print("--------------------------------------------------------------------------------")
print(f"3. Massa Total de H₂O Removida por Ciclo (6h): {mass_h2o_total:.3f} kg")
print(f"4. Energia Térmica Total Requerida (Q_Total): {q_total_j / 1e6:.2f} MJ (por ciclo)")
print(f"5. Potência Térmica Média Requerida (P_Médio): {power_thermal_w / 1000:.2f} kW")
print("   (Energia necessária para aquecer o adsorvente a 250°C e desorver H₂O)")
print("================================================================================")