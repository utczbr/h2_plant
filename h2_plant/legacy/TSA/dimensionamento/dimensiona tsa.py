import numpy as np

# ==============================================================================
#                      PARTE 1: DEFINIÇÃO DE CONSTANTES E PARÂMETROS
# ==============================================================================

# Constantes Fundamentais e Propriedades dos Materiais
R_UNIVERSAL = 8.314        # J/(mol*K)
MM_H2 = 2.016e-3           # Massa Molar do H2 (kg/mol)
MM_H2O = 18.015e-3         # Massa Molar da H2O (kg/mol)
RHO_ADS = 700.0            # Densidade aparente do adsorvente (kg/m³) - Peneira 4A
MU_H2_4C = 1.05e-5         # Viscosidade do H2 a 4°C e 39 bar (aproximada, Pa·s)
EPSILON_BED = 0.40         # Porosidade do leito (fração de vazios)

# Parâmetros de Dimensionamento (Premissas do Engenheiro)
W_CAP = 0.05               # Capacidade de trabalho (kg_H2O/kg_ads) - 5%
T_ADS_HR = 6.0             # Tempo de adsorção por leito (horas)
L_D_RATIO = 2.5            # Relação Comprimento/Diâmetro do leito (L/D)

# Dados de Entrada do Fluxo H2 (Estado Final do KOD 2)
T_H2_C = 4.00
P_H2_BAR = 39.20
Q_MASS_H2 = 0.08527        # Vazão Mássica de Gás (kg/s)
Y_H2O_IN = 0.000207        # Fração Molar de H2O na entrada
T_H2 = T_H2_C + 273.15     # Temperatura em Kelvin
P_H2 = P_H2_BAR * 1e5      # Pressão em Pascal

# ==============================================================================
#                      PARTE 2: FUNÇÃO DE DIMENSIONAMENTO PRINCIPAL
# ==============================================================================

def dimensionar_tsa(
    q_mass_mix, mm_h2o, mm_gas, y_h2o_in, w_cap, t_ads_hr, rho_ads, l_d_ratio, t_k, p_pa
):
    """
    Calcula a massa, volume e dimensões físicas do leito TSA.
    """
    # 1. Massa Molar Média da Mistura (kg/mol)
    MM_MIX = y_h2o_in * mm_h2o + (1.0 - y_h2o_in) * mm_gas

    # 2. Vazão Mássica de H2O no Contaminante (kg_H2O/s)
    Q_MOLAR_MIX = q_mass_mix / MM_MIX
    Q_MASS_H2O = Q_MOLAR_MIX * y_h2o_in * mm_h2o

    # 3. Massa Total de H2O a ser Removida no Período de Adsorção (kg_H2O)
    T_ADS_SEC = t_ads_hr * 3600  # Tempo de adsorção em segundos
    MASS_H2O_TOTAL = Q_MASS_H2O * T_ADS_SEC

    # 4. Massa e Volume de Adsorvente Requerida por Leito
    MASS_ADS_PER_BED = MASS_H2O_TOTAL / w_cap
    VOL_ADS_PER_BED = MASS_ADS_PER_BED / rho_ads

    # 5. Dimensionamento Físico da Coluna (L/D)
    # Volume = (pi/4) * L_D_RATIO * D³
    DIAMETER_BED = (
        (VOL_ADS_PER_BED * 4.0 / np.pi) / l_d_ratio
    ) ** (1/3)
    LENGTH_BED = DIAMETER_BED * l_d_ratio

    # 6. Vazão Volumétrica e Velocidade Superficial (para Modelagem)
    Q_VOLUMETRIC_IN = Q_MOLAR_MIX * R_UNIVERSAL * t_k / p_pa
    AREA_BED = np.pi * (DIAMETER_BED**2) / 4.0
    VELOCITY = Q_VOLUMETRIC_IN / AREA_BED

    return {
        "MASS_ADS_PER_BED_KG": MASS_ADS_PER_BED,
        "VOL_ADS_PER_BED_M3": VOL_ADS_PER_BED,
        "DIAMETER_BED_M": DIAMETER_BED,
        "LENGTH_BED_M": LENGTH_BED,
        "Q_MASS_H2O_KG_S": Q_MASS_H2O,
        "Q_VOLUMETRIC_IN_M3_S": Q_VOLUMETRIC_IN,
        "AREA_BED_M2": AREA_BED,
        "VELOCITY_M_S": VELOCITY,
        "MM_MIX": MM_MIX,
    }

# ==============================================================================
#                      PARTE 3: EXECUÇÃO E SAÍDA
# ==============================================================================

# Executa o dimensionamento para o Hidrogênio
results_h2 = dimensionar_tsa(
    Q_MASS_H2, MM_H2O, MM_H2, Y_H2O_IN, W_CAP, T_ADS_HR, RHO_ADS, L_D_RATIO, T_H2, P_H2
)

# Imprime os resultados
print("================================================================================")
print("             ✅ TSA - CÓDIGO DE DIMENSIONAMENTO ESTÁTICO (H₂)                    ")
print("================================================================================")
print(f"Dados de Entrada: T={T_H2_C:.2f} °C | P={P_H2_BAR:.2f} bar | Vazão Total={Q_MASS_H2} kg/s")
print(f"Vazão Mássica de H₂O a remover: {results_h2['Q_MASS_H2O_KG_S'] * 1000:.4f} g/s")
print("--------------------------------------------------------------------------------")
print(f"1. Massa Mínima de Adsorvente (por Leito): {results_h2['MASS_ADS_PER_BED_KG']:.1f} kg")
print(f"2. Volume de Adsorvente (por Leito): {results_h2['VOL_ADS_PER_BED_M3']:.3f} m³")
print("--------------------------------------------------------------------------------")
print(f"3. Diâmetro da Coluna (D): {results_h2['DIAMETER_BED_M']:.3f} m")
print(f"4. Comprimento do Leito (L): {results_h2['LENGTH_BED_M']:.3f} m")
print("--------------------------------------------------------------------------------")

# Salva os resultados importantes para o módulo de Modelagem
dimensionamento_output = {
    "D_M": results_h2['DIAMETER_BED_M'],
    "L_M": results_h2['LENGTH_BED_M'],
    "MASS_ADS_KG": results_h2['MASS_ADS_PER_BED_KG'],
    "RHO_ADS": RHO_ADS,
    "EPSILON_BED": EPSILON_BED,
    "T_ADS_HR": T_ADS_HR,
    "Q_MASS_H2": Q_MASS_H2,
    "Y_H2O_IN": Y_H2O_IN,
}

# print(f"\n# Dados exportados para o módulo de modelagem: {dimensionamento_output}")

# Exemplo de saída para o fluxo de O2 (apenas para comparação)
# Q_MASS_O2 = 0.20053  # kg/s
# Y_H2O_O2_IN = 0.000205
# MM_O2 = 31.999e-3  # kg/mol
# results_o2 = dimensionar_tsa(
#     Q_MASS_O2, MM_H2O, MM_O2, Y_H2O_O2_IN, W_CAP, T_ADS_HR, RHO_ADS, L_D_RATIO, T_H2, P_H2
# )
# print(f"Massa de Adsorvente O₂: {results_o2['MASS_ADS_PER_BED_KG']:.1f} kg")