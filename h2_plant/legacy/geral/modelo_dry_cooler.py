import numpy as np
import sys 

# =================================================================
# === PARÂMETROS DE PROJETO (FIXOS DO DIMENSIONAMENTO) ===
# Estes valores foram obtidos na execução do 'dimensionamento.py'
# =================================================================
# Variáveis de Dimensionamento:
AREA_H2_DESIGN = 29.77      # m²
AREA_O2_DESIGN = 15.29      # m²
U_VALUE_DESIGN = 35         # W/m².K
DP_AIR_DESIGN = 500         # Pa

# Parâmetros de Operação Padrão do Ar (para simulação)
T_A_IN_OP = 20              # °C (Temperatura de entrada do ar para simulação)
M_DOT_A_H2_DESIGN = 0.704   # kg/s (Vazão Máxima de Ar para H2)
M_DOT_A_O2_DESIGN = 0.361   # kg/s (Vazão Máxima de Ar para O2)
P_PERDA_BAR = 0.05          # bar (Queda de pressão estimada no Dry Cooler)


# =================================================================
# === FUNÇÕES DE CÁLCULO GERAIS ===
# =================================================================

def get_gas_cp(gas_name):
    """Retorna o calor específico (cp) do gás em J/(kg.K) (Valores de Referência)."""
    # Valores de referência simplificados, o CoolProp será usado para entalpia na função central.
    c_p_H2 = 14300 
    c_p_O2 = 918   
    return c_p_H2 if gas_name == 'H2' else c_p_O2

def calcular_potencia_ventilador(m_dot_a, DP_air):
    """Calcula a potência elétrica do ventilador (W), assumindo eficiência de 60%."""
    rho_ar = 1.225 # kg/m³ (Densidade do ar a 20C, 1 atm)
    V_dot_a = m_dot_a / rho_ar
    P_ventilador_mecanica = V_dot_a * DP_air
    eficiencia = 0.60
    return P_ventilador_mecanica / eficiencia


# =================================================================
# === FUNÇÃO DE MODELAGEM (SIMULAÇÃO DE DESEMPENHO) ===
# =================================================================

def modelar_dry_cooler(gas_fluido: str, m_dot_g_kg_s: float, P_in_bar: float, T_g_in_C: float, m_dot_a_op: float = None) -> dict:
    """
    Modelagem do Dry Cooler usando o método NTU-Eficácia.
    Retorna os parâmetros de estado de saída e as energias envolvidas.
    """
    
    # 1. Parâmetros do Componente
    if gas_fluido == 'H2':
        Area_m2 = AREA_H2_DESIGN
        if m_dot_a_op is None: m_dot_a_op = M_DOT_A_H2_DESIGN
    elif gas_fluido == 'O2':
        Area_m2 = AREA_O2_DESIGN
        if m_dot_a_op is None: m_dot_a_op = M_DOT_A_O2_DESIGN
    else:
        return {"erro": f"Gás {gas_fluido} não suportado."}
        
    U_value = U_VALUE_DESIGN
    T_a_in_op = T_A_IN_OP
    c_p_g = get_gas_cp(gas_fluido)
    c_p_a = 1005.0 # J/(kg.K)
    
    # 2. NTU-Eficácia
    C_g = m_dot_g_kg_s * c_p_g
    C_a = m_dot_a_op * c_p_a
    
    C_min = min(C_g, C_a)
    C_max = max(C_g, C_a)
    
    if C_min <= 0:
         return {"erro": "C_min é zero. Vazão mássica ou cp inválido."}
         
    R = C_min / C_max
    NTU = U_value * Area_m2 / C_min
    
    # Cálculo da Eficácia (Contracorrente para conservação)
    if R == 1:
        E = NTU / (1 + NTU)
    else:
        try:
            E = (1 - np.exp(-NTU * (1 - R))) / (1 - R * np.exp(-NTU * (1 - R)))
        except OverflowError:
            E = 1.0 
    
    # 3. CÁLCULO DE ENERGIA E TEMPERATURA DE SAÍDA
    T_g_in_K = T_g_in_C + 273.15
    T_a_in_K = T_a_in_op + 273.15

    Q_max = C_min * (T_g_in_K - T_a_in_K) # Q_max em Watts
    Q_dot_real_W = E * Q_max
    
    T_g_out_K = T_g_in_K - Q_dot_real_W / C_g
    T_g_out_C = T_g_out_K - 273.15
    
    # 4. Consumo Elétrico (Ventilador)
    W_dot_ventilador_W = calcular_potencia_ventilador(m_dot_a_op, DP_AIR_DESIGN)
    
    # 5. Pressão de Saída
    P_out_bar = P_in_bar - P_PERDA_BAR
    
    # 6. Dicionário de Saída Padronizado
    results = {
        # Estado de Saída do Gás
        "T_C": T_g_out_C,
        "P_bar": P_out_bar,
        # Energia do Componente
        "Q_dot_fluxo_W": Q_dot_real_W * -1.0, # Negativo pois é calor removido do fluxo
        "W_dot_comp_W": W_dot_ventilador_W
    }
    
    return results

if __name__ == '__main__':
    # Exemplo de Teste
    T_in_C = 80.0
    P_in_bar = 40.0
    m_dot_h2 = 0.02472
    m_dot_o2 = 0.19778
    
    print("--- Teste Unitário Dry Cooler ---")
    res_h2 = modelar_dry_cooler('H2', m_dot_h2, P_in_bar, T_in_C)
    res_o2 = modelar_dry_cooler('O2', m_dot_o2, P_in_bar, T_in_C)
    print(f"H2 Saída: T={res_h2['T_C']:.2f}C, P={res_h2['P_bar']:.2f}bar, Q={res_h2['Q_dot_fluxo_W']/1000:.2f}kW, W={res_h2['W_dot_comp_W']:.2f}W")
    print(f"O2 Saída: T={res_o2['T_C']:.2f}C, P={res_o2['P_bar']:.2f}bar, Q={res_o2['Q_dot_fluxo_W']/1000:.2f}kW, W={res_o2['W_dot_comp_W']:.2f}W")