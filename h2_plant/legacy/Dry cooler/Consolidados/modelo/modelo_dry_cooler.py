import numpy as np
import sys 
import CoolProp.CoolProp as CP 

# =================================================================
# === PARÂMETROS DE PROJETO (FIXOS DO DIMENSIONAMENTO) ===
# ESTES VALORES FORAM ATUALIZADOS COM O NOVO DIMENSIONAMENTO (Com Água)
# =================================================================
# Variáveis de Dimensionamento:
AREA_H2_DESIGN = 219.0      # m² (Novo Valor: 219.0)
AREA_O2_DESIGN = 24251.05   # m² (Novo Valor: 24251.05)
U_VALUE_DESIGN = 35         # W/m².K (Mantido)
DP_AIR_DESIGN = 500         # Pa (Mantido)

# Parâmetros de Operação Padrão do Ar (para simulação)
T_A_IN_OP = 20              # °C (Temperatura de entrada do ar para simulação)
M_DOT_A_H2_DESIGN = 5.175   # kg/s (Nova Vazão de Ar Design H2: 5.175)
M_DOT_A_O2_DESIGN = 573.037 # kg/s (Nova Vazão de Ar Design O2: 573.037)
P_PERDA_BAR = 0.05          # bar (Queda de pressão estimada no Dry Cooler)


# =================================================================
# === FUNÇÕES DE CÁLCULO GERAIS ===
# =================================================================

def get_gas_cp_and_liquid_cp(gas_name):
    """Retorna o calor específico (cp) do gás e o cp da água líquida em J/(kg.K)."""
    # Usando valores de referência/CoolProp
    try:
        # Cp Mássico (J/kg.K)
        c_p_H2_gas = CP.PropsSI('CPMASS', 'T', 300, 'P', 1e5, 'H2') 
        c_p_O2_gas = CP.PropsSI('CPMASS', 'T', 300, 'P', 1e5, 'O2') 
        c_p_H2O_liq = CP.PropsSI('CPMASS', 'T', 300, 'P', 1e5, 'Water')
    except:
        # Fallback para valores simplificados
        c_p_H2_gas = 14300 # J/(kg.K)
        c_p_O2_gas = 918   # J/(kg.K)
        c_p_H2O_liq = 4186 # J/(kg.K)
        
    c_p_g = c_p_H2_gas if gas_name == 'H2' else c_p_O2_gas
    return c_p_g, c_p_H2O_liq

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

def modelar_dry_cooler(gas_fluido: str, m_dot_mix_kg_s: float, m_dot_H2O_liq_kg_s: float, P_in_bar: float, T_g_in_C: float, m_dot_a_op: float = None) -> dict:
    """
    Modelagem do Dry Cooler, calculando a carga térmica usando a capacidade
    térmica total (Gás + Líquido).
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
    
    # NOVAS CAPACIDADES DE CALOR ESPECÍFICO (Gás e Líquido)
    c_p_g, c_p_H2O_liq = get_gas_cp_and_liquid_cp(gas_fluido)
    c_p_a = 1005.0 # J/(kg.K)
    
    # 2. NTU-Eficácia (Cálculo da Carga Térmica Total do Fluido Quente)
    
    # Vazão mássica da fase gasosa (Gás Principal + Vapor H2O)
    m_dot_gas_fase = m_dot_mix_kg_s - m_dot_H2O_liq_kg_s
    
    # Capacidade Térmica Total do Fluxo Quente (Gás + Líquido)
    C_gas_mix = m_dot_gas_fase * c_p_g # Usa cp do gas principal/mix como aproximação para a fase gasosa
    C_liquido = m_dot_H2O_liq_kg_s * c_p_H2O_liq 
    C_quente = C_gas_mix + C_liquido
    
    C_a = m_dot_a_op * c_p_a
    
    # Cálculo NTU/Eficácia
    C_min = min(C_quente, C_a)
    C_max = max(C_quente, C_a)
    
    if C_min <= 0:
         # O NOVO DIMENSIONAMENTO DEVE TER ELIMINADO ESTE ERRO
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
    
    # A nova temperatura de saída do fluido quente (Gás + Líquido)
    T_g_out_K = T_g_in_K - Q_dot_real_W / C_quente
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
    m_dot_h2_mix = 0.51972 # Exemplo
    m_dot_h2_liq = 0.49500 # Exemplo
    m_dot_o2_mix = 69.00 # Exemplo
    m_dot_o2_liq = 68.72 # Exemplo
    
    # Testa H2
    res_h2 = modelar_dry_cooler('H2', m_dot_h2_mix, m_dot_h2_liq, P_in_bar, T_in_C)
    print("--- Teste Unitário Dry Cooler (H2 Corrigido) ---")
    print(f"H2 Saída: T={res_h2['T_C']:.2f}C, P={res_h2['P_bar']:.2f}bar, Q={res_h2['Q_dot_fluxo_W']/1000:.2f}kW, W={res_h2['W_dot_comp_W']:.2f}W")
    
    # Testa O2
    res_o2 = modelar_dry_cooler('O2', m_dot_o2_mix, m_dot_o2_liq, P_in_bar, T_in_C)
    print("--- Teste Unitário Dry Cooler (O2 Corrigido) ---")
    print(f"O2 Saída: T={res_o2['T_C']:.2f}C, P={res_o2['P_bar']:.2f}bar, Q={res_o2['Q_dot_fluxo_W']/1000:.2f}kW, W={res_o2['W_dot_comp_W']:.2f}W")