import numpy as np
import pandas as pd
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

# Parâmetros Invariáveis do Processo:
T_G_IN_PROJ = 80            # °C (Temperatura de entrada do gás)
M_DOT_G_H2 = 0.02472        # kg/s (Vazão mássica máxima de H2)
M_DOT_G_O2 = 0.19778        # kg/s (Vazão mássica máxima de O2)
# Vazões de Ar calculadas no dimensionamento (T_a_in=32C, T_g_out=40C, dT_ar=20K)
M_DOT_A_H2_DESIGN = 0.704   # kg/s (Vazão Máxima de Ar para H2)
M_DOT_A_O2_DESIGN = 0.361   # kg/s (Vazão Máxima de Ar para O2)


# =================================================================
# === FUNÇÕES DE CÁLCULO GERAIS ===
# =================================================================

def get_gas_cp(gas_name):
    """Retorna o calor específico (cp) do gás em J/(kg.K) (Valores de Referência)."""
    c_p_H2 = 14300 
    c_p_O2 = 918   
    return c_p_H2 if gas_name == 'H2' else c_p_O2

# =================================================================
# === FUNÇÃO DE MODELAGEM (SIMULAÇÃO DE DESEMPENHO) ===
# =================================================================

def cooler_modelagem_desempenho(gas_name, m_dot_g, T_g_in, T_a_in_op, Area_m2, U_value, m_dot_a_op):
    """
    Calcula a T_g_out sob condições de operação (op) usando a Área fixada.
    Utiliza o método da Eficácia - NTU.
    """
    
    # Constantes do Modelo
    c_p_g = get_gas_cp(gas_name)
    c_p_a = 1005.0
    U = U_value
    
    # 1. Capacidades Térmicas
    C_g = m_dot_g * c_p_g
    C_a = m_dot_a_op * c_p_a
    
    # 2. C_min, C_max e R (Razão de Capacidade)
    C_min = min(C_g, C_a)
    C_max = max(C_g, C_a)
    
    if C_max == 0:
        return {"erro": "Vazão mássica do gás e do ar são nulas."}
        
    R = C_min / C_max
    
    # 3. NTU (Número de Unidades de Transferência)
    if C_min == 0:
         return {"erro": "C_min é zero. Vazão mássica ou cp inválido."}
         
    NTU = U * Area_m2 / C_min
    
    # 4. Eficácia (E) (Aproximação de Contracorrente para conservação)
    # Fórmula usada: E = (1 - exp(-NTU * (1 - R))) / (1 - R * exp(-NTU * (1 - R)))
    if R == 1:
        E = NTU / (1 + NTU)
    else:
        try:
            E = (1 - np.exp(-NTU * (1 - R))) / (1 - R * np.exp(-NTU * (1 - R)))
        except OverflowError:
            # Caso os termos exponenciais sejam muito grandes ou pequenos
            E = 1.0 # Máxima eficácia
    
    # 5. Carga de Calor Máxima e Real (Q_dot_real)
    Q_max = C_min * (T_g_in - T_a_in_op)
    Q_dot_real = E * Q_max
    
    # 6. CÁLCULO DAS TEMPERATURAS DE SAÍDA
    T_g_out_calc = T_g_in - Q_dot_real / C_g
    
    results = {
        "Gás": gas_name,
        "Área Utilizada (m²)": round(Area_m2, 2),
        "Vazão Mássica Gás (kg/s)": round(m_dot_g, 5),
        "T Entrada Ar Op. (°C)": T_a_in_op,
        "Vazão Ar Op. (kg/s)": m_dot_a_op,
        "T Saída Gás (°C)": round(T_g_out_calc, 2),
        "Eficácia (E)": round(E, 3),
        "NTU": round(NTU, 2),
        "Q Real (kW)": round(Q_dot_real / 1000, 2)
    }
    
    return results

# =================================================================
# === FUNÇÕES DE EXIBIÇÃO DE TABELAS ===
# =================================================================

def display_results_vertical(title, results_h2, results_o2):
    """Exibe os resultados em uma tabela vertical (transposta) usando pandas."""
    # Garante que resultados incompletos ou erros não quebrem a tabela
    if isinstance(results_h2, dict) and "erro" in results_h2: 
        results_h2 = {k: results_h2["erro"] for k in results_h2.keys()}
    if isinstance(results_o2, dict) and "erro" in results_o2: 
        results_o2 = {k: results_o2["erro"] for k in results_o2.keys()}
        
    results_h2.pop('Gás', None)
    results_o2.pop('Gás', None)
        
    df = pd.DataFrame({
        "Hidrogênio (H2)": results_h2, 
        "Oxigênio (O2)": results_o2
    }).T.T
    
    print("\n" + "="*80)
    print(f"        {title}         ")
    print("="*80)
    df.index.name = 'Parâmetros de Saída'
    print(df.to_string())
    print("="*80)

# =================================================================
# === EXECUÇÃO DA MODELAGEM ===
# =================================================================

if __name__ == '__main__':
    
    print("Iniciando a Modelagem de Desempenho (Simulação) do Dry Cooler...")
    
    # --- CENÁRIO 1: Dia Frio (T_a_in = 20C) com Vazão de Ar Total ---
    # O cooler opera com sua capacidade total de troca térmica.
    T_a_in_op_c1 = 20
    
    sim_h2_c1 = cooler_modelagem_desempenho(
        'H2', M_DOT_G_H2, T_G_IN_PROJ, T_a_in_op_c1, AREA_H2_DESIGN, U_VALUE_DESIGN, M_DOT_A_H2_DESIGN
    )
    sim_o2_c1 = cooler_modelagem_desempenho(
        'O2', M_DOT_G_O2, T_G_IN_PROJ, T_a_in_op_c1, AREA_O2_DESIGN, U_VALUE_DESIGN, M_DOT_A_O2_DESIGN
    )
    
    display_results_vertical(
        "1. Modelagem: Dia Frio (20°C) - Vazão de Ar Total (100%)", sim_h2_c1, sim_o2_c1
    )
    
    # --- CENÁRIO 2: Dia Frio (T_a_in = 20C) com Vazão de Ar Reduzida (50%) ---
    # Simula economia de energia (menor Potência do Ventilador).
    T_a_in_op_c2 = 20
    m_dot_a_op_h2_c2 = M_DOT_A_H2_DESIGN * 0.5 
    m_dot_a_op_o2_c2 = M_DOT_A_O2_DESIGN * 0.5
    
    sim_h2_c2 = cooler_modelagem_desempenho(
        'H2', M_DOT_G_H2, T_G_IN_PROJ, T_a_in_op_c2, AREA_H2_DESIGN, U_VALUE_DESIGN, m_dot_a_op_h2_c2
    )
    sim_o2_c2 = cooler_modelagem_desempenho(
        'O2', M_DOT_G_O2, T_G_IN_PROJ, T_a_in_op_c2, AREA_O2_DESIGN, U_VALUE_DESIGN, m_dot_a_op_o2_c2
    )
    
    display_results_vertical(
        "2. Modelagem: Dia Frio (20°C) - Vazão de Ar 50% (Economia de Energia)", sim_h2_c2, sim_o2_c2
    )