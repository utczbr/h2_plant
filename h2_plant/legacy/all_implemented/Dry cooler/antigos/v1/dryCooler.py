import numpy as np
import pandas as pd

def dry_cooler_model(gas_name, m_dot_g, T_g_in, T_g_out, P_g, T_a_in, U_value, m_dot_a=None):
    """
    Modela um dry cooler para resfriamento de gases (H2 ou O2),
    considerando T_a_in como a temperatura de projeto (pior cenário).
    """
    
    # === 1. PARÂMETROS FÍSICOS E DE PROJETO ===
    c_p_H2 = 14300 # J/(kg.K) 
    c_p_O2 = 918   # J/(kg.K)
    
    c_p_g = c_p_H2 if gas_name == 'H2' else c_p_O2
    c_p_a = 1005   # J/(kg.K) (Ar)
    
    rho_a = 1.15    # kg/m3 (Densidade média do ar)
    delta_P_a = 500 # Pa (Queda de Pressão do ar, estimado)
    eta_fan = 0.65  # Eficiência do ventilador
    U = U_value 
    F = 0.85        # Fator de Correção LMTD (Estimativa p/ Fluxo Cruzado)
    
    # === 2. CÁLCULO DA CARGA DE CALOR (Q_dot) ===
    Q_dot = m_dot_g * c_p_g * (T_g_in - T_g_out)
    
    # === 3. VAZÃO DE AR E T_a_out ===
    if m_dot_a is None:
        delta_T_a_proj = 20 # K
        if Q_dot <= 0:
            return {"erro": "Carga térmica (Q_dot) inválida ou zero."}
        m_dot_a = Q_dot / (c_p_a * delta_T_a_proj)
        
    T_a_out = T_a_in + Q_dot / (m_dot_a * c_p_a)
    
    # === 4. CÁLCULO LMTD ===
    Delta_T1 = T_g_in - T_a_out
    Delta_T2 = T_g_out - T_a_in
    
    if Delta_T1 <= 0 or Delta_T2 <= 0:
        return {"erro": "Pinch Point/Impossível. T_g_out é menor ou igual a T_a_in ou T_g_in é menor ou igual a T_a_out."}
    
    Delta_T_log = (Delta_T1 - Delta_T2) / np.log(Delta_T1 / Delta_T2)
    LMTD = F * Delta_T_log
    
    # === 5. CÁLCULO DA ÁREA (A) ===
    Area_m2 = Q_dot / (U * LMTD)
    
    # === 6. CÁLCULO DA ENERGIA (Ventilador) ===
    V_dot_a = m_dot_a / rho_a 
    W_fan_watts = (V_dot_a * delta_P_a) / eta_fan
    
    # Retorna resultados formatados para a tabela vertical
    results = {
        "Calor Específico (J/kg.K)": round(c_p_g),
        "Carga Térmica (kW)": round(Q_dot / 1000, 2),
        "Vazão Mássica do Ar (kg/s)": round(m_dot_a, 3),
        "T Saída Ar (°C)": round(T_a_out, 1),
        "LMTD Corrigido (°C)": round(LMTD, 2),
        "Área Mínima (m²)": round(Area_m2, 2), 
        "Potência Máx. Fan (kW)": round(W_fan_watts / 1000, 3)
    }
    
    return results

def display_inputs(T_g_in, T_g_out, P_g, T_a_in, U_value, m_dot_g):
    """Formata e exibe os parâmetros de entrada em uma tabela vertical."""
    
    # Parâmetros adicionais fixos no modelo (para transparência)
    delta_P_a = 500  # Pa
    eta_fan = 0.65   # -
    
    data = {
        "Parâmetro de Entrada": [
            "Vazão Mássica Gases (kg/s)", 
            "T Entrada Gás (°C)", 
            "T Saída Gás (Meta) (°C)", 
            "Pressão Gás (bar)", 
            "T Entrada Ar (Projeto) (°C)",
            "Coef. Global U (W/m².K)",
            "Queda de Pressão Ar (Pa) [Est.]",
            "Eficiência do Ventilador [-] [Est.]"
        ],
        "Valor": [
            m_dot_g, 
            T_g_in, 
            T_g_out, 
            P_g, 
            T_a_in, 
            U_value,
            delta_P_a,
            eta_fan
        ]
    }
    
    df = pd.DataFrame(data)
    
    print("\n" + "="*55)
    print("      🧾 Parâmetros de Entrada da Simulação     ")
    print("="*55)
    print(df.to_string(index=False))
    print("="*55)


def display_results_vertical(results_h2, results_o2):
    """Formata e exibe os resultados em uma tabela vertical (transposta) usando pandas."""
    
    # Cria um DataFrame a partir dos resultados
    df = pd.DataFrame({
        "Hidrogênio (H2)": results_h2, 
        "Oxigênio (O2)": results_o2
    }).T # Transpõe o DataFrame (linhas viram colunas e vice-versa)

    # Reverte a transposição para ter os parâmetros nas linhas
    df = df.T
    
    print("\n" + "="*80)
    print("        📊 Resultados do Modelo Dry Cooler (Tabela Vertical)         ")
    print("="*80)
    
    df.index.name = 'Parâmetros de Saída'
    print(df.to_string())
    print("="*80)

# =================================================================
# === EXECUÇÃO DO MODELO COM OS DADOS FORNECIDOS ===
# =================================================================

# Parâmetros Comuns (Pior Cenário de Projeto)
T_g_in_proj = 80        # C
T_g_out_proj = 40       # C (Meta de resfriamento)
P_g_proj = 40           # bar
T_pico_enschede = 32    # C (T_a_in: Temperatura de projeto, Pior Cenário)
U_referencia = 35       # W/m2.K
m_dot_g_proj = 0.2      # kg/s (Vazão Mássica de exemplo para ambos)

print("Iniciando a simulação do modelo...")

# Exibe a tabela de inputs primeiro
display_inputs(
    T_g_in=T_g_in_proj,
    T_g_out=T_g_out_proj,
    P_g=P_g_proj,
    T_a_in=T_pico_enschede,
    U_value=U_referencia,
    m_dot_g=m_dot_g_proj
)

# 1. Simulação para Hidrogênio (H2)
resultados_h2 = dry_cooler_model(
    gas_name='H2', 
    m_dot_g=m_dot_g_proj, 
    T_g_in=T_g_in_proj, 
    T_g_out=T_g_out_proj, 
    P_g=P_g_proj, 
    T_a_in=T_pico_enschede, 
    U_value=U_referencia
)

# 2. Simulação para Oxigênio (O2)
resultados_o2 = dry_cooler_model(
    gas_name='O2', 
    m_dot_g=m_dot_g_proj, 
    T_g_in=T_g_in_proj, 
    T_g_out=T_g_out_proj, 
    P_g=P_g_proj, 
    T_a_in=T_pico_enschede, 
    U_value=U_referencia
)

# Verifica se houve erros antes de exibir a tabela de resultados
if isinstance(resultados_h2, dict) and "erro" in resultados_h2:
    print(f"\nErro no cálculo do H2: {resultados_h2['erro']}")
elif isinstance(resultados_o2, dict) and "erro" in resultados_o2:
    print(f"\nErro no cálculo do O2: {resultados_o2['erro']}")
else:
    # Exibe os resultados em formato de tabela vertical
    display_results_vertical(resultados_h2, resultados_o2)

# FIM DO CÓDIGO