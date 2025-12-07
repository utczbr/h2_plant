import numpy as np
from tabulate import tabulate 

# A função assume que o Dry Cooler já reduziu a temperatura
# para a 'T_DC_out', que se torna o T_in do Chiller.

def modelar_chiller_gas(gas_name, m_dot, P_in_bar, T_DC_out_C, T_out_C, COP_chiller, Delta_P_estimado):
    """
    Modela o chiller para um fluxo de gás (O2 ou H2), considerando o pré-resfriamento por Dry Cooler.

    Dados de Entrada:
    - gas_name: 'O2' ou 'H2'
    - m_dot: Vazão mássica do gás (kg/s)
    - P_in_bar: Pressão de entrada do gás no chiller (bar)
    - T_DC_out_C: Temperatura de saída do Dry Cooler (Temperatura de entrada no Chiller) (°C)
    - T_out_C: Temperatura de saída desejada do chiller (°C)
    - COP_chiller: Coeficiente de Performance do chiller (adimensional)
    - Delta_P_estimado: Queda de pressão estimada no trocador de calor do chiller (bar)

    Dados de Saída:
    - estado_entrada: Dados de entrada do fluxo e chiller
    - performance_chiller: Carga térmica e consumo energético do chiller
    - estado_saida: Estado final do fluxo de gás
    """

    # --- 1. Propriedades Termodinâmicas (Exemplo Simplificado) ---
    if gas_name == 'O2':
        # Calor específico do O2 a alta pressão
        Cp_avg = 1.05  # kJ/kg.C
    elif gas_name == 'H2':
        # Calor específico do H2 a alta pressão
        Cp_avg = 14.5  # kJ/kg.C
    else:
        Cp_avg = 1.05 
        print(f"Aviso: Gás '{gas_name}' não mapeado. Usando Cp padrão.")


    # --- 2. Cálculo da Carga Térmica do Chiller (Q_dot) ---
    Delta_T_chiller = T_DC_out_C - T_out_C
    
    Q_dot_CHILLER = m_dot * Cp_avg * Delta_T_chiller

    # --- 3. Cálculo do Consumo Energético do Chiller ---
    if COP_chiller > 0:
        W_dot_eletrico = Q_dot_CHILLER / COP_chiller
    else:
        W_dot_eletrico = 0.0

    # --- 4. Cálculo da Pressão de Saída ---
    P_out_bar = P_in_bar - Delta_P_estimado

    # --- 5. Fluxos para a Troca (Dimensionamento do Fluido Secundário) ---
    Cp_agua = 4.18  # kJ/kg.C (Água)
    Delta_T_agua = 5.0 # Delta T do fluido secundário
    
    if Cp_agua * Delta_T_agua > 0:
        m_dot_secundario = Q_dot_CHILLER / (Cp_agua * Delta_T_agua)
    else:
        m_dot_secundario = 0.0

    # --- 6. Dados de Saída (Organizados) ---
    
    # 6.1. DADOS DE ENTRADA
    estado_entrada = [
        ["Gás Analisado", gas_name],
        ["Vazão Mássica do Gás (kg/s)", f"{m_dot:.3f}"],
        ["Cp Médio do Gás (kJ/kg.C)", f"{Cp_avg:.2f}"],
        ["P de Entrada no Chiller (bar)", f"{P_in_bar:.2f}"],
        ["T de Entrada no Chiller (C)", f"{T_DC_out_C:.1f}"],
        ["T de Saída Desejada (C)", f"{T_out_C:.1f}"]
    ]

    # 6.2. DADOS DO CHILLER E TROCA
    performance_chiller = [
        ["Carga Térmica do Chiller (kW)", f"{Q_dot_CHILLER:.2f}"],
        ["COP (Coef. de Performance)", f"{COP_chiller:.2f}"],
        ["Consumo Elétrico (kW)", f"{W_dot_eletrico:.2f}"],
        ["Fluido Secundário", "Água/Glicol (Exemplo)"],
        ["Vazão Mássica Secundária (kg/s)", f"{m_dot_secundario:.3f}"],
        ["Delta T Secundário (C)", f"{Delta_T_agua:.1f}"]
    ]

    # 6.3. DADOS DE SAÍDA DO GÁS
    estado_saida = [
        ["T de Saída do Gás (C)", f"{T_out_C:.1f}"],
        ["Queda de Pressão Estimada (bar)", f"{Delta_P_estimado:.2f}"],
        ["P de Saída do Gás (bar)", f"{P_out_bar:.2f}"]
    ]

    return estado_entrada, performance_chiller, estado_saida

# --- Exemplo de Uso ---

# Parametros de Entrada (O2)
m_dot_O2 = 0.5       # kg/s
T_DC_out = 28.0      # C (Saída do Dry Cooler, Entrada do Chiller)
P_in = 40.0          # bar
T_out = 4.0          # C (Saída desejada)
COP_chiller = 4.0
Delta_P_estimado = 0.2 # bar

# Chamada para O2
# As variáveis de retorno foram renomeadas de 'estado_O2, consumo_O2, troca_O2'
# para 'entrada_O2, performance_O2, saida_O2' para refletir a nova organização.
entrada_O2, performance_O2, saida_O2 = modelar_chiller_gas('O2', m_dot_O2, P_in, T_DC_out, T_out, COP_chiller, Delta_P_estimado)

print("\n--- Resultados de Modelagem do CHILLER (O2) ---")

# Tabela 1: Dados de Entrada
print("\n## 1. DADOS DE ENTRADA DO FLUXO")
print(tabulate(entrada_O2, headers=["Parâmetro", "Valor"], tablefmt="fancy_grid"))

# Tabela 2: Dados de Performance do Chiller e Troca
print("\n## 2. PERFORMANCE DO CHILLER E FLUXOS DE TROCA")
print(tabulate(performance_O2, headers=["Parâmetro", "Valor"], tablefmt="fancy_grid"))

# Tabela 3: Dados de Saída do Gás (Estado Final)
print("\n## 3. ESTADO DE SAÍDA DO GÁS")
print(tabulate(saida_O2, headers=["Parâmetro", "Valor"], tablefmt="fancy_grid"))