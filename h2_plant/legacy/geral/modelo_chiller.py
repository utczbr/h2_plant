import numpy as np

def modelar_chiller_gas(gas_fluido: str, m_dot_g_kg_s: float, P_in_bar: float, T_in_C: float, T_out_C_desejada: float, COP_chiller: float = 4.0, Delta_P_estimado: float = 0.2) -> dict:
    """
    Modela o chiller para um fluxo de gás (O2 ou H2).

    Dados de Entrada:
    - gas_fluido: 'O2' ou 'H2'
    - m_dot_g_kg_s: Vazão mássica do gás (kg/s)
    - P_in_bar: Pressão de entrada do gás no chiller (bar)
    - T_in_C: Temperatura de entrada no Chiller (°C)
    - T_out_C_desejada: Temperatura de saída desejada do chiller (°C)
    - COP_chiller: Coeficiente de Performance do chiller (adimensional)
    - Delta_P_estimado: Queda de pressão estimada no trocador de calor do chiller (bar)

    Retorno:
    - Dicionário com T_out, P_out, Carga Térmica (Q_dot) e Consumo Elétrico (W_dot).
    """

    # --- 1. Propriedades Termodinâmicas (Exemplo Simplificado - Usar CoolProp na função central para precisão) ---
    if gas_fluido == 'O2':
        Cp_avg = 918.0  # J/kg.C
    elif gas_fluido == 'H2':
        Cp_avg = 14300.0  # J/kg.C
    else:
        Cp_avg = 1000.0
        print(f"Aviso: Gás '{gas_fluido}' não mapeado. Usando Cp padrão.")


    # --- 2. Cálculo da Carga Térmica do Chiller (Q_dot) ---
    Delta_T_chiller = T_in_C - T_out_C_desejada
    
    # Q_dot_CHILLER em Watts (J/s)
    Q_dot_CHILLER = m_dot_g_kg_s * Cp_avg * Delta_T_chiller 

    # --- 3. Cálculo do Consumo Energético do Chiller ---
    if COP_chiller > 0:
        W_dot_eletrico = Q_dot_CHILLER / COP_chiller
    else:
        W_dot_eletrico = 0.0

    # --- 4. Cálculo da Pressão de Saída ---
    P_out_bar = P_in_bar - Delta_P_estimado

    # --- 5. Dicionário de Saída Padronizado ---
    results = {
        # Estado de Saída do Gás
        "T_C": T_out_C_desejada,
        "P_bar": P_out_bar,
        # Energia do Componente
        "Q_dot_fluxo_W": Q_dot_CHILLER * -1.0, # Negativo pois é calor removido do fluxo
        "W_dot_comp_W": W_dot_eletrico
    }

    return results

if __name__ == '__main__':
    # Exemplo de Teste
    T_in_C = 28.0
    P_in_bar = 39.95 # Saída do Dry Cooler
    m_dot_h2 = 0.02472
    m_dot_o2 = 0.19778
    T_out_desejada = 4.0
    
    print("--- Teste Unitário Chiller ---")
    res_h2 = modelar_chiller_gas('H2', m_dot_h2, P_in_bar, T_in_C, T_out_desejada)
    res_o2 = modelar_chiller_gas('O2', m_dot_o2, P_in_bar, T_in_C, T_out_desejada)
    print(f"H2 Saída: T={res_h2['T_C']:.2f}C, P={res_h2['P_bar']:.2f}bar, Q={res_h2['Q_dot_fluxo_W']/1000:.2f}kW, W={res_h2['W_dot_comp_W']:.2f}W")
    print(f"O2 Saída: T={res_o2['T_C']:.2f}C, P={res_o2['P_bar']:.2f}bar, Q={res_o2['Q_dot_fluxo_W']/1000:.2f}kW, W={res_o2['W_dot_comp_W']:.2f}W")