import numpy as np

def modelar_chiller_gas(
    gas_fluido: str, 
    m_dot_mix_kg_s: float, 
    P_in_bar: float, 
    T_in_C: float, 
    T_out_C_desejada: float, 
    COP_chiller: float = 4.0, 
    Delta_P_estimado: float = 0.2, 
    H_in_J_kg: float = 0.0, 
    H_out_J_kg: float = 0.0, 
    y_H2O_in: float = 0.0
) -> dict:
    """
    Modela o chiller para um fluxo de gás sem qualquer lógica de fallback.
    
    ESTA FUNÇÃO EXIGE ENTALPIAS EXPLÍCITAS. 
    Se as entalpias não forem fornecidas pelo sistema central, o código 
    gerará um erro fatal para evitar resultados mascarados.
    """

    # --- VERIFICAÇÃO DE ERRO (SUBSTITUI O FALLBACK) ---
    if H_in_J_kg == 0.0 and H_out_J_kg == 0.0:
        raise ValueError(
            f"ERRO CRÍTICO NO CHILLER ({gas_fluido}): Entalpias não fornecidas. "
            "O cálculo de fallback foi removido conforme solicitado. "
            "Verifique o cálculo de Delta H no script principal."
        )

    # --- 1. Cálculo da Carga Térmica (Q_dot) ---
    # Q = H_saída - H_entrada (Em Watts, já que H é J/s)
    Q_dot_CHILLER = H_out_J_kg - H_in_J_kg
    
    # --- 2. Cálculo do Consumo Energético ---
    if COP_chiller <= 0:
        raise ValueError("ERRO: O COP do Chiller deve ser maior que zero para evitar divisão por zero.")

    # Potência elétrica consumida (W)
    W_dot_eletrico = abs(Q_dot_CHILLER) / COP_chiller

    # --- 3. Cálculo da Pressão de Saída ---
    P_out_bar = P_in_bar - Delta_P_estimado

    # --- 4. Retorno de Resultados ---
    return {
        "T_C": T_out_C_desejada,
        "P_bar": P_out_bar,
        "Q_dot_fluxo_W": Q_dot_CHILLER,
        "W_dot_comp_W": W_dot_eletrico,
        "m_dot_kg_s": m_dot_mix_kg_s,
        "Status": "Cálculo Rigoroso (Sem Fallback)"
    }