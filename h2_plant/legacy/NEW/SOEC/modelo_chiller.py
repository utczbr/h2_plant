import numpy as np

def modelar_chiller_gas(gas_fluido: str, m_dot_mix_kg_s: float, P_in_bar: float, T_in_C: float, T_out_C_desejada: float, COP_chiller: float = 4.0, Delta_P_estimado: float = 0.2, H_in_J_kg: float = 0.0, H_out_J_kg: float = 0.0, y_H2O_in: float = 0.0) -> dict:
    """
    Modela o chiller para um fluxo de gás (O2 ou H2), calculando a carga térmica pela diferença de entalpia.
    
    H_in_J_kg e H_out_J_kg são as Entalpias TOTAIS do fluxo (J/s), calculadas pelo sistema central,
    incluindo calor sensível, latente e capacidade térmica do líquido acompanhante (se presente).
    """

    # --- 1. Cálculo da Carga Térmica do Chiller (Q_dot) ---
    
    # Q_dot = H_out - H_in (Se Q_dot < 0, calor é removido do fluido)
    # 🛑 REMOVIDO FALLBACK: O sistema deve fornecer as entalpias corretas
    Q_dot_CHILLER = H_out_J_kg - H_in_J_kg
    

    # --- 2. Cálculo do Consumo Energético do Chiller ---
    # W_dot_eletrico = |Q_dot| / COP_chiller
    if COP_chiller > 0:
        W_dot_eletrico = abs(Q_dot_CHILLER) / COP_chiller
    else:
        W_dot_eletrico = 0.0

    # --- 3. Cálculo da Pressão de Saída ---
    P_out_bar = P_in_bar - Delta_P_estimado

    # --- 4. Dicionário de Saída Padronizado (Inclui as chaves de água e vazão) ---
    results = {
        "T_C": T_out_C_desejada,
        "P_bar": P_out_bar,
        # Q_dot_fluxo_W é o calor removido do fluxo (Negativo)
        "Q_dot_fluxo_W": Q_dot_CHILLER, 
        "W_dot_comp_W": W_dot_eletrico,
        
        # CHAVES NECESSÁRIAS PARA O KOD/SISTEMA CENTRAL (Pass-Through)
        "y_H2O_out_vap": y_H2O_in,
        "m_dot_gas_out_princ": m_dot_mix_kg_s # Usando mix como placeholder, o sistema central recalcula no estado_atual
    }

    return results

if __name__ == '__main__':
    # Exemplo de Teste Unitário foi removido para evitar execução desnecessária
    pass