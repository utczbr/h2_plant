import CoolProp.CoolProp as CP
import numpy as np
import pandas as pd

# Fluido de Exemplo (Gás)
FLUIDO_PADRAO = 'Nitrogen' 

# Constantes de eficiência conforme solicitado
ETA_IS_DEFAULT = 0.75  
ETA_M_DEFAULT = 0.96
ETA_EL_DEFAULT = 0.93

def modelo_compressor_ideal(
    fluido_nome: str, 
    T_in_C: float, 
    P_in_Pa: float, 
    P_out_Pa: float, 
    m_dot_mix_kg_s: float, 
    m_dot_gas_kg_s: float,
    Eta_is: float = ETA_IS_DEFAULT,
    Eta_m: float = ETA_M_DEFAULT,
    Eta_el: float = ETA_EL_DEFAULT
) -> dict:
    """
    Calcula o estado de saída (T, P, h) e a potência de trabalho (W_dot) para um compressor de gás.
    """
    T1_K = T_in_C + 273.15
    
    # 1. Estado de Entrada
    H1 = CP.PropsSI('H', 'T', T1_K, 'P', P_in_Pa, fluido_nome)
    S1 = CP.PropsSI('S', 'T', T1_K, 'P', P_in_Pa, fluido_nome)
    
    # 2. Estado de Saída Isentrópico
    try:
        T2s_K = CP.PropsSI('T', 'S', S1, 'P', P_out_Pa, fluido_nome)
        H2s = CP.PropsSI('H', 'S', S1, 'P', P_out_Pa, fluido_nome)
    except:
        # Fallback simples para gases ideais caso o CoolProp falhe em pontos críticos
        gamma = 1.41 # Para H2/Diatômicos
        T2s_K = T1_K * (P_out_Pa / P_in_Pa)**((gamma-1)/gamma)
        cp_approx = CP.PropsSI('C', 'T', T1_K, 'P', P_in_Pa, fluido_nome)
        H2s = H1 + cp_approx * (T2s_K - T1_K)

    # 3. Trabalho Isentrópico
    W_is_dot = m_dot_gas_kg_s * (H2s - H1)
    
    # 4. Trabalho Real e Entalpia de Saída
    H2_real = H1 + (H2s - H1) / Eta_is
    
    # 5. Estado de Saída Real
    T2_K = CP.PropsSI('T', 'H', H2_real, 'P', P_out_Pa, fluido_nome)
    T2_C = T2_K - 273.15
    
    # 6. Potências
    W_real_dot = W_is_dot / Eta_is
    Potencia_do_Eixo_W = W_real_dot / Eta_m
    Potencia_Eletrica_W = Potencia_do_Eixo_W / Eta_el
    
    return {
        'T_C': T2_C,
        'P_bar': P_out_Pa / 1e5,
        'W_dot_comp_W': Potencia_do_Eixo_W,
        'W_dot_el_W': Potencia_Eletrica_W,
        'T_out_isentropic_C': T2s_K - 273.15,
        'erro': None
    }