import CoolProp.CoolProp as CP
import numpy as np
import pandas as pd # Adicionado pandas para consistência, se necessário para outras partes do modelo.

# Fluido de Exemplo (Gás)
# NOTA: O fluido será passado como argumento na simulação (e.g., 'hydrogen', 'oxygen')
FLUIDO_PADRAO = 'Nitrogen' 

# Constantes de eficiência (usadas se não forem passadas como argumentos default)
ETA_IS_DEFAULT = 0.65  # <--- ALTERADO DE 0.82 PARA 0.65 (65%)
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
    Calcula o estado de saída (T, P, h) e a potência de trabalho (W_dot) para um compressor de gás
    usando o modelo de Eficiência Isoentrópica (isentrópico/adiabático).
    
    A lógica é baseada no modo 'Entrada Conhecida' (Estado 1 -> Estado 2).
    """
    T1_C = T_in_C
    T1_K = T1_C + 273.15
    
    # 🛑 REMOVIDO BLOCO TRY/EXCEPT COM FALLBACK DE ERRO
    
    # 1. Estado de Entrada (Estado 1)
    # Assumindo que o gás é o fluido dominante, as propriedades são baseadas no gás puro
    H1 = CP.PropsSI('H', 'T', T1_K, 'P', P_in_Pa, fluido_nome)
    S1 = CP.PropsSI('S', 'T', T1_K, 'P', P_in_Pa, fluido_nome)
    
    # 2. Estado de Saída Isentrópico (Estado 2s) - s2 = s1
    T2s_K = CP.PropsSI('T', 'S', S1, 'P', P_out_Pa, fluido_nome)
    H2s = CP.PropsSI('H', 'S', S1, 'P', P_out_Pa, fluido_nome)
    
    T2s_C = T2s_K - 273.15
    
    # 3. Trabalho Isentrópico (Ideal)
    W_is_dot = m_dot_gas_kg_s * (H2s - H1)
    
    # 4. Trabalho Real
    # W_real = W_is / Eta_is. Note que W_is é (H2s - H1), que é positivo para compressão.
    H2_real = H1 + (H2s - H1) / Eta_is
    
    # 5. Estado de Saída Real (Estado 2)
    T2_K = CP.PropsSI('T', 'H', H2_real, 'P', P_out_Pa, fluido_nome)
    T2_C = T2_K - 273.15
    
    # 6. Potência de Trabalho Real e Elétrica
    # W_real_dot = m_dot * (H2_real - H1) = m_dot * (H2s - H1) / Eta_is = W_is_dot / Eta_is
    W_real_dot = W_is_dot / Eta_is
    
    # Potência do Eixo (Mecânica)
    Potencia_do_Eixo_W = W_real_dot / Eta_m
    
    # Potência Elétrica
    # Potencia_Eletrica_W = Potencia_do_Eixo_W / Eta_el
    
    
    return {
        'T_C': T2_C,
        'P_bar': P_out_Pa / 1e5,
        # Retorna a potência requerida do compressor (W_eixo)
        'W_dot_comp_W': Potencia_do_Eixo_W, 
        'Q_dot_fluxo_W': 0.0, # A compressão é assumida como adiabática (Q=0)
        # 💥 NOVO CAMPO: Temperatura Isentrópica (Para referência de diagnóstico)
        'T_out_isentropic_C': T2s_C,
        'erro': None
    }

# O bloco abaixo não será executado quando importado pelo process_execution.py
if __name__ == "__main__":
    # Exemplo de teste não interativo:
    res = modelo_compressor_ideal(
        fluido_nome='hydrogen', 
        T_in_C=25.0, 
        P_in_Pa=1.0 * 1e5, 
        P_out_Pa=2.0 * 1e5, 
        m_dot_mix_kg_s=0.0416, 
        m_dot_gas_kg_s=0.0396,
        Eta_is=0.65 # Testando com a nova eficiência
    )
    print("Teste Compressor H2 (P=1 a 2 bar, T_in=25C, Eta=0.65):")
    print(f"T_out: {res['T_C']:.2f} °C | W_eixo: {res['W_dot_comp_W']:.2f} W | Erro: {res['erro']}")
