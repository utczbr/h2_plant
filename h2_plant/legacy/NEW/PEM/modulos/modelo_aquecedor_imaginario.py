# modulos/modelo_aquecedor_imaginario.py
import numpy as np
import CoolProp.CoolProp as CP 
import sys

# Importa a função auxiliar para calcular o estado termodinâmico de saída
try:
    from aux_coolprop import calcular_estado_termodinamico
except ImportError:
    # Fallback simplificado se a importação falhar (para robustez)
    def calcular_estado_termodinamico(gas, T_C, P_bar, m_dot_gas, y_H2O, y_O2, y_H2):
        print("AVISO: Usando stub simplificado para CoolProp.")
        return {
            "T_C": T_C, "P_bar": P_bar, "y_H2O": y_H2O, "y_O2": y_O2, "y_H2": y_H2,
            "m_dot_gas_kg_s": m_dot_gas, "m_dot_mix_kg_s": m_dot_gas * 1.05, 
            "H_mix_J_kg": 1000.0 * T_C, # Estimativa muito grosseira
            "H_in_mix_J_kg": (m_dot_gas * 1.05) * 1000.0 * T_C,
            "m_dot_H2O_vap_kg_s": (m_dot_gas * 1.05) - m_dot_gas
        }


def modelar_aquecedor_imaginario(estado_in: dict, T_out_C_alvo: float = 40.0) -> dict:
    """
    Calcula a carga térmica (Q_dot) necessária para aquecer o fluxo de gás 
    do estado de entrada até a temperatura alvo (40 °C), mantendo a pressão constante.
    
    A água líquida acompanhante (m_dot_H2O_liq_accomp_kg_s) não é evaporada pelo Aquecedor.
    
    Args:
        estado_in (dict): Dicionário de estado termodinâmico do fluido de entrada.
        T_out_C_alvo (float): Temperatura de saída desejada (°C).
        
    Returns:
        dict: Resultados da simulação do aquecedor, incluindo T, P e Q_dot.
    """
    
    gas_fluido = estado_in['gas_fluido']
    P_in_bar = estado_in['P_bar']
    T_in_C = estado_in['T_C']
    
    # Se a temperatura de entrada já for igual ou superior ao alvo, não é necessário aquecimento.
    if T_in_C >= T_out_C_alvo:
        return {
            "T_C": T_in_C,
            "P_bar": P_in_bar,
            "Q_dot_fluxo_W": 0.0,
            "W_dot_comp_W": 0.0,
            "Agua_Condensada_kg_s": 0.0,
            "m_dot_H2O_liq_accomp_out_kg_s": estado_in['m_dot_H2O_liq_accomp_kg_s'],
            "erro": "Temperatura de entrada já é >= alvo. Q_dot = 0."
        }

    # 1. ESTADO DE SAÍDA ALVO (P constante, T_alvo)
    # Recalculamos o estado termodinâmico para a temperatura alvo.
    # Assumimos que o y_H2O (vapor) não muda durante o aquecimento (a menos que já estivesse saturado e fosse vaporizado, o que é improvável vindo de 4°C).
    
    # 💥 NOTA: Mantemos o y_H2O de entrada (que já está saturado ou superaquecido)
    estado_out_calc = calcular_estado_termodinamico(
        gas_fluido, 
        T_out_C_alvo, 
        P_in_bar, 
        estado_in['m_dot_gas_kg_s'], 
        estado_in['y_H2O'], 
        estado_in['y_O2'], 
        estado_in['y_H2']
    )
    
    # 2. CÁLCULO DA CARGA TÉRMICA (Q_dot)
    
    # Entalpia total de entrada (J/s)
    H_in_J_s = estado_in['H_in_mix_J_kg'] 
    
    # Entalpia total de saída (J/s) - Baseada no novo estado termodinâmico
    H_out_J_s = estado_out_calc['H_mix_J_kg'] * estado_out_calc['m_dot_mix_kg_s']
    
    # Carga térmica necessária (Q_dot = H_out - H_in)
    Q_dot_fluxo_W = H_out_J_s - H_in_J_s
    
    # 3. RESULTADO
    return {
        "T_C": T_out_C_alvo,
        "P_bar": P_in_bar,
        "Q_dot_fluxo_W": Q_dot_fluxo_W,
        "W_dot_comp_W": 0.0, # Nenhum trabalho de compressão
        "Agua_Condensada_kg_s": 0.0, # Não há condensação (apenas aquecimento)
        # O líquido acompanhante (arraste) não é removido nem condensado/evaporado (simplificação)
        "m_dot_H2O_liq_accomp_out_kg_s": estado_in['m_dot_H2O_liq_accomp_kg_s'] 
    }

if __name__ == '__main__':
    # Exemplo de teste unitário (simples)
    # Assumindo entrada de 4°C, 39.7 bar (Saída do Coalescedor)
    test_in = {
        'T_C': 4.0, 'P_bar': 39.7, 'm_dot_gas_kg_s': 0.02472, 'y_H2O': 2.04e-4, 
        'y_O2': 1.98e-4, 'y_H2': 1.0, 'gas_fluido': 'H2',
        'm_dot_H2O_liq_accomp_kg_s': 0.0, 'H_in_mix_J_kg': 598.0, # Valor fictício
        'm_dot_mix_kg_s': 0.02484
    }
    
    res = modelar_aquecedor_imaginario(test_in, T_out_C_alvo=40.0)
    print(f"Q_dot necessário para aquecer de {test_in['T_C']}°C para 40°C: {res['Q_dot_fluxo_W']:.2f} W")