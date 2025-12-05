import CoolProp.CoolProp as CP
import numpy as np

# --- CONSTANTES DE PROCESSO E PROJETO ---
R_UNIV = 8.31446    # J/(mol*K)
RHO_L_WATER = 1000.0 # kg/m³
K_SOUDERS_BROWN = 0.08 # m/s
DIAMETRO_VASO_M = 1.0 # m
DELTA_P_BAR = 0.05 # bar

def modelar_knock_out_drum(gas_fluido: str, m_dot_g_kg_s: float, P_in_bar: float, T_in_C: float, y_H2O_in: float) -> dict:
    """
    Modelagem de um Knock-Out Drum (KOD) para fluxos de Hidrogênio ou Oxigênio.
    
    O KOD opera de forma isotérmica (T_out = T_in) e isobárica (ignora-se a perda
    de pressão para cálculo da separação, mas ela é incluída no estado de saída).
    Assume-se que o gás de saída está saturado com água.
    """
    
    T_IN_K = T_in_C + 273.15
    P_IN_PA = P_in_bar * 1e5
    
    # 1. CÁLCULO DA SATURAÇÃO (Temperatura de entrada)
    try:
        # Pressão de saturação da água a T_in
        P_SAT_H2O_PA = CP.PropsSI('P', 'T', T_IN_K, 'Q', 0, 'Water')
        
        # Fracão molar de água na saturação (y_H2O_out)
        P_out_pa_ideal = P_IN_PA
        y_H2O_sat = P_SAT_H2O_PA / P_out_pa_ideal
        y_gas_sat = 1.0 - y_H2O_sat
        
        # O KOD remove toda a água condensável se T_out < T_in de saturação.
        # Se y_H2O_in > y_H2O_sat, há condensação e y_H2O_out será y_H2O_sat.
        # Como o Dry Cooler e Chiller foram modelados como resfriamento sensível (sem condensação),
        # assumimos y_H2O_in é mantido, mas o KOD irá saturar o gás na T_in.
        y_H2O_out = y_H2O_sat 
        
        # Molar Massas
        M_H2O = CP.PropsSI('M', 'Water')
        M_GAS = CP.PropsSI('M', gas_fluido)
        
        # CÁLCULO DA VAZÃO MÁSSICA DE ÁGUA REMOVIDA
        # Este é um passo crucial que deve ser feito no módulo central
        # A modelagem completa exigiria a vazão molar total de entrada.
        
        # 2. CÁLCULO DE PROPRIEDADES DE SAÍDA
        M_MIX_G_out = y_gas_sat * M_GAS + y_H2O_sat * M_H2O
        P_OUT_BAR = P_in_bar - DELTA_P_BAR
        P_OUT_PA = P_OUT_BAR * 1e5
        
        Z_gas = CP.PropsSI('Z', 'T', T_IN_K, 'P', P_OUT_PA, gas_fluido)
        
        # Densidade da mistura
        rho_G_out = P_OUT_PA * M_MIX_G_out / (Z_gas * R_UNIV * T_IN_K)
        
        # Vazão Volumétrica (Vazão Mássica de Gás que Passa / Densidade de Gás)
        # Assumindo que a vazão mássica do gás (sem água) é constante no KOD:
        m_dot_gas_out_kg_s = m_dot_g_kg_s * (y_gas_sat / (y_H2O_in * M_H2O / M_GAS + y_gas_sat)) # Correção de vazão massica por mudança de composição
        
        vazao_volumetrica_gas_out = m_dot_gas_out_kg_s / rho_G_out
        
    except Exception as e:
        return {"erro": f"Erro no cálculo de propriedades do CoolProp para {gas_fluido}: {e}"}

    
    # 3. CÁLCULO DE DIMENSIONAMENTO (Verificação da Velocidade)
    
    # V_max (Velocidade Máxima Permissível)
    V_max = K_SOUDERS_BROWN * np.sqrt((RHO_L_WATER - rho_G_out) / rho_G_out)
    
    # V_real (Velocidade Superficial Real)
    A_vaso = np.pi * (DIAMETRO_VASO_M / 2)**2
    V_superficial_real = vazao_volumetrica_gas_out / A_vaso
    
    # Status
    status_separacao = ("OK" if V_superficial_real < V_max else "ATENÇÃO: Vaso subdimensionado!")
    
    # 4. CÁLCULO ENERGÉTICO (Potência adicional para vencer o Delta P)
    # Potência elétrica (W) para vencer a queda de pressão: W = V_dot * Delta P
    W_dot_adicional_W = vazao_volumetrica_gas_out * (DELTA_P_BAR * 1e5)
    
    # 5. Dicionário de Saída Padronizado
    results = {
        # Estado de Saída do Gás
        "T_C": T_in_C,
        "P_bar": P_OUT_BAR,
        "y_H2O": y_H2O_out,
        "m_dot_gas_out_kg_s": m_dot_gas_out_kg_s,
        # Energia do Componente (Calor = 0, Trabalho é a perda de pressão)
        "Q_dot_fluxo_W": 0.0,
        "W_dot_comp_W": W_dot_adicional_W,
        # Status
        "Status_KOD": status_separacao,
        "V_real": V_superficial_real,
        "V_max": V_max
    }
    
    return results

if __name__ == '__main__':
    # Exemplo de Teste
    T_in_C = 4.0
    P_in_bar = 39.80 # Saída do Chiller
    m_dot_h2 = 0.02472
    m_dot_o2 = 0.19778
    y_h2o_in = 0.001
    
    print("--- Teste Unitário KOD ---")
    res_h2 = modelar_knock_out_drum('H2', m_dot_h2, P_in_bar, T_in_C, y_h2o_in)
    res_o2 = modelar_knock_out_drum('O2', m_dot_o2, P_in_bar, T_in_C, y_h2o_in)
    print(f"H2 Saída: T={res_h2['T_C']:.2f}C, P={res_h2['P_bar']:.2f}bar, y_H2O={res_h2['y_H2O']:.6f}, Status={res_h2['Status_KOD']}")
    print(f"O2 Saída: T={res_o2['T_C']:.2f}C, P={res_o2['P_bar']:.2f}bar, y_H2O={res_o2['y_H2O']:.6f}, Status={res_o2['Status_KOD']}")