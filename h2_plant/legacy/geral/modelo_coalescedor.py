import math

# --- CONSTANTES GLOBAIS (Sistema SI) ---
BAR_TO_PA = 1e5    # 1 bar = 10^5 Pa
PA_TO_BAR = 1e-5   # 1 Pa = 10^-5 bar
M_H2O = 0.018015   # Peso Molecular da Água (kg/mol)
CELSIUS_TO_KELVIN = 273.15
KG_S_TO_KG_H = 3600.0 # Conversão de kg/s para kg/h

# --- PARÂMETROS DE DIMENSIONAMENTO E EMPÍRICOS (Ajustados do CoalescerModel.py) ---
VISCOSIDADE_H2_REF = 9.0e-6 # Viscosidade Dinâmica do H2 (μ_g) em Pa·s a 30°C e 30 bar
T_REF_K = 303.15          # 30°C em Kelvin
K_PERDA_EMPIRICO = 0.5e6  # Fator de Perda (Proxy para ΔP)
ETA_LIQ_REMOCAO = 0.9999  # Eficiência de remoção de Água Líquida (99.99%)
CARGA_LIQUIDA_RESIDUAL_MGM3 = 100.0 # mg/m³ - Carga líquida remanescente antes do KOD (para cálculo de entrada)

# Parâmetros de Projeto (simplificados para o modelo)
DELTA_P_COALESCER_BAR = 0.15 # Queda de pressão estimada no coalescedor (bar)

FLUXOS_DE_GAS_MODELO = {
    'H2': {
        'M_gas': 0.002016,
        'Q_M_nominal_kg_h': 0.02472 * KG_S_TO_KG_H, # Vazão mássica de H2
        'D_shell_dim': 0.32,
    },
    'O2': {
        'M_gas': 0.031998,
        'Q_M_nominal_kg_h': 0.19778 * KG_S_TO_KG_H, # Vazão mássica de O2
        'D_shell_dim': 0.32,
    }
}

def modelar_coalescedor(gas_fluido: str, m_dot_g_kg_s: float, P_in_bar: float, T_in_C: float, y_H2O: float) -> dict:
    """
    Modelagem simplificada do Coalescedor.
    Assume remoção de aerossóis (água líquida residual) e queda de pressão.
    
    A modelagem utiliza a vazão volumétrica para estimar o gasto de energia (ΔP).
    """
    
    T_K = T_in_C + CELSIUS_TO_KELVIN
    P_total_Pa = P_in_bar * BAR_TO_PA
    
    # 1. PARÂMETROS DO GÁS (Para Densidade e Gasto de Energia)
    try:
        dados = FLUXOS_DE_GAS_MODELO[gas_fluido]
        M_gas_principal = dados['M_gas']
        D_shell = dados['D_shell_dim']
        Q_M_OP_nominal = dados['Q_M_nominal_kg_h'] # kg/h
    except KeyError:
        return {"erro": f"Gás {gas_fluido} não mapeado no modelo Coalescedor."}

    # 2. CÁLCULO DA DENSIDADE DA MISTURA (ρ_mix)
    y_gas = 1.0 - y_H2O
    M_avg = (y_gas * M_gas_principal) + (y_H2O * M_H2O)
    rho_mix = (P_total_Pa * M_avg) / (8.31446 * T_K) # Usando R_J_K_mol simplificado
    
    # --- MODELAGEM DE ENERGIA E FLUXO ---
    
    # 3. Vazão Volumétrica (V_dot) (usando vazão Mássica nominal para ser consistente com o dimensionamento)
    # Convertendo a vazão mássica de H2/O2 de kg/s para kg/h para usar a nominal
    Q_M_OP_kg_h = m_dot_g_kg_s * KG_S_TO_KG_H
    
    # Cálculo da Vazão Volumétrica (m³/h)
    Q_V_OP_total_m3_h = Q_M_OP_kg_h / rho_mix 
    Q_V_OP_total_m3_s = Q_V_OP_total_m3_h / KG_S_TO_KG_H # m³/s
    
    # 4. Gasto de Energia (Potência)
    # Usando a queda de pressão estimada (simplificação)
    Delta_P_bar = DELTA_P_COALESCER_BAR
    Delta_P_Pa = Delta_P_bar * BAR_TO_PA
    Potencia_gasta_W = Q_V_OP_total_m3_s * Delta_P_Pa
    
    # --- MODELAGEM DA REMOÇÃO DE LÍQUIDO (Aerossóis) ---
    
    # 5. Água Líquida de Entrada (Q_M_liq_in) - Baseado na CARGA_LIQUIDA_RESIDUAL_MGM3
    # Assumimos que o KOD deixou 100 mg/m³ de aerossóis no fluxo volumétrico antes da condensação.
    # Como o KOD já removeu a água condensável, aqui tratamos apenas os aerossóis.
    Q_M_liq_in_kg_h = (CARGA_LIQUIDA_RESIDUAL_MGM3 * Q_V_OP_total_m3_h) / 1000000.0 # kg/h
    
    # 6. Água Líquida Removida
    Agua_Removida_Coalescer_kg_h = Q_M_liq_in_kg_h * ETA_LIQ_REMOCAO
    
    # 7. Propriedades de Saída
    P_out_bar = P_in_bar - Delta_P_bar
    
    # 8. Dicionário de Saída Padronizado
    results = {
        # Estado de Saída do Gás
        "T_C": T_in_C, # T_out = T_in (Processo Isotérmico)
        "P_bar": P_out_bar,
        "y_H2O": y_H2O, # y_H2O não muda (remove-se apenas aerossóis)
        "m_dot_gas_out_kg_s": m_dot_g_kg_s, # Vazão de gás in = out
        # Energia do Componente
        "Q_dot_fluxo_W": 0.0, # Nenhum calor trocado
        "W_dot_comp_W": Potencia_gasta_W,
        # Saída Líquida
        "Agua_Condensada_kg_s": Agua_Removida_Coalescer_kg_h / KG_S_TO_KG_H, # Convertido para kg/s
        # Status
        "C_liq_out_mg_m3": (Q_M_liq_in_kg_h * (1.0 - ETA_LIQ_REMOCAO) * 1000000.0) / Q_V_OP_total_m3_h
    }
    
    return results

if __name__ == '__main__':
    # Exemplo de Teste
    T_in_C = 4.0
    P_in_bar = 39.75 # Saída do KOD
    m_dot_h2 = 0.02472
    m_dot_o2 = 0.19778
    y_h2o_sat_at_4C = 0.00171 # Valor de CoolProp para 4C e 39.75 bar

    print("--- Teste Unitário Coalescedor ---")
    res_h2 = modelar_coalescedor('H2', m_dot_h2, P_in_bar, T_in_C, y_h2o_sat_at_4C)
    res_o2 = modelar_coalescedor('O2', m_dot_o2, P_in_bar, T_in_C, y_h2o_sat_at_4C)
    print(f"H2 Saída: T={res_h2['T_C']:.2f}C, P={res_h2['P_bar']:.2f}bar, W={res_h2['W_dot_comp_W']:.2f}W, Água Removida={res_h2['Agua_Condensada_kg_s']:.8f} kg/s")
    print(f"O2 Saída: T={res_o2['T_C']:.2f}C, P={res_o2['P_bar']:.2f}bar, W={res_o2['W_dot_comp_W']:.2f}W, Água Removida={res_o2['Agua_Condensada_kg_s']:.8f} kg/s")