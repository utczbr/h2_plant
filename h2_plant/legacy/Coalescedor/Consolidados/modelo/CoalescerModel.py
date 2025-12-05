import math

# --- CONSTANTES GLOBAIS (Sistema SI) ---
BAR_TO_PA = 1e5    # 1 bar = 10^5 Pa
PA_TO_BAR = 1e-5   # 1 Pa = 10^-5 bar
M_H2O = 0.018015   # Peso Molecular da Água (kg/mol)
M_H2 = 0.002016    # Peso Molecular do Hidrogênio (kg/mol)
M_O2 = 0.031998    # Peso Molecular do Oxigênio (kg/mol)
CELSIUS_TO_KELVIN = 273.15
R_J_K_mol = 8.31446 
KG_S_TO_KG_H = 3600.0 # Conversão de kg/s para kg/h

# --- CONSTANTES DE SIMULAÇÃO (NOMINAIS) E VARIACÃO DE SEGURANÇA ---
P_NOMINAL_BAR = 39.70
T_NOMINAL_C = 4.00
VARIACAO_PERCENTUAL = 0.10 # 10% de variação de segurança

# --- DADOS DE FLUXO E DIMENSIONAMENTO (RESULTADOS DO CÓDIGO ANTERIOR) ---
# Estes dados devem ser consistentes com a saída do dimensionador.
# N_elem e D_shell são arredondados para segurança.
FLUXOS_DE_GAS_MODELO = {
    'H₂': {
        'M_gas': M_H2,
        'Q_M_nominal_kg_h': 0.02235 * KG_S_TO_KG_H, # 80.46 kg/h
        'N_elem_dim': 1,
        'D_shell_dim': 0.32, # 32 cm
        'D_elem_m': 0.20,
        'L_elem_m': 1.00
    },
    'O₂': {
        'M_gas': M_O2,
        'Q_M_nominal_kg_h': 0.19647 * KG_S_TO_KG_H, # 707.29 kg/h
        'N_elem_dim': 1,
        'D_shell_dim': 0.32,
        'D_elem_m': 0.20,
        'L_elem_m': 1.00
    }
}

# --- PARÂMETROS EMPÍRICOS DE MODELAGEM ---
VISCOSIDADE_H2_REF = 9.0e-6 # Viscosidade Dinâmica do H2 (μ_g) em Pa·s a 30°C e 30 bar
T_REF_K = 303.15          # 30°C em Kelvin
K_PERDA_EMPIRICO = 0.5e6  # Fator de Perda (Proxy para ΔP, calibrado pelo fabricante)
ETA_LIQ_REMOCAO = 0.9999  # Eficiência de remoção de Água Líquida (99.99%)
CARGA_LIQUIDA_RESIDUAL_MGM3 = 100.0 # mg/m³ - Valor fixo para a modelagem do estado final

def calcular_propriedades_do_fluido_e_pior_cenario(M_gas_principal):
    """
    Calcula as condições do Pior Cenário e as propriedades da mistura (ρ_mix) para esse ponto.
    """
    # 1. CÁLCULO DO PIOR CENÁRIO
    P_min_bar = P_NOMINAL_BAR * (1 - VARIACAO_PERCENTUAL)
    T_nominal_K = T_NOMINAL_C + CELSIUS_TO_KELVIN
    T_max_op_K = T_nominal_K * (1 + VARIACAO_PERCENTUAL)
    T_max_op_C = T_max_op_K - CELSIUS_TO_KELVIN
    
    T_K = T_max_op_K
    P_total_Pa = P_min_bar * BAR_TO_PA

    # 2. ESTIMAÇÃO DA PRESSÃO DE SATURAÇÃO (P_sat)
    P_sat_bar = 0.0469 # Valor de referência para T_max_op_C (~31.72°C)

    # 3. CÁLCULO DA DENSIDADE DA MISTURA (ρ_mix)
    y_H2O = P_sat_bar / P_min_bar
    y_gas = 1.0 - y_H2O
    M_avg = (y_gas * M_gas_principal) + (y_H2O * M_H2O)
    rho_mix = (P_total_Pa * M_avg) / (R_J_K_mol * T_K) 
    
    # 4. VISCOSIDADE DINÂMICA DO GÁS (μ_g)
    # Modelo de Sutherland Simplificado para H2 (ΔP é proporcional à viscosidade)
    mu_g = VISCOSIDADE_H2_REF * (T_K / T_REF_K)**0.7 

    return {
        'P_min_bar': P_min_bar,
        'T_max_C': T_max_op_C,
        'rho_mix': rho_mix,
        'T_K': T_K,
        'mu_g': mu_g
    }

def modelar_desempenho_coalescedor(
    Q_M_OP, P_total_bar, T_K, rho_mix, mu_g,
    N_elem, D_shell, L_elem_m, D_elem_m):
    
    # --- CÁLCULO DE VAZÃO VOLUMÉTRICA MÁXIMA ---
    Q_V_OP_total = Q_M_OP / rho_mix * (1/3600) # m³/s
    Q_V_OP_total_m3_h = Q_M_OP / rho_mix # m³/h

    # --- MODELAGEM DA QUEDA DE PRESSÃO (ΔP) - GASTO DE ENERGIA ---
    
    # 1. Área da Seção Transversal do Vaso (A_shell)
    A_shell = (math.pi / 4) * (D_shell ** 2) 
    
    # 2. Velocidade Superficial Média (U_superficial)
    U_superficial_m_s = Q_V_OP_total / A_shell

    # 3. Estimativa da Queda de Pressão Limpa (ΔP)
    # ΔP/L ∝ (μ_g * U_superficial) - Modelo Carman-Kozeny Simplificado
    Delta_P_Pa_limpa = K_PERDA_EMPIRICO * mu_g * L_elem_m * U_superficial_m_s
    Delta_P_bar_limpa = Delta_P_Pa_limpa * PA_TO_BAR
    
    # 4. Gasto de Energia (Potência)
    Potencia_gasta_W = Q_V_OP_total * Delta_P_Pa_limpa
    
    # --- MODELAGEM DA PUREZA DE SAÍDA (ÁGUA LÍQUIDA REMANESCENTE) ---
    
    # 1. Água Líquida de Entrada (Q_M_liq_in) - Baseada na CARGA_LIQUIDA_RESIDUAL_MGM3
    Q_M_liq_in_kg_h = (CARGA_LIQUIDA_RESIDUAL_MGM3 * Q_V_OP_total_m3_h) / 1000000.0
    
    # 2. Água Líquida de Saída (Q_M_liq_out)
    Q_M_liq_out_kg_h = Q_M_liq_in_kg_h * (1.0 - ETA_LIQ_REMOCAO)
    
    # 3. Concentração Líquida na Saída (C_liq_out)
    C_liq_out_mg_m3 = (Q_M_liq_out_kg_h * 1000000.0) / Q_V_OP_total_m3_h
    
    return {
        'Delta_P_bar_limpa': Delta_P_bar_limpa,
        'Potencia_gasta_W': Potencia_gasta_W,
        'Q_M_liq_out_kg_h': Q_M_liq_out_kg_h,
        'C_liq_out_mg_m3': C_liq_out_mg_m3,
        'U_superficial_m_s': U_superficial_m_s,
        'Q_V_OP_total_m3_h': Q_V_OP_total_m3_h
    }

def main():
    """Função principal para modelagem do desempenho."""
    print("=========================================================")
    print("  2. MODELO DE DESEMPENHO CONSOLIDADO (H₂ E O₂)")
    print("=========================================================")
    print(f"Critério de Modelagem: C_liq = {CARGA_LIQUIDA_RESIDUAL_MGM3:.0f} mg/m³")
    
    try:
        propriedades_pior_cenario = calcular_propriedades_do_fluido_e_pior_cenario(M_H2) # Usa M_H2 apenas para obter P_min/T_max
        
        # --- LOOP DE PROCESSAMENTO PARA H₂ E O₂ ---
        for gas_name, dados_fluxo in FLUXOS_DE_GAS_MODELO.items():
            
            # 1. PARÂMETROS ESPECÍFICOS DO FLUXO
            Q_M_OP = dados_fluxo['Q_M_nominal_kg_h']
            M_gas_principal = dados_fluxo['M_gas']
            N_elem = dados_fluxo['N_elem_dim']
            D_shell = dados_fluxo['D_shell_dim']
            D_elem_m = dados_fluxo['D_elem_m']
            L_elem_m = dados_fluxo['L_elem_m']
            
            # 2. CALCULAR PROPRIEDADES ESPECÍFICAS DO GÁS NO PIOR CENÁRIO
            props = calcular_propriedades_do_fluido_e_pior_cenario(M_gas_principal)
            
            # 3. MODELAGEM DE DESEMPENHO
            results = modelar_desempenho_coalescedor(
                Q_M_OP, 
                props['P_min_bar'], 
                props['T_K'], 
                props['rho_mix'], 
                props['mu_g'],
                N_elem, 
                D_shell, 
                L_elem_m, 
                D_elem_m
            )
            
            # --- APRESENTAÇÃO DOS RESULTADOS ---
            print(f"\n--- DESEMPENHO DO COALESCEDOR DE {gas_name} (N_elem={N_elem}) ---")
            print(f"Pior Cenário Modelado: P={props['P_min_bar']:.2f} bar, T={props['T_max_C']:.2f}°C")
            print(f"Vazão Volumétrica MÁXIMA: {results['Q_V_OP_total_m3_h']:.2f} m³/h")
            print(f"Velocidade Superficial (U_sup): {results['U_superficial_m_s']:.4f} m/s")

            # Gasto de Energia
            print("\n* GASTO DE ENERGIA (Perda de Carga)")
            print(f"1. Queda de Pressão Limpa (ΔP): {results['Delta_P_bar_limpa']:.4f} bar")
            print(f"2. Potência Gasta: {results['Potencia_gasta_W']:.2f} W")
            
            # Estado Final do Fluxo
            print("\n* ESTADO FINAL DO FLUXO")
            print(f"1. Água Líquida Remanescente (Saída): {results['Q_M_liq_out_kg_h']:.8f} kg/h")
            print(f"2. Concentração Líquida na Saída: {results['C_liq_out_mg_m3']:.4f} mg/m³")
            print(f"3. Pureza de Remoção: {ETA_LIQ_REMOCAO * 100.0:.2f}\% de aerossóis removidos.")
            
        print("\n=========================================================")

    except Exception as e:
        print(f"\nOcorreu um erro inesperado: {e}")

if __name__ == "__main__":
    main()