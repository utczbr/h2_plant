import math

# --- CONSTANTES GLOBAIS (Sistema SI) ---
R_J_K_mol = 8.31446 # Constante Universal dos Gases (J/(mol*K))
M_H2O = 0.018015   # Peso Molecular da Água (kg/mol)
M_H2 = 0.002016    # Peso Molecular do Hidrogênio (kg/mol)
M_O2 = 0.031998    # Peso Molecular do Oxigênio (kg/mol)
BAR_TO_PA = 1e5    # 1 bar = 10^5 Pa
T_DESIGN_MARGIN = 1.10 # Margem de segurança de 10% para Pressão de Projeto (Estrutural)
CELSIUS_TO_KELVIN = 273.15
KG_S_TO_KG_H = 3600.0 # Conversão de kg/s para kg/h

# --- CONSTANTES DE SIMULAÇÃO (NOMINAIS) E VARIACÃO DE SEGURANÇA ---
P_NOMINAL_BAR = 39.70
T_NOMINAL_C = 4.00
VARIACAO_PERCENTUAL = 0.10 # 10% de variação de segurança
C_LIQ_DESIGN_MG_M3 = 100.0 # CONCENTRAÇÃO LÍQUIDA DE PROJETO (Valor Fixo)

# --- DADOS DE ENTRADA DO FLUXO (Da Simulação KOD) ---
FLUXOS_DE_GAS = {
    'H₂': {
        'M_gas': M_H2,
        'Q_M_nominal_kg_h': 0.02235 * KG_S_TO_KG_H 
    },
    'O₂': {
        'M_gas': M_O2,
        'Q_M_nominal_kg_h': 0.19647 * KG_S_TO_KG_H 
    }
}

# --- DADOS DE REFERÊNCIA DE FABRICANTES (BOROSSILICATO) ---
PROCESS_REFERENCES = {
    'BOROSSILICATO': {
        'desc': 'Fibra de Borossilicato (Alta Vazão, Tmax 200°C)',
        'q_elem_spec_m3_h': 150.0,  # Vazão Volumétrica Específica Máxima (m³/h por elemento)
        'D_elem_m': 0.20,          # Diâmetro do Elemento (m)
        'L_elem_m': 1.00,          # Comprimento do Elemento (m)
        'CAP_DRENAGEM_KG_H': 5.0   # Capacidade de Drenagem Líquida por Elemento (kg/h/elemento)
    }
}

def calcular_propriedades_do_fluido(P_min_bar, T_max_C, M_gas_principal):
    """
    Calcula propriedades da mistura gasosa (Gás Principal + H2O) saturada na condição mais desfavorável.
    """
    T_K = T_max_C + CELSIUS_TO_KELVIN
    P_total_Pa = P_min_bar * BAR_TO_PA

    # --- ESTIMAÇÃO DA PRESSÃO DE SATURAÇÃO DA ÁGUA (P_sat) ---
    # Usando interpolação/referência para T_max_C = 31.72°C (10% de variação)
    if T_max_C == 4.00:
        P_sat_bar = 0.0081 
    elif T_max_C > 31.71 and T_max_C < 31.73:
        P_sat_bar = 0.0469 # Valor de referência para 31.72°C
    else:
        P_sat_bar = 0.0582
        
    if P_min_bar <= P_sat_bar:
        raise ValueError("Pressão Mínima Total abaixo da pressão de saturação. O gás entraria em condensação total.")
    
    # --- CÁLCULO DA FRAÇÃO MOLAR (y_H2O) ---
    y_H2O = P_sat_bar / P_min_bar
    P_gas_bar = P_min_bar - P_sat_bar
    y_gas = P_gas_bar / P_min_bar
    
    # --- PESO MOLECULAR MÉDIO (M_avg) ---
    M_avg = (y_gas * M_gas_principal) + (y_H2O * M_H2O)
    
    # --- DENSIDADE MÉDIA DO GÁS (ρ_mix) ---
    rho_mix = (P_total_Pa * M_avg) / (R_J_K_mol * T_K) 

    # --- VAZÃO VOLUMÉTRICA DE OPERAÇÃO POR KG/H ---
    Q_V_OP_m3_h_por_kg_h = 1.0 / rho_mix

    return {
        'rho_mix': rho_mix,
        'Q_V_OP_m3_h_por_kg_h': Q_V_OP_m3_h_por_kg_h,
        'y_H2O': y_H2O,
        'P_sat_bar': P_sat_bar,
        'T_K': T_K
    }

def calcular_dimensionamento_coalescedor(
    Q_V_OP_total, 
    q_elem_spec_m3_h, 
    D_elem_m, 
    L_elem_m,
    P_max_op_bar,
    N_elem_cap_drenagem
):
    """Calcula o número de elementos e o diâmetro do vaso coalescedor."""
    
    # 1. CÁLCULO DO NÚMERO DE ELEMENTOS (N_elem)
    FATOR_SOBREDIMENSIONAMENTO = 1.25
    N_elem_vazao = math.ceil((Q_V_OP_total / q_elem_spec_m3_h) * FATOR_SOBREDIMENSIONAMENTO)
    
    N_elem = max(N_elem_vazao, N_elem_cap_drenagem)
    
    # 2. CÁLCULO DO DIÂMETRO DO VASO (D_shell)
    A_elem_proj = (math.pi / 4) * (D_elem_m ** 2)
    k_packing = 2.5 
    A_shell_req = N_elem * A_elem_proj * k_packing
    D_shell = math.sqrt((4 * A_shell_req) / math.pi)
    
    # 3. ALTURA ÚTIL DO VASO (H_shell)
    H_shell = L_elem_m + 0.5 

    # 4. PRESSÃO DE PROJETO
    P_design_bar = P_max_op_bar * T_DESIGN_MARGIN
    
    return {
        'Q_V_OP_total': Q_V_OP_total,
        'N_elem': N_elem,
        'D_shell': D_shell,
        'H_shell': H_shell,
        'P_design_bar': P_design_bar,
        'N_elem_vazao': N_elem_vazao,
        'N_elem_drenagem': N_elem_cap_drenagem
    }

def main():
    """Função principal para obter entradas e executar o dimensionamento."""
    print("=========================================================")
    print("  1. DIMENSIONADOR CONSOLIDADO (H₂ E O₂ NO PIOR CENÁRIO)")
    print("=========================================================")

    try:
        # --- CÁLCULO AUTOMÁTICO DO PIOR CENÁRIO (10% de variação) ---
        P_max_op_bar = P_NOMINAL_BAR * (1 + VARIACAO_PERCENTUAL)
        P_min_op_bar = P_NOMINAL_BAR * (1 - VARIACAO_PERCENTUAL)
        T_nominal_K = T_NOMINAL_C + CELSIUS_TO_KELVIN
        T_max_op_K = T_nominal_K * (1 + VARIACAO_PERCENTUAL)
        T_max_op_C = T_max_op_K - CELSIUS_TO_KELVIN
        
        # --- REFERÊNCIA DO ELEMENTO ---
        REF = PROCESS_REFERENCES['BOROSSILICATO']
        q_elem_spec_m3_h = REF['q_elem_spec_m3_h']
        D_elem_m = REF['D_elem_m']
        L_elem_m = REF['L_elem_m']
        cap_drenagem_kg_h = REF['CAP_DRENAGEM_KG_H']
        ref_key = 'BOROSSILICATO'
        
        print(f"--- Ponto Nominal (KOD Sim.): P={P_NOMINAL_BAR:.2f} bar, T={T_NOMINAL_C:.2f}°C ---")
        print(f"--- Pior Cenário de Vazão (Q_V máx): P={P_min_op_bar:.2f} bar, T={T_max_op_C:.2f}°C ---")
        print(f"Concentração Líquida de Projeto (Fixa): {C_LIQ_DESIGN_MG_M3:.0f} mg/m³")
        
        resultados_finais = {}
        
        # --- LOOP DE PROCESSAMENTO PARA H₂ E O₂ ---
        for gas_name, dados_fluxo in FLUXOS_DE_GAS.items():
            
            Q_M_OP = dados_fluxo['Q_M_nominal_kg_h']
            M_gas_principal = dados_fluxo['M_gas']
            
            # CÁLCULO DE PROPRIEDADES (PIOR CENÁRIO)
            props = calcular_propriedades_do_fluido(P_min_op_bar, T_max_op_C, M_gas_principal)
            Q_V_OP_m3_h_por_kg_h = props['Q_V_OP_m3_h_por_kg_h']
            Q_V_OP_total = Q_M_OP * Q_V_OP_m3_h_por_kg_h # Vazão Volumétrica MÁXIMA

            # CÁLCULO DA CARGA LÍQUIDA MÁXIMA PARA DRENAGEM
            Q_M_liq_max_kg_h = (C_LIQ_DESIGN_MG_M3 * Q_V_OP_total) / 1000000.0
            N_elem_cap_drenagem = math.ceil(Q_M_liq_max_kg_h / cap_drenagem_kg_h)

            # DIMENSIONAMENTO
            results = calcular_dimensionamento_coalescedor(
                Q_V_OP_total, q_elem_spec_m3_h, D_elem_m, L_elem_m, P_max_op_bar, N_elem_cap_drenagem
            )
            
            resultados_finais[gas_name] = {
                'Q_M_OP': Q_M_OP,
                'Q_V_OP_total': Q_V_OP_total,
                'N_elem': results['N_elem'],
                'D_shell': results['D_shell'],
                'H_shell': results['H_shell'],
                'P_design_bar': results['P_design_bar'],
                'Q_M_liq_max_kg_h': Q_M_liq_max_kg_h,
                'N_elem_vazao': results['N_elem_vazao'],
                'N_elem_drenagem': results['N_elem_drenagem'],
                'rho_mix': props['rho_mix']
            }

        # --- APRESENTAÇÃO DOS RESULTADOS CONSOLIDADOS ---
        
        print("\n=========================================================")
        print("           RESULTADOS CONSOLIDADOS DO DIMENSIONAMENTO")
        print("=========================================================")
        print(f"Ponto Nominal (KOD Sim.): P={P_NOMINAL_BAR:.2f} bar, T={T_NOMINAL_C:.2f}°C")
        print(f"Pior Cenário (Dimensionamento Qv, máx): P={P_min_op_bar:.2f} bar, T={T_max_op_C:.2f}°C")
        print(f"Elemento de Referência: {ref_key}")
        print(f"Concentração Líquida de Projeto (Drenagem): {C_LIQ_DESIGN_MG_M3:.0f} mg/m³")
        print(f"Pressão de Projeto do Vaso (Estrutural): {P_max_op_bar * T_DESIGN_MARGIN:.2f} bar")

        print("\n--- DIMENSIONAMENTO DO COALESCEDOR ---")

        for gas, res in resultados_finais.items():
            print(f"\n-- FLUXO: {gas} --")
            print(f"Vazão Mássica de Projeto: {res['Q_M_OP']:.2f} kg/h")
            print(f"Densidade da Mistura (ρ_mix): {res['rho_mix']:.4f} kg/m³")
            print(f"Vazão Volumétrica MÁXIMA (Qv, máx): {res['Q_V_OP_total']:.2f} m³/h")
            print(f"Carga Líquida Máxima (Drenagem): {res['Q_M_liq_max_kg_h']:.4f} kg/h")
            print(f"1. N° de Elementos (N_elem): {res['N_elem']} (Máx. entre Vazão={res['N_elem_vazao']}, Drenagem={res['N_elem_drenagem']})")
            print(f"2. Diâmetro MÍNIMO do Vaso (D_shell): {res['D_shell']:.2f} m")
            print(f"3. Altura MÍNIMA do Vaso (H_shell): {res['H_shell']:.2f} m")
            
        print("\n=========================================================")

    except Exception as e:
        print(f"\nOcorreu um erro inesperado: {e}")

if __name__ == "__main__":
    main()