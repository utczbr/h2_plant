import numpy as np
import pandas as pd
import sys 

# =================================================================
# === FUNÇÕES DE CÁLCULO GERAIS ===
# =================================================================

def get_gas_cp(gas_name):
    """Retorna o calor específico (cp) do gás em J/(kg.K) (Valores de Referência a 80C e 40 bar)."""
    c_p_H2 = 14300 
    c_p_O2 = 918   
    return c_p_H2 if gas_name == 'H2' else c_p_O2

def calculate_LMTD(T_g_in, T_g_out, T_a_in, T_a_out, F=0.85):
    """Calcula a Diferença de Temperatura Média Logarítmica (LMTD) corrigida."""
    Delta_T1 = T_g_in - T_a_out
    Delta_T2 = T_g_out - T_a_in
    
    if Delta_T1 <= 0 or Delta_T2 <= 0:
        return {"erro": "Pinch Point/Impossível. T_g_out é menor ou igual a T_a_in ou T_g_in é menor ou igual a T_a_out."}
    
    Delta_T_log = (Delta_T1 - Delta_T2) / np.log(Delta_T1 / Delta_T2)
    return F * Delta_T_log

# =================================================================
# === FUNÇÃO DE DIMENSIONAMENTO (PROJETO) ===
# =================================================================

def cooler_dimensionamento(gas_name, m_dot_g, T_g_in, T_g_out_meta, P_g, T_a_in_design, U_value):
    """
    Calcula os parâmetros de projeto (Área e Potência Máxima do Ventilador)
    para o pior cenário.
    """
    # Constantes do Modelo (Valores fixos de referência de projeto)
    c_p_g = get_gas_cp(gas_name)
    c_p_a = 1005.0
    rho_a = 1.15
    delta_P_a = 500  # Pa (Queda de Pressão Ar - Estimada)
    eta_fan = 0.65
    F = 0.85
    U = U_value
    
    # 1. CÁLCULO DA CARGA DE CALOR (Q_dot)
    Q_dot = m_dot_g * c_p_g * (T_g_in - T_g_out_meta)
    
    # 2. VAZÃO DE AR E T_a_out (Determinado pelo ponto de projeto)
    delta_T_a_proj = 20 # K (Delta T típico para dimensionar a vazão de ar)
    if Q_dot <= 0:
        return {"erro": "Carga térmica (Q_dot) inválida ou zero."}
    
    m_dot_a_design = Q_dot / (c_p_a * delta_T_a_proj)
    T_a_out_design = T_a_in_design + Q_dot / (m_dot_a_design * c_p_a)
    
    # 3. CÁLCULO LMTD
    LMTD = calculate_LMTD(T_g_in, T_g_out_meta, T_a_in_design, T_a_out_design, F)
    if isinstance(LMTD, dict): return LMTD 
        
    # 4. CÁLCULO DA ÁREA (A)
    Area_m2 = Q_dot / (U * LMTD)
    
    # 5. CÁLCULO DA ENERGIA (Potência Máxima do Ventilador)
    V_dot_a = m_dot_a_design / rho_a 
    W_fan_watts = (V_dot_a * delta_P_a) / eta_fan
    
    # Resultados de Saída aprimorados
    results = {
        "Gás": gas_name,
        "Modelo Dry Cooler": "Casco e Tubos Aletados (Fluxo Cruzado)", # Informação do Modelo
        "Vazão Mássica (kg/s)": round(m_dot_g, 5),
        "Carga Térmica (kW)": round(Q_dot / 1000, 2),
        "Área Mínima (m²)": round(Area_m2, 2), 
        "Potência Máx. Fan (kW)": round(W_fan_watts / 1000, 3),
        "Vazão Ar Design (kg/s)": round(m_dot_a_design, 3),
        "Coef. Global U (W/m².K)": U, # Parâmetro crucial para a Modelagem
        "Queda de Pressão Ar (Pa)": delta_P_a # Parâmetro crucial para a Modelagem
    }
    
    return results

# =================================================================
# === FUNÇÕES DE EXIBIÇÃO DE TABELAS ===
# =================================================================

def display_inputs(P_el, E_spec, T_g_in, T_g_out, P_g, T_a_in, U_value, m_dot_H2_user=None):
    """Exibe os parâmetros de entrada em uma tabela vertical."""
    
    dados_vazao = f"P_el={P_el} MW, E_spec={E_spec} kWh/kg H2"
    if m_dot_H2_user is not None:
        dados_vazao = f"Entrada Manual: {m_dot_H2_user} kg/s (H2)"
        
    data = {
        "Parâmetro de Entrada": [
            "Dados de Vazão Utilizados",
            "T Entrada Gás (Projeto) (°C)", 
            "T Saída Gás (Meta) (°C)", 
            "Pressão Gás (bar)", 
            "T Entrada Ar (Pior Cenário) (°C)", 
            "Coef. Global U (W/m².K) [Est.]"
        ],
        "Valor": [dados_vazao, T_g_in, T_g_out, P_g, T_a_in, U_value]
    }
    df = pd.DataFrame(data)
    print("\n" + "="*70)
    print("      🧾 Parâmetros de Entrada do Dimensionamento (Pior Cenário)     ")
    print("="*70)
    print(df.to_string(index=False))
    print("="*70)


def display_results_vertical(title, results_h2, results_o2):
    """Exibe os resultados do dimensionamento em uma tabela vertical (transposta)."""
    # Garante que resultados incompletos ou erros não quebrem a tabela
    if isinstance(results_h2, dict) and "erro" in results_h2: 
        results_h2 = {k: "ERRO" for k in results_h2.keys()}
    if isinstance(results_o2, dict) and "erro" in results_o2: 
        results_o2 = {k: "ERRO" for k in results_o2.keys()}
        
    # Remove as chaves de erro para garantir a integridade da tabela
    results_h2.pop('Gás', None)
    results_o2.pop('Gás', None)

    df = pd.DataFrame({
        "Hidrogênio (H2)": results_h2, 
        "Oxigênio (O2)": results_o2
    }).T.T
    
    print("\n" + "="*80)
    print(f"        {title}         ")
    print("="*80)
    df.index.name = 'Parâmetros de Saída'
    print(df.to_string())
    print("="*80)

# =================================================================
# === CÁLCULO DA VAZÃO MÁSSICA BASEADA NA POTÊNCIA ===
# =================================================================

def calculate_max_flow(P_el_max, E_spec_min):
    """Calcula a vazão mássica máxima de H2 e O2 baseada na potência e eficiência."""
    try:
        E_spec_kJ_kg = E_spec_min * 3600  # kWh/kg -> kJ/kg
        P_el_kW = P_el_max * 1000         # MW -> kW
        
        m_dot_H2_max = P_el_kW / E_spec_kJ_kg
        m_dot_O2_max = m_dot_H2_max * (32/4) # Relação estequiométrica 8:1
        
        return m_dot_H2_max, m_dot_O2_max
    except ZeroDivisionError:
        print("[ERRO FATAL] O Consumo Específico (E_spec) não pode ser zero.")
        sys.exit(1)
    except Exception as e:
        print(f"[ERRO FATAL] Falha no cálculo do fluxo: {e}")
        sys.exit(1)

# =================================================================
# === EXECUÇÃO PRINCIPAL COM MENU ===
# =================================================================

if __name__ == '__main__':
    
    # -----------------------------------------------------------
    # VARIÁVEIS DE PROJETO (FIXAS PARA O DIMENSIONAMENTO)
    # -----------------------------------------------------------
    P_el_max = 5.0              # MW
    E_spec_min = 56.18          # kWh/kg H2
    
    T_g_in_proj = 80            # C (Pior Cenário: Máxima temperatura de operação)
    T_g_out_proj_meta = 40      # C (Meta de resfriamento desejada)
    P_g_proj = 40               # bar
    T_a_in_design = 32          # C (Pior Cenário: Pico de temperatura ambiente)
    U_referencia = 35           # W/m2.K (Coeficiente Global Estimado)
    # -----------------------------------------------------------
    
    m_dot_H2 = 0.0
    m_dot_O2 = 0.0
    m_dot_H2_input = None # Usado apenas para exibir na tabela de inputs

    print("="*50)
    print("  DIMENSIONAMENTO DE DRY COOLER PARA ELETROLISADOR")
    print("="*50)
    print("Selecione a opção para definir a VAZÃO MÁSSICA (m_dot) de dimensionamento:")
    print("1: Dimensionamento via Potência Máxima do PEM (Calcula H2 e O2)")
    print("2: Dimensionamento Personalizado (Entrada manual de m_dot_H2)")
    print("="*50)
    
    try:
        choice = int(input("Digite sua escolha (1 ou 2): "))
    except ValueError:
        print("\n[ERRO] Entrada inválida. Por favor, digite 1 ou 2.")
        sys.exit(1)

    if choice == 1:
        m_dot_H2, m_dot_O2 = calculate_max_flow(P_el_max, E_spec_min)
        print(f"\n[INFO] Vazão de H2 calculada: {m_dot_H2:.5f} kg/s (Usando 5 MW e 56.18 kWh/kg).")
        print(f"[INFO] Vazão de O2 calculada: {m_dot_O2:.5f} kg/s (Estequiometria).")
        
    elif choice == 2:
        try:
            m_dot_H2_input = float(input("\nDigite a Vazão Mássica de H2 (kg/s) para dimensionamento: "))
            if m_dot_H2_input <= 0:
                raise ValueError
            m_dot_H2 = m_dot_H2_input
            m_dot_O2 = m_dot_H2 * (32/4) # O O2 é sempre calculado via H2
            print(f"[INFO] Vazão de H2 definida: {m_dot_H2:.5f} kg/s.")
            print(f"[INFO] Vazão de O2 correspondente: {m_dot_O2:.5f} kg/s.")
        except ValueError:
            print("\n[ERRO] Vazão mássica deve ser um número positivo.")
            sys.exit(1)
            
    else:
        print("\n[ERRO] Opção não reconhecida.")
        sys.exit(1)

    # Exibe os parâmetros de entrada
    display_inputs(
        P_el=P_el_max, E_spec=E_spec_min, T_g_in=T_g_in_proj, T_g_out=T_g_out_proj_meta, 
        P_g=P_g_proj, T_a_in=T_a_in_design, U_value=U_referencia, m_dot_H2_user=m_dot_H2_input
    )

    # 4. CÁLCULO DO DIMENSIONAMENTO PARA H2
    dim_h2 = cooler_dimensionamento(
        'H2', m_dot_H2, T_g_in_proj, T_g_out_proj_meta, P_g_proj, T_a_in_design, U_referencia
    )
    
    # 5. CÁLCULO DO DIMENSIONAMENTO PARA O2
    dim_o2 = cooler_dimensionamento(
        'O2', m_dot_O2, T_g_in_proj, T_g_out_proj_meta, P_g_proj, T_a_in_design, U_referencia
    )

    # Exibe os resultados
    display_results_vertical("Resultados do Dimensionamento (Projeto - Pior Cenário)", dim_h2, dim_o2)

    # Informação para o próximo passo (modelagem)
    try:
        AREA_H2_DESIGN = dim_h2["Área Mínima (m²)"]
        AREA_O2_DESIGN = dim_o2["Área Mínima (m²)"]
        # Informa o usuário sobre o U e a Queda de Pressão do Ar necessários para a Modelagem
        U_value_info = dim_h2.get("Coef. Global U (W/m².K)", U_referencia)
        dP_a_info = dim_h2.get("Queda de Pressão Ar (Pa)", 500)

        print(f"\n[INFO] Parâmetros de Projeto FIXOS para 'modelagem.py':")
        print(f"Área H2: {AREA_H2_DESIGN} m² | Área O2: {AREA_O2_DESIGN} m².")
        print(f"Coeficiente U: {U_value_info} W/m².K | Queda de Pressão Ar: {dP_a_info} Pa.")
    except (KeyError, TypeError):
        print("\n[ERRO] Não foi possível obter as áreas de projeto.")