import numpy as np
import pandas as pd
import sys 

# =================================================================
# === FUNÇÕES DE CÁLCULO GERAIS ===
# =================================================================

def get_gas_cp(gas_name):
    """Retorna o calor específico (cp) do gás seco em J/(kg.K) (Valores de Referência a 80C e 40 bar)."""
    # Gás Principal (Seco)
    c_p_H2 = 14300 
    c_p_O2 = 918   
    return c_p_H2 if gas_name == 'H2' else c_p_O2

def get_liquid_water_properties():
    """Retorna propriedades da água líquida e de vaporização (Valores de Referência)."""
    # cp da água líquida (J/(kg.K))
    c_p_H2O_liq = 4186 
    # Calor Latente de Vaporização (J/kg) - Aproximado para 40C (condensação)
    h_fg = 2393000  
    return c_p_H2O_liq, h_fg


def calculate_LMTD(T_g_in, T_g_out, T_a_in, T_a_out, F=0.85):
    """Calcula a Diferença de Temperatura Média Logarítmica (LMTD) corrigida."""
    Delta_T1 = T_g_in - T_a_out
    Delta_T2 = T_g_out - T_a_in
    
    if Delta_T1 <= 0 or Delta_T2 <= 0:
        return {"erro": "Pinch Point/Impossível. T_g_out é menor ou igual a T_a_in ou T_g_in é menor ou igual a T_a_out."}
    
    Delta_T_log = (Delta_T1 - Delta_T2) / np.log(Delta_T1 / Delta_T2)
    return F * Delta_T_log

# =================================================================
# === NOVO CÁLCULO DE CARGA TÉRMICA TOTAL (Q_dot) ===
# =================================================================

def calculate_Q_dot_total(gas_name, m_dot_g_princ, m_dot_mistura_in, m_dot_h2o_liq_acomp_kg_s, T_g_in, T_g_out_meta):
    """
    Calcula a Carga de Calor Total (Q_dot), incluindo resfriamento sensível 
    e o calor latente liberado pela água.
    """
    c_p_g_princ = get_gas_cp(gas_name)
    c_p_H2O_liq, h_fg = get_liquid_water_properties()
    c_p_H2O_vap = 1860 # cp da água vapor (J/(kg.K)) - Valor de referência

    # 1. CÁLCULO DA VAZÃO DE VAPOR DE ÁGUA NA MISTURA DE ENTRADA
    # Assume-se que a diferença entre a mistura e o gás principal é o vapor de água
    m_dot_h2o_vap_in = m_dot_mistura_in - m_dot_g_princ
    
    if m_dot_h2o_vap_in < 0:
         # Isso pode ocorrer se os dados de entrada estiverem incorretos ou se for gás seco
         m_dot_h2o_vap_in = 0
         
    # 2. CÁLCULO DA VAZÃO TOTAL DE ÁGUA (LÍQUIDA + VAPOR)
    m_dot_h2o_in_total = m_dot_h2o_vap_in + m_dot_h2o_liq_acomp_kg_s
    
    # 3. CARGA DE CALOR SENSIÍVEL DO GÁS PRINCIPAL
    # Resfriamento do gás principal (H2 ou O2)
    Q_dot_sensivel_gas = m_dot_g_princ * c_p_g_princ * (T_g_in - T_g_out_meta)
    
    # 4. CARGA DE CALOR SENSIÍVEL DA ÁGUA (LÍQUIDA + VAPOR)
    # Calor Latente: A água que entra como vapor (m_dot_h2o_vap_in) condensa.
    Q_dot_latente_condensacao = m_dot_h2o_vap_in * h_fg 
    
    # Resfriamento de toda a água como LÍQUIDO (conservador)
    Q_dot_sensivel_liquido_final = m_dot_h2o_in_total * c_p_H2O_liq * (T_g_in - T_g_out_meta)
    
    # CARGA TÉRMICA TOTAL (Simplificação Conservadora)
    # Q_dot_total = Sensível Gás Princ. + (Latente Condensação + Sensível Líquido Total)
    Q_dot_total = Q_dot_sensivel_gas + Q_dot_latente_condensacao + Q_dot_sensivel_liquido_final
    
    # 5. VAZÃO MÁSSICA TOTAL DO FLUIDO QUENTE (Para referência)
    m_dot_total_quente = m_dot_mistura_in + m_dot_h2o_liq_acomp_kg_s
    
    return Q_dot_total, m_dot_total_quente


# =================================================================
# === FUNÇÃO DE DIMENSIONAMENTO (PROJETO) ATUALIZADA ===
# =================================================================

def cooler_dimensionamento(gas_name, m_dot_g_princ, m_dot_mistura_in, m_dot_h2o_liq_acomp_kg_h, T_g_in, T_g_out_meta, P_g, T_a_in_design, U_value):
    """
    Calcula os parâmetros de projeto (Área e Potência Máxima do Ventilador)
    para o pior cenário, incluindo a água líquida extra e o vapor na mistura.
    """
    # Conversão de kg/h para kg/s
    m_dot_h2o_liq_acomp_kg_s = m_dot_h2o_liq_acomp_kg_h / 3600

    # Constantes do Modelo (Valores fixos de referência de projeto)
    c_p_a = 1005.0
    rho_a = 1.15
    delta_P_a = 500  # Pa (Queda de Pressão Ar - Estimada)
    eta_fan = 0.65
    F = 0.85
    U = U_value
    
    # 1. CÁLCULO DA CARGA DE CALOR (Q_dot) e Vazão Mássica Total do lado quente
    Q_dot, m_dot_total_quente = calculate_Q_dot_total(
        gas_name, m_dot_g_princ, m_dot_mistura_in, m_dot_h2o_liq_acomp_kg_s, T_g_in, T_g_out_meta
    )
    
    if Q_dot <= 0:
        return {"erro": "Carga térmica (Q_dot) inválida ou zero."}

    # 2. VAZÃO DE AR E T_a_out (Determinado pelo ponto de projeto)
    delta_T_a_proj = 20 # K (Delta T típico para dimensionar a vazão de ar)
        
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
        "Vazão Mássica Total (kg/s)": round(m_dot_total_quente, 5), # Vazão total quente
        "Carga Térmica Total (kW)": round(Q_dot / 1000, 2), # Q_dot corrigido
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
    
    dados_vazao = f"P_el={P_el} MW, E_spec={E_spec} kWh/kg H2 (Usado para referência de projeto)"
    if m_dot_H2_user is None:
        dados_vazao = "Vazões Detalhadas Fornecidas (Direto)"
        
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
# === CÁLCULO DA VAZÃO MÁSSICA BASEADA NA POTÊNCIA (Mantida) ===
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
# === EXECUÇÃO PRINCIPAL DIRETA (SEM MENU) ===
# =================================================================

if __name__ == '__main__':
    
    # -----------------------------------------------------------
    # VARIÁVEIS DE PROJETO (FIXAS PARA O DIMENSIONAMENTO)
    # -----------------------------------------------------------
    P_el_max = 5.0              # MW (Apenas para exibir na tabela de inputs)
    E_spec_min = 56.18          # kWh/kg H2 (Apenas para exibir na tabela de inputs)
    
    T_g_in_proj = 80            # C (Pior Cenário: Máxima temperatura de operação)
    T_g_out_proj_meta = 40      # C (Meta de resfriamento desejada)
    P_g_proj = 40               # bar
    T_a_in_design = 32          # C (Pior Cenário: Pico de temperatura ambiente)
    U_referencia = 35           # W/m2.K (Coeficiente Global Estimado)
    
    # VAZÕES DETALHADAS FORNECIDAS PELO USUÁRIO
    # H2
    m_dot_H2_principal_ref = 0.02472    # kg/s (Gás Principal H2)
    m_dot_H2_mistura_ref = 0.02745      # kg/s (Mistura H2 + H2O vapor)
    m_dot_H2O_liq_H2_ref = 1782.00      # kg/h (Água Líquida Acompanhante no fluxo H2)
    
    # O2
    m_dot_O2_principal_ref = 0.19776    # kg/s (Gás Principal O2)
    m_dot_O2_mistura_ref = 0.19915      # kg/s (Mistura O2 + H2O vapor)
    m_dot_H2O_liq_O2_ref = 247408.00    # kg/h (Água Líquida Acompanhante no fluxo O2)
    
    # -----------------------------------------------------------
    
    m_dot_H2_principal = 0.0
    m_dot_O2_principal = 0.0
    m_dot_H2_input = None # Não há entrada manual
    
    # Vazões Totais Corrigidas para serem usadas no dimensionamento
    m_dot_mistura_H2 = m_dot_H2_mistura_ref
    m_dot_mistura_O2 = m_dot_O2_mistura_ref
    m_dot_H2O_liq_H2 = m_dot_H2O_liq_H2_ref
    m_dot_H2O_liq_O2 = m_dot_H2O_liq_O2_ref
    
    print("="*50)
    print("  DIMENSIONAMENTO DE DRY COOLER PARA ELETROLISADOR")
    print("="*50)
    
    # Execução Direta (Usando Vazões Detalhadas Fornecidas)
    m_dot_H2_principal = m_dot_H2_principal_ref
    m_dot_O2_principal = m_dot_O2_principal_ref

    print("[INFO] Dimensionamento forçado usando Vazões Detalhadas Fixas:")
    print(f"       H2 Principal={m_dot_H2_principal:.5f} kg/s, O2 Principal={m_dot_O2_principal:.5f} kg/s.")
    print(f"       H2 Mistura={m_dot_mistura_H2} kg/s, H2O Líq.={m_dot_H2O_liq_H2} kg/h.")
    print(f"       O2 Mistura={m_dot_mistura_O2} kg/s, H2O Líq.={m_dot_H2O_liq_O2} kg/h.")


    # Exibe os parâmetros de entrada
    display_inputs(
        P_el=P_el_max, E_spec=E_spec_min, T_g_in=T_g_in_proj, T_g_out=T_g_out_proj_meta, 
        P_g=P_g_proj, T_a_in=T_a_in_design, U_value=U_referencia, m_dot_H2_user=m_dot_H2_input
    )

    # 4. CÁLCULO DO DIMENSIONAMENTO PARA H2
    dim_h2 = cooler_dimensionamento(
        'H2', m_dot_H2_principal, m_dot_mistura_H2, m_dot_H2O_liq_H2, 
        T_g_in_proj, T_g_out_proj_meta, P_g_proj, T_a_in_design, U_referencia
    )
    
    # 5. CÁLCULO DO DIMENSIONAMENTO PARA O2
    dim_o2 = cooler_dimensionamento(
        'O2', m_dot_O2_principal, m_dot_mistura_O2, m_dot_H2O_liq_O2, 
        T_g_in_proj, T_g_out_proj_meta, P_g_proj, T_a_in_design, U_referencia
    )

    # Exibe os resultados
    display_results_vertical("Resultados do Dimensionamento (Projeto - Pior Cenário, Com Água)", dim_h2, dim_o2)

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