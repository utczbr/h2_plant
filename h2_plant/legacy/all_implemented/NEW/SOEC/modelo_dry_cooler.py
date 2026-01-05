# modelo_dry_cooler.py
import numpy as np
import sys 
import CoolProp.CoolProp as CP 

# 🛑 NOVO: Importa P_PERDA_BAR do arquivo de constantes
try:
    from constants_and_config import P_PERDA_BAR 
except ImportError:
    # Fallback seguro caso P_PERDA_BAR não seja encontrada
    P_PERDA_BAR = 0.0 # Define a perda como 0.0 se não conseguir importar

# =================================================================
# === PARÂMETROS DE PROJETO (FIXOS DO DIMENSIONAMENTO DO USUÁRIO) ===
# =================================================================
# Variáveis de Dimensionamento (Valores do Dimensionamento do Usuário - Pior Cenário 32°C):
# NOVOS VALORES DE DIMENSIONAMENTO PARA SOEC (Saída SOEC -> DC 1)
AREA_H2_TQC_DESIGN = 10.0       # m² (Área do Trocador Quente H2) 🛑 NOVO
AREA_H2_DC_DESIGN = 453.62      # m² (Área do Dry Cooler H2 - Mantido, agora resfria o glicol)
AREA_O2_TQC_DESIGN = 5.0        # m² (Área do Trocador Quente O2) 🛑 NOVO
AREA_O2_DC_DESIGN = 42.95       # m² (Área do Dry Cooler O2 - Mantido, agora resfria o glicol)

U_VALUE_TQC_DESIGN = 1000       # W/m².K (U-Value do Trocador Quente - Fluido-Fluido) 🛑 NOVO
U_VALUE_DC_DESIGN = 35          # W/m².K (U-Value do Dry Cooler - Fluido-Ar) 
DP_AIR_DESIGN = 500             # Pa (Queda de Pressão do Ar de Projeto) 
DP_LIQ_TQC = 0.5                # bar (Queda de pressão estimada no lado do glicol do TQC) 🛑 NOVO
DP_LIQ_DC = 0.5                 # bar (Queda de pressão estimada no lado do glicol do DC) 🛑 NOVO


# Parâmetros de Operação Padrão do Ar (para simulação)
T_A_IN_OP = 32                  # °C (Temperatura de entrada do ar para simulação - Pior Cenário 32°C)
# NOVAS VAZÕES DE AR DE DIMENSIONAMENTO PARA SOEC (Mantidas, agora para resfriar glicol)
M_DOT_A_H2_DESIGN = 25.887      # kg/s (Vazão de Ar Design Dry Cooler H2)
M_DOT_A_O2_DESIGN = 3.552       # kg/s (Vazão de Ar Design Dry Cooler O2)

# Parâmetros do Circuito Intermediário (Água + Etilenoglicol) 🛑 NOVO
GLYCOL_FRACTION = 0.40          # Fração em massa de Etilenoglicol (40%)
M_DOT_REF_H2 = 5.0              # kg/s (Vazão de refrigerante para o sistema H2/DC) 🛑 NOVO
M_DOT_REF_O2 = 1.0              # kg/s (Vazão de refrigerante para o sistema O2/DC) 🛑 NOVO
T_REF_IN_TQC = 30.0             # °C (Temperatura de entrada do refrigerante no TQC - Saída do DC) 🛑 NOVO


# =================================================================
# === FUNÇÕES DE CÁLCULO GERAIS ===
# =================================================================

def get_gas_cp_and_liquid_cp(gas_name):
    """Retorna o calor específico (cp) do gás e o cp da água líquida em J/(kg.K)."""
    # Cp Mássico (J/kg.K) a 300 K (aprox. 27 °C)
    c_p_H2_gas = CP.PropsSI('CPMASS', 'T', 300, 'P', 1e5, 'H2') 
    c_p_O2_gas = CP.PropsSI('CPMASS', 'T', 300, 'P', 1e5, 'O2') 
    c_p_H2O_liq = CP.PropsSI('CPMASS', 'T', 300, 'P', 1e5, 'Water')
    
    c_p_g = c_p_H2_gas if gas_name == 'H2' else c_p_O2_gas
    return c_p_g, c_p_H2O_liq

def get_cp_ref_water_glycol(mass_fraction_glycol=GLYCOL_FRACTION):
    """
    Retorna o calor específico da mistura água+etilenoglicol (J/kg.K).
    Aproximação linear usando valores de referência a 25°C:
    Cp_agua = 4180 J/kg.K
    Cp_glicol = 2430 J/kg.K
    """
    c_p_H2O = 4180 # J/kg.K
    c_p_glycol = 2430 # J/kg.K
    # Aproximação do cp da mistura
    c_p_ref = (1 - mass_fraction_glycol) * c_p_H2O + mass_fraction_glycol * c_p_glycol
    return c_p_ref

def calcular_potencia_ventilador(m_dot_a, DP_air):
    """Calcula a potência elétrica do ventilador (W), assumindo eficiência de 60%."""
    rho_ar = 1.225 # kg/m³ (Densidade do ar a 20C, 1 atm)
    V_dot_a = m_dot_a / rho_ar
    P_ventilador_mecanica = V_dot_a * DP_air
    eficiencia = 0.60
    return P_ventilador_mecanica / eficiencia


# =================================================================
# === FUNÇÕES DE MODELAGEM (NTU-EFICÁCIA) ===
# =================================================================

def calcular_desempenho_trocador(Area_m2, U_value, C_quente, C_frio, T_q_in_K, T_f_in_K, tipo_fluxo='Contracorrente') -> dict:
    """
    Calcula a eficácia (E) e a potência térmica (Q) de um trocador de calor.
    Assume Fluxo Contracorrente ou Fluxo Cruzado dependendo dos fluidos.
    """
    
    C_min = min(C_quente, C_frio)
    C_max = max(C_quente, C_frio)
    
    if C_min <= 0:
         raise ValueError("C_min é zero. Vazão mássica ou cp inválido.")
         
    R = C_min / C_max
    NTU = U_value * Area_m2 / C_min
    
    # Cálculo da Eficácia (E)
    if tipo_fluxo == 'Contracorrente':
        # E = (1 - exp(-NTU * (1 - R))) / (1 - R * exp(-NTU * (1 - R)))
        if R != 1.0:
            E = (1 - np.exp(-NTU * (1 - R))) / (1 - R * np.exp(-NTU * (1 - R)))
        else: # R=1
            E = NTU / (1 + NTU)
    elif tipo_fluxo == 'Fluxo Cruzado':
        # Aproximação comum utilizada para Fluxo Cruzado (usada anteriormente no Dry Cooler)
        if R != 0:
            E = 1 - np.exp( (1/R) * (NTU**0.22) * (np.exp(-R * NTU**0.78) - 1) )
        else: # R=0
            E = 1 - np.exp(-NTU)
    else:
        raise ValueError("Tipo de fluxo não suportado.")
    
    # Cálculo de Energia e Temperatura de Saída
    Q_max = C_min * (T_q_in_K - T_f_in_K) # Q_max em Watts
    Q_dot_real_W = E * Q_max
    
    T_q_out_K = T_q_in_K - Q_dot_real_W / C_quente
    T_f_out_K = T_f_in_K + Q_dot_real_W / C_frio
    
    return {
        "Q_dot_W": Q_dot_real_W,
        "T_quente_out_C": T_q_out_K - 273.15,
        "T_frio_out_C": T_f_out_K - 273.15,
        "Eficacia": E,
        "NTU": NTU,
    }


def modelar_trocador_fluido_fluido(gas_fluido: str, m_dot_mix_kg_s: float, m_dot_H2O_liq_kg_s: float, T_g_in_C: float, m_dot_ref: float, T_ref_in_C: float) -> dict:
    """
    Modelagem do Trocador de Calor (TQC) - Fluido Quente (Gás+Líquido) -> Refrigerante (Glicol).
    Assume Fluxo Contracorrente.
    """
    
    # 1. Parâmetros do Componente
    Area_m2 = AREA_H2_TQC_DESIGN if gas_fluido == 'H2' else AREA_O2_TQC_DESIGN
    U_value = U_VALUE_TQC_DESIGN
    
    # Capacidades de Calor Específico
    c_p_g, c_p_H2O_liq = get_gas_cp_and_liquid_cp(gas_fluido)
    c_p_ref = get_cp_ref_water_glycol()
    
    # 2. Capacidades Térmicas
    m_dot_gas_fase = m_dot_mix_kg_s - m_dot_H2O_liq_kg_s 
    C_gas_mix = m_dot_gas_fase * c_p_g
    C_liquido = m_dot_H2O_liq_kg_s * c_p_H2O_liq 
    C_quente = C_gas_mix + C_liquido
    
    C_frio = m_dot_ref * c_p_ref
    
    T_q_in_K = T_g_in_C + 273.15
    T_f_in_K = T_ref_in_C + 273.15
    
    # 3. Cálculo de Desempenho
    resultados = calcular_desempenho_trocador(Area_m2, U_value, C_quente, C_frio, T_q_in_K, T_f_in_K, tipo_fluxo='Contracorrente')

    # 4. Potência de Saída e Perda de Pressão
    P_perda = DP_LIQ_TQC # Perda de Pressão apenas no lado do glicol (fluido frio)
    
    return {
        # Estado de Saída do Gás (Fluido Quente)
        "T_g_out_C": resultados["T_quente_out_C"],
        # Estado de Saída do Refrigerante (Fluido Frio)
        "T_ref_out_C": resultados["T_frio_out_C"],
        "P_ref_out_bar_perda": P_perda,
        # Energia do Componente
        "Q_dot_TQC_W": resultados["Q_dot_W"]
    }


def modelar_dry_cooler_ar(gas_fluido: str, m_dot_ref: float, T_ref_in_C: float, m_dot_a_op: float = None) -> dict:
    """
    Modelagem do Dry Cooler (DC) - Refrigerante (Glicol) -> Ar Ambiente.
    Assume Fluxo Cruzado.
    """
    
    # 1. Parâmetros do Componente
    Area_m2 = AREA_H2_DC_DESIGN if gas_fluido == 'H2' else AREA_O2_DC_DESIGN
    U_value = U_VALUE_DC_DESIGN
    T_a_in_op = T_A_IN_OP
    DP_air = DP_AIR_DESIGN
    
    if gas_fluido == 'H2':
        if m_dot_a_op is None: m_dot_a_op = M_DOT_A_H2_DESIGN
    elif gas_fluido == 'O2':
        if m_dot_a_op is None: m_dot_a_op = M_DOT_A_O2_DESIGN
    else:
        raise ValueError(f"Gás {gas_fluido} não suportado.")
        
    # Capacidades de Calor Específico
    c_p_ref = get_cp_ref_water_glycol()
    c_p_a = 1005.0 # J/(kg.K)
    
    # 2. Capacidades Térmicas
    C_quente = m_dot_ref * c_p_ref # Refrigerante (Glicol)
    C_frio = m_dot_a_op * c_p_a    # Ar
    
    T_q_in_K = T_ref_in_C + 273.15
    T_f_in_K = T_a_in_op + 273.15
    
    # 3. Cálculo de Desempenho
    resultados = calcular_desempenho_trocador(Area_m2, U_value, C_quente, C_frio, T_q_in_K, T_f_in_K, tipo_fluxo='Fluxo Cruzado')

    # 4. Potência de Saída e Consumo Elétrico
    W_dot_ventilador_W = calcular_potencia_ventilador(m_dot_a_op, DP_air)
    P_perda = DP_LIQ_DC # Perda de Pressão apenas no lado do glicol (fluido quente)
    
    return {
        # Estado de Saída do Refrigerante (Fluido Quente)
        "T_ref_out_C": resultados["T_quente_out_C"],
        # Energia do Componente
        "Q_dot_DC_W": resultados["Q_dot_W"],
        "W_dot_ventilador_W": W_dot_ventilador_W,
        "P_ref_out_bar_perda": P_perda,
    }

# =================================================================
# === FUNÇÃO DE SIMULAÇÃO DO SISTEMA COMPLETO ===
# =================================================================

def simular_sistema_resfriamento(gas_fluido: str, m_dot_mix_kg_s: float, m_dot_H2O_liq_kg_s: float, P_in_bar: float, T_g_in_C: float, m_dot_a_op: float = None) -> dict:
    """
    Simula o sistema Dry Cooler com Trocador de Calor Quente (TQC) intermediário
    usando a mistura Água/Glicol.
    """
    
    if gas_fluido == 'H2':
        m_dot_ref = M_DOT_REF_H2
        # T_ref_in_TQC é a T_ref_out_DC (Temperatura de retorno do refrigerante resfriado)
        T_ref_in_TQC_C = T_REF_IN_TQC 
    elif gas_fluido == 'O2':
        m_dot_ref = M_DOT_REF_O2
        T_ref_in_TQC_C = T_REF_IN_TQC
    else:
        raise ValueError(f"Gás {gas_fluido} não suportado.")
        
    # --- 1. Trocador de Calor Quente (TQC) ---
    # Fluido Quente: Gás+Líquido -> Fluido Frio: Refrigerante
    try:
        resultado_tqc = modelar_trocador_fluido_fluido(gas_fluido, m_dot_mix_kg_s, m_dot_H2O_liq_kg_s, T_g_in_C, m_dot_ref, T_ref_in_TQC_C)
    except Exception as e:
        return {"erro": f"Erro no TQC: {e}"}
        
    # Saídas do TQC (Entradas para a próxima etapa)
    T_g_out_C = resultado_tqc["T_g_out_C"] # Temperatura final do gás/mix (saída do sistema)
    T_ref_in_DC_C = resultado_tqc["T_ref_out_C"] # Refrigerante quente (entrada do DC)
    
    # --- 2. Dry Cooler (DC) ---
    # Fluido Quente: Refrigerante -> Fluido Frio: Ar
    try:
        resultado_dc = modelar_dry_cooler_ar(gas_fluido, m_dot_ref, T_ref_in_DC_C, m_dot_a_op)
    except Exception as e:
        return {"erro": f"Erro no DC: {e}"}

    # A temperatura de saída do refrigerante do DC é a T_ref_in_TQC (assumimos que T_ref_in_TQC_C é um setpoint)
    # T_ref_out_DC_C = resultado_dc["T_ref_out_C"]

    # --- 3. Resultados Finais e Perdas de Pressão ---
    
    # Perda de Pressão Total (apenas fluido do processo, o gás)
    # 🛑 CÁLCULO CORRIGIDO: Usa a constante P_PERDA_BAR importada (0.05 bar)
    P_out_bar = P_in_bar - P_PERDA_BAR 
    
    # Potência total removida do fluxo do processo (Q_dot_TQC)
    Q_dot_fluxo_W = resultado_tqc["Q_dot_TQC_W"] * -1.0 # Negativo pois é calor removido do fluxo
    
    # Potência total consumida (Ventilador)
    W_dot_comp_W = resultado_dc["W_dot_ventilador_W"]
    
    # 4. Dicionário de Saída Padronizado
    results = {
        # Estado de Saída do Gás
        "T_C": T_g_out_C,
        "P_bar": P_out_bar,
        # Energia do Componente
        "Q_dot_fluxo_W": Q_dot_fluxo_W, 
        "W_dot_comp_W": W_dot_comp_W,
        # Resultados Intermediários
        "T_ref_in_DC_C": T_ref_in_DC_C,
        "Q_dot_DC_W": resultado_dc["Q_dot_DC_W"] * -1.0,
        "T_ref_out_DC_C": resultado_dc["T_ref_out_C"],
    }
    
    return results

if __name__ == '__main__':
    # Exemplo de teste unitário. A função principal deve ser capaz de lidar com erros.
    # A função original `modelar_dry_cooler` foi substituída por `simular_sistema_resfriamento`
    
    T_g_in_C_TEST = 150.0 
    P_in_bar = 1.0        # bar

    # Teste Unitário (H2 - SOEC)
    print("--- Teste Unitário (H2 - SOEC) - Sistema Água/Glicol ---")
    m_dot_mix_H2 = 0.320  # kg/s (Vazão mássica total H2+H2O)
    m_dot_H2O_liq = 0.0   # kg/s (Para simulação de desempenho)
    
    resultado_h2 = simular_sistema_resfriamento('H2', m_dot_mix_H2, m_dot_H2O_liq, P_in_bar, T_g_in_C=T_g_in_C_TEST)
    
    if "erro" not in resultado_h2:
        print(f"Temperatura de Saída H2 (Gás): {resultado_h2['T_C']:.2f} °C")
        print(f"Pressão de Saída H2 (Gás): {resultado_h2['P_bar']:.2f} bar") # 🛑 Adicionado P_out para ver a perda
        print(f"Potência Térmica Removida (TQC): {resultado_h2['Q_dot_fluxo_W']/-1000:.2f} kW")
        print(f"Potência Térmica Rejeitada (DC): {resultado_h2['Q_dot_DC_W']/-1000:.2f} kW")
        print(f"T. Glicol (TQC out / DC in): {resultado_h2['T_ref_in_DC_C']:.2f} °C")
        print(f"T. Glicol (DC out / TQC in): {resultado_h2['T_ref_out_DC_C']:.2f} °C")
        print(f"Potência Elétrica do Ventilador: {resultado_h2['W_dot_comp_W']/1000:.3f} kW")
    else:
        print(f"Erro no cálculo H2: {resultado_h2['erro']}")
        
    # Teste Unitário (O2 - SOEC)
    print("\n--- Teste Unitário (O2 - SOEC) - Sistema Água/Glicol ---")
    m_dot_mix_O2 = 0.85333  # kg/s (Vazão mássica total O2+H2O)
    m_dot_O2O_liq = 0.0     # kg/s
    
    resultado_o2 = simular_sistema_resfriamento('O2', m_dot_mix_O2, m_dot_O2O_liq, P_in_bar, T_g_in_C=T_g_in_C_TEST)
    
    if "erro" not in resultado_o2:
        print(f"Temperatura de Saída O2 (Gás): {resultado_o2['T_C']:.2f} °C")
        print(f"Pressão de Saída O2 (Gás): {resultado_o2['P_bar']:.2f} bar") # 🛑 Adicionado P_out para ver a perda
        print(f"Potência Térmica Removida (TQC): {resultado_o2['Q_dot_fluxo_W']/-1000:.2f} kW")
        print(f"Potência Térmica Rejeitada (DC): {resultado_o2['Q_dot_DC_W']/-1000:.2f} kW")
        print(f"T. Glicol (TQC out / DC in): {resultado_o2['T_ref_in_DC_C']:.2f} °C")
        print(f"T. Glicol (DC out / TQC in): {resultado_o2['T_ref_out_DC_C']:.2f} °C")
        print(f"Potência Elétrica do Ventilador: {resultado_o2['W_dot_comp_W']/1000:.3f} kW")
    else:
        print(f"Erro no cálculo O2: {resultado_o2['erro']}")