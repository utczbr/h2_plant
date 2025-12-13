# process_flow_blocks.py
# Lógica modularizada de execução de componentes para Dry Coolers, Chillers e Compressores O2.

import numpy as np
import CoolProp.CoolProp as CP

# Importações de funções de CoolProp/Estado termodinâmico
from aux_coolprop import (
    calcular_estado_termodinamico, calcular_y_H2O_inicial, verificar_limites_operacionais
)
# Importações de modelos auxiliares
from aux_models import calcular_pressao_maxima_analitica, modelar_resfriador_simples
# Importações de constantes
from constants_and_config import (
    LIMITES, T_DRY_COOLER_OUT_O2_C, T_DRY_COOLER_OUT_H2_C_DC2, T_DRENO_OUT_ALVO_C, 
    T_CHILLER_OUT_O2_C_FINAL, T_CHILLER_OUT_H2_C_C1, P_TARGET_COMPRESSOR_O2_EST2_BAR,
    P_TARGET_COMPRESSOR_O2_EST3_BAR, P_TARGET_COMPRESSOR_O2_EST4_BAR, T_CHILLER_OUT_O2_C,
    M_DOT_CHUTE_DRENO_TROC_KGS, T_CHUTE_DRENO_TROC_C, # Importa o chute inicial
    
    # NOVO: Importa os alvos de pressão do H2
    P_TARGET_COMPRESSOR_H2_EST1_BAR, P_TARGET_COMPRESSOR_H2_EST2_BAR,
    P_TARGET_COMPRESSOR_H2_EST3_BAR, P_TARGET_COMPRESSOR_H2_EST4_BAR, P_TARGET_COMPRESSOR_H2_EST5_BAR,
)

# Importações de modelos de componentes (AGORA DIRETAMENTE DO DIRETÓRIO RAIZ)
# 🛑 REMOVIDO STUB DE COMPONENTE. DEVE HAVER IMPORTAÇÃO BEM SUCEDIDA.
from modelo_chiller import modelar_chiller_gas
from modelo_compressor import modelo_compressor_ideal
# -----------------------------------------------------------------------------------


# =================================================================
# === NOVO BLOCO: TROCADOR DE CALOR (ÁGUA DRENO) - CORRIGIDO ===
# =================================================================
def executar_trocador_calor_agua_dreno(comp: str, estado_in: dict, gas_fluido: str, m_dot_H2O_liq_accomp_in: float, estado_dreno_agregado: dict, calcular_estado_termodinamico_func):
    """
    Simula um trocador de calor contrafluxo (Heat Recovery)
    onde o Fluido Quente (H2/H2O) cede calor para o Fluido Frio (Água Drenada).
    O objetivo é aquecer o Fluido Frio até T_DRENO_OUT_ALVO_C (99 °C) no máximo.
    
    O estado_dreno_agregado é o estado da água de recirculação/dreno usado na iteração.
    """
    T_h_e_C = estado_in['T_C'] # Temperatura de entrada do H2 quente
    P_h_bar = estado_in['P_bar'] # Pressão total do H2
    m_dot_mix_h = estado_in['m_dot_mix_kg_s'] # Vazão total da mistura H2/H2O
    
    # Entalpia ESPECÍFICA de entrada (J/kg)
    h_mix_h_in_J_kg = estado_in['H_mix_J_kg'] 

    # 1. Parâmetros do Fluido Frio (Água Drenada - Entrada)
    
    # Se o estado do dreno for None (e.g., primeira iteração), usa o chute inicial
    if estado_dreno_agregado is None or estado_dreno_agregado.get('M_dot_H2O_final_kg_s', 0.0) < 1e-6:
        # Usa o chute inicial da constante (128 kg/h a 20C)
        T_c_e_C = T_CHUTE_DRENO_TROC_C
        P_c_bar = LIMITES['Trocador de Calor (Água Dreno)']['P_MAX_C'] # Usamos P_out_SOEC
        m_dot_c = M_DOT_CHUTE_DRENO_TROC_KGS
        H_c_in_J_kg = CP.PropsSI('H', 'T', T_c_e_C + 273.15, 'Q', 0, 'Water') 
        print(f"[AVISO - {comp}] Usando CHUTE INICIAL: ṁ={m_dot_c*3600:.1f} kg/h @ T={T_c_e_C:.1f} °C.")

    else:
        # Usa o resultado da última simulação da linha de drenos (Mixer 1 OUT)
        T_c_e_C = estado_dreno_agregado['T'] 
        P_c_bar = estado_dreno_agregado['P_bar'] 
        m_dot_c = estado_dreno_agregado['M_dot_H2O_final_kg_s'] 
        H_c_in_J_kg = estado_dreno_agregado['H_liq_out_J_kg'] 
    
    T_c_s_alvo_C = T_DRENO_OUT_ALVO_C # Alvo: 99 °C
    
    # 2. CÁLCULO MÁXIMO DE CALOR QUE PODE SER ABSORVIDO PELO FLUIDO FRIO (Água)
    try:
        T_c_s_alvo_K = T_c_s_alvo_C + 273.15
        P_c_Pa = P_c_bar * 1e5
        H_c_s_alvo_J_kg = CP.PropsSI('H', 'T', T_c_s_alvo_K, 'Q', 0, 'Water') 
        
        Q_max_absorvivel_W = m_dot_c * (H_c_s_alvo_J_kg - H_c_in_J_kg) 
        Q_max_absorvivel_W = max(0.0, Q_max_absorvivel_W)
        
    except Exception as e:
        print(f"[ERRO COOLPROP - {comp}] Falha ao calcular H_out da água fria: {e}")
        Q_max_absorvivel_W = 0.0 

    # 3. BALANÇO DE ENERGIA (Lado Quente: H2/H2O)
    Q_cedido_W = Q_max_absorvivel_W 
    
    # 4. RESOLUÇÃO DA T DE SAÍDA DO GÁS QUENTE (Cálculo estável via Balanço Térmico)
    
    T_h_s_C = T_h_e_C 
    P_h_Pa = P_h_bar * 1e5
    
    try:
        T_h_e_K = T_h_e_C + 273.15
        
        # 4.1. CÁLCULO DO CP MÉDIO PONDERADO DO FLUXO DE GÁS (H2 + VAPOR H2O)
        w_H2 = estado_in['m_dot_gas_kg_s'] / m_dot_mix_h
        w_H2O_vap = estado_in['m_dot_H2O_vap_kg_s'] / m_dot_mix_h
        
        if w_H2 + w_H2O_vap > 0:
             Cp_H2 = CP.PropsSI('CPMASS', 'T', T_h_e_K, 'P', P_h_Pa, 'H2')
             Cp_H2O = CP.PropsSI('CPMASS', 'T', T_h_e_K, 'P', P_h_Pa, 'Water') 
             
             # Cp da mistura (média mássica ponderada)
             Cp_mix_J_kgK = (w_H2 * Cp_H2 + w_H2O_vap * Cp_H2O) / (w_H2 + w_H2O_vap)
        else:
             Cp_mix_J_kgK = 4000.0 # Fallback seguro
             
        # CÁLCULO DIRETO DO DELTA T (ΔT = Q / (ṁ * Cp))
        Delta_H_cedido_J_kg = Q_cedido_W / m_dot_mix_h if m_dot_mix_h > 0 else 0.0
        Delta_T_C = Delta_H_cedido_J_kg / Cp_mix_J_kgK
        
        T_h_s_C = T_h_e_C - Delta_T_C
        
        # 📌 Clamp físico final (ΔT mínimo de 3°C, T_h_s_C > T_c_e_C + 3.0)
        T_min_fisico = T_c_e_C + 3.0
        T_h_s_C = max(T_h_s_C, T_min_fisico) 
        
        # Se o clamp foi aplicado, ajusta Q_cedido_W para refletir o ΔT real
        if T_h_s_C > T_h_e_C: 
             Q_cedido_W = 0.0
             T_h_s_C = T_h_e_C
        elif T_h_e_C - T_h_s_C < Delta_T_C:
             # Recalcula Q cedido
             Q_cedido_W = m_dot_mix_h * Cp_mix_J_kgK * (T_h_e_C - T_h_s_C)
             Q_cedido_W = max(0.0, Q_cedido_W)
             
    except Exception as e:
        print(f"[ERRO CÁLCULO CP - {comp}] Falha crítica ao calcular T_out do gás quente: {e}. Bypassando ΔT.")
        Q_cedido_W = 0.0
        T_h_s_C = T_h_e_C
        
    # 5. RECALCULA ESTADO (para obter H_mix_J_kg correto na T_h_s_C)
    y_H2O_out_vap = estado_in['y_H2O']
    
    estado_out_calc = calcular_estado_termodinamico_func(
        gas_fluido, T_h_s_C, P_h_bar, estado_in['m_dot_gas_kg_s'], y_H2O_out_vap, estado_in['y_O2'], estado_in['y_H2']
    )
    
    # 6. CÁLCULO DA T_out DO FLUIDO FRIO (DRENO)
    H_frio_out_J_kg = H_c_in_J_kg + Q_cedido_W / m_dot_c if m_dot_c > 0 else H_c_in_J_kg
    
    try:
        T_frio_out_K = CP.PropsSI('T', 'H', H_frio_out_J_kg, 'P', P_c_Pa, 'Water')
        T_frio_out_C = T_frio_out_K - 273.15
        T_frio_out_C = min(T_c_s_alvo_C, T_frio_out_C) 
    except Exception as e:
        print(f"[AVISO COOLPROP T_frio] Falha ao calcular T_out do dreno: {e}. Usando aproximação linear.")
        T_frio_out_C = T_c_e_C + Q_cedido_W / (m_dot_c * 4183.0) 
        T_frio_out_C = min(T_c_s_alvo_C, T_frio_out_C) 
    
    
    res = {
        'Q_dot_fluxo_W': -Q_cedido_W, # Q cedido é negativo para o fluxo de H2
        'W_dot_comp_W': 0.0,
        'Agua_Condensada_kg_s': 0.0,
        'Agua_Pura_Removida_H2O_kg_s': 0.0, 
        'm_dot_frio_out_kgs': m_dot_c,
        'T_frio_out_C': T_frio_out_C,
        'H_frio_out_J_kg': H_frio_out_J_kg,
        'T_C': T_h_s_C, 
        'P_bar': P_h_bar,
        'Q_recuperado_W': Q_cedido_W # Calor Útil Recuperado
    }
    
    print(f"Executando {comp} ({gas_fluido}). T_in={T_h_e_C:.2f}°C, T_out={T_h_s_C:.2f}°C. Dreno: T_in={T_c_e_C:.2f}°C, T_out={T_frio_out_C:.2f}°C. Q̇_recuperado={Q_cedido_W/1000:.2f} kW.")

    
    m_dot_H2O_liq_accomp_out = m_dot_H2O_liq_accomp_in
    m_dot_H2O_liquida_pool_out = estado_in.get('m_dot_H2O_liq_pool_kgs', 0.0)

    # 🛑 NOVO: Extra data com o estado de saída do DRENO (lado frio)
    extra_data = {
        'Q_dot_H2_Gas': res['Q_dot_fluxo_W'], 
        'Q_dot_H2O_Total': 0.0, 
        'Dreno_m_dot_kgs': m_dot_c,
        'Dreno_T_out_C': T_frio_out_C,
        'Dreno_H_out_J_kg': H_frio_out_J_kg
    }
    
    # O estado termodinâmico retornado é o do FLUXO DE GÁS H2
    return estado_out_calc, res, m_dot_H2O_liq_accomp_out, m_dot_H2O_liquida_pool_out, extra_data


def executar_dry_cooler_estagio(comp: str, estado_in: dict, gas_fluido: str, m_dot_H2O_liq_accomp_in: float, m_dot_H2O_liquida_pool_out: float):
    """Lógica unificada para Dry Coolers O2 e H2 de estágios."""
    
    # Determina a Temperatura Alvo (T_DRY_COOLER_OUT_O2_C = 60 °C)
    T_TARGET = T_DRY_COOLER_OUT_O2_C if gas_fluido == 'O2' else T_DRY_COOLER_OUT_H2_C_DC2
    
    # Simulação do resfriamento e condensação
    res_dc_calc = modelar_resfriador_simples(
         estado_in, 
         T_target_C=T_TARGET, 
         gas_fluido=gas_fluido, 
         calcular_estado_termodinamico_func=calcular_estado_termodinamico
    )
    
    Agua_Condensada_removida_kg_s = res_dc_calc['Agua_Condensada_kg_s']
    
    # Dry Cooler: Adiciona Condensado ao líquido acompanhante.
    Agua_Removida_Componente_total = Agua_Condensada_removida_kg_s + m_dot_H2O_liq_accomp_in
    
    # 🛑 DRY COOLER NÃO DRENA. O líquido acumulado APENAS SEGUE para o próximo KOD.
    
    m_dot_H2O_liq_accomp_out = m_dot_H2O_liq_accomp_in + Agua_Condensada_removida_kg_s 
    
    estado_atual = res_dc_calc['estado_termodinamico']
    
    res = {
        'Q_dot_fluxo_W': res_dc_calc['Q_dot_fluxo_W'],
        'Agua_Condensada_kg_s': Agua_Condensada_removida_kg_s,
        'Agua_Pura_Removida_H2O_kg_s': 0.0, # Nenhuma remoção do Pool aqui
        'W_dot_comp_W': 0.0,
        'T_cold_out_C': T_TARGET,
        'm_dot_cold_liq_kg_s': 0.0, 
        'Gas_Dissolvido_removido_kg_s': 0.0, 
        'm_dot_cold_H2O_pura_kg_s': 0.0,
        'T_C': estado_atual['T_C'], 
        'P_bar': estado_atual['P_bar']
    }
    
    return estado_atual, res, m_dot_H2O_liq_accomp_out, m_dot_H2O_liquida_pool_out, {} # Retorna estado, resultado, vazões e extra_data


def executar_chiller_o2_estagio(comp: str, estado_in: dict, gas_fluido: str, m_dot_H2O_liq_accomp_in: float, m_dot_H2O_liquida_pool_out: float):
    """Lógica unificada para Chillers de O2 (4 °C) e Chillers H2 de estágios (4 °C)."""
    
    # Determina a Temperatura Alvo (T_CHILLER_OUT_O2_C_FINAL = 4 °C)
    T_TARGET_CH = T_CHILLER_OUT_O2_C_FINAL # 4 °C para O2 e H2 (Estágios)
    if comp == 'Chiller 1':
        T_TARGET_CH = T_CHILLER_OUT_H2_C_C1 # 4 °C para H2 (Estágio 1/Chiller 1)
        
    print(f"Executando {comp} ({gas_fluido}). T_target={T_TARGET_CH:.1f} °C.")
    
    # 1. Simula o resfriador simples para obter a condensação potencial na T alvo
    res_resf_calc = modelar_resfriador_simples(estado_in, T_TARGET_CH, gas_fluido, calcular_estado_termodinamico)
    Agua_Condensada = res_resf_calc['Agua_Condensada_kg_s']
    
    # 2. Simula o chiller (para obter W_dot e Q_dot)
    estado_out_ch_calc = res_resf_calc['estado_termodinamico']
    
    res_ch = modelar_chiller_gas(
        gas_fluido, 
        estado_in['m_dot_mix_kg_s'] + m_dot_H2O_liq_accomp_in, 
        estado_in['P_bar'], 
        estado_in['T_C'], 
        T_out_C_desejada=T_TARGET_CH, 
        H_in_J_kg=estado_in['H_in_mix_J_kg'], 
        H_out_J_kg=estado_out_ch_calc['H_in_mix_J_kg'], 
        y_H2O_in=estado_in['y_H2O']
    )
    
    # 3. Balanço de massa: Chillers NÃO DRENAM. Apenas condensam e o líquido segue.
    m_dot_H2O_liq_accomp_out = m_dot_H2O_liq_accomp_in + Agua_Condensada 
    estado_atual = estado_out_ch_calc
    
    # 4. Ajusta chaves de resultado
    res = {
        'Q_dot_fluxo_W': res_ch['Q_dot_fluxo_W'], 
        'W_dot_comp_W': res_ch.get('W_chiller_W', 0.0), 
        'Agua_Condensada_kg_s': Agua_Condensada, 
        'Agua_Pura_Removida_H2O_kg_s': 0.0, # Chillers não drenam
        'T_cold_out_C': T_TARGET_CH, 
        'Gas_Dissolvido_removido_kg_s': 0.0, 
        'm_dot_cold_H2O_pura_kg_s': 0.0,
        'T_C': estado_atual['T_C'], 
        'P_bar': estado_atual['P_bar']
    }
    extra_data = {'Q_dot_H2_Gas': res['Q_dot_fluxo_W'], 'Q_dot_H2O_Total': 0.0}
    
    return estado_atual, res, m_dot_H2O_liq_accomp_out, m_dot_H2O_liquida_pool_out, extra_data


def executar_compressor_h2_estagio(comp: str, estado_in: dict, m_dot_H2O_liq_accomp_in: float):
    """Lógica unificada para Compressores H2 (Estágios 1 a 5)."""
    
    # 1. Determina a Pressão Alvo e Limite de Temperatura
    T_ALVO_MAX = LIMITES[comp]['T_MAX_C'] 
    
    # Mapeamento das pressões alvo dos estágios H2
    mapa_pressoes_h2 = {
        'Compressor H2 (Estágio 1)': P_TARGET_COMPRESSOR_H2_EST1_BAR,
        'Compressor H2 (Estágio 2)': P_TARGET_COMPRESSOR_H2_EST2_BAR,
        'Compressor H2 (Estágio 3)': P_TARGET_COMPRESSOR_H2_EST3_BAR,
        'Compressor H2 (Estágio 4)': P_TARGET_COMPRESSOR_H2_EST4_BAR,
        'Compressor H2 (Estágio 5)': P_TARGET_COMPRESSOR_H2_EST5_BAR,
    }
    
    P_ALVO = mapa_pressoes_h2.get(comp, 9999.0)

    # 2. Calcula a P_max analítica baseada na T_max
    P_out_max_bar = calcular_pressao_maxima_analitica(
        estado_in, 
        fluido_nome='hydrogen', # Gás Principal é H2
        T_alvo_max_C=T_ALVO_MAX 
    )
    
    # 3. Limita a pressão à P_alvo de projeto (ou P_max analítica se for menor)
    P_out_final_bar = min(P_out_max_bar, P_ALVO)
    
    res = {'erro': None}
    
    # Lógica de falha/desligamento se P_out não for maior que P_in
    if P_out_final_bar <= estado_in['P_bar'] + 1e-6:
        # Se a P_in já é a P_alvo ou T_max restringe, P_out é igual a P_in.
        # 🛑 REMOVIDO FALLBACK: FORÇA ERRO SE NÃO HÁ COMPRESSÃO
        raise ValueError(f"ERRO DE COMPRESSÃO: P_out ({P_out_final_bar:.2f} bar) não é maior que P_in ({estado_in['P_bar']:.2f} bar). Compressão desabilitada para {comp}.")
    else:
        print(f"Executando {comp}: P_in={estado_in['P_bar']:.2f} bar. P_out_alvo: {P_ALVO:.2f} bar. P_out_final: {P_out_final_bar:.2f} bar.")
        
        # Simula o compressor ideal/isentrópico
        res_comp_analitico = modelo_compressor_ideal(
            fluido_nome='hydrogen', # Gás Principal é H2
            T_in_C=estado_in['T_C'], 
            P_in_Pa=estado_in['P_bar'] * 1e5, 
            P_out_Pa=P_out_final_bar * 1e5, 
            m_dot_mix_kg_s=estado_in['m_dot_mix_kg_s'], 
            m_dot_gas_kg_s=estado_in['m_dot_gas_kg_s']
        )
        
        # 🛑 REMOVIDO FALLBACK: O modelo_compressor_ideal deve falhar se houver erro.
        
        res = res_comp_analitico
        T_out_C = res['T_C']; P_out_bar = res['P_bar']
        
        # Recalcula estado termodinâmico para atualizar y_H2O (pressão parcial)
        y_H2O_out_sat = calcular_y_H2O_inicial(T_out_C, P_out_bar) 
        y_H2O_out_vap = min(y_H2O_out_sat, estado_in['y_H2O']) # O vapor não condensa no compressor ideal
        
        estado_atual = calcular_estado_termodinamico('H2', T_out_C, P_out_bar, estado_in['m_dot_gas_kg_s'], y_H2O_out_vap, estado_in['y_O2'], estado_in['y_H2'])
        m_dot_H2O_liquida_pool_out = estado_in.get('m_dot_H2O_liq_pool_kgs', 0.0)

    
    # O compressor não remove líquido, apenas aquece e pressuriza
    m_dot_H2O_liq_accomp_out = m_dot_H2O_liq_accomp_in
    
    res['Agua_Pura_Removida_H2O_kg_s'] = 0.0
    res['Agua_Condensada_kg_s'] = 0.0
    res['Q_dot_fluxo_W'] = 0.0 
    res['W_dot_comp_W'] = res.get('W_dot_comp_W', 0.0)
    
    return estado_atual, res, m_dot_H2O_liq_accomp_out, m_dot_H2O_liquida_pool_out, {}


def executar_compressor_o2_estagio(comp: str, estado_in: dict, m_dot_H2O_liq_accomp_in: float):
    """Lógica unificada para Compressores O2 (Estágio 1, 2, 3, 4)."""
    
    # 1. Determina a Pressão Alvo e Limite de Temperatura
    T_ALVO_MAX = LIMITES[comp]['T_MAX_C'] 
    
    if comp == 'Compressor O2 (Estágio 2)':
        P_ALVO = P_TARGET_COMPRESSOR_O2_EST2_BAR
    elif comp == 'Compressor O2 (Estágio 3)':
        P_ALVO = P_TARGET_COMPRESSOR_O2_EST3_BAR
    elif comp == 'Compressor O2 (Estágio 4)':
        P_ALVO = P_TARGET_COMPRESSOR_O2_EST4_BAR
    else: # Estágio 1
        P_ALVO = 9999.0 

    # 2. Calcula a P_max analítica baseada na T_max
    P_out_max_bar = calcular_pressao_maxima_analitica(
        estado_in, 
        fluido_nome='oxygen', 
        T_alvo_max_C=T_ALVO_MAX 
    )
    
    # 3. Limita a pressão à P_alvo de projeto (ou P_max analítica se for menor)
    P_out_final_bar = min(P_out_max_bar, P_ALVO)
    
    res = {'erro': None}
    
    if P_out_final_bar <= estado_in['P_bar'] + 1e-6:
        # 🛑 REMOVIDO FALLBACK: FORÇA ERRO SE NÃO HÁ COMPRESSÃO
        raise ValueError(f"ERRO DE COMPRESSÃO: P_out ({P_out_final_bar:.2f} bar) não é maior que P_in ({estado_in['P_bar']:.2f} bar). Compressão desabilitada para {comp}.")
    else:
        print(f"Executando {comp}: P_in={estado_in['P_bar']:.2f} bar. P_out_alvo: {P_ALVO:.2f} bar. P_out_final: {P_out_final_bar:.2f} bar.")
        
        # Simula o compressor ideal/isentrópico
        res_comp_analitico = modelo_compressor_ideal(
            fluido_nome='oxygen', 
            T_in_C=estado_in['T_C'], 
            P_in_Pa=estado_in['P_bar'] * 1e5, 
            P_out_Pa=P_out_final_bar * 1e5, 
            m_dot_mix_kg_s=estado_in['m_dot_mix_kg_s'], 
            m_dot_gas_kg_s=estado_in['m_dot_gas_kg_s']
        )
        
        # 🛑 REMOVIDO FALLBACK: O modelo_compressor_ideal deve falhar se houver erro.
        
        res = res_comp_analitico
        T_out_C = res['T_C']; P_out_bar = res['P_bar']
        
        # Recalcula estado termodinâmico para atualizar y_H2O (pressão parcial)
        y_H2O_out_sat = calcular_y_H2O_inicial(T_out_C, P_out_bar) 
        y_H2O_out_vap = min(y_H2O_out_sat, estado_in['y_H2O']) # O vapor não condensa no compressor ideal
        
        estado_atual = calcular_estado_termodinamico('O2', T_out_C, P_out_bar, estado_in['m_dot_gas_kg_s'], y_H2O_out_vap, estado_in['y_O2'], estado_in['y_H2'])
        m_dot_H2O_liquida_pool_out = estado_in.get('m_dot_H2O_liq_pool_kgs', 0.0)

    
    # O compressor não remove líquido, apenas aquece e pressuriza
    m_dot_H2O_liq_accomp_out = m_dot_H2O_liq_accomp_in
    
    res['Agua_Pura_Removida_H2O_kg_s'] = 0.0
    res['Agua_Condensada_kg_s'] = 0.0
    res['Q_dot_fluxo_W'] = 0.0 
    res['W_dot_comp_W'] = res.get('W_dot_comp_W', 0.0)
    
    return estado_atual, res, m_dot_H2O_liq_accomp_out, m_dot_H2O_liquida_pool_out, {}