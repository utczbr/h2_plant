# process_execution.py
# Contém a função principal de simulação componente por componente (simular_sistema)

import numpy as np
import pandas as pd
import sys
import CoolProp.CoolProp as CP
from datetime import datetime

# Importações de constantes e funções auxiliares
from constants_and_config import (
    LIMITES, T_IN_C, P_IN_BAR, 
    T_CHILLER_OUT_H2_C_C1, T_CHILLER_OUT_O2_C, Y_H2O_LIMIT_MOLAR,
    COMPONENTS_H2, COMPONENTS_O2, P_OUT_VALVULA_O2_BAR, P_VSA_PROD_BAR, P_VSA_REG_BAR,
    M_DOT_H2O_RECIRC_TOTAL_KGS, M_DOT_H2O_CONSUMIDA_KGS, M_DOT_G_H2, M_DOT_G_O2,
    FATOR_CROSSOVER_H2, LIMITE_LIQUIDO_DEMISTER_G_NM3, V_MOLAR_PADRAO_NM3_KMOL, 
    MM_H2_CALC, MM_O2_CALC,
    M_H2O_TOTAL_H2_KGS, M_H2O_TOTAL_O2_KGS, # Novos valores de água total por fluxo
    M_DOT_REF_H2, M_DOT_REF_O2 
)
from aux_coolprop import (
    calcular_estado_termodinamico, calcular_y_H2O_inicial, verificar_limites_operacionais
)

# 💥 CORREÇÃO: Importações dos Modelos de Componentes
try:
    from modulos.modelo_dry_cooler import modelar_dry_cooler
    from modulos.modelo_chiller import modelar_chiller_gas
    from modulos.modelo_kod import modelar_knock_out_drum
    from modulos.modelo_coalescedor import modelar_coalescedor 
    from modulos.modelo_deoxo import modelar_deoxo 
    from modulos.modelo_psa import modelar_psa 
    from modulos.modelo_valvula import modelo_valvula_isoentalpica
    
    # 🌟 NOVO: Importação do Aquecedor Imaginário
    from modulos.modelo_aquecedor_imaginario import modelar_aquecedor_imaginario
except ImportError as e:
    raise ImportError(f"Falha na importação do modelo de componente em process_execution: {e}")


def calcular_vazao_demister(m_dot_gas_kg_s: float, mm_gas_kg_kmol: float) -> float:
# ... (Corpo da função inalterado) ...
    """Calcula a vazão máxima de líquido (kg/s) permitida pelo demister (20 mg/Nm³)."""
    # 1. Vazão Molar (kmol/s)
    F_molar_kmol_s = m_dot_gas_kg_s / mm_gas_kg_kmol
    # 2. Vazão Volumétrica Padrão (Nm³/s)
    # CORREÇÃO: Removida a divisão por 3600.0, garantindo o cálculo de Nm³/s a partir de kg/s.
    Q_vol_nm3_s = F_molar_kmol_s * V_MOLAR_PADRAO_NM3_KMOL
    # 3. Massa Líquida Máxima (kg/s)
    M_liq_max_kg_s = Q_vol_nm3_s * (LIMITE_LIQUIDO_DEMISTER_G_NM3 / 1000.0)
    return M_liq_max_kg_s


def simular_sistema(gas_fluido: str, m_dot_g_kg_s: float, deoxo_mode: str, L_deoxo: float, dc2_mode: str, comp_c1: str, m_dot_H2O_total_fluxo_input: float, y_O2_in: float, y_H2_in: float, m_dot_liq_max_h2_ref: float = None):
# ... (código inicial inalterado) ...
    """
    Executa a simulação componente por componente para um dado fluxo de gás.
    
    m_dot_H2O_total_fluxo_input: É o M_H2O_TOTAL_H2_KGS ou M_H2O_TOTAL_O2_KGS, 
    que inclui vapor e líquido arraste, mas ainda não foi separado.
    m_dot_liq_max_h2_ref: Valor de referência do M_DOT_LIQ_MAX_DEMISTER_KGS do H2.
    """
    
    # --- 1. CONFIGURAÇÃO BASE DO FLUXO ---
    if gas_fluido == 'H2':
        MM_GAS = MM_H2_CALC
        M_DOT_GAS = M_DOT_G_H2
        base_list = COMPONENTS_H2 
        M_DOT_REF = M_DOT_REF_H2 
    else: # O2
        MM_GAS = MM_O2_CALC
        M_DOT_GAS = M_DOT_G_O2
        base_list = COMPONENTS_O2 
        M_DOT_REF = M_DOT_REF_O2 
    
    M_H2O_TOTAL_FLUXO = m_dot_H2O_total_fluxo_input # Vazão total de H2O (vapor + líquido) que entra na linha de purificação
    
    # 1.2. PARÂMETROS TERMODINÂMICOS DE ENTRADA
    y_H2O_in_sat = calcular_y_H2O_inicial(T_IN_C, P_IN_BAR)
    
    # 1.3. CÁLCULO DA PARTIÇÃO INICIAL (Vapor, Arraste e Dreno PEM)
    
    # 1.3.a. Vapor Saturado (limite máximo de H2O que o gás pode carregar)
    estado_sat_calc = calcular_estado_termodinamico(gas_fluido, T_IN_C, P_IN_BAR, M_DOT_GAS, y_H2O_in_sat, y_O2_in, y_H2_in)
    M_DOT_VAPOR_SAT_KGS = estado_sat_calc['m_dot_H2O_vap_kg_s']
    
    # 1.3.b. Líquido Acompanhante Calculado (z/w) - Valor determinado pelo critério do demister (20 mg/Nm³)
    M_DOT_LIQ_MAX_DEMISTER_KGS = calcular_vazao_demister(M_DOT_GAS, MM_GAS)
    
    # 🌟 CORREÇÃO FORÇADA: Se O2, usamos exatamente metade do valor de H2 (conforme solicitado).
    if gas_fluido == 'O2' and m_dot_liq_max_h2_ref is not None:
         # O valor de M_DOT_LIQ_MAX_DEMISTER_KGS para O2 é agora forçado a ser M_DOT_LIQ_MAX_DEMISTER_KGS_H2 / 2.0
         M_DOT_LIQ_MAX_DEMISTER_KGS = m_dot_liq_max_h2_ref / 2.0
    
    # 1.3.c. Partição de Entrada
    
    # 🌟 1. VAPOR NA ENTRADA: MÍNIMO entre o que está disponível e a saturação
    M_DOT_VAPOR_ENTRADA_KGS = min(M_H2O_TOTAL_FLUXO, M_DOT_VAPOR_SAT_KGS)
    
    # 🌟 2. LÍQUIDO ARRASTE TOTAL DISPONÍVEL (Excessso que não virou vapor)
    M_DOT_LIQ_TOTAL_ARRAS_KGS = max(0.0, M_H2O_TOTAL_FLUXO - M_DOT_VAPOR_ENTRADA_KGS)
    
    # 🌟 3. ARRASTE QUE SEGUE PARA O KOD 1 (Líquido Acompanhante - z ou w)
    # Valor é definido pelo cálculo do demister (que foi forçado/calculado corretamente) 
    # e limitado pela disponibilidade (M_DOT_LIQ_TOTAL_ARRAS_KGS).
    m_dot_H2O_liq_in_arraste = min(M_DOT_LIQ_MAX_DEMISTER_KGS, M_DOT_LIQ_TOTAL_ARRAS_KGS)

    # 🌟 4. DRENO DO PEM (Residual do balanço de massa, conforme a lógica do usuário)
    # Dreno = Total H2O - Vapor Saturado (x/y) - Líquido Acompanhante (z/w)
    M_DOT_DRENO_PEM_KGS = max(0.0, M_H2O_TOTAL_FLUXO - M_DOT_VAPOR_ENTRADA_KGS - m_dot_H2O_liq_in_arraste)
    
    # Novo y_H2O de entrada (Baseado no M_DOT_VAPOR_ENTRADA_KGS real)
    M_H2O = CP.PropsSI('M', 'Water'); M_GAS_CP = CP.PropsSI('M', gas_fluido)
    if M_DOT_VAPOR_ENTRADA_KGS > 0:
        w_H2O_in_vap = M_DOT_VAPOR_ENTRADA_KGS / (M_DOT_GAS + M_DOT_VAPOR_ENTRADA_KGS)
        y_H2O_in_calc = (w_H2O_in_vap / M_H2O) / (w_H2O_in_vap / M_H2O + (1-w_H2O_in_vap) / M_GAS_CP)
    else:
        y_H2O_in_calc = 0.0

    # --- FIM DO CÁLCULO DE BALANÇO DE ÁGUA DE ENTRADA ---
    
    
    # 2. ESTADO DE ENTRADA (INICIAL)
    estado_atual = calcular_estado_termodinamico(gas_fluido, T_IN_C, P_IN_BAR, M_DOT_GAS, y_H2O_in_calc, y_O2_in, y_H2_in)
    
    
    # 🌟 INVENTÁRIO RASTREADOR DE ÁGUA LÍQUIDA A PARTIR DA QUAL O VAPOR É GERADO
    # O Pool Inicial é a água líquida total que entrou (M_H2O_TOTAL_FLUXO) menos o que saiu no Dreno PEM.
    # O consumo estequiométrico foi deduzido antes do cálculo de M_H2O_TOTAL_FLUXO (em constants_and_config)
    m_dot_H2O_liquida_pool_kgs = M_H2O_TOTAL_FLUXO - M_DOT_DRENO_PEM_KGS
    
    # 3. A lista de componentes a simular (sem a 'Entrada')
    component_list_filtered = [comp for comp in base_list if comp != 'Entrada']
    
    history = [
        {
            **estado_atual,
            'Componente': 'Entrada',
            'Q_dot_fluxo_W': 0.0, 'W_dot_comp_W': 0.0, 'Agua_Condensada_kg_s': 0.0,
            'm_dot_H2O_vap_kg_s': estado_atual['m_dot_H2O_vap_kg_s'], 'W_dot_regen_W': 0.0,
            'm_dot_H2O_liq_accomp_kg_s': m_dot_H2O_liq_in_arraste, # Arraste que segue para o KOD 1
            'Status_KOD': 'N/A', 
            'T_profile_C': None, 'L_span': None, 'X_O2': 0.0, 'T_max_calc': None,
            'Q_dot_H2_Gas': 0.0, 'Q_dot_H2O_Total': 0.0,
            'T_cold_out_C': T_IN_C, 'm_dot_cold_liq_kg_s': 0.0, 
            'Gas_Dissolvido_removido_kg_s': 0.0, 
            'Agua_Pura_Removida_H2O_kg_s': M_DOT_DRENO_PEM_KGS, # Dreno coletado no PEM (que vai para o Mixer)
            'm_dot_cold_H2O_pura_kg_s': 0.0, 
            'm_dot_H2O_liq_pool_kgs': m_dot_H2O_liquida_pool_kgs, # 🌟 ESTOQUE INICIAL
            # --- NOVAS VARIÁVEIS DE BALANÇO DE ÁGUA NA ENTRADA (Para exibição) ---
            'M_DOT_VAPOR_ENTRADA_KGS_X_Y': M_DOT_VAPOR_ENTRADA_KGS, 
            'M_DOT_LIQ_ACOMP_KGS_Z_W': m_dot_H2O_liq_in_arraste, 
            'M_DOT_LIQ_ARRAS_TOTAL_KGS': M_DOT_LIQ_TOTAL_ARRAS_KGS, # Total que deveria ser arraste
            'M_DOT_LIQ_MAX_DEMISTER_KGS': M_DOT_LIQ_MAX_DEMISTER_KGS, # Limite do demister
        }
    ]
    
    deoxo_L_span, deoxo_T_profile_C, deoxo_X_O2, deoxo_T_max_calc = None, None, None, None

    # --------------------------------------------------------------------------
    # 4. LOOP DE SIMULAÇÃO
    # --------------------------------------------------------------------------
    
    for comp in component_list_filtered:
            
        estado_in = history[-1].copy()
        m_dot_H2O_liq_accomp_in = estado_in.get('m_dot_H2O_liq_accomp_kg_s', 0.0) 
        m_dot_H2O_liquida_pool_in = estado_in.get('m_dot_H2O_liq_pool_kgs', 0.0)
        m_dot_H2O_liq_accomp_out = m_dot_H2O_liq_accomp_in # Inicializa com o valor de entrada
        m_dot_gas_in_princ = estado_in['m_dot_gas_kg_s'] 
        
        # Inicializa o dicionário de resultados
        res = {'T_C': estado_in['T_C'], 'P_bar': estado_in['P_bar'], 'Q_dot_fluxo_W': 0.0, 'W_dot_comp_W': 0.0,
               'Agua_Condensada_kg_s': 0.0, 'T_cold_out_C': estado_in['T_C'], 'm_dot_cold_liq_kg_s': 0.0,
               'Gas_Dissolvido_removido_kg_s': 0.0, 'Agua_Pura_Removida_H2O_kg_s': 0.0, 'm_dot_cold_H2O_pura_kg_s': 0.0}
        extra_data = {} 
        
        perda_gas_dissolvido_kg_s = 0.0
        m_dot_H2O_liquida_pool_out = m_dot_H2O_liquida_pool_in 
        
        # --- MODELOS DE PROCESSO ---
        
        # 🌟 NOVO COMPONENTE: AQUECEDOR IMAGINÁRIO (SÓ NO FLUXO H2, ANTES DO DEOXO)
        if comp == 'Aquecedor Imaginário':
            # 💥 NOVO: Chamando o Aquecedor Imaginário
            T_OUT_ALVO = 40.0
            res_heater = modelar_aquecedor_imaginario(estado_in, T_out_C_alvo=T_OUT_ALVO)
            
            if 'erro' in res_heater and res_heater.get('Q_dot_fluxo_W', 0.0) == 0.0:
                 # Tratamento para quando a temperatura de entrada já é alta o suficiente
                 print(f"!!! ALERTA em {comp} ({gas_fluido})! Mensagem: {res_heater['erro']} !!!")
                 res = res_heater
                 estado_atual = estado_in
                 m_dot_H2O_liq_accomp_out = res_heater.get('m_dot_H2O_liq_accomp_out_kg_s', m_dot_H2O_liq_accomp_in)
            else:
                 res = res_heater
                 T_out_C = res['T_C']; P_out_bar = res['P_bar']

                 # Recalcula o estado para a T_out, P_out (confirmando o y_H2O/saturação)
                 y_H2O_out_sat = calcular_y_H2O_inicial(T_out_C, P_out_bar) 
                 y_H2O_out_vap = min(estado_in['y_H2O'], y_H2O_out_sat)
                 
                 estado_atual = calcular_estado_termodinamico(gas_fluido, T_out_C, P_out_bar, estado_in['m_dot_gas_kg_s'], y_H2O_out_vap, estado_in['y_O2'], estado_in['y_H2'])
                 m_dot_H2O_liq_accomp_out = res.get('m_dot_H2O_liq_accomp_out_kg_s', m_dot_H2O_liq_accomp_in)
            
            res['T_cold_out_C'] = estado_in['T_C']; res['m_dot_cold_liq_kg_s'] = 0.0 
            res['Gas_Dissolvido_removido_kg_s'] = 0.0; res['Agua_Pura_Removida_H2O_kg_s'] = 0.0; res['m_dot_cold_H2O_pura_kg_s'] = 0.0
            
        elif comp == 'Deoxo':
            # 💥 CORREÇÃO: Chamando a função do novo nome 'modelar_deoxo'
            res_deoxo = modelar_deoxo(estado_in['m_dot_gas_kg_s'], estado_in['P_bar'], estado_in['T_C'], estado_in['y_H2O'], estado_in['y_O2'], L_deoxo)
            if 'erro' in res_deoxo: 
                print(f"!!! ERRO em {comp} ({gas_fluido})! Mensagem: {res_deoxo['erro']} !!!")
                continue 
            res = res_deoxo
            
            m_dot_gas_out_princ = res['m_dot_gas_out_kg_s'] 
            T_MAX_CALCULADA = res.get('T_max_calc', res['T_C']) 
            if T_MAX_CALCULADA > LIMITES['Deoxo']['T_MAX_C']: verificar_limites_operacionais('Deoxo', estado_in, T_max_calc=T_MAX_CALCULADA)
            
            # 🌟 ATUALIZAÇÃO DO ESTOQUE (Água Gerada)
            Agua_Gerada_Deoxo = res.get('Agua_Gerada_kg_s', 0.0)
            m_dot_H2O_liquida_pool_out += Agua_Gerada_Deoxo 
            
            # Novo y_H2O de saída: min(y_H2O_out_deoxo, saturação da T_out)
            y_H2O_next_stage_sat = calcular_y_H2O_inicial(res['T_C'], res['P_bar'])
            y_H2O_out_vap = min(res['y_H2O_out'], y_H2O_next_stage_sat)
            
            estado_atual = calcular_estado_termodinamico(gas_fluido, res['T_C'], res['P_bar'], m_dot_gas_out_princ, y_H2O_out_vap, res['y_O2_out'], estado_in['y_H2'])
            
            # 💥 CORREÇÃO (DEOXO): Água Líquida Acompanhante AUMENTA pela água gerada
            m_dot_H2O_liq_accomp_out = m_dot_H2O_liq_accomp_in + Agua_Gerada_Deoxo 
            
            deoxo_L_span = res.get('L_span', None); deoxo_T_profile_C = res.get('T_profile_C', None); 
            deoxo_X_O2 = res.get('X_O2', 0.0); deoxo_T_max_calc = T_MAX_CALCULADA
            
            res['T_cold_out_C'] = estado_in['T_C'] ; res['m_dot_cold_liq_kg_s'] = 0.0 
            res['Gas_Dissolvido_removido_kg_s'] = 0.0; res['Agua_Pura_Removida_H2O_kg_s'] = 0.0
            res['m_dot_cold_H2O_pura_kg_s'] = 0.0
            
        elif comp == 'PSA': 
            # 💥 CORREÇÃO: Chamando a função do novo nome 'modelar_psa'
            res_psa = modelar_psa(estado_in['m_dot_mix_kg_s'], estado_in['P_bar'], estado_in['T_C'], estado_in['y_H2O'], estado_in['y_O2'])
            if 'erro' in res_psa: 
                print(f"!!! ERRO em {comp} ({gas_fluido})! Mensagem: {res_psa['erro']} !!!")
                continue 
                
            m_dot_gas_out_princ = res_psa['m_dot_gas_out_kg_s'] 
            perda_gas_purga_kg_s = estado_in['m_dot_gas_kg_s'] - m_dot_gas_out_princ
            perda_gas_dissolvido_kg_s = perda_gas_purga_kg_s
            res['Gas_Dissolvido_removido_kg_s'] = perda_gas_dissolvido_kg_s
            
            y_H2O_out_vap = Y_H2O_LIMIT_MOLAR 
            
            # --- ATUALIZADO: Usando a pressão de saída calculada no modelo_psa (com Delta P fixo) ---
            P_out_bar = res_psa['P_out_bar']; 
            T_out_C = res_psa['T_C']         
            # ----------------------------------------------------------------------------------------
            
            # 🌟 ATUALIZAÇÃO DO ESTOQUE (Água Removida)
            Agua_Removida_PSA_Pool = res_psa.get('Agua_Removida_kg_s', 0.0) + m_dot_H2O_liq_accomp_in
            m_dot_H2O_liquida_pool_out -= Agua_Removida_PSA_Pool
            
            estado_atual = calcular_estado_termodinamico(gas_fluido, T_out_C, P_out_bar, m_dot_gas_out_princ, y_H2O_out_vap, estado_in['y_O2'], estado_in['y_H2'])
            Agua_Removida_PSA = res_psa.get('Agua_Removida_kg_s', 0.0)
            res['Agua_Condensada_kg_s'] = Agua_Removida_PSA
            # 💥 CORREÇÃO (PSA): Todo o líquido acompanhante e o vapor condensado são removidos.
            m_dot_H2O_liq_accomp_out = 0.0
            
            res = {'T_C': T_out_C, 'P_bar': P_out_bar, 'Q_dot_fluxo_W': res_psa.get('Q_dot_fluxo_W', 0.0), 'W_dot_comp_W': res_psa['W_dot_comp_W'],
                   'Agua_Condensada_kg_s': Agua_Removida_PSA, 'T_cold_out_C': T_out_C, 'm_dot_cold_liq_kg_s': 0.0,
                   'Gas_Dissolvido_removido_kg_s': perda_gas_dissolvido_kg_s, 'Agua_Pura_Removida_H2O_kg_s': Agua_Removida_PSA, 'm_dot_cold_H2O_pura_kg_s': 0.0}
            extra_data = {}
            
        elif comp == 'Dry Cooler 1':
            # 💥 CORREÇÃO: Chamando a função do novo nome 'modelar_dry_cooler' (que agora é o wrapper para o sistema TQC + DC)
            # A função modelar_dry_cooler agora retorna Q_dot_fluxo_W e W_dot_comp_W do sistema Dry Cooler + TQC.
            res_dc = modelar_dry_cooler(gas_fluido, estado_in['m_dot_mix_kg_s'], m_dot_H2O_liq_accomp_in, estado_in['P_bar'], estado_in['T_C'])
            if 'erro' in res_dc: 
                print(f"!!! ERRO em {comp} ({gas_fluido})! Mensagem: {res_dc['erro']} !!!")
                continue 
            res = res_dc
            
            T_out_C = res['T_C']; P_out_bar = res['P_bar']
            
            # --- CÁLCULO DA CONDENSAÇÃO E ATUALIZAÇÃO DO ESTOQUE ---
            
            # Novo limite de saturação na T_out do Dry Cooler
            y_H2O_out_sat = calcular_y_H2O_inicial(T_out_C, P_out_bar) 
            # O vapor de saída é o mínimo entre o y_H2O de entrada e a saturação na T_out
            y_H2O_out_vap = min(y_H2O_out_sat, estado_in['y_H2O'])
            
            # Calcula o estado termodinâmico para obter a vazão de vapor de saída
            estado_atual_calc = calcular_estado_termodinamico(gas_fluido, T_out_C, P_out_bar, estado_in['m_dot_gas_kg_s'], y_H2O_out_vap, estado_in['y_O2'], estado_in['y_H2'])
            
            m_dot_H2O_vap_out = estado_atual_calc['m_dot_H2O_vap_kg_s']
            m_dot_H2O_vap_in = estado_in['m_dot_H2O_vap_kg_s']
            
            # Água Condensada no Dry Cooler (Vapor In - Vapor Out)
            Agua_Condensada_DC = max(0.0, m_dot_H2O_vap_in - m_dot_H2O_vap_out)
            
            # 🌟 ATUALIZAÇÃO DO ESTOQUE (Vapor condensado volta para o pool líquido)
            m_dot_H2O_liquida_pool_out += Agua_Condensada_DC 
            
            # 💥 CORREÇÃO (DRY COOLER): Água Líquida Acompanhante AUMENTA pela condensação
            m_dot_H2O_liq_accomp_out = m_dot_H2O_liq_accomp_in + Agua_Condensada_DC 
            
            estado_atual = estado_atual_calc 
            
            # Q-dot breakdown (mantido para plots de energia)
            w_H2O_total_in = (estado_in['m_dot_H2O_vap_kg_s'] + estado_in['m_dot_H2O_liq_accomp_kg_s']) / estado_in['m_dot_mix_kg_s']
            Q_dot_H2O_Total_W = res['Q_dot_fluxo_W'] * w_H2O_total_in; Q_dot_gas_sensivel_W = res['Q_dot_fluxo_W'] * (1.0 - w_H2O_total_in)
            extra_data['Q_dot_H2_Gas'] = Q_dot_gas_sensivel_W; extra_data['Q_dot_H2O_Total'] = Q_dot_H2O_Total_W
            
            # Resfriamento/Aquecimento da água de resfriamento (glicol)
            T_ref_out_DC_C = res.get('T_ref_out_DC_C', T_CHILLER_OUT_H2_C_C1 if gas_fluido == 'H2' else T_CHILLER_OUT_O2_C)
            
            # Vazão de refrigerante (NOVO: Usando a constante importada M_DOT_REF)
            m_dot_ref = M_DOT_REF
            
            res['T_cold_out_C'] = T_ref_out_DC_C; 
            res['m_dot_cold_liq_kg_s'] = m_dot_ref 
            
            # O Dry Cooler não drena (o KOD 2 e o Coalescedor fazem isso)
            res['Agua_Condensada_kg_s'] = 0.0 # A condensação apenas aumenta o arraste, o KOD 2 drena
            res['Gas_Dissolvido_removido_kg_s'] = 0.0; res['Agua_Pura_Removida_H2O_kg_s'] = 0.0; res['m_dot_cold_H2O_pura_kg_s'] = 0.0

        elif comp == 'Chiller 1':
            T_CHILLER_TARGET_LOCAL = T_CHILLER_OUT_H2_C_C1 if gas_fluido == 'H2' else T_CHILLER_OUT_O2_C
            
            T_out_C = T_CHILLER_TARGET_LOCAL; P_out_bar = estado_in['P_bar'] 
            y_H2O_out_sat = calcular_y_H2O_inicial(T_out_C, P_out_bar) 
            y_H2O_out_vap = min(y_H2O_out_sat, estado_in['y_H2O'])
            
            estado_out_ch1_calc = calcular_estado_termodinamico(gas_fluido, T_out_C, P_out_bar, estado_in['m_dot_gas_kg_s'], y_H2O_out_vap, estado_in['y_O2'], estado_in['y_H2'])

            # 💥 CORREÇÃO: Chamando a função do novo nome 'modelar_chiller_gas'
            res_ch = modelar_chiller_gas(gas_fluido, estado_in['m_dot_mix_kg_s'] + estado_in['m_dot_H2O_liq_accomp_kg_s'], estado_in['P_bar'], estado_in['T_C'], 
                T_out_C_desejada=T_CHILLER_TARGET_LOCAL, H_in_J_kg=estado_in['H_in_mix_J_kg'], H_out_J_kg=estado_out_ch1_calc['H_in_mix_J_kg'], y_H2O_in=estado_in['y_H2O'])
            if 'erro' in res_ch: 
                print(f"!!! ERRO em {comp} ({gas_fluido})! Mensagem: {res_ch['erro']} !!!")
                continue 
            res = res_ch 
            
            m_dot_H2O_vap_out = estado_out_ch1_calc['m_dot_H2O_vap_kg_s']
            m_dot_H2O_vap_in = estado_in['m_dot_H2O_vap_kg_s']
            
            Agua_Condensada_Chiller = max(0.0, m_dot_H2O_vap_in - m_dot_H2O_vap_out)
            
            # 🌟 ATUALIZAÇÃO DO ESTOQUE (Vapor condensado volta para o pool líquido)
            m_dot_H2O_liquida_pool_out += Agua_Condensada_Chiller
            
            # 💥 CORREÇÃO (CHILLER): Água Líquida Acompanhante AUMENTA pela condensação
            m_dot_H2O_liq_accomp_out = m_dot_H2O_liq_accomp_in + Agua_Condensada_Chiller 
            
            estado_atual = estado_out_ch1_calc
            
            w_H2O_total_in = (estado_in['m_dot_H2O_vap_kg_s'] + estado_in['m_dot_H2O_liq_accomp_kg_s']) / estado_in['m_dot_mix_kg_s']
            Q_dot_H2O_Total_W = res['Q_dot_fluxo_W'] * w_H2O_total_in; Q_dot_gas_sensivel_W = res['Q_dot_fluxo_W'] * (1.0 - w_H2O_total_in)
            extra_data['Q_dot_H2_Gas'] = Q_dot_gas_sensivel_W; extra_data['Q_dot_H2O_Total'] = Q_dot_H2O_Total_W
            res['T_cold_out_C'] = T_CHILLER_TARGET_LOCAL; res['m_dot_cold_liq_kg_s'] = 0.0 
            
            res['Agua_Condensada_kg_s'] = 0.0 
            res['Gas_Dissolvido_removido_kg_s'] = 0.0; res['Agua_Pura_Removida_H2O_kg_s'] = 0.0; res['m_dot_cold_H2O_pura_kg_s'] = 0.0

        elif comp in ['KOD 1', 'KOD 2']: 
            # 💥 CORREÇÃO: Chamando a função do novo nome 'modelar_knock_out_drum'
            res_kod = modelar_knock_out_drum(gas_fluido, m_dot_gas_in_princ, estado_in['P_bar'], estado_in['T_C'], estado_in['y_H2O'], m_dot_H2O_liq_accomp_in)
            if 'erro' in res_kod: 
                print(f"!!! ERRO em {comp} ({gas_fluido})! Mensagem: {res_kod['erro']} !!!")
                continue 
            res = res_kod 
            y_H2O_out_vap = res['y_H2O_out_vap']; m_dot_gas_out_princ = res['m_dot_gas_out_kg_s']; extra_data = {'Status_KOD': res['Status_KOD']}
            
            perda_gas_dissolvido_kg_s = res.get('Gas_Dissolvido_removido_kg_s', 0.0)
            
            Agua_Removida_KOD = res.get('Agua_Condensada_removida_kg_s', 0.0); res['Agua_Condensada_kg_s'] = Agua_Removida_KOD 
            # 💥 CORREÇÃO (KOD): A água acompanhante é o que SOBRA após a remoção.
            m_dot_H2O_liq_accomp_out = res.get('Agua_Liquida_Residual_kg_s', 0.0) 
            
            # 🌟 ATUALIZAÇÃO DO ESTOQUE (Remoção de água do pool)
            m_dot_H2O_liquida_pool_out -= res.get('Agua_Pura_Removida_H2O_kg_s', 0.0)
            m_dot_H2O_liquida_pool_out = max(0.0, m_dot_H2O_liquida_pool_out)
            
            estado_atual = calcular_estado_termodinamico(gas_fluido, res['T_C'], res['P_bar'], m_dot_gas_out_princ, y_H2O_out_vap, estado_in['y_O2'], estado_in['y_H2'])
            extra_data['Q_dot_H2_Gas'] = 0.0; extra_data['Q_dot_H2O_Total'] = 0.0
            
        elif comp == 'Coalescedor 1':
            # 💥 CORREÇÃO: Chamando a função do novo nome 'modelar_coalescedor'
            res_co = modelar_coalescedor(gas_fluido, m_dot_gas_in_princ, estado_in['P_bar'], estado_in['T_C'], estado_in['y_H2O'], m_dot_H2O_liq_accomp_in)
            if 'erro' in res_co: 
                print(f"!!! ERRO em {comp} ({gas_fluido})! Mensagem: {res_co['erro']} !!!")
                continue 
            res = res_co 
            
            perda_gas_dissolvido_kg_s = res.get('Gas_Dissolvido_removido_kg_s', 0.0)
            m_dot_gas_out_princ = res['m_dot_gas_out_kg_s'] 
            
            # 🌟 ATUALIZAÇÃO DO ESTOQUE (Remoção de água do pool)
            m_dot_H2O_liquida_pool_out -= res.get('Agua_Pura_Removida_H2O_kg_s', 0.0)
            m_dot_H2O_liquida_pool_out = max(0.0, m_dot_H2O_liquida_pool_out)
            
            estado_atual = calcular_estado_termodinamico(gas_fluido, res['T_C'], res['P_bar'], m_dot_gas_out_princ, estado_in['y_H2O'], estado_in['y_O2'], estado_in['y_H2'])
            Agua_Removida_Coalescer = res.get('Agua_Removida_Coalescer_kg_s', 0.0)
            res['Agua_Condensada_kg_s'] = Agua_Removida_Coalescer
            # 💥 CORREÇÃO (COALESCEDOR): A água acompanhante é o que SOBRA após a remoção.
            m_dot_H2O_liq_accomp_out = res.get('Agua_Liquida_Residual_out_kg_s', 0.0) 
            
            res['T_cold_out_C'] = estado_in['T_C']; res['m_dot_cold_liq_kg_s'] = 0.0
            extra_data['Q_dot_H2_Gas'] = 0.0; extra_data['Q_dot_H2O_Total'] = 0.0
        
        # --- Bloco Válvula ---
        
        elif comp == 'Válvula' or comp == 'Válvula Post-Deoxo':
            fluido_coolprop = 'hydrogen' if gas_fluido == 'H2' else 'oxygen'
            P_in_Pa = estado_in['P_bar'] * 1e5
            
            # P_out forçada para o O2, ou apenas uma pequena queda para o H2 (Post-Deoxo)
            P_out_Pa = P_OUT_VALVULA_O2_BAR * 1e5 if comp == 'Válvula' else P_in_Pa - (1.0 * 1e5) 
            T_in_K = estado_in['T_C'] + 273.15
            
            # 💥 CORREÇÃO: Chamando a função do novo nome 'modelo_valvula_isoentalpica'
            res_valvula = modelo_valvula_isoentalpica(fluido_coolprop, T_in_K, P_in_Pa, P_out_Pa)
            if res_valvula is None or 'erro' in res_valvula: 
                print(f"!!! ERRO em {comp} ({gas_fluido})! Mensagem: Falha na simulação isentálpica. !!!")
                T_out_C = estado_in['T_C']; P_out_bar = estado_in['P_bar']
            else: 
                # Não é um erro que a temperatura mude, mas o modelo está em processo de estrangulamento.
                # Para evitar surpresas: 
                # T_out_C = res_valvula['SAIDA']['T_K'] - 273.15
                T_out_C = estado_in['T_C'] # MANTÉM A T, processo simplificado para estrangulamento
                P_out_bar = res_valvula['SAIDA']['P_Pa'] / 1e5
            
            y_H2O_out_sat = calcular_y_H2O_inicial(T_out_C, P_out_bar)
            y_H2O_out_vap = min(y_H2O_out_sat, estado_in['y_H2O'])
            
            res = {'T_C': T_out_C, 'P_bar': P_out_bar, 'Q_dot_fluxo_W': 0.0, 'W_dot_comp_W': 0.0, 'T_cold_out_C': T_out_C, 'm_dot_cold_liq_kg_s': 0.0, 'Gas_Dissolvido_removido_kg_s': 0.0, 'Agua_Pura_Removida_H2O_kg_s': 0.0, 'm_dot_cold_H2O_pura_kg_s': 0.0}
            estado_atual = calcular_estado_termodinamico(gas_fluido, res['T_C'], res['P_bar'], estado_in['m_dot_gas_kg_s'], y_H2O_out_vap, estado_in['y_O2'], estado_in['y_H2'])
            # 💥 CORREÇÃO (Válvula): O líquido acompanhante não muda
            m_dot_H2O_liq_accomp_out = m_dot_H2O_liq_accomp_in; extra_data = {} 
        
        # --- REGISTRO DO ESTADO NO HISTORY ---
        
        Agua_Condensada_removida_kg_s = res.get('Agua_Condensada_kg_s', 0.0)
        Agua_Pura_Removida_H2O_kg_s = res.get('Agua_Pura_Removida_H2O_kg_s', 0.0)
        Gas_Dissolvido_removido_kg_s_total = res.get('Gas_Dissolvido_removido_kg_s', 0.0)
        
        # 💥 CORREÇÃO: PSA/VSA são os únicos que removem gás principal como subproduto (purga)
        if comp not in ['PSA']: 
             Gas_Dissolvido_removido_kg_s_total = perda_gas_dissolvido_kg_s 
        
        m_dot_cold_H2O_pura_kg_s = res.get('m_dot_cold_H2O_pura_kg_s', 0.0)
        
        is_mass_separator = comp in ['KOD 1', 'KOD 2', 'Coalescedor 1', 'PSA'] or comp == 'Aquecedor Imaginário'
        
        if not is_mass_separator and comp not in ['Deoxo', 'Dry Cooler 1', 'Chiller 1']: 
            Agua_Condensada_removida_kg_s = 0.0 
            Agua_Pura_Removida_H2O_kg_s = 0.0
            Gas_Dissolvido_removido_kg_s_total = 0.0
            m_dot_H2O_liq_accomp_out = m_dot_H2O_liq_accomp_in 
            
        m_dot_H2O_liquida_pool_out = max(0.0, m_dot_H2O_liquida_pool_out)
        
        history.append({
            **estado_atual,
            'Componente': comp,
            'Q_dot_fluxo_W': res.get('Q_dot_fluxo_W', 0.0),
            'W_dot_comp_W': res.get('W_dot_comp_W', 0.0),
            'Agua_Condensada_kg_s': Agua_Condensada_removida_kg_s, 
            'm_dot_H2O_vap_kg_s': estado_atual['m_dot_H2O_vap_kg_s'],
            'W_dot_regen_W': 0.0,
            # 💥 CORREÇÃO PRINCIPAL: Usar o valor atualizado de saída (m_dot_H2O_liq_accomp_out)
            'm_dot_H2O_liq_accomp_kg_s': m_dot_H2O_liq_accomp_out, 
            'Status_KOD': extra_data.get('Status_KOD', 'N/A'),
            'T_profile_C': deoxo_T_profile_C, 
            'L_span': deoxo_L_span,           
            'X_O2': deoxo_X_O2,               
            'T_max_calc': deoxo_T_max_calc,
            'Q_dot_H2_Gas': extra_data.get('Q_dot_H2_Gas', 0.0), 
            'Q_dot_H2O_Total': extra_data.get('Q_dot_H2O_Total', 0.0),
            'T_cold_out_C': res.get('T_cold_out_C', estado_in['T_C']),
            'm_dot_cold_liq_kg_s': res.get('m_dot_cold_liq_kg_s', 0.0), 
            'Gas_Dissolvido_removido_kg_s': Gas_Dissolvido_removido_kg_s_total, 
            'Agua_Pura_Removida_H2O_kg_s': Agua_Pura_Removida_H2O_kg_s, 
            'm_dot_cold_H2O_pura_kg_s': m_dot_cold_H2O_pura_kg_s,
            'm_dot_H2O_liq_pool_kgs': m_dot_H2O_liquida_pool_out, # 🌟 ESTOQUE ATUALIZADO
            # --- Variáveis de Balanço Inicial (Mantidas como None nos componentes subsequentes) ---
            'M_DOT_VAPOR_ENTRADA_KGS_X_Y': estado_in['M_DOT_VAPOR_ENTRADA_KGS_X_Y'], 
            'M_DOT_LIQ_ACOMP_KGS_Z_W': estado_in['M_DOT_LIQ_ACOMP_KGS_Z_W'], 
            'M_DOT_LIQ_ARRAS_TOTAL_KGS': estado_in['M_DOT_LIQ_ARRAS_TOTAL_KGS'],
            'M_DOT_LIQ_MAX_DEMISTER_KGS': estado_in['M_DOT_LIQ_MAX_DEMISTER_KGS'],
        })
        
    df_history = pd.DataFrame(history)
    
    return {
        'dataframe': df_history,
        'deoxo_L_span': deoxo_L_span,
        'deoxo_T_profile_C': deoxo_T_profile_C,
        'deoxo_X_O2': deoxo_X_O2,
        'deoxo_T_max_calc': deoxo_T_max_calc,
        'deoxo_mode': deoxo_mode,
        'L_deoxo': L_deoxo,
        'dc2_mode': dc2_mode,
    }