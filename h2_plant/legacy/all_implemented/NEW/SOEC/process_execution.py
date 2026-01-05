# process_execution.py (Versão Completa e Final com Lógica de Deoxo/PSA ATIVADA)

import numpy as np
import pandas as pd
import sys
import CoolProp.CoolProp as CP
from datetime import datetime

# 💥 CORREÇÃO DA IMPORTAÇÃO: Importando TUDO de constants_and_config para garantir que M_DOT_G_H2 seja visível.
from constants_and_config import *

from aux_coolprop import (
    calcular_estado_termodinamico, calcular_y_H2O_inicial, verificar_limites_operacionais
)

# IMPORTAÇÃO DA NOVA FUNÇÃO ANALÍTICA E RESFRIADOR SIMPLES DO NOVO MÓDULO (aux_models)
from aux_models import calcular_pressao_maxima_analitica, modelar_resfriador_simples

# NOVO: Importação dos blocos de lógica de fluxo (process_flow_blocks)
from process_flow_blocks import (
    executar_dry_cooler_estagio, executar_chiller_o2_estagio, 
    executar_compressor_o2_estagio, executar_compressor_h2_estagio,
    executar_trocador_calor_agua_dreno # <--- IMPORTADO NOVO TROCADOR
)

# ----------------------------------------------------------------------
# Imports e Stubs (Bloco Central Unificado)
# ----------------------------------------------------------------------
# 🛑 CORREÇÃO DE IMPORTAÇÃO: Modelos de componente
from modelo_chiller import modelar_chiller_gas
from modelo_compressor import modelo_compressor_ideal
from modelo_kod import modelar_knock_out_drum
from modelo_coalescedor import modelar_coalescedor 
from modelo_deoxo import modelar_deoxo 
from modelo_psa import modelar_psa 

# 🛑 NOVO: Importa a nova função principal do dry cooler com um alias
from modelo_dry_cooler import simular_sistema_resfriamento as modelar_dry_cooler

from drain_mixer import extrair_dados_dreno # <--- IMPORTA FUNÇÕES DE DRENO PARA PRÉ-CÁLCULO

# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# FUNÇÃO PRINCIPAL DE SIMULAÇÃO
# --------------------------------------------------------------------------
def simular_sistema(gas_fluido: str, m_dot_g_kg_s: float, deoxo_mode: str, L_deoxo: float, dc2_mode: str, m_dot_H2O_total_fluxo_input: float, y_O2_in: float, y_H2_in: float, m_dot_liq_max_h2_ref: float = None):
    """
    Executa a simulação componente por componente para um dado fluxo de gás.
    
    m_dot_liq_max_h2_ref é ignorado, pois o Demister foi removido.
    """
    
    # --- 0. ESTADO INICIAL DO DRENO (CHUTE INICIAL PARA O TROCADOR DE CALOR) ---
    # É o estado da água fria que entra no lado frio do Trocador (Makeup + recirc.)
    estado_dreno_agregado_h2 = None
    if gas_fluido == 'H2':
         
         # CHUTE INICIAL: Vazão mínima de água não reagida a 20°C e P=1 bar (P_DRENO_OUT_BAR)
         T_DRENO_IN_C_ESTIMADA = T_CHUTE_DRENO_TROC_C # 20.0 C
         P_DRENO_IN_BAR = P_DRENO_OUT_BAR # 1.0 bar
         M_DOT_DRENO_IN_KGS = M_DOT_CHUTE_DRENO_TROC_KGS # 736.88 kg/h (novo chute)
         
         if M_DOT_DRENO_IN_KGS > 1e-6:
             T_DRENO_IN_K = T_DRENO_IN_C_ESTIMADA + 273.15
             P_DRENO_PA = P_DRENO_IN_BAR * 1e5
             H_liq_in_J_kg = CP.PropsSI('H', 'T', T_DRENO_IN_K, 'Q', 0, 'Water') 
             
             estado_dreno_agregado_h2 = {
                 'Componente': 'Dreno H2 - CHUTE INICIAL',
                 'T': T_DRENO_IN_C_ESTIMADA,
                 'P_bar': P_DRENO_IN_BAR,
                 'P_kPa': P_DRENO_IN_BAR * 100,
                 'h_kJ_kg': H_liq_in_J_kg / 1000.0,
                 'C_diss_mg_kg': 0.0,
                 'M_dot_H2O_final_kg_s': M_DOT_DRENO_IN_KGS,
                 'H_liq_out_J_kg': H_liq_in_J_kg # J/kg
             }

    
    # --- 1. CONFIGURAÇÃO BASE DO FLUXO ---
    if gas_fluido == 'H2':
        MM_GAS = MM_H2_CALC
        M_DOT_GAS = M_DOT_G_H2 # Variável agora visível
        base_list = COMPONENTS_H2 
        
    else: # O2
        MM_GAS = MM_O2_CALC
        M_DOT_GAS = M_DOT_G_O2 # Variável agora visível
        base_list = COMPONENTS_O2 
        
    
    M_H2O_TOTAL_FLUXO = m_dot_H2O_total_fluxo_input 
    
    # 1.1. Propriedades na Entrada e Saída do SOEC...
    
    T_SOEC_IN_C = T_SAT_5BAR_C
    P_SOEC_IN_BAR = P_IN_SOEC_BAR
    y_H2O_in_calc_soec = 1.0 
    m_dot_H2O_liq_accomp_in_soec = M_DOT_H2O_RECIRC_TOTAL_KGS
    
    T_START_C = T_OUT_SOEC 
    P_START_BAR = P_OUT_SOEC_BAR 
    
    M_DOT_LIQ_MAX_DEMISTER_KGS = 0.0 
         
    M_H2O_TOTAL_FLUXO_SAIDA = m_dot_H2O_total_fluxo_input
    
    M_DOT_DRENO_SOEC_KGS = 0.0 
    
    # 1. Vazão de Vapor (M_DOT_VAPOR_ENTRADA_KGS)
    M_H2O_M = CP.PropsSI('M', 'Water'); M_GAS_CP = CP.PropsSI('M', gas_fluido) # Usar CoolProp para MM localmente
    
    m_dot_mix_total_nao_consumida = M_DOT_GAS + M_H2O_TOTAL_FLUXO_SAIDA 
    
    w_H2O_total_input = M_H2O_TOTAL_FLUXO_SAIDA / m_dot_mix_total_nao_consumida
    
    y_H2O_in_calc_alvo = (w_H2O_total_input / M_H2O_M) / (w_H2O_total_input / M_H2O_M + (1-w_H2O_total_input) / M_GAS_CP)
    
    estado_soec_out_calc = calcular_estado_termodinamico(gas_fluido, T_START_C, P_START_BAR, M_DOT_GAS, y_H2O_in_calc_alvo, y_O2_in, y_H2_in)
    
    M_DOT_VAPOR_ENTRADA_KGS = estado_soec_out_calc['m_dot_H2O_vap_kg_s']
    
    m_dot_H2O_liq_in_arraste = M_H2O_TOTAL_FLUXO_SAIDA - M_DOT_VAPOR_ENTRADA_KGS
    m_dot_H2O_liq_in_arraste = max(0.0, m_dot_H2O_liq_in_arraste)
    
    y_H2O_in_calc_saida = estado_soec_out_calc['y_H2O'] 

    # --- FIM DO CÁLCULO DE BALANÇO DE ÁGUA NA SAÍDA ---
    
    M_DOT_GAS_SOEC_IN = 0.0
    
    # 2. ESTADO INICIAL (SOEC ENTRADA)
    estado_soec_in = calcular_estado_termodinamico(gas_fluido, T_SOEC_IN_C, P_SOEC_IN_BAR, M_DOT_GAS_SOEC_IN, y_H2O_in_calc_soec, y_O2_in, y_H2_in)
    
    # 3. ESTADO SOEC SAÍDA (ENTRADA PURIFICAÇÃO) - USA O ESTADO CALCULADO
    estado_soec_out = estado_soec_out_calc
    
    m_dot_H2O_liquida_pool_kgs = M_H2O_TOTAL_FLUXO_SAIDA 
    m_dot_H2O_liquida_pool_out = m_dot_H2O_liquida_pool_kgs 

    # 4. A lista de componentes a simular
    component_list_filtered = [comp for comp in base_list if comp not in ['SOEC (Entrada)', 'SOEC (Saída)']]
    
    # ADIÇÃO DO 'SOEC (Entrada)' e 'SOEC (Saída)' no histórico
    history = [
        # --- PONTO 1: SOEC (ENTRADA) ---
        {
            **estado_soec_in, 'Componente': 'SOEC (Entrada)', 'Q_dot_fluxo_W': 0.0, 'W_dot_comp_W': 0.0, 'Agua_Condensada_kg_s': 0.0,
            'm_dot_H2O_vap_kg_s': estado_soec_in['m_dot_H2O_vap_kg_s'], 'W_dot_regen_W': 0.0,
            'm_dot_H2O_liq_accomp_kg_s': m_dot_H2O_liq_accomp_in_soec, 
            'Status_KOD': 'N/A', 'T_profile_C': None, 'L_span': None, 'X_O2': 0.0, 'T_max_calc': None,
            'Q_dot_H2_Gas': 0.0, 'Q_dot_H2O_Total': 0.0, 'T_cold_out_C': T_SOEC_IN_C, 'm_dot_cold_liq_kg_s': 0.0, 
            'Gas_Dissolvido_removido_kg_s': 0.0, 'Agua_Pura_Removida_H2O_kg_s': M_DOT_H2O_CONSUMIDA_KGS, 
            'm_dot_cold_H2O_pura_kg_s': 0.0, 'm_dot_H2O_liq_pool_kgs': M_DOT_H2O_RECIRC_TOTAL_KGS, 
            'M_DOT_VAPOR_ENTRADA_KGS_X_Y': M_DOT_VAPOR_ENTRADA_KGS, 'M_DOT_LIQ_ACOMP_KGS_Z_W': m_dot_H2O_liq_in_arraste, 
            'M_DOT_LIQ_ARRAS_TOTAL_KGS': M_H2O_TOTAL_FLUXO_SAIDA, 'M_DOT_LIQ_MAX_DEMISTER_KGS': M_DOT_LIQ_MAX_DEMISTER_KGS,
            'multistage_history': None
        },
        # --- PONTO 2: SOEC (SAÍDA) / ENTRADA DA PURIFICAÇÃO ---
        {
            **estado_soec_out, 'Componente': 'SOEC (Saída)', 'Q_dot_fluxo_W': 0.0, 'W_dot_comp_W': 0.0, 'Agua_Condensada_kg_s': 0.0,
            'm_dot_H2O_vap_kg_s': estado_soec_out['m_dot_H2O_vap_kg_s'], 'W_dot_regen_W': 0.0,
            'm_dot_H2O_liq_accomp_kg_s': m_dot_H2O_liq_in_arraste, 
            'Status_KOD': 'N/A', 'T_profile_C': None, 'L_span': None, 'X_O2': 0.0, 'T_max_calc': None,
            'Q_dot_H2_Gas': 0.0, 'Q_dot_H2O_Total': 0.0, 'T_cold_out_C': T_START_C, 'm_dot_cold_liq_kg_s': 0.0, 
            'Gas_Dissolvido_removido_kg_s': 0.0, 'Agua_Pura_Removida_H2O_kg_s': M_DOT_DRENO_SOEC_KGS, 
            'm_dot_cold_H2O_pura_kg_s': 0.0, 'm_dot_H2O_liq_pool_kgs': m_dot_H2O_liquida_pool_out, 
            'M_DOT_VAPOR_ENTRADA_KGS_X_Y': M_DOT_VAPOR_ENTRADA_KGS, 'M_DOT_LIQ_ACOMP_KGS_Z_W': m_dot_H2O_liq_in_arraste, 
            'M_DOT_LIQ_ARRAS_TOTAL_KGS': M_H2O_TOTAL_FLUXO_SAIDA, 'M_DOT_LIQ_MAX_DEMISTER_KGS': M_DOT_LIQ_MAX_DEMISTER_KGS,
            'multistage_history': None
        }
    ]
    
    # Variáveis de retorno do Deoxo (agora inicializadas no escopo da função principal)
    deoxo_L_span, deoxo_T_profile_C, deoxo_X_O2, deoxo_T_max_calc = None, None, None, None
    multistage_history = None 
    estado_atual = history[-1].copy() 

    # --------------------------------------------------------------------------
    # 5. LOOP DE SIMULAÇÃO (Começa após SOEC (Saída))
    # --------------------------------------------------------------------------
    
    for comp in component_list_filtered:
            
        estado_in = history[-1].copy()
        m_dot_H2O_liq_accomp_in = estado_in.get('m_dot_H2O_liq_accomp_kg_s', 0.0) 
        
        m_dot_H2O_liquida_pool_in = estado_in.get('m_dot_H2O_liq_pool_kgs', 0.0)
        m_dot_H2O_liq_accomp_out = m_dot_H2O_liq_accomp_in 
        m_dot_gas_in_princ = estado_in['m_dot_gas_kg_s'] 
        
        # Inicializa res 
        res = {'T_C': estado_in['T_C'], 'P_bar': estado_in['P_bar'], 'Q_dot_fluxo_W': 0.0, 'W_dot_comp_W': 0.0,
               'Agua_Condensada_kg_s': 0.0, 'T_cold_out_C': estado_in['T_C'], 'm_dot_cold_liq_kg_s': 0.0,
               'Gas_Dissolvido_removido_kg_s': 0.0, 'Agua_Pura_Removida_H2O_kg_s': 0.0, 'm_dot_cold_H2O_pura_kg_s': 0.0}
        extra_data = {} 
        
        perda_gas_dissolvido_kg_s = 0.0
        m_dot_H2O_liquida_pool_out = m_dot_H2O_liquida_pool_in 
        Agua_Condensada_removida_kg_s = 0.0
        Agua_Pura_Removida_H2O_kg_s = 0.0
        
        Agua_Removida_Componente_total = 0.0
        
        # --- EXECUÇÃO MODULAR (DRY COOLER, CHILLER, COMPRESSOR O2) ---
        is_o2_dry_cooler_estagio = comp in ['Dry Cooler O2 (Estágio 1)', 'Dry Cooler O2 (Estágio 2)', 'Dry Cooler O2 (Estágio 3)', 'Dry Cooler O2 (Estágio 4)']
        is_o2_chiller_estagio = comp in ['Chiller O2', 'Chiller O2 (Estágio 2)', 'Chiller O2 (Estágio 3)']
        is_o2_compressor_estagio = comp in ['Compressor O2 (Estágio 1)', 'Compressor O2 (Estágio 2)', 'Compressor O2 (Estágio 3)', 'Compressor O2 (Estágio 4)']
        
        is_h2_dry_cooler_estagio = comp in ['Dry Cooler (Estágio 1)', 'Dry Cooler (Estágio 2)', 'Dry Cooler (Estágio 3)', 'Dry Cooler (Estágio 4)', 'Dry Cooler (Estágio 5)']
        is_h2_chiller_estagio = comp in ['Chiller (Estágio 1)', 'Chiller (Estágio 2)', 'Chiller (Estágio 3)', 'Chiller (Estágio 4)', 'Chiller (Estágio 5)', 'Chiller (Estágio (Estágio 5)', 'Chiller 1']
        is_h2_compressor_estagio = comp in ['Compressor H2 (Estágio 1)', 'Compressor H2 (Estágio 2)', 'Compressor H2 (Estágio 3)', 'Compressor H2 (Estágio 4)', 'Compressor H2 (Estágio 5)']

        
        if (gas_fluido == 'O2' and is_o2_dry_cooler_estagio) or (gas_fluido == 'H2' and is_h2_dry_cooler_estagio):
            estado_atual, res, m_dot_H2O_liq_accomp_out, m_dot_H2O_liquida_pool_out, extra_data = executar_dry_cooler_estagio(
                comp, estado_in, gas_fluido, m_dot_H2O_liq_accomp_in, m_dot_H2O_liquida_pool_out
            )
            Agua_Condensada_removida_kg_s = res.get('Agua_Condensada_kg_s', 0.0)
            Agua_Pura_Removida_H2O_kg_s = res.get('Agua_Pura_Removida_H2O_kg_s', 0.0) # <--- Contém a água drenada
            Agua_Removida_Componente_total = Agua_Pura_Removida_H2O_kg_s # Rastrear para o balanço

        elif (gas_fluido == 'O2' and is_o2_chiller_estagio) or (gas_fluido == 'H2' and is_h2_chiller_estagio):
            estado_atual, res, m_dot_H2O_liq_accomp_out, m_dot_H2O_liquida_pool_out, extra_data = executar_chiller_o2_estagio(
                comp, estado_in, gas_fluido, m_dot_H2O_liq_accomp_in, m_dot_H2O_liquida_pool_out
            )
            Agua_Condensada_removida_kg_s = res.get('Agua_Condensada_kg_s', 0.0)
            Agua_Pura_Removida_H2O_kg_s = res.get('Agua_Pura_Removida_H2O_kg_s', 0.0)
            Agua_Removida_Componente_total = Agua_Pura_Removida_H2O_kg_s

        elif is_o2_compressor_estagio:
             estado_atual, res, m_dot_H2O_liq_accomp_out, m_dot_H2O_liquida_pool_out, extra_data = executar_compressor_o2_estagio(
                 comp, estado_in, m_dot_H2O_liq_accomp_in
             )
             Agua_Condensada_removida_kg_s = res.get('Agua_Condensada_kg_s', 0.0)
             Agua_Pura_Removida_H2O_kg_s = res.get('Agua_Pura_Removida_H2O_kg_s', 0.0)
             Agua_Removida_Componente_total = Agua_Pura_Removida_H2O_kg_s
             
        elif is_h2_compressor_estagio:
             estado_atual, res, m_dot_H2O_liq_accomp_out, m_dot_H2O_liquida_pool_out, extra_data = executar_compressor_h2_estagio(
                 comp, estado_in, m_dot_H2O_liq_accomp_in
             )
             Agua_Condensada_removida_kg_s = res.get('Agua_Condensada_kg_s', 0.0)
             Agua_Pura_Removida_H2O_kg_s = res.get('Agua_Pura_Removida_H2O_kg_s', 0.0)
             Agua_Removida_Componente_total = Agua_Pura_Removida_H2O_kg_s

        # --- Trocador de Calor (Água Dreno) ---
        elif comp == 'Trocador de Calor (Água Dreno)' and gas_fluido == 'H2':
            estado_atual, res, m_dot_H2O_liq_accomp_out, m_dot_H2O_liquida_pool_out, extra_data = executar_trocador_calor_agua_dreno(
                comp, estado_in, gas_fluido, m_dot_H2O_liq_accomp_in, estado_dreno_agregado_h2, calcular_estado_termodinamico
            )
            Agua_Condensada_removida_kg_s = 0.0
            Agua_Pura_Removida_H2O_kg_s = 0.0
            Agua_Removida_Componente_total = 0.0
            
            estado_dreno_agregado_h2 = {
                 'Componente': 'Dreno H2 - PÓS TROCADOR',
                 'T': extra_data['Dreno_T_out_C'],
                 'P_bar': P_DRENO_OUT_BAR, 
                 'P_kPa': P_DRENO_OUT_BAR * 100,
                 'h_kJ_kg': extra_data['Dreno_H_out_J_kg'] / 1000.0,
                 'C_diss_mg_kg': 0.0, 
                 'M_dot_H2O_final_kg_s': extra_data['Dreno_m_dot_kgs'],
                 'H_liq_out_J_kg': extra_data['Dreno_H_out_J_kg']
             }

        
        elif comp == 'Dry Cooler 1':
            # --- Dry Cooler 1 (Inicial) ---
            T_target = T_DRY_COOLER_OUT_H2_C if gas_fluido == 'H2' else T_DRY_COOLER_OUT_O2_C
            
            # 🛑 CHAMADA AO NOVO MODELO (simular_sistema_resfriamento via alias) 🛑
            res_dc = modelar_dry_cooler(
                gas_fluido, 
                estado_in['m_dot_mix_kg_s'], 
                m_dot_H2O_liq_accomp_in, 
                estado_in['P_bar'], 
                estado_in['T_C']
            )
            
            if "erro" in res_dc:
                 raise ValueError(f"Erro na simulação do Dry Cooler 1: {res_dc['erro']}")
                 
            res = res_dc
            T_out_C = res['T_C']; P_out_bar = res['P_bar']
            
            # Lógica de condensação (basicamente a mesma)
            y_H2O_out_sat = calcular_y_H2O_inicial(T_out_C, P_out_bar) 
            y_H2O_out_vap = min(y_H2O_out_sat, estado_in['y_H2O'])
            
            estado_atual_calc = calcular_estado_termodinamico(gas_fluido, T_out_C, P_out_bar, estado_in['m_dot_gas_kg_s'], y_H2O_out_vap, estado_in['y_O2'], estado_in['y_H2'])
            
            m_dot_H2O_vap_out = estado_atual_calc['m_dot_H2O_vap_kg_s']
            m_dot_H2O_vap_in = estado_in['m_dot_H2O_vap_kg_s']
            
            Agua_Condensada_removida_kg_s = max(0.0, m_dot_H2O_vap_in - m_dot_H2O_vap_out)
            
            # O condensado acumula no líquido acompanhante, não é drenado aqui
            m_dot_H2O_liquida_pool_out = m_dot_H2O_liquida_pool_in
            m_dot_H2O_liq_accomp_out = m_dot_H2O_liq_accomp_in + Agua_Condensada_removida_kg_s 
            estado_atual = estado_atual_calc 
            
            # Dados extras para rastreamento de energia
            extra_data['Q_dot_H2_Gas'] = res_dc.get('Q_dot_fluxo_W', 0.0) 
            extra_data['Q_dot_H2O_Total'] = 0.0 # O modelo Dry Cooler não separa o calor
            
            # Atualiza os resultados de res
            res['Agua_Condensada_kg_s'] = Agua_Condensada_removida_kg_s # Condensada
            res['Agua_Pura_Removida_H2O_kg_s'] = 0.0 # Não drena líquido acompanhante aqui
            res['Gas_Dissolvido_removido_kg_s'] = 0.0
            res['m_dot_cold_H2O_pura_kg_s'] = 0.0
            Agua_Removida_Componente_total = 0.0 # Nenhuma remoção de líquido aqui


        
        elif comp in ['KOD 1', 'KOD 2', 'KOD 3', 'KOD 4', 'KOD 5', 'KOD 1 O2']: 
            # KOD drena TODO o líquido acompanhante que entrou (m_dot_H2O_liq_accomp_in)
            Agua_Pura_Removida_H2O_kg_s = m_dot_H2O_liq_accomp_in
            
            if comp == 'KOD 2':
                m_dot_H2O_liquida_pool_out -= Agua_Pura_Removida_H2O_kg_s 
            
            if Agua_Pura_Removida_H2O_kg_s > 0:
                m_dot_H2O_liq_accomp_out = 0.0 
            else: 
                m_dot_H2O_liq_accomp_out = m_dot_H2O_liq_accomp_in
                
            res_kod = modelar_knock_out_drum(gas_fluido, m_dot_gas_in_princ, estado_in['P_bar'], estado_in['T_C'], estado_in['y_H2O'], m_dot_H2O_liq_accomp_in)
            res = res_kod 
            T_out_C_KOD = estado_in['T_C'] 
            
            y_H2O_out_vap = res['y_H2O_out_vap']; m_dot_gas_out_princ = res['m_dot_gas_out_kg_s']; extra_data = {'Status_KOD': res['Status_KOD']}
            perda_gas_dissolvido_kg_s = res.get('Gas_Dissolvido_removido_kg_s', 0.0)
            res['Agua_Condensada_kg_s'] = 0.0
            res['Agua_Pura_Removida_H2O_kg_s'] = Agua_Pura_Removida_H2O_kg_s # O que foi drenado
            m_dot_H2O_liquida_pool_out = max(0.0, m_dot_H2O_liquida_pool_out)
            res['T_cold_out_C'] = T_out_C_KOD; res['m_dot_cold_liq_kg_s'] = 0.0
            estado_atual = calcular_estado_termodinamico(gas_fluido, T_out_C_KOD, res['P_bar'], m_dot_gas_out_princ, y_H2O_out_vap, estado_in['y_O2'], estado_in['y_H2'])
            extra_data['Q_dot_H2_Gas'] = 0.0; extra_data['Q_dot_H2O_Total'] = 0.0
            Agua_Removida_Componente_total = Agua_Pura_Removida_H2O_kg_s # Rastrear para o balanço

        elif comp == 'Coalescedor 1': 
             # Coalescedor drena TODO o líquido acompanhante que entrou (m_dot_H2O_liq_accomp_in)
             Agua_Pura_Removida_H2O_kg_s = m_dot_H2O_liq_accomp_in
             
             if Agua_Pura_Removida_H2O_kg_s > 0:
                 m_dot_H2O_liq_accomp_out = 0.0 
             else: 
                 m_dot_H2O_liq_accomp_out = m_dot_H2O_liq_accomp_in
                 
             res_coalescer = modelar_coalescedor(gas_fluido, m_dot_gas_in_princ, estado_in['P_bar'], estado_in['T_C'], estado_in['y_H2O'], m_dot_H2O_liq_accomp_in)
             res = res_coalescer
             T_out_C_COAL = estado_in['T_C'] 
             
             y_H2O_out_vap = res['y_H2O_out_vap']; m_dot_gas_out_princ = res['m_dot_gas_out_kg_s']; 
             perda_gas_dissolvido_kg_s = res.get('Gas_Dissolvido_removido_kg_s', 0.0)
             res['Agua_Condensada_kg_s'] = 0.0
             res['Agua_Pura_Removida_H2O_kg_s'] = Agua_Pura_Removida_H2O_kg_s # O que foi drenado
             m_dot_H2O_liquida_pool_out = max(0.0, m_dot_H2O_liquida_pool_out)
             estado_atual = calcular_estado_termodinamico(gas_fluido, T_out_C_COAL, res['P_bar'], m_dot_gas_out_princ, y_H2O_out_vap, estado_in['y_O2'], estado_in['y_H2'])
             Agua_Removida_Componente_total = Agua_Pura_Removida_H2O_kg_s # Rastrear para o balanço

        # 🛑 LÓGICA DE EXECUÇÃO DO DEOXO (ATIVADA)
        elif comp == 'Deoxo' and gas_fluido == 'H2':
             print(f"Executando {comp} (H2). T_in={estado_in['T_C']:.2f} °C, y_O2={estado_in['y_O2']*1e6:.2f} ppm.")
             
             # 1. Simular Deoxo
             res_deoxo = modelar_deoxo(
                 estado_in['m_dot_gas_kg_s'], estado_in['P_bar'], estado_in['T_C'], 
                 estado_in['y_H2O'], estado_in['y_O2'], L_deoxo
             )
             
             # 2. Atualizar estados pós-reação
             res = res_deoxo
             T_out_C = res['T_C']; P_out_bar = res['P_bar']
             m_dot_gas_out_princ = res['m_dot_gas_out_kg_s']
             y_H2O_out = res['y_H2O_out'] # Fração molar de vapor de H2O (nova + antiga)
             y_O2_out = res['y_O2_out'] # Nova fração molar de O2 (muito baixa)
             
             # 3. Recalcular estado termodinâmico
             estado_atual = calcular_estado_termodinamico(
                 gas_fluido, T_out_C, P_out_bar, m_dot_gas_out_princ, 
                 y_H2O_out, y_O2_out, estado_in['y_H2']
             )
             
             # 4. Rastrear dados para plotagem
             deoxo_L_span = res.get('L_span')
             deoxo_T_profile_C = res.get('T_profile_C')
             deoxo_X_O2 = res.get('X_O2')
             deoxo_T_max_calc = res.get('T_max_calc')
             
             # 5. Balanço de Massa
             Agua_Pura_Removida_H2O_kg_s = 0.0 # Deoxo gera vapor
             # Adiciona a água gerada ao líquido acompanhante (embora seja vapor, é rastreado aqui)
             m_dot_H2O_liq_accomp_out = m_dot_H2O_liq_accomp_in + res['Agua_Gerada_kg_s']
             m_dot_H2O_liquida_pool_out = m_dot_H2O_liquida_pool_in
             Agua_Removida_Componente_total = 0.0 # Não remove líquido do fluxo principal

        # 🛑 LÓGICA DE EXECUÇÃO DO PSA (ATIVADA)
        elif comp == 'PSA' and gas_fluido == 'H2':
             print(f"Executando {comp} (H2). T_in={estado_in['T_C']:.2f} °C, P_in={estado_in['P_bar']:.2f} bar.")
             
             res_psa = modelar_psa(estado_in['m_dot_mix_kg_s'], estado_in['P_bar'], estado_in['T_C'], estado_in['y_H2O'], estado_in['y_O2'])
             
             res = res_psa
             m_dot_gas_out_princ = res['m_dot_gas_out_kg_s']
             P_out_bar_PSA = res['P_out_bar']
             m_dot_H2O_liq_accomp_out = res.get('m_dot_H2O_liq_accomp_out', m_dot_H2O_liq_accomp_in)
             perda_gas_dissolvido_kg_s = res.get('Gas_Dissolvido_removido_kg_s', 0.0)
             Agua_Pura_Removida_H2O_kg_s = res.get('Agua_Pura_Removida_H2O_kg_s', 0.0)
             m_dot_H2O_liquida_pool_out -= Agua_Pura_Removida_H2O_kg_s
             
             # O PSA muda as frações molares de todas as impurezas (H2O e O2)
             estado_atual = calcular_estado_termodinamico(
                 gas_fluido, res['T_C'], P_out_bar_PSA, m_dot_gas_out_princ, 
                 res['y_H2O_out'], res['y_O2_out'], estado_in['y_H2']
             )
             res['W_dot_regen_W'] = res_psa.get('W_dot_regen_W', 0.0)
             Agua_Removida_Componente_total = Agua_Pura_Removida_H2O_kg_s # Rastrear para o balanço

        else:
             # Componente não modelado ou fluxo O2 na purificação final (se houver)
             print(f"[AVISO - {comp}] Componente ignorado ou não possui lógica de execução definida.")
             Agua_Condensada_removida_kg_s = 0.0
             Agua_Pura_Removida_H2O_kg_s = 0.0
             Agua_Removida_Componente_total = 0.0
             # Nenhuma mudança no estado, apenas repassa
             estado_atual = estado_in.copy()
             m_dot_H2O_liq_accomp_out = m_dot_H2O_liq_accomp_in
        
        
        # ----------------------------------------------------------------------------------------
        # --- REGISTRO DO ESTADO NO HISTORY (Lógica unificada de balanço) ---
        
        Agua_Condensada_removida_kg_s = res.get('Agua_Condensada_kg_s', 0.0)
        Agua_Pura_Removida_H2O_kg_s = res.get('Agua_Pura_Removida_H2O_kg_s', 0.0)
        Gas_Dissolvido_removido_kg_s_total = res.get('Gas_Dissolvido_removido_kg_s', 0.0)
        
        is_kod_coalescer = comp in ['KOD 1', 'KOD 2', 'KOD 1 O2', 'KOD 3', 'KOD 4', 'KOD 5', 'Coalescedor 1']

        # Garante que o pool não é negativo
        m_dot_H2O_liquida_pool_out = max(0.0, m_dot_H2O_liquida_pool_out)
        
        # Atualiza a água removida no res com o valor final rastreado (Agua_Pura_Removida_H2O_kg_s)
        res['Agua_Pura_Removida_H2O_kg_s'] = Agua_Pura_Removida_H2O_kg_s
        
        history.append({
            **estado_atual,
            'Componente': comp,
            'Q_dot_fluxo_W': res.get('Q_dot_fluxo_W', 0.0),
            'W_dot_comp_W': res.get('W_dot_comp_W', 0.0),
            'Agua_Condensada_kg_s': Agua_Condensada_removida_kg_s, 
            'm_dot_H2O_vap_kg_s': estado_atual['m_dot_H2O_vap_kg_s'],
            'W_dot_regen_W': res.get('W_dot_regen_W', 0.0),
            'm_dot_H2O_liq_accomp_kg_s': m_dot_H2O_liq_accomp_out, 
            'Status_KOD': extra_data.get('Status_KOD', 'N/A'),
            'T_profile_C': deoxo_T_profile_C, 
            'L_span': deoxo_L_span,           
            'X_O2': deoxo_X_O2,               
            'T_max_calc': extra_data.get('T_peak_C', deoxo_T_max_calc),
            'Q_dot_H2_Gas': extra_data.get('Q_dot_H2_Gas', 0.0), 
            'Q_dot_H2O_Total': extra_data.get('Q_dot_H2O_Total', 0.0),
            'T_cold_out_C': res.get('T_cold_out_C', estado_in['T_C']),
            'm_dot_cold_liq_kg_s': res.get('m_dot_cold_liq_kg_s', 0.0), 
            'Gas_Dissolvido_removido_kg_s': Gas_Dissolvido_removido_kg_s_total, 
            'Agua_Pura_Removida_H2O_kg_s': Agua_Pura_Removida_H2O_kg_s, 
            'm_dot_cold_H2O_pura_kg_s': res.get('m_dot_cold_H2O_pura_kg_s', 0.0),
            'm_dot_H2O_liq_pool_kgs': m_dot_H2O_liquida_pool_out, 
            'M_DOT_VAPOR_ENTRADA_KGS_X_Y': estado_in['M_DOT_VAPOR_ENTRADA_KGS_X_Y'], 
            'M_DOT_LIQ_ACOMP_KGS_Z_W': estado_in['M_DOT_LIQ_ACOMP_KGS_Z_W'], 
            'M_DOT_LIQ_ARRAS_TOTAL_KGS': estado_in['M_DOT_LIQ_ARRAS_TOTAL_KGS'], 
            'M_DOT_LIQ_MAX_DEMISTER_KGS': estado_in['M_DOT_LIQ_MAX_DEMISTER_KGS'],
            'multistage_history': None,
            # Dreno H/T Pós-Trocador (mantido para a lógica de iteração ou visualização)
            'Dreno_H_out_J_kg': extra_data.get('Dreno_H_out_J_kg', 0.0),
            'Dreno_T_out_C': extra_data.get('Dreno_T_out_C', 0.0)
        })
        
    df_history = pd.DataFrame(history)
    
    # --------------------------------------------------------------------------
    # 7. CONSOLIDAÇÃO FINAL
    # --------------------------------------------------------------------------
    
    # Garante que a coluna 'Agua_Pura_Removida_H2O_kg_s' seja float
    if 'Agua_Pura_Removida_H2O_kg_s' not in df_history.columns:
         df_history['Agua_Pura_Removida_H2O_kg_s'] = 0.0
         
    # Garante que a coluna 'Gas_Dissolvido_removido_kg_s' seja float
    if 'Gas_Dissolvido_removido_kg_s' not in df_history.columns:
         df_history['Gas_Dissolvido_removido_kg_s'] = 0.0
    
    # Preenche NaN de campos específicos de Dry Cooler/Deoxo para facilitar a plotagem/exibição
    # 🛑 CORREÇÃO DO FUTURE WARNING E TIPO DE DADO
    df_history['L_span'] = (
        df_history['L_span']
        .infer_objects(copy=False)
        .fillna(0.0)
    )
    df_history['X_O2'] = df_history['X_O2'].fillna(0.0)
    df_history['T_max_calc'] = df_history['T_max_calc'].fillna(df_history['T_C'])
    df_history['Q_dot_H2_Gas'] = df_history['Q_dot_H2_Gas'].fillna(0.0)
    df_history['Q_dot_H2O_Total'] = df_history['Q_dot_H2O_Total'].fillna(0.0)
    df_history['T_cold_out_C'] = df_history['T_cold_out_C'].fillna(df_history['T_C'])
    df_history['m_dot_cold_liq_kg_s'] = df_history['m_dot_cold_liq_kg_s'].fillna(0.0)
    df_history['m_dot_cold_H2O_pura_kg_s'] = df_history['m_dot_cold_H2O_pura_kg_s'].fillna(0.0)
    
    # Ajuste de Vazão (para PSA e Valvulas)
    # Vazão mássica principal (gás)
    df_history['Vazao_Massica_Principal (kg/s)'] = df_history['m_dot_gas_kg_s']
    # Vazão mássica total (gás + vapor)
    df_history['Vazao_Massica_Total (kg/s)'] = df_history['m_dot_mix_kg_s'] 
    
    # Calcula Vazão Volumétrica (STP) - Nm³/h
    if gas_fluido == 'H2':
        MM_GAS_PRINCIPAL = MM_H2_CALC
    else:
        MM_GAS_PRINCIPAL = MM_O2_CALC
        
    df_history['Vazao_Volumetrica_Nm3_h'] = (df_history['m_dot_gas_kg_s'] / MM_GAS_PRINCIPAL) * V_MOLAR_PADRAO_NM3_KMOL * 3600

    return {
        'dataframe': df_history,
        'deoxo_L_span': deoxo_L_span,
        'deoxo_T_profile_C': deoxo_T_profile_C,
        'deoxo_X_O2': deoxo_X_O2,
        'deoxo_T_max_calc': deoxo_T_max_calc,
        'deoxo_mode': deoxo_mode,
        'L_deoxo': L_deoxo,
        'dc2_mode': dc2_mode,
        # 🛑 NOVO: Retorna o estado do Dreno Pós-Trocador para uma possível próxima iteração
        'dreno_agregado_h2': estado_dreno_agregado_h2 
    }