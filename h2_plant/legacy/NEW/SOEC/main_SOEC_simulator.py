# main_SOEC_simulator.py
# Script principal que orquestra a simulação completa.

import numpy as np
import pandas as pd
import sys
from datetime import datetime
import os 

# 🛑 CORREÇÃO CRÍTICA: REMOÇÃO DO BLOCO DE INJEÇÃO DO SYS.PATH
# Os módulos agora devem ser importáveis diretamente no diretório raiz.

# ----------------------------------------------------------------------
# AS IMPORTAÇÕES ABAIXO DEVEM AGORA ENCONTRAR OS MÓDULOS DE COMPONENTE 
# E AUXILIARES NO DIRETÓRIO RAIZ.
# ----------------------------------------------------------------------


# Importa a lógica de execução do processo
from process_execution import simular_sistema

# 1. Importações de Módulos Auxiliares (do diretório)
from constants_and_config import (
    LIMITES, T_IN_C, P_IN_BAR, T_DRY_COOLER_OUT_H2_C_DC2,
    M_DOT_G_H2, M_DOT_G_O2, M_H2O_TOTAL_H2_KGS, M_H2O_TOTAL_O2_KGS,
    MODE_DEOXO_FINAL, L_DEOXO_OTIMIZADO_M, DC2_MODE_FINAL, Y_O2_IN_H2, Y_H2_IN_O2,
    T_OUT_SOEC # 🛑 NOVO: Importa T_OUT_SOEC (152.0 °C)
)

# 📌 CORREÇÃO DE ERRO: Definir o stub_plot localmente antes de tentar importar (ou removê-lo da importação)
def stub_plot(*args, **kwargs):
    print("AVISO: Plotagem não executada (Módulo ausente/Erro).")


# Importações de funções de Reporting (que realmente existem em plot_reporter_base)
from plot_reporter_base import (
    salvar_e_exibir_plot, 
    exibir_estado_final,
    exibir_tabela_detalhada, 
    log_runtime, 
    exibir_validacao_balanco_global,
    exibir_balanco_agua_inicial,
)

# 3. Importação de Plotagens de Perfis (Com Fallback)
try:
    # 🛑 CORREÇÃO: Removido o prefixo 'plots_modulos.'
    from plot_propriedades_empilhadas import plot_propriedades_empilhadas
    from plot_vazao_massica_total_e_removida import plot_vazao_massica_total_e_removida
    from plot_vazao_liquida_acompanhante import plot_vazao_liquida_acompanhante
    from plot_q_breakdown import plot_q_breakdown
    from plot_vazao_agua_separada import plot_vazao_agua_separada
    from plot_fluxos_energia import plot_fluxos_energia
    from plot_agua_removida_total import plot_agua_removida_total
    from plot_impurezas_crossover import plot_impurezas_crossover
    from plot_deoxo_perfil import plot_deoxo_perfil
    from plot_esquema_processo import plot_esquema_processo
    from plot_drenos_descartados import plot_drenos_descartados
    # Importação da função de plotagem de drenos (também deve estar no diretório raiz)
    from plot_esquema_drenos import plot_esquema_drenos
    
    PLOT_MODULES_LOADED = True
except ImportError as e:
    # Este bloco é para captura de erros de módulo/plotagem e usa os stubs
    print(f"AVISO: Módulo de plotagem/drenagem ausente. Alguns plots podem ser stubs: {e}")
    PLOT_MODULES_LOADED = False

# 🛑 IMPORTAÇÃO CRÍTICA FORA DO TRY/EXCEPT DE PLOTAGEM
try:
    import drain_mixer as mixer_drenos
except ImportError as e:
    print(f"!!! ERRO CRÍTICO: Falha ao importar módulo drain_mixer: {e} !!!")
    mixer_drenos = None

# ----------------------------------------------------------------------
# 4. FUNÇÃO DE EXECUÇÃO ÚNICA (CORRIGIDA)
# ----------------------------------------------------------------------

def executar_simulacao_unica(gas_fluido: str, m_dot_g_kg_s: float, deoxo_mode: str, L_deoxo: float, dc2_mode: str, m_dot_H2O_total_fluxo_input: float, y_O2_in: float, y_H2_in: float, m_dot_liq_max_h2_ref: float = None):
    """
    Executa a simulação completa.
    
    💥 CORREÇÃO: Usando a variável de argumento m_dot_H2O_total_fluxo_input
    """
    
    # --- FASE ÚNICA: Simulação Direta ---
    print(f"\n--- [FASE ÚNICA: FLUXO {gas_fluido} - SIMULAÇÃO DIRETA (KOD + Dry Cooler Inicial)] ---")
    
    result_real = simular_sistema(gas_fluido, m_dot_g_kg_s, deoxo_mode, L_deoxo, dc2_mode, 
                                  # 📌 Variável CORRIGIDA AQUI:
                                  m_dot_H2O_total_fluxo_input=m_dot_H2O_total_fluxo_input, y_O2_in=y_O2_in, y_H2_in=y_H2_in, 
                                  m_dot_liq_max_h2_ref=m_dot_liq_max_h2_ref) 
    
    df_final = result_real['dataframe']
    
    # Captura o M_DOT_LIQ_MAX_DEMISTER_KGS (que pode ter sido forçado para O2)
    # A referência é no 'SOEC (Saída)'
    m_dot_liq_max_kgs = df_final[df_final['Componente'] == 'SOEC (Saída)']['M_DOT_LIQ_MAX_DEMISTER_KGS'].iloc[0]
    
    return df_final, result_real, m_dot_liq_max_kgs


if __name__ == '__main__':
    
    start_time = datetime.now()
    
    MODE = MODE_DEOXO_FINAL 
    L_DEOXO_M = L_DEOXO_OTIMIZADO_M
    DC2_MODE = DC2_MODE_FINAL 
    
    # Variáveis de entrada para a função (obtidas das constantes, agora visíveis)
    m_dot_H2O_total_a_processar_H2 = M_H2O_TOTAL_H2_KGS
    m_dot_H2O_total_a_processar_O2 = M_H2O_TOTAL_O2_KGS
    
    
    print("\n--- INICIANDO SIMULAÇÃO OTIMIZADA (SOEC Entrada -> SOEC Saída -> Purificação) ---")
    print(f"Modo Deoxo: {MODE}, Comprimento (L): {L_DEOXO_M:.3f} m, Pós-Deoxo: {DC2_MODE}")
    
    # --- 1. Executa a Simulação Única para H2 ---
    start_time_h2 = datetime.now()
    # Usando o nome da variável de escopo local (m_dot_H2O_total_a_processar_H2)
    df_h2_plot, result_h2, m_dot_liq_max_h2_ref = executar_simulacao_unica('H2', M_DOT_G_H2, MODE, L_DEOXO_M, DC2_MODE, m_dot_H2O_total_a_processar_H2, Y_O2_IN_H2, 0.999) 
    end_time_h2 = datetime.now()
    log_runtime(start_time_h2, end_time_h2)
    
    # --- 2. Executa a Simulação Única para O2 ---
    start_time_o2 = datetime.now()
    # Usando o nome da variável de escopo local (m_dot_H2O_total_a_processar_O2)
    df_o2_plot, result_o2, _ = executar_simulacao_unica('O2', M_DOT_G_O2, MODE, L_DEOXO_M, 'N/A', m_dot_H2O_total_a_processar_O2, 0.001, Y_H2_IN_O2, m_dot_liq_max_h2_ref=m_dot_liq_max_h2_ref) 
    end_time_o2 = datetime.now()
    log_runtime(start_time_o2, end_time_o2)
    
    # --- 3. Geração de Gráficos e Exibição ---
    
    print("\n\n" + "="*80)
    print("RESUMO DA SIMULAÇÃO COMPLETA (H2 e O2)")
    print("="*80)

    # CORREÇÃO: Criação de DataFrames Filtrados para remover SOEC (Entrada)
    COMPONENT_TO_REMOVE = 'SOEC (Entrada)'
    df_h2_plot_filtered = df_h2_plot[df_h2_plot['Componente'] != COMPONENT_TO_REMOVE].copy()
    df_o2_plot_filtered = df_o2_plot[df_o2_plot['Componente'] != COMPONENT_TO_REMOVE].copy()
    
    # Exibir estado final (usando o DF original para pegar o último ponto de purificação)
    exibir_estado_final(df_h2_plot, 'H2', MODE, L_DEOXO_M, DC2_MODE)
    exibir_estado_final(df_o2_plot, 'O2', MODE, L_DEOXO_M, 'N/A')
    
    # Exibir tabela detalhada e Resumo VSA
    exibir_tabela_detalhada(df_h2_plot_filtered, 'H2') 
    exibir_tabela_detalhada(df_o2_plot_filtered, 'O2') 
    
    # Exibição do Balanço Inicial de Água (Tabela)
    exibir_balanco_agua_inicial(df_h2_plot, df_o2_plot) 
    
    # Decisão de mostrar gráficos (mantido o stub para interatividade)
    MOSTRAR_GRAFICOS_FINAL = False # Definido como False para manter a saída consistente
    
    print("\nINICIANDO GERAÇÃO DE GRAFICOS (Salvando apenas no disco)...")

    # --- GERAÇÃO DE PLOTS (Usando os nomes das funções reais) ---
    if PLOT_MODULES_LOADED:
        try:
            plot_vazao_massica_total_e_removida(df_h2_plot_filtered, 'H2', MODE, L_DEOXO_M, DC2_MODE, MOSTRAR_GRAFICOS_FINAL)
            plot_vazao_massica_total_e_removida(df_o2_plot_filtered, 'O2', MODE, L_DEOXO_M, 'N/A', MOSTRAR_GRAFICOS_FINAL)
            
            plot_propriedades_empilhadas(df_h2_plot_filtered, 'H2', MODE, L_DEOXO_M, DC2_MODE, MOSTRAR_GRAFICOS_FINAL)
            plot_propriedades_empilhadas(df_o2_plot_filtered, 'O2', MODE, L_DEOXO_M, 'N/A', MOSTRAR_GRAFICOS_FINAL) 

            plot_vazao_liquida_acompanhante(df_h2_plot_filtered, 'H2', MODE, L_DEOXO_M, DC2_MODE, MOSTRAR_GRAFICOS_FINAL)
            plot_vazao_liquida_acompanhante(df_o2_plot_filtered, 'O2', MODE, L_DEOXO_M, 'N/A', MOSTRAR_GRAFICOS_FINAL)
            
            plot_q_breakdown(df_h2_plot, df_o2_plot, MODE, L_DEOXO_M, DC2_MODE, MOSTRAR_GRAFICOS_FINAL) 
            
            plot_vazao_agua_separada(df_h2_plot, 'H2', MODE, L_DEOXO_M, DC2_MODE, MOSTRAR_GRAFICOS_FINAL)
            plot_vazao_agua_separada(df_o2_plot, 'O2', MODE, L_DEOXO_M, DC2_MODE, MOSTRAR_GRAFICOS_FINAL)
            
            plot_fluxos_energia(df_h2_plot, df_o2_plot, MODE, L_DEOXO_M, DC2_MODE, MOSTRAR_GRAFICOS_FINAL)
            plot_agua_removida_total(df_h2_plot, df_o2_plot, MODE, L_DEOXO_M, DC2_MODE, MOSTRAR_GRAFICOS_FINAL)
            plot_impurezas_crossover(df_h2_plot, df_o2_plot, MODE, L_DEOXO_M, DC2_MODE, MOSTRAR_GRAFICOS_FINAL)
            
            # 🛑 CORREÇÃO FINAL DO ERRO FLOAT/LISTA DO DEOXO
            L_span_plot = result_h2['deoxo_L_span']
            T_profile_plot = result_h2['deoxo_T_profile_C']
            
            # Garante que L_span_plot e T_profile_plot são listas, transformando float/None
            # Usando uma verificação mais robusta para isenção de array numpy de 1 elemento (caso tenha vindo de outro modulo)
            if L_span_plot is None or isinstance(L_span_plot, (int, float)):
                L_span_plot = [float(L_span_plot)] if L_span_plot is not None else []
            elif isinstance(L_span_plot, np.ndarray):
                L_span_plot = L_span_plot.tolist()
                
            if T_profile_plot is None or isinstance(T_profile_plot, (int, float)):
                T_profile_plot = [float(T_profile_plot)] if T_profile_plot is not None else []
            elif isinstance(T_profile_plot, np.ndarray):
                T_profile_plot = T_profile_plot.tolist()

            # A verificação de erro do array agora é delegada ao loop for/len.
            if len(L_span_plot) > 0 and L_span_plot[0] is not None:
                 plot_deoxo_perfil(df_h2_plot, L_span_plot, T_profile_plot, result_h2['deoxo_X_O2'], result_h2['deoxo_T_max_calc'], MODE, L_DEOXO_M, MOSTRAR_GRAFICOS_FINAL)
            
            # Esquemas
            if 'plot_esquema_processo' in globals():
                plot_esquema_processo(df_h2_plot['Componente'].tolist(), 'H2 (SOEC + Purificacao)', MOSTRAR_GRAFICOS_FINAL)
                plot_esquema_processo(df_o2_plot['Componente'].tolist(), 'O2 (SOEC + Purificacao)', MOSTRAR_GRAFICOS_FINAL)
            if 'plot_drenos_descartados' in globals():
                plot_drenos_descartados(df_h2_plot, df_o2_plot, MOSTRAR_GRAFICOS_FINAL)
            
        except Exception as e:
            print(f"!!! ERRO na GERAÇÃO DE PLOTS: {e} !!!")
    else:
        print("AVISO: Plotagem desabilitada devido a erros de importação na inicialização.")

    
    # --- 5.3. SIMULAÇÃO DA LINHA DE DRENOS E RECIRCULAÇÃO ---
    if df_h2_plot is not None and df_o2_plot is not None and mixer_drenos:
        print("\n--- INICIANDO SIMULAÇÃO E PLOTAGEM DA LINHA DE DRENOS ---")
        try:
            if PLOT_MODULES_LOADED and 'plot_esquema_drenos' in globals():
                 plot_esquema_drenos(MOSTRAR_GRAFICOS_FINAL)
                 
            # 🛑 CORRIGIDO: Passando T_OUT_SOEC (agora 152.0 °C) como alvo do Boiler
            T_ALVO_BOILER_C = T_OUT_SOEC
            
            # Chamada do mixer_drenos, passando a nova temperatura alvo
            drenos_plot_data, _, estado_recirculacao_final = mixer_drenos.executar_simulacao_mixer(
                df_h2_plot, df_o2_plot, MOSTRAR_GRAFICOS_FINAL, T_ALVO_BOILER_C
            )
        except Exception as e:
             print(f"!!! ERRO na simulação/plotagem da Linha de Drenos: {e} !!!")
             drenos_plot_data = None
             estado_recirculacao_final = None

    # --- 6. ENCERRAMENTO ---
    end_time = datetime.now()
    log_runtime(start_time, end_time)