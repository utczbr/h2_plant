# main_simulator.py
# Script principal que orquestra a simulação completa.

import numpy as np
import pandas as pd
import sys
from datetime import datetime
import os # Necessário para manipular pastas

# Importa a lógica de execução do processo
# Importação mantida, pois process_execution está na raiz.
from process_execution import simular_sistema

# 1. Importações de Módulos Auxiliares (do diretório)
from constants_and_config import (
    LIMITES, T_IN_C, P_IN_BAR, M_DOT_G_H2, M_DOT_G_O2, 
    T_CHILLER_OUT_H2_C_C1, T_CHILLER_OUT_O2_C, 
    Y_O2_IN_H2, Y_H2_IN_O2, COMPONENTS_H2, COMPONENTS_O2, GASES, 
    MODE_DEOXO_FINAL, L_DEOXO_OTIMIZADO_M, DC2_MODE_FINAL, 
    M_DOT_H2O_LIQ_IN_H2_TOTAL_KGS, M_DOT_H2O_LIQ_IN_O2_TOTAL_KGS, # Total de Água Líquida Gerada no PEM (BASE)
    R_ARRASTE_H2, R_ARRASTE_O2, # Taxas de Arraste (Placeholder)
    P_IN_BAR 
)

# === MÓDULO DE PLOTAGEM E EXIBIÇÃO UNIFICADO (plot_reporter_base) ===
from plot_reporter_base import (
    exibir_tabela_detalhada, exibir_estado_final, exibir_resumo_vsa,
    log_runtime, salvar_e_exibir_plot, exibir_validacao_balanco_global,
    exibir_estado_recirculacao, exibir_balanco_agua_inicial
)

# === NOVAS IMPORTAÇÕES MODULARES (DA PASTA plots_modulos) - COM TRATAMENTO DE ERRO ROBUSTO ===
# Função de placeholder (stub)
stub_plot = lambda *args, **kwargs: print("AVISO: Plotagem não executada (Módulo ausente).")

# Lista dos módulos de plotagem a serem importados dinamicamente
PLOT_MODULES = [
    'plot_agua_removida_total', 'plot_vazao_massica_total_e_removida',
    'plot_propriedades_empilhadas', 'plot_impurezas_crossover', 
    'plot_vazao_agua_separada', 'plot_fluxos_energia', 
    'plot_deoxo_perfil', 'plot_esquema_processo', 
    'plot_vazao_liquida_acompanhante', 'plot_q_breakdown', 
    'plot_drenos_descartados', 'plot_esquema_drenos', 
    'plot_esquema_planta_completa', 'plot_recirculacao_mixer'
]

# Inicializa as variáveis de plotagem para o stub
for mod_name in PLOT_MODULES:
    globals()[mod_name] = stub_plot

# Tenta importar cada módulo e atribui a função principal ou o stub
for mod_name in PLOT_MODULES:
    try:
        module = __import__(f'plots_modulos.{mod_name}', fromlist=[mod_name])
        # Assume que o nome da função é o mesmo do módulo
        globals()[mod_name] = getattr(module, mod_name) 
    except ImportError as e:
        print(f"AVISO: Módulo '{mod_name}' ausente. {e}")

# Importação da nova linha de simulação (Drenos)
try:
    # 🌟 CORREÇÃO ROBUSTA: Tentar importar e verificar a presença de funções chave.
    import drain_mixer as mixer_drenos
    
    if not hasattr(mixer_drenos, 'executar_simulacao_mixer'):
        raise ImportError("Módulo drain_mixer encontrado, mas função chave ausente.")
        
except ImportError as e:
    print(f"ERRO CRÍTICO NA IMPORTAÇÃO DE DRENOS: {e}")
    print("!!! AVISO: A simulação da Linha de Drenos está desabilitada. !!!")
    mixer_drenos = None
    plot_esquema_drenos = stub_plot
    plot_esquema_planta_completa = stub_plot
    plot_recirculacao_mixer = stub_plot

    
# =================================================================
# === FUNÇÃO DE CONTROLE DE PLOTAGEM (AGORA INTERATIVA) ===
# =================================================================

def decidir_se_mostra_graficos():
    """Pergunta ao usuário se ele deseja exibir os gráficos (plt.show()).
    O salvamento é SEMPRE realizado via salvar_e_exibir_plot().
    """
    try:
        # 💥 CORREÇÃO: Tornando a função interativa novamente
        resposta = input("Deseja exibir os gráficos (plt.show())? (s/N): ").strip().lower()
        return resposta == 's'
    except EOFError:
        # Ambiente não interativo (ex: rodando como script não terminal)
        print("\nAVISO: Ambiente não interativo. Gráficos serão apenas salvos.")
        return False
    except Exception:
        # Fallback
        return False

# =================================================================
# === FUNÇÃO DE EXECUÇÃO ÚNICA (ADAPTADA AO NOVO BALANÇO E REFERÊNCIA) ===
# =================================================================

def executar_simulacao_unica(gas_fluido: str, m_dot_g_kg_s: float, deoxo_mode: str, L_deoxo: float, dc2_mode: str, m_dot_liq_max_h2_ref: float = None):
    """
    Executa a simulação completa.
    
    m_dot_liq_max_h2_ref: Valor de referência do M_DOT_LIQ_MAX_DEMISTER_KGS do H2,
                          usado para forçar o cálculo 1:2 no O2.
    """
    
    # --- CÁLCULO DA VAZÃO DE ÁGUA TOTAL QUE ENTRA NO FLUXO (Base) ---
    if gas_fluido == 'H2':
        m_dot_H2O_total_a_processar = M_DOT_H2O_LIQ_IN_H2_TOTAL_KGS 
    else:
        m_dot_H2O_total_a_processar = M_DOT_H2O_LIQ_IN_O2_TOTAL_KGS
    
    y_O2_in = Y_O2_IN_H2 if gas_fluido == 'H2' else 0.0
    y_H2_in = Y_H2_IN_O2 if gas_fluido == 'O2' else 0.0
    
    # --- FASE ÚNICA: Simulação Direta ---
    print(f"\n--- [FASE ÚNICA: FLUXO {gas_fluido} - SIMULAÇÃO DIRETA (KOD + Dry Cooler Inicial)] ---")
    
    # Passando o argumento de referência do H2
    result_real = simular_sistema(gas_fluido, m_dot_g_kg_s, deoxo_mode, L_deoxo, dc2_mode, 'KOD 1', 
                                  m_dot_H2O_total_fluxo_input=m_dot_H2O_total_a_processar, y_O2_in=y_O2_in, y_H2_in=y_H2_in, 
                                  m_dot_liq_max_h2_ref=m_dot_liq_max_h2_ref) 
    
    df_final = result_real['dataframe']
    
    # Captura o M_DOT_LIQ_MAX_DEMISTER_KGS (que pode ter sido forçado para O2)
    m_dot_liq_max_kgs = df_final[df_final['Componente'] == 'Entrada']['M_DOT_LIQ_MAX_DEMISTER_KGS'].iloc[0]
    
    # Retorna o valor de referência para a próxima chamada
    return df_final, result_real, m_dot_liq_max_kgs


# =================================================================
# === EXECUÇÃO DA SIMULAÇÃO PRINCIPAL ===
# =================================================================

if __name__ == '__main__':
    
    # 🌟 CORREÇÃO: Definir o tempo de início GLOBAL
    start_time = datetime.now()
    
    MODE = MODE_DEOXO_FINAL 
    L_DEOXO_M = L_DEOXO_OTIMIZADO_M
    DC2_MODE = DC2_MODE_FINAL 

    
    print("\n--- INICIANDO SIMULAÇÃO OTIMIZADA (Sem WGHE - KOD + Dry Cooler Inicial) ---")
    print(f"Modo Deoxo: {MODE}, Comprimento (L): {L_DEOXO_M:.3f} m, Pós-Deoxo: {DC2_MODE}")
    print(f"CHILLER H2 C1: {T_CHILLER_OUT_H2_C_C1:.1f} °C | CHILLER O2: {T_CHILLER_OUT_O2_C:.1f} °C")
    
    # --- 1. Executa a Simulação Única para H2 (Calcula a referência) ---
    start_time_h2 = datetime.now()
    # Capturando o valor de referência do H2
    df_h2_plot, result_h2, m_dot_liq_max_h2_ref = executar_simulacao_unica(GASES[0], M_DOT_G_H2, MODE, L_DEOXO_M, DC2_MODE)
    end_time_h2 = datetime.now()
    log_runtime(start_time_h2, end_time_h2)
    
    # --- 2. Executa a Simulação Única para O2 (Força a razão 1:2) ---
    start_time_o2 = datetime.now()
    # Passando o valor de referência do H2 para forçar o balanço 1:2
    df_o2_plot, result_o2, _ = executar_simulacao_unica(GASES[1], M_DOT_G_O2, MODE, L_DEOXO_M, 'N/A', m_dot_liq_max_h2_ref=m_dot_liq_max_h2_ref) 
    end_time_o2 = datetime.now()
    log_runtime(start_time_o2, end_time_o2)
    
    # --- 3. Geração de Gráficos e Exibição ---
    
    COMPONENTS_H2_FLOW = df_h2_plot['Componente'].tolist()
    COMPONENTS_O2_FLOW = df_o2_plot['Componente'].tolist()
    
    
    print("\n\n" + "="*80)
    print("RESUMO DA SIMULAÇÃO COMPLETA (H2 e O2) - Cenário Dry Cooler Inicial")
    print("="*80)
    
    # 2. Exibir o estado final
    exibir_estado_final(df_h2_plot, 'H2', MODE, L_DEOXO_M, DC2_MODE)
    exibir_estado_final(df_o2_plot, 'O2', MODE, L_DEOXO_M, 'N/A')
    
    # 3. Exibir tabela detalhada e Resumo VSA
    exibir_tabela_detalhada(df_h2_plot, 'H2') 
    exibir_tabela_detalhada(df_o2_plot, 'O2') 
    exibir_resumo_vsa(df_h2_plot) 
    
    # 🌟 Exibição do Balanço Inicial de Água (Tabela)
    exibir_balanco_agua_inicial(df_h2_plot, df_o2_plot) 
    
    # Decide se EXIBE os gráficos (o salvamento já é obrigatório via salvar_e_exibir_plot)
    MOSTRAR_GRAFICOS = decidir_se_mostra_graficos()
    
    if MOSTRAR_GRAFICOS:
        print("\nINICIANDO GERAÇÃO DE GRÁFICOS (E exibição)...")
    else:
        print("\nINICIANDO GERAÇÃO DE GRÁFICOS (Salvando apenas no disco)...")


    # --- GERAÇÃO DE GRÁFICOS ---
    # Implementação de TRY/EXCEPT em cada bloco para evitar que uma falha interrompa o restante
    
    
    if plot_vazao_massica_total_e_removida is not stub_plot:
        try:
            print("\nGerando Gráfico de Rastreamento de Vazões Mássicas (Gás, Vapor e Total) (H2)...")
            plot_vazao_massica_total_e_removida(df_h2_plot, 'H2', MODE, L_DEOXO_M, DC2_MODE, MOSTRAR_GRAFICOS)
            print("Gerando Gráfico de Rastreamento de Vazões Mássicas (Gás, Vapor e Total) (O2)...")
            plot_vazao_massica_total_e_removida(df_o2_plot, 'O2', MODE, L_DEOXO_M, 'N/A', MOSTRAR_GRAFICOS)
        except Exception as e:
            print(f"!!! ERRO na plotagem plot_vazao_massica_total_e_removida: {e} !!!")


    if plot_propriedades_empilhadas is not stub_plot:
        try:
            print("\nGerando Gráficos de Propriedades Empilhadas (H2)...")
            plot_propriedades_empilhadas(df_h2_plot, 'H2', MODE, L_DEOXO_M, DC2_MODE, MOSTRAR_GRAFICOS)
            print("Gerando Gráficos de Propriedades Empilhadas (O2)...")
            plot_propriedades_empilhadas(df_o2_plot, 'O2', MODE, L_DEOXO_M, 'N/A', MOSTRAR_GRAFICOS) 
        except Exception as e:
            print(f"!!! ERRO na plotagem plot_propriedades_empilhadas: {e} !!!")


    if plot_vazao_liquida_acompanhante is not stub_plot:
        try:
            print("\nGerando Gráfico da Vazão Mássica de Água Líquida Acompanhante (H2)...")
            plot_vazao_liquida_acompanhante(df_h2_plot, 'H2', MODE, L_DEOXO_M, DC2_MODE, MOSTRAR_GRAFICOS)
            print("Gerando Gráfico da Vazão Mássica de Água Líquida Acompanhante (O2)...")
            plot_vazao_liquida_acompanhante(df_o2_plot, 'O2', MODE, L_DEOXO_M, 'N/A', MOSTRAR_GRAFICOS)
        except Exception as e:
             print(f"!!! ERRO na plotagem plot_vazao_liquida_acompanhante: {e} !!!")

    if plot_q_breakdown is not stub_plot:
        try:
            print("\nGerando Gráfico de Quebra de Carga Térmica (Q_dot) por Fase...")
            plot_q_breakdown(df_h2_plot, df_o2_plot, MODE, L_DEOXO_M, DC2_MODE, MOSTRAR_GRAFICOS) 
        except Exception as e:
             print(f"!!! ERRO na plotagem plot_q_breakdown: {e} !!!")

    if plot_vazao_agua_separada is not stub_plot:
        try:
            print("\nGerando Gráfico de Rastreamento de Água (H2) em PPM Molar...")
            plot_vazao_agua_separada(df_h2_plot, 'H2', MODE, L_DEOXO_M, DC2_MODE, MOSTRAR_GRAFICOS)
            print("Gerando Gráfico de Rastreamento de Água (O2) em PPM Molar...")
            plot_vazao_agua_separada(df_o2_plot, 'O2', MODE, L_DEOXO_M, DC2_MODE, MOSTRAR_GRAFICOS)
        except Exception as e:
             print(f"!!! ERRO na plotagem plot_vazao_agua_separada: {e} !!!")
    
    if plot_fluxos_energia is not stub_plot:
        try:
            print("\nGerando Gráficos de Fluxos de Energia (Completo - Térmica e Elétrica)...")
            plot_fluxos_energia(df_h2_plot, df_o2_plot, MODE, L_DEOXO_M, DC2_MODE, MOSTRAR_GRAFICOS)
        except Exception as e:
             print(f"!!! ERRO na plotagem plot_fluxos_energia: {e} !!!")

    if plot_agua_removida_total is not stub_plot:
        try:
            print("\nGerando Gráfico da Vazão de Água Líquida Removida por Componente...")
            plot_agua_removida_total(df_h2_plot, df_o2_plot, MODE, L_DEOXO_M, DC2_MODE, MOSTRAR_GRAFICOS)
        except Exception as e:
             print(f"!!! ERRO na plotagem plot_agua_removida_total: {e} !!!")
    
    if plot_impurezas_crossover is not stub_plot:
        try:
            print("\nGerando Gráfico da Evolução da Impureza O₂ (H2) no Deoxo...")
            plot_impurezas_crossover(df_h2_plot, df_o2_plot, MODE, L_DEOXO_M, DC2_MODE, MOSTRAR_GRAFICOS)
        except Exception as e:
             print(f"!!! ERRO na plotagem plot_impurezas_crossover: {e} !!!")
    
    if plot_deoxo_perfil is not stub_plot:
        try:
            print("\nGerando Gráfico do Perfil de Temperatura e Conversão no Reator Deoxo (H2)...")
            plot_deoxo_perfil(df_h2_plot, result_h2['deoxo_L_span'], result_h2['deoxo_T_profile_C'], result_h2['deoxo_X_O2'], result_h2['deoxo_T_max_calc'], MODE, L_DEOXO_M, MOSTRAR_GRAFICOS)
        except Exception as e:
             print(f"!!! ERRO na plotagem plot_deoxo_perfil: {e} !!!")
    
    if plot_esquema_processo is not stub_plot:
        try:
            print("\nGerando Esquema do Processo (H2)...")
            plot_esquema_processo(COMPONENTS_H2_FLOW, 'H2 (KOD+DC Inicial)', MOSTRAR_GRAFICOS)
            print("Gerando Esquema do Processo (O2)...")
            plot_esquema_processo(COMPONENTS_O2_FLOW, 'O2 (KOD+DC Inicial)', MOSTRAR_GRAFICOS)
        except Exception as e:
             print(f"!!! ERRO na plotagem plot_esquema_processo: {e} !!!")
        
    if plot_drenos_descartados is not stub_plot:
        try:
            print("\nGerando Gráficos dos Drenos Não Reaproveitados (Coalescedor, VSA, PSA)...")
            plot_drenos_descartados(df_h2_plot, df_o2_plot, MOSTRAR_GRAFICOS)
        except Exception as e:
             print(f"!!! ERRO na plotagem plot_drenos_descartados: {e} !!!")
    
    # --- EXECUÇÃO DA LINHA DE DRENOS (Mixer) ---
    estado_recirculacao_final = None 
    drenos_plot_data = None
    
    if mixer_drenos is not None and hasattr(mixer_drenos, 'executar_simulacao_mixer'):
        try:
            if plot_esquema_drenos is not stub_plot:
                 print("\nGerando Esquema do Processo da Linha de Drenos...")
                 plot_esquema_drenos(MOSTRAR_GRAFICOS)
                 
            # Chamada do mixer_drenos (passando df_h2_plot e df_o2_plot)
            drenos_plot_data, _, estado_recirculacao_final = mixer_drenos.executar_simulacao_mixer(df_h2_plot, df_o2_plot, MOSTRAR_GRAFICOS)
        except Exception as e:
             print(f"!!! ERRO na simulação/plotagem da Linha de Drenos: {e} !!!")
             drenos_plot_data = None
             estado_recirculacao_final = None
    else:
        print("\n!!! AVISO: A simulação da Linha de Drenos não foi executada pois o módulo 'drain_mixer' ou suas funções não foram importadas. !!!")

    # --- PLOTAGEM DO ESQUEMA COMPLETO DA PLANTA ---
    if plot_esquema_planta_completa is not stub_plot and estado_recirculacao_final:
         try:
             print("\nGerando Esquema COMPLETO da Planta (H2, O2 e Drenos)...")
             plot_esquema_planta_completa(df_h2_plot, df_o2_plot, estado_recirculacao_final, MODE, L_DEOXO_M, DC2_MODE, MOSTRAR_GRAFICOS)
         except Exception as e:
              print(f"!!! ERRO na plotagem do Esquema Completo da Planta: {e} !!!")
    elif plot_esquema_planta_completa is not stub_plot:
        print("\nAVISO: Esquema da Planta Completa NÃO gerado devido a dados de dreno ausentes ou módulo não importado.")
    
    # --- 5. TEMPO DE EXECUÇÃO ---
    # Registrar o tempo de fim GLOBAL e usar o tempo GLOBAL no log
    end_time = datetime.now()
    log_runtime(start_time, end_time)