# plot_reporter_base.py (Versão COMPLETA com funções de estado final garantidas)

import numpy as np 
import pandas as pd 
import CoolProp.CoolProp as CP
import sys
import matplotlib.pyplot as plt 
import os 

# Importa as constantes globais do módulo dedicado
from constants_and_config import (
    LIMITES, Y_H2O_LIMIT_MOLAR, P_IN_BAR, T_IN_C, M_DOT_G_H2, M_DOT_G_O2,
    T_JACKET_DEOXO_C, P_VSA_PROD_BAR, P_VSA_REG_BAR, COMPONENTS_H2, COMPONENTS_O2,
    M_DOT_H2O_RECIRC_TOTAL_KGS, M_DOT_H2O_CONSUMIDA_KGS, 
    M_H2O_TOTAL_H2_KGS, M_H2O_TOTAL_O2_KGS 
)

# 💥 REMOVIDO: Importação de modelo_vsa
try:
    pass
except ImportError as e:
    print(f"AVISO: Falha na importação de modelos auxiliares para plots comparativos: {e}")
    
    def modelo_valvula_isoentalpica(fluido, T_in_K, P_in_Pa, P_out_Pa):
        if fluido == 'hydrogen':
            T_out_K = T_in_K - 0.5 
        else:
            T_out_K = T_in_K
        return {'SAIDA': {'T_K': T_out_K, 'P_Pa': P_out_Pa}}


# =================================================================
# === FUNÇÃO CENTRAL DE SALVAMENTO E EXIBIÇÃO ===
# =================================================================

# 🛑 NOVO CAMINHO RAIZ FORNECIDO PELO USUÁRIO
CAMINHO_RAIZ_GRAFICOS = r'C:\Users\tusaw\OneDrive\Documentos\projeto hidrogenio\SOEC\Graficos'

def salvar_e_exibir_plot(nome_arquivo: str, mostrar_grafico: bool = True):
    """Salva o gráfico no caminho absoluto predefinido e, opcionalmente, o exibe."""
    try:
        if not os.path.exists(CAMINHO_RAIZ_GRAFICOS):
            os.makedirs(CAMINHO_RAIZ_GRAFICOS, exist_ok=True)
            
        caminho_completo = os.path.join(CAMINHO_RAIZ_GRAFICOS, nome_arquivo)
        plt.savefig(caminho_completo)
        print(f"Gráfico '{nome_arquivo}' salvo com sucesso em: {caminho_completo}")
        
        if mostrar_grafico:
            plt.show()
            
        plt.close()
        
    except Exception as e:
        print(f"❌ ERRO ao salvar ou exibir o gráfico '{nome_arquivo}': {e}")
        plt.close()

# =================================================================
# === FUNÇÕES AUXILIARES DE PLOTAGEM/CÁLCULO ===
# =================================================================

def calcular_vazao_massica_total_completa(df: pd.DataFrame) -> pd.Series:
    """
    Calcula a vazão mássica total real (Gás Principal + Vapor H2O + H2O Líquida Acompanhante).
    """
    m_dot_total_completa = df['m_dot_mix_kg_s'] + df['m_dot_H2O_liq_accomp_kg_s']
    return m_dot_total_completa


def log_runtime(start_time, end_time):
    """Calcula e imprime a duração total da execução do pipeline."""
    duration = end_time - start_time
    print(f"\nTempo de Execução Total: {duration}")

# =================================================================
# === FUNÇÕES DE EXIBIÇÃO DE TABELA E RESUMO ===
# =================================================================
def exibir_tabela_detalhada(df: pd.DataFrame, gas_fluido: str):
    """Exibe uma tabela detalhada dos estados do fluido em cada componente."""
    print("\n" + "="*140)
    print(f"TABELA DE DADOS DETALHADOS - FLUXO DE {gas_fluido}")
    print("="*140)
    
    # Colunas comuns e essenciais
    cols = ['Componente', 'T_C', 'P_bar', 'm_dot_gas_kg_s', 'm_dot_mix_kg_s', 'y_H2O', 'm_dot_H2O_liq_accomp_kg_s', 'Q_dot_fluxo_W', 'W_dot_comp_W'] 
    
    # Adicionar impurezas relevantes
    if gas_fluido == 'H2':
        cols.append('y_O2')
        df_display = df[cols].copy()
        imp_col = 'y_O2'
        imp_name = 'y_O2 (ppm molar)' 
    else:
        cols.append('y_H2')
        df_display = df[cols].copy()
        imp_col = 'y_H2'
        imp_name = 'y_H2 (ppm molar)' 

    # Formatação dos dados
    df_display['T_C'] = df_display['T_C'].map('{:.2f}'.format)
    df_display['P_bar'] = df_display['P_bar'].map('{:.2f}'.format)
    df_display['m_dot_gas_kg_s'] = df_display['m_dot_gas_kg_s'].map('{:.5f}'.format)
    df_display['m_dot_mix_kg_s'] = df_display['m_dot_mix_kg_s'].map('{:.5f}'.format)
    df_display['y_H2O'] = df_display['y_H2O'].map('{:.2e}'.format)
    
    # CONVERSÃO: m_dot H2O Líq. (kg/s) para kg/h
    df_display['m_dot_H2O_liq_accomp_kg_h'] = (df_display['m_dot_H2O_liq_accomp_kg_s'].astype(float) * 3600).map('{:.2f}'.format) 
    df_display.drop(columns=['m_dot_H2O_liq_accomp_kg_s'], inplace=True)
    
    df_display['Q_dot_fluxo_W'] = df_display['Q_dot_fluxo_W'].map('{:.2f}'.format)
    df_display['W_dot_comp_W'] = df_display['W_dot_comp_W'].map('{:.2f}'.format)
    
    # Formatação das impurezas (PPM e notação científica)
    df_display[imp_name] = df_display[imp_col].map(lambda x: f'{x:.2e} ({(x * 1e6):.2f})')
    df_display.drop(columns=[imp_col], inplace=True)


    df_display.rename(columns={
        'T_C': 'T (°C)',
        'P_bar': 'P (bar)',
        'm_dot_gas_kg_s': 'm_dot Gás Princ. (kg/s)',
        'm_dot_mix_kg_s': 'm_dot Mistura (kg/s)',
        'y_H2O': 'y_H2O (molar)',
        'm_dot_H2O_liq_accomp_kg_h': 'm_dot H₂O Líq. Acomp. (kg/h)', 
        'Q_dot_fluxo_W': 'Q dot (W)',
        'W_dot_comp_W': 'W dot (W)'
    }, inplace=True)

    print(df_display.to_string(index=False))
    print("="*140)


        
# FIM DA FUNÇÃO exibir_tabela_detalhada

# 💥 REMOVIDO: def exibir_resumo_compressor_multiestagio(history: list):

# 🛑 FUNÇÃO QUE ESTAVA FALTANDO/PROBLEMA DE ESCOPO
def exibir_estado_final(df: pd.DataFrame, gas_fluido: str, deoxo_mode: str, L_deoxo: float, dc2_mode: str):
    """Exibe o estado final do fluido."""
    
    # 💥 CORREÇÃO: Pega o último componente no DataFrame, que é o último componente de purificação.
    estado_final = df.iloc[-1] 
    comp_final = estado_final['Componente']
    
    print("\n" + "="*80)
    print(f"ESTADO FINAL DO FLUIDO: {gas_fluido} (SAÍDA DA PURIFICAÇÃO)")
    print(f"Modo Deoxo: {deoxo_mode} | L_Deoxo: {L_deoxo:.3f} m")
    if gas_fluido == 'H2':
        # 💥 CORREÇÃO: Pós-Deoxo é apenas PSA
        print(f"Processo Final H2: {comp_final} (PSA)")
    print("="*80)
    print(f"Componente Final: {estado_final['Componente']}")
    print(f"Temperatura (T): {estado_final['T_C']:.2f} °C")
    print(f"Pressão (P): {estado_final['P_bar']:.2f} bar")
    print(f"Vazão Mássica de Gás Principal: {estado_final['m_dot_gas_kg_s']:.5f} kg/s") 
    print(f"Vazão Mássica da Mistura Total: {estado_final['m_dot_mix_kg_s']:.5f} kg/s")
    
    # NOVO CAMPO
    m_dot_H2O_liq_out = estado_final.get('m_dot_H2O_liq_accomp_kg_s', 0.0)
    print(f"Vazão Mássica de Água Líquida Acompanhante: {m_dot_H2O_liq_out * 3600:.2f} kg/h") # CONVERSÃO
    
    print(f"Fração Mássica de H₂O (w_H₂O): {estado_final['w_H2O']:.2e} ({estado_final['w_H2O'] * 1e6:.6f} ppm)")
    
    if gas_fluido == 'H2':
        y_o2_val = estado_final['y_O2']
        print(f"Fração Molar de O₂ (y_O₂): {y_o2_val:.2e} ({y_o2_val * 1e6:.6f} ppm)")
    else: # O2
        y_h2_val = estado_final['y_H2']
        print(f"Fração Molar de H₂ (y_H₂): {y_h2_val:.2e} ({y_h2_val * 1e6:.6f} ppm)")

    print(f"Entalpia Mássica da Mistura: {estado_final['H_mix_J_kg'] / 1000:.2f} kJ/kg")
    print(f"Estado da Água: O gás de saída está {estado_final['Estado_H2O']}")
    
    # Soma de Água Líquida (Drenos) Removida TOTAL (Exclui SOEC Entrada/Saída)
    componentes_purificacao = [comp for comp in df['Componente'].tolist() if comp not in ['SOEC (Entrada)', 'SOEC (Saída)']]
    agua_removida_total = df[df['Componente'].isin(componentes_purificacao)]['Agua_Pura_Removida_H2O_kg_s'].sum()
    print(f"Água Líquida (Drenos) Removida TOTAL (Purificação): {agua_removida_total * 3600:.2f} kg/h") # CONVERSÃO
    
    if 'KOD 1' in df['Componente'].values:
        status_kod_1 = df[df['Componente'] == 'KOD 1']['Status_KOD'].iloc[0]
        print(f"Status do KOD 1: {status_kod_1}")

    print("="*80)

def exibir_resumo_vsa(df_h2: pd.DataFrame):
    """Função removida. Manter apenas um stub para evitar erros de chamada."""
    print("\n--- VSA não encontrado no fluxo H2. ---")
    return
# FIM DA FUNÇÃO exibir_resumo_vsa


# =================================================================
# === FUNÇÕES DE EXIBIÇÃO DE ENERGIA (Para a Bomba e Boiler) ===
# =================================================================

def exibir_resultados_bomba(res_bomba: dict):
    """Exibe os resultados da simulação da bomba de forma organizada."""
    print("\n" + "="*50)
    print("RESUMO DA SIMULAÇÃO DA BOMBA")
    print("="*50)
    
    data = {
        'Propriedade': ['Trabalho Real (w_real)', 'Pot. Fluido (Ẇ_fluido)', 'Pot. Eixo (Ẇ_eixo)', 'Pot. Elétrica (Ẇ_elétrico)'],
        'Valor': [res_bomba['W_real_kJ_kg'], res_bomba['Pot_Fluido_kW'], res_bomba['Pot_Eixo_kW'], res_bomba['Pot_Eletrica_kW']],
        'Unidade': ['kJ/kg', 'kW', 'kW', 'kW']
    }
    df_display = pd.DataFrame(data)
    df_display['Valor'] = df_display['Valor'].map('{:.3f}'.format)
    
    print(df_display.to_string(index=False))
    
    print(f"\nEstado Final (2): P={res_bomba['P_out_kPa']/100:.2f} bar, T={res_bomba['T_out_C']:.2f} °C, h={res_bomba['h_out_kJ_kg']:.2f} kJ/kg")
    print("="*50)


def exibir_resultados_boiler(res_boiler: dict, T_final_C: float):
    """Exibe os resultados da simulação do boiler de forma organizada."""
    print("\n" + "="*50)
    print("RESUMO DA SIMULAÇÃO DO BOILER ELÉTRICO")
    print("="*50)
    
    data = {
        'Propriedade': ['Delta H (Específico)', 'Pot. Térmica Necessária (Q̇_necessário)', 'Pot. Elétrica de Consumo (Ẇ_elétrico)', 'Temp. de Saída (T_final)'],
        'Valor': [res_boiler['Delta_H_kJ_kg'], res_boiler['Q_necessario_kW'], res_boiler['W_eletrico_kW'], T_final_C],
        'Unidade': ['kJ/kg', 'kW', 'kW', '°C']
    }
    df_display = pd.DataFrame(data)
    df_display['Valor'] = df_display['Valor'].map(lambda x: f'{x:.3f}' if isinstance(x, (int, float)) else str(x))
    
    print(df_display.to_string(index=False))
    print("="*50)


# =================================================================
# === FUNÇÕES DE EXIBIÇÃO DE DRENOS ===
# =================================================================

def exibir_tabela_drenos_raw(drenos_list: list, gas_fluido: str):
    """Exibe uma tabela detalhada dos drenos brutos."""
    
    if not drenos_list:
        print(f"\n--- DRENOS BRUTOS {gas_fluido} ---")
        print("Nenhum dreno bruto para exibir.")
        return

    df_drenos = pd.DataFrame(drenos_list).copy()
    
    df_drenos['m_dot_kg_h'] = df_drenos['m_dot'] * 3600 
    df_drenos.rename(columns={'m_dot': 'm_dot_kg_s'}, inplace=True)

    impureza_dissolvida = 'O₂' 
    
    cols = ['Componente', 'm_dot_kg_s', 'm_dot_kg_h', 'T', 'P_bar', 'Gas_Dissolvido_in_mg_kg']
    df_display = df_drenos[cols].rename(columns={
        'T': 'T (°C)',
        'P_bar': 'P (bar)',
        'Gas_Dissolvido_in_mg_kg': f'Conc. {impureza_dissolvida} Diss. (mg/kg)',
        'm_dot_kg_s': 'm_dot (kg/s)',
        'm_dot_kg_h': 'm_dot (kg/h)'
    })

    # Formatação dos dados
    df_display['m_dot (kg/s)'] = df_display['m_dot (kg/s)'].map('{:.5f}'.format) 
    df_display['m_dot (kg/h)'] = df_display['m_dot (kg/h)'].map('{:.2f}'.format)
    df_display['T (°C)'] = df_display['T (°C)'].map('{:.1f}'.format)
    df_display['P (bar)'] = df_display['P (bar)'].map('{:.1f}'.format)
    df_display[f'Conc. {impureza_dissolvida} Diss. (mg/kg)'] = df_display[f'Conc. {impureza_dissolvida} Diss. (mg/kg)'].map('{:.4f}'.format)
    
    print("\n" + "="*100)
    print(f"TABELA DE DADOS DOS DRENOS BRUTOS (INPUTS) - FLUXO DE {gas_fluido}")
    print("="*100)
    try:
        print(df_display.to_markdown(index=False)) 
    except ImportError:
         print(df_display.to_string(index=False)) 
    print("="*100)


def exibir_tabela_processo_dreno(entrada: dict, saida: dict, gas_fluido: str):
    """
    Exibe uma tabela formatada do Dreno Agregado (Mixer 1 OUT).
    """
    if not entrada or not saida:
        return

    impureza_dissolvida = 'O₂'
    
    data = {
        'Propriedade': ['Componente', 'Vazão (kg/h)', 'Temperatura (°C)', 'Pressão (bar)', 'Entalpia (kJ/kg)', f'Conc. {impureza_dissolvida} Diss. (mg/kg)'],
        'Mixer 1 OUT': [
            entrada['Componente'],
            entrada['m_dot_kg_h'],
            entrada['T'],
            entrada['P_bar'],
            entrada['h_kJ_kg'],
            entrada['C_diss_mg_kg']
        ]
    }

    df_display = pd.DataFrame(data).set_index('Propriedade')
    
    # Formatação dos floats
    df_display.iloc[1:4] = df_display.iloc[1:4].map(lambda x: f'{x:.2f}')
    df_display.iloc[4:] = df_display.iloc[4:].map(lambda x: f'{x:.4f}')

    print("\n" + "="*80)
    print(f"TABELA DE DADOS DO DRENO AGREGADO: FLUXO DE {gas_fluido}")
    print("="*80)
    try:
        print(df_display.to_markdown())
    except ImportError:
        print(df_display.to_string()) 
    print("="*80)
    
    
def exibir_estado_final_mixer(resultado_mixer: dict):
    """
    Exibe uma tabela de resumo das propriedades termodinâmicas e vazão
    da corrente final de água drenada (saída do Mixer).
    """
    if not resultado_mixer or 'erro' in resultado_mixer:
        print("\n--- ERRO/AVISO: Não foi possível obter o estado final do Mixer de Drenos. ---")
        if 'erro' in resultado_mixer:
             print(f"Detalhes do Erro: {resultado_mixer['erro']}")
        return
        
    print("\n" + "="*80)
    print("ESTADO FINAL DA ÁGUA DRENADA (SAÍDA DO MIXER)")
    print("="*80)
    
    # Extração segura dos dados
    T_C = resultado_mixer.get('T_out_C', np.nan)
    P_bar = resultado_mixer.get('P_out_bar', np.nan)
    M_dot_kg_s = resultado_mixer.get('M_dot_H2O_final_kg_s', 0.0)
    H_J_kg = resultado_mixer.get('H_liq_out_J_kg', np.nan)
    
    # Concentrações (Baseado na saída do Flash Drum, se disponível)
    Conc_H2_mg_kg = resultado_mixer.get('Conc_H2_final_mg_kg', np.nan)
    Conc_O2_mg_kg = resultado_mixer.get('Conc_O2_final_mg_kg', np.nan)
    
    # Criação do DataFrame para exibição como Tabela
    data = {
        'Propriedade': ['Temperatura', 'Pressão', 'Vazão Mássica Total', 'Entalpia Mássica', 'Concentração H₂ Dissolvido', 'Concentração O₂ Dissolvido'],
        'Valor': [T_C, P_bar, M_dot_kg_s, H_J_kg, Conc_H2_mg_kg, Conc_O2_mg_kg],
        'Unidade': ['°C', 'bar', 'kg/s', 'kJ/kg', 'mg/kg', 'mg/kg']
    }
    df_display = pd.DataFrame(data)
    
    # Formatação da saída
    df_display['Valor'] = df_display.apply(
        lambda row: f"{row['Valor']:.2f}" if row['Unidade'] in ['°C', 'bar'] else 
                    f"{row['Valor']:.5f}" if row['Unidade'] == 'kg/s' else
                    f"{row['Valor'] / 1000:.2f}" if row['Unidade'] == 'kJ/kg' else # Converte J/kg para kJ/kg
                    f"{row['Valor']:.4f}", axis=1 # Concentrações com 4 casas
    )
    
    # Ajuste da Entalpia
    df_display.loc[df_display['Propriedade'] == 'Entalpia Mássica', 'Unidade'] = 'kJ/kg'
    
    print(df_display.to_string(index=False))
    print("="*80)
    print(f"Vazão de Água (kg/h): {M_dot_kg_s * 3600:.2f}")
    print("="*80)
    
    
def exibir_estado_recirculacao(resultado_recirculacao: dict):
    """
    Exibe uma tabela de resumo das propriedades termodinâmicas da
    corrente de água após a reposição (pronta para o SOEC).
    """
    if not resultado_recirculacao or 'erro' in resultado_recirculacao:
        print("\n--- ERRO/AVISO: Não foi possível obter o estado final da Água de Recirculação. ---")
        if 'erro' in resultado_recirculacao:
             print(f"Detalhes do Erro: {resultado_recirculacao['erro']}")
        return
        
    print("\n" + "="*80)
    print("ESTADO FINAL DA ÁGUA DE RECIRCULAÇÃO (PÓS-REPOSIÇÃO)")
    print("="*80)
    
    # Extração segura dos dados
    T_C = resultado_recirculacao.get('T_out_C', np.nan)
    P_bar = resultado_recirculacao.get('P_out_bar', np.nan)
    M_dot_kg_s = resultado_recirculacao.get('M_dot_out_kgs', 0.0)
    H_J_kg = resultado_recirculacao.get('H_out_J_kg', np.nan)
    M_dot_makeup_kgs = resultado_recirculacao.get('M_dot_makeup_kgs', 0.0)
    
    # Criação do DataFrame para exibição como Tabela
    data = {
        'Propriedade': ['Temperatura', 'Pressão', 'Vazão Mássica Total', 'Entalpia Mássica', 'Água Reposição Adicionada'],
        'Valor': [T_C, P_bar, M_dot_kg_s, H_J_kg, M_dot_makeup_kgs],
        'Unidade': ['°C', 'bar', 'kg/s', 'kJ/kg', 'kg/s']
    }
    df_display = pd.DataFrame(data)
    
    # Formatação da saída
    df_display['Valor'] = df_display.apply(
        lambda row: f"{row['Valor']:.2f}" if row['Unidade'] in ['°C', 'bar'] else 
                    f"{row['Valor']:.5f}" if row['Unidade'] == 'kg/s' else
                    f"{row['Valor'] / 1000:.2f}" if row['Unidade'] == 'kJ/kg' else 
                    f"{row['Valor']:.4f}", axis=1 
    )
    
    # Ajuste da Entalpia
    df_display.loc[df_display['Propriedade'] == 'Entalpia Mássica', 'Unidade'] = 'kJ/kg'
    
    print(df_display.to_string(index=False))
    print("="*80)
    print(f"Vazão de Água (kg/h): {M_dot_kg_s * 3600:.2f} (Alvo: {M_DOT_H2O_RECIRC_TOTAL_KGS * 3600:.2f} kg/h)")
    print(f"Água de Reposição Necessária: {M_dot_makeup_kgs * 3600:.2f} kg/h")
    print("="*80)
    
    
def exibir_validacao_balanco_global(m_dot_drenada_total_kgs: float, m_dot_consumida_kgs: float):
    """
    Calcula e exibe a validação do balanço de massa global de água:
    Água Total Drenada (Pool) vs. Consumo Estequiométrico.
    """
    
    V_drenada_total_kg_h = m_dot_drenada_total_kgs * 3600
    V_recirc_total_kg_h = M_DOT_H2O_RECIRC_TOTAL_KGS * 3600
    V_consumida_esteq_kg_h = m_dot_consumida_kgs * 3600
    
    # Água que deveria ser drenada (Pool Total)
    V_pool_esperado_kg_h = V_recirc_total_kg_h - V_consumida_esteq_kg_h
    
    # Diferença (Pool Esperado - Pool Real)
    Diferenca_Pool_kg_h = V_pool_esperado_kg_h - V_drenada_total_kg_h
    
    # Verificação de Fechamento (Fechamento = 1 - Abs(Desvio/Alvo))
    Fechamento_pct = 100.0 - (abs(Diferenca_Pool_kg_h) / V_pool_esperado_kg_h) * 100
    
    # Cálculo da Reposição que seria necessária se TODA a água drenada voltasse
    Reposição_Global_Necessaria_kg_h = V_recirc_total_kg_h - V_drenada_total_kg_h
    
    print("\n" + "#"*80)
    print("### VALIDAÇÃO DO BALANÇO DE MASSA GLOBAL (MÉTRICA DE CONTROLE) ###")
    print("#"*80)
    
    print(f"Vazão Total de Recirculação (Alvo): {V_recirc_total_kg_h:.2f} kg/h")
    print(f"Vazão Consumida (Esteq.):          {V_consumida_esteq_kg_h:.2f} kg/h")
    print("-" * 50)
    print(f"Água do Pool Esperada (Alvo - Consumo): {V_pool_esperado_kg_h:.2f} kg/h")
    print(f"Água Total Drenada (Massa Removida):   {V_drenada_total_kg_h:.2f} kg/h")
    print("\n--- Análise ---")
    print(f"Diferença no Pool (Esperada - Real):    {Diferenca_Pool_kg_h:.2f} kg/h")
    print(f"FECHAMENTO FÍSICO DO BALANÇO:           {Fechamento_pct:.4f} %")
    
    print(f"\nREPOSIÇÃO NECESSÁRIA (SE TUDO DRENADO VOLTASSE): {Reposição_Global_Necessaria_kg_h:.2f} kg/h")
    print("#"*80)
    
    
def exibir_balanco_agua_inicial(df_h2: pd.DataFrame, df_o2: pd.DataFrame):
    """
    Exibe uma tabela de resumo dos cálculos iniciais de balanço de água
    e partição Vapor/Líquido/Dreno SOEC, rastreando as vazões.
    """
    
    # 💥 CORREÇÃO: Extrair a linha de 'SOEC (Saída)'
    h2_out = df_h2[df_h2['Componente'] == 'SOEC (Saída)'].iloc[0]
    o2_out = df_o2[df_o2['Componente'] == 'SOEC (Saída)'].iloc[0]
    
    # --------------------------------------------------------------------------------
    # CÁLCULOS DOS VALORES (x, y, z, w) em kg/h
    # --------------------------------------------------------------------------------
    
    # H2 Side (x and z)
    x_vapor_kgs = h2_out.get('M_DOT_VAPOR_ENTRADA_KGS_X_Y', 0.0) 
    z_liq_accomp_kgs = h2_out.get('m_dot_H2O_liq_accomp_kg_s', 0.0) 
    m_dot_h2o_total_h2_kgs = M_H2O_TOTAL_H2_KGS 
    
    # Água em Vapor (x): 
    x_vapor_kg_h = x_vapor_kgs * 3600 
    # Água Líquida Acompanhante (z): 
    z_liq_accomp_kg_h = z_liq_accomp_kgs * 3600
    # Água Removida no Dreno SOEC: (AGORA SEMPRE ZERO)
    dreno_soec_h2_kg_h = h2_out.get('Agua_Pura_Removida_H2O_kg_s', 0.0) * 3600 
    
    # O2 Side (y and w)
    y_vapor_kgs = o2_out.get('M_DOT_VAPOR_ENTRADA_KGS_X_Y', 0.0) 
    w_liq_accomp_kgs = o2_out.get('m_dot_H2O_liq_accomp_kg_s', 0.0) 
    m_dot_h2o_total_o2_kgs = M_H2O_TOTAL_O2_KGS 
    
    # Água em Vapor (y): 
    y_vapor_kg_h = y_vapor_kgs * 3600
    # Água Líquida Acompanhante (w): 
    w_liq_accomp_kg_h = w_liq_accomp_kgs * 3600
    # Água Removida no Dreno SOEC: (AGORA SEMPRE ZERO)
    dreno_soec_o2_kg_h = o2_out.get('Agua_Pura_Removida_H2O_kg_s', 0.0) * 3600
    
    # --------------------------------------------------------------------------------
    # CONSTRUÇÃO DA TABELA
    # --------------------------------------------------------------------------------
    
    # CÁLCULO DE VALIDAÇÃO (Massa Total)
    Total_H2_Entrada_kg_h = m_dot_h2o_total_h2_kgs * 3600
    Total_O2_Entrada_kg_h = m_dot_h2o_total_o2_kgs * 3600
    
    data = {
        'Fluxo': ['H₂', 'O₂'],
        # Mudar a ordem para que todas as colunas sejam float e convertidas juntas.
        'Vazão Total de Água que Segue (kg/h)': [Total_H2_Entrada_kg_h, Total_O2_Entrada_kg_h],
        'Vapor Saturado (kg/h)': [x_vapor_kg_h, y_vapor_kg_h],
        'Líquido Acomp. que Segue (kg/h)': [z_liq_accomp_kg_h, w_liq_accomp_kg_h],
        'Dreno SOEC Removido (kg/h)': [dreno_soec_h2_kg_h, dreno_soec_o2_kg_h],
    }
    
    df_display = pd.DataFrame(data).set_index('Fluxo')
    
    # FUNÇÃO DE FORMATAÇÃO: Aplica notação científica para garantir que valores pequenos não sejam zero
    def formatar_vazao_cientifica(x):
        # GARANTE QUE X É UM FLOAT antes de chamar abs() para evitar o erro 'str'
        try:
             x_float = float(x)
        except ValueError:
             return x # Retorna o valor original se não for conversível (embora não deva acontecer aqui)
             
        # Usamos 5 casas decimais (f) para valores maiores que 0.0001 (1e-4)
        if abs(x_float) > 1e-4:
             return f'{x_float:.5f}'
        else: 
            # Notação científica (3e) para os fluxos minúsculos.
            return f'{x_float:.3e}'
            
    # Aplica a formatação em todas as colunas de dados de uma só vez.
    df_display['Vazão Total de Água que Segue (kg/h)'] = df_display['Vazão Total de Água que Segue (kg/h)'].apply(formatar_vazao_cientifica)
    df_display['Vapor Saturado (kg/h)'] = df_display['Vazão Total de Água que Segue (kg/h)'].apply(formatar_vazao_cientifica)
    df_display['Líquido Acomp. que Segue (kg/h)'] = df_display['Líquido Acomp. que Segue (kg/h)'].apply(formatar_vazao_cientifica)
    df_display['Dreno SOEC Removido (kg/h)'] = df_display['Dreno SOEC Removido (kg/h)'].apply(formatar_vazao_cientifica)


    print("\n" + "="*140)
    print("RESUMO DO BALANÇO DE ÁGUA NA SAÍDA DO SOEC / ENTRADA DA PURIFICAÇÃO (Vazões Mássicas em kg/h)")
    print("---------------------------------------------------------------------------------------------")
    print(f"Água Consumida Estequiometricamente: {M_DOT_H2O_CONSUMIDA_KGS * 3600:.3f} kg/h")
    print(f"Água Total Não Consumida: {M_DOT_H2O_RECIRC_TOTAL_KGS * 3600 - M_DOT_H2O_CONSUMIDA_KGS * 3600:.3f} kg/h")
    print(f"Crossover H₂ (M_H2O_TOTAL_H2_KGS): {Total_H2_Entrada_kg_h:.3f} kg/h")
    print(f"Água para Fluxo O₂ (M_H2O_TOTAL_O2_KGS): {Total_O2_Entrada_kg_h:.3f} kg/h")
    print("="*140)
    
    # Imprime a tabela com alinhamento melhorado
    print(df_display.to_markdown(colalign=['right', 'right', 'right', 'right', 'right'])) # Ajustado para 5 colunas
        
    print(f"\nVerificação Balanço H₂: {x_vapor_kg_h + z_liq_accomp_kg_h + dreno_soec_h2_kg_h:.3f} kg/h (Total na Saída do SOEC: {Total_H2_Entrada_kg_h:.3f} kg/h)")
    print(f"Verificação Balanço O₂: {y_vapor_kg_h + w_liq_accomp_kg_h + dreno_soec_o2_kg_h:.3f} kg/h (Total na Saída do SOEC: {Total_O2_Entrada_kg_h:.3f} kg/h)")
    print("="*140)
