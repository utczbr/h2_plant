# plot_reporter_base.py
# Funções de Suporte (Auxiliares e Exibição de Terminal) - FUNÇÕES DE PLOTAGEM MOVIDAS PARA plots_modulos/

import numpy as np 
import pandas as pd 
import CoolProp.CoolProp as CP
import sys
import matplotlib.pyplot as plt # Importação necessária para o salvamento
import os # Necessário para manipular pastas

# Importa as constantes globais do módulo dedicado
from constants_and_config import (
    LIMITES, Y_H2O_LIMIT_MOLAR, P_IN_BAR, T_IN_C, M_DOT_G_H2, M_DOT_G_O2,
    T_JACKET_DEOXO_C, P_VSA_PROD_BAR, P_VSA_REG_BAR, COMPONENTS_H2, COMPONENTS_O2,
    M_DOT_H2O_RECIRC_TOTAL_KGS, M_DOT_H2O_CONSUMIDA_KGS, # Importa constantes globais para validação
    M_H2O_TOTAL_H2_KGS, M_H2O_TOTAL_O2_KGS # Adicionada importação das constantes de fluxo total
)

# 💥 CORREÇÃO: Importa os modelos de componente necessários para o re-cálculo de dados nos plots do novo caminho 'modulos' e nomes.
try:
    from modulos.modelo_vsa import modelo_vsa_dimensionamento_parcial 
    from modulos.modelo_valvula import modelo_valvula_isoentalpica # Importação do modelo real

except ImportError as e:
    # Fallback/Stub se os módulos não forem encontrados
    print(f"AVISO: Modelos de componente (VSA/Valvula) para plots comparativos não encontrados: {e}")
    # Cria stubs apenas para funções usadas no re-cálculo de plots
    def modelo_vsa_dimensionamento_parcial(*args, **kwargs): return {'dimensionamento_parcial': {}, 'consumo_energetico': {}}
    
    # Função dummy para Válvula (necessária para simular o efeito Joule-Thomson para plots)
    def modelo_valvula_isoentalpica(fluido, T_in_K, P_in_Pa, P_out_Pa):
        if fluido == 'hydrogen':
            T_out_K = T_in_K - 0.5 
        else:
            T_out_K = T_in_K
        return {'SAIDA': {'T_K': T_out_K, 'P_Pa': P_out_Pa}}


# =================================================================
# === FUNÇÃO CENTRAL DE SALVAMENTO E EXIBIÇÃO (NOVO) ===
# =================================================================

# CORREÇÃO: Atualizando o caminho raiz para o local que você indicou.
CAMINHO_RAIZ_GRAFICOS = r'C:\Users\tusaw\OneDrive\Documentos\projeto hidrogenio\PEM\plots_modulos\Graficos'

def salvar_e_exibir_plot(nome_arquivo: str, mostrar_grafico: bool = True):
    """
    Salva o gráfico no caminho absoluto predefinido e, opcionalmente, o exibe.
    Esta função deve ser chamada ao final de cada função de plotagem.
    """
    try:
        # 1. Cria a pasta 'Graficos' (se não existir)
        if not os.path.exists(CAMINHO_RAIZ_GRAFICOS):
            os.makedirs(CAMINHO_RAIZ_GRAFICOS, exist_ok=True)
            
        caminho_completo = os.path.join(CAMINHO_RAIZ_GRAFICOS, nome_arquivo)
        
        # 2. Salva o arquivo (SEMPRE)
        plt.savefig(caminho_completo)
        print(f"Gráfico '{nome_arquivo}' salvo com sucesso em: {caminho_completo}")
        
        # 3. Exibe (CONDICIONALMENTE)
        if mostrar_grafico:
            plt.show()
            
        plt.close() # Garante que o recurso seja liberado
        
    except Exception as e:
        print(f"❌ ERRO ao salvar ou exibir o gráfico '{nome_arquivo}': {e}")
        plt.close()

# =================================================================
# === FUNÇÕES AUXILIARES DE PLOTAGEM/CÁLCULO (MANTIDAS AQUI) ===
# =================================================================

# 💥 CORREÇÃO: Remoção dos comentários de REMOVIDA A FUNÇÃO: adicionar_entalpia_pura
# e REMOVIDA A FUNÇÃO: calcular_entalpia_total_fluxo
# (O Python pode, em raras configurações, tentar importar funções de módulos
# baseados em comentários se as funções stub não estiverem presentes.)


def calcular_vazao_massica_total_completa(df: pd.DataFrame) -> pd.Series:
    """
    Calcula a vazão mássica total real (Gás Principal + Vapor H2O + H2O Líquida Acompanhante).
    Utiliza as colunas m_dot_mix_kg_s (que é Gás Principal + Vapor H2O)
    e m_dot_H2O_liq_accomp_kg_s (Água Líquida Acompanhante).
    Retorna em kg/s.
    """
    # A coluna 'm_dot_mix_kg_s' já é a soma do Gás Principal e Vapor H2O.
    # Soma essa mistura com a água líquida acompanhante.
    m_dot_total_completa = df['m_dot_mix_kg_s'] + df['m_dot_H2O_liq_accomp_kg_s']
    return m_dot_total_completa


def log_runtime(start_time, end_time):
    """Calcula e imprime a duração total da execução do pipeline."""
    duration = end_time - start_time
    print(f"\nTempo de Execução Total: {duration}")

# =================================================================
# === FUNÇÕES DE EXIBIÇÃO DE TABELA E RESUMO (MANTIDAS AQUI) ===
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
    df_display[imp_name] = df_display[imp_col].map('{:.2e}'.format)
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


def exibir_estado_final(df: pd.DataFrame, gas_fluido: str, deoxo_mode: str, L_deoxo: float, dc2_mode: str):
    """Exibe o estado final do fluido."""
    estado_final = df[df['Componente'] == df['Componente'].iloc[-1]].iloc[0] 
    
    print("\n" + "="*80)
    print(f"ESTADO FINAL DO FLUIDO: {gas_fluido}")
    print(f"Modo Deoxo: {deoxo_mode} | L_Deoxo: {L_deoxo:.3f} m")
    if gas_fluido == 'H2':
        comp_final = estado_final['Componente']
        print(f"Processo Final H2: {comp_final}")
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
    
    # Soma de Água Líquida (Condensado + Acompanhante) Removida TOTAL
    # Nota: A coluna 'Agua_Condensada_kg_s' não é mais usada para rastrear remoção total.
    # Usamos a soma de 'Agua_Pura_Removida_H2O_kg_s'
    agua_removida_total = df['Agua_Pura_Removida_H2O_kg_s'].sum()
    print(f"Água Líquida (Drenos) Removida TOTAL: {agua_removida_total * 3600:.2f} kg/h") # CONVERSÃO
    
    if 'KOD 1' in df['Componente'].values:
        status_kod_1 = df[df['Componente'] == 'KOD 1']['Status_KOD'].iloc[0]
        print(f"Status do KOD 1: {status_kod_1}")

    print("="*80)

def exibir_resumo_vsa(df_h2: pd.DataFrame):
    """Exibe um resumo dos parâmetros de desempenho e dimensionamento do VSA."""
    vsa_data = df_h2[df_h2['Componente'] == 'VSA']
    if vsa_data.empty:
        print("\n--- VSA não encontrado no fluxo H2. ---")
        return
        
    vsa_data = df_h2[df_h2['Componente'] == 'VSA'].iloc[0]
    idx_vsa = df_h2[df_h2['Componente'] == 'VSA'].index[0]
    vsa_in = df_h2.iloc[idx_vsa - 1] 
    
    delta_p = vsa_in['P_bar'] - vsa_data['P_bar']
    
    print("\n" + "="*80)
    print("RESUMO DE DESEMPENHO E CUSTOS DO VSA (H2)")
    print("="*80)
    
    # Inicialização segura das variáveis 
    M_ads_total_kg, H2O_Removida_kg_h, P_total_kW, E_especifica_kwh_kg, H2_Perdido_kg_s = None, None, None, None, None
    
    try:
        T_K = vsa_in['T_C'] + 273.15
        P_Pa = vsa_in['P_bar'] * 1e5
        
        # Otimização da densidade (mantendo o fallback para robustez)
        try:
            rho_in = CP.PropsSI('D', 'T', T_K, 'P', P_Pa, 'H2')
        except:
             R_UNIV = 8.31446 
             F_molar_total = vsa_in['F_molar_total'] if vsa_in['F_molar_total'] > 0 else 1.0
             M_H2_MEDIO = vsa_in['m_dot_mix_kg_s'] / F_molar_total
             rho_in = P_Pa * M_H2_MEDIO / (R_UNIV * T_K)
             
        Vazao_m3_h = (vsa_in['m_dot_mix_kg_s'] / rho_in) * 3600
        
        # Re-chama o modelo VSA (modelo_vsa_dimensionamento_parcial deve ser importado corretamente)
        res_full = modelo_vsa_dimensionamento_parcial(
            T_entrada_C=vsa_in['T_C'],
            P_entrada_bar=vsa_in['P_bar'],
            vazao_m3_h=Vazao_m3_h,
            umidade_molar_entrada_ppm=vsa_in['y_H2O'] * 1e6,
            P_adsorcao_bar=vsa_in['P_bar'], 
            P_produto_bar=P_VSA_PROD_BAR,
            P_regeneracao_bar=P_VSA_REG_BAR,
            recuperacao_h2=0.90
        )
        
        # Extrai resultados com segurança
        if res_full and 'dimensionamento_parcial' in res_full:
             dim_res = res_full['dimensionamento_parcial']
             M_ads_total_kg = dim_res['massa_adsorvente_total_kg']
             H2O_Removida_kg_h = dim_res['vazao_h2o_removida_kg_h']
        
        if res_full and 'consumo_energetico' in res_full:
            cons_res = res_full['consumo_energetico']
            P_total_kW = cons_res['potencia_total_kW']
            E_especifica_kwh_kg = cons_res['energia_especifica_kwh_por_kg_h2']
            
        # Vazão de H2 Perdido (baseado em 90% de recuperação)
        H2_Perdido_kg_s = (vsa_in['m_dot_gas_kg_s'] * (1.0 - 0.90)) 

    except Exception as e:
        print(f"Aviso: Falha ao re-executar o modelo VSA para obter dados de dimensionamento/custo. Erro: {e}")
        
    
    print("--- PARÂMETROS DE DIMENSIONAMENTO E CUSTO ---")
    if M_ads_total_kg is not None:
        print(f"Massa Total de Adsorvente (M_ads): {M_ads_total_kg:.2f} kg")
        print(f"Vazão de H₂O a Remover: {H2O_Removida_kg_h:.2f} kg/h")
    else:
        print("Massa/Vazão de Adsorvente/H2O: Não disponível.")
        
    if H2_Perdido_kg_s is not None:
        print(f"Vazão de H₂ Perdido (Custo de Purga): {H2_Perdido_kg_s * 3600:.2f} kg/h (10% de perda)") # CONVERSÃO
        
    N_VASOS_ATUAL = 3
    
    print(f"\nConfiguração do Ciclo:")
    print(f"Nº de Leitos (Modelo): {N_VASOS_ATUAL}")
    print(f"Tempo de Ciclo (T_ciclo): 10.0 min")
    print(f"Recuperação de H₂ (Modelo): 90.00 %") 
        
    print("\n--- PARÂMETROS OPERACIONAIS ---")
    print(f"Pressão de Entrada (P_in): {vsa_in['P_bar']:.2f} bar")
    print(f"Pressão de Saída (P_out): {vsa_data['P_bar']:.2f} bar")
    print(f"Queda de Pressão (Delta P): {delta_p:.4f} bar")
    
    if P_total_kW is not None:
        print(f"Potência de Compressão/Vácuo (W dot): {P_total_kW:.4f} kW")
        print(f"Energia Específica: {E_especifica_kwh_kg:.3f} kWh/kg H₂")
    else:
         print("Potência/Energia Específica: Não disponível.")
         
    print(f"H₂O de Saída (Pureza Alvo): {vsa_data['w_H2O'] * 1e6:.6f} ppm")
    print("="*80)
# FIM DA FUNÇÃO exibir_resumo_vsa


# =================================================================
# === NOVO: EXIBIÇÃO DO ESTADO FINAL DO MIXER DE DRENOS ===
# =================================================================

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
    
# FIM DA FUNÇÃO exibir_estado_final_mixer

# =================================================================
# === NOVO: EXIBIÇÃO DO ESTADO FINAL DA RECIRCULAÇÃO (Pós-Reposição) ===
# =================================================================

def exibir_estado_recirculacao(resultado_recirculacao: dict):
    """
    Exibe uma tabela de resumo das propriedades termodinâmicas da
    corrente de água após a reposição (pronta para o PEM).
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
    
# FIM DA FUNÇÃO exibir_estado_recirculacao


# =================================================================
# === NOVO: FUNÇÃO DE VALIDAÇÃO DO BALANÇO DE MASSA GLOBAL ===
# =================================================================

def exibir_validacao_balanco_global(m_dot_drenada_total_kgs: float, m_dot_consumida_kgs: float):
    """
    Calcula e exibe a validação do balanço de massa global de água:
    Água Total Drenada (Pool) vs. Consumo Estequiométrico.
    
    Esta métrica serve como controle, independentemente das restrições de recirculação.
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
    
# FIM DA FUNÇÃO exibir_validacao_balanco_global


# =================================================================
# === NOVO: EXIBIÇÃO DO BALANÇO INICIAL DE ÁGUA (x, y, z, w) ===
# =================================================================

def exibir_balanco_agua_inicial(df_h2: pd.DataFrame, df_o2: pd.DataFrame):
    """
    Exibe uma tabela de resumo dos cálculos iniciais de balanço de água
    e partição Vapor/Líquido/Dreno PEM, rastreando as vazões.
    
    Variáveis:
    x/y: Vazão de água na forma de Vapor Saturado.
    z/w: Vazão de Água Líquida que efetivamente segue com o fluxo gasoso.
    """
    
    # Extrair a linha de 'Entrada'
    h2_in = df_h2[df_h2['Componente'] == 'Entrada'].iloc[0]
    o2_in = df_o2[df_o2['Componente'] == 'Entrada'].iloc[0]
    
    # --------------------------------------------------------------------------------
    # CÁLCULOS DOS VALORES (x, y, z, w) em kg/h
    # --------------------------------------------------------------------------------
    
    # H2 Side (x and z)
    x_vapor_kgs = h2_in.get('M_DOT_VAPOR_ENTRADA_KGS_X_Y', 0.0)
    # CORRIGIDO: Agora extraindo o valor CORRETO do m_dot_H2O_liq_accomp_kg_s (z)
    z_liq_accomp_kgs = h2_in.get('m_dot_H2O_liq_accomp_kg_s', 0.0) 
    
    # CORREÇÃO AQUI: Usar M_H2O_TOTAL_H2_KGS (que já está correto em constants_and_config)
    m_dot_h2o_total_h2_kgs = M_H2O_TOTAL_H2_KGS 
    
    # Água em Vapor (x): 
    x_vapor_kg_h = x_vapor_kgs * 3600 
    # Água Líquida Acompanhante (z): 
    z_liq_accomp_kg_h = z_liq_accomp_kgs * 3600
    # Água Removida no Dreno PEM:
    dreno_pem_h2_kg_h = h2_in.get('Agua_Pura_Removida_H2O_kg_s', 0.0) * 3600
    
    # O2 Side (y and w)
    y_vapor_kgs = o2_in.get('M_DOT_VAPOR_ENTRADA_KGS_X_Y', 0.0)
    # CORRIGIDO: Agora extraindo o valor CORRETO do m_dot_H2O_liq_accomp_kg_s (w)
    w_liq_accomp_kgs = o2_in.get('m_dot_H2O_liq_accomp_kg_s', 0.0) 
    
    # CORREÇÃO AQUI: Usar M_H2O_TOTAL_O2_KGS (que já está correto em constants_and_config)
    m_dot_h2o_total_o2_kgs = M_H2O_TOTAL_O2_KGS 
    
    # Água em Vapor (y): 
    y_vapor_kg_h = y_vapor_kgs * 3600
    # Água Líquida Acompanhante (w): 
    w_liq_accomp_kg_h = w_liq_accomp_kgs * 3600
    # Água Removida no Dreno PEM:
    dreno_pem_o2_kg_h = o2_in.get('Agua_Pura_Removida_H2O_kg_s', 0.0) * 3600
    
    # --------------------------------------------------------------------------------
    # CONSTRUÇÃO DA TABELA
    # --------------------------------------------------------------------------------
    
    # CÁLCULO DE VALIDAÇÃO (Massa Total)
    Total_H2_Entrada_kg_h = m_dot_h2o_total_h2_kgs * 3600
    Total_O2_Entrada_kg_h = m_dot_h2o_total_o2_kgs * 3600
    
    data = {
        'Fluxo': ['H₂', 'O₂'],
        # Mudar a ordem para que todas as colunas sejam float e convertidas juntas.
        'Vazão Total de Água de Entrada (kg/h)': [Total_H2_Entrada_kg_h, Total_O2_Entrada_kg_h],
        'Vapor Saturado (kg/h)': [x_vapor_kg_h, y_vapor_kg_h],
        'Líquido Acomp. que Segue (kg/h)': [z_liq_accomp_kg_h, w_liq_accomp_kg_h],
        'Dreno PEM Removido (kg/h)': [dreno_pem_h2_kg_h, dreno_pem_o2_kg_h],
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
    df_display['Vazão Total de Água de Entrada (kg/h)'] = df_display['Vazão Total de Água de Entrada (kg/h)'].apply(formatar_vazao_cientifica)
    df_display['Vapor Saturado (kg/h)'] = df_display['Vapor Saturado (kg/h)'].apply(formatar_vazao_cientifica)
    df_display['Líquido Acomp. que Segue (kg/h)'] = df_display['Líquido Acomp. que Segue (kg/h)'].apply(formatar_vazao_cientifica)
    df_display['Dreno PEM Removido (kg/h)'] = df_display['Dreno PEM Removido (kg/h)'].apply(formatar_vazao_cientifica)


    print("\n" + "="*140)
    print("RESUMO DO BALANÇO INICIAL DE ÁGUA (PARTIÇÃO PEM) - Vazões Mássicas em kg/h")
    print("--------------------------------------------------------------------------")
    print(f"Água Consumida Estequiometricamente: {M_DOT_H2O_CONSUMIDA_KGS * 3600:.3f} kg/h")
    print(f"Água Total Não Consumida: {M_DOT_H2O_RECIRC_TOTAL_KGS * 3600 - M_DOT_H2O_CONSUMIDA_KGS * 3600:.3f} kg/h")
    print(f"Crossover H₂ (5 x Consumo): {Total_H2_Entrada_kg_h:.3f} kg/h")
    print(f"Água para Fluxo O₂ (Restante): {Total_O2_Entrada_kg_h:.3f} kg/h")
    print("="*140)
    
    # Imprime a tabela com alinhamento melhorado
    print(df_display.to_markdown(colalign=['right', 'right', 'right', 'right', 'right'])) # Ajustado para 5 colunas
        
    print(f"\nVerificação Balanço H₂: {x_vapor_kg_h + z_liq_accomp_kg_h + dreno_pem_h2_kg_h:.3f} kg/h (Total de Entrada H₂: {Total_H2_Entrada_kg_h:.3f} kg/h)")
    print(f"Verificação Balanço O₂: {y_vapor_kg_h + w_liq_accomp_kg_h + dreno_pem_o2_kg_h:.3f} kg/h (Total de Entrada O₂: {Total_O2_Entrada_kg_h:.3f} kg/h)")
    print("="*140)