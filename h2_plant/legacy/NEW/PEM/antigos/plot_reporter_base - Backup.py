# plot_reporter_base.py
# Funções de Plotagem e Exibição de Resultados da Simulação (Base).

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import CoolProp.CoolProp as CP
import sys

# Importa as constantes globais dos módulos dedicados
from constants_and_config import (
    LIMITES, Y_H2O_LIMIT_MOLAR, P_IN_BAR, T_IN_C, M_DOT_G_H2, M_DOT_G_O2,
    T_JACKET_DEOXO_C, P_VSA_PROD_BAR, P_VSA_REG_BAR, COMPONENTS_H2, COMPONENTS_O2,
    T_COLD_IN_WGHE_C # Adicionado para a nova função de comparação
)

# Importa os modelos de componente necessários para o re-cálculo de dados nos plots
try:
    from modelo_vsa_EDO import modelo_vsa_dimensionamento_parcial 
    # REMOVIDO: TSA
    
    # Função dummy para Válvula (necessária para simular o efeito Joule-Thomson para plots)
    def modelo_valvula_isoentalpica(fluido, T_in_K, P_in_Pa, P_out_Pa):
        if fluido == 'hydrogen':
            T_out_K = T_in_K - 0.5 
        else:
            T_out_K = T_in_K
        return {'SAIDA': {'T_K': T_out_K, 'P_Pa': P_out_Pa}}
        
except ImportError as e:
    print(f"AVISO: Modelos de componente para plots comparativos não encontrados: {e}")
    # Cria stubs apenas para funções usadas no re-cálculo de plots
    def modelo_vsa_dimensionamento_parcial(*args, **kwargs): return {'dimensionamento_parcial': {}, 'consumo_energetico': {}}

# =================================================================
# === FUNÇÕES AUXILIARES DE PLOTAGEM/CÁLCULO ===
# =================================================================

def adicionar_entalpia_pura(df: pd.DataFrame, gas_fluido: str) -> pd.Series:
    """Calcula a entalpia mássica do componente puro (H2 ou O2) nas condições T e P do fluxo."""
    
    fluido = 'H2' if gas_fluido == 'H2' else 'O2'
        
    H_pure_list = []
    for index, row in df.iterrows():
        T_K = row['T_C'] + 273.15
        P_Pa = row['P_bar'] * 1e5
        
        try:
            # Calcula a entalpia massica (J/kg) do componente puro nas condições do fluxo
            H_pure = CP.PropsSI('H', 'T', T_K, 'P', P_Pa, fluido)
            H_pure_list.append(H_pure)
        except Exception:
            # Em caso de erro, usa a entalpia da mistura como fallback
            H_pure_list.append(row['H_mix_J_kg'])
            
    return pd.Series(H_pure_list)

def calcular_entalpia_total_fluxo(df: pd.DataFrame, gas_fluido: str) -> pd.Series:
    """
    Calcula a entalpia mássica total (gás/vapor + líquido acompanhante) do fluxo.
    Assume entalpia específica do líquido a 0°C (273.15 K) como zero.
    """
    H_total_list = []
    C_p_liq = 4186.0 # J/kg.K (Cp da água líquida)
    T_ref_K = 273.15 # 0 C (Entalpia zero para água líquida)
    
    for _, row in df.iterrows():
        m_dot_mix = row['m_dot_mix_kg_s']
        m_dot_liq_accomp = row['m_dot_H2O_liq_accomp_kg_s']
        
        # H_mix_J_kg já inclui Gás Principal + Vapor H2O
        H_mix_gas_vap = row['H_mix_J_kg']
        
        # Entalpia do líquido acompanhante: h_liq ~ Cp * (T - T_ref)
        T_K = row['T_C'] + 273.15
        H_liq_J_kg = C_p_liq * (T_K - T_ref_K)
        
        # Vazão mássica total (Gás/Vapor + Líquido)
        m_dot_total = m_dot_mix + m_dot_liq_accomp
        
        # Entalpia TOTAL (J/s)
        H_dot_total_J_s = (H_mix_gas_vap * m_dot_mix) + (H_liq_J_kg * m_dot_liq_accomp)
        
        # Entalpia MÁSSICA TOTAL (J/kg_total)
        if m_dot_total > 0:
            H_total_massica = H_dot_total_J_s / m_dot_total
        else:
            H_total_massica = 0.0
            
        H_total_list.append(H_total_massica)
        
    return pd.Series(H_total_list)


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
    df_display[imp_name] = df_display[imp_col].map('{:.2e}'.format)
    df_display[imp_name] = df_display[imp_name] + ' (' + (df[imp_col] * 1e6).map('{:.2f}'.format) + ')'
    df_display.drop(columns=[imp_col], inplace=True)


    df_display.rename(columns={
        'T_C': 'T (°C)',
        'P_bar': 'P (bar)',
        'm_dot_gas_kg_s': 'm_dot Gás Princ. (kg/s)',
        'm_dot_mix_kg_s': 'm_dot Mistura (kg/s)',
        'y_H2O': 'y_H2O (molar)',
        'm_dot_H2O_liq_accomp_kg_h': 'm_dot H₂O Líq. Acomp. (kg/h)', # RENAME
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
    agua_removida_total = df['Agua_Condensada_kg_s'].sum()
    print(f"Água Líquida (Condensado + Acompanhante) Removida TOTAL: {agua_removida_total * 3600:.2f} kg/h") # CONVERSÃO
    
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
        
        # Re-chama o modelo VSA
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
        # AQUI AVISAMOS, MAS AS VARIÁVEIS JÁ ESTÃO DEFINIDAS COMO NONE
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

# =================================================================
# === FUNÇÕES DE PLOTAGEM (BASE) ===
# =================================================================

def plot_comparacao_wghe_lado_frio(df_h2_fase1: pd.DataFrame, df_h2_fase2: pd.DataFrame, df_o2_fase1: pd.DataFrame, df_o2_fase2: pd.DataFrame):
    """
    Gera gráfico comparativo da temperatura e vazão mássica do fluido de resfriamento 
    no WGHE, exibindo apenas o resultado da Fase 2 (Real).
    """
    
    # --- Função Auxiliar de Extração ---
    def extract_wghe_cold_data(df, gas_fluido, phase_label):
        # Filtra apenas a Fase 2 (Real)
        if phase_label not in ['Fase 2 (Real)', 'Fase 2']:
            return None
            
        # Garante que a linha 'WGHE 1' exista antes de tentar extrair
        if 'WGHE 1' not in df['Componente'].values:
             return None
             
        df_wghe = df[df['Componente'] == 'WGHE 1'].iloc[0]
        
        T_cold_in = T_COLD_IN_WGHE_C
        T_cold_out = df_wghe['T_cold_out_C']
        
        Delta_T_C = T_cold_out - T_cold_in
        m_dot_cold_kg_h = df_wghe['m_dot_cold_liq_kg_s'] * 3600
        
        # Para a Fase 2, Delta_T_bar = Delta_T_C
        Delta_T_bar = Delta_T_C
        
        return {
            'Gás': gas_fluido,
            'Fase': 'Fase 2', # Rótulo simplificado
            'T_cold_in_C': T_cold_in,
            'T_cold_out_C': T_cold_out,
            'Delta_T_C': Delta_T_C, 
            'Delta_T_bar': Delta_T_bar, 
            'm_dot_total_reuso_kg_h': m_dot_cold_kg_h,
        }

    # --- Coleta de Dados ---
    data = []
    
    # Processa apenas o dataframe da Fase 2
    h2_f2 = extract_wghe_cold_data(df_h2_fase2, 'H₂', 'Fase 2')
    o2_f2 = extract_wghe_cold_data(df_o2_fase2, 'O₂', 'Fase 2')
    
    if h2_f2: data.append(h2_f2)
    if o2_f2: data.append(o2_f2)

    if not data:
        print("\n--- Aviso: WGHE não está nos dataframes da Fase 2 ou dados insuficientes para plotagem. ---")
        return
        
    df_plot = pd.DataFrame(data)
    
    # --- Plotagem ---
    # Título ajustado: Remove todo o conteúdo extra
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    fig.suptitle(f'Desempenho do WGHE (Lado Frio - Fluido de Resfriamento)', fontsize=14)
    
    bar_width = 0.35
    
    # 1. Subplot H2
    ax1 = axes[0]
    h2_plot = df_plot[df_plot['Gás'] == 'H₂'].reset_index(drop=True)
    r1 = np.arange(len(h2_plot))
    
    # Eixo Y1: Temperatura (Delta T em barra)
    color_temp = 'tab:red'
    ax1.set_ylabel('T (°C)', color=color_temp)
    ax1.tick_params(axis='y', labelcolor=color_temp)
    
    # Define os limites do eixo Y1 (Temperatura)
    y_max_h2 = h2_plot['T_cold_out_C'].max() if not h2_plot.empty else T_COLD_IN_WGHE_C
    ax1.set_ylim(T_COLD_IN_WGHE_C - 1, y_max_h2 * 1.1 + 1)
    
    # Plota o Delta T como barra
    bottoms_h2 = h2_plot['T_cold_in_C'].apply(lambda x: x if x > 0 else 0) # Base da barra
    bars_t_h2 = ax1.bar(r1, h2_plot['Delta_T_bar'], bar_width, bottom=bottoms_h2, label='Delta T (°C) (Barra)', color='skyblue')
    
    # Linha de referência T_cold_in
    ax1.axhline(T_COLD_IN_WGHE_C, color='gray', linestyle='--', linewidth=1.0, label=f'T_cold_in = {T_COLD_IN_WGHE_C:.1f} °C')
    
    
    # Eixo Y2: Vazão Mássica de Reuso
    ax2 = ax1.twinx()
    color_mdot = 'tab:blue'
    ax2.set_ylabel('Vazão Mássica de Reuso (kg/h)', color=color_mdot)
    ax2.tick_params(axis='y', labelcolor=color_mdot)
    
    # Ajuste de escala do eixo Y2
    m_dot_max_h2 = h2_plot['m_dot_total_reuso_kg_h'].max()
    if m_dot_max_h2 > 0:
        y_max_h2_mdot = m_dot_max_h2 * 1.5 
    else:
        y_max_h2_mdot = 100 
        
    ax2.set_ylim(0, y_max_h2_mdot) # Escala ajustada
        
    # Plotagem da vazão mássica TOTAL
    ax2.plot(r1, h2_plot['m_dot_total_reuso_kg_h'], marker='o', linestyle='-', color=color_mdot, label='m_dot Reuso TOTAL (kg/h)')
    
    ax1.set_title('Fluxo de H₂')
    ax1.set_xticks(r1)
    ax1.set_xticklabels(h2_plot['Fase']) # Rótulos sem (Ideal) ou (Real)
    ax1.grid(axis='y', linestyle='--')
    
    # Rótulos da Barra de Temperatura
    for i, row in h2_plot.iterrows():
        # Rótulo T_in (na base)
        ax1.text(r1[i], row['T_cold_in_C'] * 1.05 if row['T_cold_in_C'] > 0 else 0.5, 
                 f'T_in: {row["T_cold_in_C"]:.2f}°C', 
                 ha='center', va='bottom', fontsize=8, color='black', fontweight='bold')
                 
        # Posição do rótulo T_out
        y_out_pos = row['T_cold_in_C'] + row['Delta_T_bar']
        
        # Rótulo T_out (no topo)
        ax1.text(r1[i], y_out_pos * 1.02, 
                 f'T_out: {row["T_cold_out_C"]:.2f}°C', 
                 ha='center', va='bottom', fontsize=8, color='black', fontweight='bold')
                 
        # Rótulo Delta T (no meio da barra)
        pos_y_delta_t = row['T_cold_in_C'] + row['Delta_T_bar'] / 2
        ax1.text(r1[i], pos_y_delta_t, 
                 f'ΔT: +{row["Delta_T_C"]:.2f}°C', 
                 ha='center', va='center', fontsize=7, color='tab:red', fontweight='bold')
        
        # Rótulo da Vazão Mássica (no eixo 2)
        if row['m_dot_total_reuso_kg_h'] > 1e-3:
            ax2.text(r1[i] + 0.1, row['m_dot_total_reuso_kg_h'] * 1.05, 
                     f'{row["m_dot_total_reuso_kg_h"]:.2f} kg/h', 
                     ha='center', va='bottom', fontsize=8, color='tab:blue')
    
    # 2. Subplot O2
    ax3 = axes[1]
    o2_plot = df_plot[df_plot['Gás'] == 'O₂'].reset_index(drop=True)
    r2 = np.arange(len(o2_plot))
    
    # Eixo Y3: Temperatura (Delta T em barra)
    ax3.set_ylabel('T (°C)', color=color_temp)
    ax3.tick_params(axis='y', labelcolor=color_temp)
    
    # Define os limites do eixo Y3 (Temperatura)
    y_max_o2 = o2_plot['T_cold_out_C'].max() if not o2_plot.empty else T_COLD_IN_WGHE_C
    ax3.set_ylim(T_COLD_IN_WGHE_C - 1, y_max_o2 * 1.1 + 1)
    
    # Plota o Delta T como barra
    bottoms_o2 = o2_plot['T_cold_in_C'].apply(lambda x: x if x > 0 else 0)
    bars_t_o2 = ax3.bar(r2, o2_plot['Delta_T_bar'], bar_width, bottom=bottoms_o2, label='Delta T (°C) (Barra)', color='salmon')
    
    ax3.axhline(T_COLD_IN_WGHE_C, color='gray', linestyle='--', linewidth=1.0, label=f'T_cold_in = {T_COLD_IN_WGHE_C:.1f} °C')
    
    
    # Eixo Y4: Vazão Mássica de Reuso
    ax4 = ax3.twinx()
    ax4.set_ylabel('Vazão Mássica de Reuso (kg/h)', color='tab:blue')
    ax4.tick_params(axis='y', labelcolor='tab:blue')
    
    # Ajuste de escala do eixo Y4
    m_dot_max_o2 = o2_plot['m_dot_total_reuso_kg_h'].max()
    if m_dot_max_o2 > 0:
        y_max_o2_mdot = m_dot_max_o2 * 1.5
    else:
        y_max_o2_mdot = 100 
        
    ax4.set_ylim(0, y_max_o2_mdot) # Escala ajustada
    
    # Plotagem da vazão mássica TOTAL
    ax4.plot(r2, o2_plot['m_dot_total_reuso_kg_h'], marker='o', linestyle='-', color='tab:blue', label='m_dot Reuso TOTAL (kg/h)')
    
    ax3.set_title('Fluxo de O₂')
    ax3.set_xticks(r2)
    ax3.set_xticklabels(o2_plot['Fase']) # Rótulos sem (Ideal) ou (Real)
    ax3.set_xlabel('Fase da Simulação')
    ax3.grid(axis='y', linestyle='--')
    
    # Rótulos da Barra de Temperatura
    for i, row in o2_plot.iterrows():
        # Rótulo T_in (na base)
        ax3.text(r2[i], row['T_cold_in_C'] * 1.05 if row['T_cold_in_C'] > 0 else 0.5, 
                 f'T_in: {row["T_cold_in_C"]:.2f}°C', 
                 ha='center', va='bottom', fontsize=8, color='black', fontweight='bold')
                 
        # Posição do rótulo T_out
        y_out_pos = row['T_cold_in_C'] + row['Delta_T_bar']
                 
        # Rótulo T_out (no topo)
        ax3.text(r2[i], y_out_pos * 1.02, 
                 f'T_out: {row["T_cold_out_C"]:.2f}°C', 
                 ha='center', va='bottom', fontsize=8, color='black', fontweight='bold')
                 
        # Rótulo Delta T (no meio da barra)
        pos_y_delta_t = row['T_cold_in_C'] + row['Delta_T_bar'] / 2
        ax3.text(r2[i], pos_y_delta_t, 
                 f'ΔT: +{row["Delta_T_C"]:.2f}°C', 
                 ha='center', va='center', fontsize=7, color='tab:red', fontweight='bold')

        # Rótulo da Vazão Mássica (no eixo 4)
        if row['m_dot_total_reuso_kg_h'] > 1e-3:
            ax4.text(r2[i] + 0.1, row['m_dot_total_reuso_kg_h'] * 1.05, 
                     f'{row["m_dot_total_reuso_kg_h"]:.2f} kg/h', 
                     ha='center', va='bottom', fontsize=8, color='tab:blue')
            
    # Combina legendas (Apenas no primeiro subplot)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    
    unique_labels = {}
    for h, l in zip(h1 + h2, l1 + l2):
        if l not in unique_labels:
            unique_labels[l] = h
            
    fig.legend(unique_labels.values(), unique_labels.keys(), loc='upper left', bbox_to_anchor=(0.08, 0.95), ncol=4)

    plt.tight_layout(rect=[0, 0, 1.0, 0.90])
    plt.show()

# =================================================================
# === FUNÇÕES DE PLOTAGEM DE RASTREAMENTO (Base) ===
# =================================================================

def plot_propriedades_empilhadas(df: pd.DataFrame, gas_fluido: str, deoxo_mode: str, L_deoxo: float, dc2_mode: str):
    """Gera gráficos empilhados de T, P, w_H2O, H vs. Componente para um único fluxo."""
    
    df_plot = df.copy()

    x_labels = df_plot['Componente']
    fig, axes = plt.subplots(4, 1, figsize=(10, 15), sharex=True)
    fig.suptitle(f'Evolução das Propriedades do Fluxo de {gas_fluido}', y=1.0) 
    
    
    # 1. Temperatura
    axes[0].plot(x_labels, df_plot['T_C'], marker='o', label=f'{gas_fluido} - Temperatura (°C)', color='blue')
    axes[0].set_ylabel('T (°C)')
    axes[0].grid(True, linestyle='--')
    axes[0].legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    for x, y in zip(range(len(x_labels)), df_plot['T_C']):
        axes[0].text(x, y + 0.05 * df_plot['T_C'].max(), f'{y:.2f}', ha='center', va='bottom', fontsize=8, color='blue')


    # 2. Pressão
    axes[1].plot(x_labels, df_plot['P_bar'], marker='o', label=f'{gas_fluido} - Pressão (bar)', color='red')
    axes[1].set_ylabel('P (bar)')
    axes[1].grid(True, linestyle='--')
    for x, y in zip(range(len(x_labels)), df_plot['P_bar']):
         axes[1].text(x, y + 0.005 * df_plot['P_bar'].max(), f'{y:.2f}', ha='center', va='bottom', fontsize=8, color='red')
    axes[1].legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))

    # 3. Fração Mássica de H2O (w_H2O)
    axes[2].plot(x_labels, df_plot['w_H2O'] * 100, marker='o', label=f'{gas_fluido} - w_H₂O Mássica (%)', color='green')
    axes[2].set_ylabel('w_H₂O Mássica (%)')
    axes[2].grid(True, linestyle='--')
    for x, y in zip(range(len(x_labels)), df_plot['w_H2O'] * 100):
         if y > 0.0001: 
             axes[2].text(x, y * 1.05 + 0.005, f'{y:.4f}', ha='center', va='bottom', fontsize=8, color='green')
    axes[2].legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))

    # 4. Entalpia Mássica (Com Entalpia Pura)
    df_plot['H_pure_J_kg'] = adicionar_entalpia_pura(df_plot, gas_fluido)
    df_plot['H_total_fluxo_J_kg'] = calcular_entalpia_total_fluxo(df_plot, gas_fluido)
    
    axes[3].plot(x_labels, df_plot['H_mix_J_kg'] / 1000, marker='o', 
                 label=f'H_mix (Gás+Vapor) (kJ/kg)', color='purple')
    axes[3].plot(x_labels, df_plot['H_total_fluxo_J_kg'] / 1000, marker='x', linestyle='-', 
                 label=f'H_TOTAL (Gás+Vapor+Líq) (kJ/kg)', color='red')
        
    axes[3].set_ylabel('H (kJ/kg)')
    axes[3].set_xlabel('Componente')
    axes[3].grid(True, linestyle='--')
    axes[3].tick_params(axis='x', rotation=15)
    
    # Rótulos para H_TOTAL (novo ponto de referência)
    for x, y in zip(range(len(x_labels)), df_plot['H_total_fluxo_J_kg'] / 1000):
         axes[3].text(x, y * 1.02, f'{y:.2f}', ha='center', va='bottom', fontsize=8, color='red')
         
    axes[3].legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))

    plt.tight_layout(rect=[0, 0, 0.88, 0.95]) 
    plt.show()

def plot_impurezas_crossover(df_h2: pd.DataFrame, df_o2: pd.DataFrame, deoxo_mode: str, L_deoxo: float, dc2_mode: str):
    """Gera gráfico das impurezas de crossover (O2 no H2 e H2 no O2) em subplots."""
    
    df_h2_plot = df_h2.copy()
        
    x_labels_h2 = df_h2_plot['Componente']
    x_labels_o2 = df_o2['Componente']
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False) 
    fig.suptitle(f'Evolução das Impurezas de Crossover (Escala Logarítmica)', y=1.0)
    
    # --- Subplot 1: Fluxo H2 (Contaminante O2) ---
    ax1 = axes[0]
    y_O2_profile_ppm = df_h2_plot['y_O2'].copy() * 1e6
    
    # Lógica de correção visual (mantida)
    if 'Deoxo' in x_labels_h2.values:
         idx_deoxo = x_labels_h2[x_labels_h2 == 'Deoxo'].index[0]
         y_O2_depois_deoxo = df_h2_plot.loc[idx_deoxo, 'y_O2'] * 1e6
         y_O2_profile_ppm.loc[idx_deoxo:] = y_O2_depois_deoxo

    # Adiciona linha de limite (5 ppm)
    LIMITE_O2_DEOXO_PPM = 5.0
    ax1.axhline(LIMITE_O2_DEOXO_PPM, color='red', linestyle='--', linewidth=1.5, label=f'Limite Deoxo (5 ppm)')
        
    ax1.plot(x_labels_h2, y_O2_profile_ppm, marker='o', label='H2 - Impureza O₂ (ppm)', color='darkgreen')
    ax1.set_ylabel('y_O₂ (ppm molar)') 
    ax1.set_title(f'Impureza O₂ no Fluxo de H₂')
    ax1.set_yscale('log')
    ax1.grid(True, which="both", linestyle='--')
    ax1.tick_params(axis='x', rotation=15)
    
    for x, y in zip(range(len(x_labels_h2)), y_O2_profile_ppm):
         label = f'{y:.2e}' if y < 1.0 and y > 0 else f'{y:.2f}' if y > 0 else '0.00'
         if y > 0:
            ax1.text(x, y * 1.2, label, ha='center', va='bottom', fontsize=7, color='darkgreen')
    ax1.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    
    # --- Subplot 2: Fluxo O2 (Contaminante H2) ---
    ax2 = axes[1]
    y_H2_profile_ppm = df_o2['y_H2'].copy() * 1e6
    
    ax2.plot(x_labels_o2, y_H2_profile_ppm, marker='s', label='O2 - Impureza H₂ (ppm)', color='orange')
    ax2.set_ylabel('y_H₂ (ppm molar)') 
    ax2.set_title('Impureza H₂ no Fluxo de O₂')
    ax2.set_xlabel('Componente')
    # CORREÇÃO: Ajuste de escala logarítmica para visualização de valores mais altos
    ax2.set_yscale('log')
    y_min = y_H2_profile_ppm[y_H2_profile_ppm > 0].min()
    ax2.set_ylim(y_min * 0.9, y_H2_profile_ppm.max() * 1.5) 
        
    ax2.grid(True, which="both", linestyle='--')
    ax2.tick_params(axis='x', rotation=15)
    
    for x, y in zip(range(len(x_labels_o2)), y_H2_profile_ppm):
         label = f'{y:.2e}' if y < 1.0 and y > 0 else f'{y:.2f}' if y > 0 else '0.00'
         if y > 0:
            ax2.text(x, y * 1.2, label, ha='center', va='bottom', fontsize=7, color='orange')
    ax2.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))

    plt.tight_layout(rect=[0, 0, 0.95, 0.95])
    plt.show()


def plot_vazao_agua_separada(df: pd.DataFrame, gas_fluido: str, deoxo_mode: str, L_deoxo: float, dc2_mode: str):
    """Gera gráfico da vazão mássica de vapor de água e adiciona o limite em PPM molar."""
    
    df_plot = df.copy()
        
    fig, ax = plt.subplots(figsize=(10, 5))
    x_labels = df_plot['Componente']
    fig.suptitle(f'Rastreamento da Vazão de Vapor de Água ({gas_fluido}) (Rótulos em Concentração Molar PPM)', y=1.0)
    
    # 1. Vazão Mássica de H2O na Fase Gasosa (Linha)
    # CONVERSÃO: kg/s para kg/h
    y_data_kg_h = df_plot['m_dot_H2O_vap_kg_s'] * 3600
    line, = ax.plot(x_labels, y_data_kg_h, 
            marker='o', linestyle='-', label=f'Vazão Mássica de H₂O na Fase Gasosa (kg/h)',
            color='blue' if gas_fluido == 'H2' else 'red')

    # 2. Adiciona o percentual de água molar (y_H2O) em PPM como rótulo sobre a linha 
    y_h2o_ppm = df_plot['y_H2O'] * 1e6 
    
    for i, txt in enumerate(y_h2o_ppm):
        if txt > 1.0: 
             label = f'{txt:.2f}' 
        elif txt > 0.0:
            label = f'{txt:.2e}' 
        else:
             continue
             
        # Usa y_data_kg_h
        ax.text(x_labels.iloc[i], y_data_kg_h.iloc[i] + 0.05 * y_data_kg_h.max(), 
                     label, 
                     ha='center', va='bottom', fontsize=8, color=line.get_color())


    # 3. Adiciona o Limite Molar de 100 PPM (y_H2O_LIMIT_MOLAR)
    
    if gas_fluido == 'H2':
        comp_limite = 'VSA' 
        m_dot_gas_princ_entrada = M_DOT_G_H2 
    else:
        comp_limite = 'Secador Adsorvente'
        m_dot_gas_princ_entrada = M_DOT_G_O2

    M_H2O = CP.PropsSI('M', 'Water')
    M_GAS_PRINCIPAL = CP.PropsSI('M', gas_fluido)
    y_H2O_limite_molar = Y_H2O_LIMIT_MOLAR
    
    F_gas_molar_entrada = m_dot_gas_princ_entrada / M_GAS_PRINCIPAL
    
    F_H2O_molar_limite = F_gas_molar_entrada * (y_H2O_limite_molar / (1.0 - y_H2O_limite_molar))
    m_dot_h2o_limite_kg_h = F_H2O_molar_limite * M_H2O * 3600 # kg/h
    
    # Plota a linha de limite
    ax.axhline(m_dot_h2o_limite_kg_h, color='green', linestyle='--', 
               label=f'Limite de Saída {comp_limite} ({y_H2O_limite_molar*1e6:.0f} ppm molar)')
        
    ax.set_ylabel('Vazão Mássica de H₂O na Fase Gasosa (kg/h)')
    ax.set_xlabel('Componente')
    ax.grid(True, linestyle='--')
    
    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize='small')
    
    plt.tight_layout(rect=[0, 0, 0.88, 0.95])
    plt.show()


def plot_vazao_liquida_acompanhante(df: pd.DataFrame, gas_fluido: str, deoxo_mode: str, L_deoxo: float, dc2_mode: str):
    """
    Gera gráfico da vazão mássica de água líquida que acompanha o fluxo (Recirculação/Drag).
    """
    
    df_plot = df.copy()

    fig, ax1 = plt.subplots(figsize=(10, 5))
    x_labels = df_plot['Componente'] 
    fig.suptitle(f'Rastreamento da Vazão de H₂O Líquida Acompanhante ({gas_fluido})', y=1.0)

    # --- Eixo Principal (Ax1): Vazão Mássica de Água Líquida Acompanhante (kg/h) ---
    
    # Conversão: kg/s para kg/h
    m_dot_liq_plot = df_plot['m_dot_H2O_liq_accomp_kg_s'] * 3600 
    unidade_liq = 'kg/h'
    
    color_liq_accomp = 'darkgreen'
    line_liq, = ax1.plot(x_labels, m_dot_liq_plot, 
                         marker='s', linestyle='-', 
                         label=f'H₂O Líquida Acompanhante ({unidade_liq})', 
                         color=color_liq_accomp)
    
    ax1.set_ylabel(f'Vazão Mássica de H₂O Líquida Acompanhante ({unidade_liq})', color=color_liq_accomp)
    ax1.tick_params(axis='y', labelcolor=color_liq_accomp)
    
    # Rótulos para a linha da Água Líquida Acompanhante (Garante 0.00 para saídas nulas)
    for x, y in zip(range(len(x_labels)), m_dot_liq_plot):
         # Usa 2 casas decimais para valores muito pequenos/zero para clareza visual (VSA/PSA)
         label = f'{y:.2f}' if y < 0.1 else f'{y:.4f}'
         if y > 100:
             label = f'{y:.2f}'
             
         if y > -0.001: 
             ax1.text(x, y * 1.05, label, ha='center', va='bottom', fontsize=8, color=color_liq_accomp)


    ax1.set_xlabel('Componente')
    ax1.grid(True, linestyle='--')
    ax1.tick_params(axis='x', rotation=15)
    
    # Manter a legenda do líquido
    ax1.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    
    plt.tight_layout(rect=[0, 0, 0.88, 0.95])
    plt.show()


def plot_fluxos_energia(df_h2: pd.DataFrame, df_o2: pd.DataFrame, deoxo_mode: str, L_deoxo: float, dc2_mode: str):
    """
    Gera gráficos de Energia do Fluxo e Energia do Componente. 
    Mantido para mostrar a carga do fluxo e o consumo (W dot).
    """
    
    df_plot_h2 = df_h2[df_h2['Componente'] != 'Entrada'].copy()
    df_plot_o2 = df_o2[df_o2['Componente'] != 'Entrada'].copy()
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle('Fluxos de Energia e Consumo Total por Componente', y=1.0) 
    
    bar_width = 0.35
    r1_h2 = np.arange(len(df_plot_h2))
    r2_o2 = np.arange(len(df_plot_o2))
    
    comp_labels_h2 = df_plot_h2['Componente'].tolist()
    
    # 1. Energia que sai ou entra do FLUXO (Q_dot_fluxo_W)
    q_fluxo_h2 = df_plot_h2['Q_dot_fluxo_W'] / 1000 # kW
    q_fluxo_o2 = df_plot_o2['Q_dot_fluxo_W'] / 1000 # kW
    
    barras_q_h2 = axes[0].bar(r1_h2, q_fluxo_h2, color='blue', width=bar_width, edgecolor='grey', label='H2 - Calor Trocado (Fluxo)')
    barras_q_o2 = axes[0].bar(r2_o2 + bar_width, q_fluxo_o2, color='red', width=bar_width, edgecolor='grey', label='O2 - Calor Trocado (Fluxo)')
    
    axes[0].set_title('Carga Térmica (Q dot Trocado com o Fluxo) - Comparativo H2 vs O2 (Inclui todas as fases)')
    axes[0].set_ylabel('Potência Térmica (kW)')
    
    axes[0].set_xticks(np.arange(len(comp_labels_h2)) + bar_width / 2)
    axes[0].set_xticklabels(comp_labels_h2)
    
    axes[0].grid(axis='y', linestyle='--')
    axes[0].legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    
    # Rótulos Q
    for bar in barras_q_h2:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2, yval * 1.05 + 0.1, f"{yval:.2f}", ha='center', va='bottom', fontsize=7, color='blue')
    for bar in barras_q_o2:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2, yval * 1.05 + 0.1, f"{yval:.2f}", ha='center', va='bottom', fontsize=7, color='red')


    # 2. Entrada/Saída de Energia do COMPONENTE (W_dot_comp_W)
    w_comp_h2 = df_plot_h2['W_dot_comp_W'] / 1000 # kW
    w_comp_o2 = df_plot_o2['W_dot_comp_W'] / 1000 # kW
    
    barras_w_h2 = axes[1].bar(r1_h2, w_comp_h2, color='skyblue', width=bar_width, edgecolor='grey', label='H2 - Consumo Total (Elétrico)')
    barras_w_o2 = axes[1].bar(r2_o2 + bar_width, w_comp_o2, color='salmon', width=bar_width, edgecolor='grey', label='O2 - Consumo Total (Elétrico)')
    
    axes[1].set_title('Consumo Total de Potência por Componente (W dot Elétrico)')
    axes[1].set_ylabel('Potência (kW)')
    axes[1].set_xticks(np.arange(len(comp_labels_h2)) + bar_width / 2)
    axes[1].set_xticklabels(comp_labels_h2, rotation=15)
    axes[1].set_xlabel('Componente')
    axes[1].grid(axis='y', linestyle='--')
    axes[1].legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))

    # Rótulos W
    for bar in barras_w_h2:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2, yval * 1.05 + 0.01, f"{yval:.2f}", ha='center', va='bottom', fontsize=7, color='blue')
    for bar in barras_w_o2:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2, yval * 1.05 + 0.01, f"{yval:.2f}", ha='center', va='bottom', fontsize=7, color='red')


    plt.tight_layout(rect=[0, 0, 0.95, 0.95])
    plt.show()

def plot_q_breakdown(df_h2: pd.DataFrame, df_o2: pd.DataFrame, deoxo_mode: str, L_deoxo: float, dc2_mode: str):
    """
    Gera gráfico da quebra de energia térmica (Q_dot) por fase: Total, Gás e Líquido/Latente.
    """
    
    df_plot_h2 = df_h2[df_h2['Componente'] != 'Entrada'].copy()
    df_plot_o2 = df_o2[df_o2['Componente'] != 'Entrada'].copy()

    # O filtro .get() garantirá que as colunas existam, ou usará 0.0
    df_plot_h2['Q_dot_Gas'] = df_plot_h2.get('Q_dot_H2_Gas', 0.0) 
    df_plot_h2['Q_dot_H2O_Total'] = df_plot_h2.get('Q_dot_H2O_Total', 0.0)
    
    df_plot_o2['Q_dot_Gas'] = df_plot_o2.get('Q_dot_H2_Gas', 0.0) 
    df_plot_o2['Q_dot_H2O_Total'] = df_plot_o2.get('Q_dot_H2O_Total', 0.0) 

    # Converte Q_dot para kW
    df_plot_h2['Q_dot_Gas_kW'] = df_plot_h2['Q_dot_Gas'] / 1000
    df_plot_h2['Q_dot_H2O_Total_kW'] = df_plot_h2['Q_dot_H2O_Total'] / 1000
    df_plot_o2['Q_dot_Gas_kW'] = df_plot_o2['Q_dot_Gas'] / 1000
    df_plot_o2['Q_dot_H2O_Total_kW'] = df_plot_o2['Q_dot_H2O_Total'] / 1000
    
    
    # 2. DEFINIÇÃO DOS EIXOS X 
    comp_labels_h2 = df_plot_h2['Componente'].tolist()
    r_h2 = np.arange(len(comp_labels_h2))
    
    comp_labels_o2 = df_plot_o2['Componente'].tolist()
    r_o2 = np.arange(len(comp_labels_o2))
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=False) 
    fig.suptitle('Quebra de Carga Térmica (Q dot) por Componente e Fase', y=1.0)
    
    # --- Subplot 1: Fluxo H2
    ax1 = axes[0]
    
    # Plota a quebra de carga em barras empilhadas para H2
    barras_h2_agua = ax1.bar(r_h2, df_plot_h2['Q_dot_H2O_Total_kW'], color='skyblue', edgecolor='grey', label='H2O (Vapor + Líquido)')
    barras_h2_gas = ax1.bar(r_h2, df_plot_h2['Q_dot_Gas_kW'], bottom=df_plot_h2['Q_dot_H2O_Total_kW'], color='blue', edgecolor='grey', label='H2 (Gás Principal)')
    
    ax1.set_title('Fluxo H₂')
    ax1.set_ylabel('Carga Térmica Removida (Q dot) (kW)')
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.grid(axis='y', linestyle='--')
    ax1.set_xticks(r_h2)
    ax1.set_xticklabels(comp_labels_h2)
    ax1.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))

    # Adicionar rótulos para a soma total (Q_dot_fluxo_W)
    for i in range(len(r_h2)):
        total = df_plot_h2['Q_dot_Gas_kW'].iloc[i] + df_plot_h2['Q_dot_H2O_Total_kW'].iloc[i]
        if total != 0:
            ax1.text(r_h2[i], total * 1.05 if total > 0 else total * 0.95, f'{total:.2f}', ha='center', va='center', fontsize=7, color='black')

    # --- Subplot 2: Fluxo O2
    ax2 = axes[1]
    
    # Plota a quebra de carga em barras empilhadas para O2
    barras_o2_agua = ax2.bar(r_o2, df_plot_o2['Q_dot_H2O_Total_kW'], color='salmon', edgecolor='grey', label='H2O (Vapor + Líquido)')
    barras_o2_gas = ax2.bar(r_o2, df_plot_o2['Q_dot_Gas_kW'], bottom=df_plot_o2['Q_dot_H2O_Total_kW'], color='red', edgecolor='grey', label='O2 (Gás Principal)')
    
    ax2.set_title('Fluxo O₂')
    ax2.set_ylabel('Carga Térmica Removida (Q dot) (kW)')
    ax2.set_xlabel('Componente')
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.grid(axis='y', linestyle='--')
    ax2.set_xticks(r_o2)
    ax2.set_xticklabels(comp_labels_o2, rotation=15)
    ax2.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    
    # Adicionar rótulos para a soma total (Q_dot_fluxo_W)
    for i in range(len(r_o2)):
        total = df_plot_o2['Q_dot_Gas_kW'].iloc[i] + df_plot_o2['Q_dot_H2O_Total_kW'].iloc[i]
        if total != 0:
             ax2.text(r_o2[i], total * 1.05 if total > 0 else total * 0.95, f'{total:.2f}', ha='center', va='center', fontsize=7, color='black')

    plt.tight_layout(rect=[0, 0, 0.95, 0.95])
    plt.show()

def plot_aproveitamento_termico(df_h2: pd.DataFrame, df_o2: pd.DataFrame):
    """Placeholder: Gráfico de Aproveitamento Térmico desativado (TSA/Regeneração removidos)."""
    # MANTIDO: O aviso existe no seu código original e deve ser mantido na estrutura.
    print("--- Aviso: O gráfico de Aproveitamento Térmico foi desativado (TSA/Regeneração removidos). ---")
    pass

def plot_agua_removida_total(df_h2: pd.DataFrame, df_o2: pd.DataFrame, deoxo_mode: str, L_deoxo: float, dc2_mode: str):
    """Gera gráfico da quantidade total de água retirada por componente."""

    comp_remover = ['KOD 1', 'Coalescedor 1', 'Secador Adsorvente', 'VSA'] 
    
    # Inicializa as remoções: Conversão de kg/s para kg/h
    remocao_h2 = {comp: df_h2[df_h2['Componente'] == comp]['Agua_Condensada_kg_s'].iloc[0] * 3600 for comp in comp_remover if comp in df_h2['Componente'].values}
    remocao_o2 = {comp: df_o2[df_o2['Componente'] == comp]['Agua_Condensada_kg_s'].iloc[0] * 3600 for comp in comp_remover if comp in df_o2['Componente'].values}
    
    componentes_x = [c for c in comp_remover if c in df_h2['Componente'].values or c in df_o2['Componente'].values]
    
    dados_h2 = [remocao_h2.get(c, 0) for c in componentes_x]
    dados_o2 = [remocao_o2.get(c, 0) for c in componentes_x] 
    
    df_plot = pd.DataFrame({
        'H2': dados_h2,
        'O2': dados_o2
    }, index=componentes_x)
    
    componentes_x = df_plot.index.to_list()

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle(f'Vazão de Água Líquida Removida por Componente', y=1.0)
    
    bar_width = 0.35
    r1 = np.arange(len(componentes_x))
    r2 = [x + bar_width for x in r1]
    
    barras_h2 = ax.bar(r1, df_plot['H2'], color='blue', width=bar_width, edgecolor='black', label='Fluxo de H₂')
    barras_o2 = ax.bar(r2, df_plot['O2'], color='red', width=bar_width, edgecolor='black', label='Fluxo de O₂')

    ax.set_title(f'Vazão de Água Líquida Removida por Componente')
    ax.set_ylabel('Vazão Mássica de Água Líquida (kg/h)')
    ax.set_xticks([r + bar_width/2 for r in r1])
    ax.set_xticklabels(componentes_x)
    ax.grid(axis='y', linestyle='--')
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0))
    
    for bar in barras_h2:
        yval = bar.get_height()
        if yval > 0.000001:
             ax.text(bar.get_x() + bar.get_width()/2, yval + 0.000000001, f"{yval:.2f}", ha='center', va='bottom', fontsize=8) 
    
    for bar in barras_o2:
        yval = bar.get_height()
        if yval > 0.000001:
             ax.text(bar.get_x() + bar.get_width()/2, yval + 0.000000001, f"{yval:.2f}", ha='center', va='bottom', fontsize=8)

    plt.tight_layout(rect=[0, 0, 1.0, 0.98]) 
    plt.show()

def plot_deoxo_perfil(df_h2: pd.DataFrame, L_span: np.ndarray, T_profile_C: np.ndarray, X_O2: float, T_max_calc: float, deoxo_mode: str, L_deoxo: float):
    """
    Gera o gráfico do perfil de Temperatura e Conversão no reator Deoxo.
    """
    
    deoxo_df = df_h2[df_h2['Componente'] == 'Deoxo']

    if not deoxo_df.empty and L_span is not None and T_profile_C is not None and T_profile_C.size > 1:
        
        idx_deoxo = df_h2[df_h2['Componente'] == 'Deoxo'].index[0]
        T_in_C = df_h2.iloc[idx_deoxo - 1]['T_C'] if idx_deoxo > 0 else T_profile_C[0]

        X_profile = np.linspace(0, X_O2, len(L_span))
        T_jacket_C = T_JACKET_DEOXO_C

        fig, ax1 = plt.subplots(figsize=(10, 6))
        fig.suptitle(f'Perfil de Temperatura e Conversão no Reator Deoxo (H2)', y=1.0)

        # Perfil de Conversão
        ax1.set_xlabel(f'Comprimento do Reator (L) (m) - L_total: {L_deoxo:.3f} m')
        ax1.set_ylabel('Conversão de O₂ (X) - Curva Azul', color='tab:blue')
        line_conv, = ax1.plot(L_span, X_profile, label='Conversão de O₂ (X)', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.set_ylim(0, 1.1)
        
        step = max(1, len(L_span) // 10) 
        for i in range(0, len(L_span), step):
            if X_profile[i] > 0:
                 ax1.text(L_span[i], X_profile[i] * 1.05, f'{X_profile[i]:.4f}', ha='center', va='bottom', fontsize=8, color=line_conv.get_color())
        

        # Perfil de Temperatura
        ax2 = ax1.twinx()
        ax2.set_ylabel('Temperatura (T) (C) - Curva Vermelha', color='tab:red')
        line_temp, = ax2.plot(L_span, T_profile_C, label='Temperatura do Fluxo (T)', color='tab:red')
        
        step = max(1, len(L_span) // 10)
        for i in range(0, len(L_span), step):
            ax2.text(L_span[i], T_profile_C[i], f'{T_profile_C[i]:.2f}', ha='center', va='bottom', fontsize=8, color='tab:red')


        # Linhas de referência de temperatura
        line_t_in = ax2.axhline(T_in_C, color='k', linestyle=':', label=f'T_in real = {T_in_C:.2f} C')
        lines = [line_t_in]

        if deoxo_mode == 'JACKET':
            line_t_jacket = ax2.axhline(T_JACKET_DEOXO_C, color='g', linestyle='-.', label=f'T_jacket = {T_JACKET_DEOXO_C:.2f} C')
            lines.append(line_t_jacket)
        
        if T_max_calc is not None:
             line_t_max = ax2.axhline(T_max_calc, color='r', linestyle='--', label=f'T_max calculada = {T_max_calc:.2f} C')
             lines.append(line_t_max)

        ax2.tick_params(axis='y', labelcolor='tab:red')
        
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        
        fig.legend(h1 + h2 + lines, l1 + l2 + [l.get_label() for l in lines], loc='upper left', bbox_to_anchor=(1.05, 1.05))

        plt.tight_layout(rect=[0, 0, 1.0, 0.96])
        plt.show()
        return
            
    print(f"\n--- Aviso: O gráfico do Perfil Deoxo (MODO {deoxo_mode}, L: {L_deoxo:.3f} m) não pode ser gerado, pois o H2 não passou pelo Deoxo ou dados insuficientes. ---")


def plot_esquema_processo(component_list: list, gas_fluido: str):
    """Gera um esquema simples da ordem dos processos com fluxo de massa/energia, com mais espaço."""
    
    fig, ax = plt.subplots(figsize=(20, 6))
    
    espacamento_x = 2.5
    posicoes = {comp: (i * espacamento_x, 0) for i, comp in enumerate(component_list)}
    
    # 1. Desenha os retângulos dos componentes
    for comp, (x, y) in posicoes.items():
        if comp != 'Entrada':
            rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color='lightgray', alpha=0.6, ec='black', lw=1.5, zorder=1)
            ax.add_patch(rect)
            ax.text(x, y, comp, ha='center', va='center', fontsize=10, fontweight='bold', zorder=2)
        else:
            ax.text(x, y, comp, ha='center', va='center', fontsize=10, fontweight='bold', zorder=2)

    # 2. Desenha as setas (Fluxo de Massa)
    fluxo_y = 0.0
    for i in range(len(component_list) - 1):
        comp_i = component_list[i]
        comp_i1 = component_list[i+1]
        x_start = posicoes[comp_i][0] + 0.5
        x_end = posicoes[comp_i1][0] - 0.5
        
        ax.annotate('', xy=(x_end, fluxo_y), xytext=(x_start, fluxo_y),
                    arrowprops=dict(facecolor='blue', shrink=0.05, width=2, headwidth=10))
        ax.text((x_start + x_end) / 2, fluxo_y + 0.45, 'Fluxo de Gás e Vapor', ha='center', va='bottom', fontsize=7, color='blue') 
        
    # 3. Desenha as setas de Energia (Calor/Trabalho)
    
    energia_detalhes = {
        'WGHE 1': [('Q (Resfriamento)', -1.0, -0.6, 'red'), ('W (Bomba)', 1.0, 0.6, 'green')],
        'Chiller 1': [('Q (Resfriamento)', -1.0, -0.6, 'red'), ('W (Elétrico)', 1.0, 0.6, 'green')],
        # ALTERADO: A água líquida agora é o fluxo principal de líquido a ser removido
        'KOD 1': [('M_H2O (Líq/Recirculação)', -1.0, -0.6, 'brown'), ('W (Perda P)', 1.0, 0.6, 'green')],
        'Coalescedor 1': [('M_Liq (Residual/Aerossóis)', -1.0, -0.6, 'brown'), ('W (Perda P)', 1.0, 0.6, 'green')],
        'Deoxo': [('Q (Trocado Jaqueta)', -1.0, -0.6, 'red'), ('H₂O (Produto, adicionado ao Líq)', -0.7, -0.3, 'purple'), ('O₂ (Consumido)', 1.0, 0.6, 'orange')],
        'VSA': [('M_H2O (Vapor Removida)', -1.0, -0.6, 'brown'), ('W (Comp./Vácuo)', 1.0, 0.6, 'green'), ('H₂ (Perdido)', -0.7, 0.3, 'orange')],
        'Secador Adsorvente': [('M_H2O (Vapor Removida)', -1.0, -0.6, 'brown'), ('W (Purga/Comp)', 1.0, 0.6, 'green')],
        'Válvula': [('W (Perda P)', 1.0, 0.6, 'green')], 
    }
    
    for comp in component_list:
        if comp in energia_detalhes:
            for rotulo, y_arr, y_text, cor in energia_detalhes[comp]:
                x = posicoes[comp][0]
                
                if gas_fluido == 'O2' and comp == 'Deoxo':
                    continue
                
                offset = 0.5 if 'Líq' in rotulo or 'Produto' in rotulo or 'Calor' in rotulo or 'Reação' in rotulo else -0.5
                
                ax.annotate('', xy=(x + offset, y_arr), xytext=(x + offset, y_text), arrowprops=dict(facecolor=cor, shrink=0.05, width=1, headwidth=5), ha='center', zorder=3)
                ax.text(x + offset, y_text + 0.1, rotulo, ha='center', va='bottom', fontsize=7, color=cor)
        
    ax.set_xlim(-1, posicoes[component_list[-1]][0] + 1.5)
    ax.set_ylim(-1.5, 1.5) 
    ax.axis('off')
    plt.title(f'Esquema do Sistema de Tratamento de Gás ({gas_fluido})', fontsize=14)
    plt.show()

def plot_vazao_massica_total_e_removida(df: pd.DataFrame, gas_fluido: str, deoxo_mode: str, L_deoxo: float, dc2_mode: str):
    """
    Gera gráfico comparativo das vazões mássicas do Gás Principal, Vapor de H2O e Total.
    Tudo em kg/h com rótulos de valor.
    """
    
    df_plot = df.copy()
    
    # Converte tudo para kg/h
    df_plot['m_dot_gas_princ_kg_h'] = df_plot['m_dot_gas_kg_s'] * 3600
    df_plot['m_dot_H2O_vap_kg_h'] = df_plot['m_dot_H2O_vap_kg_s'] * 3600
    df_plot['m_dot_mix_kg_h'] = df_plot['m_dot_mix_kg_s'] * 3600

    x_labels = df_plot['Componente']
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(f'Rastreamento das Vazões Mássicas do Fluxo ({gas_fluido})', y=1.0)

    # Curva 1: Vazão Mássica Total da Mistura
    line_mix, = ax.plot(x_labels, df_plot['m_dot_mix_kg_h'], marker='o', linestyle='-', color='purple', label='Vazão Mássica TOTAL (Gás+Vapor) (kg/h)')
    
    # Curva 2: Vazão Mássica do Gás Principal
    line_gas, = ax.plot(x_labels, df_plot['m_dot_gas_princ_kg_h'], marker='s', linestyle='--', color='blue', label=f'Vazão Mássica {gas_fluido} Principal (kg/h)')

    # Curva 3: Vazão Mássica do Vapor de H2O
    line_vap, = ax.plot(x_labels, df_plot['m_dot_H2O_vap_kg_h'], marker='^', linestyle=':', color='red', label='Vazão Mássica H₂O Vapor (kg/h)')

    ax.set_ylabel('Vazão Mássica (kg/h)')
    ax.set_xlabel('Componente')
    ax.grid(True, linestyle='--')
    ax.tick_params(axis='x', rotation=15)

    # Adicionar Rótulos (apenas onde o valor não é zero, exceto se for o H2O vapor no final)
    
    # Rótulos da Mistura (TOTAL)
    for x, y in zip(range(len(x_labels)), df_plot['m_dot_mix_kg_h']):
        if y > 0.01:
            ax.text(x, y * 1.01, f'{y:.2f}', ha='center', va='bottom', fontsize=8, color='purple')
            
    # Rótulos do Gás Principal
    for x, y in zip(range(len(x_labels)), df_plot['m_dot_gas_princ_kg_h']):
        if y > 0.01:
            ax.text(x, y * 0.99, f'{y:.2f}', ha='center', va='top', fontsize=8, color='blue')

    # Rótulos do Vapor (focando nos valores de entrada e saída)
    for x, y in zip(range(len(x_labels)), df_plot['m_dot_H2O_vap_kg_h']):
        # Rótulos em estágios iniciais, no Deoxo e no estágio final (VSA/PSA)
        if (y > 0.01 and x < 3) or x >= len(x_labels) - 3:
            # Usa notação científica para valores muito pequenos no final
            label = f'{y:.2f}' if y > 0.1 else f'{y:.2e}' if y > 0 else '0.00'
            ax.text(x, y * 1.1 + 0.05, label, ha='center', va='bottom', fontsize=8, color='red')


    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout(rect=[0, 0, 0.88, 0.95])
    plt.show()