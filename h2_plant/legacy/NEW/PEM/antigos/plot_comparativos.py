# plot_comparativos.py
# Funções de Plotagem para Comparações de Estágios (C1: Dry Cooler vs WGHE)

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import sys

# Importa as constantes globais
from constants_and_config import (
    T_IN_C, T_COLD_IN_WGHE_C, 
    T_A_IN_OP # <--- CORREÇÃO: Constante adicionada aqui
)

# =================================================================
# === GRÁFICO 1: COMPARAÇÃO DE QUEDA DE TEMPERATURA C1 ===
# =================================================================

def plot_comparacao_temperatura_c1(df_h2_dc: pd.DataFrame, df_h2_wghe: pd.DataFrame, df_o2_dc: pd.DataFrame, df_o2_wghe: pd.DataFrame):
    """
    Gera gráfico comparativo da queda de temperatura no primeiro estágio (C1) 
    para Dry Cooler vs. WGHE (H2 e O2).
    """
    
    # --- Função Auxiliar para Extração Segura ---
    def safe_extract_t_out(df, component_name, default_t=T_IN_C):
        df_filtered = df[df['Componente'] == component_name]
        if not df_filtered.empty:
            return df_filtered['T_C'].iloc[0]
        return default_t

    # --- Extração de Dados (Com tratamento de erro) ---
    T_in_h2 = T_IN_C
    T_in_o2 = T_IN_C
    
    # H2
    T_out_h2_dc = safe_extract_t_out(df_h2_dc, 'Dry Cooler 1')
    T_out_h2_wghe = safe_extract_t_out(df_h2_wghe, 'WGHE 1')
    
    # O2
    T_out_o2_dc = safe_extract_t_out(df_o2_dc, 'Dry Cooler 1')
    T_out_o2_wghe = safe_extract_t_out(df_o2_wghe, 'WGHE 1')
    
    data = {
        'Gás': ['H₂', 'H₂', 'O₂', 'O₂'],
        'Sistema C1': ['Dry Cooler', 'WGHE', 'Dry Cooler', 'WGHE'],
        'T_Entrada (°C)': [T_in_h2, T_in_h2, T_in_o2, T_in_o2],
        'T_Saída (°C)': [T_out_h2_dc, T_out_h2_wghe, T_out_o2_dc, T_out_o2_wghe],
    }
    df_plot = pd.DataFrame(data)
    
    df_plot['Delta_T (°C)'] = df_plot['T_Entrada (°C)'] - df_plot['T_Saída (°C)']
    
    # --- Plotagem ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    fig.suptitle('Comparação da Queda de Temperatura no Primeiro Estágio de Resfriamento (C1)', fontsize=14)
    
    x = np.arange(2) * 1.5 
    width = 0.35
    
    # Subplot H2
    ax1 = axes[0]
    h2_plot = df_plot.loc[df_plot['Gás'] == 'H₂']
    h2_dc_delta = h2_plot[h2_plot['Sistema C1'] == 'Dry Cooler']['Delta_T (°C)'].iloc[0]
    h2_wghe_delta = h2_plot[h2_plot['Sistema C1'] == 'WGHE']['Delta_T (°C)'].iloc[0]

    bar1 = ax1.bar(width/2, h2_dc_delta, width, label='Dry Cooler (Ar)', color='tab:blue')
    bar2 = ax1.bar(width/2 + width + 0.1, h2_wghe_delta, width, label='WGHE (Água Reutilizada)', color='tab:green')
    
    ax1.set_title('Fluxo de H2')
    ax1.set_ylabel('Queda de Temperatura (Delta T) (°C)')
    ax1.set_xticks([width/2, width/2 + width + 0.1])
    ax1.set_xticklabels(['Dry Cooler', 'WGHE'])
    
    # Rótulos H2
    for bar_container in [bar1, bar2]:
        for bar in bar_container:
            yval = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, yval * 1.05, f'+{yval:.2f}', ha='center', va='bottom', fontsize=9)
        
    
    # Subplot O2
    ax2 = axes[1]
    o2_plot = df_plot.loc[df_plot['Gás'] == 'O₂']
    o2_dc_delta = o2_plot[o2_plot['Sistema C1'] == 'Dry Cooler']['Delta_T (°C)'].iloc[0]
    o2_wghe_delta = o2_plot[o2_plot['Sistema C1'] == 'WGHE']['Delta_T (°C)'].iloc[0]
    
    bar3 = ax2.bar(width/2, o2_dc_delta, width, label='Dry Cooler (Ar)', color='tab:blue')
    bar4 = ax2.bar(width/2 + width + 0.1, o2_wghe_delta, width, label='WGHE (Água Reutilizada)', color='tab:green')
    
    ax2.set_title('Fluxo de O2')
    ax2.set_xticks([width/2, width/2 + width + 0.1])
    ax2.set_xticklabels(['Dry Cooler', 'WGHE'])
    
    # Rótulos O2
    for bar_container in [bar3, bar4]:
        for bar in bar_container:
            yval = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, yval * 1.05, f'+{yval:.2f}', ha='center', va='bottom', fontsize=9)
        
    ax1.grid(axis='y', linestyle='--')
    ax2.grid(axis='y', linestyle='--')
    ax1.legend(loc='upper left', bbox_to_anchor=(0.0, -0.15))
    ax2.legend(loc='upper left', bbox_to_anchor=(0.0, -0.15))

    plt.tight_layout(rect=[0, 0, 1.0, 0.95])
    plt.show()

# =================================================================
# === GRÁFICO 2: COMPARAÇÃO DE ENERGIA C1 (Q por Fase vs W) ===
# =================================================================

def plot_comparacao_energia_c1(df_h2_dc: pd.DataFrame, df_h2_wghe: pd.DataFrame, df_o2_dc: pd.DataFrame, df_o2_wghe: pd.DataFrame):
    """
    Gera gráfico comparativo da energia (Q removido por fase e W consumido)
    no primeiro estágio (C1) para Dry Cooler vs. WGHE (H2 e O2).
    """
    
    # --- Função Auxiliar para Extração Segura de Dados de Energia ---
    def safe_extract_energy(df, component_name):
        df_filtered = df[df['Componente'] == component_name]
        if not df_filtered.empty:
            data = df_filtered.iloc[0]
            q_gas = data.get('Q_dot_H2_Gas', 0.0) / 1000 
            q_h2o = data.get('Q_dot_H2O_Total', 0.0) / 1000
            w_comp = data.get('W_dot_comp_W', 0.0) / 1000
            q_total = data.get('Q_dot_fluxo_W', 0.0) / 1000
            return q_gas, q_h2o, w_comp, q_total
        return 0.0, 0.0, 0.0, 0.0


    # --- Extração de Dados (Dry Cooler) ---
    q_gas_h2_dc, q_h2o_h2_dc, w_comp_h2_dc, q_total_h2_dc = safe_extract_energy(df_h2_dc, 'Dry Cooler 1')
    q_gas_o2_dc, q_h2o_o2_dc, w_comp_o2_dc, q_total_o2_dc = safe_extract_energy(df_o2_dc, 'Dry Cooler 1')
    
    # --- Extração de Dados (WGHE) ---
    q_gas_h2_wghe, q_h2o_h2_wghe, w_comp_h2_wghe, q_total_h2_wghe = safe_extract_energy(df_h2_wghe, 'WGHE 1')
    q_gas_o2_wghe, q_h2o_o2_wghe, w_comp_o2_wghe, q_total_o2_wghe = safe_extract_energy(df_o2_wghe, 'WGHE 1')
    
    data = {
        'Gás': ['H₂', 'H₂', 'O₂', 'O₂'],
        'Sistema C1': ['Dry Cooler', 'WGHE', 'Dry Cooler', 'WGHE'],
        'Q_Gas_kW': [-q_gas_h2_dc, -q_gas_h2_wghe, -q_gas_o2_dc, -q_gas_o2_wghe], 
        'Q_H2O_kW': [-q_h2o_h2_dc, -q_h2o_h2_wghe, -q_h2o_o2_dc, -q_h2o_o2_wghe], 
        'Q_Total_kW': [-q_total_h2_dc, -q_total_h2_wghe, -q_total_o2_dc, -q_total_o2_wghe], 
        'W_Comp_kW': [w_comp_h2_dc, w_comp_h2_wghe, w_comp_o2_dc, w_comp_o2_wghe]
    }
    df_plot = pd.DataFrame(data)
    
    # --- Plotagem ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # CORREÇÃO: Usando texto simples
    fig.suptitle('Comparação da Carga Térmica (Q dot) Removida por Fase e Consumo de Potência (W dot) no Estágio C1', fontsize=14)
    
    bar_width = 0.35
    
    # Subplot H2
    ax1 = axes[0]
    h2_plot = df_plot[df_plot['Gás'] == 'H₂']
    r1 = np.arange(len(h2_plot))
    
    # Barras de Q (Empilhadas)
    bar1 = ax1.bar(r1, h2_plot['Q_Gas_kW'], bar_width, label='Q dot Gás Principal', color='blue')
    bar2 = ax1.bar(r1, h2_plot['Q_H2O_kW'], bar_width, bottom=h2_plot['Q_Gas_kW'], label='Q dot H₂O (Vapor+Líq)', color='skyblue')
    
    # Linha de W (Consumo)
    line1 = ax1.plot(r1, h2_plot['W_Comp_kW'], marker='s', linestyle='--', color='red', label='W dot Consumido (Elétrico) [Linha]')
    
    ax1.set_title('Fluxo de H2')
    ax1.set_ylabel('Potência (kW)')
    ax1.set_xticks(r1)
    ax1.set_xticklabels(h2_plot['Sistema C1'])
    ax1.grid(axis='y', linestyle='--')
    ax1.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0))
    
    # Rótulos Q Total e W Total
    for i in r1:
        total_q = h2_plot['Q_Total_kW'].iloc[i]
        total_w = h2_plot['W_Comp_kW'].iloc[i]
        # Rótulo Q
        ax1.text(r1[i], total_q * 1.05, f'Q: {total_q:.2f}', ha='center', va='bottom', fontsize=8, color='black')
        # Rótulo W
        ax1.text(r1[i], total_w + 0.1, f'W: {total_w:.4f}', ha='center', va='bottom', fontsize=8, color='red')
        
    
    # Subplot O2
    ax2 = axes[1]
    o2_plot = df_plot[df_plot['Gás'] == 'O₂']
    r2 = np.arange(len(o2_plot))
    
    # Barras de Q (Empilhadas)
    bar3 = ax2.bar(r2, o2_plot['Q_Gas_kW'], bar_width, label='Q dot Gás Principal', color='darkgreen')
    bar4 = ax2.bar(r2, o2_plot['Q_H2O_kW'], bar_width, bottom=o2_plot['Q_Gas_kW'], label='Q dot H₂O (Vapor+Líq)', color='lightgreen')
    
    # Linha de W (Consumo)
    line2 = ax2.plot(r2, o2_plot['W_Comp_kW'], marker='s', linestyle='--', color='red', label='W dot Consumido (Elétrico) [Linha]')

    ax2.set_title('Fluxo de O2')
    ax2.set_ylabel('Potência (kW)')
    ax2.set_xticks(r2)
    ax2.set_xticklabels(o2_plot['Sistema C1'])
    ax2.grid(axis='y', linestyle='--')
    ax2.set_xlabel('Sistema C1')
    ax2.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0))

    # Rótulos Q Total e W Total
    for i in r2:
        total_q = o2_plot['Q_Total_kW'].iloc[i]
        total_w = o2_plot['W_Comp_kW'].iloc[i]
        # Rótulo Q
        ax2.text(r2[i], total_q * 1.05, f'Q: {total_q:.2f}', ha='center', va='bottom', fontsize=8, color='black')
        # Rótulo W
        ax2.text(r2[i], total_w + 0.1, f'W: {total_w:.4f}', ha='center', va='bottom', fontsize=8, color='red')

    plt.tight_layout(rect=[0, 0, 1.0, 0.95])
    plt.show()

# =================================================================
# === NOVO GRÁFICO: COMPARAÇÃO DO LADO FRIO (T_out e m_dot) ===
# =================================================================

def plot_comparacao_lado_frio_c1(df_h2_dc: pd.DataFrame, df_h2_wghe: pd.DataFrame, df_o2_dc: pd.DataFrame, df_o2_wghe: pd.DataFrame):
    """
    Gera gráfico comparativo da temperatura de saída do fluido de resfriamento 
    (lado frio) e a vazão mássica desse fluido no estágio C1.
    """
    
    # --- Função Auxiliar para Extração Segura de Dados Frios ---
    def extract_cold_data(df, component_name, default_t_in):
        df_filtered = df[df['Componente'] == component_name]
        if not df_filtered.empty:
            data = df_filtered.iloc[0]
            T_out = data.get('T_cold_out_C', default_t_in)
            m_dot = data.get('m_dot_cold_liq_kg_s', 0.0)
            return T_out, m_dot
        return default_t_in, 0.0 # Usa a T de entrada como T de saída em caso de erro

    # --- Extração de Dados ---
    T_out_h2_dc, m_dot_h2_dc = extract_cold_data(df_h2_dc, 'Dry Cooler 1', T_A_IN_OP) 
    T_out_h2_wghe, m_dot_h2_wghe = extract_cold_data(df_h2_wghe, 'WGHE 1', T_COLD_IN_WGHE_C)
    T_out_o2_dc, m_dot_o2_dc = extract_cold_data(df_o2_dc, 'Dry Cooler 1', T_A_IN_OP)
    T_out_o2_wghe, m_dot_o2_wghe = extract_cold_data(df_o2_wghe, 'WGHE 1', T_COLD_IN_WGHE_C)
    
    data = {
        'Gás': ['H₂', 'H₂', 'O₂', 'O₂'],
        'Sistema C1': ['Dry Cooler (Ar)', 'WGHE (Água)', 'Dry Cooler (Ar)', 'WGHE (Água)'],
        'T_cold_in_C': [T_A_IN_OP, T_COLD_IN_WGHE_C, T_A_IN_OP, T_COLD_IN_WGHE_C], 
        'T_cold_out_C': [T_out_h2_dc, T_out_h2_wghe, T_out_o2_dc, T_out_o2_wghe],
        'm_dot_cold_kg_s': [m_dot_h2_dc, m_dot_h2_wghe, m_dot_o2_dc, m_dot_o2_wghe],
    }
    df_plot = pd.DataFrame(data)
    
    # Converter m_dot para kg/h para melhor visualização
    df_plot['m_dot_cold_kg_h'] = df_plot['m_dot_cold_kg_s'] * 3600
    
    # --- Plotagem ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('Comparação do Lado Frio (Cooling Medium) no Estágio C1', fontsize=14)
    
    x = np.arange(2) * 1.5 
    width = 0.35
    
    # Subplot H2
    ax1 = axes[0]
    h2_plot = df_plot.loc[df_plot['Gás'] == 'H₂'].reset_index(drop=True)
    
    # Eixo Y1: Temperatura de Saída (T_cold_out_C)
    color_temp = 'tab:red'
    ax1.set_ylabel('T_cold_out (°C)', color=color_temp)
    ax1.tick_params(axis='y', labelcolor=color_temp)
    
    bar_h2_temp_dc = ax1.bar(width/2, h2_plot['T_cold_out_C'].iloc[0], width, label='Dry Cooler (Ar) T_out', color=color_temp, alpha=0.6)
    bar_h2_temp_wghe = ax1.bar(width/2 + width + 0.1, h2_plot['T_cold_out_C'].iloc[1], width, label='WGHE (Água) T_out', color=color_temp, alpha=0.3)
    
    # Linha de T_in (referência)
    ax1.axhline(h2_plot['T_cold_in_C'].iloc[0], color='gray', linestyle='--', linewidth=1.0, label=f'T_cold_in (DC) = {h2_plot["T_cold_in_C"].iloc[0]:.1f} °C')
    ax1.axhline(h2_plot['T_cold_in_C'].iloc[1], color='darkgray', linestyle=':', linewidth=1.0, label=f'T_cold_in (WGHE) = {h2_plot["T_cold_in_C"].iloc[1]:.1f} °C')
    
    # Eixo Y2: Vazão Mássica (m_dot_cold_kg_h)
    ax2 = ax1.twinx()
    color_mdot = 'tab:blue'
    ax2.set_ylabel('Vazão Mássica (kg/h)', color=color_mdot)
    ax2.tick_params(axis='y', labelcolor=color_mdot)
    
    line_h2_mdot = ax2.plot([width/2, width/2 + width + 0.1], h2_plot['m_dot_cold_kg_h'], 
                            marker='o', linestyle='-', color=color_mdot, label='m_dot_cold (kg/h) [Linha]')
    
    ax1.set_title('Fluxo de H2')
    ax1.set_xticks([width/2, width/2 + width + 0.1])
    ax1.set_xticklabels(['Dry Cooler (Ar)', 'WGHE (Água)'])
    
    # Rótulos (T e m_dot)
    
    # Variável para o valor máximo para a normalização do rótulo
    max_mdot_h2 = h2_plot['m_dot_cold_kg_h'].max()
    
    for i in h2_plot.index:
        row = h2_plot.loc[i]
        # Rótulo T
        ax1.text(ax1.get_xticks()[i], row['T_cold_out_C'] * 1.05, f'T: {row["T_cold_out_C"]:.2f}', ha='center', va='bottom', fontsize=8, color=color_temp)
        # Rótulo m_dot (CORRIGIDO: usa o max da COLUNA)
        ax2.text(ax1.get_xticks()[i] + 0.1, row['m_dot_cold_kg_h'] + 0.05 * max_mdot_h2, f'm: {row["m_dot_cold_kg_h"]:.2f}', ha='center', va='bottom', fontsize=8, color=color_mdot)
    
    # Combina legendas
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper left', bbox_to_anchor=(1.05, 1.0))
    ax1.grid(axis='y', linestyle='--')


    # Subplot O2
    ax3 = axes[1]
    o2_plot = df_plot.loc[df_plot['Gás'] == 'O₂'].reset_index(drop=True)
    
    # Eixo Y1: Temperatura de Saída (T_cold_out_C)
    color_temp = 'tab:red'
    ax3.set_ylabel('T_cold_out (°C)', color=color_temp)
    ax3.tick_params(axis='y', labelcolor=color_temp)
    
    bar_o2_temp_dc = ax3.bar(width/2, o2_plot['T_cold_out_C'].iloc[0], width, label='Dry Cooler (Ar) T_out', color=color_temp, alpha=0.6)
    bar_o2_temp_wghe = ax3.bar(width/2 + width + 0.1, o2_plot['T_cold_out_C'].iloc[1], width, label='WGHE (Água) T_out', color=color_temp, alpha=0.3)
    
    # Linha de T_in (referência)
    ax3.axhline(o2_plot['T_cold_in_C'].iloc[0], color='gray', linestyle='--', linewidth=1.0, label=f'T_cold_in (DC) = {o2_plot["T_cold_in_C"].iloc[0]:.1f} °C')
    ax3.axhline(o2_plot['T_cold_in_C'].iloc[1], color='darkgray', linestyle=':', linewidth=1.0, label=f'T_cold_in (WGHE) = {o2_plot["T_cold_in_C"].iloc[1]:.1f} °C')

    # Eixo Y2: Vazão Mássica (m_dot_cold_kg_h)
    ax4 = ax3.twinx()
    color_mdot = 'tab:blue'
    ax4.set_ylabel('Vazão Mássica (kg/h)', color=color_mdot)
    ax4.tick_params(axis='y', labelcolor=color_mdot)
    
    line_o2_mdot = ax4.plot([width/2, width/2 + width + 0.1], o2_plot['m_dot_cold_kg_h'], 
                            marker='o', linestyle='-', color=color_mdot, label='m_dot_cold (kg/h) [Linha]')

    ax3.set_title('Fluxo de O2')
    ax3.set_xticks([width/2, width/2 + width + 0.1])
    ax3.set_xticklabels(['Dry Cooler (Ar)', 'WGHE (Água)'])
    ax3.set_xlabel('Sistema C1')
    
    # Rótulos (T e m_dot)
    
    # Variável para o valor máximo para a normalização do rótulo
    max_mdot_o2 = o2_plot['m_dot_cold_kg_h'].max()
    
    for i in o2_plot.index:
        row = o2_plot.loc[i]
        # Rótulo T
        ax3.text(ax3.get_xticks()[i], row['T_cold_out_C'] * 1.05, f'T: {row["T_cold_out_C"]:.2f}', ha='center', va='bottom', fontsize=8, color=color_temp)
        # Rótulo m_dot (CORRIGIDO: usa o max da COLUNA)
        ax4.text(ax3.get_xticks()[i] + 0.1, row['m_dot_cold_kg_h'] + 0.05 * max_mdot_o2, f'm: {row["m_dot_cold_kg_h"]:.2f}', ha='center', va='bottom', fontsize=8, color=color_mdot)

    # Combina legendas
    h3, l3 = ax3.get_legend_handles_labels()
    h4, l4 = ax4.get_legend_handles_labels()
    ax3.legend(h3 + h4, l3 + l4, loc='upper left', bbox_to_anchor=(1.05, 1.0))
    ax3.grid(axis='y', linestyle='--')

    plt.tight_layout(rect=[0, 0, 1.0, 0.95])
    plt.show()