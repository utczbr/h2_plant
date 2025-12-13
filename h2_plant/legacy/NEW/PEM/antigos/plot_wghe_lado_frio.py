# plots_modulos/plot_wghe_lado_frio.py

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

from constants_and_config import T_COLD_IN_WGHE_C 

def plot_comparacao_wghe_lado_frio(df_h2_fase1: pd.DataFrame, df_h2_fase2: pd.DataFrame, df_o2_fase1: pd.DataFrame, df_o2_fase2: pd.DataFrame):
    """
    Gera gráfico do desempenho do WGHE (Lado Frio), exibindo apenas o resultado da Fase 2 (Real).
    """
    
    def extract_wghe_cold_data(df, gas_fluido, phase_label):
        if phase_label not in ['Fase 2 (Real)', 'Fase 2']:
            return None
            
        if 'WGHE 1' not in df['Componente'].values:
             return None
             
        df_wghe = df[df['Componente'] == 'WGHE 1'].iloc[0]
        
        T_cold_in = T_COLD_IN_WGHE_C
        T_cold_out = df_wghe['T_cold_out_C']
        
        Delta_T_C = T_cold_out - T_cold_in
        m_dot_cold_kg_h = df_wghe['m_dot_cold_liq_kg_s'] * 3600
        
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

    data = []
    
    h2_f2 = extract_wghe_cold_data(df_h2_fase2, 'H₂', 'Fase 2')
    o2_f2 = extract_wghe_cold_data(df_o2_fase2, 'O₂', 'Fase 2')
    
    if h2_f2: data.append(h2_f2)
    if o2_f2: data.append(o2_f2)

    if not data:
        print("\n--- Aviso: WGHE não está nos dataframes da Fase 2 ou dados insuficientes para plotagem. ---")
        return
        
    df_plot = pd.DataFrame(data)
    
    # --- Plotagem ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    fig.suptitle(f'Desempenho do WGHE (Lado Frio - Fluido de Resfriamento)', fontsize=14)
    
    bar_width = 0.35
    
    # 1. Subplot H2
    ax1 = axes[0]
    h2_plot = df_plot[df_plot['Gás'] == 'H₂'].reset_index(drop=True)
    r1 = np.arange(len(h2_plot))
    
    color_temp = 'tab:red'
    ax1.set_ylabel('T (°C)', color=color_temp)
    ax1.tick_params(axis='y', labelcolor=color_temp)
    
    y_max_h2 = h2_plot['T_cold_out_C'].max() if not h2_plot.empty else T_COLD_IN_WGHE_C
    ax1.set_ylim(T_COLD_IN_WGHE_C - 1, y_max_h2 * 1.1 + 1)
    
    bottoms_h2 = h2_plot['T_cold_in_C'].apply(lambda x: x if x > 0 else 0) 
    bars_t_h2 = ax1.bar(r1, h2_plot['Delta_T_bar'], bar_width, bottom=bottoms_h2, label='Delta T (°C) (Barra)', color='skyblue')
    
    ax1.axhline(T_COLD_IN_WGHE_C, color='gray', linestyle='--', linewidth=1.0, label=f'T_cold_in = {T_COLD_IN_WGHE_C:.1f} °C')
    
    
    ax2 = ax1.twinx()
    color_mdot = 'tab:blue'
    ax2.set_ylabel('Vazão Mássica de Reuso (kg/h)', color=color_mdot)
    ax2.tick_params(axis='y', labelcolor=color_mdot)
    
    m_dot_max_h2 = h2_plot['m_dot_total_reuso_kg_h'].max()
    if m_dot_max_h2 > 0:
        y_max_h2_mdot = m_dot_max_h2 * 1.5 
    else:
        y_max_h2_mdot = 100 
        
    ax2.set_ylim(0, y_max_h2_mdot) 
        
    ax2.plot(r1, h2_plot['m_dot_total_reuso_kg_h'], marker='o', linestyle='-', color=color_mdot, label='m_dot Reuso TOTAL (kg/h)')
    
    ax1.set_title('Fluxo de H₂')
    ax1.set_xticks(r1)
    ax1.set_xticklabels(h2_plot['Fase'])
    ax1.set_xlabel('Fase da Simulação') 
    ax1.grid(axis='y', linestyle='--')
    
    for i, row in h2_plot.iterrows():
        ax1.text(r1[i], row['T_cold_in_C'] * 1.05 if row['T_cold_in_C'] > 0 else 0.5, 
                 f'T_in: {row["T_cold_in_C"]:.2f}°C', 
                 ha='center', va='bottom', fontsize=8, color='black', fontweight='bold')
                 
        y_out_pos = row['T_cold_in_C'] + row['Delta_T_bar']
        
        ax1.text(r1[i], y_out_pos * 1.02, 
                 f'T_out: {row["T_cold_out_C"]:.2f}°C', 
                 ha='center', va='bottom', fontsize=8, color='black', fontweight='bold')
                 
        pos_y_delta_t = row['T_cold_in_C'] + row['Delta_T_bar'] / 2
        ax1.text(r1[i], pos_y_delta_t, 
                 f'ΔT: +{row["Delta_T_C"]:.2f}°C', 
                 ha='center', va='center', fontsize=7, color='tab:red', fontweight='bold')
        
        if row['m_dot_total_reuso_kg_h'] > 1e-3:
            ax2.text(r1[i] + 0.1, row['m_dot_total_reuso_kg_h'] * 1.05, 
                     f'{row["m_dot_total_reuso_kg_h"]:.2f} kg/h', 
                     ha='center', va='bottom', fontsize=8, color='tab:blue')
    
    # 2. Subplot O2
    ax3 = axes[1]
    o2_plot = df_plot[df_plot['Gás'] == 'O₂'].reset_index(drop=True)
    r2 = np.arange(len(o2_plot))
    
    ax3.set_ylabel('T (°C)', color=color_temp)
    ax3.tick_params(axis='y', labelcolor=color_temp)
    
    y_max_o2 = o2_plot['T_cold_out_C'].max() if not o2_plot.empty else T_COLD_IN_WGHE_C
    ax3.set_ylim(T_COLD_IN_WGHE_C - 1, y_max_o2 * 1.1 + 1)
    
    bottoms_o2 = o2_plot['T_cold_in_C'].apply(lambda x: x if x > 0 else 0)
    bars_t_o2 = ax3.bar(r2, o2_plot['Delta_T_bar'], bar_width, bottom=bottoms_o2, label='Delta T (°C) (Barra)', color='salmon')
    
    ax3.axhline(T_COLD_IN_WGHE_C, color='gray', linestyle='--', linewidth=1.0, label=f'T_cold_in = {T_COLD_IN_WGHE_C:.1f} °C')
    
    
    ax4 = ax3.twinx()
    ax4.set_ylabel('Vazão Mássica de Reuso (kg/h)', color='tab:blue')
    ax4.tick_params(axis='y', labelcolor='tab:blue')
    
    m_dot_max_o2 = o2_plot['m_dot_total_reuso_kg_h'].max()
    if m_dot_max_o2 > 0:
        y_max_o2_mdot = m_dot_max_o2 * 1.5
    else:
        y_max_o2_mdot = 100 
        
    ax4.set_ylim(0, y_max_o2_mdot) 
    
    ax4.plot(r2, o2_plot['m_dot_total_reuso_kg_h'], marker='o', linestyle='-', color='tab:blue', label='m_dot Reuso TOTAL (kg/h)')
    
    ax3.set_title('Fluxo de O₂')
    ax3.set_xticks(r2)
    ax3.set_xticklabels(o2_plot['Fase'])
    ax3.set_xlabel('Fase da Simulação')
    ax3.grid(axis='y', linestyle='--')
    
    for i, row in o2_plot.iterrows():
        ax3.text(r2[i], row['T_cold_in_C'] * 1.05 if row['T_cold_in_C'] > 0 else 0.5, 
                 f'T_in: {row["T_cold_in_C"]:.2f}°C', 
                 ha='center', va='bottom', fontsize=8, color='black', fontweight='bold')
                 
        y_out_pos = row['T_cold_in_C'] + row['Delta_T_bar']
                 
        ax3.text(r2[i], y_out_pos * 1.02, 
                 f'T_out: {row["T_cold_out_C"]:.2f}°C', 
                 ha='center', va='bottom', fontsize=8, color='black', fontweight='bold')
                 
        pos_y_delta_t = row['T_cold_in_C'] + row['Delta_T_bar'] / 2
        ax3.text(r2[i], pos_y_delta_t, 
                 f'ΔT: +{row["Delta_T_C"]:.2f}°C', 
                 ha='center', va='center', fontsize=7, color='tab:red', fontweight='bold')

        if row['m_dot_total_reuso_kg_h'] > 1e-3:
            ax4.text(r2[i] + 0.1, row['m_dot_total_reuso_kg_h'] * 1.05, 
                     f'{row["m_dot_total_reuso_kg_h"]:.2f} kg/h', 
                     ha='center', va='bottom', fontsize=8, color='tab:blue')
            
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    
    unique_labels = {}
    for h, l in zip(h1 + h2, l1 + l2):
        if l not in unique_labels:
            unique_labels[l] = h
            
    fig.legend(unique_labels.values(), unique_labels.keys(), loc='upper left', bbox_to_anchor=(0.08, 0.95), ncol=4)

    plt.tight_layout(rect=[0, 0, 1.0, 0.90])
    plt.show()