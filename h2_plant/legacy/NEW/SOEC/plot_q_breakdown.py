# plots_modulos/plot_q_breakdown.py

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from plot_reporter_base import salvar_e_exibir_plot # Importa a função central

# CORREÇÃO: Adicionando 'mostrar_grafico' como 6º argumento posicional
def plot_q_breakdown(df_h2: pd.DataFrame, df_o2: pd.DataFrame, deoxo_mode: str, L_deoxo: float, dc2_mode: str, mostrar_grafico: bool = False):
    """
    Gera gráfico da carga térmica (Q_dot) por fase: Total, Gás e Líquido/Latente.
    """
    
    # Remove a entrada (Assumindo que 8 e 6 são o número de componentes após a entrada)
    df_plot_h2 = df_h2[df_h2['Componente'] != 'Entrada'].copy()
    df_plot_o2 = df_o2[df_o2['Componente'] != 'Entrada'].copy()

    # Filtros de Q dot
    df_plot_h2['Q_dot_Gas'] = df_plot_h2.get('Q_dot_H2_Gas', 0.0) 
    df_plot_h2['Q_dot_H2O_Total'] = df_plot_h2.get('Q_dot_H2O_Total', 0.0)
    df_plot_o2['Q_dot_Gas'] = df_plot_o2.get('Q_dot_H2_Gas', 0.0) 
    df_plot_o2['Q_dot_H2O_Total'] = df_plot_o2.get('Q_dot_H2O_Total', 0.0) 

    # Converte Q_dot para kW
    df_plot_h2['Q_dot_Gas_kW'] = df_plot_h2['Q_dot_Gas'] / 1000
    df_plot_h2['Q_dot_H2O_Total_kW'] = df_plot_h2['Q_dot_H2O_Total'] / 1000
    df_plot_o2['Q_dot_Gas_kW'] = df_plot_o2['Q_dot_Gas'] / 1000
    df_plot_o2['Q_dot_H2O_Total_kW'] = df_plot_o2['Q_dot_H2O_Total'] / 1000
    
    
    # 2. DEFINIÇÃO DOS EIXOS X (r_h2 deve ser o eixo principal)
    comp_labels_h2 = df_plot_h2['Componente'].tolist()
    comp_labels_o2 = df_plot_o2['Componente'].tolist()

    r_h2 = np.arange(len(comp_labels_h2)) # Array X para H2 (8 pontos)
    r_o2 = np.arange(len(comp_labels_o2)) # Array X para O2 (6 pontos)
    
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=False) 
    # TÍTULO CORRIGIDO (Instrução do usuário)
    fig.suptitle('Carga Térmica (Q dot) por Componente e Fase', y=1.0) 
    
    # --- Subplot 1: Fluxo H2 (Baseado no r_h2)
    ax1 = axes[0]
    bar_width = 0.35
    
    barras_q_h2_agua = ax1.bar(r_h2, df_plot_h2['Q_dot_H2O_Total_kW'], color='skyblue', width=bar_width, edgecolor='grey', label='H2O (Vapor + Líquido)')
    barras_q_h2_gas = ax1.bar(r_h2, df_plot_h2['Q_dot_Gas_kW'], bottom=df_plot_h2['Q_dot_H2O_Total_kW'], color='blue', width=bar_width, edgecolor='grey', label='H2 (Gás Principal)')
    
    ax1.set_title('Fluxo H₂')
    ax1.set_ylabel('Carga Térmica Removida (Q dot) (kW)')
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.grid(axis='y', linestyle='--')
    ax1.set_xticks(r_h2 + bar_width / 2) # Ajuste dos ticks para o centro das barras
    ax1.set_xticklabels(comp_labels_h2)
    ax1.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    
    # Rótulos Q - H2 (CORRIGIDOS para posicionamento lateral)
    for i in range(len(r_h2)):
        total = df_plot_h2['Q_dot_Gas_kW'].iloc[i] + df_plot_h2['Q_dot_H2O_Total_kW'].iloc[i]
        if total != 0:
            # Posiciona o rótulo à direita da barra
            ax1.text(r_h2[i] + bar_width + 0.05, total if total > 0 else total, f'{total:.2f}', ha='left', va='center', fontsize=7, color='black')


    # --- Subplot 2: Fluxo O2 (Baseado no r_o2)
    ax2 = axes[1]
    
    # CORREÇÃO CRÍTICA: O eixo X deve usar r_o2 (que tem 6 pontos)
    barras_q_o2_agua = ax2.bar(r_o2, df_plot_o2['Q_dot_H2O_Total_kW'], color='salmon', width=bar_width, edgecolor='grey', label='H2O (Vapor + Líquido)')
    barras_q_o2_gas = ax2.bar(r_o2, df_plot_o2['Q_dot_Gas_kW'], bottom=df_plot_o2['Q_dot_H2O_Total_kW'], color='red', width=bar_width, edgecolor='grey', label='O2 (Gás Principal)')

    ax2.set_title('Fluxo O₂')
    ax2.set_ylabel('Carga Térmica Removida (Q dot) (kW)')
    ax2.set_xlabel('Componente')
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.grid(axis='y', linestyle='--')
    
    # Ajuste dos ticks para o centro das barras e labels O2 (6 pontos)
    ax2.set_xticks(r_o2 + bar_width / 2) 
    ax2.set_xticklabels(comp_labels_o2, rotation=15)
    ax2.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))

    # Rótulos Q - O2 (CORRIGIDOS para posicionamento lateral)
    for i in range(len(r_o2)):
        total = df_plot_o2['Q_dot_Gas_kW'].iloc[i] + df_plot_o2['Q_dot_H2O_Total_kW'].iloc[i]
        if total != 0:
             # Posiciona o rótulo à direita da barra
             ax2.text(r_o2[i] + bar_width + 0.05, total if total > 0 else total, f'{total:.2f}', ha='left', va='center', fontsize=7, color='black')

    plt.tight_layout(rect=[0, 0, 0.95, 0.95])
    
    # SALVAMENTO CENTRALIZADO
    nome_arquivo = 'plot_q_breakdown.png'
    salvar_e_exibir_plot(nome_arquivo, mostrar_grafico)