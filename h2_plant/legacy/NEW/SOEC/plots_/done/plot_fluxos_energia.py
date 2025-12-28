# plots_modulos/plot_fluxos_energia.py

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from plot_reporter_base import salvar_e_exibir_plot # Importa a função central

# CORREÇÃO: Adicionando 'mostrar_grafico' como 6º argumento posicional
def plot_fluxos_energia(df_h2: pd.DataFrame, df_o2: pd.DataFrame, deoxo_mode: str, L_deoxo: float, dc2_mode: str, mostrar_grafico: bool = False):
    """
    Gera gráficos de Fluxos de Energia (Q dot trocado e W dot elétrico) por componente.
    """
    
    # Remove a entrada (Assumindo que 8 e 6 são o número de componentes após a entrada)
    df_plot_h2 = df_h2[df_h2['Componente'] != 'Entrada'].copy()
    df_plot_o2 = df_o2[df_o2['Componente'] != 'Entrada'].copy()
    
    # 2. DEFINIÇÃO DOS EIXOS X e Rótulos
    comp_labels_h2 = df_plot_h2['Componente'].tolist()
    comp_labels_o2 = df_plot_o2['Componente'].tolist()
    
    # Eixos X independentes para garantir o shape matching
    r_h2 = np.arange(len(comp_labels_h2)) # Array X para H2 (8 pontos)
    r_o2 = np.arange(len(comp_labels_o2)) # Array X para O2 (6 pontos)
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=False) # Mudado para 10x10 para melhor visualização
    fig.suptitle('Fluxos de Energia e Consumo Total por Componente', y=1.0) 
    
    bar_width = 0.35
    
    # 1. Energia que sai ou entra do FLUXO (Q_dot_fluxo_W)
    ax1 = axes[0]
    q_fluxo_h2 = df_plot_h2['Q_dot_fluxo_W'] / 1000 # kW
    q_fluxo_o2 = df_plot_o2['Q_dot_fluxo_W'] / 1000 # kW
    
    # CORREÇÃO CRÍTICA 1: Plotar H2 vs O2 em eixos X diferentes (r_h2 e r_o2)
    
    # H2 (Plotado na posição r_h2)
    barras_q_h2 = ax1.bar(r_h2 - bar_width / 2, q_fluxo_h2, color='blue', width=bar_width, edgecolor='grey', label='H2 - Calor Trocado (Fluxo)')
    
    # O2 (Plotado na posição r_o2)
    barras_q_o2 = ax1.bar(r_o2 + bar_width / 2, q_fluxo_o2, color='red', width=bar_width, edgecolor='grey', label='O2 - Calor Trocado (Fluxo)')

    ax1.set_title('Carga Térmica (Q dot Trocado com o Fluxo) - Comparativo H2 vs O2')
    ax1.set_ylabel('Potência Térmica (kW)')
    
    # Ticks devem ser definidos pelo maior array (H2)
    ax1.set_xticks(r_h2)
    ax1.set_xticklabels(comp_labels_h2)
    
    ax1.grid(axis='y', linestyle='--')
    ax1.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    
    # Rótulos Q - H2
    for bar in barras_q_h2:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval * 1.05 if yval > 0 else yval * 0.95, f"{yval:.2f}", ha='center', va='center', fontsize=7, color='blue')
    # Rótulos Q - O2
    for bar in barras_q_o2:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval * 1.05 if yval > 0 else yval * 0.95, f"{yval:.2f}", ha='center', va='center', fontsize=7, color='red')


    # 2. Entrada/Saída de Energia do COMPONENTE (W_dot_comp_W)
    ax2 = axes[1]
    w_comp_h2 = df_plot_h2['W_dot_comp_W'] / 1000 # kW
    w_comp_o2 = df_plot_o2['W_dot_comp_W'] / 1000 # kW
    
    # H2 (Plotado na posição r_h2)
    barras_w_h2 = ax2.bar(r_h2 - bar_width / 2, w_comp_h2, color='skyblue', width=bar_width, edgecolor='grey', label='H2 - Consumo Total (Elétrico)')
    # O2 (Plotado na posição r_o2)
    barras_w_o2 = ax2.bar(r_o2 + bar_width / 2, w_comp_o2, color='salmon', width=bar_width, edgecolor='grey', label='O2 - Consumo Total (Elétrico)')
    
    ax2.set_title('Consumo Total de Potência por Componente (W dot Elétrico)')
    ax2.set_ylabel('Potência (kW)')
    
    # Ticks devem ser definidos pelo maior array (H2)
    ax2.set_xticks(r_h2)
    ax2.set_xticklabels(comp_labels_h2, rotation=15)
    ax2.set_xlabel('Componente')
    
    ax2.grid(axis='y', linestyle='--')
    ax2.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))

    # Rótulos W - H2
    for bar in barras_w_h2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval * 1.05, f"{yval:.2f}", ha='center', va='bottom', fontsize=7, color='blue')
    # Rótulos W - O2
    for bar in barras_w_o2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval * 1.05, f"{yval:.2f}", ha='center', va='bottom', fontsize=7, color='red')


    plt.tight_layout(rect=[0, 0, 0.95, 0.95])
    
    # SALVAMENTO CENTRALIZADO
    nome_arquivo = 'plot_fluxos_energia.png'
    salvar_e_exibir_plot(nome_arquivo, mostrar_grafico)