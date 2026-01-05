# plots_modulos/plot_impurezas_crossover.py

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from plot_reporter_base import salvar_e_exibir_plot

# CORREÇÃO: Adicionando 'mostrar_grafico' como 6º argumento posicional
def plot_impurezas_crossover(df_h2: pd.DataFrame, df_o2: pd.DataFrame, deoxo_mode: str, L_deoxo: float, dc2_mode: str, mostrar_grafico: bool = False):
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
    
    # SALVAMENTO CENTRALIZADO
    nome_arquivo = f'plot_impurezas_crossover.png'
    salvar_e_exibir_plot(nome_arquivo, mostrar_grafico)