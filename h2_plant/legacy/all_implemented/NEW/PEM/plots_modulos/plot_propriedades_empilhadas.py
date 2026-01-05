# plots_modulos/plot_propriedades_empilhadas.py

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
# REMOVIDO: from plot_reporter_base import adicionar_entalpia_pura, calcular_entalpia_total_fluxo, salvar_e_exibir_plot
from plot_reporter_base import salvar_e_exibir_plot

# CORREÇÃO: Adicionando 'mostrar_grafico' como 6º argumento posicional
def plot_propriedades_empilhadas(df: pd.DataFrame, gas_fluido: str, deoxo_mode: str, L_deoxo: float, dc2_mode: str, mostrar_grafico: bool = False):
    """Gera gráficos empilhados de T, P, w_H2O, H vs. Componente para um único fluxo."""
    
    df_plot = df.copy()

    x_labels = df_plot['Componente']
    # Apenas 3 eixos são necessários agora (T, P, w_H2O e Entalpia Mássica)
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

    # 4. Entalpia Mássica (APENAS H_mix)
    
    axes[3].plot(x_labels, df_plot['H_mix_J_kg'] / 1000, marker='o', 
                 label=f'H_mix (Gás+Vapor) (kJ/kg)', color='purple')
        
    axes[3].set_ylabel('H (kJ/kg)')
    axes[3].set_xlabel('Componente')
    axes[3].grid(True, linestyle='--')
    axes[3].tick_params(axis='x', rotation=15)
    
    # Rótulos para H_mix (o único valor)
    for x, y in zip(range(len(x_labels)), df_plot['H_mix_J_kg'] / 1000):
         axes[3].text(x, y * 1.02, f'{y:.2f}', ha='center', va='bottom', fontsize=8, color='purple')
         
    axes[3].legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))

    plt.tight_layout(rect=[0, 0, 0.88, 0.95]) 
    
    # SALVAMENTO CENTRALIZADO
    nome_arquivo = f'plot_propriedades_empilhadas_{gas_fluido}.png'
    salvar_e_exibir_plot(nome_arquivo, mostrar_grafico)