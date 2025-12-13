# plots_modulos/plot_propriedades_empilhadas.py

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from plot_reporter_base import salvar_e_exibir_plot

# CORREÇÃO: Adicionando 'mostrar_grafico' como 6º argumento posicional
def plot_propriedades_empilhadas(df: pd.DataFrame, gas_fluido: str, deoxo_mode: str, L_deoxo: float, dc2_mode: str, mostrar_grafico: bool = False):
    """
    Gera gráficos empilhados de T, P, w_H2O, H vs. Componente para um único fluxo,
    incluindo anotações de valor em cada ponto.
    """
    
    df_plot = df.copy()

    x_labels = df_plot['Componente']
    
    # 💥 NOVO DEBUG: Confirma se o compressor está aqui
    print(f"DEBUG {gas_fluido} PLOT: Componentes no eixo X: {x_labels.tolist()}") 
    
    # Apenas 4 eixos são necessários (T, P, w_H2O e Entalpia Mássica)
    fig, axes = plt.subplots(4, 1, figsize=(10, 15), sharex=True)
    fig.suptitle(f'Evolução das Propriedades do Fluxo de {gas_fluido}', y=1.0) 
    
    
    # --- Função Auxiliar para Adicionar Rótulos de Valor (Anotações) ---
    def add_value_labels(ax, x, y, format_str, offset=0.0):
        """Adiciona o valor de y como rótulo em cada ponto (x, y)."""
        for i in range(len(x)):
            # Tenta centralizar o texto um pouco acima do ponto
            ax.annotate(format_str.format(y.iloc[i]), 
                        (x.iloc[i], y.iloc[i] + offset), 
                        textcoords="offset points", 
                        xytext=(0, 5), 
                        ha='center',
                        fontsize=8)
    # --- FIM da Função Auxiliar ---


    # 1. Temperatura
    axes[0].plot(x_labels, df_plot['T_C'], marker='o', label=f'{gas_fluido} - Temperatura (°C)', color='blue')
    axes[0].set_ylabel('T (°C)')
    axes[0].grid(True, linestyle='--')
    axes[0].legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    # Adicionar Rótulos de Valor
    add_value_labels(axes[0], x_labels, df_plot['T_C'], '{:.1f}') 


    # 2. Pressão
    axes[1].plot(x_labels, df_plot['P_bar'], marker='o', label=f'{gas_fluido} - Pressão (bar)', color='red')
    axes[1].set_ylabel('P (bar)')
    axes[1].grid(True, linestyle='--')
    axes[1].legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    # Adicionar Rótulos de Valor
    add_value_labels(axes[1], x_labels, df_plot['P_bar'], '{:.2f}') 


    # 3. Fração Mássica de H2O (w_H2O)
    w_H2O_percent = df_plot['w_H2O'] * 100
    axes[2].plot(x_labels, w_H2O_percent, marker='o', label=f'{gas_fluido} - w_H₂O Mássica (%)', color='green')
    axes[2].set_ylabel('w_H₂O Mássica (%)')
    axes[2].grid(True, linestyle='--')
    axes[2].legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    # Adicionar Rótulos de Valor
    # Usa notação científica para valores muito pequenos (H2O < 0.01%) e notação decimal para o resto
    for i in range(len(x_labels)):
        val = w_H2O_percent.iloc[i]
        format_str = '{:.4f}' if abs(val) >= 0.0001 else '{:.2e}'
        axes[2].annotate(format_str.format(val),
                         (x_labels.iloc[i], val), 
                         textcoords="offset points", 
                         xytext=(0, 5), 
                         ha='center',
                         fontsize=8)

    # 4. Entalpia Mássica (APENAS H_mix)
    H_mix_kJ_kg = df_plot['H_mix_J_kg'] / 1000
    axes[3].plot(x_labels, H_mix_kJ_kg, marker='o', 
                 label=f'H_mix (Gás+Vapor) (kJ/kg)', color='purple')
        
    axes[3].set_ylabel('H (kJ/kg)')
    axes[3].set_xlabel('Componente')
    axes[3].grid(True, linestyle='--')
    
    # 💥 CORREÇÃO: Aumenta a rotação para evitar sobreposição nos rótulos de componente
    axes[3].tick_params(axis='x', rotation=45) 
    
    # Adicionar Rótulos de Valor
    add_value_labels(axes[3], x_labels, H_mix_kJ_kg, '{:.2f}') 
         
    axes[3].legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))

    # 💥 CORREÇÃO PRINCIPAL: Garante que todos os rótulos X sejam mostrados (Ajuste para rotação 45)
    plt.xticks(range(len(x_labels)), x_labels, rotation=45, ha='right')

    plt.tight_layout(rect=[0, 0, 0.88, 0.95]) 
    
    # SALVAMENTO CENTRALIZADO
    nome_arquivo = f'plot_propriedades_empilhadas_{gas_fluido}.png'
    salvar_e_exibir_plot(nome_arquivo, mostrar_grafico)