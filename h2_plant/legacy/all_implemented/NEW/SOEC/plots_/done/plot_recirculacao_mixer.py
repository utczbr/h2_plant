# plots_modulos/plot_recirculacao_mixer.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Importa a função de salvamento centralizada
try:
    # Ajusta o path se necessário para acessar o nível superior
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from plot_reporter_base import salvar_e_exibir_plot 
except ImportError:
    print("ERRO: Não foi possível importar salvar_e_exibir_plot. Usando stub.")
    def salvar_e_exibir_plot(nome_arquivo: str, mostrar_grafico: bool = True):
        plt.close()
        print(f"AVISO: Stub de salvamento para {nome_arquivo} chamado.")


def plot_recirculacao_mixer(df_dados: pd.DataFrame, mostrar_grafico: bool = True):
    """
    Gera o gráfico comparativo das propriedades da água (Vazão, P, T)
    antes (Dreno Final) e depois (Água de Recirculação Pós-Reposição).
    
    MODIFICAÇÕES:
    1. Gráfico de Linha com Marcadores (Pontos) em vez de Barras.
    2. Unidade da Vazão alterada de kg/s para kg/h.
    3. Uso da função de salvamento centralizada.
    
    Args:
        df_dados (pd.DataFrame): DataFrame com as linhas 'Dreno Final (Antes)' e 
                                 'Recirculação (Depois)'.
        mostrar_grafico (bool): Se deve exibir o gráfico.
    """
    
    fig, axes = plt.subplots(3, 1, figsize=(8, 10))
    x_labels = df_dados['Estado']
    x_indices = np.arange(len(x_labels)) # Índices para o eixo X
    
    # 0. Preparação de Vazão em kg/h
    df_plot = df_dados.copy()
    df_plot['Vazão (kg/h)'] = df_plot['Vazão (kg/s)'] * 3600 # Nova unidade
    
    # -----------------------------------------------------
    # 1. Vazão (AGORA COMO LINHA E EM kg/h)
    axes[0].plot(x_indices, df_plot['Vazão (kg/h)'], marker='o', linestyle='-', color='darkblue')
    axes[0].set_title('Vazão Mássica (kg/h)')
    axes[0].set_ylabel('Vazão (kg/h)')
    axes[0].set_xticks(x_indices, x_labels) # Define os rótulos do eixo X
    axes[0].grid(axis='y', linestyle='--')
    
    # Adiciona rótulos de valor
    for i, v in enumerate(df_plot['Vazão (kg/h)']):
        axes[0].text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
    
    # -----------------------------------------------------
    # 2. Pressão (AGORA COMO LINHA)
    axes[1].plot(x_indices, df_plot['Pressão (bar)'], marker='o', linestyle='-', color='darkorange')
    axes[1].set_title('Pressão (bar)')
    axes[1].set_ylabel('Pressão (bar)')
    axes[1].set_xticks(x_indices, x_labels)
    axes[1].grid(axis='y', linestyle='--')
    
    # Adiciona rótulos de valor
    for i, v in enumerate(df_plot['Pressão (bar)']):
        axes[1].text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
    
    # -----------------------------------------------------
    # 3. Temperatura (AGORA COMO LINHA)
    axes[2].plot(x_indices, df_plot['Temperatura (°C)'], marker='o', linestyle='-', color='darkgreen')
    axes[2].set_title('Temperatura (°C)')
    axes[2].set_ylabel('Temperatura (°C)')
    axes[2].set_xticks(x_indices, x_labels)
    axes[2].grid(axis='y', linestyle='--')
    
    # Adiciona rótulos de valor
    for i, v in enumerate(df_plot['Temperatura (°C)']):
        axes[2].text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
    
    # -----------------------------------------------------
    
    plt.tight_layout()
    
    # Chamada à função de salvamento centralizada (salva e exibe)
    salvar_e_exibir_plot("plot_recirculacao_mixer.png", mostrar_grafico)