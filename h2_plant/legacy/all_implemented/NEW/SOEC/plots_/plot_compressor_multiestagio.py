# plots_modulos/plot_compressor_multiestagio.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Importa a função de salvamento centralizada
try:
    from plot_reporter_base import salvar_e_exibir_plot 
except ImportError:
    # Fallback/Stub
    def salvar_e_exibir_plot(nome_arquivo, mostrar_grafico):
        print(f"AVISO: Plotagem '{nome_arquivo}' apenas salva no diretório de saída padrão.")
        plt.savefig(os.path.join(os.getcwd(), 'Graficos', nome_arquivo))
        plt.close()


def plot_compressor_multiestagio(multistage_history: list, mostrar_grafico: bool = False):
    """
    Gera um gráfico do perfil de Pressão e Temperatura ao longo dos estágios do 
    compressor multiestágio de H2, incluindo o calor removido.
    """
    if not multistage_history:
        print("AVISO: Não há dados de histórico do compressor multiestágio para plotar.")
        return

    df = pd.DataFrame(multistage_history)
    n_stages = len(df)
    stage_labels = [f"Estágio {i}" for i in range(1, n_stages + 1)]

    # 1. Gráfico de Pressão e Temperatura
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Pressão (Eixo Y Esquerdo)
    color = 'tab:blue'
    ax1.set_xlabel('Estágio de Compressão')
    ax1.set_ylabel('Pressão (bar)', color=color)
    # Ponto de entrada no estágio (P_in)
    ax1.plot(stage_labels, df['P_in_bar'], marker='o', linestyle='--', color='lightblue', label='P Entrada (bar)')
    # Ponto de saída do compressor (P_after_comp) - Pressão final após compressão, mas antes do resfriamento
    ax1.plot(stage_labels, df['P_after_comp_bar'], marker='s', linestyle='-', color=color, label='P Pós-Compressão (bar)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Temperatura (Eixo Y Direito)
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Temperatura (°C)', color=color)
    # T Pós-Compressão (Pico)
    ax2.plot(stage_labels, df['T_after_comp_C'], marker='^', linestyle='-', color=color, label='T Pós-Comp (Pico) (°C)')
    # T Pós-Resfriamento (Entrada do Próximo Estágio)
    ax2.plot(stage_labels, df['T_after_chiller_C'], marker='v', linestyle='-', color='orange', label='T Pós-Resfriamento (Entrada Próx. Estágio) (°C)')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Legendas (Unindo de ambos os eixos)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.title(f'Perfil de Pressão e Temperatura - Compressor H₂ Multiestágio ({n_stages} Estágios)')
    plt.tight_layout()
    salvar_e_exibir_plot('plot_compressor_multiestagio_PT.png', mostrar_grafico)
    
    # 2. Gráfico de Carga Térmica (Calor Removido e Potência de Compressão)
    
    fig, ax3 = plt.subplots(figsize=(10, 6))
    
    # Potência de Compressão (Eixo Y Esquerdo)
    color_w = 'tab:purple'
    ax3.set_xlabel('Estágio de Compressão')
    ax3.set_ylabel('Potência de Compressão (kW)', color=color_w)
    ax3.bar(df['stage'] - 0.2, df['W_comp_W'] / 1000.0, width=0.4, color=color_w, alpha=0.6, label='W Compressão (kW)')
    ax3.tick_params(axis='y', labelcolor=color_w)
    ax3.grid(True, linestyle='--', alpha=0.6)

    # Calor Removido (Eixo Y Direito)
    ax4 = ax3.twinx()
    color_q = 'tab:green'
    ax4.set_ylabel('Calor Removido (kW)', color=color_q)
    # Calor Removido Total (Dry + Chiller)
    Q_removido_total = (df['Q_dry_W'] + df['Q_chiller_W']) / 1000.0
    ax4.bar(df['stage'] + 0.2, Q_removido_total, width=0.4, color=color_q, alpha=0.6, label='Q Removido Total (kW)')
    ax4.tick_params(axis='y', labelcolor=color_q)
    ax4.set_xticks(df['stage'])
    ax4.set_xticklabels(stage_labels)
    
    # Legendas (Unindo de ambos os eixos)
    lines3, labels3 = ax3.get_legend_handles_labels()
    lines4, labels4 = ax4.get_legend_handles_labels()
    ax3.legend(lines3 + lines4, labels3 + labels4, loc='upper left')
    
    plt.title(f'Potência e Remoção de Calor por Estágio - Compressor H₂ Multiestágio')
    plt.tight_layout()
    salvar_e_exibir_plot('plot_compressor_multiestagio_QW.png', mostrar_grafico)

# --- FIM DO NOVO MÓDULO DE PLOTAGEM ---