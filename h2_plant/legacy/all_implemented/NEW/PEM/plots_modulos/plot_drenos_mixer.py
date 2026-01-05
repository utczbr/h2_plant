# plots_modulos/plot_drenos_mixer.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import os

# Importa a função central de salvamento/exibição
try:
    from plot_reporter_base import salvar_e_exibir_plot 
except ImportError:
    print("AVISO: Dependência 'salvar_e_exibir_plot' não encontrada.")
    def salvar_e_exibir_plot(nome, mostrar): pass


def plot_propriedades_dreno(drenos_plot_data: dict, mostrar_grafico: bool):
    """
    Gera um gráfico das propriedades empilhadas (T, P, m_dot, h) para 
    os fluxos de dreno H2 e O2 antes (IN) e depois (OUT) do Flash Drum,
    e a saída final do Mixer.

    Args:
        drenos_plot_data (dict): Dicionário contendo H2_IN, H2_OUT, O2_IN, O2_OUT e FINAL_MIXER.
        mostrar_grafico (bool): Se deve exibir o gráfico.
    """
    
    H2_IN = drenos_plot_data.get('H2_IN')
    H2_OUT = drenos_plot_data.get('H2_OUT')
    O2_IN = drenos_plot_data.get('O2_IN')
    O2_OUT = drenos_plot_data.get('O2_OUT')
    FINAL_MIXER = drenos_plot_data.get('FINAL_MIXER')
    P_OUT_BAR = drenos_plot_data.get('P_OUT_BAR', 4.0)

    # 1. Preparar os dados para o DataFrame
    data_list = []
    labels = []

    if H2_IN:
        data_list.append(H2_IN)
        labels.append('Dreno H₂ IN (Agregado)')
    if H2_OUT:
        data_list.append(H2_OUT)
        labels.append(f'Dreno H₂ OUT ({P_OUT_BAR:.1f} bar)')
    if O2_IN:
        data_list.append(O2_IN)
        labels.append('Dreno O₂ IN (Agregado)')
    if O2_OUT:
        data_list.append(O2_OUT)
        labels.append(f'Dreno O₂ OUT ({P_OUT_BAR:.1f} bar)')
    if FINAL_MIXER:
        # Extrai os dados do Mixer (índice 3, assumindo 2 entradas)
        m_dot_out = FINAL_MIXER.get('Vazão Mássica de Saída (kg/s) (m_dot_3)', np.nan)
        T_out = FINAL_MIXER.get('Temperatura de Saída (°C) (T_3)', np.nan)
        h_out = FINAL_MIXER.get('Entalpia Específica de Saída (kJ/kg) (h_3)', np.nan)
        P_out = FINAL_MIXER.get('Pressão de Saída (kPa) (P_3)', np.nan) / 100
        
        data_list.append({
            'm_dot': m_dot_out,
            'T': T_out,
            'P_bar': P_out,
            'h_kJ_kg': h_out,
            'C_diss_mg_kg': 0.0 # N/A para este plot
        })
        labels.append('Saída Mixer Final')

    df_plot = pd.DataFrame(data_list, index=labels)
    df_plot['m_dot_kg_h'] = df_plot['m_dot'] * 3600

    # 2. Configuração da Figura
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    fig.suptitle(f'Propriedades da Linha de Drenos: Válvula $\\rightarrow$ Flash Drum $\\rightarrow$ Mixer', fontsize=14, fontweight='bold')
    x = np.arange(len(df_plot))

    # --- Plot 1: Temperatura ---
    ax = axes[0]
    ax.bar(x, df_plot['T'], width=0.6, color=['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c'])
    ax.set_ylabel('Temperatura (°C)')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # --- Plot 2: Pressão ---
    ax = axes[1]
    ax.bar(x, df_plot['P_bar'], width=0.6, color=['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c'])
    ax.set_ylabel('Pressão (bar)')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # --- Plot 3: Vazão Mássica ---
    ax = axes[2]
    ax.bar(x, df_plot['m_dot_kg_h'], width=0.6, color=['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c'])
    ax.set_ylabel('Vazão Mássica (kg/h)')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # --- Plot 4: Entalpia Específica ---
    ax = axes[3]
    ax.bar(x, df_plot['h_kJ_kg'], width=0.6, color=['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c'])
    ax.set_ylabel('Entalpia Específica (kJ/kg)')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Configurações do eixo X
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(df_plot.index, rotation=15, ha='right', fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Ajusta o layout para o suptitle

    # 3. Salva e Exibe
    salvar_e_exibir_plot('drenos_propriedades_empilhadas.png', mostrar_grafico)


def plot_concentracao_dreno(*args, **kwargs):
    """Stub para garantir compatibilidade reversa. O novo plot de concentração é gerado em plot_concentracao_dreno.py."""
    pass