# plots_modulos/plot_agua_removida_total.py

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from plot_reporter_base import salvar_e_exibir_plot

# CORREÇÃO: Adicionando 'mostrar_grafico' como 6º argumento posicional
def plot_agua_removida_total(df_h2: pd.DataFrame, df_o2: pd.DataFrame, deoxo_mode: str, L_deoxo: float, dc2_mode: str, mostrar_grafico: bool = False):
    """Gera gráfico da quantidade total de água retirada por componente."""

    # LISTA DE COMPONENTES REMOVEDORES CORRIGIDA (Instrução do usuário: Incluir PSA)
    comp_remover = ['KOD 1', 'KOD 2', 'Coalescedor 1', 'VSA', 'PSA'] 
    
    # Inicializa as remoções: Conversão de kg/s para kg/h
    remocao_h2 = {comp: df_h2[df_h2['Componente'] == comp]['Agua_Condensada_kg_s'].iloc[0] * 3600 for comp in comp_remover if comp in df_h2['Componente'].values}
    # O fluxo de O2 não tem PSA/VSA, então filtramos apenas os componentes presentes no df_o2
    remocao_o2 = {comp: df_o2[df_o2['Componente'] == comp]['Agua_Condensada_kg_s'].iloc[0] * 3600 for comp in comp_remover if comp in df_o2['Componente'].values}
    
    # Garante que os componentes do eixo X sejam os que apareceram em H2 ou O2
    componentes_x = [c for c in comp_remover if c in df_h2['Componente'].values or c in df_o2['Componente'].values]
    
    dados_h2 = [remocao_h2.get(c, 0) for c in componentes_x]
    dados_o2 = [remocao_o2.get(c, 0) for c in componentes_x] 
    
    df_plot = pd.DataFrame({
        'H2': dados_h2,
        'O2': dados_o2
    }, index=componentes_x)
    
    componentes_x = df_plot.index.to_list()

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle(f'Vazão de Água Líquida Removida por Componente', y=1.0)
    
    bar_width = 0.35
    r1 = np.arange(len(componentes_x))
    r2 = [x + bar_width for x in r1]
    
    barras_h2 = ax.bar(r1, df_plot['H2'], color='blue', width=bar_width, edgecolor='black', label='Fluxo de H₂')
    barras_o2 = ax.bar(r2, df_plot['O2'], color='red', width=bar_width, edgecolor='black', label='Fluxo de O₂')

    ax.set_title(f'Vazão de Água Líquida Removida por Componente')
    ax.set_ylabel('Vazão Mássica de Água Líquida (kg/h)')
    ax.set_xticks([r + bar_width/2 for r in r1])
    ax.set_xticklabels(componentes_x)
    ax.grid(axis='y', linestyle='--')
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0))
    
    # Rótulos (Mantive a lógica original de posicionamento que está OK para este gráfico)
    for bar in barras_h2:
        yval = bar.get_height()
        if yval > 0.000001:
             ax.text(bar.get_x() + bar.get_width()/2, yval + 0.000000001, f"{yval:.2f}", ha='center', va='bottom', fontsize=8) 
    
    for bar in barras_o2:
        yval = bar.get_height()
        if yval > 0.000001:
             ax.text(bar.get_x() + bar.get_width()/2, yval + 0.000000001, f"{yval:.2f}", ha='center', va='bottom', fontsize=8)

    plt.tight_layout(rect=[0, 0, 1.0, 0.98]) 
    
    # SALVAMENTO CENTRALIZADO
    nome_arquivo = 'plot_agua_removida_total.png'
    salvar_e_exibir_plot(nome_arquivo, mostrar_grafico)