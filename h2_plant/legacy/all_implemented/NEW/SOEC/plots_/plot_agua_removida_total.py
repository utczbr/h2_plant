# plots_modulos/plot_agua_removida_total.py

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from plot_reporter_base import salvar_e_exibir_plot

# CORREÇÃO: Adicionando 'mostrar_grafico' como 6º argumento posicional
def plot_agua_removida_total(df_h2: pd.DataFrame, df_o2: pd.DataFrame, deoxo_mode: str, L_deoxo: float, dc2_mode: str, mostrar_grafico: bool = False):
    """Gera gráfico da quantidade total de água retirada por componente (DRENADA/REMOVIDA)."""

    # LISTA CORRIGIDA: Foca nos componentes que REMOVEM o líquido do fluxo (drenos/pool)
    comp_remover = ['KOD 1', 'KOD 2', 'Coalescedor 1', 'PSA', 'VSA'] 
    
    # Função para extrair o valor da água REMOVIDA (drenada) e garantir que é um escalar
    def get_remocao_value(df, comp):
        # Localiza a linha e extrai o valor de Agua_Pura_Removida_H2O_kg_s
        if comp in df['Componente'].values:
            # 🛑 MUDANÇA CRÍTICA: Usando 'Agua_Pura_Removida_H2O_kg_s' para água drenada.
            val = df[df['Componente'] == comp]['Agua_Pura_Removida_H2O_kg_s'].iloc[0] * 3600
            return float(val) # Garante que seja um escalar float
        return 0.0

    remocao_h2 = {comp: get_remocao_value(df_h2, comp) for comp in comp_remover}
    remocao_o2 = {comp: get_remocao_value(df_o2, comp) for comp in comp_remover}
    
    # Garante que os componentes do eixo X sejam os que apareceram em H2 ou O2
    componentes_x = [c for c in comp_remover if remocao_h2[c] > 0 or remocao_o2[c] > 0]
    
    dados_h2 = [remocao_h2.get(c, 0) for c in componentes_x]
    dados_o2 = [remocao_o2.get(c, 0) for c in componentes_x] 
    
    if not componentes_x:
         print("AVISO: Nenhuma remoção líquida relevante para plotar em plot_agua_removida_total (KODs/Coalescedor/PSA/VSA).")
         return
         
    df_plot = pd.DataFrame({
        'H2': dados_h2,
        'O2': dados_o2
    }, index=componentes_x)
    
    componentes_x = df_plot.index.to_list()

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle(f'Vazão de Água Líquida DRENADA por Componente', y=1.0)
    
    bar_width = 0.35
    r1 = np.arange(len(componentes_x))
    r2 = [x + bar_width for x in r1]
    
    barras_h2 = ax.bar(r1, df_plot['H2'], color='blue', width=bar_width, edgecolor='black', label='Fluxo de H₂')
    barras_o2 = ax.bar(r2, df_plot['O2'], color='red', width=bar_width, edgecolor='black', label='Fluxo de O₂')

    ax.set_title(f'Vazão de Água Líquida DRENADA por Componente')
    ax.set_ylabel('Vazão Mássica de Água Líquida (kg/h)')
    ax.set_xticks([r + bar_width/2 for r in r1])
    ax.set_xticklabels(componentes_x)
    ax.grid(axis='y', linestyle='--')
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0))
    
    # Rótulos (CORRIGIDO PARA EVITAR AMBIGUIDADE)
    def autolabel_bars(bars):
        for bar in bars:
            yval = bar.get_height()
            # Usamos uma checagem simples de valor escalar
            if yval > 0.000001: 
                 ax.text(bar.get_x() + bar.get_width()/2, yval, f"{yval:.2f}", ha='center', va='bottom', fontsize=8) 
    
    autolabel_bars(barras_h2)
    autolabel_bars(barras_o2)

    plt.tight_layout(rect=[0, 0, 1.0, 0.98]) 
    
    # SALVAMENTO CENTRALIZADO
    nome_arquivo = 'plot_agua_removida_total.png'
    salvar_e_exibir_plot(nome_arquivo, mostrar_grafico)