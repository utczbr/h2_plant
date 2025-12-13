# plot_concentracao_linha_dreno.py
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


def plot_concentracao_linha_dreno(drenos_plot_data: dict, mostrar_grafico: bool):
    """
    Gera um gráfico empilhado rastreando a concentração de H2 e O2 dissolvido (ppm)
    ao longo dos 3 pontos do processo de drenagem: Agregação, Pós-Válvula, e Tank + Vent OUT.
    """
    
    H2_IN = drenos_plot_data.get('H2_IN', {'C_diss_mg_kg': 0.0})
    H2_OUT = drenos_plot_data.get('H2_OUT', {'C_diss_mg_kg': 0.0})
    O2_IN = drenos_plot_data.get('O2_IN', {'C_diss_mg_kg': 0.0})
    O2_OUT = drenos_plot_data.get('O2_OUT', {'C_diss_mg_kg': 0.0})
    
    # Concentrações Pós-Válvula: A concentração dissolvida não muda no estrangulamento.
    C_H2_IN = H2_IN['C_diss_mg_kg']
    C_O2_IN = O2_IN['C_diss_mg_kg']
    C_H2_OUT_TV = H2_OUT['C_diss_mg_kg'] # C pós Tank+Vent
    C_O2_OUT_TV = O2_OUT['C_diss_mg_kg'] # C pós Tank+Vent
    
    
    # 1. Preparar os dados para plotagem em linha (3 pontos)
    data = {
        'Ponto': ['Agregação IN', 'Pós-Válvula', 'Tank + Vent OUT'],
        'H2 Dissolvido (ppm)': [C_H2_IN, C_H2_IN, C_H2_OUT_TV],
        'O2 Dissolvido (ppm)': [C_O2_IN, C_O2_IN, C_O2_OUT_TV],
    }
    df_plot = pd.DataFrame(data).set_index('Ponto')

    x_indices = np.arange(len(df_plot))

    # 2. Configuração da Figura
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plotagem H2
    ax.plot(x_indices, df_plot['H2 Dissolvido (ppm)'], marker='o', linestyle='-', 
            color='darkred', linewidth=2, label='H₂ Dissolvido (ppm)')
            
    # Plotagem O2
    ax.plot(x_indices, df_plot['O2 Dissolvido (ppm)'], marker='s', linestyle='--', 
            color='darkblue', linewidth=2, label='O₂ Dissolvido (ppm)')
    
    # Rótulos de valor para H2 e O2
    def formatar_valor_significativo(val):
        """Formata o valor para exibir casas significativas ou notação científica."""
        if val > 1e-4: 
            return f'{val:.4f}'
        if val > 1e-6:
            return f'{val:.2e}'
        return f'{val:.4f}'
        
    for i, row in df_plot.iterrows():
        x_pos = x_indices[df_plot.index.get_loc(i)]
        
        # Rótulo H2
        ax.text(x_pos, row['H2 Dissolvido (ppm)'] * 1.1, 
                formatar_valor_significativo(row['H2 Dissolvido (ppm)']), 
                ha='center', va='bottom', fontsize=8, color='darkred')
        # Rótulo O2
        ax.text(x_pos, row['O2 Dissolvido (ppm)'] * 0.9, 
                formatar_valor_significativo(row['O2 Dissolvido (ppm)']), 
                ha='center', va='top', fontsize=8, color='darkblue')

    # Configurações do Gráfico
    ax.set_ylabel('Concentração Mássica Dissolvida (ppm)')
    ax.set_title('Rastreamento de Gases Dissolvidos (H₂ e O₂) na Linha de Dreno', fontsize=14, fontweight='bold')
    ax.set_xticks(x_indices)
    ax.set_xticklabels(df_plot.index, rotation=15, ha='right')
    ax.legend(loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Ajuste do limite Y 
    y_max = max(df_plot['H2 Dissolvido (ppm)'].max(), df_plot['O2 Dissolvido (ppm)'].max())
    ax.set_ylim(0, max(y_max * 1.5, 0.075)) 

    plt.tight_layout()

    # 3. Salva e Exibe
    salvar_e_exibir_plot('drenos_concentracao_linha_processo_ppm.png', mostrar_grafico)