# plots_modulos/plot_concentracao_dreno.py
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


def plot_concentracao_dreno(drenos_plot_data: dict, mostrar_grafico: bool):
    """
    Gera um gráfico comparando a Concentração de Gás Dissolvido (H2 e O2) 
    na água antes (IN) e depois (OUT) do Flash Drum, usando dados agregados.
    
    CORREÇÃO: Unidade mudada para ppm. Rótulos formatados para exibir casas significativas.
    """
    
    H2_IN = drenos_plot_data.get('H2_IN', {'C_diss_mg_kg': 0.0})
    H2_OUT = drenos_plot_data.get('H2_OUT', {'C_diss_mg_kg': 0.0})
    O2_IN = drenos_plot_data.get('O2_IN', {'C_diss_mg_kg': 0.0})
    O2_OUT = drenos_plot_data.get('O2_OUT', {'C_diss_mg_kg': 0.0})
    FINAL_MIXER = drenos_plot_data.get('FINAL_MIXER', {'Conc_H2_final_mg_kg': 0.0, 'Conc_O2_final_mg_kg': 0.0})
    
    # Prepara os dados (Apenas H2 e O2)
    gases = ['H₂ Dissolvido', 'O₂ Dissolvido']
    
    # Concentração de Entrada Agregada (IN)
    c_in = [H2_IN['C_diss_mg_kg'], O2_IN['C_diss_mg_kg']]
    
    # Concentração de Saída do Flash Drum (OUT)
    c_out = [H2_OUT['C_diss_mg_kg'], O2_OUT['C_diss_mg_kg']]
    
    # Concentração do Mixer Final (Ponderada)
    c_final_mixer = [
    FINAL_MIXER.get('Conc_H2_final_mg_kg', 0.0),
    FINAL_MIXER.get('Conc_O2_final_mg_kg', 0.0)
    ]


    df_plot = pd.DataFrame({
        'Gás': gases,
        'Concentração IN (ppm)': c_in,
        'Concentração OUT (ppm)': c_out,
        'Concentração Final Mixer (ppm)': c_final_mixer
    }).set_index('Gás')

    # Calcula a eficiência de remoção (para rótulos)
    eficiencia_h2 = (c_in[0] - c_out[0]) / c_in[0] if c_in[0] > 0 else 0
    eficiencia_o2 = (c_in[1] - c_out[1]) / c_in[1] if c_in[1] > 0 else 0
    
    
    # 1. Configuração da Figura
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(gases))
    width = 0.25
    
    # Títulos e Rótulos ajustados para PPM
    rects1 = ax.bar(x - width, df_plot['Concentração IN (ppm)'], width, label='Entrada Agregada (Pré-Flash)', color='#1f77b4')
    rects2 = ax.bar(x, df_plot['Concentração OUT (ppm)'], width, label='Saída do Tank + Vent (Desgaseificado)', color='#ff7f0e')
    rects3 = ax.bar(x + width, df_plot['Concentração Final Mixer (ppm)'], width, label='Saída Mixer Final (Ponderado)', color='#2ca02c')

    # --- AJUSTE CRÍTICO DE ESCALA ---
    max_conc = max(c_in[0], c_in[1])
    # Define o limite Y como o maior valor de entrada + uma margem (ou 0.075, o que for maior)
    y_limit = max(max_conc * 1.5, 0.075) 
    ax.set_ylim(0, y_limit)
    # ----------------------------------------
    
    # Rótulos
    # Eixo Y agora em ppm
    ax.set_ylabel('Concentração Mássica Dissolvida (ppm $\\text{H}_2\text{O}$)')
    ax.set_title('Eficiência de Remoção de Gases Dissolvidos e Concentração Final (ppm)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(gases)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Função de formatação para exibir o primeiro dígito significativo
    def formatar_valor_significativo(val):
        if val > 1e-4: # Acima de 0.0001 (4 casas decimais)
            return f'{val:.4f}'
        if val > 1e-6: # Acima de 0.000001 (notação científica para valores muito pequenos)
            return f'{val:.2e}'
        return f'{val:.2f}' # Default para zero ou valores próximos
    
    # Adicionar rótulos de eficiência (apenas para o Tank + Vent OUT)
    def autolabel_eficiencia(rects, eficiencias):
        for i, rect in enumerate(rects):
            height = rect.get_height()
            eficiencia = eficiencias[i]
            if height > 0:
                 ax.annotate(f'Remoção: {eficiencia:.1%}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, -10), 
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8, color='black')
                        
    autolabel_eficiencia(rects2, [eficiencia_h2, eficiencia_o2])
    
    # Adicionar rótulos de valor para todas as barras (usando formatação significativa)
    def autolabel_valor(rects):
        for rect in rects:
            height = rect.get_height()
            if height >= 0: # Inclui zero para evitar erro
                 label = formatar_valor_significativo(height)
                 ax.annotate(label,
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), 
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8, color='black')
                            
    autolabel_valor(rects1)
    autolabel_valor(rects2)
    autolabel_valor(rects3) # Rótulos de valor para o Mixer

    plt.tight_layout()

    # 2. Salva e Exibe
    # Renomeando o arquivo para refletir a mudança no rótulo do eixo Y (PPM)
    salvar_e_exibir_plot('drenos_concentracao_removida_ppm.png', mostrar_grafico)