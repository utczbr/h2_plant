# plots_modulos/plot_drenos_individuais.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import os
import matplotlib.ticker as ticker 

# Importa a função central de salvamento/exibição e constantes
try:
    from plot_reporter_base import salvar_e_exibir_plot 
    from aux_coolprop import calcular_solubilidade_gas_henry 
    from constants_and_config import Y_O2_IN_H2, Y_H2_IN_O2, P_DRENO_OUT_BAR 
    # Importação das novas constantes de total de água líquida gerada no PEM
    from constants_and_config import M_DOT_H2O_LIQ_IN_H2_TOTAL_KGS, M_DOT_H2O_LIQ_IN_O2_TOTAL_KGS
    
except ImportError:
    print("AVISO: Dependências não encontradas.")
    def salvar_e_exibir_plot(nome, mostrar): pass
    def calcular_solubilidade_gas_henry(*args): return 0.0
    Y_O2_IN_H2 = 0.0002
    Y_H2_IN_O2 = 0.004
    P_DRENO_OUT_BAR = 4.0 
    M_DOT_H2O_LIQ_IN_H2_TOTAL_KGS = 0.013844
    M_DOT_H2O_LIQ_IN_O2_TOTAL_KGS = 0.069222

CONC_MAX_H2_PERIGO_MG_KG = 20.0 # Limite de segurança

def calcular_conc_crossover(dreno_list, gas_fluido):
    """Calcula a concentração do gás crossover (impureza) para cada dreno."""
    crossover_list = []
    
    is_h2_flow = (gas_fluido == 'H2')
    gas_crossover = 'O2' if is_h2_flow else 'H2'
    y_crossover = Y_O2_IN_H2 if is_h2_flow else Y_H2_IN_O2
    
    for dreno in dreno_list:
        T_C = dreno['T']
        P_bar = dreno['P_bar']
        
        # Assumimos que o dreno em equilíbrio com a fase gás tem a pressão parcial do crossover
        conc_crossover = calcular_solubilidade_gas_henry(gas_crossover, T_C, P_bar, y_crossover)
        crossover_list.append(conc_crossover)
        
    return crossover_list


def plot_drenos_individuais(drenos_h2_raw: list, drenos_o2_raw: list, mostrar_grafico: bool):
    """
    Gera dois gráficos separados (H2 e O2) das propriedades dos drenos brutos, usando barras.
    O primeiro subplot utiliza um eixo Y secundário para visualizar as pequenas vazões.
    """
    
    if not drenos_h2_raw and not drenos_o2_raw:
        print("AVISO: Dados de drenos brutos vazios para plotagem.")
        return

    # --- 1. PREPARAÇÃO DOS DADOS H2 ---
    data_h2 = []
    crossover_h2 = calcular_conc_crossover(drenos_h2_raw, 'H2')
    for i, d in enumerate(drenos_h2_raw):
        data_h2.append({
            'Componente': d['Componente'],
            'Vazão (kg/h)': d['m_dot'] * 3600,
            'T (°C)': d['T'],
            'P (bar)': d['P_bar'],
            'C_dominante': d['Gas_Dissolvido_in_mg_kg'], # H2
            'C_crossover': crossover_h2[i], # O2
        })
    df_h2 = pd.DataFrame(data_h2).set_index('Componente')
    
    # --- 2. PREPARAÇÃO DOS DADOS O2 ---
    data_o2 = []
    crossover_o2 = calcular_conc_crossover(drenos_o2_raw, 'O2')
    for i, d in enumerate(drenos_o2_raw):
        data_o2.append({
            'Componente': d['Componente'],
            'Vazão (kg/h)': d['m_dot'] * 3600,
            'T (°C)': d['T'],
            'P (bar)': d['P_bar'],
            'C_dominante': d['Gas_Dissolvido_in_mg_kg'], # O2
            'C_crossover': crossover_o2[i], # H2
        })
    df_o2 = pd.DataFrame(data_o2).set_index('Componente')


    def gerar_plot_individual(df_plot: pd.DataFrame, gas_fluido: str):
        """Função auxiliar para gerar um único gráfico empilhado de barras com eixo Y secundário."""
        
        fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
        fig.suptitle(f'Propriedades dos Drenos Individuais - Fluxo de {gas_fluido}', fontsize=14, fontweight='bold')
        x = np.arange(len(df_plot))
        
        # Cores
        cor_principal = 'darkred' if gas_fluido == 'H₂' else 'darkblue'
        cor_secundaria = 'maroon' if gas_fluido == 'H₂' else 'navy'
        
        # --- Plot 1: Vazão Mássica (Eixo Y Principal e Secundário) ---
        ax1 = axes[0]
        
        # Classifica as vazões: Dreno PEM Recirc. (principal) vs Drenos de Processo (pequeno)
        # Assumindo que a primeira linha é o Dreno Recirc.
        if df_plot.empty:
            return 
            
        Vazao_Principal = df_plot.iloc[0]['Vazão (kg/h)']
        Vazoes_Pequenas = df_plot.iloc[1:]['Vazão (kg/h)']
        
        # Eixo Y principal para Vazões grandes (usando um marcador discreto)
        bars_principal = ax1.bar(x[0], Vazao_Principal, width=0.8, color=cor_principal, label='Vazão Recirculação (Eixo Esquerdo)')
        ax1.set_ylabel('Vazão Recirc. (kg/h)', color=cor_principal)
        ax1.tick_params(axis='y', labelcolor=cor_principal)
        ax1.set_ylim(0, Vazao_Principal * 1.1)
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        ax1.text(x[0], Vazao_Principal, f'{Vazao_Principal:.2f}', ha='center', va='bottom', fontsize=8)
        
        # Eixo Y secundário para Vazões Pequenas (usando barras)
        ax2 = ax1.twinx()
        bars_pequenas = ax2.bar(x[1:], Vazoes_Pequenas, width=0.8, color=cor_secundaria, alpha=0.7, label='Drenos de Processo (Eixo Direito)')
        ax2.set_ylabel('Vazão Proc. (kg/h)', color=cor_secundaria)
        ax2.tick_params(axis='y', labelcolor=cor_secundaria)
        
        max_pequenas = Vazoes_Pequenas.max() if Vazoes_Pequenas.size > 0 else 10
        ax2.set_ylim(0, max_pequenas * 1.5 if max_pequenas > 0 else 10) # Escala ampliada
        
        # Rótulos para as vazões pequenas
        for bar in bars_pequenas:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)
            
        # Unir legendas
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=8)


        # --- Plot 2: Temperatura ---
        ax = axes[1]
        bars = ax.bar(x, df_plot['T (°C)'], width=0.8, color=cor_secundaria)
        ax.set_ylabel('Temperatura (°C)')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)

        # --- Plot 3: Pressão ---
        ax = axes[2]
        bars = ax.bar(x, df_plot['P (bar)'], width=0.8, color=cor_principal)
        ax.set_ylabel('Pressão (bar)')
        ax.set_ylim(bottom=P_DRENO_OUT_BAR * 0.9, top=df_plot['P (bar)'].max() * 1.05) 
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)

        # --- Plot 4: Concentração de Gás Dissolvido ---
        ax = axes[3]
        bars_dom = ax.bar(x - 0.2, df_plot['C_dominante'], width=0.4, color=cor_principal, label=f'{gas_fluido} Dissolvido')
        
        gas_cross_label = 'O₂ Dissolvido' if gas_fluido == 'H₂' else 'H₂ Dissolvido'
        bars_cross = ax.bar(x + 0.2, df_plot['C_crossover'], width=0.4, color='gray', alpha=0.6, label=gas_cross_label)
        
        ax.set_ylabel('Conc. Dissolvida (mg/kg)')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Linha de segurança (apenas para H2)
        if gas_fluido == 'H₂':
            ax.axhline(CONC_MAX_H2_PERIGO_MG_KG, color='red', linestyle='--', label=f'Risco H₂ ({CONC_MAX_H2_PERIGO_MG_KG:.1f} mg/kg)')
            ax.legend(loc='upper right', fontsize=8)
        else:
             ax.legend(loc='upper right', fontsize=8)
        
        # Rótulos de Concentração
        for bars in [bars_dom, bars_cross]:
            for bar in bars:
                yval = bar.get_height()
                if yval > 0.01:
                    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f"{yval:.2f}", ha='center', va='bottom', fontsize=7)

        # Configurações do eixo X
        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels(df_plot.index, rotation=30, ha='right', fontsize=9)
        axes[-1].set_xlabel('Componente Drenado')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96]) 
        
        return fig

    # Gera e salva o gráfico H2
    if not df_h2.empty:
        fig_h2 = gerar_plot_individual(df_h2, 'H₂')
        salvar_e_exibir_plot('drenos_individuais_h2.png', mostrar_grafico)
    
    # Gera e salva o gráfico O2
    if not df_o2.empty:
        fig_o2 = gerar_plot_individual(df_o2, 'O₂')
        salvar_e_exibir_plot('drenos_individuais_o2.png', mostrar_grafico)