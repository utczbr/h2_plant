# plots_modulos/plot_vazao_agua_separada.py

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import CoolProp.CoolProp as CP
from constants_and_config import Y_H2O_LIMIT_MOLAR, M_DOT_G_H2, M_DOT_G_O2 
from plot_reporter_base import salvar_e_exibir_plot

# CORREÇÃO: Adicionando 'mostrar_grafico' como 6º argumento posicional
def plot_vazao_agua_separada(df: pd.DataFrame, gas_fluido: str, deoxo_mode: str, L_deoxo: float, dc2_mode: str, mostrar_grafico: bool = False):
    """Gera gráfico da vazão mássica de vapor de água e adiciona o limite em PPM molar."""
    
    df_plot = df.copy()
        
    fig, ax = plt.subplots(figsize=(10, 5))
    x_labels = df_plot['Componente']
    fig.suptitle(f'Rastreamento da Vazão de Vapor de Água ({gas_fluido}) (Rótulos em Concentração Molar PPM)', y=1.0)
    
    # 1. Vazão Mássica de H2O na Fase Gasosa (Linha)
    # CONVERSÃO: kg/s para kg/h
    y_data_kg_h = df_plot['m_dot_H2O_vap_kg_s'] * 3600
    line, = ax.plot(x_labels, y_data_kg_h, 
            marker='o', linestyle='-', label=f'Vazão Mássica de H₂O na Fase Gasosa (kg/h)',
            color='blue' if gas_fluido == 'H2' else 'red')

    # 2. Adiciona o percentual de água molar (y_H2O) em PPM como rótulo sobre a linha 
    y_h2o_ppm = df_plot['y_H2O'] * 1e6 
    
    for i, txt in enumerate(y_h2o_ppm):
        if txt > 1.0: 
             label = f'{txt:.2f}' 
        elif txt > 0.0:
            label = f'{txt:.2e}' 
        else:
             continue
             
        # Usa y_data_kg_h
        ax.text(x_labels.iloc[i], y_data_kg_h.iloc[i] + 0.05 * y_data_kg_h.max(), 
                     label, 
                     ha='center', va='bottom', fontsize=8, color=line.get_color())


    # 3. Adiciona o Limite Molar de 100 PPM (y_H2O_LIMIT_MOLAR)
    
    if gas_fluido == 'H2':
        comp_limite = 'VSA' 
        m_dot_gas_princ_entrada = M_DOT_G_H2 
    else:
        comp_limite = 'Secador Adsorvente'
        m_dot_gas_princ_entrada = M_DOT_G_O2

    M_H2O = CP.PropsSI('M', 'Water')
    M_GAS_PRINCIPAL = CP.PropsSI('M', gas_fluido)
    y_H2O_limite_molar = Y_H2O_LIMIT_MOLAR
    
    F_gas_molar_entrada = m_dot_gas_princ_entrada / M_GAS_PRINCIPAL
    
    F_H2O_molar_limite = F_gas_molar_entrada * (y_H2O_limite_molar / (1.0 - y_H2O_limite_molar))
    m_dot_h2o_limite_kg_h = F_H2O_molar_limite * M_H2O * 3600 # kg/h
    
    # Plota a linha de limite
    ax.axhline(m_dot_h2o_limite_kg_h, color='green', linestyle='--', 
               label=f'Limite de Saída {comp_limite} ({y_H2O_limite_molar*1e6:.0f} ppm molar)')
        
    ax.set_ylabel('Vazão Mássica de H₂O na Fase Gasosa (kg/h)')
    ax.set_xlabel('Componente')
    ax.grid(True, linestyle='--')
    
    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize='small')
    
    plt.tight_layout(rect=[0, 0, 0.88, 0.95])
    
    # SALVAMENTO CENTRALIZADO
    nome_arquivo = f'plot_vazao_agua_separada_{gas_fluido}.png'
    salvar_e_exibir_plot(nome_arquivo, mostrar_grafico)