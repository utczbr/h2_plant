# plots_modulos/plot_vazao_massica_total_e_removida.py

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from plot_reporter_base import calcular_vazao_massica_total_completa, salvar_e_exibir_plot

# CORREÇÃO: Adicionando 'mostrar_grafico' como 6º argumento posicional
def plot_vazao_massica_total_e_removida(df: pd.DataFrame, gas_fluido: str, deoxo_mode: str, L_deoxo: float, dc2_mode: str, mostrar_grafico: bool = False):
    """
    Gera gráfico comparativo das vazões mássicas do Gás Principal, Vapor de H2O, 
    Água Líquida Acompanhante, e a Vazão TOTAL COMPLETA.
    Tudo em kg/h com rótulos de valor.
    """
    
    df_plot = df.copy()
    
    # === CÁLCULO DA NOVA VAZÃO TOTAL ===
    df_plot['m_dot_total_comp_kg_s'] = calcular_vazao_massica_total_completa(df_plot)
    
    # Converte tudo para kg/h
    df_plot['m_dot_gas_princ_kg_h'] = df_plot['m_dot_gas_kg_s'] * 3600
    df_plot['m_dot_H2O_vap_kg_h'] = df_plot['m_dot_H2O_vap_kg_s'] * 3600
    df_plot['m_dot_liq_accomp_kg_h'] = df_plot['m_dot_H2O_liq_accomp_kg_s'] * 3600 
    df_plot['m_dot_total_comp_kg_h'] = df_plot['m_dot_total_comp_kg_s'] * 3600 
    
    x_labels = df_plot['Componente']
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(f'Rastreamento das Vazões Mássicas do Fluxo ({gas_fluido})', y=1.0)

    # Curva 1: Vazão Mássica TOTAL COMPLETA (NOVA)
    line_total, = ax.plot(x_labels, df_plot['m_dot_total_comp_kg_h'], marker='o', linestyle='-', color='purple', label='Vazão Mássica TOTAL COMPLETA (Gás+Vapor+Líq) (kg/h)')
    
    # Curva 2: Vazão Mássica do Gás Principal
    line_gas, = ax.plot(x_labels, df_plot['m_dot_gas_princ_kg_h'], marker='s', linestyle='--', color='blue', label=f'Vazão Mássica {gas_fluido} Principal (kg/h)')

    # Curva 3: Vazão Mássica do Vapor de H2O
    line_vap, = ax.plot(x_labels, df_plot['m_dot_H2O_vap_kg_h'], marker='^', linestyle=':', color='red', label='Vazão Mássica H₂O Vapor (kg/h)')
    
    # Curva 4: Vazão Mássica de ÁGUA LÍQUIDA ACOMPANHANTE (NOVA)
    line_liq, = ax.plot(x_labels, df_plot['m_dot_liq_accomp_kg_h'], marker='d', linestyle='-', color='brown', label='Vazão Mássica H₂O Líquida Acomp. (kg/h)')


    ax.set_ylabel('Vazão Mássica (kg/h)')
    ax.set_xlabel('Componente')
    ax.grid(True, linestyle='--')
    ax.tick_params(axis='x', rotation=15)

    # Coeficiente de ajuste vertical (em unidades Y do gráfico)
    # AJUSTE PARA MAIOR ESPAÇAMENTO (Instrução do usuário)
    max_val = df_plot['m_dot_total_comp_kg_h'].max()
    OFFSET_PEQUENO = max(1.0, max_val * 0.01) # Aumentado de 0.5 para 1.0 ou 1%
    OFFSET_GRANDE = max(4.0, max_val * 0.04) # Aumentado de 2.0 para 4.0 ou 4%
    
    
    def formatar_rotulo(y):
        """Formata o valor para 2 casas decimais ou notação científica."""
        if y >= 1.0:
            return f'{y:.2f}'
        elif y > 0 and y < 0.01:
            return f'{y:.2e}'
        else: 
            return f'{y:.2f}' if y >= 0 else '0.00'

    # 1. Rótulos da Vazão TOTAL COMPLETA (Topo, Púrpura)
    for x, y in zip(range(len(x_labels)), df_plot['m_dot_total_comp_kg_h']):
        label = formatar_rotulo(y)
        ax.text(x, y + OFFSET_PEQUENO, label, ha='center', va='bottom', fontsize=8, color='purple')
            
    # 2. Rótulos do Gás Principal (Abaixo, Azul)
    for x, y in zip(range(len(x_labels)), df_plot['m_dot_gas_princ_kg_h']):
        label = formatar_rotulo(y)
        ax.text(x, y - OFFSET_PEQUENO, label, ha='center', va='top', fontsize=8, color='blue')

    # 3. Rótulos do Vapor (Vermelho)
    for x, y in zip(range(len(x_labels)), df_plot['m_dot_H2O_vap_kg_h']):
        label = formatar_rotulo(y)
        ax.text(x, y + OFFSET_GRANDE, label, ha='center', va='bottom', fontsize=8, color='red')

    # 4. Rótulos da Água Líquida Acompanhante (Marrom)
    # Corrigindo o posicionamento para evitar colisão com o rótulo do Vapor (se Vapor for 0.00)
    for x, y in zip(range(len(x_labels)), df_plot['m_dot_liq_accomp_kg_h']):
        label = formatar_rotulo(y)
        y_vap = df_plot['m_dot_H2O_vap_kg_h'].iloc[x]
        # Se o vapor for muito baixo, posiciona para cima
        if y_vap < OFFSET_GRANDE:
             ax.text(x, y + OFFSET_PEQUENO, label, ha='center', va='bottom', fontsize=8, color='brown')
        else: # Senão, posiciona para baixo
             ax.text(x, y - (OFFSET_PEQUENO * 0.8), label, ha='center', va='top', fontsize=8, color='brown')


    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    
    # 💥 CORREÇÃO PRINCIPAL: Garante que todos os rótulos X sejam mostrados
    plt.xticks(range(len(x_labels)), x_labels, rotation=15)
    
    plt.tight_layout(rect=[0, 0, 0.88, 0.95])
    
    # SALVAMENTO CENTRALIZADO
    nome_arquivo = f'plot_vazao_massica_total_e_removida_{gas_fluido}.png'
    salvar_e_exibir_plot(nome_arquivo, mostrar_grafico)