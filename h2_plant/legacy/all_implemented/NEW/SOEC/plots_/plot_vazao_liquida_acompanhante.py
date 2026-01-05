# plots_modulos/plot_vazao_liquida_acompanhante.py

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
# IMPORTAÇÃO NECESSÁRIA PARA SALVAMENTO CENTRALIZADO
from plot_reporter_base import salvar_e_exibir_plot 


# A função agora recebe MOSTRAR_GRAFICOS como 6º argumento
def plot_vazao_liquida_acompanhante(df: pd.DataFrame, gas_fluido: str, deoxo_mode: str, L_deoxo: float, dc2_mode: str, mostrar_grafico: bool = False):
    """
    Gera gráfico da Vazão Mássica de Água Líquida Acompanhante (m_dot_H2O_liq_accomp_kg_s)
    em kg/h.
    """
    
    df_plot = df.copy()
    
    # Converte para kg/h
    df_plot['m_dot_liq_accomp_kg_h'] = df_plot['m_dot_H2O_liq_accomp_kg_s'] * 3600
    
    x_labels = df_plot['Componente']
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(f'Rastreamento da Vazão Mássica de Água Líquida Acompanhante ({gas_fluido})', y=1.0)
    
    # Curva da Vazão Líquida Acompanhante
    line, = ax.plot(x_labels, df_plot['m_dot_liq_accomp_kg_h'], marker='d', linestyle='-', color='brown', label='Vazão Mássica H₂O Líquida Acomp. (kg/h)')

    ax.set_ylabel('Vazão Mássica (kg/h)')
    ax.set_xlabel('Componente')
    ax.grid(True, linestyle='--')
    ax.tick_params(axis='x', rotation=15)
    
    # Definição do offset para rótulos
    max_y = df_plot['m_dot_liq_accomp_kg_h'].max()
    OFFSET_Y = max(0.005, max_y * 0.005) # Reduzido para ficar mais próximo do ponto

    # FUNÇÃO DE FORMATAÇÃO CORRIGIDA PARA 1 CASA DECIMAL (Instrução do usuário)
    def formatar_rotulo_liq(y):
        """Formata o valor para 1 casa decimal ou notação científica."""
        if y >= 0.1:
            return f'{y:.1f}' # 1 casa decimal
        elif y > 0 and y < 0.01:
            return f'{y:.2e}' # Notação científica para valores muito pequenos
        else:
            return f'{y:.1f}' # Mantém 1 casa decimal (ex: 0.0, 0.1)

    # Adicionar Rótulos (Posicionados POUCO ACIMA do ponto)
    for x, y in zip(range(len(x_labels)), df_plot['m_dot_liq_accomp_kg_h']):
        label = formatar_rotulo_liq(y)
        # Posiciona o rótulo ligeiramente acima do ponto
        ax.text(x, y + OFFSET_Y, label, ha='center', va='bottom', fontsize=8, color='brown')

    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    
    # 💥 CORREÇÃO PRINCIPAL: Garante que todos os rótulos X sejam mostrados
    plt.xticks(range(len(x_labels)), x_labels, rotation=15)
    
    plt.tight_layout(rect=[0, 0, 0.88, 0.95])
    
    # SALVAMENTO CENTRALIZADO
    nome_arquivo = f'plot_vazao_liquida_acompanhante_{gas_fluido}.png'
    salvar_e_exibir_plot(nome_arquivo, mostrar_grafico)