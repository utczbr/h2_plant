# plots_modulos/plot_esquema_processo.py

import numpy as np 
import matplotlib.pyplot as plt
# IMPORTAÇÃO NECESSÁRIA PARA SALVAMENTO CENTRALIZADO
from plot_reporter_base import salvar_e_exibir_plot 

# A função agora recebe MOSTRAR_GRAFICOS como 3º argumento
def plot_esquema_processo(component_list: list, gas_fluido: str, mostrar_grafico: bool = False):
    """Gera um esquema simples da ordem dos processos com fluxo de massa/energia, com mais espaço."""
    
    fig, ax = plt.subplots(figsize=(20, 6))
    
    espacamento_x = 2.5
    posicoes = {comp: (i * espacamento_x, 0) for i, comp in enumerate(component_list)}
    
    # 1. Desenha os retângulos dos componentes
    for comp, (x, y) in posicoes.items():
        if comp != 'Entrada':
            rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color='lightgray', alpha=0.6, ec='black', lw=1.5, zorder=1)
            ax.add_patch(rect)
            ax.text(x, y, comp, ha='center', va='center', fontsize=10, fontweight='bold', zorder=2)
        else:
            ax.text(x, y, comp, ha='center', va='center', fontsize=10, fontweight='bold', zorder=2)

    # 2. Desenha as setas (Fluxo de Massa)
    fluxo_y = 0.0
    for i in range(len(component_list) - 1):
        comp_i = component_list[i]
        comp_i1 = component_list[i+1]
        x_start = posicoes[comp_i][0] + 0.5
        x_end = posicoes[comp_i1][0] - 0.5
        
        ax.annotate('', xy=(x_end, fluxo_y), xytext=(x_start, fluxo_y),
                    arrowprops=dict(facecolor='blue', shrink=0.05, width=2, headwidth=10))
        ax.text((x_start + x_end) / 2, fluxo_y + 0.45, 'Fluxo de Gás e Vapor', ha='center', va='bottom', fontsize=7, color='blue') 
        
    # 3. Desenha as setas de Energia (Calor/Trabalho)
    
    energia_detalhes = {
        'Chiller 1': [('Q (Resfriamento)', -1.0, -0.6, 'red'), ('W (Elétrico)', 1.0, 0.6, 'green')],
        'Dry Cooler 1': [('Q (Resfriamento)', -1.0, -0.6, 'red'), ('W (Ventilador)', 1.0, 0.6, 'green')],
        'KOD 1': [('M_H2O (Líq/Recirculação)', -1.0, -0.6, 'brown'), ('W (Perda P)', 1.0, 0.6, 'green')],
        'KOD 2': [('M_H2O (Líq/Condensado)', -1.0, -0.6, 'brown'), ('W (Perda P)', 1.0, 0.6, 'green')],
        'Coalescedor 1': [('M_Liq (Residual/Aerossóis)', -1.0, -0.6, 'brown'), ('W (Perda P)', 1.0, 0.6, 'green')],
        'Deoxo': [('Q (Trocado Jaqueta)', -1.0, -0.6, 'red'), ('H₂O (Produto, adicionado ao Líq)', -0.7, -0.3, 'purple'), ('O₂ (Consumido)', 1.0, 0.6, 'orange')],
        'VSA': [('M_H2O (Vapor Removida)', -1.0, -0.6, 'brown'), ('W (Comp./Vácuo)', 1.0, 0.6, 'green'), ('H₂ (Perdido)', -0.7, 0.3, 'orange')],
        'Secador Adsorvente': [('M_H2O (Vapor Removida)', -1.0, -0.6, 'brown'), ('W (Purga/Comp)', 1.0, 0.6, 'green')],
        'Válvula': [('W (Perda P)', 1.0, 0.6, 'green')], 
    }
    
    for comp in component_list:
        if comp in energia_detalhes:
            for rotulo, y_arr, y_text, cor in energia_detalhes[comp]:
                x = posicoes[comp][0]
                
                if gas_fluido == 'O2' and comp == 'Deoxo':
                    continue
                
                offset = 0.5 if 'Líq' in rotulo or 'Produto' in rotulo or 'Calor' in rotulo or 'Reação' in rotulo else -0.5
                
                ax.annotate('', xy=(x + offset, y_arr), xytext=(x + offset, y_text), arrowprops=dict(facecolor=cor, shrink=0.05, width=1, headwidth=5), ha='center', zorder=3)
                ax.text(x + offset, y_text + 0.1, rotulo, ha='center', va='bottom', fontsize=7, color=cor)
        
    ax.set_xlim(-1, posicoes[component_list[-1]][0] + 1.5)
    ax.set_ylim(-1.5, 1.5) 
    ax.axis('off')
    plt.title(f'Esquema do Sistema de Tratamento de Gás ({gas_fluido})', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1.0, 0.98])
    
    # SALVAMENTO CENTRALIZADO
    nome_arquivo = f'plot_esquema_processo_{gas_fluido}.png'
    salvar_e_exibir_plot(nome_arquivo, mostrar_grafico)