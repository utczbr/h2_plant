# plot_esquema_drenos.py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from plot_reporter_base import salvar_e_exibir_plot

def plot_esquema_drenos(mostrar_grafico: bool = True):
    """
    Gera um esquema simplificado da linha de tratamento de água de dreno,
    incluindo agregação de drenos (PEM e KODs), redução de pressão e desgaseificação.
    
    CORREÇÃO: Inclui apenas os drenos PEM e KOD 1/2.
    """
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_title('Esquema do Processo de Tratamento de Água de Dreno (PEM e KODs)', fontsize=16, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off') # Remove os eixos

    # --- 1. Definição de Componentes ---
    
    # Componentes do Fluxo H2 (Vermelho - Topo)
    x_h2 = 10; y_h2 = 80
    componentes_h2 = [
        ("PEM Dreno Recirc. (H₂)", x_h2 + 5, y_h2),
        ("KOD 1 (H₂)", x_h2 + 25, y_h2),
        ("KOD 2 (H₂)", x_h2 + 45, y_h2),
    ]
    
    # Componentes do Fluxo O2 (Azul - Base)
    x_o2 = 10; y_o2 = 25
    componentes_o2 = [
        ("PEM Dreno Recirc. (O₂)", x_o2 + 5, y_o2),
        ("KOD 1 (O₂)", x_o2 + 25, y_o2),
        ("KOD 2 (O₂)", x_o2 + 45, y_o2),
    ]

    # Componentes de Processo (Comuns)
    x_valve = 65; y_valve_h2 = 80; y_valve_o2 = 25
    x_flash = 78; y_flash_h2 = 80; y_flash_o2 = 25
    x_mixer = 85; y_mixer = 52.5 # Ponto central
    
    # --- 2. Desenho das Entradas (Drenos Brutos) ---
    def draw_drenos_brutos(ax, comps, color):
        for i, (name, x, y) in enumerate(comps):
            # Círculo/Ponto de Dreno
            ax.plot(x, y, 'o', color=color, markersize=8, zorder=5)
            ax.text(x, y + 3, name, ha='center', fontsize=9, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle="round,pad=0.2"))
            
            # Linha de conexão
            if i > 0:
                prev_x = comps[i-1][1]
                # Linha horizontal
                ax.plot([prev_x + 3, x - 3], [y, y], color=color, linestyle=':', linewidth=1.5, zorder=1)
                # Linha de agregação (Vertical)
                ax.plot([x - 10, x - 10], [y, y + 15], color='gray', linestyle='--', linewidth=1)
                
            # Adicionar setas na junção para indicar o fluxo (pequena seta horizontal)
            ax.arrow(x - 3, y, 2.8, 0, head_width=1.5, head_length=1.5, fc=color, ec=color)


    draw_drenos_brutos(ax, componentes_h2, 'firebrick')
    draw_drenos_brutos(ax, componentes_o2, 'navy')
    
    # Agregação Final - Linha de entrada no Processo (Agregação é virtual)
    x_agregacao_h2 = componentes_h2[-1][1] + 10
    ax.plot([componentes_h2[-1][1], x_agregacao_h2], [componentes_h2[-1][2], componentes_h2[-1][2]], color='firebrick', linewidth=2, zorder=2)
    ax.text(x_agregacao_h2 + 0.5, y_h2 + 1, "Linha H₂ (Agregada)", fontsize=9, color='firebrick')

    x_agregacao_o2 = componentes_o2[-1][1] + 10
    ax.plot([componentes_o2[-1][1], x_agregacao_o2], [componentes_o2[-1][2], componentes_o2[-1][2]], color='navy', linewidth=2, zorder=2)
    ax.text(x_agregacao_o2 + 0.5, y_o2 + 1, "Linha O₂ (Agregada)", fontsize=9, color='navy')
    
    # --- 3. Desenho dos Componentes de Processo ---

    # Válvula (Joule-Thomson / Redução de P)
    def draw_valve(ax, x, y, color):
        # Quadrado para Válvula
        valve_rect = patches.Rectangle((x - 1.5, y - 3), 3, 6, facecolor='lightgray', edgecolor=color, linewidth=2, zorder=3)
        ax.add_patch(valve_rect)
        ax.text(x, y, "Válvula", ha='center', va='center', fontsize=9)
        ax.text(x, y - 5, "P reduzida", ha='center', va='center', fontsize=8)

    # Flash Drum (Desgaseificação)
    def draw_flash(ax, x, y, color):
        # Vaso cilíndrico
        flash_rect = patches.Rectangle((x - 3, y - 6), 6, 12, facecolor='mistyrose' if color == 'firebrick' else 'lightblue', edgecolor=color, linewidth=2, zorder=3)
        ax.add_patch(flash_rect)
        ax.text(x, y + 2, "Flash Drum", ha='center', va='center', fontsize=9)
        ax.text(x, y - 2, "Desgaseificação", ha='center', va='center', fontsize=7)
        
        # Seta de Vent (Gás Removido)
        ax.arrow(x, y + 6, 0, 8, head_width=2, head_length=2, fc='gray', ec='gray', zorder=4)
        ax.text(x + 5, y + 14, "Gás Removido (Vent)", fontsize=8, color='gray')


    # Linha H2
    draw_valve(ax, x_valve, y_valve_h2, 'firebrick')
    draw_flash(ax, x_flash, y_flash_h2, 'firebrick')
    
    # Linha O2
    draw_valve(ax, x_valve, y_valve_o2, 'navy')
    draw_flash(ax, x_flash, y_flash_o2, 'navy')

    # --- 4. Conexões e Fluxos ---

    # Conexão Agregação -> Válvula
    ax.plot([x_agregacao_h2, x_valve - 1.5], [y_h2, y_h2], color='firebrick', linewidth=2, zorder=2)
    ax.arrow(x_valve - 1.5, y_h2, 0.01, 0, head_width=1.5, head_length=1.5, fc='firebrick', ec='firebrick')

    ax.plot([x_agregacao_o2, x_valve - 1.5], [y_o2, y_o2], color='navy', linewidth=2, zorder=2)
    ax.arrow(x_valve - 1.5, y_o2, 0.01, 0, head_width=1.5, head_length=1.5, fc='navy', ec='navy')

    # Conexão Válvula -> Flash Drum
    ax.plot([x_valve + 1.5, x_flash - 3], [y_h2, y_h2], color='firebrick', linewidth=2, zorder=2)
    ax.arrow(x_flash - 3, y_h2, 0.01, 0, head_width=1.5, head_length=1.5, fc='firebrick', ec='firebrick')

    ax.plot([x_valve + 1.5, x_flash - 3], [y_o2, y_o2], color='navy', linewidth=2, zorder=2)
    ax.arrow(x_flash - 3, y_o2, 0.01, 0, head_width=1.5, head_length=1.5, fc='navy', ec='navy')

    # Mixer Final (Pós-Flash Drums)
    mixer_circle = patches.Circle((x_mixer, y_mixer), radius=5, facecolor='lightgray', edgecolor='black', linewidth=2, zorder=4)
    ax.add_patch(mixer_circle)
    ax.text(x_mixer, y_mixer, "Mixer", ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Conexão Flash Drum H2 -> Mixer
    ax.plot([x_flash + 3, x_mixer], [y_h2, y_mixer + 5], color='firebrick', linewidth=2, zorder=3)
    ax.arrow(x_mixer - 0.5, y_mixer + 5, 0.01, 0, head_width=1.5, head_length=1.5, fc='firebrick', ec='firebrick')
    
    # Conexão Flash Drum O2 -> Mixer
    ax.plot([x_flash + 3, x_mixer], [y_o2, y_mixer - 5], color='navy', linewidth=2, zorder=3)
    ax.arrow(x_mixer - 0.5, y_mixer - 5, 0.01, 0, head_width=1.5, head_length=1.5, fc='navy', ec='navy')

    # Saída do Mixer (Água Purificada + Reposição)
    ax.plot([x_mixer + 5, x_mixer + 20], [y_mixer, y_mixer], color='black', linewidth=3, zorder=4)
    ax.arrow(x_mixer + 20, y_mixer, 0.01, 0, head_width=1.5, head_length=1.5, fc='black', ec='black')
    ax.text(x_mixer + 21, y_mixer + 1, "Água de Recirculação", fontsize=10, fontweight='bold')
    ax.text(x_mixer + 21, y_mixer - 3, "(Pós-Reposição)", fontsize=8)


    # -----------------------------------------------------------
    # 5. Salva e Exibe
    # -----------------------------------------------------------
    salvar_e_exibir_plot('Esquema_Linha_Drenos_KOD_PEM.png', mostrar_grafico)