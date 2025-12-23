# plots_modulos/plot_esquema_planta_completa.py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import sys

# Importa funções auxiliares de plotagem
from plot_reporter_base import salvar_e_exibir_plot, calcular_vazao_massica_total_completa

def plot_esquema_planta_completa(df_h2: pd.DataFrame, df_o2: pd.DataFrame, estado_recirculacao: dict, mode_deoxo: str, L_deoxo: float, dc2_mode: str, mostrar_grafico: bool = False):
    """
    Gera um esquema de blocos de toda a planta (H2, O2, Drenos) com rastreamento
    dos fluxos externos (Perdas de Gás, Calor Rejeitado, Água de Reposição).
    """
    
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_title(f"Esquema de Fluxo da Planta Completa (Modo Deoxo: {mode_deoxo}, L={L_deoxo:.2f} m)", fontsize=16, pad=20)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off') # Remove os eixos e ticks

    # --- 1. DEFINIÇÃO DOS BLOCOS PRINCIPAIS ---
    
    # 1A. Bloco H2 (Esquerda)
    h2_rect = patches.Rectangle((5, 50), 30, 40, edgecolor='darkred', facecolor='red', alpha=0.1, linewidth=2, label='Processo H₂')
    ax.add_patch(h2_rect)
    ax.text(20, 85, 'Processamento de H₂', ha='center', fontsize=12, fontweight='bold', color='darkred')

    # 1B. Bloco O2 (Centro)
    o2_rect = patches.Rectangle((35, 50), 30, 40, edgecolor='darkblue', facecolor='blue', alpha=0.1, linewidth=2, label='Processo O₂')
    ax.add_patch(o2_rect)
    ax.text(50, 85, 'Processamento de O₂', ha='center', fontsize=12, fontweight='bold', color='darkblue')

    # 1C. Bloco Drenos/Recirculação (Baixo)
    drain_rect = patches.Rectangle((20, 10), 60, 30, edgecolor='darkgreen', facecolor='green', alpha=0.1, linewidth=2, label='Linha de Drenos & Recirculação')
    ax.add_patch(drain_rect)
    ax.text(50, 35, 'Linha de Drenos e Recirculação', ha='center', fontsize=12, fontweight='bold', color='darkgreen')

    # --- 2. ENTRADA PRINCIPAL (PEM) ---
    ax.arrow(50, 98, 0, -5, head_width=1.5, head_length=2, fc='black', ec='black')
    ax.text(50, 98.5, 'Entrada de Gás/Líquido (PEM)', ha='center', fontsize=10, fontweight='bold')
    
    ax.arrow(35, 93, -10, 0, head_width=1.5, head_length=2, fc='red', ec='red')
    ax.text(32, 94, 'Fluxo H₂', ha='center', fontsize=10, color='red')

    ax.arrow(65, 93, 0, 0, head_width=1.5, head_length=2, fc='blue', ec='blue')
    ax.text(68, 94, 'Fluxo O₂', ha='center', fontsize=10, color='blue')

    # --- 3. FLUXOS DE PRODUTO FINAL (SAÍDAS) ---
    
    # Produto H2 (Sai do VSA/PSA)
    m_dot_h2_final = df_h2['m_dot_gas_kg_s'].iloc[-1] * 3600
    ax.arrow(35, 75, -5, 0, head_width=1.5, head_length=2, fc='red', ec='red', linewidth=2)
    ax.text(20, 75, f"H₂ Puro: {m_dot_h2_final:.2f} kg/h", ha='center', fontsize=10, fontweight='bold', color='red')

    # Produto O2 (Sai da Válvula)
    m_dot_o2_final = df_o2['m_dot_gas_kg_s'].iloc[-1] * 3600
    ax.arrow(65, 75, 5, 0, head_width=1.5, head_length=2, fc='blue', ec='blue', linewidth=2)
    ax.text(80, 75, f"O₂ Purificado: {m_dot_o2_final:.2f} kg/h", ha='center', fontsize=10, fontweight='bold', color='blue')

    # --- 4. CONEXÕES PARA DRENOS (DENTRO DO SISTEMA) ---
    
    # Drenos do H2 para o Mixer
    ax.arrow(20, 50, 0, -10, head_width=1, head_length=1.5, fc='darkred', ec='darkred', linestyle='--')
    ax.text(14, 42, 'Drenos H₂', ha='left', fontsize=9, color='darkred')
    
    # Drenos do O2 para o Mixer
    ax.arrow(50, 50, 0, -10, head_width=1, head_length=1.5, fc='darkblue', ec='darkblue', linestyle='--')
    ax.text(44, 42, 'Drenos O₂', ha='left', fontsize=9, color='darkblue')

    # --- 5. FLUXOS DE DESCARTE (DE FORA DO SISTEMA) ---
    
    # A. Calor Rejeitado (Q_dot)
    Q_h2_total = df_h2['Q_dot_fluxo_W'].sum() / 1000 # kW
    Q_o2_total = df_o2['Q_dot_fluxo_W'].sum() / 1000 # kW
    Q_total = Q_h2_total + Q_o2_total
    
    ax.arrow(30, 45, -10, 5, head_width=1.5, head_length=2, fc='darkgray', ec='darkgray', linewidth=1)
    ax.arrow(50, 45, 10, 5, head_width=1.5, head_length=2, fc='darkgray', ec='darkgray', linewidth=1)
    ax.text(18, 55, f"Calor Rejeitado Total: {Q_total:.2f} kW", ha='right', fontsize=9, color='darkgray', fontweight='bold')
    
    # B. Gás Purga / Não Aproveitado (do VSA/PSA)
    
    # H2 Perdido no VSA (perda no VSA/PSA é a purga)
    m_dot_h2_perda = df_h2[df_h2['Componente'].isin(['VSA', 'PSA'])]['Gas_Dissolvido_removido_kg_s'].sum() * 3600
    
    # O2 Perdido (No KODs/Coalescedor - dissolvido)
    m_dot_o2_perda = df_o2[df_o2['Componente'].isin(['KOD 1', 'KOD 2', 'Coalescedor 1'])]['Gas_Dissolvido_removido_kg_s'].sum() * 3600
    
    m_dot_gas_perda_total = m_dot_h2_perda + m_dot_o2_perda
    
    ax.arrow(55, 65, 20, 10, head_width=1.5, head_length=2, fc='orange', ec='orange', linewidth=1)
    ax.text(78, 77, f"Gás Purga/Perdido: {m_dot_gas_perda_total:.3f} kg/h", ha='center', fontsize=9, color='orange', fontweight='bold')

    # C. Água de Reposição (MAKE-UP) (Entra no Mixer)
    m_dot_makeup_kg_h = estado_recirculacao.get('M_dot_makeup_kgs', 0.0) * 3600
    T_makeup = 20.0 # Temperatura assumida para reposição
    
    ax.arrow(70, 20, -5, 0, head_width=1.5, head_length=2, fc='cyan', ec='cyan', linewidth=2)
    ax.text(72, 22, f"Água de Reposição ({T_makeup:.1f}°C): {m_dot_makeup_kg_h:.2f} kg/h", ha='left', fontsize=9, color='cyan', fontweight='bold')

    # D. Água Pura Removida (Sai do KOD/Coalescedor, etc.)
    # Se esta água não vai para o dreno, ela é um descarte/reuso externo.
    # No seu modelo, a água condensada é somada ao dreno, mas vamos ilustrar o 'excesso' de líquido que sai do KODs/Coalescedor no diagrama
    agua_condensada_total = df_h2['Agua_Condensada_kg_s'].sum() * 3600 + df_o2['Agua_Condensada_kg_s'].sum() * 3600
    
    ax.arrow(70, 55, 10, -5, head_width=1.5, head_length=2, fc='lightgreen', ec='lightgreen', linewidth=1)
    ax.text(82, 57, f"Água Condensada TOTAL: {agua_condensada_total:.2f} kg/h", ha='center', fontsize=9, color='lightgreen', fontweight='bold')
    
    # --- 6. SAÍDA DE RECIRCULAÇÃO (SAI DO DRENO) ---
    m_dot_recirc_final = estado_recirculacao.get('M_dot_out_kgs', 0.0) * 3600
    T_recirc_final = estado_recirculacao.get('T_out_C', 0.0)
    
    ax.arrow(50, 10, 0, 5, head_width=1.5, head_length=2, fc='darkgreen', ec='darkgreen', linewidth=3)
    ax.text(50, 25, f"Água Recirculação: {m_dot_recirc_final:.2f} kg/h @ {T_recirc_final:.2f}°C", ha='center', fontsize=10, color='darkgreen', fontweight='bold')

    # Salva e exibe o plot
    salvar_e_exibir_plot('esquema_planta_completa.png', mostrar_grafico)


# FIM: plots_modulos/plot_esquema_planta_completa.py