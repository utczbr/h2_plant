# plots_modulos/plot_propriedades_linha_dreno.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import os
import matplotlib.ticker as ticker 

# Importa a função central de salvamento/exibição
try:
    from plot_reporter_base import salvar_e_exibir_plot 
except ImportError:
    print("AVISO: Dependência 'salvar_e_exibir_plot' não encontrada.")
    def salvar_e_exibir_plot(nome, mostrar): pass


def plot_propriedades_linha_dreno(drenos_plot_data: dict, mostrar_grafico: bool):
    """
    Gera um gráfico das propriedades (T, P, m_dot, h) usando LINHAS,
    dividindo o fluxo H2 e O2 em subplots separados.
    
    CORREÇÃO: Implementa 3 pontos de estado (IN, Pós-Válvula, Tank + Vent OUT) e
    remove o ponto "Mixer Final" para se concentrar na linha de processo individual.
    """
    
    H2_IN = drenos_plot_data.get('H2_IN')
    H2_OUT = drenos_plot_data.get('H2_OUT') # Estado Pós-Flash Drum (Tank + Vent)
    O2_IN = drenos_plot_data.get('O2_IN')
    O2_OUT = drenos_plot_data.get('O2_OUT') # Estado Pós-Flash Drum (Tank + Vent)
    P_OUT_BAR = drenos_plot_data.get('P_OUT_BAR', 4.0)

    # -------------------------------------------------------------------------
    # 1. PREPARAÇÃO DOS DADOS (3 PONTOS)
    # -------------------------------------------------------------------------
    
    # Ponto 2 (Pós-Válvula) usa P, T, h do estado OUT (Tank+Vent) e C_diss do estado IN (Agregação)
    
    h2_tank_vent_out = H2_OUT.copy()
    o2_tank_vent_out = O2_OUT.copy()

    h2_valve_out_ponto = {
        'm_dot_kg_h': H2_IN['m_dot_kg_h'],
        'T': h2_tank_vent_out['T'], 
        'P_bar': h2_tank_vent_out['P_bar'], 
        'h_kJ_kg': h2_tank_vent_out['h_kJ_kg'],
        'C_diss_mg_kg': H2_IN['C_diss_mg_kg'] 
    }
    
    o2_valve_out_ponto = {
        'm_dot_kg_h': O2_IN['m_dot_kg_h'],
        'T': o2_tank_vent_out['T'], 
        'P_bar': o2_tank_vent_out['P_bar'], 
        'h_kJ_kg': o2_tank_vent_out['h_kJ_kg'],
        'C_diss_mg_kg': O2_IN['C_diss_mg_kg'] 
    }
    
    data_h2_list = [H2_IN, h2_valve_out_ponto, h2_tank_vent_out] 
    data_o2_list = [O2_IN, o2_valve_out_ponto, o2_tank_vent_out] 
    
    df_h2 = pd.DataFrame(data_h2_list)
    df_o2 = pd.DataFrame(data_o2_list)
    
    # Renomear colunas para consistência
    cols_map = {'m_dot_kg_h': 'Vazão (kg/h)', 'h_kJ_kg': 'Entalpia (kJ/kg)', 'T': 'Temperatura (°C)', 'P_bar': 'Pressão (bar)'}
    df_h2.rename(columns=cols_map, inplace=True)
    df_o2.rename(columns=cols_map, inplace=True)
    
    # Rótulos dos Componentes (Eixos X)
    x_labels = ['Agregação IN', 'Pós-Válvula', 'Tank + Vent OUT']
    indices = np.arange(len(x_labels))

    # -------------------------------------------------------------------------
    # 2. CONFIGURAÇÃO DA FIGURA E PLOTAGEM
    # -------------------------------------------------------------------------
    
    fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex='col')
    fig.suptitle(f'Propriedades da Linha de Drenos Processadas (P-alvo: {P_OUT_BAR:.1f} bar)', fontsize=14, fontweight='bold')

    
    # Propriedades a plotar
    propriedades = ['Vazão (kg/h)', 'Entalpia (kJ/kg)', 'Temperatura (°C)', 'Pressão (bar)']
    
    # --- Loop por Propriedade ---
    for i, prop in enumerate(propriedades):
        
        # Formatação específica para rótulos de valor
        if prop in ['Temperatura (°C)', 'Pressão (bar)']:
            fmt = '{:.2f}' if prop == 'Pressão (bar)' else '{:.1f}'
        elif prop == 'Vazão (kg/h)':
            fmt = '{:.1f}'
        else: # Entalpia
            fmt = '{:.0f}'
            
        # Posição vertical para o rótulo
        offset_y_h2 = df_h2[prop].max() * 0.005 # Offset de 0.5%
        offset_y_o2 = df_o2[prop].max() * 0.005 # Offset de 0.5%
        
        # --- COLUNA 1: FLUXO H2 ---
        ax1 = axes[i, 0]
        ax1.plot(indices, df_h2[prop], marker='o', linestyle='-', color='darkred', linewidth=3, markersize=8)
        
        if i == 0:
            ax1.set_title(f'Dreno do Fluxo de Hidrogênio (H₂)')
            
        ax1.set_ylabel(prop)
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        ax1.set_xticks(indices)
        ax1.set_xticklabels(x_labels, rotation=15, ha='right', fontsize=9)
        
        # Rótulos de valor
        for idx, val in df_h2[prop].items():
            ax1.text(idx, val + offset_y_h2, fmt.format(val), 
                     ha='center', va='bottom', fontsize=8, weight='bold', color='darkred')


        # --- COLUNA 2: FLUXO O2 ---
        ax2 = axes[i, 1]
        ax2.plot(indices, df_o2[prop], marker='s', linestyle='--', color='darkblue', linewidth=3, markersize=8)
        
        if i == 0:
            ax2.set_title(f'Dreno do Fluxo de Oxigênio (O₂)')

        ax2.set_ylabel(prop)
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        ax2.set_xticks(indices)
        ax2.set_xticklabels(x_labels, rotation=15, ha='right', fontsize=9)

        # Rótulos de valor
        for idx, val in df_o2[prop].items():
            ax2.text(idx, val + offset_y_o2, fmt.format(val), 
                     ha='center', va='bottom', fontsize=8, weight='bold', color='darkblue')

        # Ajustar limite X para os 3 pontos (0, 1, 2)
        ax1.set_xlim(-0.2, len(indices) - 0.8)
        ax2.set_xlim(-0.2, len(indices) - 0.8)
                
    
    plt.tight_layout(rect=[0, 0, 1, 0.96]) 
    
    # 3. Salva e Exibe
    salvar_e_exibir_plot('drenos_propriedades_linha_processo_3_pontos_final.png', mostrar_grafico)