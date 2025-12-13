# plots_modulos/plot_drenos_descartados.py

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from plot_reporter_base import salvar_e_exibir_plot 

def plot_drenos_descartados(df_h2: pd.DataFrame, df_o2: pd.DataFrame, mostrar_grafico: bool = True):
    """
    Gera gráficos de barras para as propriedades dos drenos que são descartados/não recirculados
    (Coalescedor 1, VSA, PSA).
    """
    
    # Colunas de interesse para extração (usamos a coluna de Agua Pura Removida)
    colunas_interesse = ['Componente', 'Agua_Pura_Removida_H2O_kg_s', 'T_C', 'P_bar', 'Gas_Dissolvido_removido_kg_s', 'y_H2', 'y_O2']
    
    # --- EXTRAÇÃO DE DADOS ---
    
    componentes_descartados = ['Coalescedor 1', 'VSA', 'PSA']
    
    # Filtrar o histórico para os componentes relevantes
    df_descarte_h2 = df_h2[df_h2['Componente'].isin(componentes_descartados)].copy()
    df_descarte_o2 = df_o2[df_o2['Componente'].isin(componentes_descartados)].copy()

    # Preparar a tabela de exibição
    # Nota: Em H2, Gas Dissolvido é H2. Em O2, Gas Dissolvido é O2.
    
    # Criação do Dataframe para H2
    df_plot_h2 = pd.DataFrame({
        'Componente': df_descarte_h2['Componente'],
        'Vazao_Proc_kg_h': df_descarte_h2['Agua_Pura_Removida_H2O_kg_s'] * 3600,
        'Temperatura_C': df_descarte_h2['T_C'],
        'Pressao_bar': df_descarte_h2['P_bar'],
        'H2_Dissolvido_mg_kg': (df_descarte_h2['Gas_Dissolvido_removido_kg_s'] / df_descarte_h2['Agua_Pura_Removida_H2O_kg_s']) * 1e6 * 3600, # Aprox. M_gas/M_H2O
        'O2_Dissolvido_mg_kg': 0.0, # O2 Dissolvido é zero no fluxo H2
    }).fillna(0)
    
    # Criação do Dataframe para O2
    df_plot_o2 = pd.DataFrame({
        'Componente': df_descarte_o2['Componente'],
        'Vazao_Proc_kg_h': df_descarte_o2['Agua_Pura_Removida_H2O_kg_s'] * 3600,
        'Temperatura_C': df_descarte_o2['T_C'],
        'Pressao_bar': df_descarte_o2['P_bar'],
        'O2_Dissolvido_mg_kg': (df_descarte_o2['Gas_Dissolvido_removido_kg_s'] / df_descarte_o2['Agua_Pura_Removida_H2O_kg_s']) * 1e6 * 3600, # Aprox. M_gas/M_H2O
        'H2_Dissolvido_mg_kg': 0.0, # H2 Dissolvido é zero no fluxo O2
    }).fillna(0)
    
    
    # --- GRÁFICOS (H2) ---
    if not df_plot_h2.empty:
        fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
        fig.suptitle(f"Propriedades dos Drenos DESCARTADOS - Fluxo de H₂", fontsize=16)

        x_labels_h2 = df_plot_h2['Componente']
        
        # 1. Vazão Mássica (Eixo Único)
        axes[0].bar(x_labels_h2, df_plot_h2['Vazao_Proc_kg_h'], color='darkred')
        axes[0].set_ylabel("Vazão Proc. (kg/h)")
        axes[0].grid(axis='y', linestyle='--', alpha=0.7)
        axes[0].tick_params(axis='y', labelcolor='darkred')
        axes[0].set_ylim(0, df_plot_h2['Vazao_Proc_kg_h'].max() * 1.1)

        # 2. Temperatura
        axes[1].bar(x_labels_h2, df_plot_h2['Temperatura_C'], color='darkred')
        axes[1].set_ylabel("Temperatura (°C)")
        axes[1].grid(axis='y', linestyle='--', alpha=0.7)
        
        # 3. Pressão
        axes[2].bar(x_labels_h2, df_plot_h2['Pressao_bar'], color='darkred')
        axes[2].set_ylabel("Pressão (bar)")
        axes[2].grid(axis='y', linestyle='--', alpha=0.7)
        
        # 4. Concentração Dissolvida
        axes[3].bar(x_labels_h2, df_plot_h2['H2_Dissolvido_mg_kg'], color='darkred', label='H₂ Dissolvido')
        axes[3].set_ylabel("Conc. Dissolvida (mg/kg)")
        axes[3].set_xlabel("Componente Descartado")
        axes[3].grid(axis='y', linestyle='--', alpha=0.7)
        axes[3].legend()
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        salvar_e_exibir_plot("drenos_descartados_H2.png", mostrar_grafico)


    # --- GRÁFICOS (O2) ---
    if not df_plot_o2.empty:
        fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
        fig.suptitle(f"Propriedades dos Drenos DESCARTADOS - Fluxo de O₂", fontsize=16)

        x_labels_o2 = df_plot_o2['Componente']
        
        # 1. Vazão Mássica (Eixo Único)
        axes[0].bar(x_labels_o2, df_plot_o2['Vazao_Proc_kg_h'], color='darkblue')
        axes[0].set_ylabel("Vazão Proc. (kg/h)")
        axes[0].grid(axis='y', linestyle='--', alpha=0.7)
        axes[0].tick_params(axis='y', labelcolor='darkblue')
        axes[0].set_ylim(0, df_plot_o2['Vazao_Proc_kg_h'].max() * 1.1)

        # 2. Temperatura
        axes[1].bar(x_labels_o2, df_plot_o2['Temperatura_C'], color='darkblue')
        axes[1].set_ylabel("Temperatura (°C)")
        axes[1].grid(axis='y', linestyle='--', alpha=0.7)
        
        # 3. Pressão
        axes[2].bar(x_labels_o2, df_plot_o2['Pressao_bar'], color='darkblue')
        axes[2].set_ylabel("Pressão (bar)")
        axes[2].grid(axis='y', linestyle='--', alpha=0.7)
        
        # 4. Concentração Dissolvida
        axes[3].bar(x_labels_o2, df_plot_o2['O2_Dissolvido_mg_kg'], color='darkblue', label='O₂ Dissolvido')
        axes[3].set_ylabel("Conc. Dissolvida (mg/kg)")
        axes[3].set_xlabel("Componente Descartado")
        axes[3].grid(axis='y', linestyle='--', alpha=0.7)
        axes[3].legend()
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        salvar_e_exibir_plot("drenos_descartados_O2.png", mostrar_grafico)