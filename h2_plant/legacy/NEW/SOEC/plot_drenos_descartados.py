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
    # Usamos loc e indexação booleana para garantir a cópia
    df_descarte_h2 = df_h2.loc[df_h2['Componente'].isin(componentes_descartados)].copy()
    df_descarte_o2 = df_o2.loc[df_o2['Componente'].isin(componentes_descartados)].copy()
    
    # Função segura para calcular concentração (mg gás / kg H2O)
    def calculate_conc(row):
         m_dot_gas = row['Gas_Dissolvido_removido_kg_s']
         m_dot_h2o = row['Agua_Pura_Removida_H2O_kg_s']
         if m_dot_h2o > 1e-12:
              return (m_dot_gas / m_dot_h2o) * 1e6
         return 0.0
         
    # Preparar a tabela de exibição
    
    # Criação do Dataframe para H2
    if not df_descarte_h2.empty:
        # Usamos apply para garantir que o cálculo de concentração é escalar
        conc_o2_removed = df_descarte_h2.apply(calculate_conc, axis=1)
        
        df_plot_h2 = pd.DataFrame({
            'Componente': df_descarte_h2['Componente'],
            'Vazao_Proc_kg_h': df_descarte_h2['Agua_Pura_Removida_H2O_kg_s'] * 3600,
            'Temperatura_C': df_descarte_h2['T_C'],
            'Pressao_bar': df_descarte_h2['P_bar'],
            'O2_Dissolvido_mg_kg': conc_o2_removed,
            'H2_Dissolvido_mg_kg': 0.0,
        }).fillna(0)
    else:
         df_plot_h2 = pd.DataFrame(columns=['Componente', 'Vazao_Proc_kg_h', 'Temperatura_C', 'Pressao_bar', 'O2_Dissolvido_mg_kg', 'H2_Dissolvido_mg_kg'])
    
    # Criação do Dataframe para O2
    if not df_descarte_o2.empty:
        conc_h2_removed = df_descarte_o2.apply(calculate_conc, axis=1)

        df_plot_o2 = pd.DataFrame({
            'Componente': df_descarte_o2['Componente'],
            'Vazao_Proc_kg_h': df_descarte_o2['Agua_Pura_Removida_H2O_kg_s'] * 3600,
            'Temperatura_C': df_descarte_o2['T_C'],
            'Pressao_bar': df_descarte_o2['P_bar'],
            'H2_Dissolvido_mg_kg': conc_h2_removed,
            'O2_Dissolvido_mg_kg': 0.0,
        }).fillna(0)
    else:
         df_plot_o2 = pd.DataFrame(columns=['Componente', 'Vazao_Proc_kg_h', 'Temperatura_C', 'Pressao_bar', 'O2_Dissolvido_mg_kg', 'H2_Dissolvido_mg_kg'])

    
    # --- GRÁFICOS (H2) ---
    if not df_plot_h2.empty:
        fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
        fig.suptitle(f"Propriedades dos Drenos DESCARTADOS - Fluxo de H₂", fontsize=16)

        x_labels_h2 = df_plot_h2['Componente']
        x_indices = np.arange(len(x_labels_h2))
        
        # 1. Vazão Mássica (Eixo Único)
        axes[0].bar(x_indices, df_plot_h2['Vazao_Proc_kg_h'], color='darkred')
        axes[0].set_ylabel("Vazão Proc. (kg/h)")
        axes[0].grid(axis='y', linestyle='--', alpha=0.7)
        axes[0].tick_params(axis='y', labelcolor='darkred')
        axes[0].set_ylim(0, df_plot_h2['Vazao_Proc_kg_h'].max() * 1.1)

        # 2. Temperatura
        axes[1].bar(x_indices, df_plot_h2['Temperatura_C'], color='darkred')
        axes[1].set_ylabel("Temperatura (°C)")
        axes[1].grid(axis='y', linestyle='--', alpha=0.7)
        
        # 3. Pressão
        axes[2].bar(x_indices, df_plot_h2['Pressao_bar'], color='darkred')
        axes[2].set_ylabel("Pressão (bar)")
        axes[2].grid(axis='y', linestyle='--', alpha=0.7)
        
        # 4. Concentração Dissolvida
        # 🛑 CORREÇÃO: Usamos o O2_Dissolvido para o Fluxo H2 (Crossover/Impureza)
        axes[3].bar(x_indices, df_plot_h2['O2_Dissolvido_mg_kg'], color='darkred', label='O₂ Dissolvido')
        axes[3].set_ylabel("Conc. Dissolvida (mg/kg)")
        axes[3].set_xlabel("Componente Descartado")
        axes[3].grid(axis='y', linestyle='--', alpha=0.7)
        axes[3].legend()
        
        # Ticks X
        axes[3].set_xticks(x_indices)
        axes[3].set_xticklabels(x_labels_h2, rotation=45, ha='right')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        salvar_e_exibir_plot("drenos_descartados_H2.png", mostrar_grafico)


    # --- GRÁFICOS (O2) ---
    if not df_plot_o2.empty:
        fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
        fig.suptitle(f"Propriedades dos Drenos DESCARTADOS - Fluxo de O₂", fontsize=16)

        x_labels_o2 = df_plot_o2['Componente']
        x_indices = np.arange(len(x_labels_o2))
        
        # 1. Vazão Mássica (Eixo Único)
        axes[0].bar(x_indices, df_plot_o2['Vazao_Proc_kg_h'], color='darkblue')
        axes[0].set_ylabel("Vazão Proc. (kg/h)")
        axes[0].grid(axis='y', linestyle='--', alpha=0.7)
        axes[0].tick_params(axis='y', labelcolor='darkblue')
        axes[0].set_ylim(0, df_plot_o2['Vazao_Proc_kg_h'].max() * 1.1)

        # 2. Temperatura
        axes[1].bar(x_indices, df_plot_o2['Temperatura_C'], color='darkblue')
        axes[1].set_ylabel("Temperatura (°C)")
        axes[1].grid(axis='y', linestyle='--', alpha=0.7)
        
        # 3. Pressão
        axes[2].bar(x_indices, df_plot_o2['Pressao_bar'], color='darkblue')
        axes[2].set_ylabel("Pressão (bar)")
        axes[2].grid(axis='y', linestyle='--', alpha=0.7)
        
        # 4. Concentração Dissolvida
        # 🛑 CORREÇÃO: Usamos o H2_Dissolvido para o Fluxo O2 (Crossover/Impureza)
        axes[3].bar(x_indices, df_plot_o2['H2_Dissolvido_mg_kg'], color='darkblue', label='H₂ Dissolvido')
        axes[3].set_ylabel("Conc. Dissolvida (mg/kg)")
        axes[3].set_xlabel("Componente Descartado")
        axes[3].grid(axis='y', linestyle='--', alpha=0.7)
        axes[3].legend()
        
        # Ticks X
        axes[3].set_xticks(x_indices)
        axes[3].set_xticklabels(x_labels_o2, rotation=45, ha='right')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        salvar_e_exibir_plot("drenos_descartados_O2.png", mostrar_grafico)