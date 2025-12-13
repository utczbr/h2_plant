# plots_modulos/plot_deoxo_perfil.py

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from constants_and_config import T_JACKET_DEOXO_C
from plot_reporter_base import salvar_e_exibir_plot

# CORREÇÃO: Adicionando 'mostrar_grafico' como 7º argumento posicional
def plot_deoxo_perfil(df_h2: pd.DataFrame, L_span: np.ndarray, T_profile_C: np.ndarray, X_O2: float, T_max_calc: float, deoxo_mode: str, L_deoxo: float, mostrar_grafico: bool = False):
    """
    Gera o gráfico do perfil de Temperatura e Conversão no reator Deoxo.
    """
    
    deoxo_df = df_h2[df_h2['Componente'] == 'Deoxo']

    if not deoxo_df.empty and L_span is not None and T_profile_C is not None and T_profile_C.size > 1:
        
        idx_deoxo = df_h2[df_h2['Componente'] == 'Deoxo'].index[0]
        T_in_C = df_h2.iloc[idx_deoxo - 1]['T_C'] if idx_deoxo > 0 else T_profile_C[0]

        X_profile = np.linspace(0, X_O2, len(L_span))
        T_jacket_C = T_JACKET_DEOXO_C

        fig, ax1 = plt.subplots(figsize=(10, 6))
        fig.suptitle(f'Perfil de Temperatura e Conversão no Reator Deoxo (H2)', y=1.0)

        # Perfil de Conversão
        ax1.set_xlabel(f'Comprimento do Reator (L) (m) - L_total: {L_deoxo:.3f} m')
        ax1.set_ylabel('Conversão de O₂ (X) - Curva Azul', color='tab:blue')
        line_conv, = ax1.plot(L_span, X_profile, label='Conversão de O₂ (X)', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.set_ylim(0, 1.1)
        
        step = max(1, len(L_span) // 10) 
        for i in range(0, len(L_span), step):
            if X_profile[i] > 0:
                 ax1.text(L_span[i], X_profile[i] * 1.05, f'{X_profile[i]:.4f}', ha='center', va='bottom', fontsize=8, color=line_conv.get_color())
        

        # Perfil de Temperatura
        ax2 = ax1.twinx()
        ax2.set_ylabel('Temperatura (T) (C) - Curva Vermelha', color='tab:red')
        line_temp, = ax2.plot(L_span, T_profile_C, label='Temperatura do Fluxo (T)', color='tab:red')
        
        step = max(1, len(L_span) // 10)
        for i in range(0, len(L_span), step):
            ax2.text(L_span[i], T_profile_C[i], f'{T_profile_C[i]:.2f}', ha='center', va='bottom', fontsize=8, color='tab:red')


        # Linhas de referência de temperatura
        line_t_in = ax2.axhline(T_in_C, color='k', linestyle=':', label=f'T_in real = {T_in_C:.2f} C')
        lines = [line_t_in]

        if deoxo_mode == 'JACKET':
            line_t_jacket = ax2.axhline(T_JACKET_DEOXO_C, color='g', linestyle='-.', label=f'T_jacket = {T_JACKET_DEOXO_C:.2f} C')
            lines.append(line_t_jacket)
        
        if T_max_calc is not None:
             line_t_max = ax2.axhline(T_max_calc, color='r', linestyle='--', label=f'T_max calculada = {T_max_calc:.2f} C')
             lines.append(line_t_max)

        ax2.tick_params(axis='y', labelcolor='tab:red')
        
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        
        fig.legend(h1 + h2 + lines, l1 + l2 + [l.get_label() for l in lines], loc='upper left', bbox_to_anchor=(1.05, 1.05))

        plt.tight_layout(rect=[0, 0, 1.0, 0.96])
        
        # SALVAMENTO CENTRALIZADO
        nome_arquivo = f'plot_deoxo_perfil.png'
        salvar_e_exibir_plot(nome_arquivo, mostrar_grafico)
        
        return
            
    print(f"\n--- Aviso: O gráfico do Perfil Deoxo (MODO {deoxo_mode}, L: {L_deoxo:.3f} m) não pode ser gerado, pois o H2 não passou pelo Deoxo ou dados insuficientes. ---")