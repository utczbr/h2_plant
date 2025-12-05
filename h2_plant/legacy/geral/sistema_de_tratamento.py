import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import CoolProp.CoolProp as CP
from scipy.integrate import solve_ivp

# =================================================================
# === INSTRUÇÕES DE IMPORTAÇÃO DOS MODELOS DE COMPONENTES ===
# Por favor, garanta que os arquivos de modelo estão nomeados exatamente como:
# - 'modelo_dry_cooler.py'
# - 'modelo_chiller.py'
# - 'modelo_kod.py'
# - 'modelo_coalescedor.py'
# - 'modelo_deoxo.py'
# =================================================================
try:
    # Importa as funções dos modelos dos componentes com nomes padronizados
    from modelo_dry_cooler import modelar_dry_cooler
    from modelo_chiller import modelar_chiller_gas
    from modelo_kod import modelar_knock_out_drum
    from modelo_coalescedor import modelar_coalescedor 
    from modelo_deoxo import modelar_deoxo
except ImportError as e:
    print("="*80)
    print("ERRO CRÍTICO DE IMPORTAÇÃO DOS MODELOS DE COMPONENTES:")
    print("Não foi possível encontrar as funções de modelagem.")
    print(f"Detalhes do erro: {e}")
    print("\nPor favor, VERIFIQUE se os arquivos de modelo estão nomeados corretamente:")
    print("e garanta que estão no mesmo diretório.")
    print("="*80)
    sys.exit()

# =================================================================
# === CONSTANTES GLOBAIS DE PROCESSO (Dimensionamento) ===
# =================================================================
T_IN_C = 80.0       # °C (Saída do Eletrolisador - Temperatura de entrada no sistema)
P_IN_BAR = 40.0     # bar (Pressão do Sistema - Pressão de entrada no sistema)
M_DOT_G_H2 = 0.02472 # kg/s (Vazão mássica máxima de H2)
M_DOT_G_O2 = 0.19778 # kg/s (Vazão mássica máxima de O2)
T_CHILLER_OUT_C = 4.0 # °C (Temperatura de saída desejada do Chiller)

# Componentes expandidos para o re-resfriamento do H2
COMPONENTS = ["Entrada", "Dry Cooler 1", "Chiller 1", "KOD 1", "Coalescedor 1", "Deoxo", "Dry Cooler 2", "Chiller 2", "KOD 2"] 
GASES = ['H2', 'O2']

# Fração Molar de O2 na entrada (Assumida 1% do fluxo principal para Deoxo)
Y_O2_IN_H2 = 0.01 
Y_H2_IN_O2 = 0.0001 # 100 ppm de H2 no fluxo de O2 (valor de controle, não removido)

# =================================================================
# === FUNÇÕES AUXILIARES DE CÁLCULO DE PROPRIEDADES (COOLPROP) ===
# =================================================================

def calcular_estado_termodinamico(gas_fluido: str, T_C: float, P_bar: float, m_dot_gas_kg_s: float, y_H2O: float, y_O2: float = 0.0, y_H2: float = 0.0) -> dict:
    """Calcula propriedades termodinâmicas da mistura usando CoolProp, rastreando O2/H2."""
    T_K = T_C + 273.15
    P_PA = P_bar * 1e5
    
    # Constantes
    M_H2O = CP.PropsSI('M', 'Water')
    M_H2 = CP.PropsSI('M', 'H2')
    M_O2 = CP.PropsSI('M', 'O2')
    
    # 1. Vazões Molares e Mássicas (Recalculando frações para garantir que a soma seja 1)
    
    if gas_fluido == 'H2':
        M_PRINCIPAL = M_H2
        y_H2_princ = 1.0 - y_H2O - y_O2
        
        # Defesa contra divisão por zero se y_H2_princ for zero ou negativo
        if y_H2_princ <= 0: y_H2_princ = 1.0 # Força valor seguro
        
        F_princ_molar = m_dot_gas_kg_s / M_PRINCIPAL
        F_O2_molar = F_princ_molar * (y_O2 / y_H2_princ)
        F_H2O_molar = F_princ_molar * (y_H2O / y_H2_princ)
        F_H2_molar = F_princ_molar
        F_molar_total = F_princ_molar + F_O2_molar + F_H2O_molar
        
        y_H2_final = F_princ_molar / F_molar_total
        y_O2_final = F_O2_molar / F_molar_total
        y_H2O_final = F_H2O_molar / F_molar_total
        
        # Vazão Mássica Total da Mistura
        m_dot_mix_kg_s = F_princ_molar * M_H2 + F_O2_molar * M_O2 + F_H2O_molar * M_H2O
        
    else: # O2
        M_PRINCIPAL = M_O2
        y_O2_princ = 1.0 - y_H2O - y_H2
        
        if y_O2_princ <= 0: y_O2_princ = 1.0 # Força valor seguro
        
        F_princ_molar = m_dot_gas_kg_s / M_PRINCIPAL
        F_H2_molar = F_princ_molar * (y_H2 / y_O2_princ)
        F_H2O_molar = F_princ_molar * (y_H2O / y_O2_princ)
        F_O2_molar = F_princ_molar
        F_molar_total = F_princ_molar + F_H2_molar + F_H2O_molar
        
        y_O2_final = F_princ_molar / F_molar_total
        y_H2_final = F_H2_molar / F_molar_total
        y_H2O_final = F_H2O_molar / F_molar_total
        
        # Vazão Mássica Total da Mistura
        m_dot_mix_kg_s = F_princ_molar * M_O2 + F_H2_molar * M_H2 + F_H2O_molar * M_H2O

    
    # 2. Entalpia e Pressões
    
    # 2.1. Cálculo da Entalpia do H2O (Vapor)
    P_H2O_PA_parcial = P_PA * y_H2O_final
    
    try:
        # P_sat_H2O_PA: Pressão de saturação na T atual
        P_sat_H2O_PA = CP.PropsSI('P', 'T', T_K, 'Q', 0, 'Water')
        
        # Defesa contra erro de saturação da água (força estado superaquecido se muito próximo)
        if abs(P_H2O_PA_parcial - P_sat_H2O_PA) / P_sat_H2O_PA < 1e-6: 
             P_H2O_calc = P_H2O_PA_parcial * (1.0 - 1e-7) 
        else:
             P_H2O_calc = P_H2O_PA_parcial
             
        H_H2O_vap = CP.PropsSI('H', 'T', T_K, 'P', P_H2O_calc, 'Water') # J/kg_H2O

    except ValueError:
        # Fallback
        H_H2O_vap = CP.PropsSI('H', 'T', T_K, 'P', P_PA / 10.0, 'Water') 
    
    # 2.2. Entalpia dos Gases Principais (H2 e O2)
    
    # DEFESA CONTRA PRESSÃO ZERO/QUASE ZERO (Para CoolProp)
    P_MIN_PA = 1.0 # 1 Pascal (Evita erro Brent's method quando a concentração é < 1 ppm em 40 bar)

    # Entalpia do H2
    P_H2_PA_parcial = P_PA * y_H2_final
    P_H2_calc = max(P_H2_PA_parcial, P_MIN_PA)
    H_H2 = CP.PropsSI('H', 'T', T_K, 'P', P_H2_calc, 'H2') 

    # Entalpia do O2
    P_O2_PA_parcial = P_PA * y_O2_final
    P_O2_calc = max(P_O2_PA_parcial, P_MIN_PA)
    H_O2 = CP.PropsSI('H', 'T', T_K, 'P', P_O2_calc, 'O2') 

    # 2.3. Entalpia Mássica da Mistura (J/kg_mix)
    H_mix_J_kg = (F_H2_molar * M_H2 * H_H2 + F_O2_molar * M_O2 * H_O2 + F_H2O_molar * M_H2O * H_H2O_vap) / m_dot_mix_kg_s
    
    # 3. Pressões Parciais (Lei de Dalton)
    P_H2O_parcial_bar = P_PA * y_H2O_final / 1e5
    
    # 4. Estado de Saturação (para exibição)
    y_H2O_sat = P_sat_H2O_PA / P_PA
    estado_saturacao = "Saturado" if y_H2O_final >= y_H2O_sat else "Vapor Superaquecido"
    
    return {
        "T_C": T_C,
        "P_bar": P_bar,
        "y_H2O": y_H2O_final,
        "y_O2": y_O2_final, 
        "y_H2": y_H2_final, 
        "m_dot_gas_kg_s": m_dot_mix_kg_s, # Vazão mássica da mistura total
        "m_dot_H2O_vap_kg_s": F_H2O_molar * M_H2O, # Vazão mássica de vapor de água
        "H_mix_J_kg": H_mix_J_kg,
        "P_H2O_bar": P_H2O_parcial_bar,
        "y_H2O_sat": y_H2O_sat,
        "Estado_H2O": estado_saturacao,
        "F_molar_total": F_molar_total
    }

def calcular_y_H2O_inicial(T_C, P_bar) -> float:
    """Calcula a fração molar inicial de água (y_H2O) assumindo saturação total."""
    T_K = T_C + 273.15
    P_PA = P_bar * 1e5
    try:
        P_sat_H2O_PA = CP.PropsSI('P', 'T', T_K, 'Q', 0, 'Water')
        return P_sat_H2O_PA / P_PA
    except:
        return 0.0 # Retorna 0 em caso de erro

# =================================================================
# === FUNÇÕES DE PLOTAGEM E EXIBIÇÃO DE RESULTADOS ===
# =================================================================

def exibir_estado_final(df: pd.DataFrame, gas_fluido: str):
    """Exibe o estado final do fluido."""
    estado_final = df[df['Componente'] == COMPONENTS[-1]].iloc[0] # Último componente
    
    print("\n" + "="*80)
    print(f"ESTADO FINAL DO FLUIDO: {gas_fluido}")
    print("="*80)
    print(f"Componente Final: {estado_final['Componente']}")
    print(f"Temperatura (T): {estado_final['T_C']:.2f} °C")
    print(f"Pressão (P): {estado_final['P_bar']:.2f} bar")
    print(f"Vazão Mássica de Gás: {estado_final['m_dot_gas_kg_s']:.5f} kg/s") 
    print(f"Fração Molar de H₂O (y_H₂O): {estado_final['y_H2O']:.6f} ({estado_final['y_H2O'] * 100:.4f} %)")
    
    # Exibição seletiva de impurezas
    if gas_fluido == 'H2':
        print(f"Fração Molar de O₂ (y_O₂): {estado_final['y_O2']:.2e} ({estado_final['y_O2'] * 1e6:.2f} ppm)")
    else:
        print(f"Fração Molar de H₂ (y_H₂): {estado_final['y_H2']:.2e} ({estado_final['y_H2'] * 1e6:.2f} ppm)")

    print(f"Entalpia Mássica da Mistura: {estado_final['H_mix_J_kg'] / 1000:.2f} kJ/kg")
    print(f"Estado da Água: O gás de saída está {estado_final['Estado_H2O']}")
    # Combina a água removida no KOD e no Coalescedor
    agua_removida_total = df['Agua_Condensada_kg_s'].sum()
    print(f"Água Líquida (Condensado + Aerossóis) Removida TOTAL: {agua_removida_total:.8f} kg/s")
    if 'Status_KOD' in df.columns:
         # Pega o status do KOD antes do coalescedor
        status_kod = df[df['Componente'] == 'KOD 1']['Status_KOD'].iloc[0] # Mudança: KOD 1
        print(f"Status do KOD 1: {status_kod}")
        status_kod_2 = df[df['Componente'] == 'KOD 2']['Status_KOD'].iloc[0] # Mudança: KOD 2
        print(f"Status do KOD 2: {status_kod_2}")
    print("="*80)

def plot_propriedades(df_h2: pd.DataFrame, df_o2: pd.DataFrame):
    """Gera gráficos de T, P, y_H2O, H vs. Componente, e o novo gráfico de impureza, em 3 figuras."""
    
    x_labels = df_h2['Componente']
    
    # --- Figura 1: Temperatura e Pressão ---
    fig1, axes1 = plt.subplots(2, 1, figsize=(10, 8))
    
    # 1. Temperatura
    axes1[0].plot(x_labels, df_h2['T_C'], marker='o', label='H2 - Temperatura (°C)', color='blue')
    axes1[0].plot(x_labels, df_o2['T_C'], marker='s', label='O2 - Temperatura (°C)', color='red')
    axes1[0].set_title('Evolução da Temperatura')
    axes1[0].set_ylabel('T (°C)')
    axes1[0].grid(True, linestyle='--')
    axes1[0].legend()
    axes1[0].tick_params(axis='x', rotation=15)

    # 2. Pressão
    axes1[1].plot(x_labels, df_h2['P_bar'], marker='o', label='H2 - Pressão (bar)', color='blue')
    axes1[1].plot(x_labels, df_o2['P_bar'], marker='s', label='O2 - Pressão (bar)', color='red')
    axes1[1].set_title('Evolução da Pressão')
    axes1[1].set_ylabel('P (bar)')
    axes1[1].grid(True, linestyle='--')
    axes1[1].legend()
    axes1[1].set_xlabel('Componente')
    axes1[1].tick_params(axis='x', rotation=15)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.suptitle('Figura 1: Evolução da Temperatura e Pressão', y=1.0)
    plt.show()

    # --- Figura 2: Umidade e Entalpia ---
    fig2, axes2 = plt.subplots(2, 1, figsize=(10, 8))

    # 3. Fração Molar de H2O (y_H2O)
    axes2[0].plot(x_labels, df_h2['y_H2O'] * 100, marker='o', label='H2 - y_H2O (%)', color='blue')
    axes2[0].plot(x_labels, df_o2['y_H2O'] * 100, marker='s', label='O2 - y_H2O (%)', color='red')
    axes2[0].set_title('Evolução da Umidade (Fração Molar de H₂O)')
    axes2[0].set_ylabel('y_H2O (%)')
    axes2[0].grid(True, linestyle='--')
    axes2[0].legend()
    axes2[0].tick_params(axis='x', rotation=15)

    # 4. Entalpia Mássica da Mistura
    axes2[1].plot(x_labels, df_h2['H_mix_J_kg'] / 1000, marker='o', label='H2 - Entalpia (kJ/kg)', color='blue')
    axes2[1].plot(x_labels, df_o2['H_mix_J_kg'] / 1000, marker='s', label='O2 - Entalpia (kJ/kg)', color='red')
    axes2[1].set_title('Evolução da Entalpia Mássica da Mistura')
    axes2[1].set_ylabel('H (kJ/kg)')
    axes2[1].set_xlabel('Componente')
    axes2[1].grid(True, linestyle='--')
    axes2[1].legend()
    axes2[1].tick_params(axis='x', rotation=15)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.suptitle('Figura 2: Evolução da Umidade e Entalpia', y=1.0)
    plt.show()

    # --- Figura 3: Rastreamento de Impurezas ---
    fig3, axes3 = plt.subplots(1, 1, figsize=(10, 5))
    
    # 5. Rastreamento de Impurezas (y_O2 no H2 e y_H2 no O2)
    # y_O2 (H2) em ppm (multiplicado por 1e6)
    axes3.plot(x_labels, df_h2['y_O2'] * 1e6, marker='o', label='H2 - Impureza O₂ (ppm)', color='darkgreen')
    # y_H2 (O2) em ppm
    axes3.plot(x_labels, df_o2['y_H2'] * 1e6, marker='s', label='O2 - Impureza H₂ (ppm)', color='orange')
    
    axes3.set_title('Evolução das Principais Impurezas')
    axes3.set_ylabel('Concentração (ppm)')
    axes3.set_xlabel('Componente')
    axes3.tick_params(axis='x', rotation=15)
    axes3.set_yscale('log') # Escala logarítmica para melhor visualização da redução
    axes3.grid(True, which="both", linestyle='--')
    axes3.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.suptitle('Figura 3: Evolução das Impurezas (Escala Logarítmica)', y=1.0)
    plt.show()

def plot_vazao_massica(df_h2: pd.DataFrame, df_o2: pd.DataFrame):
    """Gera gráfico da vazão mássica do gás (H2/O2) vs. Componente."""
    
    fig, ax = plt.subplots(figsize=(10, 5))
    x_labels = df_h2['Componente']

    # Vazão Mássica de Gás (H2/O2)
    ax.plot(x_labels, df_h2['m_dot_gas_kg_s'] * 1000, 
            marker='o', linestyle='-', label='H2 - Vazão Mássica de Gás (g/s)', color='darkblue')
    ax.plot(x_labels, df_o2['m_dot_gas_kg_s'] * 1000, 
            marker='s', linestyle='-', label='O2 - Vazão Mássica de Gás (g/s)', color='darkred')

    ax.set_title('Evolução da Vazão Mássica de Gás (H₂/O₂) ao Longo do Processo')
    ax.set_ylabel('Vazão Mássica de Gás (g/s)')
    ax.set_xlabel('Componente')
    ax.grid(True, linestyle='--')
    ax.legend()
    
    plt.tight_layout()
    plt.show()

def plot_vazao_agua(df_h2: pd.DataFrame, df_o2: pd.DataFrame):
    """Gera gráfico da vazão mássica de vapor de água e condensado."""
    
    fig, ax = plt.subplots(figsize=(10, 5))
    x_labels = df_h2['Componente']

    # 1. Vazão de Vapor de Água (Linha)
    ax.plot(x_labels, df_h2['m_dot_H2O_vap_kg_s'] * 1000, 
            marker='o', linestyle='-', label='H2 - Vazão de Vapor (g/s)', color='blue')
    ax.plot(x_labels, df_o2['m_dot_H2O_vap_kg_s'] * 1000, 
            marker='s', linestyle='-', label='O2 - Vazão de Vapor (g/s)', color='red')

    # 2. Água Condensada Removida (Barra)
    # Componentes que removem água líquida
    comp_remover = ['KOD 1', 'Coalescedor 1', 'KOD 2'] # Mudança: KOD 1, Coalescedor 1, KOD 2
    
    # Extrai dados de remoção
    remocao_h2 = df_h2[df_h2['Componente'].isin(comp_remover)]['Agua_Condensada_kg_s'] * 1000
    remocao_o2 = df_o2[df_o2['Componente'].isin(comp_remover)]['Agua_Condensada_kg_s'] * 1000
    
    # Posições no eixo x
    posicoes_x_h2 = [df_h2['Componente'].to_list().index(c) for c in comp_remover]
    
    bar_width = 0.1
    ax.bar([p - 0.05 for p in posicoes_x_h2], remocao_h2, width=bar_width, color='gray', label='H2 - Líquido Removido', align='center')
    ax.bar([p + 0.05 for p in posicoes_x_h2], remocao_o2, width=bar_width, color='darkgray', label='O2 - Líquido Removido', align='center')

    ax.set_title('Rastreamento da Vazão de Água (Vapor no Fluxo e Líquido Removido)')
    ax.set_ylabel('Vazão Mássica de Água (g/s)')
    ax.set_xlabel('Componente')
    ax.grid(True, linestyle='--')
    ax.legend(loc='upper right', fontsize='small')
    
    plt.tight_layout()
    plt.show()

def plot_fluxos_energia(df_h2: pd.DataFrame, df_o2: pd.DataFrame):
    """Gera gráficos de Energia do Fluxo e Energia do Componente."""
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Componentes para o gráfico (Excluindo 'Entrada')
    df_plot_h2 = df_h2[df_h2['Componente'] != 'Entrada']
    df_plot_o2 = df_o2[df_o2['Componente'] != 'Entrada']
    x_comp = df_plot_h2['Componente']
    
    bar_width = 0.35
    r1 = np.arange(len(x_comp))
    r2 = [x + bar_width for x in r1]
    
    # 1. Energia que sai ou entra do FLUXO (Q_dot_fluxo_W)
    q_fluxo_h2 = df_plot_h2['Q_dot_fluxo_W'] / 1000 # kW
    q_fluxo_o2 = df_plot_o2['Q_dot_fluxo_W'] / 1000 # kW
    
    axes[0].bar(r1, q_fluxo_h2, color='blue', width=bar_width, edgecolor='grey', label='H2 - Calor Removido (Fluxo)')
    axes[0].bar(r2, q_fluxo_o2, color='red', width=bar_width, edgecolor='grey', label='O2 - Calor Removido (Fluxo)')
    axes[0].set_title('Carga Térmica (Calor Trocado com o Fluxo)')
    axes[0].set_ylabel('Potência Térmica (kW)')
    axes[0].set_xticks([r + bar_width/2 for r in r1])
    axes[0].set_xticklabels(x_comp)
    axes[0].grid(axis='y', linestyle='--')
    axes[0].legend(loc='lower left')

    # 2. Entrada/Saída de Energia do COMPONENTE (W_dot_comp_W)
    w_comp_h2 = df_plot_h2['W_dot_comp_W'] / 1000 # kW
    w_comp_o2 = df_plot_o2['W_dot_comp_W'] / 1000 # kW
    
    axes[1].bar(r1, w_comp_h2, color='skyblue', width=bar_width, edgecolor='grey', label='H2 - Consumo Elétrico (Componente)')
    axes[1].bar(r2, w_comp_o2, color='salmon', width=bar_width, edgecolor='grey', label='O2 - Consumo Elétrico (Componente)')
    axes[1].set_title('Consumo de Energia Elétrica (Trabalho no Componente)')
    axes[1].set_ylabel('Potência Elétrica (kW)')
    axes[1].set_xticks([r + bar_width/2 for r in r1])
    axes[1].set_xticklabels(x_comp, rotation=15)
    axes[1].set_xlabel('Componente')
    axes[1].grid(axis='y', linestyle='--')
    axes[1].legend(loc='upper left')

    plt.tight_layout()
    plt.show()

def plot_agua_removida_total(df_h2: pd.DataFrame, df_o2: pd.DataFrame):
    """Gera gráfico da quantidade total de água retirada por componente."""

    # Componentes que removem água líquida
    comp_remover = ['KOD 1', 'Coalescedor 1', 'KOD 2']
    
    # Inicializa as remoções
    remocao_h2 = {comp: df_h2[df_h2['Componente'] == comp]['Agua_Condensada_kg_s'].iloc[0] * 1000 for comp in comp_remover}
    remocao_o2 = {comp: df_o2[df_o2['Componente'] == comp]['Agua_Condensada_kg_s'].iloc[0] * 1000 for comp in comp_remover}
    
    # Remove KOD 2 e Coalescedor 1 para O2 (pois o fluxo de O2 só usa KOD 1 e Coalescedor 1)
    if 'KOD 2' in remocao_o2: remocao_o2['KOD 2'] = 0
    
    componentes_x = [c for c in comp_remover]
    
    dados_h2 = [remocao_h2[c] for c in componentes_x]
    dados_o2 = [remocao_o2.get(c, 0) for c in componentes_x] # Usa .get para evitar KeyError se o O2 não tiver KOD 2
    
    df_plot = pd.DataFrame({
        'H2': dados_h2,
        'O2': dados_o2
    }, index=componentes_x)

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plotagem em barras agrupadas
    bar_width = 0.35
    r1 = np.arange(len(componentes_x))
    r2 = [x + bar_width for x in r1]
    
    barras_h2 = ax.bar(r1, df_plot['H2'], color='blue', width=bar_width, edgecolor='black', label='Fluxo de H₂')
    barras_o2 = ax.bar(r2, df_plot['O2'], color='red', width=bar_width, edgecolor='black', label='Fluxo de O₂')

    ax.set_title('Vazão de Água Líquida Removida por Componente')
    ax.set_ylabel('Vazão Mássica de Água Líquida (g/s)')
    ax.set_xticks([r + bar_width/2 for r in r1])
    ax.set_xticklabels(componentes_x)
    ax.grid(axis='y', linestyle='--')
    ax.legend()
    
    # Adicionar os valores nas barras (H2)
    for bar in barras_h2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.000000001, f"{yval:.8f}", ha='center', va='bottom', fontsize=8)
    
    # Adicionar os valores nas barras (O2)
    for bar in barras_o2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.000000001, f"{yval:.8f}", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_deoxo_perfil(df_h2: pd.DataFrame):
    """
    Gera o gráfico do perfil de Temperatura e Conversão no reator Deoxo.
    Esta função só plota para o H2.
    """
    
    # Extrai o resultado da simulação do Deoxo para H2
    deoxo_entry = df_h2[df_h2['Componente'] == 'Deoxo'].iloc[0]
    
    # Se o H2 passou pelo Deoxo (deoxo_entry['T_profile_C'] existe e não é None)
    if deoxo_entry['T_profile_C'] is not None:
        T_profile_C = deoxo_entry['T_profile_C']
        L_span = deoxo_entry['L_span']
        X_final = deoxo_entry['X_O2']
        
        # Cria um perfil de conversão linear ou usa o perfil completo (se salvo)
        X_profile = np.linspace(0, X_final, len(L_span))
        
        T_in_C = T_profile_C[0]
        T_jacket_C = 120.0 # Parâmetro fixo do modelo Deoxo

        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Perfil de Conversão
        ax1.set_xlabel('Comprimento do Reator (L) (m)')
        ax1.set_ylabel('Conversão de O₂ (X)', color='tab:blue')
        ax1.plot(L_span, X_profile, color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.set_ylim(0, 1.1)

        # Perfil de Temperatura
        ax2 = ax1.twinx()
        ax2.set_ylabel('Temperatura (T) (C)', color='tab:red') 
        ax2.plot(L_span, T_profile_C, color='tab:red')

        # Linhas de referência de temperatura
        ax2.axhline(T_in_C, color='k', linestyle=':', label=f'T_in = {T_in_C:.0f} C')
        ax2.axhline(np.max(T_profile_C), color='r', linestyle='--', label=f'T_max = {np.max(T_profile_C):.1f} C')
        ax2.axhline(T_jacket_C, color='g', linestyle='-.', label=f'T_jacket = {T_jacket_C:.0f} C')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        plt.title(f'Perfil de Temperatura e Conversão no Reator Deoxo ({df_h2["Componente"].iloc[-1]})')
        fig.tight_layout()
        plt.legend(loc='lower right')
        plt.grid(True, linestyle='--')
        plt.show()

def plot_esquema_processo():
    """Gera um esquema simples da ordem dos processos com fluxo de massa/energia."""
    
    fig, ax = plt.subplots(figsize=(16, 6)) # Aumentei o tamanho X
    
    # Componentes e Posições (x, y) - Ajuste nas posições para incluir o Deoxo
    componentes = COMPONENTS
    posicoes = {
        'Entrada': (0, 0),
        'Dry Cooler 1': (2, 0),
        'Chiller 1': (4, 0),
        'KOD 1': (6, 0),
        'Coalescedor 1': (8, 0),
        'Deoxo': (10, 0), 
        'Dry Cooler 2': (12, 0),
        'Chiller 2': (14, 0),
        'KOD 2': (16, 0)
    }
    
    # 1. Desenha os retângulos dos componentes
    for comp, (x, y) in posicoes.items():
        if comp != 'Entrada':
            rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color='lightgray', alpha=0.6, ec='black', lw=1.5, zorder=1)
            ax.add_patch(rect)
            ax.text(x, y, comp, ha='center', va='center', fontsize=12, fontweight='bold', zorder=2)
        else:
            ax.text(x, y, comp, ha='center', va='center', fontsize=12, fontweight='bold', zorder=2)

    # 2. Desenha as setas (Fluxo de Massa)
    fluxo_y = 0.0
    for i in range(len(componentes) - 1):
        x_start = posicoes[componentes[i]][0] + 0.5
        x_end = posicoes[componentes[i+1]][0] - 0.5
        
        # Seta do fluxo principal (Massa)
        ax.annotate('', xy=(x_end, fluxo_y), xytext=(x_start, fluxo_y),
                    arrowprops=dict(facecolor='blue', shrink=0.05, width=2, headwidth=10))
        ax.text((x_start + x_end) / 2, fluxo_y + 0.3, 'Fluxo de Gás e Vapor', ha='center', va='bottom', fontsize=10, color='blue') 
        
    # 3. Desenha as setas de Energia (Calor/Trabalho)
    
    # Ciclo 1
    # Dry Cooler 1 (Q e W)
    ax.annotate('Q (Calor p/ Ar)', xy=(2.0, -1.0), xytext=(2.0, -0.6), arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=6), ha='center')
    ax.annotate('W (Ventilador)', xy=(2.5, 1.0), xytext=(2.5, 0.5), arrowprops=dict(facecolor='green', shrink=0.05, width=1, headwidth=6), ha='center')
    # Chiller 1 (Q e W)
    ax.annotate('Q (Refrigeração)', xy=(4.0, -1.0), xytext=(4.0, -0.6), arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=6), ha='center')
    ax.annotate('W (Elétrico)', xy=(4.5, 1.0), xytext=(4.5, 0.5), arrowprops=dict(facecolor='green', shrink=0.05, width=1, headwidth=6), ha='center')
    # KOD 1 (W e Água Condensada)
    ax.annotate('W (Perda P)', xy=(6.5, 1.0), xytext=(6.5, 0.5), arrowprops=dict(facecolor='green', shrink=0.05, width=1, headwidth=6), ha='center')
    ax.annotate('M_H2O (Líq)', xy=(6.0, -1.0), xytext=(6.0, -0.6), arrowprops=dict(facecolor='brown', shrink=0.05, width=1, headwidth=6), ha='center')
    # Coalescedor 1 (W e Aerossóis)
    ax.annotate('W (Perda P)', xy=(8.5, 1.0), xytext=(8.5, 0.5), arrowprops=dict(facecolor='green', shrink=0.05, width=1, headwidth=6), ha='center')
    ax.annotate('M_Liq (Aerossóis)', xy=(8.0, -1.0), xytext=(8.0, -0.6), arrowprops=dict(facecolor='brown', shrink=0.05, width=1, headwidth=6), ha='center')

    # Deoxo (Reação)
    ax.annotate('Q (Resfriamento)', xy=(10.0, -1.0), xytext=(10.0, -0.6), arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=6), ha='center')
    ax.annotate('H₂O (Produto)', xy=(10.5, -1.0), xytext=(10.5, -0.3), arrowprops=dict(facecolor='purple', shrink=0.05, width=1, headwidth=6), ha='center')
    ax.annotate('O₂ (Consumido)', xy=(9.5, 1.0), xytext=(9.5, 0.5), arrowprops=dict(facecolor='orange', shrink=0.05, width=1, headwidth=6), ha='center')
    
    # Ciclo 2 (Re-resfriamento)
    # Dry Cooler 2 (Q e W)
    ax.annotate('Q (Calor p/ Ar)', xy=(12.0, -1.0), xytext=(12.0, -0.6), arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=6), ha='center')
    ax.annotate('W (Ventilador)', xy=(12.5, 1.0), xytext=(12.5, 0.5), arrowprops=dict(facecolor='green', shrink=0.05, width=1, headwidth=6), ha='center')
    # Chiller 2 (Q e W)
    ax.annotate('Q (Refrigeração)', xy=(14.0, -1.0), xytext=(14.0, -0.6), arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=6), ha='center')
    ax.annotate('W (Elétrico)', xy=(14.5, 1.0), xytext=(14.5, 0.5), arrowprops=dict(facecolor='green', shrink=0.05, width=1, headwidth=6), ha='center')
    # KOD 2 (W e Água Condensada)
    ax.annotate('W (Perda P)', xy=(16.5, 1.0), xytext=(16.5, 0.5), arrowprops=dict(facecolor='green', shrink=0.05, width=1, headwidth=6), ha='center')
    ax.annotate('M_H2O (Líq)', xy=(16.0, -1.0), xytext=(16.0, -0.6), arrowprops=dict(facecolor='brown', shrink=0.05, width=1, headwidth=6), ha='center')


    ax.set_xlim(-1, 18) # Ajuste do limite X
    ax.set_ylim(-1.5, 1.5) 
    ax.axis('off')
    ax.set_title('Esquema do Sistema de Tratamento de Gás')
    plt.show()

# =================================================================
# === FUNÇÃO DE SIMULAÇÃO CENTRAL ===
# =================================================================

def simular_sistema(gas_fluido: str, m_dot_g_kg_s: float):
    
    # Rastreamento de impurezas
    y_O2_in = Y_O2_IN_H2 if gas_fluido == 'H2' else 0.0
    y_H2_in = Y_H2_IN_O2 if gas_fluido == 'O2' else 0.0

    # 1. ESTADO DE ENTRADA (INICIAL)
    y_H2O_in = calcular_y_H2O_inicial(T_IN_C, P_IN_BAR)
    estado_atual = calcular_estado_termodinamico(gas_fluido, T_IN_C, P_IN_BAR, m_dot_g_kg_s, y_H2O_in, y_O2_in, y_H2_in)
    
    # Inicializa o rastreamento
    history = [
        {
            **estado_atual,
            'Componente': 'Entrada',
            'Q_dot_fluxo_W': 0.0,
            'W_dot_comp_W': 0.0,
            'Agua_Condensada_kg_s': 0.0,
            'm_dot_H2O_vap_kg_s': estado_atual['m_dot_H2O_vap_kg_s'] 
        }
    ]
    
    # --- CICLO 1: DRY COOLER 1 / CHILLER 1 / KOD 1 / COALESCEDOR 1 ---
    
    # 2. DRY COOLER 1
    print(f"\n--- Processando {gas_fluido} no Dry Cooler 1 ---")
    res_dc = modelar_dry_cooler(gas_fluido, estado_atual['m_dot_gas_kg_s'], estado_atual['P_bar'], estado_atual['T_C'])
    estado_atual = calcular_estado_termodinamico(gas_fluido, res_dc['T_C'], res_dc['P_bar'], estado_atual['m_dot_gas_kg_s'], estado_atual['y_H2O'], estado_atual['y_O2'], estado_atual['y_H2'])
    
    history.append({
        **estado_atual,
        'Componente': 'Dry Cooler 1',
        'Q_dot_fluxo_W': res_dc['Q_dot_fluxo_W'],
        'W_dot_comp_W': res_dc['W_dot_comp_W'],
        'Agua_Condensada_kg_s': 0.0,
        'm_dot_H2O_vap_kg_s': estado_atual['m_dot_H2O_vap_kg_s']
    })

    # 3. CHILLER 1
    print(f"--- Processando {gas_fluido} no Chiller 1 ---")
    res_chiller = modelar_chiller_gas(
        gas_fluido, 
        estado_atual['m_dot_gas_kg_s'], 
        estado_atual['P_bar'], 
        estado_atual['T_C'], 
        T_CHILLER_OUT_C
    )
    estado_atual = calcular_estado_termodinamico(gas_fluido, res_chiller['T_C'], res_chiller['P_bar'], estado_atual['m_dot_gas_kg_s'], estado_atual['y_H2O'], estado_atual['y_O2'], estado_atual['y_H2'])
    
    history.append({
        **estado_atual,
        'Componente': 'Chiller 1',
        'Q_dot_fluxo_W': res_chiller['Q_dot_fluxo_W'],
        'W_dot_comp_W': res_chiller['W_dot_comp_W'],
        'Agua_Condensada_kg_s': 0.0,
        'm_dot_H2O_vap_kg_s': estado_atual['m_dot_H2O_vap_kg_s']
    })

    # 4. KNOCK-OUT DRUM 1 (KOD 1)
    print(f"--- Processando {gas_fluido} no KOD 1 ---")
    res_kod = modelar_knock_out_drum(
        gas_fluido, 
        estado_atual['m_dot_gas_kg_s'], 
        estado_atual['P_bar'], 
        estado_atual['T_C'], 
        estado_atual['y_H2O']
    )
    
    y_H2O_out_kod = res_kod['y_H2O']
    m_dot_gas_out_kod = res_kod['m_dot_gas_out_kg_s']
    F_molar_total_in = history[-1]['F_molar_total']
    F_H2O_molar_in = F_molar_total_in * history[-1]['y_H2O']
    F_gas_molar_const = F_molar_total_in - F_H2O_molar_in
    F_molar_total_out = F_gas_molar_const / (1.0 - y_H2O_out_kod)
    F_H2O_molar_out = F_molar_total_out * y_H2O_out_kod
    M_H2O = CP.PropsSI('M', 'Water')
    Agua_Condensada_kg_s_kod = (F_H2O_molar_in - F_H2O_molar_out) * M_H2O
    
    estado_atual = calcular_estado_termodinamico(gas_fluido, res_kod['T_C'], res_kod['P_bar'], m_dot_gas_out_kod, y_H2O_out_kod, estado_atual['y_O2'], estado_atual['y_H2'])
    
    history.append({
        **estado_atual,
        'Componente': 'KOD 1',
        'Q_dot_fluxo_W': res_kod['Q_dot_fluxo_W'],
        'W_dot_comp_W': res_kod['W_dot_comp_W'],
        'Agua_Condensada_kg_s': Agua_Condensada_kg_s_kod,
        'Status_KOD': res_kod['Status_KOD'],
        'm_dot_H2O_vap_kg_s': estado_atual['m_dot_H2O_vap_kg_s']
    })
    
    # 5. COALESCEDOR 1
    print(f"--- Processando {gas_fluido} no Coalescedor 1 ---")
    res_coal = modelar_coalescedor(
        gas_fluido,
        estado_atual['m_dot_gas_kg_s'],
        estado_atual['P_bar'],
        estado_atual['T_C'],
        estado_atual['y_H2O']
    )
    
    Agua_Aerossois_kg_s_coal = res_coal['Agua_Condensada_kg_s']
    estado_atual = calcular_estado_termodinamico(
        gas_fluido, 
        res_coal['T_C'], 
        res_coal['P_bar'], 
        estado_atual['m_dot_gas_kg_s'], 
        estado_atual['y_H2O'],
        estado_atual['y_O2'],
        estado_atual['y_H2']
    )
    
    history.append({
        **estado_atual,
        'Componente': 'Coalescedor 1',
        'Q_dot_fluxo_W': res_coal['Q_dot_fluxo_W'],
        'W_dot_comp_W': res_coal['W_dot_comp_W'],
        'Agua_Condensada_kg_s': Agua_Aerossois_kg_s_coal,
        'm_dot_H2O_vap_kg_s': estado_atual['m_dot_H2O_vap_kg_s'] 
    })
    
    # --- ETAPA CONDICIONAL: DEOXO (H2) ---
    
    # 6. DEOXO (Condicional: Apenas para H2)
    print(f"--- Processando {gas_fluido} no Deoxo ---")
    
    if gas_fluido == 'H2':
        res_deoxo = modelar_deoxo(
            estado_atual['m_dot_gas_kg_s'],
            estado_atual['P_bar'],
            estado_atual['T_C'],
            estado_atual['y_H2O'],
            estado_atual['y_O2']
        )
        
        # O Deoxo altera T, P, y_H2O (gera água), y_O2 (consome) e m_dot_gas_kg_s (consome H2)
        estado_atual_pos_deoxo = calcular_estado_termodinamico(
            gas_fluido, 
            res_deoxo['T_C'], 
            res_deoxo['P_bar'], 
            res_deoxo['m_dot_gas_out_kg_s'], 
            res_deoxo['y_H2O_out'],
            res_deoxo['y_O2_out'],
            estado_atual['y_H2'] 
        )
        
        extra_data = {
            'Agua_Condensada_kg_s': 0.0, 
            'Q_dot_fluxo_W': res_deoxo['Q_dot_fluxo_W'],
            'W_dot_comp_W': res_deoxo['W_dot_comp_W'],
            'm_dot_H2O_vap_kg_s': estado_atual_pos_deoxo['m_dot_H2O_vap_kg_s'],
            'T_profile_C': res_deoxo['T_profile_C'],
            'L_span': res_deoxo['L_span'],
            'X_O2': res_deoxo['X_O2']
        }
    
    else: # O2 não passa pelo Deoxo, apenas transborda
        estado_atual_pos_deoxo = estado_atual # Mantém o estado do Coalescedor 1
        extra_data = {
            'Agua_Condensada_kg_s': 0.0, 
            'Q_dot_fluxo_W': 0.0,
            'W_dot_comp_W': 0.0,
            'm_dot_H2O_vap_kg_s': estado_atual_pos_deoxo['m_dot_H2O_vap_kg_s'],
            'T_profile_C': None,
            'L_span': None,
            'X_O2': 0.0
        }
        
    history.append({
        **estado_atual_pos_deoxo,
        'Componente': 'Deoxo',
        **extra_data
    })
    
    # --- CICLO 2 (CONDICIONAL): DRY COOLER 2 / CHILLER 2 / KOD 2 ---
    
    if gas_fluido == 'H2':
        
        estado_atual = estado_atual_pos_deoxo
        
        # 7. DRY COOLER 2
        print(f"--- Processando {gas_fluido} no Dry Cooler 2 (Re-Cool) ---")
        res_dc2 = modelar_dry_cooler(gas_fluido, estado_atual['m_dot_gas_kg_s'], estado_atual['P_bar'], estado_atual['T_C'])
        estado_atual = calcular_estado_termodinamico(gas_fluido, res_dc2['T_C'], res_dc2['P_bar'], estado_atual['m_dot_gas_kg_s'], estado_atual['y_H2O'], estado_atual['y_O2'], estado_atual['y_H2'])
        
        history.append({
            **estado_atual,
            'Componente': 'Dry Cooler 2',
            'Q_dot_fluxo_W': res_dc2['Q_dot_fluxo_W'],
            'W_dot_comp_W': res_dc2['W_dot_comp_W'],
            'Agua_Condensada_kg_s': 0.0,
            'm_dot_H2O_vap_kg_s': estado_atual['m_dot_H2O_vap_kg_s']
        })
        
        # 8. CHILLER 2
        print(f"--- Processando {gas_fluido} no Chiller 2 (Re-Cool) ---")
        res_chiller2 = modelar_chiller_gas(
            gas_fluido, 
            estado_atual['m_dot_gas_kg_s'], 
            estado_atual['P_bar'], 
            estado_atual['T_C'], 
            T_CHILLER_OUT_C
        )
        estado_atual = calcular_estado_termodinamico(gas_fluido, res_chiller2['T_C'], res_chiller2['P_bar'], estado_atual['m_dot_gas_kg_s'], estado_atual['y_H2O'], estado_atual['y_O2'], estado_atual['y_H2'])
        
        history.append({
            **estado_atual,
            'Componente': 'Chiller 2',
            'Q_dot_fluxo_W': res_chiller2['Q_dot_fluxo_W'],
            'W_dot_comp_W': res_chiller2['W_dot_comp_W'],
            'Agua_Condensada_kg_s': 0.0,
            'm_dot_H2O_vap_kg_s': estado_atual['m_dot_H2O_vap_kg_s']
        })
        
        # 9. KNOCK-OUT DRUM 2 (KOD 2)
        print(f"--- Processando {gas_fluido} no KOD 2 (Re-Separação) ---")
        res_kod2 = modelar_knock_out_drum(
            gas_fluido, 
            estado_atual['m_dot_gas_kg_s'], 
            estado_atual['P_bar'], 
            estado_atual['T_C'], 
            estado_atual['y_H2O']
        )
        
        y_H2O_out_kod2 = res_kod2['y_H2O']
        m_dot_gas_out_kod2 = res_kod2['m_dot_gas_out_kg_s']
        F_molar_total_in = history[-1]['F_molar_total']
        F_H2O_molar_in = F_molar_total_in * history[-1]['y_H2O']
        F_gas_molar_const = F_molar_total_in - F_H2O_molar_in
        F_molar_total_out = F_gas_molar_const / (1.0 - y_H2O_out_kod2)
        F_H2O_molar_out = F_molar_total_out * y_H2O_out_kod2
        Agua_Condensada_kg_s_kod2 = (F_H2O_molar_in - F_H2O_molar_out) * M_H2O
        
        estado_atual = calcular_estado_termodinamico(gas_fluido, res_kod2['T_C'], res_kod2['P_bar'], m_dot_gas_out_kod2, y_H2O_out_kod2, estado_atual['y_O2'], estado_atual['y_H2'])
        
        history.append({
            **estado_atual,
            'Componente': 'KOD 2',
            'Q_dot_fluxo_W': res_kod2['Q_dot_fluxo_W'],
            'W_dot_comp_W': res_kod2['W_dot_comp_W'],
            'Agua_Condensada_kg_s': Agua_Condensada_kg_s_kod2,
            'Status_KOD': res_kod2['Status_KOD'],
            'm_dot_H2O_vap_kg_s': estado_atual['m_dot_H2O_vap_kg_s']
        })
        
    else:
        # Para O2, adicionamos componentes de 'transbordo' com estado final do Deoxo (Coalescedor 1)
        # Isso garante que os gráficos tenham o mesmo número de pontos.
        print(f"--- Fluxo {gas_fluido} ignorando re-resfriamento ---")
        estado_transbordo = history[-1]
        
        history.append({**estado_transbordo, 'Componente': 'Dry Cooler 2', 'Q_dot_fluxo_W': 0.0, 'W_dot_comp_W': 0.0, 'Agua_Condensada_kg_s': 0.0})
        history.append({**estado_transbordo, 'Componente': 'Chiller 2', 'Q_dot_fluxo_W': 0.0, 'W_dot_comp_W': 0.0, 'Agua_Condensada_kg_s': 0.0})
        history.append({**estado_transbordo, 'Componente': 'KOD 2', 'Q_dot_fluxo_W': 0.0, 'W_dot_comp_W': 0.0, 'Agua_Condensada_kg_s': 0.0, 'Status_KOD': 'N/A'})

    return pd.DataFrame(history)

# =================================================================
# === EXECUÇÃO DA SIMULAÇÃO E GRÁFICOS ===
# =================================================================

if __name__ == '__main__':
    
    # 1. Simular os dois fluidos
    df_h2 = simular_sistema(GASES[0], M_DOT_G_H2)
    df_o2 = simular_sistema(GASES[1], M_DOT_G_O2)
    
    print("\n\n" + "="*80)
    print("RESUMO DA SIMULAÇÃO COMPLETA (H2 e O2)")
    print("="*80)
    
    # 2. Exibir o estado final
    exibir_estado_final(df_h2, 'H2')
    exibir_estado_final(df_o2, 'O2')
    
    # 3. Gerar Gráficos
    print("\nGerando Gráficos de Evolução de Propriedades...")
    plot_propriedades(df_h2, df_o2)

    # Gráfico de Vazão Mássica
    print("\nGerando Gráfico de Evolução da Vazão Mássica de Gás...")
    plot_vazao_massica(df_h2, df_o2)

    # Gráfico de Rastreamento da Água (Vapor e Líquido)
    print("\nGerando Gráfico de Rastreamento de Água (Vapor no Fluxo e Líquido Removido)...")
    plot_vazao_agua(df_h2, df_o2)
    
    # Gráfico de Fluxos de Energia
    print("\nGerando Gráficos de Fluxos de Energia...")
    plot_fluxos_energia(df_h2, df_o2)

    # Gráfico de Água Removida Total
    print("\nGerando Gráfico da Vazão de Água Líquida Removida por Componente...")
    plot_agua_removida_total(df_h2, df_o2)
    
    # Gráfico do Perfil Deoxo (Apenas H2)
    print("\nGerando Gráfico do Perfil de Temperatura e Conversão no Reator Deoxo (H2)...")
    plot_deoxo_perfil(df_h2)
    
    # Gráfico de Esquema do Processo
    print("\nGerando Esquema do Processo...")
    plot_esquema_processo()