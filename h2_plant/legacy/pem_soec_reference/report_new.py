import matplotlib.pyplot as plt
import numpy as np
import pem_operator  # Importando para garantir consistência física (Área, T, Pressão)
import soec_operator

# ==============================================================================
# CONFIGURAÇÕES GERAIS E MAPA DE CORES
# ==============================================================================
plt.style.use('default') # Ou 'seaborn-v0_8-whitegrid' se preferir
COLORS = {
    'offer': 'black',
    'soec': '#4CAF50',    # Verde
    'pem': '#2196F3',     # Azul
    'sold': '#FF9800',    # Laranja
    'price': 'black',
    'ppa': 'red',
    'limit': 'blue',
    'h2_total': 'black',
    'water_total': 'navy'
}

# ==============================================================================
# GRUPO A: GRÁFICOS OPERACIONAIS (Baseados no Histórico do Manager)
# ==============================================================================

def plot_dispatch(history):
    """Gera o gráfico de despacho de potência (Oferta vs SOEC vs PEM vs Venda)."""
    minutes = history['minute']
    P_offer = np.array(history['P_offer'])
    P_soec = np.array(history['P_soec_actual'])
    P_pem = np.array(history['P_pem'])
    P_sold = np.array(history['P_sold'])

    plt.figure(figsize=(12, 6))
    
    # Áreas empilhadas
    plt.fill_between(minutes, 0, P_soec, label='Consumo SOEC', color=COLORS['soec'], alpha=0.6)
    plt.fill_between(minutes, P_soec, P_soec + P_pem, label='Consumo PEM', color=COLORS['pem'], alpha=0.6)
    plt.fill_between(minutes, P_soec + P_pem, P_soec + P_pem + P_sold, label='Venda ao Grid', color=COLORS['sold'], alpha=0.6)
    
    # Linhas de referência
    plt.plot(minutes, P_offer, label='Potência Ofertada', color=COLORS['offer'], linestyle='--', linewidth=1.5)
    
    plt.title('Despacho Híbrido: SOEC + PEM + Arbitragem', fontsize=14)
    plt.xlabel('Tempo (Minutos)')
    plt.ylabel('Potência (MW)')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('report_dispatch.png')
    plt.close() # Importante para liberar memória
    print("   -> Salvo: report_dispatch.png")

def plot_arbitrage(history):
    """Gera o gráfico de preços e decisões de arbitragem."""
    minutes = history['minute']
    spot_price = np.array(history['spot_price'])
    sell_decision = np.array(history['sell_decision'])
    
    # Preço PPA (Fixo no Manager, mas idealmente passaria como argumento. Usando valor típico do código)
    PPA_PRICE = 50.0 
    
    plt.figure(figsize=(12, 6))
    plt.plot(minutes, spot_price, label='Preço Spot (EUR/MWh)', color=COLORS['price'])
    plt.axhline(y=PPA_PRICE, color=COLORS['ppa'], linestyle='--', label='Preço PPA (Contrato)')
    
    # Pontos de Venda
    sell_idx = np.where(sell_decision == 1)[0]
    if len(sell_idx) > 0:
        plt.scatter(np.array(minutes)[sell_idx], np.array(spot_price)[sell_idx], 
                   color='red', zorder=5, label='Decisão: Venda')

    plt.title('Cenário de Preços e Decisão de Arbitragem', fontsize=14)
    plt.xlabel('Tempo (Minutos)')
    plt.ylabel('Preço (EUR/MWh)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('report_arbitrage.png')
    plt.close()
    print("   -> Salvo: report_arbitrage.png")

def plot_h2_production(history):
    """Gera gráfico de produção total de Hidrogênio."""
    minutes = history['minute']
    H2_soec = np.array(history['H2_soec_kg'])
    H2_pem = np.array(history['H2_pem_kg'])
    H2_total = H2_soec + H2_pem

    plt.figure(figsize=(12, 6))
    plt.fill_between(minutes, 0, H2_soec, label='H2 SOEC', color=COLORS['soec'], alpha=0.5)
    plt.fill_between(minutes, H2_soec, H2_total, label='H2 PEM', color=COLORS['pem'], alpha=0.5)
    plt.plot(minutes, H2_total, color=COLORS['h2_total'], linestyle='--', label='Total H2')

    plt.title('Produção de Hidrogênio Acumulada (kg/min)', fontsize=14)
    plt.ylabel('Taxa de Produção (kg/min)')
    plt.xlabel('Tempo (Minutos)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('report_h2_production.png')
    plt.close()
    print("   -> Salvo: report_h2_production.png")

def plot_water_consumption(history):
    """Gera gráfico de consumo de água (Reação + Extra)."""
    minutes = history['minute']
    # Recalculando lógica simples de extra aqui para plotagem
    water_soec = np.array(history['steam_soec_kg']) * 1.10 # 10% extra
    water_pem = np.array(history['H2O_pem_kg']) * 1.02 # 2% extra
    total = water_soec + water_pem

    plt.figure(figsize=(12, 6))
    plt.fill_between(minutes, 0, water_soec, label='H2O SOEC (Total)', color=COLORS['soec'], alpha=0.5)
    plt.fill_between(minutes, water_soec, total, label='H2O PEM (Total)', color='brown', alpha=0.5)
    plt.plot(minutes, total, color=COLORS['water_total'], linestyle='--', label='Consumo Total H2O')

    plt.title('Consumo de Água Total (Incluindo Perdas/Vapor)', fontsize=14)
    plt.ylabel('Fluxo de Água (kg/min)')
    plt.xlabel('Tempo (Minutos)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('report_water_consumption.png')
    plt.close()
    print("   -> Salvo: report_water_consumption.png")

def plot_energy_pie(history):
    """Gera gráfico de rosca (Donut Chart) profissional da distribuição de energia."""
    E_soec = np.sum(history['P_soec_actual']) / 60.0
    E_pem = np.sum(history['P_pem']) / 60.0
    E_sold = np.sum(history['P_sold']) / 60.0
    E_total = E_soec + E_pem + E_sold
    
    sizes = [E_soec, E_pem, E_sold]
    labels = ['SOEC', 'PEM', 'Venda ao Grid']
    colors = [COLORS['soec'], COLORS['pem'], COLORS['sold']]
    
    # Filtra fatias zeradas
    valid_sizes = []
    valid_labels = []
    valid_colors = []
    for s, l, c in zip(sizes, labels, colors):
        if s > 0.01:
            valid_sizes.append(s)
            valid_labels.append(l)
            valid_colors.append(c)

    if not valid_sizes:
        print("   [!] Aviso: Nenhuma energia consumida para gerar gráfico de pizza.")
        return

    # Criação do Donut Chart
    fig, ax = plt.subplots(figsize=(9, 8))
    
    # Explode a fatia de venda se ela existir para destaque
    explode = [0.05 if 'Venda' in l else 0 for l in valid_labels]
    
    wedges, texts, autotexts = ax.pie(
        valid_sizes, 
        explode=explode, 
        labels=valid_labels, 
        colors=valid_colors, 
        autopct='%1.1f%%', 
        pctdistance=0.85,
        startangle=140,
        wedgeprops=dict(width=0.4, edgecolor='w'), # Largura da rosca
        textprops=dict(color="black", fontweight='bold')
    )
    
    # Adiciona texto central com o Total
    ax.text(0, 0, f"TOTAL\n{E_total:.1f}\nMWh", ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Legenda externa melhorada
    legend_labels = [f"{l}: {s:.1f} MWh" for l, s in zip(valid_labels, valid_sizes)]
    ax.legend(wedges, legend_labels, title="Distribuição Energética", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    
    plt.title('Distribuição de Energia Consumida/Vendida', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('report_energy_pie.png')
    plt.close()
    print("   -> Salvo: report_energy_pie.png (Versão Donut Profissional)")


# ==============================================================================
# GRUPO B: GRÁFICOS TEÓRICOS/FÍSICOS (Baseados nas Constantes Reais do PEM)
# ==============================================================================

def _calculate_pem_physics_curves():
    """Função auxiliar interna para gerar as curvas V-j usando constantes REAIS do operador."""
    # 1. Recuperar constantes REAIS do arquivo pem_operator
    F = pem_operator.F
    R = pem_operator.R
    T = pem_operator.T
    P_op = pem_operator.P_op
    P_ref = pem_operator.P_ref
    z = pem_operator.z
    alpha = pem_operator.alpha
    j0 = pem_operator.j0
    delta_mem = pem_operator.delta_mem
    sigma = pem_operator.sigma_base
    j_lim = pem_operator.j_lim
    
    # 2. Gerar vetores
    j_range = np.linspace(0.01, j_lim * 0.95, 200)
    
    # 3. Recalcular tensões (reproduzindo lógica do operador localmente para plotagem)
    # Tensão Reversível
    U_rev = 1.229 - 0.9e-3 * (T - 298.15) + (R * T) / (z * F) * np.log((P_op / P_ref)**1.5)
    
    # Perdas
    eta_act = (R * T) / (alpha * z * F) * np.log(j_range / j0)
    eta_ohm = j_range * (delta_mem / sigma)
    # Proteção contra log(0) ou log(negativo) no termo de concentração
    limit_term = np.maximum(1e-6, j_lim - j_range)
    eta_conc = (R * T) / (z * F) * np.log(j_lim / limit_term)
    
    V_total = U_rev + eta_act + eta_ohm + eta_conc
    
    return j_range, U_rev, eta_act, eta_ohm, V_total

def plot_physics_polarization():
    """Gera a curva de polarização detalhada do PEM."""
    j_range, U_rev, eta_act, eta_ohm, V_total = _calculate_pem_physics_curves()
    
    plt.figure(figsize=(10, 6))
    
    # Plotar áreas empilhadas de perdas
    plt.fill_between(j_range, 0, U_rev, color='blue', alpha=0.2, label='Tensão Reversível (Termo)')
    plt.fill_between(j_range, U_rev, U_rev + eta_act, color='green', alpha=0.2, label='Perda Ativação')
    plt.fill_between(j_range, U_rev + eta_act, U_rev + eta_act + eta_ohm, color='orange', alpha=0.2, label='Perda Ôhmica')
    
    # Linha Total
    plt.plot(j_range, V_total, color='red', linewidth=2, label='Tensão da Célula (Total)')
    
    # Ponto Nominal
    plt.axvline(x=pem_operator.j_nom, color='black', linestyle='--', label=f'Nominal ({pem_operator.j_nom} A/cm²)')
    
    plt.title('Física do PEM: Curva de Polarização e Perdas (Constantes Reais)', fontsize=14)
    plt.xlabel('Densidade de Corrente (A/cm²)')
    plt.ylabel('Tensão (V)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('report_physics_polarization.png')
    plt.close()
    print("   -> Salvo: report_physics_polarization.png")

def plot_physics_efficiency():
    """Gera a curva de eficiência DO SISTEMA (Considerando BoP)."""
    # 1. Recupera vetores físicos
    j_range, _, _, _, V_total = _calculate_pem_physics_curves()
    
    # 2. Recupera constantes do Operador
    P_bop_fixo = pem_operator.P_bop_fixo
    k_bop_var = pem_operator.k_bop_var
    Area_Total = pem_operator.Area_Total
    
    # 3. Calcula Potências
    I_total = j_range * Area_Total # Corrente Total (A)
    P_stack = I_total * V_total    # Potência do Stack (W)
    P_system = P_stack + P_bop_fixo + (k_bop_var * P_stack) # Potência Total (W)
    
    # 4. Calcula Energia do H2 Produzido (LHV)
    # Fluxo H2 (kg/s) = (I * n * M) / (z * F) -> Simplificando para Potência Química (W)
    # LHV H2 ~ 1.254 V de potencial equivalente energético
    P_hydrogen_chemical = I_total * 1.254 # Aproximação W = A * V_equiv
    
    # 5. Eficiência do Sistema
    # Evita divisão por zero se P_system for muito baixo
    efficiency = np.divide(P_hydrogen_chemical, P_system, out=np.zeros_like(P_system), where=P_system!=0) * 100
    
    plt.figure(figsize=(10, 6))
    plt.plot(j_range, efficiency, color='green', linewidth=2, label='Eficiência do Sistema (Stack + BoP)')
    
    # Plota a eficiência antiga (só stack) pontilhada para comparação
    eff_stack_only = (1.254 / V_total) * 100
    plt.plot(j_range, eff_stack_only, color='gray', linestyle='--', alpha=0.5, label='Eficiência Apenas Stack (Teórica)')

    plt.axvline(x=pem_operator.j_nom, color='black', linestyle='--', label='Ponto Nominal')
    
    plt.title('Curva de Eficiência Real: Sistema vs Stack', fontsize=14)
    plt.xlabel('Densidade de Corrente (A/cm²)')
    plt.ylabel('Eficiência (% LHV)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('report_physics_efficiency.png')
    plt.close()
    print("   -> Salvo: report_physics_efficiency.png (Atualizado para Sistema)")



def plot_physics_power_balance():
 
    j_range, _, _, _, _, V_total = _calculate_pem_physics_curves()
    
    # Recupera constantes REAIS do operador
    Area_Total = pem_operator.Area_Total # cm2
    P_bop_fixo = pem_operator.P_bop_fixo # Watts
    k_bop_var = pem_operator.k_bop_var   # %
    
    # Cálculos de Potência (W -> kW)
    I_total = j_range * Area_Total
    P_stack_W = I_total * V_total
    P_bop_var_W = k_bop_var * P_stack_W
    P_total_W = P_stack_W + P_bop_fixo + P_bop_var_W
    
    # Conversão para kW
    P_stack_kW = P_stack_W / 1000.0
    P_total_kW = P_total_W / 1000.0
    
    plt.figure(figsize=(10, 6))
    
    # Área do Stack
    plt.fill_between(j_range, 0, P_stack_kW, color=COLORS['pem'], alpha=0.3, label='Potência Útil (Stack)')
    
    # Área do BoP (Diferença entre Total e Stack)
    plt.fill_between(j_range, P_stack_kW, P_total_kW, color=COLORS['loss_bop'], alpha=0.5, label='Perda BoP (Auxiliares)')
    
    # Linha Total
    plt.plot(j_range, P_total_kW, color='darkred', linewidth=2, label='Consumo Total do Sistema')
    
    # Linha Nominal
    plt.axvline(x=pem_operator.j_nom, color='black', linestyle='--', alpha=0.5)
    plt.text(pem_operator.j_nom, max(P_total_kW)*0.1, f' Nominal', rotation=90)
    
    plt.title('Balanço de Potência: Stack vs Balance of Plant (BoP)', fontsize=14, fontweight='bold')
    plt.xlabel('Densidade de Corrente (A/cm²)')
    plt.ylabel('Potência (kW)')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('report_physics_power_balance.png')
    plt.close()
    print("   -> Salvo: report_physics_power_balance.png")

    
def plot_degradation_projection(pem_state):
    """Plota a projeção de degradação baseada no estado atual."""
    # Recupera a tabela de degradação do operador
    time_table = pem_operator.T_OP_H_TABLE / 8760.0 # Anos
    voltage_table = pem_operator.V_CELL_TABLE # V
    
    current_time_years = pem_state.t_op_h / 8760.0
    
    plt.figure(figsize=(10, 6))
    plt.plot(time_table, voltage_table, marker='o', label='Curva de Degradação (Modelo)')
    
    # Ponto atual
    # Precisamos calcular a tensão atual (baseado em j_nom para referência)
    V_current = pem_operator.calculate_Vcell(pem_operator.j_nom, pem_operator.T, pem_operator.P_op, pem_state.t_op_h)
    
    plt.scatter([current_time_years], [V_current], color='red', s=100, zorder=5, label='Estado Atual da Simulação')
    
    plt.title('Evolução da Tensão Nominal (Degradação)', fontsize=14)
    plt.axhline(y=2.0, color='green', linestyle='--', alpha=0.5, label='BOL')
    plt.axhline(y=2.2, color='red', linestyle='--', alpha=0.5, label='EOL')
    plt.xlabel('Tempo de Operação (Anos)')
    plt.ylabel('Tensão de Célula (V)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('report_degradation.png')
    plt.close()
    print("   -> Salvo: report_degradation.png")

# ==============================================================================
# FUNÇÃO MASTER: CONTROLADOR DE RELATÓRIOS
# ==============================================================================

def generate_selected_reports(history, pem_state=None, selection=['all']):
    """
    Função principal chamada pelo Manager.
    
    Args:
        history (dict): Dados da simulação.
        pem_state (obj): Objeto de estado do PEM (para gráficos físicos).
        selection (list): Lista de strings com os gráficos desejados.
                          Opções: 'dispatch', 'arbitrage', 'pie', 'h2', 'water',
                                  'polarization', 'efficiency', 'degradation', 'all'.
    """
    print("\n--- [REPORT] Iniciando Geração de Gráficos ---")
    
    # Mapa de seleções para funções
    # Operacionais
    ops_map = {
        'dispatch': lambda: plot_dispatch(history),
        'arbitrage': lambda: plot_arbitrage(history),
        'pie': lambda: plot_energy_pie(history),
        'h2': lambda: plot_h2_production(history),
        'water': lambda: plot_water_consumption(history)
    }
    
    # Físicos (exigem pem_state ou apenas acesso ao modulo)
    phys_map = {
        'polarization': plot_physics_polarization,
        'efficiency': plot_physics_efficiency,
        'degradation': lambda: plot_degradation_projection(pem_state) if pem_state else print("   [!] Erro: pem_state necessário para gráfico de degradação."),
    }
    
    run_all = 'all' in selection
    
    # Executar Operacionais
    for key, func in ops_map.items():
        if run_all or key in selection:
            func()
            
    # Executar Físicos
    for key, func in phys_map.items():
        if run_all or key in selection:
            func()
            
    print("--- [REPORT] Geração Concluída ---\n")

# Bloco de teste direto (se rodar o arquivo isolado)
if __name__ == "__main__":
    print("Este módulo deve ser importado pelo manager.py.")

