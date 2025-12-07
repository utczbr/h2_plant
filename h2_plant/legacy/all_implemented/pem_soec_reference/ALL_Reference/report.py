import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # Necessário para resample (médias temporais)
import pem_operator  
import soec_operator

# ==============================================================================
# GENERAL SETTINGS AND COLOR MAP
# ==============================================================================
plt.style.use('default') 
COLORS = {
    'offer': 'black',
    'soec': '#4CAF50',    # Green
    'pem': '#2196F3',     # Blue
    'sold': '#FF9800',    # Orange
    'price': 'black',
    'ppa': 'red',
    'limit': 'blue',
    'h2_total': 'black',
    'water_total': 'navy',
    'oxygen': 'purple'    # New color for O2
}

# ==============================================================================
# GROUP A: OPERATIONAL PLOTS (Based on Manager History)
# ==============================================================================

def plot_dispatch(history):
    """Generates the power dispatch plot (Offer vs SOEC vs PEM vs Sale)."""
    minutes = history['minute']
    P_offer = np.array(history['P_offer'])
    P_soec = np.array(history['P_soec_actual'])
    P_pem = np.array(history['P_pem'])
    P_sold = np.array(history['P_sold'])

    plt.figure(figsize=(12, 6))
    
    # Stacked areas
    plt.fill_between(minutes, 0, P_soec, label='SOEC Consumption', color=COLORS['soec'], alpha=0.6)
    plt.fill_between(minutes, P_soec, P_soec + P_pem, label='PEM Consumption', color=COLORS['pem'], alpha=0.6)
    plt.fill_between(minutes, P_soec + P_pem, P_soec + P_pem + P_sold, label='Grid Sale', color=COLORS['sold'], alpha=0.6)
    
    # Reference lines
    plt.plot(minutes, P_offer, label='Offered Power', color=COLORS['offer'], linestyle='--', linewidth=1.5)
    
    plt.title('Hybrid Dispatch: SOEC + PEM + Arbitrage', fontsize=14)
    plt.xlabel('Time (Minutes)')
    plt.ylabel('Power (MW)')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('report_dispatch.png')
    plt.close()
    print("   -> Saved: report_dispatch.png")

def plot_arbitrage(history, h2_price_eur_kg=9.6):
    """Generates price plot with an extra line for H2 Breakeven."""
    minutes = history['minute']
    spot_price = np.array(history['spot_price'])
    sell_decision = np.array(history['sell_decision'])
    
    PPA_PRICE = 50.0 
    
    EFF_ESTIMATE_MWH_KG = 0.05 # 50 kWh
    H2_EQUIV_PRICE = h2_price_eur_kg / EFF_ESTIMATE_MWH_KG

    plt.figure(figsize=(12, 6))
    plt.plot(minutes, spot_price, label='Spot Price (EUR/MWh)', color=COLORS['price'])
    plt.axhline(y=PPA_PRICE, color=COLORS['ppa'], linestyle='--', label='PPA Price (Contract)')
    
    plt.axhline(y=H2_EQUIV_PRICE, color='green', linestyle='-.', label=f'H2 Breakeven (~{H2_EQUIV_PRICE:.0f} EUR/MWh)')
    
    sell_idx = np.where(sell_decision == 1)[0]
    if len(sell_idx) > 0:
        plt.scatter(np.array(minutes)[sell_idx], np.array(spot_price)[sell_idx], 
                   color='red', zorder=5, label='Decision: Sell')

    plt.title('Price Scenario, PPA and H2 Opportunity Cost', fontsize=14)
    plt.xlabel('Time (Minutes)')
    plt.ylabel('Price (EUR/MWh)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('report_arbitrage.png')
    plt.close()
    print("   -> Saved: report_arbitrage.png")

def plot_h2_production(history):
    """Generates total Hydrogen production plot."""
    minutes = history['minute']
    H2_soec = np.array(history['H2_soec_kg'])
    H2_pem = np.array(history['H2_pem_kg'])
    H2_total = H2_soec + H2_pem

    plt.figure(figsize=(12, 6))
    plt.fill_between(minutes, 0, H2_soec, label='H2 SOEC', color=COLORS['soec'], alpha=0.5)
    plt.fill_between(minutes, H2_soec, H2_total, label='H2 PEM', color=COLORS['pem'], alpha=0.5)
    plt.plot(minutes, H2_total, color=COLORS['h2_total'], linestyle='--', label='Total H2')

    plt.title('Accumulated Hydrogen Production (kg/min)', fontsize=14)
    plt.ylabel('Production Rate (kg/min)')
    plt.xlabel('Time (Minutes)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('report_h2_production.png')
    plt.close()
    print("   -> Saved: report_h2_production.png")

def plot_oxygen_production(history):
    """Generates joint Oxygen production plot."""
    minutes = history['minute']
    
    if 'O2_pem_kg' in history:
         O2_pem = np.array(history['O2_pem_kg'])
    else:
         O2_pem = np.array(history['H2_pem_kg']) * 8.0
         
    O2_soec = np.array(history['H2_soec_kg']) * 8.0 
    O2_total = O2_soec + O2_pem

    plt.figure(figsize=(12, 6))
    plt.fill_between(minutes, 0, O2_soec, label='O2 SOEC', color=COLORS['soec'], alpha=0.5)
    plt.fill_between(minutes, O2_soec, O2_total, label='O2 PEM', color='purple', alpha=0.5)
    plt.plot(minutes, O2_total, color='black', linestyle='--', label='Total O2')

    plt.title('Joint Oxygen Production (kg/min)', fontsize=14)
    plt.ylabel('Production Rate (kg/min)')
    plt.xlabel('Time (Minutes)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('report_oxygen_production.png')
    plt.close()
    print("   -> Saved: report_oxygen_production.png")

def plot_water_consumption(history):
    """Generates water consumption plot."""
    minutes = history['minute']
    water_soec = np.array(history['steam_soec_kg']) * 1.10 
    water_pem = np.array(history['H2O_pem_kg']) * 1.02 
    total = water_soec + water_pem

    plt.figure(figsize=(12, 6))
    plt.fill_between(minutes, 0, water_soec, label='H2O SOEC (Total)', color=COLORS['soec'], alpha=0.5)
    plt.fill_between(minutes, water_soec, total, label='H2O PEM (Total)', color='brown', alpha=0.5)
    plt.plot(minutes, total, color=COLORS['water_total'], linestyle='--', label='Total H2O Consumption')

    plt.title('Total Water Consumption (Including Losses/Steam)', fontsize=14)
    plt.ylabel('Water Flow (kg/min)')
    plt.xlabel('Time (Minutes)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('report_water_consumption.png')
    plt.close()
    print("   -> Saved: report_water_consumption.png")

def plot_energy_pie(history):
    """Generates donut chart of energy distribution."""
    E_soec = np.sum(history['P_soec_actual']) / 60.0
    E_pem = np.sum(history['P_pem']) / 60.0
    E_sold = np.sum(history['P_sold']) / 60.0
    E_total = E_soec + E_pem + E_sold
    
    sizes = [E_soec, E_pem, E_sold]
    labels = ['SOEC', 'PEM', 'Grid Sale']
    colors = [COLORS['soec'], COLORS['pem'], COLORS['sold']]
    
    valid_sizes = []
    valid_labels = []
    valid_colors = []
    for s, l, c in zip(sizes, labels, colors):
        if s > 0.01:
            valid_sizes.append(s)
            valid_labels.append(l)
            valid_colors.append(c)

    if not valid_sizes:
        print("   [!] Warning: No energy consumed to generate pie chart.")
        return

    fig, ax = plt.subplots(figsize=(9, 8))
    explode = [0.05 if 'Sale' in l else 0 for l in valid_labels]
    
    wedges, texts, autotexts = ax.pie(
        valid_sizes, 
        explode=explode, 
        labels=valid_labels, 
        colors=valid_colors, 
        autopct='%1.1f%%', 
        pctdistance=0.85,
        startangle=140,
        wedgeprops=dict(width=0.4, edgecolor='w'),
        textprops=dict(color="black", fontweight='bold')
    )
    
    ax.text(0, 0, f"TOTAL\n{E_total:.1f}\nMWh", ha='center', va='center', fontsize=12, fontweight='bold')
    legend_labels = [f"{l}: {s:.1f} MWh" for l, s in zip(valid_labels, valid_sizes)]
    ax.legend(wedges, legend_labels, title="Energy Distribution", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    
    plt.title('Consumed/Sold Energy Distribution', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('report_energy_pie.png')
    plt.close()
    print("   -> Saved: report_energy_pie.png")

def plot_price_histogram(history):
    """Generates price distribution histogram."""
    spot_price = np.array(history['spot_price'])
    
    plt.figure(figsize=(10, 6))
    plt.hist(spot_price, bins=30, color='gray', edgecolor='black', alpha=0.7)
    
    mean_price = np.mean(spot_price)
    max_price = np.max(spot_price)
    
    plt.axvline(mean_price, color='blue', linestyle='--', linewidth=2, label=f'Mean: {mean_price:.2f} EUR')
    plt.axvline(max_price, color='red', linestyle='--', linewidth=2, label=f'Max: {max_price:.2f} EUR')
    
    plt.title('Spot Price Frequency Distribution', fontsize=14)
    plt.xlabel('Price (EUR/MWh)')
    plt.ylabel('Frequency (Minutes)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('report_price_histogram.png')
    plt.close()
    print("   -> Saved: report_price_histogram.png")

def plot_dispatch_curve(history):
    """Generates Dispatch Curve: H2 Produced vs Total Power."""
    P_total = np.array(history['P_soec_actual']) + np.array(history['P_pem'])
    H2_total = np.array(history['H2_soec_kg']) + np.array(history['H2_pem_kg'])
    
    plt.figure(figsize=(10, 6))
    plt.scatter(P_total, H2_total, alpha=0.5, c='blue', edgecolors='none')
    
    plt.title('Real Dispatch Curve: H2 Production vs Power', fontsize=14)
    plt.xlabel('Total Input Power (MW)')
    plt.ylabel('H2 Production (kg/min)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('report_dispatch_curve.png')
    plt.close()
    print("   -> Saved: report_dispatch_curve.png")

# ==============================================================================
# NOVO GRUPO: MÉDIAS TEMPORAIS (Hourly, Daily, Monthly)
# ==============================================================================

def plot_temporal_averages(history):
    """
    NOVA FUNÇÃO: Gera gráficos de médias horárias, diárias e mensais.
    Usa pandas para resample (agregação).
    """
    print("   -> Generating Temporal Average Reports...")
    
    # 1. Converter Histórico para DataFrame Pandas
    df = pd.DataFrame(history)
    
    # Criação de um Índice Temporal (TimeIndex)
    # Como o histórico original pode não ter data real, criamos uma dummy começando em 2024
    # Se o seu manager já passar datas, seria ideal usar aqui, mas isso garante que funcione sempre.
    start_date = "2024-01-01 00:00"
    df.index = pd.date_range(start=start_date, periods=len(df), freq='min')
    
    # Adicionar colunas calculadas úteis
    df['H2_total_kg'] = df['H2_soec_kg'] + df['H2_pem_kg']
    
    # Definição dos períodos para plotagem
    # (NomeArquivo, CodigoPandas, Titulo)
    periods = [
        ('hourly', 'h', 'Hourly Average (Média Horária)'),
        ('daily', 'D', 'Daily Average (Média Diária)'),
        ('monthly', 'ME', 'Monthly Average (Média Mensal)') # 'ME' = Month End
    ]
    
    for fname, freq_code, title_text in periods:
        # Tenta fazer o Resample (Cálculo da Média)
        try:
            # Observação: Para potência e preço, a MÉDIA faz sentido.
            # Para produção (kg/min), a média continua sendo "taxa média".
            df_res = df.resample(freq_code).mean()
        except ValueError:
            # Fallback para pandas antigo se 'ME' falhar
            if freq_code == 'ME':
                df_res = df.resample('M').mean()
            else:
                continue

        # Se a simulação for muito curta para esse período (ex: simulação de 8h tentando plotar mensal), pula
        if len(df_res) < 2:
            print(f"      [i] Skipping {fname} plot: Not enough data points (Simulated time too short).")
            continue

        # Configuração do Plot (3 Subplots verticais)
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # 1. Preço Médio
        axes[0].plot(df_res.index, df_res['spot_price'], color='black', marker='.', linestyle='-', linewidth=1, label='Avg Spot Price')
        axes[0].set_ylabel('Price (EUR/MWh)')
        axes[0].set_title(f'{title_text}: Spot Market Prices')
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)
        
        # 2. Potência Média (Stacked)
        axes[1].stackplot(df_res.index, 
                          df_res['P_soec_actual'], 
                          df_res['P_pem'], 
                          df_res['P_sold'],
                          labels=['SOEC Power', 'PEM Power', 'Sold Power'],
                          colors=[COLORS['soec'], COLORS['pem'], COLORS['sold']], 
                          alpha=0.7)
        axes[1].set_ylabel('Avg Power (MW)')
        axes[1].set_title(f'{title_text}: Power Dispatch Distribution')
        axes[1].legend(loc='upper left')
        axes[1].grid(True, alpha=0.3)
        
        # 3. Taxa de Produção Média (H2)
        axes[2].plot(df_res.index, df_res['H2_total_kg'], color=COLORS['h2_total'], linewidth=2, label='Avg H2 Rate (kg/min)')
        # Opcional: Adicionar área sombreada para SOEC vs PEM se desejar, mas linha é mais limpa para médias
        axes[2].fill_between(df_res.index, 0, df_res['H2_total_kg'], color=COLORS['h2_total'], alpha=0.1)
        
        axes[2].set_ylabel('Production Rate (kg/min)')
        axes[2].set_title(f'{title_text}: Hydrogen Production Rate')
        axes[2].legend(loc='upper left')
        axes[2].grid(True, alpha=0.3)
        
        plt.xlabel('Simulation Time')
        plt.tight_layout()
        
        filename = f'report_avg_{fname}.png'
        plt.savefig(filename)
        plt.close()
        print(f"   -> Saved: {filename}")


# ==============================================================================
# GROUP B: THEORETICAL/PHYSICAL PLOTS (Based on Real PEM Constants)
# ==============================================================================

def _calculate_pem_physics_curves(pem_state=None, override_t_op_h=None):
    """Internal helper function to generate V-j curves."""
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
    
    j_range = np.linspace(0.01, j_lim * 0.95, 200)
    
    # Physical Calculations
    U_rev = 1.229 - 0.9e-3 * (T - 298.15) + (R * T) / (z * F) * np.log((P_op / P_ref)**1.5)
    eta_act = (R * T) / (alpha * z * F) * np.log(j_range / j0)
    eta_ohm = j_range * (delta_mem / sigma)
    limit_term = np.maximum(1e-6, j_lim - j_range)
    eta_conc = (R * T) / (z * F) * np.log(j_lim / limit_term)
    
    # Degradation Logic
    t_op = 0.0
    if override_t_op_h is not None:
        t_op = override_t_op_h
    elif pem_state is not None:
        t_op = pem_state.t_op_h
        
    U_deg = pem_operator.calculate_U_deg_from_table(t_op)
    
    V_total = U_rev + eta_act + eta_ohm + eta_conc + U_deg
    
    return j_range, U_rev, eta_act, eta_ohm, eta_conc, V_total, U_deg

def plot_physics_polarization(pem_state=None):
    """Generates polarization curve with BOL, EOL and Current State."""
    
    j_range, U_rev, eta_act, eta_ohm, eta_conc, V_total, U_deg = _calculate_pem_physics_curves(pem_state)
    _, _, _, _, _, V_bol, _ = _calculate_pem_physics_curves(override_t_op_h=0)
    _, _, _, _, _, V_eol, _ = _calculate_pem_physics_curves(override_t_op_h=87600)
    
    plt.figure(figsize=(10, 6))
    plt.fill_between(j_range, 0, U_rev, color='blue', alpha=0.05, label='Reversible Voltage')
    plt.plot(j_range, V_bol, color='green', linestyle='--', alpha=0.8, label='BOL (Beginning of Life)')
    plt.plot(j_range, V_eol, color='red', linestyle='--', alpha=0.8, label='EOL (10 Years)')
    
    label_total = f'Total Voltage (Year {pem_state.t_op_h/8760:.1f})' if pem_state else 'Total Voltage (Current)'
    plt.plot(j_range, V_total, color='blue', linewidth=2, label=label_total)
    
    plt.axvline(x=pem_operator.j_nom, color='black', linestyle='--', label=f'Nominal ({pem_operator.j_nom} A/cm²)')
    
    plt.title('PEM Physics: BOL vs Current vs EOL Comparison', fontsize=14)
    plt.xlabel('Current Density (A/cm²)')
    plt.ylabel('Voltage (V)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('report_physics_polarization.png')
    plt.close()
    print("   -> Saved: report_physics_polarization.png")

def plot_physics_efficiency():
    """Generates SYSTEM efficiency curve."""
    j_range, _, _, _, _, V_total, _ = _calculate_pem_physics_curves()
    
    P_bop_fixo = pem_operator.P_bop_fixo
    k_bop_var = pem_operator.k_bop_var
    Area_Total = pem_operator.Area_Total
    
    I_total = j_range * Area_Total 
    P_stack = I_total * V_total    
    P_system = P_stack + P_bop_fixo + (k_bop_var * P_stack)
    
    P_hydrogen_chemical = I_total * 1.254 
    efficiency = np.divide(P_hydrogen_chemical, P_system, out=np.zeros_like(P_system), where=P_system!=0) * 100
    
    plt.figure(figsize=(10, 6))
    plt.plot(j_range, efficiency, color='green', linewidth=2, label='System Efficiency (Stack + BoP)')
    
    eff_stack_only = (1.254 / V_total) * 100
    plt.plot(j_range, eff_stack_only, color='gray', linestyle='--', alpha=0.5, label='Stack Only Efficiency')

    plt.axvline(x=pem_operator.j_nom, color='black', linestyle='--', label='Nominal Point')
    
    plt.title('Real Efficiency Curve: System vs Stack', fontsize=14)
    plt.xlabel('Current Density (A/cm²)')
    plt.ylabel('Efficiency (% LHV)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('report_physics_efficiency.png')
    plt.close()
    print("   -> Saved: report_physics_efficiency.png")

def plot_physics_power_balance():
    """Generates power balance."""
    j_range, _, _, _, _, V_total, _ = _calculate_pem_physics_curves()
    
    Area_Total = pem_operator.Area_Total 
    P_bop_fixo = pem_operator.P_bop_fixo 
    k_bop_var = pem_operator.k_bop_var   
    
    I_total = j_range * Area_Total
    P_stack_W = I_total * V_total
    P_bop_var_W = k_bop_var * P_stack_W
    P_total_W = P_stack_W + P_bop_fixo + P_bop_var_W
    
    P_stack_kW = P_stack_W / 1000.0
    P_total_kW = P_total_W / 1000.0
    
    plt.figure(figsize=(10, 6))
    plt.fill_between(j_range, 0, P_stack_kW, color=COLORS['pem'], alpha=0.3, label='Useful Power (Stack)')
    plt.fill_between(j_range, P_stack_kW, P_total_kW, color='gray', alpha=0.5, label='BoP Losses')
    plt.plot(j_range, P_total_kW, color='darkred', linewidth=2, label='Total System Consumption')
    plt.axvline(x=pem_operator.j_nom, color='black', linestyle='--', alpha=0.5)
    
    plt.title('Power Balance: Stack vs Balance of Plant (BoP)', fontsize=14, fontweight='bold')
    plt.xlabel('Current Density (A/cm²)')
    plt.ylabel('Power (kW)')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('report_physics_power_balance.png')
    plt.close()
    print("   -> Saved: report_physics_power_balance.png")

def plot_degradation_projection(pem_state):
    """Plots degradation projection based on current state."""
    time_table = pem_operator.T_OP_H_TABLE / 8760.0 
    voltage_table = pem_operator.V_CELL_TABLE 
    
    current_time_years = pem_state.t_op_h / 8760.0
    V_current = pem_operator.calculate_Vcell(pem_operator.j_nom, pem_operator.T, pem_operator.P_op, pem_state.t_op_h)
    
    plt.figure(figsize=(10, 6))
    plt.plot(time_table, voltage_table, marker='o', label='Degradation Curve (Model)')
    plt.scatter([current_time_years], [V_current], color='red', s=100, zorder=5, label='Current Simulation State')
    
    plt.title('Nominal Voltage Evolution (Degradation)', fontsize=14)
    plt.axhline(y=2.0, color='green', linestyle='--', alpha=0.5, label='BOL')
    plt.axhline(y=2.2, color='red', linestyle='--', alpha=0.5, label='EOL')
    plt.xlabel('Operating Time (Years)')
    plt.ylabel('Cell Voltage (V)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('report_degradation.png')
    plt.close()
    print("   -> Saved: report_degradation.png")

# ==============================================================================
# GROUP C: DETAILED MODULE ANALYSIS
# ==============================================================================

def plot_modules_temporal(module_history, minutes):
    """Generates temporal plots for each module."""
    if module_history is None or len(module_history) == 0:
        print("   [!] Module data not available for temporal plots.")
        return

    data = np.array(module_history)
    time_axis = np.array(minutes)
    num_modules = data.shape[1]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    accumulated_wear = np.cumsum(data, axis=0)
    for j in range(num_modules):
        ax1.plot(time_axis, accumulated_wear[:, j], label=f'Module {j+1}')
    
    mean_wear = np.mean(accumulated_wear, axis=1)
    ax1.plot(time_axis, mean_wear, 'k--', label='Mean', linewidth=2)
    
    ax1.set_title('Accumulated Wear per Module (MW·min)', fontsize=13)
    ax1.set_ylabel('Accumulated Energy (MW·min)')
    ax1.legend(loc='upper left', fontsize='small', ncol=2)
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    for j in range(num_modules):
        ax2.plot(time_axis, data[:, j], label=f'Module {j+1}', alpha=0.8)
        
    ax2.set_title('Instantaneous Power per Module (MW)', fontsize=13)
    ax2.set_ylabel('Power (MW)')
    ax2.set_xlabel('Time (Minutes)')
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('report_modules_temporal.png')
    plt.close()
    print("   -> Saved: report_modules_temporal.png")

def plot_modules_bars(module_history, cycle_counts):
    """Generates bar charts for Total Wear and Cycles."""
    if module_history is None or cycle_counts is None:
        print("   [!] Module/cycle data not available for bar charts.")
        return

    total_wear = np.sum(module_history, axis=0)
    num_modules = len(cycle_counts)
    x_base = np.arange(num_modules)
    labels = [f'Mod {i+1}' for i in range(num_modules)]
    
    wear_mean = np.mean(total_wear)
    cycle_mean = np.mean(cycle_counts)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    bars1 = ax1.bar(x_base, total_wear, color='#FF9800', alpha=0.8, edgecolor='black')
    ax1.axhline(wear_mean, color='black', linestyle='--', linewidth=1.5, label=f'Mean: {wear_mean:.1f}')
    ax1.set_title('Total Accumulated Wear (Processed Energy)', fontsize=12)
    ax1.set_ylabel('MW·min')
    ax1.set_xticks(x_base)
    ax1.set_xticklabels(labels)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    bars2 = ax2.bar(x_base, cycle_counts, color='#2196F3', alpha=0.8, edgecolor='black')
    ax2.axhline(cycle_mean, color='black', linestyle='--', linewidth=1.5, label=f'Mean: {cycle_mean:.1f}')
    ax2.set_title('Startup Cycle Count (Hot Standby -> Ramp Up)', fontsize=12)
    ax2.set_ylabel('Cycles (#)')
    ax2.set_xticks(x_base)
    ax2.set_xticklabels(labels)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.suptitle(f'Final Module Status ({num_modules} Units)', fontsize=14)
    plt.tight_layout()
    plt.savefig('report_modules_stats.png')
    plt.close()
    print("   -> Saved: report_modules_stats.png")

def print_simulation_stats(history):
    """Prints requested statistics to console."""
    prices = np.array(history['spot_price'])
    sell_decisions = np.array(history['sell_decision'])
    
    max_price = np.max(prices)
    avg_price = np.mean(prices)
    count_sell = np.sum(sell_decisions)
    
    print("\n--- [STATS] Additional Statistics ---")
    print(f"* Max Intraday Price: {max_price:.2f} EUR/MWh")
    print(f"* Avg Intraday Price: {avg_price:.2f} EUR/MWh")
    print(f"* Sales Count (Opportunities): {int(count_sell)} times")
    print("---------------------------------------")

# ==============================================================================
# MASTER FUNCTION: REPORT CONTROLLER
# ==============================================================================

def generate_selected_reports(history, pem_state=None, selection=['all'], module_history=None, module_cycles=None):
    """
    Main function called by the Manager.
    """
    print("\n--- [REPORT] Starting Chart Generation ---")
    
    # 1. Print Stats
    print_simulation_stats(history)

    # 2. Generate Charts
    ops_map = {
        'dispatch': lambda: plot_dispatch(history),
        'arbitrage': lambda: plot_arbitrage(history),
        'pie': lambda: plot_energy_pie(history),
        'h2': lambda: plot_h2_production(history),
        'oxygen': lambda: plot_oxygen_production(history),
        'water': lambda: plot_water_consumption(history),
        'histogram': lambda: plot_price_histogram(history),
        'dispatch_curve': lambda: plot_dispatch_curve(history),
        # NOVO MAP PARA MÉDIAS TEMPORAIS
        'temporal_averages': lambda: plot_temporal_averages(history) 
    }
    
    phys_map = {
        'polarization': lambda: plot_physics_polarization(pem_state),
        'efficiency': plot_physics_efficiency,
        'power_balance': plot_physics_power_balance,
        'degradation': lambda: plot_degradation_projection(pem_state) if pem_state else print("   [!] Error: pem_state required."),
    }
    
    modules_map = {
        'modules_temporal': lambda: plot_modules_temporal(module_history, history['minute']),
        'modules_stats': lambda: plot_modules_bars(module_history, module_cycles)
    }
    
    run_all = 'all' in selection
    
    # Execute Operational
    for key, func in ops_map.items():
        if run_all or key in selection:
            try:
                func()
            except Exception as e:
                print(f"   [!] Error generating chart '{key}': {e}")
            
    # Execute Physical
    for key, func in phys_map.items():
        if run_all or key in selection:
            try:
                func()
            except Exception as e:
                print(f"   [!] Error generating physical chart '{key}': {e}")
                
    # Execute Module Analysis
    if module_history is not None:
        for key, func in modules_map.items():
            if run_all or key in selection:
                try:
                    func()
                except Exception as e:
                    print(f"   [!] Error generating module chart '{key}': {e}")
    elif run_all or 'modules_temporal' in selection:
        print("   [i] Skipping module charts: 'module_history' data not provided by manager.")
            
    print("--- [REPORT] Generation Completed ---\n")

if __name__ == "__main__":
    print("This module must be imported by manager.py.")