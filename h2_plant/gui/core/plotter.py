import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# ==============================================================================
# CONFIGURAÇÕES GERAIS E MAPA DE CORES
# ==============================================================================
plt.style.use('default') 
COLORS = {
    'offer': 'black',
    'soec': '#4CAF50',    # Verde
    'pem': '#2196F3',     # Azul
    'sold': '#FF9800',    # Laranja
    'price': 'black',
    'ppa': 'red',
    'limit': 'blue',
    'h2_total': 'black',
    'water_total': 'navy',
    'oxygen': 'purple'    # Nova cor para O2
}

def normalize_history(history):
    """
    Normalizes the history dictionary to ensure consistent keys for plotting.
    Maps backend keys to those expected by the plotting logic.
    """
    df = pd.DataFrame(history)
    
    # Map keys if they exist, otherwise create defaults or use aliases
    # Power
    if 'P_soec_actual' in df.columns and 'P_soec' not in df.columns:
        df['P_soec'] = df['P_soec_actual']
    if 'P_soec' not in df.columns: df['P_soec'] = 0.0
        
    if 'P_pem' not in df.columns: df['P_pem'] = 0.0
    if 'P_sold' not in df.columns: df['P_sold'] = 0.0
    if 'P_offer' not in df.columns: df['P_offer'] = 0.0
    
    # Prices
    if 'spot_price' in df.columns and 'Spot' not in df.columns:
        df['Spot'] = df['spot_price']
    if 'Spot' not in df.columns: df['Spot'] = 0.0
    
    # Hydrogen
    if 'H2_soec_kg' in df.columns and 'H2_soec' not in df.columns:
        df['H2_soec'] = df['H2_soec_kg']
    if 'H2_soec' not in df.columns: df['H2_soec'] = 0.0
        
    if 'H2_pem_kg' in df.columns and 'H2_pem' not in df.columns:
        df['H2_pem'] = df['H2_pem_kg']
    if 'H2_pem' not in df.columns: df['H2_pem'] = 0.0
    
    # Water
    if 'steam_soec_kg' in df.columns and 'Steam_soec' not in df.columns:
        df['Steam_soec'] = df['steam_soec_kg']
    if 'Steam_soec' not in df.columns: df['Steam_soec'] = 0.0
        
    if 'H2O_pem_kg' in df.columns and 'H2O_pem' not in df.columns:
        df['H2O_pem'] = df['H2O_pem_kg']
    if 'H2O_pem' not in df.columns: df['H2O_pem'] = 0.0
    
    # Time
    if 'minute' not in df.columns:
        df['minute'] = df.index * 60 # Assume hourly steps if minute not present
        
    return df

def generate_plots(history, output_dir="."):
    """
    Generates PNG plots from simulation history dictionary.
    Matches the filenames expected by SimulationReportWidget.
    """
    # Normalize data
    df = normalize_history(history)
    minutes = df['minute']
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # ==============================================================================
    # 1. Dispatch Chart (report_dispatch.png)
    # ==============================================================================
    try:
        plt.figure(figsize=(12, 6))
        
        P_soec = df['P_soec']
        P_pem = df['P_pem']
        P_sold = df['P_sold']
        P_offer = df['P_offer']
        
        # Stacked areas
        plt.fill_between(minutes, 0, P_soec, label='Consumo SOEC', color=COLORS['soec'], alpha=0.6)
        plt.fill_between(minutes, P_soec, P_soec + P_pem, label='Consumo PEM', color=COLORS['pem'], alpha=0.6)
        plt.fill_between(minutes, P_soec + P_pem, P_soec + P_pem + P_sold, label='Venda ao Grid', color=COLORS['sold'], alpha=0.6)
        
        # Reference lines
        plt.plot(minutes, P_offer, label='Potência Ofertada', color=COLORS['offer'], linestyle='--', linewidth=1.5)
        
        plt.title('Despacho Híbrido: SOEC + PEM + Arbitragem', fontsize=14)
        plt.xlabel('Tempo (Minutos)')
        plt.ylabel('Potência (MW)')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'report_dispatch.png'))
        plt.close()
    except Exception as e:
        print(f"Error generating dispatch plot: {e}")

    # ==============================================================================
    # 2. Arbitrage Chart (report_arbitrage.png)
    # ==============================================================================
    try:
        plt.figure(figsize=(12, 6))
        spot_price = df['Spot']
        
        # PPA and Breakeven lines
        PPA_PRICE = 50.0 
        h2_price_eur_kg = 9.6
        EFF_ESTIMATE_MWH_KG = 0.05 # 50 kWh/kg
        H2_EQUIV_PRICE = h2_price_eur_kg / EFF_ESTIMATE_MWH_KG
        
        plt.plot(minutes, spot_price, label='Preço Spot (EUR/MWh)', color=COLORS['price'])
        plt.axhline(y=PPA_PRICE, color=COLORS['ppa'], linestyle='--', label='Preço PPA (Contrato)')
        plt.axhline(y=H2_EQUIV_PRICE, color='green', linestyle='-.', label=f'H2 Breakeven (~{H2_EQUIV_PRICE:.0f} EUR/MWh)')
        
        # Sell points
        if 'sell_decision' in df.columns:
            sell_decision = df['sell_decision']
            sell_idx = df.index[sell_decision == 1].tolist()
            if sell_idx:
                plt.scatter(minutes.iloc[sell_idx], spot_price.iloc[sell_idx], 
                           color='red', zorder=5, label='Decisão: Venda')

        plt.title('Cenário de Preços, PPA e Custo de Oportunidade H2', fontsize=14)
        plt.xlabel('Tempo (Minutos)')
        plt.ylabel('Preço (EUR/MWh)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'report_arbitrage.png'))
        plt.close()
    except Exception as e:
        print(f"Error generating arbitrage plot: {e}")

    # ==============================================================================
    # 3. H2 Production (report_h2_production.png)
    # ==============================================================================
    try:
        plt.figure(figsize=(12, 6))
        H2_soec = df['H2_soec']
        H2_pem = df['H2_pem']
        H2_total = H2_soec + H2_pem

        plt.fill_between(minutes, 0, H2_soec, label='H2 SOEC', color=COLORS['soec'], alpha=0.5)
        plt.fill_between(minutes, H2_soec, H2_total, label='H2 PEM', color=COLORS['pem'], alpha=0.5)
        plt.plot(minutes, H2_total, color=COLORS['h2_total'], linestyle='--', label='Total H2')

        plt.title('Produção de Hidrogênio Acumulada (kg/min)', fontsize=14)
        plt.ylabel('Taxa de Produção (kg/min)')
        plt.xlabel('Tempo (Minutos)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'report_h2_production.png'))
        plt.close()
    except Exception as e:
        print(f"Error generating H2 production plot: {e}")

    # ==============================================================================
    # 4. Oxygen Production (report_oxygen_production.png)
    # ==============================================================================
    try:
        plt.figure(figsize=(12, 6))
        
        # Calculate O2 if not present (Stoichiometry: Mass O2 = 8 * Mass H2)
        if 'O2_pem_kg' in df.columns:
            O2_pem = df['O2_pem_kg']
        else:
            O2_pem = df['H2_pem'] * 8.0
            
        O2_soec = df['H2_soec'] * 8.0 
        O2_total = O2_soec + O2_pem

        plt.fill_between(minutes, 0, O2_soec, label='O2 SOEC', color=COLORS['soec'], alpha=0.5)
        plt.fill_between(minutes, O2_soec, O2_total, label='O2 PEM', color='purple', alpha=0.5)
        plt.plot(minutes, O2_total, color='black', linestyle='--', label='Total O2')

        plt.title('Produção de Oxigênio Conjunta (kg/min)', fontsize=14)
        plt.ylabel('Taxa de Produção (kg/min)')
        plt.xlabel('Tempo (Minutos)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'report_oxygen_production.png'))
        plt.close()
    except Exception as e:
        print(f"Error generating oxygen plot: {e}")

    # ==============================================================================
    # 5. Water Consumption (report_water_consumption.png)
    # ==============================================================================
    try:
        plt.figure(figsize=(12, 6))
        
        water_soec = df['Steam_soec'] * 1.10 
        water_pem = df['H2O_pem'] * 1.02 
        total = water_soec + water_pem

        plt.fill_between(minutes, 0, water_soec, label='H2O SOEC (Total)', color=COLORS['soec'], alpha=0.5)
        plt.fill_between(minutes, water_soec, total, label='H2O PEM (Total)', color='brown', alpha=0.5)
        plt.plot(minutes, total, color=COLORS['water_total'], linestyle='--', label='Consumo Total H2O')

        plt.title('Consumo de Água Total (Incluindo Perdas/Vapor)', fontsize=14)
        plt.ylabel('Fluxo de Água (kg/min)')
        plt.xlabel('Tempo (Minutos)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'report_water_consumption.png'))
        plt.close()
    except Exception as e:
        print(f"Error generating water consumption plot: {e}")

    # ==============================================================================
    # 6. Energy Pie (report_energy_pie.png)
    # ==============================================================================
    try:
        E_soec = df['P_soec'].sum() / 60.0
        E_pem = df['P_pem'].sum() / 60.0
        E_sold = df['P_sold'].sum() / 60.0
        E_total = E_soec + E_pem + E_sold
        
        sizes = [E_soec, E_pem, E_sold]
        labels = ['SOEC', 'PEM', 'Venda ao Grid']
        colors = [COLORS['soec'], COLORS['pem'], COLORS['sold']]
        
        valid_sizes = []
        valid_labels = []
        valid_colors = []
        for s, l, c in zip(sizes, labels, colors):
            if s > 0.01:
                valid_sizes.append(s)
                valid_labels.append(l)
                valid_colors.append(c)

        if valid_sizes:
            fig, ax = plt.subplots(figsize=(9, 8))
            explode = [0.05 if 'Venda' in l else 0 for l in valid_labels]
            
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
            ax.legend(wedges, legend_labels, title="Distribuição Energética", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
            
            plt.title('Distribuição de Energia Consumida/Vendida', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'report_energy_pie.png'))
            plt.close()
    except Exception as e:
        print(f"Error generating energy pie chart: {e}")

    # ==============================================================================
    # 7. Price Histogram (report_price_histogram.png)
    # ==============================================================================
    try:
        plt.figure(figsize=(10, 6))
        spot_price = df['Spot']
        plt.hist(spot_price, bins=30, color='gray', edgecolor='black', alpha=0.7)
        
        mean_price = spot_price.mean()
        max_price = spot_price.max()
        
        plt.axvline(mean_price, color='blue', linestyle='--', linewidth=2, label=f'Média: {mean_price:.2f} EUR')
        plt.axvline(max_price, color='red', linestyle='--', linewidth=2, label=f'Máximo: {max_price:.2f} EUR')
        
        plt.title('Distribuição de Frequência dos Preços Spot', fontsize=14)
        plt.xlabel('Preço (EUR/MWh)')
        plt.ylabel('Frequência (Minutos)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'report_price_histogram.png'))
        plt.close()
    except Exception as e:
        print(f"Error generating price histogram: {e}")

    # ==============================================================================
    # 8. Dispatch Curve (report_dispatch_curve.png)
    # ==============================================================================
    try:
        P_total = df['P_soec'] + df['P_pem']
        H2_total = df['H2_soec'] + df['H2_pem']
        
        plt.figure(figsize=(10, 6))
        plt.scatter(P_total, H2_total, alpha=0.5, c='blue', edgecolors='none')
        
        plt.title('Curva de Despacho Real: Produção H2 vs Potência', fontsize=14)
        plt.xlabel('Potência de Entrada Total (MW)')
        plt.ylabel('Produção de H2 (kg/min)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'report_dispatch_curve.png'))
        plt.close()
    except Exception as e:
        print(f"Error generating dispatch curve: {e}")
