import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

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

# ==============================================================================
# GRUPO A: GRÁFICOS OPERACIONAIS (Baseados no Histórico do Manager)
# ==============================================================================

def plot_dispatch(history, output_dir='.'):
    """Gera o gráfico de despacho de potência (Oferta vs SOEC vs PEM vs Venda)."""
    minutes = history['minute']
    P_offer = np.array(history['P_offer'])
    P_soec = np.array(history['P_soec'])
    P_pem = np.array(history['P_pem'])
    P_sold = np.array(history['P_sold'])

    plt.figure(figsize=(12, 6))
    
    # Áreas empilhadas (Mantendo estética original)
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
    plt.savefig(os.path.join(output_dir, 'report_dispatch.png'))
    plt.close()
    print("   -> Salvo: report_dispatch.png")

def plot_arbitrage(history, h2_price_eur_kg=9.6, output_dir='.'):
    """Gera gráfico de preços com linha extra para Breakeven do H2."""
    minutes = history['minute']
    spot_price = np.array(history['Spot'])
    sell_decision = np.array(history['sell_decision'])
    
    PPA_PRICE = 50.0 
    
    # Cálculo do Breakeven H2 (Estimativa: 50 kWh/kg de eficiência média do sistema)
    # Preço Eq (EUR/MWh) = (EUR/kg) / (MWh/kg)
    EFF_ESTIMATE_MWH_KG = 0.05 # 50 kWh
    H2_EQUIV_PRICE = h2_price_eur_kg / EFF_ESTIMATE_MWH_KG

    plt.figure(figsize=(12, 6))
    plt.plot(minutes, spot_price, label='Preço Spot (EUR/MWh)', color=COLORS['price'])
    plt.axhline(y=PPA_PRICE, color=COLORS['ppa'], linestyle='--', label='Preço PPA (Contrato)')
    
    # Nova Linha: Preço do Hidrogênio
    plt.axhline(y=H2_EQUIV_PRICE, color='green', linestyle='-.', label=f'H2 Breakeven (~{H2_EQUIV_PRICE:.0f} EUR/MWh)')
    
    # Pontos de Venda
    sell_idx = np.where(sell_decision == 1)[0]
    if len(sell_idx) > 0:
        plt.scatter(np.array(minutes)[sell_idx], np.array(spot_price)[sell_idx], 
                   color='red', zorder=5, label='Decisão: Venda')

    plt.title('Cenário de Preços, PPA e Custo de Oportunidade H2', fontsize=14)
    plt.xlabel('Tempo (Minutos)')
    plt.ylabel('Preço (EUR/MWh)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'report_arbitrage.png'))
    plt.close()
    print("   -> Salvo: report_arbitrage.png")

def plot_h2_production(history, output_dir='.'):
    """Gera gráfico de produção total de Hidrogênio."""
    minutes = history['minute']
    H2_soec = np.array(history['H2_soec'])
    H2_pem = np.array(history['H2_pem'])
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
    plt.savefig(os.path.join(output_dir, 'report_h2_production.png'))
    plt.close()
    print("   -> Salvo: report_h2_production.png")

def plot_oxygen_production(history, output_dir='.'):
    """(NOVO) Gera gráfico de produção conjunta de Oxigênio."""
    minutes = history['minute']
    
    # Se o manager não salvou O2, calcula pela estequiometria (Massa O2 = 8 * Massa H2)
    O2_pem = np.array(history['H2_pem']) * 8.0
    O2_soec = np.array(history['H2_soec']) * 8.0 
    O2_total = O2_soec + O2_pem

    plt.figure(figsize=(12, 6))
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
    print("   -> Salvo: report_oxygen_production.png")

def plot_water_consumption(history, output_dir='.'):
    """Gera gráfico de consumo de água."""
    minutes = history['minute']
    water_soec = np.array(history['Steam_soec']) * 1.10 
    water_pem = np.array(history['H2O_pem']) * 1.02 
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
    plt.savefig(os.path.join(output_dir, 'report_water_consumption.png'))
    plt.close()
    print("   -> Salvo: report_water_consumption.png")

def plot_energy_pie(history, output_dir='.'):
    """Gera gráfico de rosca (Donut Chart) da distribuição de energia."""
    E_soec = np.sum(history['P_soec']) / 60.0
    E_pem = np.sum(history['P_pem']) / 60.0
    E_sold = np.sum(history['P_sold']) / 60.0
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

    if not valid_sizes:
        print("   [!] Aviso: Nenhuma energia consumida para gerar gráfico de pizza.")
        return

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
    print("   -> Salvo: report_energy_pie.png")

def plot_price_histogram(history, output_dir='.'):
    """(NOVO) Gera histograma da distribuição de preços."""
    spot_price = np.array(history['Spot'])
    
    plt.figure(figsize=(10, 6))
    plt.hist(spot_price, bins=30, color='gray', edgecolor='black', alpha=0.7)
    
    mean_price = np.mean(spot_price)
    max_price = np.max(spot_price)
    
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
    print("   -> Salvo: report_price_histogram.png")

def plot_dispatch_curve(history, output_dir='.'):
    """(NOVO) Gera Curva de Despacho: H2 Produzido vs Potência Total."""
    P_total = np.array(history['P_soec']) + np.array(history['P_pem'])
    H2_total = np.array(history['H2_soec']) + np.array(history['H2_pem'])
    
    plt.figure(figsize=(10, 6))
    plt.scatter(P_total, H2_total, alpha=0.5, c='blue', edgecolors='none')
    
    plt.title('Curva de Despacho Real: Produção H2 vs Potência', fontsize=14)
    plt.xlabel('Potência de Entrada Total (MW)')
    plt.ylabel('Produção de H2 (kg/min)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'report_dispatch_curve.png'))
    plt.close()
    print("   -> Salvo: report_dispatch_curve.png")

def generate_reports(history, output_dir='.'):
    """Gera todos os relatórios disponíveis."""
    print(f"\nGenerating reports in {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    plot_dispatch(history, output_dir)
    plot_arbitrage(history, output_dir=output_dir)
    plot_h2_production(history, output_dir)
    plot_oxygen_production(history, output_dir)
    plot_water_consumption(history, output_dir)
    plot_energy_pie(history, output_dir)
    plot_price_histogram(history, output_dir)
    plot_dispatch_curve(history, output_dir)
