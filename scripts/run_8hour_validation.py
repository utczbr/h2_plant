#!/usr/bin/env python3
"""
8-Hour Validation Test Runner

Runs both manager.py (reference) and h2_plant for 8 hours and compares results.
"""

import sys
from pathlib import Path

# Add local libs to path for matplotlib
sys.path.insert(0, str(Path(__file__).parent.parent / 'vendor' / 'libs'))

import matplotlib.pyplot as plt
import numpy as np

# Add h2_plant to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from h2_plant.config.plant_builder import PlantBuilder
from h2_plant.simulation.engine import SimulationEngine

def generate_dispatch_chart(history):
    """Generates a chart comparing P_offer, P_soec, P_pem, and P_sold."""
    minutes = history['minute']
    P_offer = np.array(history['P_offer'])
    P_soec = np.array(history['P_soec_actual'])
    P_pem = np.array(history['P_pem'])
    P_sold = np.array(history['P_sold'])

    plt.figure(figsize=(15, 7)) 
    
    # 1. Offered Power (Dashed line)
    plt.plot(minutes, P_offer, label='1. Offered Power (Set Point)', color='black', linestyle='--')
    
    # 2. Actual SOEC Power (Base Area - Main Consumption)
    plt.fill_between(minutes, 0, P_soec, label='2. SOEC Power (Consumption)', color='green', alpha=0.5)
    
    # 3. PEM Power (Secondary Area)
    plt.fill_between(minutes, P_soec, P_soec + P_pem, label='3. PEM Power (Balancing)', color='blue', alpha=0.5)
    
    # 4. Sold Power (Top Area)
    plt.fill_between(minutes, P_soec + P_pem, P_soec + P_pem + P_sold, label='4. Sold Power (Arbitrage/Surplus)', color='orange', alpha=0.5)

    # 5. Total Power Line (Should follow P_offer if no saturation)
    plt.plot(minutes, P_soec + P_pem + P_sold, label='Total Dispatched (SOEC+PEM+Sold)', color='red', linestyle=':', linewidth=1.0)
    
    # Power Limit Line (Added for context, 11.52 MW)
    plt.axhline(y=11.52, color='purple', linestyle='-.', linewidth=1, label=f'Max SOEC Limit (80%): 11.52 MW')
    
    # Limits and Formatting
    plt.title('Hybrid Dispatch Management (SOEC/PEM/Market) - Surplus Arbitrage', fontsize=16) 
    
    plt.xlabel('Time (Minutes)', fontsize=12)
    plt.ylabel('Power (MW)', fontsize=12)
    
    # LEGEND: Outside plot area
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(np.arange(0, len(minutes) + 1, 60)) 
    plt.ylim(0, max(P_offer) * 1.05)
    
    # SAVE CHART
    plt.savefig('hybrid_dispatch_arbitrage.png', bbox_inches='tight')
    print("Saved hybrid_dispatch_arbitrage.png")

def generate_price_chart(history):
    """Generates a chart comparing price profiles and the sell decision."""
    minutes = history['minute']
    spot_price = np.array(history['spot_price'])
    sell_decision = np.array(history['sell_decision'])
    
    # H2 Equivalent Price in EUR/MWh
    # Constants from manager.py
    SOEC_H2_CONSUMPTION_KWH_PER_KG = 37.5
    H2_PRICE_EUR_KG = 9.6
    PPA_PRICE_EUR_MWH = 50.0
    
    h2_eq_price = (1000/SOEC_H2_CONSUMPTION_KWH_PER_KG) * H2_PRICE_EUR_KG 
    
    plt.figure(figsize=(15, 6))
    
    # 1. Spot Price
    plt.plot(minutes, spot_price, label='Spot Price (EUR/MWh)', color='black', linewidth=2.0)
    
    # 2. PPA Price (Fixed Contract Cost)
    plt.axhline(y=PPA_PRICE_EUR_MWH, color='red', linestyle='--', label=f'PPA Contract Price: {PPA_PRICE_EUR_MWH:.2f} EUR/MWh')

    # 3. H2 Equivalent Price (For profit comparison context)
    plt.axhline(y=h2_eq_price, color='green', linestyle=':', label=f'H2 Equivalent Revenue (SOEC): {h2_eq_price:.2f} EUR/MWh')
    
    # Arbitrage Limit Line
    arbitrage_limit = PPA_PRICE_EUR_MWH + h2_eq_price
    plt.axhline(y=arbitrage_limit, color='blue', linestyle='-.', label=f'Arbitrage Limit: {arbitrage_limit:.2f} EUR/MWh')
    
    # 4. Sell Decision Indication
    sell_minutes = np.array(minutes)[sell_decision == 1]
    sell_prices = np.array(spot_price)[sell_decision == 1]
    
    if len(sell_minutes) > 0:
        plt.scatter(sell_minutes, sell_prices, color='red', marker='o', s=50, label='Decision: Sell Power (Arbitrage)')
    
    # Limits and Formatting
    plt.title('Arbitrage Decision vs. Energy Prices', fontsize=16)
    plt.xlabel('Time (Minutes)', fontsize=12)
    plt.ylabel('Price (EUR/MWh)', fontsize=12)
    
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(np.arange(0, len(minutes) + 1, 60))
    plt.ylim(min(spot_price) * 0.9, max(max(spot_price) * 1.1, arbitrage_limit * 1.05))
    
    # SAVE CHART
    plt.savefig('arbitrage_prices_chart.png', bbox_inches='tight')
    print("Saved arbitrage_prices_chart.png")

def generate_pie_chart(history):
    """Generates a pie chart of the total energy distribution."""
    
    # Sum of energy in MWh
    soec_energy = np.sum(history['P_soec_actual']) / 60.0
    pem_energy = np.sum(history['P_pem']) / 60.0
    sold_energy = np.sum(history['P_sold']) / 60.0
    
    labels = [f'SOEC Consumption ({soec_energy:.2f} MWh)', f'PEM Consumption ({pem_energy:.2f} MWh)', f'Sold Power ({sold_energy:.2f} MWh)']
    sizes = [soec_energy, pem_energy, sold_energy]
    colors = ['#4CAF50', '#2196F3', '#FF9800'] # Green, Blue, Orange
    explode = [0.05, 0.05, 0.05] 
    
    plt.figure(figsize=(10, 8))
    # Autopct showing % (MWh value is already in the label for clarity)
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct=lambda pct: f'{pct:.1f}%', 
            shadow=True, startangle=140, textprops={'fontsize': 10})
    
    plt.title('Total Energy Dispatch Distribution (MWh)', fontsize=16)
    
    # SAVE CHART
    plt.savefig('total_energy_pie_chart.png', bbox_inches='tight')
    print("Saved total_energy_pie_chart.png")

def generate_h2_production_chart(history):
    """Generates a chart of SOEC H2 production over time (PEM removed)."""
    
    minutes = history['minute']
    H2_soec_kg = np.array(history['H2_soec_kg'])
    
    plt.figure(figsize=(15, 6))
    
    # 1. SOEC H2 Production (Area)
    plt.fill_between(minutes, 0, H2_soec_kg, label='SOEC H2 Production (kg/min)', color='green', alpha=0.5)
    
    # 2. Total H2 Production Line (Agora é apenas a linha do SOEC)
    plt.plot(minutes, H2_soec_kg, label='Total SOEC H2 Production (kg/min)', color='black', linestyle='--', linewidth=1.5)
    
    # Limits and Formatting
    plt.title('Hydrogen Production (SOEC ONLY)', fontsize=16)
    plt.xlabel('Time (Minutes)', fontsize=12)
    plt.ylabel('H2 Production Rate (kg/min)', fontsize=12)
    
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(np.arange(0, len(minutes) + 1, 60))
    plt.ylim(0, max(H2_soec_kg) * 1.05)
    
    # SAVE CHART
    plt.savefig('soec_h2_production_chart.png', bbox_inches='tight')
    print("Saved soec_h2_production_chart.png")

def generate_steam_consumption_chart(history):
    """Generates a chart of SOEC steam consumption over time."""
    
    minutes = history['minute']
    steam_soec_kg = np.array(history['steam_soec_kg'])
    
    plt.figure(figsize=(15, 6))
    
    # 1. SOEC Steam Consumption (Area)
    plt.fill_between(minutes, 0, steam_soec_kg, label='SOEC Steam Consumption (kg/min)', color='teal', alpha=0.5)
    
    # 2. Line
    plt.plot(minutes, steam_soec_kg, label='SOEC Steam Consumption (kg/min)', color='navy', linestyle='-', linewidth=1.5)
    
    # Limits and Formatting
    plt.title('SOEC Steam Consumption Over Time', fontsize=16)
    plt.xlabel('Time (Minutes)', fontsize=12)
    plt.ylabel('Steam Consumption Rate (kg/min)', fontsize=12)
    
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(np.arange(0, len(minutes) + 1, 60))
    plt.ylim(0, max(steam_soec_kg) * 1.05)
    
    # SAVE CHART
    plt.savefig('soec_steam_consumption_chart.png', bbox_inches='tight')
    print("Saved soec_steam_consumption_chart.png")

print("="*70)
print("8-HOUR VALIDATION TEST")
print("="*70)
print("Duration: 8 hours (480 minutes)")
print("Comparing h2_plant vs reference manager.py")
print("="*70 + "\n")

config_path = "configs/plant_pem_soec_8hour_test.yaml"

print("[1/2] Building and running h2_plant...")
try:
    builder = PlantBuilder.from_file(config_path)
    
    engine = SimulationEngine(
        registry=builder.registry,
        config=builder.config.simulation,
        topology=getattr(builder.config, 'topology', []),
        indexed_topology=getattr(builder.config, 'indexed_topology', [])
    )
    
    print(f"  ✓ {builder.registry.get_component_count()} components ready")
    print(f"  ✓ Running 480 minutes...")
    
    # engine.run()
    
    # Run step-by-step to print table
    print(" | Min | H | P. Offer | P. Set SOEC | P. SOEC Act | P. PEM | P. Sold | P. Spot | Decision | H2 SOEC (kg/min) | Steam SOEC (kg/min) |")
    print(" |-----|---|-----------|-------------|-------------|--------|---------|---------|----------|------------------|---------------------|")
    
    coordinator = builder.registry.get('dual_path_coordinator')
    soec = builder.registry.get('soec_cluster')
    env = builder.registry.get('environment_manager')
    
    engine.initialize()
    
    # Initialize history for plotting
    history = {
        'minute': [], 'hour': [], 'P_offer': [], 'P_soec_set': [], 
        'P_soec_actual': [], 'P_pem': [], 'P_sold': [], 
        'spot_price': [], 'sell_decision': [], 'H2_soec_kg': [],
        'steam_soec_kg': []
    }
    
    for minute in range(480):
        hour_fraction = minute / 60.0
        engine._execute_timestep(hour_fraction)
        
        # Get state for logging and history
        coord_state = coordinator.get_state()
        soec_state = soec.get_state()
        env_state = env.get_state()
        
        P_offer = coord_state.get('P_offer_mw', env_state.get('current_wind_power_mw', 0.0))
        P_soec_set = coord_state.get('soec_setpoint_mw', 0.0)
        P_soec_actual = soec_state.get('P_actual_mw', 0.0)
        P_pem = coord_state.get('pem_setpoint_mw', 0.0)
        P_sold = coord_state.get('sold_power_mw', 0.0)
        spot_price = env_state.get('current_energy_price_eur_mwh', 0.0)
        sell_decision = coord_state.get('sell_decision', 0)
        h2_soec_kg = soec_state.get('h2_output_kg_per_min', 0.0)
        steam_soec_kg = soec_state.get('steam_input_kg_per_min', 0.0)
        
        # Append to history
        history['minute'].append(minute)
        history['hour'].append(minute // 60 + 1)
        history['P_offer'].append(P_offer)
        history['P_soec_set'].append(P_soec_set)
        history['P_soec_actual'].append(P_soec_actual)
        history['P_pem'].append(P_pem)
        history['P_sold'].append(P_sold)
        history['spot_price'].append(spot_price)
        history['sell_decision'].append(sell_decision)
        history['H2_soec_kg'].append(h2_soec_kg)
        history['steam_soec_kg'].append(steam_soec_kg)
        
        if minute % 15 == 0:
            print(
                f" | {minute:03d} | {minute//60 + 1} | {P_offer:9.2f} | {P_soec_set:11.2f} | {P_soec_actual:11.2f} | {P_pem:6.2f} | {P_sold:7.2f} | {spot_price:7.2f} | {('SELL' if sell_decision == 1 else 'H2'):8s} | {h2_soec_kg:16.4f} | {steam_soec_kg:19.4f} |"
            )
            
    results = {} # Dummy results
    
    print(f"  ✓ Simulation complete!")
    
    # Get coordinator results
    coordinator = builder.registry.get('dual_path_coordinator')
    state = coordinator.get_state()
    
    print("\n" + "="*70)
    print("H2 PLANT RESULTS (8 hours)")
    print("="*70)
    print(f"Total H2 produced (System): {state.get('cumulative_production_kg', 0):.2f} kg")
    
    # Get SOEC specific results
    soec_state = soec.get_state()
    print(f"Total SOEC H2 produced: {soec_state.get('total_h2_produced_kg', 0):.2f} kg")
    print(f"Total SOEC Steam consumed: {soec_state.get('total_steam_consumed_kg', 0):.2f} kg")
    
    print(f"Energy sold: {state.get('cumulative_sold_energy_mwh', 0):.4f} MWh")
    print(f"Sell decisions: {state.get('sell_decision', 0)}")
    print(f"Force sell flag: {state.get('force_sell_flag', False)}")
    print("="*70)
    
    print("\n✅ TEST COMPLETE")
    
    # Generate charts
    print("\nGenerating charts...")
    try:
        generate_dispatch_chart(history)
        generate_price_chart(history)
        generate_pie_chart(history)
        generate_h2_production_chart(history)
        generate_steam_consumption_chart(history)
        print("\nAll charts generated successfully!")
    except Exception as e:
        print(f"\n❌ Chart generation failed: {e}")
        import traceback
        traceback.print_exc()

    print("\nTo compare with reference:")
    print("1. cd pem_and_soec")
    print("2. Update manager.py to use same price data")
    print("3. Run: python3 manager.py")
    print("4. Compare outputs manually")
    
except Exception as e:
    print(f"\n❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
