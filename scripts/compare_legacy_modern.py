#!/usr/bin/env python3
"""
Simple Comparison: Legacy manager.py vs Modern System

Generates side-by-side comparison graphs for precision validation.
"""

import sys
from pathlib import Path

# Add vendor libs
sys.path.insert(0, str(Path(__file__).parent.parent / 'vendor' / 'libs'))

import matplotlib.pyplot as plt
import numpy as np

# Add legacy path
legacy_path = Path(__file__).parent.parent / 'h2_plant' / 'legacy' / 'pem_soec_reference'
sys.path.insert(0, str(legacy_path))

print("=" * 80)
print("RUNNING LEGACY MANAGER.PY")
print("=" * 80)

# Import and run legacy
import manager
legacy_history = manager.run_hybrid_management()

print(f"\n✅ Legacy complete: {len(legacy_history['minute'])} minutes")
print(f"   Total H2: {sum(np.array(legacy_history['H2_soec_kg']) + np.array(legacy_history['H2_pem_kg'])):.2f} kg")
print(f"   Energy Sold: {sum(legacy_history['P_sold'])/60:.4f} MWh")

# Now run modern system and collect its history
print("\n" + "=" * 80)
print("RUNNING MODERN SYSTEM")
print("=" * 80)

sys.path.insert(0, str(Path(__file__).parent.parent))
from h2_plant.config.plant_builder import PlantBuilder
from h2_plant.simulation.engine import SimulationEngine

config_path = "configs/plant_pem_soec_8hour_test.yaml"
builder = PlantBuilder.from_file(config_path)
engine = SimulationEngine(
    registry=builder.registry,
    config=builder.config.simulation,
    topology=getattr(builder.config, 'topology', []),
    indexed_topology=getattr(builder.config, 'indexed_topology', [])
)

coordinator = builder.registry.get('dual_path_coordinator')
soec = builder.registry.get('soec_cluster')
env = builder.registry.get('environment_manager')

engine.initialize()

# Initialize modern history
modern_history = {
    'minute': [], 'hour': [], 'P_offer': [], 'P_soec_set': [],
    'P_soec_actual': [], 'P_pem': [], 'P_sold': [],
    'spot_price': [], 'sell_decision': [], 'H2_soec_kg': [],
    'steam_soec_kg': [], 'H2_pem_kg': []
}

# Run modern simulation
for minute in range(480):
    hour_fraction = minute / 60.0
    engine._execute_timestep(hour_fraction)
    
    coord_state = coordinator.get_state()
    soec_state = soec.get_state()
    env_state = env.get_state()
    
    modern_history['minute'].append(minute)
    modern_history['hour'].append(minute // 60 + 1)
    modern_history['P_offer'].append(env_state.get('current_wind_power_mw', 0))
    modern_history['P_soec_set'].append(coord_state.get('soec_setpoint_mw', 0))
    modern_history['P_soec_actual'].append(soec_state.get('P_total_mw', 0))
    modern_history['P_pem'].append(coord_state.get('pem_setpoint_mw', 0))
    modern_history['P_sold'].append(coord_state.get('sold_power_mw', 0))
    modern_history['spot_price'].append(env_state.get('current_energy_price_eur_mwh', 0))
    modern_history['sell_decision'].append(coord_state.get('sell_decision', 0))
    modern_history['H2_soec_kg'].append(soec_state.get('h2_output_kg', 0))
    modern_history['steam_soec_kg'].append(soec_state.get('steam_consumed_kg', 0))
    modern_history['H2_pem_kg'].append(0.0)  # PEM not fully operational yet

print(f"\n✅ Modern complete: {len(modern_history['minute'])} minutes")
modern_total_h2 = sum(np.array(modern_history['H2_soec_kg']) + np.array(modern_history['H2_pem_kg']))
print(f"   Total H2: {modern_total_h2:.2f} kg")
print(f"   Energy Sold: {sum(modern_history['P_sold'])/60:.4f} MWh")

# Generate comparison graphs
print("\n" + "=" * 80)
print("GENERATING COMPARISON GRAPHS")
print("=" * 80)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Legacy vs Modern Implementation Comparison (8-Hour Test)', 
             fontsize=16, fontweight='bold')

minutes_l = np.array(legacy_history['minute'])
minutes_m = np.array(modern_history['minute'])

# 1. SOEC Power
ax = axes[0, 0]
ax.plot(minutes_l, legacy_history['P_soec_actual'], 'b-', label='Legacy', linewidth=2, alpha=0.7)
ax.plot(minutes_m, modern_history['P_soec_actual'], 'r--', label='Modern', linewidth=1.5)
ax.set_title('SOEC Power Dispatch')
ax.set_xlabel('Time (minutes)')
ax.set_ylabel('Power (MW)')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. PEM Power
ax = axes[0, 1]
ax.plot(minutes_l, legacy_history['P_pem'], 'b-', label='Legacy', linewidth=2, alpha=0.7)
ax.plot(minutes_m, modern_history['P_pem'], 'r--', label='Modern', linewidth=1.5)
ax.set_title('PEM Power Dispatch')
ax.set_xlabel('Time (minutes)')
ax.set_ylabel('Power (MW)')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Sold Power
ax = axes[0, 2]
ax.plot(minutes_l, legacy_history['P_sold'], 'b-', label='Legacy', linewidth=2, alpha=0.7)
ax.plot(minutes_m, modern_history['P_sold'], 'r--', label='Modern', linewidth=1.5)
ax.set_title('Sold Power (Arbitrage)')
ax.set_xlabel('Time (minutes)')
ax.set_ylabel('Power (MW)')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. Cumulative H2
ax = axes[1, 0]
legacy_h2 = np.array(legacy_history['H2_soec_kg']) + np.array(legacy_history['H2_pem_kg'])
modern_h2 = np.array(modern_history['H2_soec_kg']) + np.array(modern_history['H2_pem_kg'])
ax.plot(minutes_l, np.cumsum(legacy_h2), 'b-', label='Legacy', linewidth=2, alpha=0.7)
ax.plot(minutes_m, np.cumsum(modern_h2), 'r--', label='Modern', linewidth=1.5)
ax.set_title('Cumulative H2 Production')
ax.set_xlabel('Time (minutes)')
ax.set_ylabel('H2 (kg)')
ax.legend()
ax.grid(True, alpha=0.3)

# 5. Sell Decisions
ax = axes[1, 1]
ax.scatter(minutes_l, legacy_history['sell_decision'], label='Legacy', 
           s=10, c='blue', alpha=0.6)
ax.scatter(minutes_m, modern_history['sell_decision'], label='Modern',
           s=10, c='red', marker='x', alpha=0.6)
ax.set_title('Sell Decisions (0=H2, 1=Sell)')
ax.set_xlabel('Time (minutes)')
ax.set_ylabel('Decision')
ax.set_yticks([0, 1])
ax.legend()
ax.grid(True, alpha=0.3)

# 6. Power Deviations
ax = axes[1, 2]
soec_dev = 1000 * np.abs(np.array(legacy_history['P_soec_actual']) - np.array(modern_history['P_soec_actual']))
pem_dev = 1000 * np.abs(np.array(legacy_history['P_pem']) - np.array(modern_history['P_pem']))
sold_dev = 1000 * np.abs(np.array(legacy_history['P_sold']) - np.array(modern_history['P_sold']))

ax.plot(minutes_l, soec_dev, label='SOEC', color='blue', alpha=0.7)
ax.plot(minutes_l, pem_dev, label='PEM', color='green', alpha=0.7)
ax.plot(minutes_l, sold_dev, label='Sold', color='red', alpha=0.7)
ax.axhline(y=1, color='black', linestyle='--', linewidth=1, label='Tolerance (±1 kW)')
ax.set_title('Power Deviations')
ax.set_xlabel('Time (minutes)')
ax.set_ylabel('|Modern - Legacy| (kW)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

plt.tight_layout()
plt.savefig('legacy_vs_modern_comparison.png', dpi=150, bbox_inches='tight')
print("\n📊 Saved: legacy_vs_modern_comparison.png")

# Print precision metrics
print("\n" + "=" * 80)
print("PRECISION METRICS")
print("=" * 80)

print("\nPower Deviations (Tolerance: ±1 kW):")
print(f"  SOEC - Max: {np.max(soec_dev):.3f} kW, Mean: {np.mean(soec_dev):.3f} kW")
print(f"  PEM  - Max: {np.max(pem_dev):.3f} kW, Mean: {np.mean(pem_dev):.3f} kW")
print(f"  Sold - Max: {np.max(sold_dev):.3f} kW, Mean: {np.mean(sold_dev):.3f} kW" )

h2_dev_grams = 1000 * np.abs(legacy_h2 - modern_h2)
print(f"\nH2 Production (Tolerance: ±0.1 gram):")
print(f"  Max: {np.max(h2_dev_grams):.1f} grams, Mean: {np.mean(h2_dev_grams):.1f} grams")

sell_matches = np.sum(np.array(legacy_history['sell_decision']) == np.array(modern_history['sell_decision']))
print(f"\nSell Decision Match: {sell_matches}/480 ({100*sell_matches/480:.1f}%)")

legacy_total = np.sum(legacy_h2)
modern_total = np.sum(modern_h2)
h2_error = 100 * abs(legacy_total - modern_total) / legacy_total
print(f"\nCumulative H2:")
print(f"  Legacy: {legacy_total:.2f} kg")
print(f"  Modern: {modern_total:.2f} kg")
print(f"  Error: {h2_error:.4f}%")

print("\n" + "=" * 80)
print("✅ Comparison complete!")
print("=" * 80)
