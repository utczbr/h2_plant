#!/usr/bin/env python3
"""
8-Hour Validation with Detailed Logging
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from h2_plant.config.plant_builder import PlantBuilder
from h2_plant.simulation.engine import SimulationEngine

print("="*70)
print("8-HOUR VALIDATION WITH DETAILED LOGGING")
print("="*70)

config_path = "configs/plant_pem_soec_8hour_test.yaml"

# Build
builder = PlantBuilder.from_file(config_path)

# Get coordinator for logging access
coordinator = builder.registry.get('dual_path_coordinator')

# Create detailed log
log_data = []

# Monkey-patch coordinator's step method to log every minute
original_step = coordinator.step

def logged_step(t):
    result = original_step(t)
    
    # Log state every minute
    minute = int(t * 60)
    env = builder.registry.get('environment_manager')
    
    log_entry = {
        'minute': minute,
        'P_offer': env.current_wind_power_mw if env else 0,
        'P_soec_set': coordinator.soec_setpoint_mw,
        'P_soec_actual': coordinator.soec_actual_mw,
        'P_pem_set': coordinator.pem_setpoint_mw,
        'P_sold': coordinator.sold_power_mw,
        'spot_price': env.current_energy_price_eur_mwh if env else 0,
        'force_sell': coordinator.force_sell_flag,
        'sell_decision': coordinator.sell_decision,
    }
    log_data.append(log_entry)
    
    return result

coordinator.step = logged_step

# Run simulation
engine = SimulationEngine(
    registry=builder.registry,
    config=builder.config.simulation,
    topology=getattr(builder.config, 'topology', []),
    indexed_topology=getattr(builder.config, 'indexed_topology', [])
)

print("Running simulation with detailed logging...")
results = engine.run()

# Save log
df = pd.DataFrame(log_data)
df.to_csv('h2plant_minute_log.csv', index=False)
print(f"\n✓ Saved detailed log: h2plant_minute_log.csv ({len(df)} rows)")

# Print summary
print("\n" + "="*70)
print("H2 PLANT DETAILED RESULTS")
print("="*70)

state = coordinator.get_state()
print(f"Total H2 produced: {state.get('cumulative_production_kg', 0):.2f} kg")
print(f"Energy sold: {state.get('cumulative_sold_energy_mwh', 0):.4f} MWh")
print(f"Total SOEC power: {df['P_soec_actual'].sum() /60:.2f} MWh")
print(f"Total PEM power: {df['P_pem_set'].sum() / 60:.2f} MWh")
print(f"Total Sold: {df['P_sold'].sum() / 60:.2f} MWh")
print("="*70)

print("\nFirst 10 minutes:")
print(df.head(10).to_string())

print("\nMinutes with high sold power (>1 MW):")
high_sold = df[df['P_sold'] > 1.0]
if not high_sold.empty:
    print(high_sold.to_string())
else:
    print("None")
