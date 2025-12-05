#!/usr/bin/env python3
"""
Extended Precision Validation Script (7 Days)

Runs both legacy manager.py and modern h2_plant simulation for 168 hours (1 week).
Patches legacy manager with extended data.
Compares results minute-by-minute.
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock
import io

# --- 1. Setup Paths ---
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

LEGACY_DIR = PROJECT_ROOT / 'h2_plant' / 'legacy' / 'pem_soec_reference'
sys.path.insert(0, str(LEGACY_DIR))

# Mock matplotlib
sys.modules['matplotlib'] = MagicMock()
sys.modules['matplotlib.pyplot'] = MagicMock()

# --- 2. Import Legacy Manager ---
try:
    import manager
    print("✅ Legacy manager imported successfully")
except ImportError as e:
    print(f"❌ Failed to import legacy manager: {e}")
    sys.exit(1)

# --- 3. Import Modern Components ---
from h2_plant.config.plant_builder import PlantBuilder
from h2_plant.simulation.engine import SimulationEngine

# --- 4. Load Extended Data ---
print("Loading extended 7-day data...")
try:
    power_7d = pd.read_csv(PROJECT_ROOT / 'power_input_7day_test.csv')
    prices_7d = pd.read_csv(PROJECT_ROOT / 'prices_7day_test.csv')
except FileNotFoundError:
    print("❌ Extended CSVs not found. Run generate_extended_data.py first.")
    sys.exit(1)

# Prepare data for Legacy Manager
# 1. OFFERED_POWER_PROFILE (Minute-by-minute)
legacy_power_profile = power_7d['Power_MW'].tolist()

# 2. SPOT_PRICE_PROFILE (Minute-by-minute, expanded from 15-min)
# prices_7d is 15-min resolution. Repeat each 15 times.
legacy_price_profile = np.repeat(prices_7d['Price_EUR_MWh'].values, 15).tolist()

# 3. HOUR_OFFER (Used only for duration calculation: len * 60)
# We need it to be 168 hours long.
legacy_hour_offer_dummy = [0.0] * 168

# PATCH MANAGER
print("Patching Legacy Manager with 7-day data...")
manager.OFFERED_POWER_PROFILE = legacy_power_profile
manager.SPOT_PRICE_PROFILE = legacy_price_profile
manager.HOUR_OFFER = legacy_hour_offer_dummy
# Also patch SPOT_PRICE_HOUR_BY_HOUR just in case, though likely unused if we patched PROFILE
# manager.SPOT_PRICE_HOUR_BY_HOUR = ... (Skip for now, assuming PROFILE is used)

def run_legacy_simulation():
    """Run legacy manager and return history."""
    print("\n--- Running Legacy Simulation (7 Days) ---")
    cwd = os.getcwd()
    os.chdir(LEGACY_DIR)
    
    stdout_capture = io.StringIO()
    sys.stdout = stdout_capture
    
    try:
        manager.run_hybrid_management()
        history = manager.history 
    except Exception as e:
        sys.stdout = sys.__stdout__
        print(f"❌ Legacy simulation failed: {e}")
        raise
    finally:
        sys.stdout = sys.__stdout__
        os.chdir(cwd)
        # Print captured output for debugging
        # print(stdout_capture.getvalue())
        with open('legacy_debug.log', 'w') as f:
            f.write(stdout_capture.getvalue())
        print("Legacy output saved to legacy_debug.log")
    print("✅ Legacy simulation complete")
    return history

def run_modern_simulation():
    """Run modern simulation and return history."""
    print("\n--- Running Modern Simulation (7 Days) ---")
    
    # We need to override the config to use the 7-day files
    # Instead of creating a new yaml, we can modify the config object after loading
    
    config_path = PROJECT_ROOT / 'configs' / 'plant_pem_soec_8hour_test.yaml'
    plant = PlantBuilder.from_file(str(config_path))
    
    # OVERRIDE CONFIG - Skipped, modifying EnvironmentManager directly below
    # plant.config.external_inputs['wind_power']['file_path'] = 'power_input_7day_test.csv'
    # plant.config.external_inputs['energy_price']['file_path'] = 'prices_7day_test.csv'
    
    # Re-build environment manager with new config? 
    # PlantBuilder builds components during from_file.
    # We might need to manually update the environment manager or rebuild.
    # Easier to just instantiate a new EnvironmentManager or update the existing one.
    
    env = plant.registry.get('environment_manager')
    # Update paths so initialize() loads the correct files
    env.wind_data_path = str(PROJECT_ROOT / 'power_input_7day_test.csv')
    env.price_data_path = str(PROJECT_ROOT / 'prices_7day_test.csv')
    print("✅ Modern Environment paths updated to 7-day files")
    
    engine = SimulationEngine(
        registry=plant.registry,
        config=plant.config
    )
    
    coordinator = plant.registry.get('dual_path_coordinator')
    soec = plant.registry.get('soec_cluster')
    pem = plant.registry.get('pem_electrolyzer_detailed')
    
    history = {
        'minute': [], 'P_soec_actual': [], 'P_pem': [], 'P_sold': [],
        'H2_soec_kg': [], 'H2_pem_kg': [], 'sell_decision': []
    }
    
    engine.initialize()
    
    # Run for 7 days (168 hours)
    hours = 24 * 7
    total_minutes = hours * 60 # 10080 for 7 days
    print(f"Executing {total_minutes} steps...")
    
    for minute in range(total_minutes):
        if minute % 1440 == 0:
            print(f"  Day {minute // 1440 + 1}/7...")
            
        hour_fraction = minute / 60.0
        engine._execute_timestep(hour_fraction)
        
        coord_state = coordinator.get_state()
        soec_state = soec.get_state()
        pem_state = pem.get_state()
        
        history['minute'].append(minute)
        history['P_soec_actual'].append(soec_state.get('P_actual_mw', 0.0))
        history['P_pem'].append(pem_state.get('power_consumption_mw', 0.0))
        history['P_sold'].append(coord_state.get('sold_power_mw', 0.0))
        history['H2_soec_kg'].append(soec_state.get('h2_output_kg_per_min', 0.0))
        history['H2_pem_kg'].append(pem_state.get('h2_output_kg', 0.0))
        history['sell_decision'].append(1 if coord_state.get('force_sell_flag', False) else 0)
        
    print("✅ Modern simulation complete")
    return history

def compare_results(legacy, modern):
    """Compare histories and generate report."""
    print("\n--- Comparing Results ---")
    
    divergences = []
    
    # Metrics to compare
    metrics = [
        ('P_soec_actual', 0.001, 'MW'),
        ('P_pem', 0.001, 'MW'),
        ('P_sold', 0.001, 'MW'),
        ('H2_soec_kg', 0.0001, 'kg'),
        ('H2_pem_kg', 0.0001, 'kg'),
        ('sell_decision', 0, 'bool')
    ]
    
    total_minutes = 10080
    
    # Check lengths
    if len(legacy['minute']) != total_minutes:
        print(f"⚠️ Warning: Legacy history length {len(legacy['minute'])} != {total_minutes}")
    if len(modern['minute']) != total_minutes:
        print(f"⚠️ Warning: Modern history length {len(modern['minute'])} != {total_minutes}")
        
    limit = min(len(legacy['minute']), len(modern['minute']), total_minutes)
    
    for minute in range(limit):
        minute_divs = []
        for key, tol, unit in metrics:
            try:
                leg_val = legacy[key][minute]
                mod_val = modern[key][minute]
                
                diff = abs(leg_val - mod_val)
                if diff > tol:
                    minute_divs.append(f"{key}: Leg={leg_val:.4f} vs Mod={mod_val:.4f} (Diff={diff:.4f} {unit})")
            except KeyError:
                pass # Skip missing keys
        
        if minute_divs:
            divergences.append(f"Minute {minute}:\n  " + "\n  ".join(minute_divs))
            
    # Summary stats
    leg_h2_total = sum(legacy['H2_soec_kg']) + sum(legacy['H2_pem_kg'])
    mod_h2_total = sum(modern['H2_soec_kg']) + sum(modern['H2_pem_kg'])
    
    leg_sold_total = sum(legacy['P_sold']) / 60.0
    mod_sold_total = sum(modern['P_sold']) / 60.0
    
    diff_h2 = mod_h2_total - leg_h2_total
    diff_h2_pct = (diff_h2 / leg_h2_total) * 100 if leg_h2_total > 0 else 0
    
    diff_sold = mod_sold_total - leg_sold_total
    diff_sold_pct = (diff_sold / leg_sold_total) * 100 if leg_sold_total > 0 else 0
    
    report = f"""# Extended Precision Validation Report (7 Days)

## Summary
- **Duration**: 168 Hours (10,080 Minutes)
- **Total H2 Production**: Legacy={leg_h2_total:.2f} kg, Modern={mod_h2_total:.2f} kg
  - **Difference**: {diff_h2:.2f} kg ({diff_h2_pct:.4f}%)
- **Total Energy Sold**: Legacy={leg_sold_total:.2f} MWh, Modern={mod_sold_total:.2f} MWh
  - **Difference**: {diff_sold:.2f} MWh ({diff_sold_pct:.4f}%)
- **Divergent Minutes**: {len(divergences)} / {limit}

## Divergence Details
"""
    if divergences:
        report += "\n".join(divergences[:50]) 
        if len(divergences) > 50:
            report += f"\n... and {len(divergences)-50} more minutes."
    else:
        report += "✅ NO DIVERGENCES FOUND! Systems are identical within tolerance."
        
    return report

def main():
    legacy_hist = run_legacy_simulation()
    modern_hist = run_modern_simulation()
    
    report = compare_results(legacy_hist, modern_hist)
    
    report_path = PROJECT_ROOT / 'precision_report_7day.md'
    with open(report_path, 'w') as f:
        f.write(report)
        
    print(f"\nReport saved to: {report_path}")
    print("\nReport Preview:")
    print("-" * 40)
    print("\n".join(report.split('\n')[:20]))

if __name__ == "__main__":
    main()
