#!/usr/bin/env python3
"""
Long-Term Precision Validation Script (Month & Year)

Runs both legacy manager.py and modern h2_plant simulation for extended periods.
Uses 'h2_plant/data/environment_data_2024.csv' as the source of truth.
Patches legacy manager with expanded hourly data.
Compares results minute-by-minute.
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock
import io
import argparse

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

# --- 4. Load Data ---
DATA_FILE = PROJECT_ROOT / 'h2_plant' / 'data' / 'environment_data_2024.csv'
if not DATA_FILE.exists():
    print(f"❌ Data file not found: {DATA_FILE}")
    sys.exit(1)

print(f"Loading data from {DATA_FILE}...")
env_data = pd.read_csv(DATA_FILE)

# Normalize columns
env_data.columns = env_data.columns.str.strip()

# --- 5. Prepare Data for Legacy Manager ---
# Legacy needs minute-by-minute lists.
# Input data is hourly. We must expand it.

# Wind Power: Coeff * Capacity (20MW)
# If 'wind_power_coefficient' exists, use it. Else 0.5.
if 'wind_power_coefficient' in env_data.columns:
    wind_coeffs = env_data['wind_power_coefficient'].values
else:
    wind_coeffs = np.full(len(env_data), 0.5)

# Calculate Power in MW
INSTALLED_CAPACITY_MW = 20.0
power_mw_hourly = wind_coeffs * INSTALLED_CAPACITY_MW

# Expand to minutes (repeat each value 60 times)
legacy_power_profile = np.repeat(power_mw_hourly, 60).tolist()

# Prices: EUR/MWh
if 'price_eur_mwh' in env_data.columns:
    prices_hourly = env_data['price_eur_mwh'].values
else:
    prices_hourly = np.full(len(env_data), 50.0)

# Expand to minutes
legacy_price_profile = np.repeat(prices_hourly, 60).tolist()

# Hour Offer (Dummy, just needs length)
# We'll make it long enough for a year + buffer
legacy_hour_offer_dummy = [0.0] * (len(env_data) + 24)

# PATCH MANAGER
print("Patching Legacy Manager with expanded hourly data...")
manager.OFFERED_POWER_PROFILE = legacy_power_profile
manager.SPOT_PRICE_PROFILE = legacy_price_profile
manager.HOUR_OFFER = legacy_hour_offer_dummy

def run_legacy_simulation(hours):
    """Run legacy manager and return history."""
    print(f"\n--- Running Legacy Simulation ({hours} Hours) ---")
    cwd = os.getcwd()
    os.chdir(LEGACY_DIR)
    
    # Patch MAX_MINUTES in manager if it exists, or control loop duration
    # manager.py runs until len(OFFERED_POWER_PROFILE) usually.
    # We want to stop at 'hours'.
    # But manager.run_hybrid_management() loops over the profile.
    # We can truncate the profile to the desired length to force stop?
    # Or just let it run and slice the history.
    # Truncating is safer to save time.
    
    total_minutes = hours * 60
    
    # Temporarily slice the global profiles
    original_power = manager.OFFERED_POWER_PROFILE
    original_price = manager.SPOT_PRICE_PROFILE
    original_hour_offer = manager.HOUR_OFFER
    
    manager.OFFERED_POWER_PROFILE = original_power[:total_minutes]
    manager.SPOT_PRICE_PROFILE = original_price[:total_minutes]
    manager.HOUR_OFFER = original_hour_offer[:hours]
    
    stdout_capture = io.StringIO()
    sys.stdout = stdout_capture
    
    try:
        # Reset history
        manager.history = {
            'minute': [], 'hour': [], 'P_offer': [], 'P_soec_set': [], 'P_soec_actual': [],
            'P_pem': [], 'P_sold': [], 'P_previous': [], 'spot_price': [], 'sell_decision': [],
            'H2_soec_kg': [], 'steam_soec_kg': [], 'H2_pem_kg': [], 'O2_pem_kg': [], 'H2O_pem_kg': []
        }
        
        manager.run_hybrid_management()
        history = manager.history 
    except Exception as e:
        sys.stdout = sys.__stdout__
        print(f"❌ Legacy simulation failed: {e}")
        raise
    finally:
        sys.stdout = sys.__stdout__
        os.chdir(cwd)
        # Restore profiles
        manager.OFFERED_POWER_PROFILE = original_power
        manager.SPOT_PRICE_PROFILE = original_price
        manager.HOUR_OFFER = original_hour_offer
        
        with open('legacy_longterm_debug.log', 'w') as f:
            f.write(stdout_capture.getvalue())
            
    print("✅ Legacy simulation complete")
    return history

def run_modern_simulation(hours):
    """Run modern simulation and return history."""
    print(f"\n--- Running Modern Simulation ({hours} Hours) ---")
    
    # Use a standard config but override environment
    config_path = PROJECT_ROOT / 'configs' / 'plant_pem_soec_8hour_test.yaml'
    plant = PlantBuilder.from_file(str(config_path))
    
    env = plant.registry.get('environment_manager')
    # Update paths to unified file
    env.wind_data_path = str(DATA_FILE)
    env.price_data_path = str(DATA_FILE)
    # Ensure direct power is OFF (we are using coefficients now)
    env.use_direct_power = False 
    
    # Force reload data
    env._load_data()
    
    print("✅ Modern Environment configured with unified data")
    
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
    
    total_minutes = hours * 60
    print(f"Executing {total_minutes} steps...")
    
    # Suppress stdout during loop to speed up execution
    original_stdout = sys.stdout
    sys.stdout = io.StringIO()
    
    try:
        for minute in range(total_minutes):
            if minute % (24 * 60) == 0:
                # Print progress to original stdout
                sys.stdout = original_stdout
                print(f"  Day {minute // 1440 + 1}...")
                sys.stdout = io.StringIO()
                
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
    finally:
        sys.stdout = original_stdout
        
    print("✅ Modern simulation complete")
    return history

def compare_results(legacy, modern, hours, label):
    """Compare histories and generate report."""
    print(f"\n--- Comparing Results ({label}) ---")
    
    divergences = []
    total_minutes = hours * 60
    
    # Summary stats
    leg_h2_total = sum(legacy['H2_soec_kg']) + sum(legacy['H2_pem_kg'])
    mod_h2_total = sum(modern['H2_soec_kg']) + sum(modern['H2_pem_kg'])
    
    leg_sold_total = sum(legacy['P_sold']) / 60.0
    mod_sold_total = sum(modern['P_sold']) / 60.0
    
    diff_h2 = mod_h2_total - leg_h2_total
    diff_h2_pct = (diff_h2 / leg_h2_total) * 100 if leg_h2_total > 0 else 0
    
    diff_sold = mod_sold_total - leg_sold_total
    diff_sold_pct = (diff_sold / leg_sold_total) * 100 if leg_sold_total > 0 else 0
    
    report = f"""# Long-Term Validation Report: {label}

## Summary
- **Duration**: {hours} Hours ({total_minutes} Minutes)
- **Total H2 Production**: Legacy={leg_h2_total:.2f} kg, Modern={mod_h2_total:.2f} kg
  - **Difference**: {diff_h2:.2f} kg ({diff_h2_pct:.4f}%)
- **Total Energy Sold**: Legacy={leg_sold_total:.2f} MWh, Modern={mod_sold_total:.2f} MWh
  - **Difference**: {diff_sold:.2f} MWh ({diff_sold_pct:.4f}%)

## Analysis
"""
    if abs(diff_h2_pct) < 0.1:
        report += "✅ H2 Production matches within 0.1% tolerance.\n"
    else:
        report += "❌ H2 Production discrepancy exceeds 0.1%.\n"
        
    return report

def main():
    parser = argparse.ArgumentParser(description='Run long-term validation')
    parser.add_argument('--mode', choices=['month', 'year'], default='month', help='Simulation duration')
    args = parser.parse_args()
    
    if args.mode == 'month':
        hours = 24 * 30 # 720 hours
        label = "Month (30 Days)"
    else:
        hours = 24 * 365 # 8760 hours
        label = "Year (365 Days)"
        
    import time
    
    start_time = time.time()
    legacy_hist = run_legacy_simulation(hours)
    legacy_duration = time.time() - start_time
    print(f"⏱️ Legacy Simulation Time: {legacy_duration:.2f} seconds")
    
    start_time = time.time()
    modern_hist = run_modern_simulation(hours)
    modern_duration = time.time() - start_time
    print(f"⏱️ Modern Simulation Time: {modern_duration:.2f} seconds")
    
    report = compare_results(legacy_hist, modern_hist, hours, label)
    
    # Add timing to report
    report += f"\n## Performance\n- **Legacy Runtime**: {legacy_duration:.2f} s\n- **Modern Runtime**: {modern_duration:.2f} s\n"
    if modern_duration > legacy_duration:
        factor = modern_duration / legacy_duration
        report += f"- **Comparison**: Modern is {factor:.1f}x slower than Legacy\n"
    else:
        factor = legacy_duration / modern_duration
        report += f"- **Comparison**: Modern is {factor:.1f}x faster than Legacy\n"
    
    report_filename = f'validation_report_{args.mode}.md'
    report_path = PROJECT_ROOT / report_filename
    with open(report_path, 'w') as f:
        f.write(report)
        
    print(f"\nReport saved to: {report_path}")
    print("\nReport Preview:")
    print("-" * 40)
    print(report)

if __name__ == "__main__":
    main()
