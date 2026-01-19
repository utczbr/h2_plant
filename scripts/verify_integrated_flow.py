
import logging
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Adjust path to include project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from h2_plant.run_integrated_simulation import run_with_dispatch_strategy

logging.basicConfig(level=logging.ERROR)

def run_verification():
    print("Starting verification simulation (24 hours)...")
    
    # Run for 24 hours
    # This should be enough for tank to fill (13h) and start discharging
    try:
        history = run_with_dispatch_strategy(
            scenarios_dir="scenarios",
            hours=24,
            output_dir=None,
            strategy="REFERENCE_HYBRID"
        )
    except Exception as e:
        print(f"Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n--- History Keys (Subset) ---")
    keys = list(history.keys())
    # Filter for relevant keys
    relevant = [k for k in keys if 'HP_Compressor_S2' in k or 'LP_Storage' in k]
    print(relevant[:10], "...")

    print("\n--- Hourly Analysis ---")
    
    # Find exact column names
    tank_mass_col = next((k for k in keys if 'LP_Storage_Tank' in k and 'total_mass_kg' in k), None)
    tank_p_col = next((k for k in keys if 'LP_Storage_Tank' in k and ('avg_pressure_bar' in k or 'pressure' in k)), None)
    
    # Compressor flow? Usually timestep_energy_kwh is recorded, but maybe not mass flow unless explicitly added?
    # Actually, DetailedTankArray has 'total_discharged_kg'
    tank_discharged_col = next((k for k in keys if 'LP_Storage_Tank' in k and 'total_discharged_kg' in k), None)
    
    # If we can't find compressor flow, we infer it from tank discharge
    
    steps = len(list(history.values())[0])
    
    print(f"{'Hour':<6} | {'Tank Mass':<10} | {'Press(bar)':<10} | {'Discharged':<10} | {'Delta':<10}")
    
    prev_discharged = 0.0
    
    for i in range(0, steps, 60): # Hourly
        t = i / 60.0
        mass = history[tank_mass_col][i] if tank_mass_col else 0
        p = history[tank_p_col][i] if tank_p_col else 0
        discharged = history[tank_discharged_col][i] if tank_discharged_col else 0
        
        delta = discharged - prev_discharged
        prev_discharged = discharged
        
        print(f"{t:<6.1f} | {mass:<10.1f} | {p:<10.1f} | {discharged:<10.1f} | {delta:<10.1f}")
        
    # Check specifically at hour 20
    idx_20h = 20 * 60
    if idx_20h < steps:
        p_20 = history[tank_p_col][idx_20h] if tank_p_col else 0
        d_19 = history[tank_discharged_col][19*60] if tank_discharged_col else 0
        d_20 = history[tank_discharged_col][20*60] if tank_discharged_col else 0
        flow_last_hour = d_20 - d_19
        
        print(f"\nAt Hour 20 (P={p_20:.1f} bar): Discharged in last hour = {flow_last_hour:.1f} kg")
        if p_20 > 35 and flow_last_hour < 100:
             print("FAIL: Tank has pressure but discharge is low!")
        elif p_20 > 35:
             print("SUCCESS: Tank is discharging!")
        else:
             print("INCONCLUSIVE: Tank didn't reach pressure.")

if __name__ == "__main__":
    run_verification()
