import sys
import os
import numpy as np
import pandas as pd
import importlib.util

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from h2_plant.orchestrator import Orchestrator

def run_new_simulation(hours=24):
    print(f"\n--- Running New Architecture Simulation ({hours} hours) ---")
    orch = Orchestrator('scenarios')
    history = orch.run_simulation(hours=hours)
    
    # Calculate Totals
    total_h2_kg = np.sum(history.get('H2_soec_kg', [])) + np.sum(history.get('H2_pem_kg', []))
    
    # Energy: Sum of P_soec, P_pem. Timestep is 1 minute.
    # MW * (1/60) h = MWh
    total_energy_mwh = (np.sum(history['P_soec_actual']) + np.sum(history['P_pem'])) / 60.0
    
    # Water/O2
    # steam_soec_kg is now Total Input Steam for SOEC
    total_h2o_kg = np.sum(history.get('steam_soec_kg', [])) + np.sum(history.get('H2O_pem_kg', []))
    
    o2_soec = np.sum(history.get('O2_soec_kg', []))
    if o2_soec == 0:
        o2_soec = np.sum(history.get('H2_soec_kg', [])) * 8.0
        
    total_o2_kg = o2_soec + np.sum(history.get('O2_pem_kg', []))
    
    return {
        "H2 Produced (kg)": total_h2_kg,
        "Energy Consumed (MWh)": total_energy_mwh,
        "H2O Consumed (kg)": total_h2o_kg,
        "O2 Produced (kg)": total_o2_kg
    }

def run_reference_hybrid_simulation():
    print(f"\n--- Running Reference Hybrid Simulation (24 hours) ---")
    
    original_cwd = os.getcwd()
    ref_dir = os.path.abspath("h2_plant/legacy/pem_soec_reference/ALL_Reference")
    
    try:
        os.chdir(ref_dir)
        sys.path.append(ref_dir)
        
        # Import manager dynamically
        import manager
        import importlib
        importlib.reload(manager) # Ensure fresh load
        
        # Suppress prints during execution if desired, but user might want to see them
        # manager.run_hybrid_management() prints a lot.
        
        # Run the simulation
        manager.run_hybrid_management()
        
        # Extract history
        hist = manager.history
        
        # Calculate Totals from History
        # manager.py calculates these at the end, but we can re-sum from history keys
        
        H2_soec_total = np.sum(hist['H2_soec_kg'])
        H2_pem_total = np.sum(hist['H2_pem_kg'])
        total_h2_kg = H2_soec_total + H2_pem_total
        
        E_soec = np.sum(hist['P_soec_actual']) / 60.0
        E_pem = np.sum(hist['P_pem']) / 60.0
        total_energy_mwh = E_soec + E_pem
        
        # Water
        # manager.py: calculate_total_water_consumed_per_step
        total_water_system_rate, _, _ = manager.calculate_total_water_consumed_per_step(hist)
        total_h2o_kg = np.sum(total_water_system_rate)
        
        # O2
        # manager.py doesn't explicitly log O2 total in history for SOEC, but we can infer
        # O2_pem is in history
        O2_pem_total = np.sum(hist['O2_pem_kg'])
        # SOEC O2 = H2_soec * 8
        O2_soec_total = H2_soec_total * 8.0
        total_o2_kg = O2_pem_total + O2_soec_total
        
        return {
            "H2 Produced (kg)": total_h2_kg,
            "Energy Consumed (MWh)": total_energy_mwh,
            "H2O Consumed (kg)": total_h2o_kg,
            "O2 Produced (kg)": total_o2_kg
        }
        
    except Exception as e:
        print(f"Error running reference simulation: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        os.chdir(original_cwd)
        if ref_dir in sys.path:
            sys.path.remove(ref_dir)

if __name__ == "__main__":
    # 1. Run Reference (Hybrid)
    ref_res = run_reference_hybrid_simulation()
    
    if ref_res:
        # 2. Run New System (24 Hours to match Reference)
        new_res = run_new_simulation(hours=24)
        
        print("\n" + "="*80)
        print(f"COMPARISON (24 HOURS)")
        print(f"NOTE: Reference System Capacity is ~19.4 MW (14.4 MW SOEC + 5 MW PEM)")
        print(f"      New System Capacity is ~19.4 MW (14.4 MW SOEC + 5 MW PEM)")
        print("="*80)
        print(f"{'Metric':<25} | {'New System':<15} | {'Reference':<15} | {'Diff (%)':<10}")
        print("-" * 80)
        
        for key in new_res:
            val_new = new_res[key]
            val_ref = ref_res[key]
            diff = ((val_new - val_ref) / val_ref * 100) if val_ref != 0 else 0.0
            print(f"{key:<25} | {val_new:15.2f} | {val_ref:15.2f} | {diff:+9.2f}%")
        print("="*80 + "\n")
