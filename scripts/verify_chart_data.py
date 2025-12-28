import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from h2_plant.run_integrated_simulation import run_with_dispatch_strategy
from h2_plant.visualization.graph_generator import GraphGenerator
from h2_plant.visualization.metrics_collector import MetricsCollector

def check_history_keys(history, components):
    """Check if history contains required keys for components."""
    missing = []
    
    # Define what we expect based on logic gaps found
    expected_data_keys = {
        'KOD': ['_water_removed_kg_h', '_drain_temp_k', '_drain_pressure_bar', '_dissolved_gas_ppm', '_outlet_o2_ppm_mol'],
        'Chiller': ['_cooling_load_kw', '_electrical_power_kw', '_water_condensed_kg_h', '_outlet_temp_c', '_outlet_o2_ppm_mol'],
        'DryCooler': ['_fan_power_kw', '_outlet_temp_c', '_outlet_o2_ppm_mol', '_heat_rejected_kw', '_tqc_duty_kw', '_dc_duty_kw'],
        'HydrogenMultiCyclone': ['_delta_p_bar', '_drain_flow_kg_h', '_outlet_o2_ppm_mol'],
        'Coalescer': ['_delta_p_bar', '_drain_flow_kg_h', '_outlet_o2_ppm_mol', '_drain_temp_k', '_drain_pressure_bar', '_dissolved_gas_ppm']
    }
    
    print("\n--- Checking History Keys ---")
    keys = set(history.keys())
    
    for comp_name, comp_type in components.items():
        if comp_type in expected_data_keys:
            for suffix in expected_data_keys[comp_type]:
                # Dynamic check for key
                # Some keys are strictly f"{comp_id}{suffix}"
                # Others might be aliases
                
                target_key = f"{comp_name}{suffix}"
                
                found = False
                if target_key in keys:
                    found = True
                else:
                    # Try looking for alias or match in existing keys
                    for k in keys:
                        if comp_id in k and suffix in k:
                            found = True
                            break
                    
                if not found:
                    missing.append(target_key)
                    print(f"[MISSING] {target_key} for {comp_id} ({comp_type})")
                else:
                    print(f"[OK] {target_key}")

    return missing

def main():
    scenarios_dir = str(project_root / "scenarios")
    
    print(f"Running short simulation (1 hour) to generate history...")
    try:
        # Run only 1 hour to be fast
        history = run_with_dispatch_strategy(scenarios_dir, hours=1)
    except Exception as e:
        print(f"Simulation failed: {e}")
        return

    # Extract component list from history keys or topology (simplified inference)
    # We'll just infer from keys present 
    print(f"Generated history with {len(history)} keys.")
    
    # We know from analysis what components likely exist, but let's try to deduce a few to verify
    # based on keys like 'KOD_1_outlet_o2_ppm_mol'
    
    components = {
        'KOD_1': 'KnockOutDrum',
        'Coalescer_2': 'Coalescer',
        'Chiller_1': 'Chiller',
        'DryCooler_HX': 'DryCooler'
    }
    
    check_history_keys(history, components)

if __name__ == "__main__":
    main()
