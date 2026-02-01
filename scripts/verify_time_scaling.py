
import sys
from pathlib import Path
import logging
import numpy as np

# Add repo root to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from h2_plant.components.electrolysis.soec_operator import SOECOperator
from h2_plant.core.component_registry import ComponentRegistry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_time")

def main():
    print("Verifying Time Scaling (1hr vs 60min)...")
    
    # Configuration
    config = {
        'num_modules': 1,
        'max_power_nominal_mw': 1.0,
        'optimal_limit': 1.0,
        'power_first_step_mw': 0.1,
        'ramp_step_mw': 0.1,
        'steam_input_ratio_kg_per_kg_h2': 9.0,
        'rotation_enabled': False
    }
    
    # === SCENARIO 1: 1 Step of 1 Hour ===
    print("\n--- Scenario A: 1 Hour Step (dt=1.0) ---")
    registry_a = ComponentRegistry()
    soec_a = SOECOperator(config)
    soec_a.set_component_id("SOEC_A")
    soec_a.initialize(dt=1.0, registry=registry_a)
    
    # Set Power Input
    power_mw = 0.5
    soec_a.receive_input('power_in', power_mw, 'electricity')
    # Force steady state (bypass ramp)
    soec_a.real_powers[:] = power_mw
    soec_a.real_states[:] = 3 # Operating
    
    # Step
    p_a, h2_a, steam_a = soec_a.step(t=0.0)
    
    print(f"Total H2 Produced: {h2_a:.6f} kg")
    print(f"Total Steam Consumed: {steam_a:.6f} kg")
    
    
    # === SCENARIO 2: 60 Steps of 1 Minute ===
    print("\n--- Scenario B: 60 Minute Steps (dt=1/60) ---")
    registry_b = ComponentRegistry()
    soec_b = SOECOperator(config)
    soec_b.set_component_id("SOEC_B")
    soec_b.initialize(dt=1.0/60.0, registry=registry_b)
    
    # Set Power Input (Constant)
    soec_b.receive_input('power_in', power_mw, 'electricity')
    # Force steady state
    soec_b.real_powers[:] = power_mw
    soec_b.real_states[:] = 3
    
    total_h2_b = 0.0
    total_steam_b = 0.0
    
    for i in range(60):
        t = i / 60.0 # Time in hours
        p_b, h2_step, steam_step = soec_b.step(t=t)
        total_h2_b += h2_step
        total_steam_b += steam_step
        
        # Power input persists in SOEC state, but we can re-send to be safe 
        # (though implementation stores it)
        # soec_b.receive_input('power_in', power_mw, 'electricity') 
        
    print(f"Total H2 Produced: {total_h2_b:.6f} kg")
    print(f"Total Steam Consumed: {total_steam_b:.6f} kg")
    
    # === COMPARISON ===
    print("\n--- Comparison ---")
    diff_h2 = abs(h2_a - total_h2_b)
    diff_steam = abs(steam_a - total_steam_b)
    
    print(f"H2 Difference: {diff_h2:.6e} kg")
    print(f"Steam Difference: {diff_steam:.6e} kg")
    
    if diff_h2 < 1e-5 and diff_steam < 1e-5:
        print("✅ Time Scaling Verified! Physics is consistent.")
    else:
        print("❌ Time Scaling FAILED! Significant discrepancy found.")
        sys.exit(1)

if __name__ == "__main__":
    main()
