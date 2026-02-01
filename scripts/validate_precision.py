#!/usr/bin/env python3
"""
Precision Validation Script

Runs both legacy manager.py and modern h2_plant simulation.
Compares results minute-by-minute to ensure < 0.01% deviation.
Generates a detailed divergence report.
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
import importlib.util
from unittest.mock import MagicMock
import io

# --- 1. Setup Paths ---
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

LEGACY_DIR = PROJECT_ROOT / 'h2_plant' / 'legacy' / 'pem_soec_reference'
sys.path.insert(0, str(LEGACY_DIR))

# Mock matplotlib to avoid dependency issues
sys.modules['matplotlib'] = MagicMock()
sys.modules['matplotlib.pyplot'] = MagicMock()

# --- 2. Import Legacy Manager ---
try:
    # Attempt direct import first
    import manager
    print("✅ Legacy manager imported successfully")
except ImportError as e:
    print(f"❌ Failed to import legacy manager: {e}")
    sys.exit(1)

# --- 3. Import Modern Components ---
from h2_plant.config.plant_builder import PlantBuilder
from h2_plant.simulation.engine import SimulationEngine

def run_legacy_simulation():
    """Run legacy manager and return history."""
    print("\n--- Running Legacy Simulation ---")
    # Redirect stdout to suppress legacy print spam
    # sys.stdout = open(os.devnull, 'w')
    # Change to legacy dir to run manager
    cwd = os.getcwd()
    os.chdir(LEGACY_DIR)
    
    # Capture stdout
    stdout_capture = io.StringIO()
    sys.stdout = stdout_capture
    
    try:
        manager.run_hybrid_management()
        # Access global history directly since get_history was removed
        history = manager.history 
    finally:
        sys.stdout = sys.__stdout__
        os.chdir(cwd)
    print("✅ Legacy simulation complete")
    return history

def run_modern_simulation():
    """Run modern simulation and return history."""
    print("\n--- Running Modern Simulation ---")
    
    # Config path
    config_path = PROJECT_ROOT / 'configs' / 'plant_pem_soec_8hour_test.yaml'
    
    # Build plant
    plant = PlantBuilder.from_file(str(config_path))
    
    # Initialize engine
    engine = SimulationEngine(
        registry=plant.registry,
        config=plant.config
    )
    
    # Get components
    coordinator = plant.registry.get('dual_path_coordinator')
    soec = plant.registry.get('soec_cluster')
    pem = plant.registry.get('pem_electrolyzer_detailed')
    env = plant.registry.get('environment_manager')
    
    # History storage
    history = {
        'minute': [], 'P_soec_actual': [], 'P_pem': [], 'P_sold': [],
        'H2_soec_kg': [], 'H2_pem_kg': [], 'sell_decision': []
    }
    
    # Initialize components
    engine.initialize()
    
    # Run loop (480 minutes)
    for minute in range(480):
        hour_fraction = minute / 60.0
        engine._execute_timestep(hour_fraction)
        
        # Capture state
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
    
    total_minutes = 480
    
    for minute in range(total_minutes):
        minute_divs = []
        for key, tol, unit in metrics:
            leg_val = legacy[key][minute]
            mod_val = modern[key][minute]
            
            diff = abs(leg_val - mod_val)
            if diff > tol:
                minute_divs.append(f"{key}: Leg={leg_val:.4f} vs Mod={mod_val:.4f} (Diff={diff:.4f} {unit})")
        
        if minute_divs:
            divergences.append(f"Minute {minute}:\n  " + "\n  ".join(minute_divs))
            
    # Summary stats
    leg_h2_total = sum(legacy['H2_soec_kg']) + sum(legacy['H2_pem_kg'])
    mod_h2_total = sum(modern['H2_soec_kg']) + sum(modern['H2_pem_kg'])
    
    leg_sold_total = sum(legacy['P_sold']) / 60.0
    mod_sold_total = sum(modern['P_sold']) / 60.0
    
    report = f"""# Precision Validation Report

## Summary
- **Total H2 Production**: Legacy={leg_h2_total:.2f} kg, Modern={mod_h2_total:.2f} kg (Diff: {mod_h2_total-leg_h2_total:.2f} kg)
- **Total Energy Sold**: Legacy={leg_sold_total:.2f} MWh, Modern={mod_sold_total:.2f} MWh (Diff: {mod_sold_total-leg_sold_total:.2f} MWh)
- **Divergent Minutes**: {len(divergences)} / {total_minutes}

## Divergence Details
"""
    if divergences:
        report += "\n".join(divergences[:50]) # Limit to first 50 to avoid huge files
        if len(divergences) > 50:
            report += f"\n... and {len(divergences)-50} more minutes."
    else:
        report += "✅ NO DIVERGENCES FOUND! Systems are identical within tolerance."
        
    return report

def main():
    legacy_hist = run_legacy_simulation()
    modern_hist = run_modern_simulation()
    
    report = compare_results(legacy_hist, modern_hist)
    
    # Save report
    report_path = Path.home() / '.gemini/antigravity/brain/d72491b0-356d-4738-a80d-cced2586b1c8/precision_report.md'
    with open(report_path, 'w') as f:
        f.write(report)
        
    print(f"\nReport saved to: {report_path}")
    print("\nReport Preview:")
    print("-" * 40)
    print("\n".join(report.split('\n')[:20]))

if __name__ == "__main__":
    main()
