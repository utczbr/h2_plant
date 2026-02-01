import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root
sys.path.append("/home/stuart/Documentos/Planta Hidrogenio")

from h2_plant.components.electrolysis.soec_operator import SOECOperator
from h2_plant.components.thermal.interchanger import Interchanger
from h2_plant.core.stream import Stream

def run_test():
    print("### Setting up Components ###")
    
    # 1. SOEC Setup
    soec_config = {
        'num_modules': 6,
        'max_power_nominal_mw': 2.4,
        'optimal_limit': 0.80,
        'steam_input_ratio_kg_per_kg_h2': 10.3, # From topology
        'o2_crossover_ppm_molar': 200.0,
        'entrained_water_fraction': 0.0,
        'out_pressure_pa': 101325.0
    }
    soec = SOECOperator(soec_config)
    # Mock registry
    class MockRegistry:
        def has(self, x): return False
        def get(self, x): return None
    soec.initialize(dt=1.0, registry=MockRegistry())
    
    # 2. Interchanger Setup
    interchanger = Interchanger(
        component_id="SOEC_H2_Interchanger_1",
        min_approach_temp_k=10.0,
        target_cold_out_temp_c=95.0
    )
    interchanger.initialize(dt=1.0, registry=MockRegistry())

    print("\n### Running SOEC Step ###")
    # Feed power and water
    soec.receive_input('power_in', 10.0) # 10 MW
    soec.receive_input('water_in', Stream(mass_flow_kg_h=5000.0, composition={'H2O': 1.0})) # Plenty of water
    
    soec.step(t=0.0)
    
    # Get SOEC Output
    h2_out = soec.get_output('h2_out')
    
    print(f"\n[SOEC Output Stream]")
    print(f"Mass Flow: {h2_out.mass_flow_kg_h:.4f} kg/h")
    print(f"Temp: {h2_out.temperature_k:.2f} K")
    print(f"Mass Fractions (Raw): {h2_out.composition}")
    print(f"Mole Fractions (Calc): {h2_out.mole_fractions}")

    print("\n### Running Interchanger Step ###")
    # Setup dummy cold stream (e.g., water return)
    cold_in = Stream(
        mass_flow_kg_h=2000.0,
        temperature_k=298.15, # 25 C
        composition={'H2O': 1.0},
        phase='liquid'
    )
    
    interchanger.receive_input('hot_in', h2_out)
    interchanger.receive_input('cold_in', cold_in)
    
    interchanger.step(t=0.0)
    
    hot_out = interchanger.get_output('hot_out')
    
    print(f"\n[Interchanger Hot Output Stream]")
    print(f"Mass Flow: {hot_out.mass_flow_kg_h:.4f} kg/h")
    print(f"Temp: {hot_out.temperature_k:.2f} K")
    print(f"Mass Fractions (Raw): {hot_out.composition}")
    print(f"Mole Fractions (Calc): {hot_out.mole_fractions}")
    
    print("\n### Comparison ###")
    diff = h2_out.mass_flow_kg_h - hot_out.mass_flow_kg_h
    print(f"Mass Loss: {diff:.6f} kg/h")
    
    # Check composition consistency
    print("\n[Mass Fraction Delta Check]")
    for s in ['H2', 'H2O', 'O2']:
        m1 = h2_out.composition.get(s, 0.0)
        m2 = hot_out.composition.get(s, 0.0)
        print(f"{s}: In={m1:.6f}, Out={m2:.6f}, Delta={m2-m1:.6e}")
        
    # Check H2O vs H2O_liq
    h2o_vap_in = h2_out.composition.get('H2O', 0.0)
    h2o_liq_in = h2_out.composition.get('H2O_liq', 0.0)
    total_h2o_in = h2o_vap_in + h2o_liq_in
    
    h2o_vap_out = hot_out.composition.get('H2O', 0.0)
    h2o_liq_out = hot_out.composition.get('H2O_liq', 0.0)
    total_h2o_out = h2o_vap_out + h2o_liq_out
    
    print(f"\nTotal Water Mass Fraction In: {total_h2o_in:.6f}")
    print(f"Total Water Mass Fraction Out: {total_h2o_out:.6f}")
    print(f"Water Fraction Delta: {total_h2o_out - total_h2o_in:.6e}")

if __name__ == "__main__":
    run_test()
