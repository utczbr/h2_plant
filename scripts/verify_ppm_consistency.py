import sys
import os
import pandas as pd
import numpy as np

# Add project root
sys.path.append("/home/stuart/Documentos/Planta Hidrogenio")

from h2_plant.components.electrolysis.soec_operator import SOECOperator
from h2_plant.core.stream import Stream

def run_test():
    print("### PPM Consistency Verification ###")
    
    # 1. Setup SOEC with entrainment (0.0009 fraction -> ~1kg/h mist)
    soec_config = {
        'num_modules': 6, 'max_power_nominal_mw': 2.4, 'optimal_limit': 0.80,
        'steam_input_ratio_kg_per_kg_h2': 10.3,
        'entrained_water_fraction': 0.0009, 
        'o2_crossover_ppm_molar': 200.0, # Target 200 ppm relative to H2
        'out_pressure_pa': 101325.0
    }
    
    soec = SOECOperator(soec_config)
    class MockRegistry:
        def has(self, x): return False
        def get(self, x): return None
    soec.initialize(dt=1.0, registry=MockRegistry())
    
    # 2. Execute Step (produce H2 + Steam + Mist + O2 crossover)
    soec.receive_input('power_in', 3.0) 
    soec.receive_input('water_in', Stream(2000.0, composition={'H2O': 1.0}))
    soec.step(t=0.0)
    
    state = soec.get_state()
    h2_out = soec.get_output('h2_out')
    
    print(f"\n[Output Stream]")
    print(f"Total Mass (Stream Obj, Bulk): {h2_out.mass_flow_kg_h:.4f} kg/h")
    entrained_kg_h = h2_out.extra.get('m_dot_H2O_liq_accomp_kg_s', 0.0) * 3600.0
    print(f"Entrained Liquid (Metadata): {entrained_kg_h:.4f} kg/h")
    
    print(f"\n[Reported State]")
    print(f"Outlet Mass Flow (Reported): {state['outlet_mass_flow_kg_h']:.4f} kg/h")
    print(f"Outlet O2 PPM (Reported):    {state['outlet_o2_ppm_mol']:.6f} ppm")
    
    # 3. Manual Verification Calculation
    # Calculate Total Moles (Gas + Liq)
    # Gas Moles = mass_flow / MW_mix
    # Liq Moles = entrained_mass / 18.015
    
    # Get physical moles from stream helper (Ground Truth)
    true_o2_frac = h2_out.get_total_mole_frac('O2')
    true_o2_ppm = true_o2_frac * 1e6
    
    print(f"\n[Verification]")
    print(f"Calculated True O2 PPM:      {true_o2_ppm:.6f} ppm")
    
    diff_ppm = abs(state['outlet_o2_ppm_mol'] - true_o2_ppm)
    print(f"Difference: {diff_ppm:.6e} ppm")
    
    if diff_ppm < 1e-9:
        print("SUCCESS: Reported PPM matches rigorous physical calculation including entrained liquid.")
    else:
        print("FAILURE: Reported PPM mismatch.")

if __name__ == "__main__":
    run_test()
