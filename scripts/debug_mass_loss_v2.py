import sys
import os
import pandas as pd
import numpy as np

# Add project root
sys.path.append("/home/stuart/Documentos/Planta Hidrogenio")

from h2_plant.components.electrolysis.soec_operator import SOECOperator
from h2_plant.components.thermal.interchanger import Interchanger
from h2_plant.core.stream import Stream

def run_test():
    print("### Debugging Mass Loss V2 (User Composition Match) ###")
    
    # 1. Config for High H2 / Lower Water (Match ~230,000 PPM = 23% Mole Frac)
    # 23% H2O mole frac means mostly H2.
    soec_config = {
        'num_modules': 6, 'max_power_nominal_mw': 2.4, 
        'entrained_water_fraction': 0.0, 
        'o2_crossover_ppm_molar': 200.0,
        'steam_input_ratio_kg_per_kg_h2': 2.5, # Lower steam ratio
        'out_pressure_pa': 101325.0
    }
    
    soec = SOECOperator(soec_config)
    class MockRegistry:
        def has(self, x): return False
        def get(self, x): return None
    mock_reg = MockRegistry()
    soec.initialize(dt=1.0, registry=mock_reg)
    
    interchanger = Interchanger("Interchanger")
    interchanger.initialize(dt=1.0, registry=mock_reg)
    
    # 2. Step
    soec.receive_input('power_in', 3.0) 
    soec.receive_input('water_in', Stream(2000.0, composition={'H2O': 1.0}))
    soec.step(t=0.0)
    
    # Get outputs
    h2_out = soec.get_output('h2_out')
    soec_state = soec.get_state()
    
    print("\n[SOEC Report]")
    print(f"Bulk Mass: {soec_state['outlet_mass_flow_kg_h']:.4f}")
    if 'outlet_H2O_molf' in soec_state:
        print(f"H2O PPM: {soec_state['outlet_H2O_molf'] * 1e6:.0f}")
    
    # Pass to Interchanger
    interchanger.receive_input('hot_in', h2_out)
    interchanger.receive_input('cold_in', Stream(2000.0, 25.0+273.15, 101325.0, {'H2O': 1.0}, 'liquid'))
    interchanger.step(t=0.0)
    
    ic_out = interchanger.get_output('hot_out')
    ic_state = interchanger.get_state()
    
    print("\n[Interchanger Report]")
    print(f"Bulk Mass: {ic_state['outlet_mass_flow_kg_h']:.4f}")
    if 'outlet_H2O_molf' in ic_state:
        print(f"H2O PPM: {ic_state['outlet_H2O_molf'] * 1e6:.0f}")
        
    print(f"\n[Comparison]")
    diff = soec_state['outlet_mass_flow_kg_h'] - ic_state['outlet_mass_flow_kg_h']
    print(f"Mass Drop (SOEC -> IC): {diff:.6f} kg/h")
    
    if abs(diff) > 0.001:
        print("FAILURE: Reproduced Mass Drop!")
    else:
        print("SUCCESS: No Mass Drop observed.")

if __name__ == "__main__":
    run_test()
