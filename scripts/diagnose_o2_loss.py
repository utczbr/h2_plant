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
    print("### Diagnostic: Species Mass Conservation ###")
    
    # 1. Config (High O2 Crossover to make it visible)
    soec_config = {
        'num_modules': 6, 'max_power_nominal_mw': 2.4, 
        'entrained_water_fraction': 0.0, 
        'o2_crossover_ppm_molar': 5000.0, # 5000 ppm = 0.5% (Large enough to track)
        'steam_input_ratio_kg_per_kg_h2': 2.5,
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
    
    # 2. Step SOEC
    soec.receive_input('power_in', 3.0) 
    soec.receive_input('water_in', Stream(2000.0, composition={'H2O': 1.0}))
    soec.step(t=0.0)
    
    h2_out = soec.get_output('h2_out')
    
    # 3. Step Interchanger
    interchanger.receive_input('hot_in', h2_out)
    interchanger.receive_input('cold_in', Stream(2000.0, 25.0+273.15, 101325.0, {'H2O': 1.0}, 'liquid'))
    interchanger.step(t=0.0)
    
    # 4. Detailed Analysis
    def analyze_stream(name, s):
        print(f"\n[{name}]")
        print(f"Total Mass: {s.mass_flow_kg_h:.4f} kg/h")
        print(f"Composition: {s.composition}")
        
        species_mass = {}
        total_check = 0.0
        for sp, frac in s.composition.items():
            m = frac * s.mass_flow_kg_h
            species_mass[sp] = m
            total_check += m
            print(f"  - {sp}: {m:.6f} kg/h")
            
        print(f"  SUM of Species: {total_check:.6f} kg/h")
        return species_mass

    in_mass = analyze_stream("SOEC Output (Interchanger Input)", h2_out)
    ic_out = interchanger.get_output('hot_out')
    out_mass = analyze_stream("Interchanger Output", ic_out)
    
    print("\n[Conservation Check]")
    loss = h2_out.mass_flow_kg_h - ic_out.mass_flow_kg_h
    print(f"Global Mass Loss: {loss:.6f} kg/h")
    
    for sp in in_mass:
        m_in = in_mass[sp]
        m_out = out_mass.get(sp, 0.0)
        d = m_in - m_out
        if abs(d) > 1e-6:
            print(f"FAILURE: {sp} Loss = {d:.6f} kg/h")
        else:
            print(f"SUCCESS: {sp} Conserved")

if __name__ == "__main__":
    run_test()
