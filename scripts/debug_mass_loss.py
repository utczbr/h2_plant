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
    print("### Debugging Mass Loss (Zero Entrainment) ###")
    
    # 1. Config matching user's likely range (zero entrain)
    soec_config = {
        'num_modules': 6, 'max_power_nominal_mw': 2.4, 
        'entrained_water_fraction': 0.0, # Explicit 0
        'o2_crossover_ppm_molar': 200.0,
        'steam_input_ratio_kg_per_kg_h2': 10.3, # This yields ~23% H2O mole frac??
        # Let's check: 10.3 kg H2O / 1 kg H2.
        # Moles H2O = 10.3/18 = 0.57. Moles H2 = 1/2 = 0.5.
        # Mole Frac H2O = 0.57 / 1.07 = 53%. 
        # User has 23% H2O (234764 ppm). 
        # So User Input Ratio is LOWER. Probably ~3 kg/kg?
        # Or running at low utilization?
        # Anyhow, let's just run ANY config and see if Mass Drop appears.
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
    # Using 3MW to get ~1100 kg/h
    soec.receive_input('power_in', 3.0) 
    soec.receive_input('water_in', Stream(2000.0, composition={'H2O': 1.0}))
    soec.step(t=0.0)
    
    h2_out = soec.get_output('h2_out')
    soec_state = soec.get_state()
    
    # 3. Interchanger Step
    interchanger.receive_input('hot_in', h2_out)
    interchanger.receive_input('cold_in', Stream(2000.0, 298.15, 101325.0, {'H2O': 1.0}, 'liquid'))
    interchanger.step(t=0.0)
    
    ic_out = interchanger.get_output('hot_out')
    ic_state = interchanger.get_state()
    
    print(f"\n[SOEC Report]")
    print(f"Bulk Mass: {soec_state['outlet_mass_flow_kg_h']:.4f}")
    print(f"Entrained: {soec_state.get('outlet_entrained_mass_kg_h', 0):.4f}")
    
    print(f"\n[Interchanger Report]")
    print(f"Bulk Mass: {ic_state['outlet_mass_flow_kg_h']:.4f}")
    print(f"Entrained: {ic_state.get('outlet_entrained_mass_kg_h', 0):.4f}")
    
    print(f"\n[Comparison]")
    diff = soec_state['outlet_mass_flow_kg_h'] - ic_state['outlet_mass_flow_kg_h']
    print(f"Mass Drop (SOEC -> IC): {diff:.6f} kg/h")
    
    if abs(diff) > 0.001:
        print("FAILURE: Reproduced Mass Drop!")
        
        # Analyze Composition
        print("\n[Composition Check]")
        for s in ['H2', 'H2O', 'O2']:
            m1 = h2_out.composition.get(s, 0) * h2_out.mass_flow_kg_h
            m2 = ic_out.composition.get(s, 0) * ic_out.mass_flow_kg_h
            print(f"{s} Mass: {m1:.4f} -> {m2:.4f} (Diff: {m1-m2:.4f})")
    else:
        print("SUCCESS: No Mass Drop observed.")

if __name__ == "__main__":
    run_test()
