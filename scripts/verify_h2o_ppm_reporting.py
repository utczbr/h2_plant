import sys
import os
import pandas as pd
import numpy as np

# Add project root
sys.path.append("/home/stuart/Documentos/Planta Hidrogenio")

from h2_plant.components.electrolysis.soec_operator import SOECOperator
from h2_plant.components.thermal.interchanger import Interchanger
from h2_plant.components.cooling.dry_cooler import DryCooler
from h2_plant.core.stream import Stream

def run_test():
    print("### H2O PPM Reporting Verification ###")
    
    # 1. Setup Chain
    soec_config = {
        'num_modules': 6, 'max_power_nominal_mw': 2.4, 'optimal_limit': 0.80,
        'steam_input_ratio_kg_per_kg_h2': 10.3,
        'entrained_water_fraction': 0.0009, 
        'o2_crossover_ppm_molar': 200.0,
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
    
    dry_cooler = DryCooler("DryCooler")
    dry_cooler.initialize(dt=1.0, registry=mock_reg)
    
    # 2. Execute Step
    soec.receive_input('power_in', 3.0) 
    soec.receive_input('water_in', Stream(2000.0, composition={'H2O': 1.0}))
    soec.step(t=0.0)
    h2_out = soec.get_output('h2_out')
    
    # Interchanger Pass-Through
    interchanger.receive_input('hot_in', h2_out)
    interchanger.receive_input('cold_in', Stream(2000.0, 298.15, 101325.0, {'H2O': 1.0}, 'liquid'))
    interchanger.step(t=0.0)
    ic_out = interchanger.get_output('hot_out')
    
    # Dry Cooler Handling
    dry_cooler.receive_input('fluid_in', ic_out)
    dry_cooler.step(t=0.0)
    
    # 3. Verify State Exports
    components = [
        ("SOEC", soec), 
        ("Interchanger", interchanger), 
        ("DryCooler", dry_cooler)
    ]
    
    for name, comp in components:
        state = comp.get_state()
        export_val = state.get('outlet_H2O_molf', None)
        
        print(f"\n[{name}]")
        if export_val is None:
            print("FAILURE: 'outlet_H2O_molf' NOT exported.")
        else:
            ppm = export_val * 1e6
            print(f"Exported H2O Mole Fraction: {export_val:.6f}")
            print(f"Equivalent PPM:             {ppm:.1f}")
            
            # Use Stream Helper as Ground Truth
            stream = comp.get_output('h2_out' if name=='SOEC' else 'hot_out' if name=='Interchanger' else 'fluid_out')
            truth = stream.get_total_mole_frac('H2O')
            
            if abs(export_val - truth) < 1e-9:
                print("SUCCESS: Matches physical stream total water.")
            else:
                print(f"FAILURE: Mismatch (Export={export_val} vs Truth={truth})")

if __name__ == "__main__":
    run_test()
