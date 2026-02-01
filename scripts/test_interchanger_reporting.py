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
    print("### Reporting Consistency Test ###")
    
    # Setup components
    soec_config = {
        'num_modules': 6, 'max_power_nominal_mw': 2.4, 'optimal_limit': 0.80,
        'steam_input_ratio_kg_per_kg_h2': 10.3,
        'entrained_water_fraction': 0.0009, 
        'out_pressure_pa': 101325.0
    }
    soec = SOECOperator(soec_config)
    class MockRegistry:
        def has(self, x): return False
        def get(self, x): return None
    soec.initialize(dt=1.0, registry=MockRegistry())
    
    interchanger = Interchanger("Interchanger")
    interchanger.initialize(dt=1.0, registry=MockRegistry())
    
    dry_cooler = DryCooler("DryCooler")
    dry_cooler.initialize(dt=1.0, registry=MockRegistry())

    # Execution
    soec.receive_input('power_in', 3.0) 
    soec.receive_input('water_in', Stream(2000.0, composition={'H2O': 1.0}))
    soec.step(t=0.0)
    h2_out = soec.get_output('h2_out')
    
    cold_in = Stream(2000.0, 298.15, 101325.0, {'H2O': 1.0}, 'liquid')
    interchanger.receive_input('hot_in', h2_out)
    interchanger.receive_input('cold_in', cold_in)
    interchanger.step(t=0.0)
    ic_out = interchanger.get_output('hot_out')
    
    dry_cooler.receive_input('fluid_in', ic_out)
    dry_cooler.step(t=0.0)
    dc_out = dry_cooler.get_output('fluid_out')
    
    # Verification
    print(f"\n[Physical Stream Mass Flow]")
    print(f"Interchanger Stream: {ic_out.mass_flow_kg_h:.4f} kg/h (Bulk Only)")
    print(f"DryCooler Stream:    {dc_out.mass_flow_kg_h:.4f} kg/h (Merged)")
    
    print(f"\n[Reported State Mass Flow (Graph Data)]")
    ic_state = interchanger.get_state()
    # DryCooler state doesn't report mass explicitly in get_state by default in typical Component, 
    # but we compare against the Stream value usually plotted or the downstream input.
    # We check if Interchanger State now MATCHES DryCooler Stream.
    
    ic_reported = ic_state.get('outlet_mass_flow_kg_h', 0.0)
    print(f"Interchanger Reported: {ic_reported:.4f} kg/h")
    
    diff = abs(ic_reported - dc_out.mass_flow_kg_h)
    print(f"\nDifference (Reported IC vs Physical DC): {diff:.4f} kg/h")
    
    if diff < 0.001:
        print("SUCCESS: Interchanger reporting matches downstream total mass.")
    else:
        print("FAILURE: Reporting mismatch persists.")

if __name__ == "__main__":
    run_test()
