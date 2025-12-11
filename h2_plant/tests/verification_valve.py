
import sys
import os
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parents[1]))

from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.component_ids import ComponentID
from h2_plant.core.stream import Stream
from h2_plant.components.control.valve import ThrottlingValve
from h2_plant.optimization.lut_manager import LUTManager

def test_throttling_valve():
    print("=== Testing Throttling Valve ===")
    
    # 1. Setup Registry and LUT
    registry = ComponentRegistry()
    lut_mgr = LUTManager()
    lut_mgr.initialize()
    registry.register(ComponentID.LUT_MANAGER.value, lut_mgr)
    
    # 2. Create Valve (Target: 30 bar -> 5 bar)
    config = {
        'component_id': 'Valve-Test',
        'P_out_pa': 5.0e5, # 5 bar
        'fluid': 'H2'
    }
    valve = ThrottlingValve(config)
    valve.initialize(1.0, registry)
    
    # 3. Create Input Stream (30 bar, 300 K, 10 kg/h)
    inlet = Stream(
        mass_flow_kg_h=10.0,
        temperature_k=300.0,
        pressure_pa=30.0e5, # 30 bar
        composition={'H2': 1.0}
    )
    
    print(f"Inlet: P={inlet.pressure_pa/1e5:.1f} bar, T={inlet.temperature_k:.2f} K")
    
    # 4. Run Step
    valve.receive_input('inlet', inlet, 'hydrogen')
    valve.step(0.0)
    
    outlet = valve.get_output('outlet')
    
    if outlet:
        print(f"Outlet: P={outlet.pressure_pa/1e5:.1f} bar, T={outlet.temperature_k:.2f} K")
        print(f"Delta T: {valve.delta_T:.2f} K")
        
        # Reference Check: H2 at 300K has NEGATIVE Joule-Thomson coefficient (Heats up on expansion)
        # So T_out should be > T_in
        if outlet.temperature_k > inlet.temperature_k:
             print("SUCCESS: Temperature increased as expected for H2 at 300K.")
        else:
             print("WARNING: Temperature decreased? Check LUT or Physics.")
             
        # Check Mass Conservation
        if abs(outlet.mass_flow_kg_h - inlet.mass_flow_kg_h) < 1e-6:
             print("SUCCESS: Mass conserved.")
        else:
             print("FAILURE: Mass leak.")
             
    else:
        print("FAILURE: No output stream.")
        
if __name__ == "__main__":
    test_throttling_valve()
