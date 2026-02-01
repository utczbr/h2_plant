import sys
import os
import logging
from pathlib import Path

# Setup path
sys.path.append(os.getcwd())

from h2_plant.components.electrolysis.pem_electrolyzer import DetailedPEMElectrolyzer
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.stream import Stream

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DebugPEMWater")

def test_water_starvation():
    print("\n--- Testing PEM Water Starvation ---")
    
    # 1. Setup Component
    registry = ComponentRegistry()
    config = {
        'max_power_mw': 5.0,
        'base_efficiency': 0.65,
        'component_id': 'pem_test'
    }
    pem = DetailedPEMElectrolyzer(config)
    registry.register('pem_test', pem)
    pem.initialize(dt=1/60, registry=registry)
    
    # 2. Case A: Power ON, Water OFF
    print("\n[Case A] Power: 2.5 MW, Water: None")
    pem.set_power_input_mw(2.5)
    pem.step(0.0)
    
    state_a = pem.get_state()
    print(f"  Water Buffer: {pem.water_buffer_kg:.4f} kg")
    print(f"  H2 Production: {state_a['h2_production_kg_h']:.4f} kg/h")
    
    if state_a['h2_production_kg_h'] < 0.1:
        print("  -> Result: No Production (Expected due to starvation)")
    else:
        print("  -> Result: UNEXPECTED PRODUCTION!")

    # 3. Case B: Power ON, Water ON
    print("\n[Case B] Power: 2.5 MW, Water: 1000 kg/h")
    
    # Create water stream
    water_stream = Stream(
        mass_flow_kg_h=1000.0,
        temperature_k=298.15,
        pressure_pa=101325,
        composition={'H2O': 1.0},
        phase='liquid'
    )
    
    # Manually simulate FlowNetwork delivery
    accepted = pem.receive_input('water_in', water_stream, 'water')
    print(f"  PEM accepted: {accepted:.2f} kg/h water")
    
    pem.step(1/60) # Next step
    state_b = pem.get_state()
    
    print(f"  Water Buffer: {pem.water_buffer_kg:.4f} kg")
    print(f"  H2 Production: {state_b['h2_production_kg_h']:.4f} kg/h")
    
    if state_b['h2_production_kg_h'] > 10.0:
        print("  -> Result: Production Active (Confirmed)")
    else:
        print("  -> Result: STILL NO PRODUCTION (Bug in component logic?)")

if __name__ == "__main__":
    test_water_starvation()
