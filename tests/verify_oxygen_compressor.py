
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath("/home/stuart/Documentos/Planta Hidrogenio"))

from h2_plant.components.compression.compressor_single import CompressorSingle
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.stream import Stream

def test_oxygen_compressor():
    print("--- Verifying Oxygen Compression ---")
    
    registry = ComponentRegistry()
    
    # Instantiate Compressor
    comp = CompressorSingle(
        max_flow_kg_h=200.0,
        inlet_pressure_bar=1.0,
        outlet_pressure_bar=2.0,
        isentropic_efficiency=0.75
    )
    comp.initialize(dt=1.0, registry=registry)
    
    # Create Oxygen Stream
    o2_stream = Stream(
        mass_flow_kg_h=100.0,
        temperature_k=298.15,
        pressure_pa=100000.0,
        composition={'O2': 1.0},
        phase='gas'
    )
    
    # Feed Stream
    comp.receive_input('h2_in', o2_stream, 'gas')
    comp.step(0.0)
    
    # Check Properties
    fluid, cp, gamma = comp._get_fluid_properties()
    print(f"Detected Fluid: {fluid} (Target: Oxygen)")
    print(f"Detected Cp: {cp} (Target: 918.0)")
    
    if fluid != 'Oxygen' or cp != 918.0:
        print("FAIL: Fluid detection failed")
    else:
        print("PASS: Fluid detection")
        
    print(f"Outlet Temperature: {comp.outlet_temperature_c:.2f} C")
    print(f"Specific Energy: {comp.specific_energy_kwh_kg:.4f} kWh/kg")
    
    # Sanity Check: O2 compression should take LESS work per kg than H2 due to lower Cp/R
    # H2 approx Cp 14300, O2 918.
    # W ~ Cp * dT.
    # So W_O2 should be order of magnitude lower than W_H2.
    # (Actually R/MM is similar ratio. R_H2 = 4124, R_O2 = 260).
    
    if comp.specific_energy_kwh_kg < 0.1: # Threshold for sanity
        print("PASS: Energy consumption reasonable for O2")
    else:
        print("WARN: Energy consumption looks high, check calc")

if __name__ == "__main__":
    test_oxygen_compressor()
