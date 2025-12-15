import sys
import os
import numpy as np

# Adjust path to find modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from h2_plant.core.stream import Stream
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.components.purification.deoxo_reactor import DeoxoReactor

def test_new_deoxo():
    print("\n=== NEW DEOXO COMPONENT ISOLATED TEST ===")
    
    # Inputs
    m_dot_kg_h = 92.0
    P_Pa = 39.0 * 1e5
    T_K = 293.15 # 20 C
    
    # New Component expects Mass Fractions (x)
    # To match Legacy 200 ppm MOLE, we need to convert.
    # y_O2 = 200e-6.
    # M_mix ~ 2.
    # M_O2 = 32.
    # x_O2 = (y_O2 * M_O2) / M_mix ~ 200e-6 * 32 / 2 ~ 3200e-6 = 0.0032 (Mass)
    # Let's perform precise conversion to be fair.
    
    y_O2 = 200e-6
    y_H2 = 1.0 - y_O2
    MW_H2 = 2.016e-3
    MW_O2 = 32.00e-3
    
    MW_mix = y_H2 * MW_H2 + y_O2 * MW_O2
    x_O2 = (y_O2 * MW_O2) / MW_mix
    x_H2 = (y_H2 * MW_H2) / MW_mix
    
    print(f"Inputs: Flow={m_dot_kg_h} kg/h, P={P_Pa/1e5} bar, T={T_K - 273.15} C")
    print(f"Composition (Target Mole): y_O2={y_O2:.2e}")
    print(f"Composition (Mass Input): x_O2={x_O2:.6f}, x_H2={x_H2:.6f}")
    
    composition = {'H2': x_H2, 'O2': x_O2}
    
    # Initialize
    deoxo = DeoxoReactor(component_id='deoxo_iso')
    registry = ComponentRegistry()
    deoxo.initialize(dt=1/60, registry=registry)
    
    inlet = Stream(
        mass_flow_kg_h=m_dot_kg_h,
        temperature_k=T_K,
        pressure_pa=P_Pa,
        composition=composition,
        phase='gas'
    )
    
    # Run
    deoxo.receive_input('inlet', inlet)
    deoxo.step(t=0.0)
    
    outlet = deoxo.get_output('outlet')
    state = deoxo.get_state()
    
    # Report
    print("\n--- Outputs ---")
    T_out_C = outlet.temperature_k - 273.15
    P_out_bar = outlet.pressure_pa / 1e5
    
    # Output Composition (Stream is Mass Fraction)
    x_O2_out = outlet.composition.get('O2', 0.0)
    
    # Convert Mass out -> Mole out for comparison
    # Ideally should correspond to y_O2_out from Legacy
    # MW_mix_out approx matched
    # y_O2_out = (x_O2_out / 32) * MW_mix
    
    print(f"T_out: {T_out_C:.4f} C")
    print(f"P_out: {P_out_bar:.4f} bar")
    print(f"x_O2_out: {x_O2_out:.2e} (Mass)")
    print(f"Conversion: {state.get('conversion_o2_percent', 0):.4f} %")
    print(f"T_peak: {state.get('peak_temperature_c', 0):.4f} C")
    
    delta_T = T_out_C - (T_K - 273.15)
    print(f"Delta T: {delta_T:.4f} C")

if __name__ == "__main__":
    test_new_deoxo()
