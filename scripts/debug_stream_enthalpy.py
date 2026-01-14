
import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from h2_plant.core.stream import Stream

def test_enthalpy():
    print("Testing Stream Enthalpy...")
    
    # Create Inlet Stream (matched to SOEC_Chiller_1 diagnosis)
    # Flow: 530 kg/h
    # Comp: H2: 0.57, H2O: 0.43
    # T_in: 313.15 K (40 C)
    
    mass_flow = 529.77
    comp = {'H2': 0.5691, 'H2O': 0.4289, 'O2': 0.0018}
    
    s_in = Stream(
        mass_flow_kg_h=mass_flow,
        temperature_k=313.15,
        pressure_pa=0.9e5,
        composition=comp.copy()
    )
    
    # Target Stream (4 C)
    s_out = Stream(
        mass_flow_kg_h=mass_flow,
        temperature_k=277.15,
        pressure_pa=0.9e5,
        composition=comp.copy()
    )
    
    h_in = s_in.specific_enthalpy_j_kg
    h_out = s_out.specific_enthalpy_j_kg
    
    print(f"H_in (40C): {h_in:.2f} J/kg")
    print(f"H_out (4C): {h_out:.2f} J/kg")
    
    delta_h = h_in - h_out
    print(f"Delta H: {delta_h:.2f} J/kg")
    
    mass_flow_kg_s = mass_flow / 3600.0
    Q_kW = mass_flow_kg_s * delta_h / 1000.0
    
    print(f"Mass Flow: {mass_flow_kg_s:.4f} kg/s")
    print(f"Calculated Q: {Q_kW:.2f} kW")
    
    # Compare with 95 MW
    if Q_kW > 1000:
        print("RESULT: BUG REPRODUCED (High Value)")
    else:
        print("RESULT: NORMAL VALUE (Physics correct)")

if __name__ == "__main__":
    test_enthalpy()
