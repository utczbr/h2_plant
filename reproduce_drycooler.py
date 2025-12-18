
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from h2_plant.core.stream import Stream
from h2_plant.components.cooling.dry_cooler import DryCooler

def reproduction():
    print(" reproducing DryCooler O2 impurity issue...")
    
    # 1. Create Input Stream with O2 impurity and Extra Water Mist
    # Mass flow 100 kg/h
    # Composition: 99.9% H2, 0.1% O2 (by mass)
    # Extra: entrained water mist = 0.01 kg/s = 36 kg/h
    
    m_dot_h = 100.0
    comp = {'H2': 0.999, 'O2': 0.001}
    # 0.01 kg/s = 36 kg/h
    extra = {'m_dot_H2O_liq_accomp_kg_s': 0.01}
    
    inlet_stream = Stream(
        mass_flow_kg_h=m_dot_h,
        temperature_k=350.0,
        pressure_pa=30e5, # 30 bar
        composition=comp,
        extra=extra
    )
    
    print(f"Inlet Stream:")
    print(f"  Mass Flow: {inlet_stream.mass_flow_kg_h} kg/h")
    print(f"  Composition: {inlet_stream.composition}")
    print(f"  Extra: {inlet_stream.extra}")
    
    # Calculate O2 mole fraction (should include water in denominator)
    o2_frac_in = inlet_stream.get_total_mole_frac('O2')
    o2_ppm_in = o2_frac_in * 1e6
    print(f"  Inlet O2 impurity: {o2_ppm_in:.2f} ppm")

    # 2. Simulate DryCooler
    dc = DryCooler("DC-test")
    dc.initialize(dt=1/60, registry=None)
    dc.receive_input('fluid_in', inlet_stream)
    dc.step(0.0)
    
    outlet_stream = dc.get_output('fluid_out')
    
    print("\nDryCooler Step Executed.\n")
    
    print(f"Outlet Stream:")
    if outlet_stream:
        print(f"  Mass Flow: {outlet_stream.mass_flow_kg_h} kg/h")
        print(f"  Composition: {outlet_stream.composition}")
        print(f"  Extra: {outlet_stream.extra}")
        
        o2_frac_out = outlet_stream.get_total_mole_frac('O2')
        o2_ppm_out = o2_frac_out * 1e6
        print(f"  Outlet O2 impurity: {o2_ppm_out:.2f} ppm")
        
        diff = o2_ppm_out - o2_ppm_in
        print(f"\nDifference: {diff:.2f} ppm")
        
        if diff > 1.0: # Threshold for significant change
            print("\nFAIL: Significant increase in O2 impurity observed!")
            print("Explanation: DryCooler dropped the 'extra' water content, reducing total moles, thus increasing O2 mole fraction.")
        else:
            print("\nPASS: No significant change.")
    else:
        print("Error: No outlet stream produced.")

if __name__ == "__main__":
    reproduction()
