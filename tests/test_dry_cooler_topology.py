
import pytest
import sys
import os
sys.path.append(os.getcwd())

from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.stream import Stream
from h2_plant.core.constants import DryCoolerConstants as DCC
from h2_plant.components.cooling.dry_cooler import DryCooler
from h2_plant.components.separation.knock_out_drum import KnockOutDrum

def test_dry_cooler_separator_chain():
    """
    Test topology: Hot Wet H2 -> Dry Cooler -> Knock-Out Drum -> Dry(er) H2 + Water
    """
    registry = ComponentRegistry()
    dt = 1.0
    
    # 1. Initialize Components
    dc = DryCooler("dry_cooler_h2")
    kod = KnockOutDrum(diameter_m=0.5, gas_species='H2')
    
    dc.initialize(dt, registry)
    kod.initialize(dt, registry)
    
    # 2. Define Inlet Stream (Hot, Wet H2 from Electrolyzer)
    # 80°C, 30 bar, Saturation assumed or manually specified high water content
    # Composition: 90% H2, 10% H2O (mass? no, mole for stream usually)
    # Stream composition is Mole Fraction.
    input_stream = Stream(
        mass_flow_kg_h=100.0,
        temperature_k=353.15, # 80°C
        pressure_pa=30e5, # 30 bar
        composition={'H2': 0.8, 'H2O': 0.2}, # Very wet
        phase='mixed'
    )
    
    print("\n--- SIMULATION START ---")
    print(f"Inlet Source: {input_stream.mass_flow_kg_h} kg/h @ {input_stream.temperature_k:.2f} K")
    
    # 3. Dry Cooler Step
    dc.receive_input("fluid_in", input_stream)
    dc.receive_input("electricity_in", 999.0) # Unlimited power available
    dc.step(0.0)
    
    dc_out = dc.get_output("fluid_out")
    
    print("\n[Step 1] Dry Cooler Output:")
    print(f"  T: {dc_out.temperature_k:.2f} K")
    print(f"  P: {dc_out.pressure_pa/1e5:.4f} bar")
    print(f"  Fan Power: {dc.fan_power_kw:.4f} kW")
    
    assert dc_out.temperature_k < input_stream.temperature_k, "Dry Cooler failed to cool"
    
    # 4. Separator Step
    kod.receive_input("gas_inlet", dc_out)
    kod.step(0.0)
    
    gas_out = kod.get_output("gas_outlet")
    liq_drain = kod.get_output("liquid_drain")
    
    print("\n[Step 2] Knock-Out Drum Output:")
    if gas_out:
        print(f"  Gas Flow: {gas_out.mass_flow_kg_h:.4f} kg/h")
        print(f"  Gas H2O Mole Frac: {gas_out.composition.get('H2O'):.6f}")
    
    if liq_drain:
        print(f"  Liquid Drain: {liq_drain.mass_flow_kg_h:.4f} kg/h")
    else:
        print("  Liquid Drain: None")
        
    # Validation Logic
    # 1. Cooling happened
    assert dc_out.temperature_k < 313.15, "Should cool below 40C (approx)"
    
    # 2. Separation happened
    # Inlet has 20% water. Cooling to ~30C (303K) at 30 bar.
    # Psat(30C) ~ 4.2 kPa. P_tot = 3000 kPa. K ~ 4.2/3000 ~ 0.0014.
    # Equilibrium y_h2o should be very low.
    # Initial z_h2o = 0.2. Significant condensation expected.
    
    assert liq_drain is not None and liq_drain.mass_flow_kg_h > 1.0, "Separator should drain liquid water"
    assert gas_out.composition['H2O'] < 0.01, f"Gas outlet should be dry(er), got {gas_out.composition['H2O']}"
    assert gas_out.mass_flow_kg_h + liq_drain.mass_flow_kg_h == pytest.approx(input_stream.mass_flow_kg_h, rel=0.01), "Mass Balance Error"

if __name__ == "__main__":
    test_dry_cooler_separator_chain()
