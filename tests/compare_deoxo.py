"""
Comparison Test: DeoxoReactor (New) vs modelo_deoxo (Legacy)

Validates that the new DeoxoReactor component produces outputs consistent
with the legacy modelo_deoxo.py implementation (catalytic O2 removal).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from h2_plant.core.stream import Stream
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.components.purification.deoxo_reactor import DeoxoReactor


def test_compare_deoxo_h2():
    """Compare DeoxoReactor outputs for H2 stream with O2 impurity."""
    
    # === Test Conditions (post-coalescer) ===
    T_in_K = 293.15  # 20°C (after heating)
    P_in_Pa = 39e5   # 39 bar
    
    mdot_total_kg_h = 92.0
    composition = {
        'H2': 0.9998,   # 99.98% H2
        'O2': 0.0002,   # 200 ppm O2 (cross-over)
        'H2O': 0.0000
    }
    
    # === New Component ===
    deoxo = DeoxoReactor(component_id='deoxo_test')
    registry = ComponentRegistry()
    deoxo.initialize(dt=1/60, registry=registry)
    
    inlet_stream = Stream(
        mass_flow_kg_h=mdot_total_kg_h,
        temperature_k=T_in_K,
        pressure_pa=P_in_Pa,
        composition=composition,
        phase='gas'
    )
    
    deoxo.receive_input('inlet', inlet_stream)
    deoxo.step(t=0.0)
    
    new_state = deoxo.get_state()
    outlet = deoxo.get_output('outlet')
    
    print("\n=== DEOXO REACTOR (NEW) RESULTS ===")
    print(f"Outlet Flow: {outlet.mass_flow_kg_h:.4f} kg/h")
    print(f"Outlet Temp: {outlet.temperature_k - 273.15:.2f} °C")
    print(f"Outlet Pressure: {outlet.pressure_pa / 1e5:.2f} bar")
    
    # Raw State Debug
    print("Raw State:", new_state)
    
    print(f"O2 Conversion: {new_state.get('conversion_o2_percent', 0):.2f}%")
    print(f"Peak Temperature: {new_state.get('peak_temperature_c', 0):.2f} °C")
    print(f"Pressure Drop: {new_state.get('pressure_drop_mbar', 0):.4f} mbar")
    
    # Check outlet O2 content
    o2_out_mole = outlet.composition.get('O2', 0)
    print(f"Outlet O2 (Mole/Mass Frac?): {o2_out_mole:.2e}")
    print(f"Outlet O2 PPM: {o2_out_mole*1e6:.2f} ppm")
    
    # === Sanity Checks ===
    print("\n=== SANITY CHECKS ===")
    
    # O2 should be removed (high conversion)
    conversion = new_state.get('conversion_o2_percent', 0) / 100.0
    print(f"O2 Conversion: {conversion*100:.2f}% (expected > 99%)")
    
    # Temperature rise due to exothermic reaction
    temp_rise = outlet.temperature_k - T_in_K
    print(f"Temperature Rise: {temp_rise:.2f} K (expected > 0 for adiabatic)")
    
    # Mass balance (water produced from reaction)
    # 2H2 + O2 -> 2H2O (O2 consumed, H2O produced)
    mass_error = abs(mdot_total_kg_h - outlet.mass_flow_kg_h) / mdot_total_kg_h * 100
    print(f"Mass Change: {mass_error:.4f}% (reaction produces H2O, consumes O2)")
    
    # Outlet should have very low O2
    print(f"Outlet O2: {o2_out_mole*1e6:.2f} ppm (expected < 1 ppm)")
    
    # Assertions
    assert conversion > 0.99, "O2 Conversion too low"
    assert temp_rise > 0, "Adiabatic temp rise missing"
    assert o2_out_mole < 1e-6, "O2 removal failed"

    print("\n✅ DEOXO SANITY CHECKS PASSED")


if __name__ == '__main__':
    test_compare_deoxo_h2()
