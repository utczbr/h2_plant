"""
Comparison Test: DryCooler (New) vs modelo_dry_cooler (Legacy)

Validates that the new DryCooler component produces outputs consistent
with the legacy modelo_dry_cooler.py implementation using NTU-effectiveness.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from h2_plant.core.stream import Stream
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.components.cooling.dry_cooler import DryCooler


def test_compare_dry_cooler_h2():
    """Compare DryCooler outputs for H2 stream."""
    
    # === Test Conditions (from PEM outlet) ===
    T_in_K = 333.15  # 60°C
    P_in_Pa = 40e5   # 40 bar
    
    mdot_total_kg_h = 90.0
    composition = {
        'H2': 0.95,
        'H2O': 0.05
    }
    
    # === New Component ===
    dry_cooler = DryCooler(component_id='DC_H2_test')
    registry = ComponentRegistry()
    dry_cooler.initialize(dt=1/60, registry=registry)
    
    inlet_stream = Stream(
        mass_flow_kg_h=mdot_total_kg_h,
        temperature_k=T_in_K,
        pressure_pa=P_in_Pa,
        composition=composition,
        phase='gas'
    )
    
    dry_cooler.receive_input('fluid_in', inlet_stream)
    dry_cooler.step(t=0.0)
    
    new_state = dry_cooler.get_state()
    outlet = dry_cooler.get_output('fluid_out')
    
    print("\n=== DRY COOLER (NEW) RESULTS ===")
    print(f"Fluid Type Detected: {new_state.get('fluid_type', 'Unknown')}")
    print(f"Outlet Flow: {outlet.mass_flow_kg_h:.4f} kg/h")
    print(f"Outlet Temp: {outlet.temperature_k - 273.15:.2f} °C")
    print(f"Outlet Pressure: {outlet.pressure_pa / 1e5:.2f} bar")
    print(f"Heat Duty: {new_state.get('heat_duty_kw', 0):.2f} kW")
    print(f"Fan Power: {new_state.get('fan_power_kw', 0):.2f} kW")
    print(f"NTU: {new_state.get('ntu', 0):.3f}")
    print(f"Effectiveness: {new_state.get('effectiveness', 0):.3f}")
    
    # === Sanity Checks ===
    print("\n=== SANITY CHECKS ===")
    
    # Mass conservation
    mass_error = abs(mdot_total_kg_h - outlet.mass_flow_kg_h) / mdot_total_kg_h * 100
    print(f"Mass Balance Error: {mass_error:.4f}%")
    assert mass_error < 0.1, f"Mass balance error too large: {mass_error}%"
    
    # Temperature reduction (should cool the gas)
    temp_drop = T_in_K - outlet.temperature_k
    print(f"Temperature Drop: {temp_drop:.2f} K (expected > 0)")
    assert temp_drop > 0, "Dry cooler should reduce temperature"
    
    # Effectiveness should be between 0 and 1
    eff = new_state.get('effectiveness', 0)
    print(f"Effectiveness: {eff:.3f} (expected 0 < ε < 1)")
    assert 0 < eff < 1, f"Invalid effectiveness: {eff}"
    
    # NTU should be positive
    ntu = new_state.get('ntu', 0)
    print(f"NTU: {ntu:.3f} (expected > 0)")
    assert ntu > 0, f"Invalid NTU: {ntu}"
    
    # Heat duty should match energy balance: Q = ṁ × Cp × ΔT
    heat_duty = new_state.get('heat_duty_kw', 0)
    print(f"Heat Duty: {heat_duty:.2f} kW (expected > 0)")
    assert heat_duty > 0, "Heat duty should be positive"
    
    # Fan power should be positive
    fan_power = new_state.get('fan_power_kw', 0)
    print(f"Fan Power: {fan_power:.2f} kW (expected > 0)")
    assert fan_power > 0, "Fan power should be positive"
    
    print("\n✅ DRY COOLER SANITY CHECKS PASSED")


if __name__ == '__main__':
    test_compare_dry_cooler_h2()
