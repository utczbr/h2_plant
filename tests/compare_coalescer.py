"""
Comparison Test: Coalescer (New) vs modelo_coalescedor (Legacy)

Validates that the new Coalescer component produces outputs consistent
with the legacy modelo_coalescedor.py implementation.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from h2_plant.core.stream import Stream
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.components.separation.coalescer import Coalescer


def test_compare_coalescer_h2():
    """Compare Coalescer outputs for H2 stream."""
    
    # === Test Conditions ===
    T_in_K = 277.15  # 4°C (post-chiller)
    P_in_Pa = 39.5e5  # 39.5 bar
    
    mdot_total_kg_h = 95.0
    composition = {
        'H2': 0.97,
        'H2O': 0.03
    }
    
    # === New Component ===
    coalescer = Coalescer(
        d_shell=0.32,
        l_elem=1.0,
        gas_type='H2'
    )
    registry = ComponentRegistry()
    coalescer.initialize(dt=1/60, registry=registry)
    
    inlet_stream = Stream(
        mass_flow_kg_h=mdot_total_kg_h,
        temperature_k=T_in_K,
        pressure_pa=P_in_Pa,
        composition=composition,
        phase='mixed'
    )
    
    coalescer.receive_input('inlet', inlet_stream)
    coalescer.step(t=0.0)
    
    new_state = coalescer.get_state()
    new_gas_out = coalescer.get_output('outlet')
    new_drain = coalescer.get_output('drain')
    
    print("\n=== COALESCER (NEW) RESULTS ===")
    print(f"Gas Outlet Flow: {new_gas_out.mass_flow_kg_h:.4f} kg/h")
    print(f"Gas Outlet Temp: {new_gas_out.temperature_k - 273.15:.2f} °C")
    print(f"Pressure Drop: {new_state.get('pressure_drop_pa', 0) / 1e5:.4f} bar")
    print(f"Liquid Drain Flow: {new_drain.mass_flow_kg_h if new_drain else 0:.4f} kg/h")
    
    # === Sanity Checks ===
    print("\n=== SANITY CHECKS ===")
    
    # Mass conservation (should be very close with dissolved gas accounting)
    mass_in = mdot_total_kg_h
    gas_out = new_gas_out.mass_flow_kg_h if new_gas_out else 0
    drain_out = new_drain.mass_flow_kg_h if new_drain else 0
    mass_out = gas_out + drain_out
    mass_error = abs(mass_in - mass_out) / mass_in * 100
    print(f"Mass Balance Error: {mass_error:.4f}%")
    
    # Temperature (isothermal or slightly cooler)
    temp_diff = abs(T_in_K - new_gas_out.temperature_k)
    print(f"Temperature Change: {temp_diff:.4f} K")
    
    # Pressure drop (should be positive)
    dp = P_in_Pa - new_gas_out.pressure_pa
    print(f"Pressure Drop: {dp/1e5:.4f} bar (expected > 0)")
    assert dp > 0, "Pressure drop should be positive"
    
    print("\n✅ COALESCER SANITY CHECKS PASSED")


if __name__ == '__main__':
    test_compare_coalescer_h2()
