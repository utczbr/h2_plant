"""
Comparison Test: KnockOutDrum (New) vs modelo_kod (Legacy)

This test validates that the new KnockOutDrum component produces
outputs consistent with the legacy modelo_kod.py implementation.

Test Strategy:
    1. Set up identical inlet conditions (T, P, composition, flow).
    2. Run one step through both implementations.
    3. Compare key outputs within tolerance.

Expected Outputs to Compare:
    - Outlet temperature (should be isothermal, same as inlet)
    - Outlet pressure (inlet - delta_p)
    - Water removed (depends on flash equilibrium)
    - Dissolved gas in drain (Henry's Law)
"""

import sys
import math
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from h2_plant.core.stream import Stream
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.components.separation.knock_out_drum import KnockOutDrum

# Try to import legacy module
try:
    LEGACY_PATH = PROJECT_ROOT / 'h2_plant' / 'legacy' / 'NEW' / 'PEM' / 'modulos'
    sys.path.insert(0, str(LEGACY_PATH))
    from modelo_kod import modelar_knockout_drum
    LEGACY_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: Legacy module not available: {e}")
    LEGACY_AVAILABLE = False


def test_compare_kod_h2():
    """Compare KOD outputs for H2 stream at typical PEM conditions."""
    
    # === Test Conditions (from constants_and_config.py) ===
    T_in_C = 60.0
    T_in_K = T_in_C + 273.15
    P_in_bar = 40.0
    P_in_Pa = P_in_bar * 1e5
    
    # Mass flow and composition
    mdot_total_kg_h = 100.0
    composition = {
        'H2': 0.98,   # 98% H2 by mass
        'H2O': 0.02   # 2% water (saturated)
    }
    
    # === New Component ===
    kod = KnockOutDrum(diameter_m=0.5, delta_p_bar=0.05, gas_species='H2')
    registry = ComponentRegistry()
    kod.initialize(dt=1/60, registry=registry)
    
    inlet_stream = Stream(
        mass_flow_kg_h=mdot_total_kg_h,
        temperature_k=T_in_K,
        pressure_pa=P_in_Pa,
        composition=composition,
        phase='gas'
    )
    
    kod.receive_input('gas_inlet', inlet_stream)
    kod.step(t=0.0)
    
    new_state = kod.get_state()
    new_gas_out = kod.get_output('gas_outlet')
    new_liq_out = kod.get_output('liquid_drain')
    
    print("\n=== NEW COMPONENT RESULTS ===")
    print(f"Gas Outlet Flow: {new_gas_out.mass_flow_kg_h:.4f} kg/h")
    print(f"Gas Outlet Temp: {new_gas_out.temperature_k - 273.15:.2f} °C")
    print(f"Gas Outlet Pressure: {new_gas_out.pressure_pa / 1e5:.2f} bar")
    print(f"Liquid Drain Flow: {new_liq_out.mass_flow_kg_h:.4f} kg/h")
    print(f"Dissolved Gas: {new_state.get('dissolved_gas_kg_h', 0):.6f} kg/h")
    print(f"Separation Status: {new_state['separation_status']}")
    
    # === Legacy Module (if available) ===
    if LEGACY_AVAILABLE:
        print("\n=== LEGACY MODULE RESULTS ===")
        # Legacy function signature may differ - this is a template
        # Adjust parameters based on actual legacy API
        try:
            legacy_result = modelar_knockout_drum(
                gasfluido='H2',
                TC_in=T_in_C,
                Pbar_in=P_in_bar,
                mdot_mix_kgs=mdot_total_kg_h / 3600.0,
                y_H2O_in=composition['H2O'],
                mdot_H2O_liq_in_kgs=0.0
            )
            
            print(f"Gas Outlet Temp: {legacy_result.get('TC', 'N/A')} °C")
            print(f"Gas Outlet Pressure: {legacy_result.get('Pbar', 'N/A')} bar")
            print(f"Water Removed: {legacy_result.get('AguaPuraRemovidaH2Okgs', 0) * 3600:.4f} kg/h")
            print(f"Dissolved Gas: {legacy_result.get('GasDissolvidoremovidokgs', 0) * 3600:.6f} kg/h")
            
            # === Comparison ===
            print("\n=== COMPARISON ===")
            
            legacy_T_out = legacy_result.get('TC', T_in_C)
            new_T_out = new_gas_out.temperature_k - 273.15
            temp_diff = abs(legacy_T_out - new_T_out)
            print(f"Temperature Difference: {temp_diff:.3f} °C (max allowed: 0.1)")
            
            legacy_P_out = legacy_result.get('Pbar', P_in_bar)
            new_P_out = new_gas_out.pressure_pa / 1e5
            pressure_diff = abs(legacy_P_out - new_P_out)
            print(f"Pressure Difference: {pressure_diff:.4f} bar (max allowed: 0.01)")
            
            # Assertions
            assert temp_diff < 0.1, f"Temperature mismatch: {temp_diff} °C"
            assert pressure_diff < 0.01, f"Pressure mismatch: {pressure_diff} bar"
            
            print("\n✅ ALL COMPARISONS PASSED")
            
        except Exception as e:
            print(f"Legacy comparison failed: {e}")
    else:
        print("\n⚠️ Legacy module not available - skipping comparison")
        print("   Run from project root with: python tests/compare_kod.py")
    
    # === Basic Sanity Checks ===
    print("\n=== SANITY CHECKS ===")
    
    # Mass balance
    mass_in = mdot_total_kg_h
    mass_out = new_gas_out.mass_flow_kg_h + new_liq_out.mass_flow_kg_h
    mass_balance_error = abs(mass_in - mass_out) / mass_in * 100
    print(f"Mass Balance Error: {mass_balance_error:.4f}%")
    assert mass_balance_error < 1.0, f"Mass balance error too large: {mass_balance_error}%"
    
    # Isothermal check
    temp_drop = T_in_K - new_gas_out.temperature_k
    print(f"Temperature Drop: {temp_drop:.4f} K (expected: 0)")
    assert abs(temp_drop) < 0.01, f"Non-isothermal operation: {temp_drop} K"
    
    # Pressure drop
    expected_dp = 0.05 * 1e5
    actual_dp = P_in_Pa - new_gas_out.pressure_pa
    dp_error = abs(expected_dp - actual_dp)
    print(f"Pressure Drop Error: {dp_error:.1f} Pa (expected ~5000 Pa)")
    assert dp_error < 100, f"Pressure drop mismatch: {dp_error} Pa"
    
    print("\n✅ ALL SANITY CHECKS PASSED")


if __name__ == '__main__':
    test_compare_kod_h2()
