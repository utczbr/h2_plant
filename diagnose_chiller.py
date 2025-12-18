#!/usr/bin/env python3
"""Diagnose Chiller water double-counting."""

import sys
sys.path.insert(0, '.')

from h2_plant.core.stream import Stream
from h2_plant.components.thermal.chiller import Chiller

def test_chiller_water():
    """Test if Chiller double-counts water."""
    
    print("=" * 70)
    print("Simulating DryCooler_1 -> Chiller_1 transition")
    print("=" * 70)
    
    # Create stream similar to DryCooler_1 outlet
    # From table: H2O = 6030 ppm, 5.6577 kg/h, 17.4% liquid
    # Total mass flow ~110 kg/h (H2 + H2O + O2)
    
    m_dot_total = 110.34  # kg/h approx
    m_h2 = 104.348
    m_h2o = 5.6577  # Total water
    m_o2 = 0.33297
    
    # Composition by mass fraction
    x_h2 = m_h2 / m_dot_total
    x_h2o_vap = m_h2o * 0.826 / m_dot_total  # 82.6% vapor
    x_h2o_liq = m_h2o * 0.174 / m_dot_total  # 17.4% liquid
    x_o2 = m_o2 / m_dot_total
    
    # DryCooler output: liquid in 'extra', NOT in composition
    inlet = Stream(
        mass_flow_kg_h=m_dot_total,
        temperature_k=305.45,  # 32.3°C
        pressure_pa=39.9e5,
        composition={'H2': x_h2, 'H2O': x_h2o_vap + x_h2o_liq, 'O2': x_o2},
        extra={'m_dot_H2O_liq_accomp_kg_s': (m_h2o * 0.174) / 3600.0}
    )
    
    print(f"\nInlet (DryCooler_1 outlet):")
    print(f"  Mass Flow: {inlet.mass_flow_kg_h:.2f} kg/h")
    print(f"  Composition: H2={inlet.composition.get('H2', 0):.4f}, H2O={inlet.composition.get('H2O', 0):.6f}, O2={inlet.composition.get('O2', 0):.6f}")
    print(f"  Comp H2O_liq: {inlet.composition.get('H2O_liq', 0):.6f}")
    print(f"  Extra liq: {inlet.extra.get('m_dot_H2O_liq_accomp_kg_s', 0) * 3600:.4f} kg/h")
    
    # Calculate total water (should use get_total_mole_frac logic)
    m_h2o_vapor = inlet.composition.get('H2O', 0) * inlet.mass_flow_kg_h
    m_h2o_liq_comp = inlet.composition.get('H2O_liq', 0) * inlet.mass_flow_kg_h
    m_h2o_extra = inlet.extra.get('m_dot_H2O_liq_accomp_kg_s', 0) * 3600.0
    m_h2o_total_in = m_h2o_vapor + m_h2o_liq_comp + m_h2o_extra
    print(f"  Total H2O mass: {m_h2o_total_in:.4f} kg/h")
    print(f"  O2 ppm (inlet): {inlet.get_total_mole_frac('O2') * 1e6:.1f}")
    
    # Run through Chiller
    chiller = Chiller("Chiller_1", target_temp_k=277.15)  # 4°C
    chiller.initialize(dt=1/60, registry=None)
    chiller.receive_input('fluid_in', inlet)
    chiller.step(0.0)
    
    outlet = chiller.get_output('fluid_out')
    
    print(f"\nOutlet (Chiller_1 outlet):")
    print(f"  Mass Flow: {outlet.mass_flow_kg_h:.2f} kg/h")
    print(f"  Composition: H2={outlet.composition.get('H2', 0):.4f}, H2O={outlet.composition.get('H2O', 0):.6f}, O2={outlet.composition.get('O2', 0):.6f}")
    print(f"  Comp H2O_liq: {outlet.composition.get('H2O_liq', 0):.6f}")
    print(f"  Extra liq: {outlet.extra.get('m_dot_H2O_liq_accomp_kg_s', 0) * 3600:.4f} kg/h")
    
    # Calculate total water at outlet
    m_h2o_vapor_out = outlet.composition.get('H2O', 0) * outlet.mass_flow_kg_h
    m_h2o_liq_comp_out = outlet.composition.get('H2O_liq', 0) * outlet.mass_flow_kg_h
    m_h2o_extra_out = outlet.extra.get('m_dot_H2O_liq_accomp_kg_s', 0) * 3600.0
    m_h2o_total_out = m_h2o_vapor_out + m_h2o_liq_comp_out + m_h2o_extra_out
    
    print(f"\n  H2O breakdown:")
    print(f"    Vapor (comp['H2O']): {m_h2o_vapor_out:.4f} kg/h")
    print(f"    Liquid (comp['H2O_liq']): {m_h2o_liq_comp_out:.4f} kg/h")
    print(f"    Liquid (extra): {m_h2o_extra_out:.4f} kg/h")
    print(f"    ─────────────────────")
    print(f"    TOTAL: {m_h2o_total_out:.4f} kg/h")
    
    print(f"\n  O2 ppm (outlet): {outlet.get_total_mole_frac('O2') * 1e6:.1f}")
    
    # Check for double counting
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    if m_h2o_liq_comp_out > 0 and m_h2o_extra_out > 0:
        print(f"⚠️  DOUBLE COUNTING DETECTED!")
        print(f"    Liquid is in BOTH composition ({m_h2o_liq_comp_out:.4f}) AND extra ({m_h2o_extra_out:.4f})")
        print(f"    This causes get_total_mole_frac to count liquid water TWICE!")
        print(f"    Reported total: {m_h2o_total_out:.4f} kg/h")
        print(f"    Actual total should be: {m_h2o_total_in:.4f} kg/h (mass conserved)")
    else:
        print("✓ No double counting detected")

if __name__ == "__main__":
    test_chiller_water()
