#!/usr/bin/env python3
"""Diagnose mole fraction calculation in DryCooler outlet."""

import sys
sys.path.insert(0, '.')

from h2_plant.core.stream import Stream

def test_mole_frac_calculation():
    """Test get_total_mole_frac with different scenarios."""
    
    print("=" * 60)
    print("TEST 1: Stream with O2 impurity, NO water")
    print("=" * 60)
    
    s1 = Stream(
        mass_flow_kg_h=100.0,
        temperature_k=300.0,
        pressure_pa=30e5,
        composition={'H2': 0.999, 'O2': 0.001}
    )
    
    o2_ppm_1 = s1.get_total_mole_frac('O2') * 1e6
    print(f"  Composition: {s1.composition}")
    print(f"  Extra: {s1.extra}")
    print(f"  O2 ppm (molar): {o2_ppm_1:.2f}")
    
    # Manual calculation
    m_h2 = 0.999 * 100  # kg/h
    m_o2 = 0.001 * 100  # kg/h
    n_h2 = m_h2 / 2.016
    n_o2 = m_o2 / 32.0
    n_total = n_h2 + n_o2
    y_o2_manual = n_o2 / n_total
    print(f"  Manual calculation: {y_o2_manual * 1e6:.2f} ppm")
    print()
    
    print("=" * 60)
    print("TEST 2: Stream with O2 impurity + extra liquid water")
    print("=" * 60)
    
    # 0.01 kg/s = 36 kg/h of liquid water
    s2 = Stream(
        mass_flow_kg_h=100.0,
        temperature_k=300.0,
        pressure_pa=30e5,
        composition={'H2': 0.999, 'O2': 0.001},
        extra={'m_dot_H2O_liq_accomp_kg_s': 0.01}
    )
    
    o2_ppm_2 = s2.get_total_mole_frac('O2') * 1e6
    print(f"  Composition: {s2.composition}")
    print(f"  Extra: {s2.extra}")
    print(f"  O2 ppm (molar): {o2_ppm_2:.2f}")
    
    # Manual calculation with water
    m_h2o_extra = 0.01 * 3600  # 36 kg/h
    n_h2o = m_h2o_extra / 18.015
    n_total_with_water = n_h2 + n_o2 + n_h2o
    y_o2_with_water = n_o2 / n_total_with_water
    print(f"  Manual calculation (with water): {y_o2_with_water * 1e6:.2f} ppm")
    print()
    
    print("=" * 60)
    print("TEST 3: Stream with BOTH H2O_liq in composition AND extra")
    print("=" * 60)
    
    s3 = Stream(
        mass_flow_kg_h=100.0,
        temperature_k=300.0,
        pressure_pa=30e5,
        composition={'H2': 0.949, 'O2': 0.001, 'H2O_liq': 0.05},  # 5% liquid water in comp
        extra={'m_dot_H2O_liq_accomp_kg_s': 0.01}  # ADDITIONAL extra
    )
    
    o2_ppm_3 = s3.get_total_mole_frac('O2') * 1e6
    print(f"  Composition: {s3.composition}")
    print(f"  Extra: {s3.extra}")
    print(f"  O2 ppm (molar): {o2_ppm_3:.2f}")
    
    # What SHOULD happen: both sources of water should be included
    m_h2_3 = 0.949 * 100
    m_o2_3 = 0.001 * 100
    m_h2o_comp = 0.05 * 100  # 5 kg/h from composition
    m_h2o_extra = 0.01 * 3600  # 36 kg/h from extra
    m_h2o_total_should_be = m_h2o_comp + m_h2o_extra
    
    n_h2_3 = m_h2_3 / 2.016
    n_o2_3 = m_o2_3 / 32.0
    n_h2o_should_be = m_h2o_total_should_be / 18.015
    n_total_should_be = n_h2_3 + n_o2_3 + n_h2o_should_be
    y_o2_should_be = n_o2_3 / n_total_should_be
    print(f"  Manual: SHOULD be {y_o2_should_be * 1e6:.2f} ppm (both water sources)")
    
    # What the CURRENT code does: ignores extra if composition has H2O_liq
    n_h2o_current = m_h2o_comp / 18.015  # Only composition, NOT extra
    n_total_current = n_h2_3 + n_o2_3 + n_h2o_current
    y_o2_current = n_o2_3 / n_total_current
    print(f"  Current code: gives {y_o2_current * 1e6:.2f} ppm (ignores extra)")
    print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Test 1 (no water): {o2_ppm_1:.2f} ppm - OK")
    print(f"Test 2 (extra only): {o2_ppm_2:.2f} ppm - OK if matches {y_o2_with_water * 1e6:.2f}")
    print(f"Test 3 (both): {o2_ppm_3:.2f} ppm - BUG if ignores extra ({y_o2_current * 1e6:.2f})")

if __name__ == "__main__":
    test_mole_frac_calculation()
