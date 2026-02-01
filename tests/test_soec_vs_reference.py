"""
Quick validation: Compare new soec_operation.py implementation with reference soec_operator.py
"""
import sys
import numpy as np
from pathlib import Path

# Add legacy path
legacy_path = Path(__file__).parent / 'h2_plant' / 'legacy' / 'pem_soec_reference' / 'soec_to_implement'
sys.path.insert(0, str(legacy_path))

# Import reference
import soec_operator as ref

# Import new implementation
from h2_plant.models import soec_operation as new

print("=== SOEC Implementation Comparison ===\n")

# Test degradation interpolators
print("1. Degradation Interpolators:")
for year in [0, 3, 5, 7]:
    ref_eff = ref.efficiency_interpolator(year)
    new_eff = new.efficiency_interpolator(year)
    ref_cap = ref.capacity_interpolator(year)
    new_cap = new.capacity_interpolator(year)
    
    print(f"  Year {year}: Eff ref={ref_eff:.2f} new={new_eff:.2f}, Cap ref={ref_cap:.1f}% new={new_cap:.1f}%")
    assert abs(ref_eff - new_eff) < 1e-9, f"Efficiency mismatch at year {year}"
    assert abs(ref_cap - new_cap) < 1e-9, f"Capacity mismatch at year {year}"

print("  ✅ Interpolators match\n")

# Test state initialization
print("2. State Initialization:")
for year in [0.0, 7.0]:
    ref_state = ref.initialize_soec_simulation(year=year)
    new_state = new.initialize_soec_simulation(year=year)
    
    print(f"  Year {year}:")
    print(f"    Eff: ref={ref_state.current_efficiency_kwh_kg:.4f} new={new_state.current_efficiency_kwh_kg:.4f}")
    print(f"    Cap: ref={ref_state.current_capacity_factor:.4f} new={new_state.current_capacity_factor:.4f}")
    print(f"    Max: ref={ref_state.effective_max_module_power:.4f} new={new_state.effective_max_module_power:.4f}")
    
    assert abs(ref_state.current_efficiency_kwh_kg - new_state.current_efficiency_kwh_kg) < 1e-9
    assert abs(ref_state.current_capacity_factor - new_state.current_capacity_factor) < 1e-9
    assert abs(ref_state.effective_max_module_power - new_state.effective_max_module_power) < 1e-9

print("  ✅ Initialization matches\n")

# Test simulation step by step
print("3. Step-by-Step Simulation (100 steps, Year 0):")
ref_state = ref.initialize_soec_simulation(year=0.0)
new_state = new.initialize_soec_simulation(year=0.0)

for t in range(100):
    # Vary power demand
    if t < 20:
        p_ref = (t/20) * 10.0
    elif t < 60:
        p_ref = 10.0
    elif t < 80:
        p_ref = 2.0
    else:
        p_ref = 0.0
    
    # Run both
    ref_power, ref_state, ref_h2, ref_steam = ref.run_soec_step(p_ref, ref_state)
    new_power, new_state, new_h2, new_steam = new.run_soec_step(p_ref, new_state)
    
    # Compare
    if abs(ref_power - new_power) > 1e-6:
        print(f"  ❌ FAILED at step {t}: Power mismatch ref={ref_power:.6f} new={new_power:.6f}")
        sys.exit(1)
    
    if abs(ref_h2 - new_h2) > 1e-6:
        print(f"  ❌ FAILED at step {t}: H2 mismatch ref={ref_h2:.6f} new={new_h2:.6f}")
        sys.exit(1)
    
    if abs(ref_steam - new_steam) > 1e-6:
        print(f"  ❌ FAILED at step {t}: Steam mismatch ref={ref_steam:.6f} new={new_steam:.6f}")
        sys.exit(1)
    
    if not np.allclose(ref_state.real_powers, new_state.real_powers, atol=1e-6):
        print(f"  ❌ FAILED at step {t}: Module powers mismatch")
        sys.exit(1)
    
    if not np.array_equal(ref_state.real_states, new_state.real_states):
        print(f"  ❌ FAILED at step {t}: Module states mismatch")
        sys.exit(1)

# Check accumulators
print(f"  Final totals:")
print(f"    H2: ref={ref_state.total_h2_produced:.3f} new={new_state.total_h2_produced:.3f} kg")
print(f"    Steam: ref={ref_state.total_steam_consumed:.3f} new={new_state.total_steam_consumed:.3f} kg")
print(f"    Cycles: ref={np.sum(ref_state.cycle_counts)} new={np.sum(new_state.cycle_counts)}")

assert abs(ref_state.total_h2_produced - new_state.total_h2_produced) < 1e-3
assert abs(ref_state.total_steam_consumed - new_state.total_steam_consumed) < 1e-3
assert np.array_equal(ref_state.cycle_counts, new_state.cycle_counts)

print("  ✅ 100-step simulation matches perfectly\n")

print("=== ALL TESTS PASSED ===")
print("New soec_operation.py implementation is identical to reference soec_operator.py")
