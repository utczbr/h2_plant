import numpy as np
import matplotlib.pyplot as plt
import soec_operator

def run_simulation(year, duration_min=120):
    print(f"\n--- Simulating Year {year} ---")
    
    # Initialize with Year
    state = soec_operator.initialize_soec_simulation(
        off_real_modules=[], 
        rotation_enabled=True, 
        use_optimal_limit=True, 
        year=year
    )
    
    print(f"  > Efficiency: {state.current_efficiency_kwh_kg:.2f} kWh/kg")
    print(f"  > Capacity Factor: {state.current_capacity_factor*100:.1f}%")
    print(f"  > Effective Module Max: {state.effective_max_module_power:.3f} MW")
    
    # Profile: Ramp to full capacity then ramp down
    # With 6 modules, Year 0 Max is ~11.52 MW.
    # We request 12 MW to force saturation.
    power_requests = [12.0] * duration_min
    
    results_power = []
    results_h2 = []
    
    for i in range(duration_min):
        p_ref = power_requests[i]
        # Pulse demand to 0 at minute 60 to force a cycle restart
        if 50 <= i < 70:
            p_ref = 0.0
            
        p_act, state, h2, steam = soec_operator.run_soec_step(p_ref, state)
        
        results_power.append(p_act)
        results_h2.append(h2)
        
    total_h2 = sum(results_h2)
    avg_power = np.mean(results_power)
    cycles = np.sum(state.cycle_counts)
    
    print(f"  > Avg Power: {avg_power:.3f} MW")
    print(f"  > Total H2: {total_h2:.3f} kg")
    print(f"  > Total Cycles Detected: {cycles}")
    
    return results_power, results_h2, state

# Run Comparison: Year 0 vs Year 7
pow_0, h2_0, state_0 = run_simulation(0.0)
pow_7, h2_7, state_7 = run_simulation(7.0)

# Validation Logic
print("\n--- COMPARATIVE RESULTS ---")
# 1. Capacity Check
max_pow_0 = max(pow_0)
max_pow_7 = max(pow_7)
expected_drop = 0.75 # Year 7 is 75% capacity
ratio = max_pow_7 / max_pow_0

print(f"1. Peak Power (Capacity Check):")
print(f"   Year 0: {max_pow_0:.2f} MW")
print(f"   Year 7: {max_pow_7:.2f} MW")
print(f"   Ratio: {ratio:.3f} (Expected ~0.75)")

if abs(ratio - 0.75) < 0.01:
    print("   ✅ SUCCESS: Capacity degraded correctly.")
else:
    print("   ❌ FAILURE: Capacity degradation incorrect.")

# 2. Efficiency Check
# H2 Rate Year 0 = 1000/37 = 27.02 kg/MWh
# H2 Rate Year 7 = 1000/42 = 23.81 kg/MWh
# We compare the production rate at full power intervals (e.g., minute 10)
rate_0 = h2_0[10] / (pow_0[10]/60) # kg / MWh
rate_7 = h2_7[10] / (pow_7[10]/60)

print(f"\n2. Production Efficiency Check (at saturation):")
print(f"   Year 0 Rate: {rate_0:.2f} kg/MWh")
print(f"   Year 7 Rate: {rate_7:.2f} kg/MWh")

expected_rate_7 = 1000 / 42.0
if abs(rate_7 - expected_rate_7) < 0.1:
    print("   ✅ SUCCESS: Efficiency degraded correctly.")
else:
    print("   ❌ FAILURE: Efficiency degradation incorrect.")

# 3. Cycle Check
# We forced a shutdown (min 50-70) and restart. Should see at least 1 cycle per module (initial start) + 1 cycle (restart) = 12 total? 
# Or just initial start if they start at Hot Standby.
# Initial: Hot Standby(1) -> Ramp Up(2) [Cycle 1]
# Minute 50: Ramp Down -> ... -> Hot Standby
# Minute 70: Hot Standby -> Ramp Up [Cycle 2]
print(f"\n3. Cycle Counting Check:")
print(f"   Cycles Year 0: {np.sum(state_0.cycle_counts)}")
if np.sum(state_0.cycle_counts) >= 6:
     print("   ✅ SUCCESS: Cycles detected.")
else:
     print("   ❌ FAILURE: No cycles detected.")

# Plot
plt.figure(figsize=(10,6))
plt.plot(pow_0, label='Year 0 Power (MW)')
plt.plot(pow_7, label='Year 7 Power (MW)')
plt.title("Degradation Impact: Capacity Fade (Input Power Limit)")
plt.ylabel("MW")
plt.xlabel("Minute")
plt.legend()
plt.grid(True)
plt.savefig('test_degradation_proof.png')
print("\nPlot saved to test_degradation_proof.png")