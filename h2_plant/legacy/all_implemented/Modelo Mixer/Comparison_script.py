import sys
import os

# Ensure we can import the uploaded Mixer.py
# (Assuming Mixer.py is in the same directory)
try:
    from Mixer import mixer_model, get_example_data
except ImportError:
    print("Error: Could not import 'Mixer.py'. Ensure it is in the same directory.")
    sys.exit(1)

from Refactored_Mixer import Mixer
from Framework import Stream, ComponentRegistry

def run_comparison():
    print("=========================================================")
    print("      MIXER REFACTOR VALIDATION: ARCHITECTURE VS LEGACY")
    print("=========================================================")

    # 1. Get Data from Legacy System
    # This ensures we are using the exact same inputs
    legacy_streams, legacy_p_out_kpa = get_example_data()
    
    print("\n[INPUTS]")
    for i, s in enumerate(legacy_streams):
        print(f"  Stream {i+1}: {s['m_dot']} kg/s, {s['T']} °C, {s['P']} kPa")
    print(f"  Target P_out: {legacy_p_out_kpa} kPa")

    # -------------------------------------------------------
    # 2. RUN LEGACY SYSTEM
    # -------------------------------------------------------
    print("\n>>> Running Legacy Mixer.py...")
    _, legacy_results = mixer_model(legacy_streams, legacy_p_out_kpa)
    
    legacy_m_out = legacy_results['Output Mass Flow Rate (kg/s)']
    legacy_h_out = legacy_results['Output Specific Enthalpy (kJ/kg)']
    legacy_t_out = legacy_results['Output Temperature (°C)']

    # -------------------------------------------------------
    # 3. RUN NEW ARCHITECTURE SYSTEM
    # -------------------------------------------------------
    print("\n>>> Running New Refactored Mixer...")
    
    # Initialize Component
    new_mixer = Mixer(
        mixer_id="ValidationMixer", 
        fluid_type="Water", 
        outlet_pressure_kpa=legacy_p_out_kpa
    )
    registry = ComponentRegistry()
    new_mixer.initialize(dt=1.0, registry=registry)
    
    # Convert Legacy Dict Inputs -> New Architecture Stream Objects
    # Note: Stream expects (kg/h, K, Pa), Legacy has (kg/s, C, kPa)
    for i, s in enumerate(legacy_streams):
        stream_obj = Stream(
            mass_flow_kg_h=s['m_dot'] * 3600.0,   # kg/s -> kg/h
            temperature_k=s['T'] + 273.15,        # C -> K
            pressure_pa=s['P'] * 1000.0,          # kPa -> Pa
            composition={'H2O': 1.0},
            phase='liquid'
        )
        new_mixer.receive_input(f"stream_{i+1}", stream_obj, "water")
    
    # Execute Step
    new_mixer.step(t=0.0)
    
    # Get Results (using the internal state variables we added for parity checking)
    new_state = new_mixer.get_state()
    new_m_out = new_state['calc_mass_flow_kg_s']
    new_h_out = new_state['calc_enthalpy_kj_kg']
    new_t_out = new_state['calc_temp_c']

    # -------------------------------------------------------
    # 4. COMPARE RESULTS
    # -------------------------------------------------------
    print("\n[COMPARISON RESULTS]")
    print(f"{'Metric':<20} | {'Legacy Value':<15} | {'New Arch Value':<15} | {'Difference':<15}")
    print("-" * 75)
    
    diff_m = abs(legacy_m_out - new_m_out)
    diff_h = abs(legacy_h_out - new_h_out)
    diff_t = abs(legacy_t_out - new_t_out)
    
    print(f"{'Mass Flow (kg/s)':<20} | {legacy_m_out:<15.5f} | {new_m_out:<15.5f} | {diff_m:<15.5e}")
    print(f"{'Enthalpy (kJ/kg)':<20} | {legacy_h_out:<15.5f} | {new_h_out:<15.5f} | {diff_h:<15.5e}")
    print(f"{'Temperature (°C)':<20} | {legacy_t_out:<15.5f} | {new_t_out:<15.5f} | {diff_t:<15.5e}")
    
    print("-" * 75)
    if diff_m < 1e-9 and diff_h < 1e-9 and diff_t < 1e-9:
        print("✅ SUCCESS: Results match exactly!")
    else:
        print("❌ FAILURE: Discrepancies detected.")

if __name__ == "__main__":
    run_comparison()