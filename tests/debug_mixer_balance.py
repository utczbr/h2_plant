
import sys
import os
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from h2_plant.components.mixing.multicomponent_mixer import MultiComponentMixer
from h2_plant.core.stream import Stream
from h2_plant.core.component_registry import ComponentRegistry

# Setup logging
logging.basicConfig(level=logging.INFO)

def test_mixer_balance():
    print("Initializing Mixer Test...")
    registry = ComponentRegistry()
    mixer = MultiComponentMixer(volume_m3=5.0, continuous_flow=True)
    mixer.initialize(dt=1/60, registry=registry) # 1 minute steps

    # Define Inputs (Approximating User Scenario)
    # Stream 1: PEM (270 kg/h, 6 ppm H2O mol)
    # y = 6e-6. MW_H2O=18, MW_H2=2. w = y * (18/2) = 54e-6
    s1_comp = {'H2': 1.0 - 54e-6, 'H2O': 54e-6}
    s1 = Stream(mass_flow_kg_h=270.0, temperature_k=300.0, pressure_pa=30e5, composition=s1_comp)

    # Stream 2: SOEC (90 kg/h, 6 ppm H2O mol)
    s2_comp = {'H2': 1.0 - 54e-6, 'H2O': 54e-6}
    s2 = Stream(mass_flow_kg_h=90.0, temperature_k=300.0, pressure_pa=30e5, composition=s2_comp)

    # Stream 3: ATR (204 kg/h, 2 ppm CH4 mol, 0 ppm H2O)
    # 2 ppm mol CH4. MW=16. H2=2. ratio 8. w = 16e-6.
    s3_comp = {'H2': 1.0 - 16e-6, 'CH4': 16e-6}
    s3 = Stream(mass_flow_kg_h=204.0, temperature_k=300.0, pressure_pa=30e5, composition=s3_comp)

    print(f"Stream 1 (PEM): {s1.mass_flow_kg_h} kg/h, H2O={s1.get_total_mole_frac('H2O')*1e6:.2f} ppm")
    print(f"Stream 2 (SOEC): {s2.mass_flow_kg_h} kg/h, H2O={s2.get_total_mole_frac('H2O')*1e6:.2f} ppm")
    print(f"Stream 3 (ATR): {s3.mass_flow_kg_h} kg/h, H2O={s3.get_total_mole_frac('H2O')*1e6:.2f} ppm")

    # Push to Mixer
    mixer.receive_input('inlet_1', s1)
    mixer.receive_input('inlet_2', s2)
    mixer.receive_input('inlet_3', s3)

    # Execute Step
    mixer.step(t=1.0)

    # Check Output
    out = mixer.get_output('outlet')
    h2o_ppm = out.get_total_mole_frac('H2O') * 1e6
    mass_flow = out.mass_flow_kg_h
    
    print("-" * 30)
    print(f"Mixer Output: {mass_flow:.2f} kg/h (Expected ~564.0)")
    print(f"Mixer H2O: {h2o_ppm:.4f} ppm (Expected ~3.8-4.0)")
    
    expected_ppm = (270*54e-6 + 90*54e-6 + 0) / 564.0 / 18 * 2 * 1e6 # Rough calc
    # Validating mass balance
    if abs(mass_flow - 564.0) > 1.0:
        print("FAIL: Mass Balance Warning!")
    else:
        print("PASS: Mass Balance OK")
        
    if h2o_ppm > 8.0:
        print("FAIL: Impurity Explosion Detected!")
    elif h2o_ppm < 2.0:
        print("FAIL: Impurity Loss Detected!")
    else:
        print(f"PASS: Impurity Conserved ({h2o_ppm:.2f} ppm)")

if __name__ == "__main__":
    test_mixer_balance()
