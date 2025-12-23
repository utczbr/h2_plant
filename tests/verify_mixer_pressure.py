
import sys
import os
sys.path.append(os.getcwd())

from h2_plant.components.mixing.multicomponent_mixer import MultiComponentMixer
from h2_plant.core.stream import Stream
from h2_plant.core.component_registry import ComponentRegistry

def verify_mixer_pressure():
    print("--- Verifying MultiComponentMixer Pressure Logic ---")
    
    # 1. Setup Mixer in Continuous Flow Mode (Pipe Junction)
    mixer = MultiComponentMixer(volume_m3=5.0, continuous_flow=True)
    registry = ComponentRegistry()
    mixer.initialize(dt=1/60, registry=registry) # 1 minute timestep
    
    print(f"Mixer Mode: Continuous Flow")
    
    # 2. Create Input Streams (simulating SOEC output)
    # Stream 1: Hydrogen (Gas)
    # 384 kg/h H2 @ 1 bar, 152 C
    s1 = Stream(
        mass_flow_kg_h=384.0,
        temperature_k=152.0 + 273.15,
        pressure_pa=100000.0,
        composition={'H2': 1.0}
    )
    
    # Stream 2: Steam (Gas)
    # 1200 kg/h H2O @ 1 bar, 152 C
    s2 = Stream(
        mass_flow_kg_h=1200.0,
        temperature_k=152.0 + 273.15,
        pressure_pa=100000.0,
        composition={'H2O': 1.0}
    )
    
    print(f"Input Stream 1: {s1.pressure_pa/1e5:.2f} bar (H2)")
    print(f"Input Stream 2: {s2.pressure_pa/1e5:.2f} bar (H2O)")
    
    # 3. Feed Mixer
    mixer.receive_input("inlet_1", s1)
    mixer.receive_input("inlet_2", s2)
    
    # 4. Step
    mixer.step(t=0.0)
    
    # 5. Check Output
    state = mixer.get_state()
    p_out_bar = state['pressure_pa'] / 1e5
    t_out_c = state['temperature_k'] - 273.15
    
    print(f"Output Pressure: {p_out_bar:.4f} bar")
    print(f"Output Temperature: {t_out_c:.2f} C")
    
    # Validation
    if 0.95 <= p_out_bar <= 1.05:
        print("PASS: Output pressure is consistent with inputs (approx 1 bar).")
        # Also check if mass is conserved in output stream
        out_stream = mixer.get_output('outlet')
        total_in = 384.0 + 1200.0
        if abs(out_stream.mass_flow_kg_h - total_in) < 0.1:
             print(f"PASS: Mass balance conserved ({out_stream.mass_flow_kg_h:.1f} kg/h).")
        else:
             print(f"FAIL: Mass balance mismatch. Expected {total_in}, got {out_stream.mass_flow_kg_h}")
    else:
        print(f"FAIL: High pressure detected ({p_out_bar:.2f} bar). Logic fix failed.")
        sys.exit(1)

if __name__ == "__main__":
    verify_mixer_pressure()
