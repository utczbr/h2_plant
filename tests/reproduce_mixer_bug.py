
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath("/home/stuart/Documentos/Planta Hidrogenio"))

from h2_plant.components.mixing.multicomponent_mixer import MultiComponentMixer
from h2_plant.core.stream import Stream
from h2_plant.core.component_registry import ComponentRegistry

def test_mixer_enthalpy_bug():
    registry = ComponentRegistry()
    mixer = MultiComponentMixer(volume_m3=5.0)
    mixer.initialize(dt=1/60, registry=registry)
    
    # Create input stream at 800 C (1073.15 K)
    # H2 flow
    t_in = 1073.15
    p_in = 30e5 # 30 bar
    stream_in = Stream(
        mass_flow_kg_h=100.0,
        temperature_k=t_in,
        pressure_pa=p_in,
        composition={'H2': 1.0},
        phase='gas'
    )
    
    print(f"Input Temp: {t_in:.2f} K ({t_in-273.15:.2f} C)")
    
    # Step 1
    mixer.receive_input('inlet', stream_in)
    mixer.step(0.0)
    
    out_stream = mixer.get_output('outlet')
    t_out = out_stream.temperature_k
    
    print(f"Output Temp: {t_out:.2f} K ({t_out-273.15:.2f} C)")
    
    delta_t = t_out - t_in
    print(f"Delta T: {delta_t:.2f} K")
    
    if delta_t > 10.0:
        print("FAIL: Significant temperature rise detected without heat source. Flow work (PV) error confirmed.")
    else:
        print("PASS: Temperature conserved.")

if __name__ == "__main__":
    test_mixer_enthalpy_bug()
