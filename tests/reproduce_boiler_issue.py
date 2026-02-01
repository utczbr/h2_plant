
import sys
import os
sys.path.append(os.getcwd())

from h2_plant.components.thermal.electric_boiler import ElectricBoiler
from h2_plant.core.stream import Stream
from h2_plant.optimization.lut_manager import LUTManager
from h2_plant.core.component_registry import ComponentRegistry

def verify_boiler_steam():
    print("--- Verifying ElectricBoiler Steam Generation ---")
    
    # 1. Setup Boiler
    # Target 152 C, 5 bar.
    config = {
        'max_power_kw': 1000.0,
        'efficiency': 1.0, # Simplify
        'design_pressure_bar': 10.0,
        'target_temp_c': 152.0
    }
    boiler = ElectricBoiler(config)
    
    # Needs LUT Manager for accurate props
    registry = ComponentRegistry()
    lut = LUTManager()
    lut.initialize()
    registry.register('lut_manager', lut)
    
    boiler.initialize(dt=1/60, registry=registry)
    
    # 2. Input Stream: Liquid Water, 95 C, 5 bar, 1200 kg/h
    # (From Interchanger)
    s_in = Stream(
        mass_flow_kg_h=1200.0,
        temperature_k=95.0 + 273.15,
        pressure_pa=500000.0, # 5 bar
        composition={'H2O': 1.0},
        phase='liquid'
    )
    
    print(f"Input: {s_in.temperature_k-273.15:.2f} C, {s_in.pressure_pa/1e5:.1f} bar, Phase: {s_in.phase}")
    
    # 3. Step
    boiler.receive_input("fluid_in", s_in)
    boiler.step(t=0.0)
    
    # 4. Check Output
    state = boiler.get_state()
    t_out_c = state['outlet_temp_c']
    p_out_bar = s_in.pressure_pa / 1e5
    phase = state['phase']
    vap_frac = state['vapor_fraction']
    power_kw = state['power_input_kw']
    
    print(f"Output Temp: {t_out_c:.2f} C")
    print(f"Output Phase: {phase} (Vapor Frac: {vap_frac:.4f})")
    print(f"Power Input: {power_kw:.2f} kW")
    
    # Validation
    # Saturation at 5 bar is ~151.8 C.
    # We expect Steam (Gas or Mixed if target matches sat)
    # If using CP*dT, power will be low (~ sensible heat only)
    
    # Sensible heat estimate: 1200/3600 * 4.18 * (152-95) * 1000 ~ 79 kW
    # Latent heat estimate: 1200/3600 * 2100 ~ 700 kW
    # Total needed ~ 780 kW
    
    if power_kw < 200.0:
        print("FAIL: Power input too low (ignoring latent heat).")
    elif t_out_c < 150.0:
        print(f"FAIL: Temp too low ({t_out_c:.2f} C).")
    else:
        print("PASS: Boiler functioning correctly.")

if __name__ == "__main__":
    verify_boiler_steam()
