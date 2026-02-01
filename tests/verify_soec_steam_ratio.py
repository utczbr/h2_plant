
import sys
import os
sys.path.append(os.getcwd())

from h2_plant.components.electrolysis.soec_operator import SOECOperator
from h2_plant.core.stream import Stream

def verify_soec_steam_ratio():
    print("--- Verifying SOEC Steam Ratio Configuration ---")
    
    # 1. Config with 10.5 ratio (Legacy = 9.0)
    config = {
        'num_modules': 1,
        'max_power_nominal_mw': 2.4,
        'steam_input_ratio_kg_per_kg_h2': 10.5,
        'out_pressure_pa': 100000.0
    }
    
    soec = SOECOperator(config)
    print(f"Configured Steam Ratio: {soec.steam_input_ratio}")
    
    # 2. Check internal ratio
    if abs(soec.steam_input_ratio - 10.5) < 0.01:
        print("PASS: SOEC correctly read the steam ratio from config.")
    else:
        print(f"FAIL: SOEC ratio is {soec.steam_input_ratio}, expected 10.5.")
        return

    # Initialize component
    from h2_plant.core.component_registry import ComponentRegistry
    registry = ComponentRegistry()
    soec.initialize(dt=1/60, registry=registry)

    # 3. Simulate production
    # Provide enough power for 100 kg H2 (approx 3.7 MWh/100kg -> ~37 kWh/kg)
    # 1 hr step. Power = 3.7 MW
    soec.dt = 1.0
    
    h2_target = 100.0
    energy_needed_mwh = h2_target * (soec.current_efficiency_kwh_kg / 1000.0) 
    power_mw = energy_needed_mwh # since dt=1
    
    # Needs water input
    # Ratio = 10.5. Input needed = 1050 kg.
    s_in = Stream(mass_flow_kg_h=2000.0)
    soec.receive_input("steam_in", s_in)
    soec.receive_input("power_in", power_mw)
    
    soec.step(t=0.0)
    
    # 4. Check Output
    h2_out = soec.get_output('h2_out')
    steam_out = soec.get_output('steam_out')
    
    print(f"H2 Produced: {h2_out.mass_flow_kg_h:.2f} kg/h")
    print(f"Steam Out: {steam_out.mass_flow_kg_h:.2f} kg/h")
    
    # Expected steam out:
    # Consumed = 100 * 9.0 = 900 kg
    # Input consumed from stream logic: 100 * 10.5 = 1050 kg (limit check in step)
    # Excess = 1050 - 900 = 150 kg
    
    expected_excess = h2_target * (10.5 - 9.0)
    
    if abs(steam_out.mass_flow_kg_h - expected_excess) < 1.0:
        print(f"PASS: Steam output ({steam_out.mass_flow_kg_h:.2f}) matches expected excess.")
    else:
         print(f"FAIL: Steam output mismatch. Expected {expected_excess}, got {steam_out.mass_flow_kg_h}")

if __name__ == "__main__":
    verify_soec_steam_ratio()
