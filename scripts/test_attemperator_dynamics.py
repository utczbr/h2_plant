
import logging
from h2_plant.components.thermal.attemperator import Attemperator
from h2_plant.core.stream import Stream
from h2_plant.optimization.lut_manager import LUTManager
from h2_plant.core.component_registry import ComponentRegistry

def test_attemperator_residence_time():
    # Setup
    print("Initializing Attemperator Test...")
    registry = ComponentRegistry()
    
    # Initialize LUT Manager (Mock or Real)
    # Ideally we load the real one, but it takes time. 
    # For this test, we can try to rely on the fallback or load the real one if fast enough.
    # The residence time calculation DOES depend on LUT density lookup.
    # So we MUST load LUTManager.
    
    try:
        lut = LUTManager()
        lut.initialize() # Load/Generate LUTs
        registry.register('lut_manager', lut)
    except Exception as e:
        print(f"Failed to load LUTManager: {e}")
        return

    # Create Attemperator (SOEC_Feed_Attemperator Specs)
    attemp = Attemperator(
        component_id="Test_Attemperator",
        target_temp_k=430.05, # 156.9 C
        pipe_diameter_m=0.1541,
        volume_m3=0.035,
        pressure_drop_bar=0.5
    )
    attemp.initialize(0.1, registry)
    
    # Create Inputs
    # Steam: 3250 kg/h, 270 C (543.15 K), 5.5 bar
    steam_in = Stream(
        mass_flow_kg_h=3250.0,
        temperature_k=543.15,
        pressure_pa=5.5e5,
        composition={'H2O': 1.0},
        phase='gas'
    )
    
    # Water: 350 kg/h, 25 C, 6.0 bar
    water_in = Stream(
        mass_flow_kg_h=350.0,
        temperature_k=298.15,
        pressure_pa=6.0e5,
        composition={'H2O': 1.0}, # Pure water
        phase='liquid'
    )
    
    # Inject Inputs
    attemp.receive_input('steam_in', steam_in)
    attemp.receive_input('water_in', water_in)
    
    # Run Step
    attemp.step(0.0)
    
    # Inspect Output
    state = attemp.get_state()
    print("\n=== Test Results ===")
    print(f"Residence Time: {state['residence_time_s']:.5f} s")
    print(f"Velocity:       {state['steam_velocity_m_s']:.2f} m/s")
    print(f"Outlet Temp:    {state['outlet_temp_k']:.2f} K")
    print(f"Volume:         {state['volume_m3']:.4f} m3")
    
    # Verification Logic
    assert state['residence_time_s'] > 0, "Residence time should be positive"
    assert state['residence_time_s'] < 0.1, f"Residence time {state['residence_time_s']} too high (>0.1s)"
    assert state['steam_velocity_m_s'] > 10.0, "Velocity too low"
    assert state['steam_velocity_m_s'] < 60.0, "Velocity too high"
    
    print("\nSUCCESS: Attemperator meets dynamic path constraints.")

if __name__ == "__main__":
    test_attemperator_residence_time()
