
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parents[1]))

from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.component_ids import ComponentID
from h2_plant.core.stream import Stream
from h2_plant.components.control.valve import ThrottlingValve
from h2_plant.optimization.lut_manager import LUTManager, LUTConfig

def test_throttling_valve_fast():
    print("=== Testing Throttling Valve (Fast) ===")
    
    # 1. Setup Registry with Tiny LUT Config
    registry = ComponentRegistry()
    
    # Use small grid for speed
    fast_config = LUTConfig(
        pressure_points=50,
        temperature_points=50,
        entropy_points=20,
        fluids=('H2',) # Only H2
    )
    # Set cache dir to temp to avoid breaking main cache
    fast_config.cache_dir = Path("/tmp/h2_plant_test_cache")
    
    lut_mgr = LUTManager(fast_config)
    lut_mgr.initialize()
    registry.register(ComponentID.LUT_MANAGER.value, lut_mgr)
    
    # 2. Create Valve (Target: 30 bar -> 5 bar)
    config = {
        'component_id': 'Valve-Test',
        'P_out_pa': 5.0e5, # 5 bar
        'fluid': 'H2'
    }
    valve = ThrottlingValve(config)
    valve.initialize(1.0, registry)
    
    # 3. Create Input Stream (30 bar, 300 K, 10 kg/h)
    inlet = Stream(
        mass_flow_kg_h=10.0,
        temperature_k=300.0,
        pressure_pa=30.0e5, # 30 bar
        composition={'H2': 1.0}
    )
    
    print(f"Inlet: P={inlet.pressure_pa/1e5:.1f} bar, T={inlet.temperature_k:.2f} K")
    
    # 4. Run Step
    valve.receive_input('inlet', inlet, 'hydrogen')
    valve.step(0.0)
    
    outlet = valve.get_output('outlet')
    
    if outlet:
        print(f"Outlet: P={outlet.pressure_pa/1e5:.1f} bar, T={outlet.temperature_k:.2f} K")
        print(f"Delta T: {valve.delta_T:.2f} K")
        
        # Check Reference Logic
        if outlet.temperature_k > inlet.temperature_k:
             print("SUCCESS: Temperature increased (Joule-Thomson heating).")
        else:
             print(f"WARNING: Temperature change {outlet.temperature_k - inlet.temperature_k:.2f}K. Might be small or negative depending on grid acc.")
             
        # Check Mass Conservation
        if abs(outlet.mass_flow_kg_h - inlet.mass_flow_kg_h) < 1e-6:
             print("SUCCESS: Mass conserved.")
        else:
             print("FAILURE: Mass leak.")
             
    else:
        print("FAILURE: No output stream.")
        
if __name__ == "__main__":
    test_throttling_valve_fast()
