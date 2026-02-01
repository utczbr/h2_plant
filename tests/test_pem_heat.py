import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.component import Component
from h2_plant.components.production.pem_electrolyzer_detailed import DetailedPEMElectrolyzer
from h2_plant.core.component_ids import ComponentID

class MockCoordinator(Component):
    def __init__(self):
        super().__init__()
        self.pem_setpoint_mw = 5.0 # Full power
    def initialize(self, dt, registry):
        super().initialize(dt, registry)
    def step(self, t):
        pass
    def get_state(self):
        return {}

class MockLUT(Component):
    def initialize(self, dt, registry):
        super().initialize(dt, registry)
    def step(self, t):
        pass
    def get_state(self):
        return {}
    def get_efficiency(self, power):
        return 0.65

def test_pem_heat():
    print("Testing PEM Heat Model...")
    
    registry = ComponentRegistry()
    
    # Register mocks
    coord = MockCoordinator()
    registry.register(ComponentID.DUAL_PATH_COORDINATOR, coord)
    
    lut = MockLUT()
    registry.register(ComponentID.LUT_MANAGER, lut)
    
    # Register PEM
    pem = DetailedPEMElectrolyzer(max_power_mw=5.0)
    registry.register(ComponentID.PEM_ELECTROLYZER_DETAILED, pem)
    
    # Initialize
    registry.initialize_all(dt=1.0)
    
    # Run step
    registry.step_all(0.0)
    
    # Check heat output
    heat_kw = pem.heat_output_kw
    print(f"PEM Power Consumed: {pem.P_consumed_W/1e6:.4f} MW")
    print(f"PEM Heat Output: {heat_kw:.4f} kW")
    
    if heat_kw <= 0:
        print("❌ Heat output should be positive!")
        sys.exit(1)
        
    # Check get_output
    heat_out = pem.get_output('heat_out')
    print(f"get_output('heat_out'): {heat_out:.4f} kW")
    
    if abs(heat_out - heat_kw) > 1e-6:
        print("❌ get_output('heat_out') does not match heat_output_kw!")
        sys.exit(1)
        
    # Rough physics check
    # 5 MW input. Efficiency ~65-70% (LHV).
    # Heat should be roughly 30% of 5 MW = 1.5 MW = 1500 kW.
    # Let's see what the model says.
    if heat_kw < 500 or heat_kw > 2500:
        print(f"⚠️ Heat output {heat_kw} kW seems out of expected range (500-2500 kW for 5MW input)")
    else:
        print("✅ Heat output in reasonable range")

    print("✅ PEM Heat Model Verified")

if __name__ == "__main__":
    test_pem_heat()
