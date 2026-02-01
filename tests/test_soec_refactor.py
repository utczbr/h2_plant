import sys
import os
import unittest
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.component_ids import ComponentID
from h2_plant.core.component import Component
from h2_plant.components.production.soec_electrolyzer_detailed import DetailedSOECElectrolyzer

class MockCoordinator(Component):
    def __init__(self):
        super().__init__()
        self.soec_setpoint_mw = 2.0 # 2 MW setpoint
    def initialize(self, dt, registry): super().initialize(dt, registry)
    def step(self, t): pass
    def get_state(self): return {}

class TestSOECRefactor(unittest.TestCase):
    def setUp(self):
        self.registry = ComponentRegistry()
        
    def test_soec_coordinator_integration(self):
        # Register Coordinator
        coord = MockCoordinator()
        self.registry.register(ComponentID.DUAL_PATH_COORDINATOR, coord)
        
        # Register SOEC
        soec = DetailedSOECElectrolyzer(max_power_kw=5000.0) # 5 MW max
        soec.water_input_kg_h = 1000.0 # Supply water
        self.registry.register(ComponentID.SOEC_CLUSTER, soec)
        
        # Initialize
        self.registry.initialize_all(dt=1.0)
        
        # Step
        self.registry.step_all(0.0)
        
        # Verify SOEC picked up the setpoint
        # 2 MW = 2000 kW
        print(f"Coordinator Setpoint: {coord.soec_setpoint_mw} MW")
        print(f"SOEC Rectifier Input: {soec.rectifier_rt2.ac_input_kw} kW")
        
        self.assertEqual(soec.rectifier_rt2.ac_input_kw, 2000.0)
        
        # Verify output generated
        print(f"SOEC H2 Output: {soec.h2_product_kg_h} kg/h")
        self.assertGreater(soec.h2_product_kg_h, 0.0)

if __name__ == '__main__':
    unittest.main()
