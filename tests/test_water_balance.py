import sys
import os
import unittest
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.component_ids import ComponentID
from h2_plant.core.component import Component
from h2_plant.components.water.water_balance_tracker import WaterBalanceTracker

class MockPEM(Component):
    def __init__(self):
        super().__init__()
        self.water_input_kg_h = 100.0
    def initialize(self, dt, registry): super().initialize(dt, registry)
    def step(self, t): pass
    def get_state(self): return {}

class MockSeparator:
    def __init__(self):
        self.water_return_kg_h = 20.0

class MockSOEC(Component):
    def __init__(self):
        super().__init__()
        self.water_input_kg_h = 50.0
        self.separator_sp3 = MockSeparator()
    def initialize(self, dt, registry): super().initialize(dt, registry)
    def step(self, t): pass
    def get_state(self): return {}

class TestWaterBalance(unittest.TestCase):
    def setUp(self):
        self.registry = ComponentRegistry()
        self.tracker = WaterBalanceTracker()
        self.registry.register(ComponentID.WATER_BALANCE_TRACKER, self.tracker)
        
    def test_water_tracking(self):
        # Register Mocks
        pem = MockPEM()
        self.registry.register(ComponentID.PEM_ELECTROLYZER_DETAILED, pem)
        
        soec = MockSOEC()
        self.registry.register(ComponentID.SOEC_CLUSTER, soec)
        
        # Initialize
        self.registry.initialize_all(dt=1.0)
        
        # Step
        self.tracker.step(0.0)
        
        # Verify
        print(f"Total Consumption: {self.tracker.total_consumption_kg_h} kg/h")
        print(f"Total Recovery: {self.tracker.total_recovery_kg_h} kg/h")
        print(f"Net Demand: {self.tracker.net_demand_kg_h} kg/h")
        
        # PEM (100) + SOEC (50) = 150
        self.assertEqual(self.tracker.total_consumption_kg_h, 150.0)
        
        # SOEC Recovery (20)
        self.assertEqual(self.tracker.total_recovery_kg_h, 20.0)
        
        # Net = 130
        self.assertEqual(self.tracker.net_demand_kg_h, 130.0)

if __name__ == '__main__':
    unittest.main()
