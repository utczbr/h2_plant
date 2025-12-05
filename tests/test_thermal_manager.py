import sys
import os
import unittest
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.component_ids import ComponentID
from h2_plant.core.component import Component
from h2_plant.components.thermal.thermal_manager import ThermalManager

class MockPEM(Component):
    def __init__(self):
        super().__init__()
        self.heat_output_kw = 500.0
    def initialize(self, dt, registry): super().initialize(dt, registry)
    def step(self, t): pass
    def get_state(self): return {}

class MockSteamGen:
    def __init__(self):
        self.heat_input_kw = 200.0 # Old attr
        self.total_heat_demand_kw = 200.0 # New attr
        self.external_heat_input_kw = 0.0 # Input from TM

class MockSOEC(Component):
    def __init__(self):
        super().__init__()
        self.steam_gen_hx4 = MockSteamGen()
    def initialize(self, dt, registry): super().initialize(dt, registry)
    def step(self, t): pass
    def get_state(self): return {}

class TestThermalManager(unittest.TestCase):
    def setUp(self):
        self.registry = ComponentRegistry()
        self.tm = ThermalManager()
        self.registry.register(ComponentID.THERMAL_MANAGER, self.tm)
        
    def test_heat_collection_and_distribution(self):
        # Register Mocks
        pem = MockPEM()
        self.registry.register(ComponentID.PEM_ELECTROLYZER_DETAILED, pem)
        
        soec = MockSOEC()
        self.registry.register(ComponentID.SOEC_CLUSTER, soec)
        
        # Initialize
        self.registry.initialize_all(dt=1.0)
        
        # Step
        self.tm.step(0.0)
        
        # Verify
        print(f"Total Heat Available: {self.tm.total_heat_available_kw} kW")
        print(f"Total Heat Demand: {self.tm.total_heat_demand_kw} kW")
        print(f"Heat Utilized: {self.tm.heat_utilized_kw} kW")
        print(f"Heat Wasted: {self.tm.heat_wasted_kw} kW")
        
        self.assertEqual(self.tm.total_heat_available_kw, 500.0)
        self.assertEqual(self.tm.total_heat_demand_kw, 200.0)
        self.assertEqual(self.tm.heat_utilized_kw, 200.0)
        self.assertEqual(self.tm.heat_wasted_kw, 300.0)
        
        # Verify active distribution
        print(f"SOEC External Heat Input: {soec.steam_gen_hx4.external_heat_input_kw} kW")
        self.assertEqual(soec.steam_gen_hx4.external_heat_input_kw, 200.0)
        
        # Check accumulators
        self.assertEqual(self.tm.cumulative_heat_recovered_kwh, 500.0)
        self.assertEqual(self.tm.cumulative_heat_utilized_kwh, 200.0)

if __name__ == '__main__':
    unittest.main()
