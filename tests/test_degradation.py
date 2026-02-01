import sys
import os
import unittest
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.component_ids import ComponentID
from h2_plant.core.component import Component
from h2_plant.components.production.pem_electrolyzer_detailed import DetailedPEMElectrolyzer
from h2_plant.components.production.soec_electrolyzer_detailed import SOECStackArray

class MockLUTManager(Component):
    def initialize(self, dt, registry): super().initialize(dt, registry)
    def step(self, t): pass
    def get_state(self): return {}

class TestDegradation(unittest.TestCase):
    def setUp(self):
        self.registry = ComponentRegistry()
        # Mock LUT Manager
        self.registry.register(ComponentID.LUT_MANAGER, MockLUTManager())
        
    def test_pem_degradation(self):
        # 0 hours
        pem_new = DetailedPEMElectrolyzer(t_op_h_initial=0.0)
        pem_new.initialize(1.0, self.registry)
        
        # 10,000 hours
        pem_old = DetailedPEMElectrolyzer(t_op_h_initial=10000.0)
        pem_old.initialize(1.0, self.registry)
        
        # Run both at same setpoint
        # Need to mock coordinator or manually set setpoint if possible
        # DetailedPEMElectrolyzer reads from coordinator in step()
        # But we can inspect internal method _calculate_degradation_voltage
        
        v_deg_new = pem_new._calculate_degradation_voltage(0.0)
        v_deg_old = pem_old._calculate_degradation_voltage(10000.0)
        
        print(f"PEM V_deg (0h): {v_deg_new} V")
        print(f"PEM V_deg (10000h): {v_deg_old} V")
        
        self.assertEqual(v_deg_new, 0.0)
        self.assertGreater(v_deg_old, 0.0)
        self.assertAlmostEqual(v_deg_old, 0.04, places=3) # 4uV * 10000 = 0.04V
        
    def test_soec_degradation(self):
        # SOECStackArray tracks its own time
        stack = SOECStackArray(max_power_kw=100.0)
        stack.initialize(1.0, self.registry) # dt = 1.0 hour
        
        # Initial efficiency (0 hours)
        stack.power_input_kw = 100.0
        stack.steam_input_kg_h = 1000.0 # Plenty of steam
        stack.step(0.0)
        h2_new = stack.syngas_output_kg_h
        
        # Simulate aging
        stack.t_op_h = 10000.0 # Force 10k hours
        stack.step(0.0)
        h2_old = stack.syngas_output_kg_h
        
        print(f"SOEC H2 (0h): {h2_new} kg/h")
        print(f"SOEC H2 (10000h): {h2_old} kg/h")
        
        self.assertLess(h2_old, h2_new)
        
        # Check specific energy increase
        # Factor = 1 + 0.015 * 10 = 1.15
        # H2 should be ~ 1/1.15 = 0.869 of original
        ratio = h2_old / h2_new
        print(f"SOEC Efficiency Ratio: {ratio}")
        self.assertAlmostEqual(ratio, 1.0/1.15, places=2)

if __name__ == '__main__':
    unittest.main()
