
import unittest
import numpy as np
from unittest.mock import MagicMock
from h2_plant.components.electrolysis.soec_operator import SOECOperator
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.stream import Stream

class TestSOECRobustness(unittest.TestCase):
    def setUp(self):
        self.config = {
            'num_modules': 6,
            'max_power_nominal_mw': 2.4,
            'optimal_limit': 0.8
        }
        self.soec = SOECOperator(self.config)
        self.registry = ComponentRegistry()
        self.soec.initialize(1.0, self.registry) # 1 hour dt

    def test_step_api_responds_to_port(self):
        """Test that step() responds to power_in port (Fix Verified)."""
        # Simulate receiving power
        self.soec.receive_input('power_in', 2.0, 'electricity')
        
        # Call step via standard API (just time)
        power, h2, steam = self.soec.step(t=0.0)
        
        # Should be > 0 (starts ramping up)
        self.assertGreater(power, 0.0, "SOEC should respond to port input")
        
    def test_negative_power_clamping(self):
        """Test that negative power setpoints are handled gracefully."""
        self.soec.receive_input('power_in', -5.0, 'electricity')
        power, h2, steam = self.soec.step(t=0.0)
        # Should treat as 0 -> ramp down or standby
        # Starts at standby ??
        # Just ensure no crash
        self.assertGreaterEqual(power, 0.0)

    def test_production_limitation_by_steam(self):
        """Test that production IS limited by steam input (Fix Verified)."""
        # Set high power
        self.soec.receive_input('power_in', 2.0, 'electricity')
        # self.soec.step(t=0.0) # Start ramping
        
        # Now restrict steam to 0.1 kg/h
        # 0.1 kg steam -> ~0.011 kg H2
        self.soec.receive_input('steam_in', Stream(mass_flow_kg_h=0.1, temperature_k=300, pressure_pa=1e5, composition={'H2O':1}), 'water')
        
        # Step
        power, h2, steam = self.soec.step(t=0.0)
        
        # If fixed, h2 should be small (limited by steam)
        # H2 produced in this step (which is 1 hour dt)
        # Max H2 = 0.1 / 10.5 (ratio) * 1h? No steam_input_ratio is kg_steam / kg_h2.
        # h2 = steam / ratio = 0.1 / 10.5 = 0.0095 kg.
        
        self.assertLess(h2, 0.02, "Production should be limited by steam availability")


if __name__ == '__main__':
    unittest.main()
