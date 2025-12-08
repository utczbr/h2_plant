
import unittest
import logging
from h2_plant.components.water.water_pump import WaterPumpThermodynamic
from h2_plant.core.component_registry import ComponentRegistry

class TestPumpRobustness(unittest.TestCase):
    def setUp(self):
        self.registry = ComponentRegistry()

    def test_initialization_logging_truncation(self):
        """Test that initialization log is not truncated when P_target is None."""
        pump = WaterPumpThermodynamic(pump_id='test_pump', target_pressure_pa=None)
        
        with self.assertLogs(level='INFO') as cm:
            pump.initialize(1.0, self.registry)
        
        # Current bug: log message becomes "P_target=not set" and loses efficiencies
        log_msg = cm.output[0]
        self.assertIn("P_target=not set", log_msg)
        
        # Verify efficiencies are present (Fail expectation)
        if "η_is=" not in log_msg:
             print(f"Reproduced truncation: {log_msg}")
        else:
             print("Logging appears fixed?")
             
        self.assertIn("η_is=", log_msg, "Log message truncated! Lost efficiency info.")
        self.assertIn("η_m=", log_msg)

if __name__ == '__main__':
    unittest.main()
