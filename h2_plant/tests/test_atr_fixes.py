import sys
import os
import unittest
import numpy as np
from unittest.mock import MagicMock

# Add project root to path
# Add project root to path
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from h2_plant.components.reforming.atr_reactor import ATRReactor, linear_interp_scalar, cubic_interp_scalar

class TestATRFixes(unittest.TestCase):
    def setUp(self):
        self.reactor = ATRReactor('atr_test', 1000)
        
    def _inject_mock_model(self):
        self.reactor.model = MagicMock()
        self.reactor.model.get_outputs.return_value = {
            'h2_production': 100.0,
            'total_heat_duty': 500.0,
            'biogas_required': 40.0,
            'steam_required': 60.0,
            'water_required': 10.0
        }

    def test_heat_accumulation_logic(self):
        dt = 0.5 # hours
        self.reactor.initialize(dt=dt, registry=None)
        self._inject_mock_model() # Inject mock AFTER initialize to avoid overwrite
        
        # Simulate input
        self.reactor.receive_input('o2_in', 50.0, 'oxygen')
        
        # Run step
        self.reactor.step(0.0)
        
        # Check internal state
        self.assertEqual(self.reactor.heat_duty_kw, 500.0)
        
        # Check buffer accumulation: should be Power * dt = 500 * 0.5 = 250 kWh (or equiv energy unit)
        # The code implementation: self._heat_output_buffer_kw += self.heat_duty_kw * self.dt
        self.assertAlmostEqual(self.reactor._heat_output_buffer_kw, 250.0)
        
        # Check output retrieval: should return Power = Buffer / dt = 250 / 0.5 = 500 kW
        heat_out = self.reactor.get_output('heat_out')
        self.assertEqual(heat_out, 500.0)

    def test_water_flow_handling(self):
        self.reactor.initialize(dt=1.0, registry=None)
        self._inject_mock_model()
        self.reactor.receive_input('o2_in', 50.0, 'oxygen')
        self.reactor.step(0.0)
        
        # Verify water flow was captured from model outputs
        self.assertEqual(self.reactor.water_input_kmol_h, 10.0)
        
        # Verify it appears in get_state
        state = self.reactor.get_state()
        self.assertIn('water_input_kmol_h', state)
        self.assertEqual(state['water_input_kmol_h'], 10.0)

    def test_interpolation_safety(self):
        # Test Numba compiled functions
        x_empty = np.array([], dtype=np.float64)
        y_empty = np.array([], dtype=np.float64)
        
        x_single = np.array([1.0], dtype=np.float64)
        y_single = np.array([10.0], dtype=np.float64)
        
        # Linear
        res = linear_interp_scalar(0.5, x_empty, y_empty)
        self.assertEqual(res, 0.0)
        
        res = linear_interp_scalar(0.5, x_single, y_single)
        self.assertEqual(res, 10.0)
        
        # Cubic
        res = cubic_interp_scalar(0.5, x_empty, y_empty)
        self.assertEqual(res, 0.0)
        
        res = cubic_interp_scalar(0.5, x_single, y_single)
        self.assertEqual(res, 10.0)

if __name__ == '__main__':
    unittest.main()
