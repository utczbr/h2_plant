
import unittest
from h2_plant.components.separation.coalescer import Coalescer
from h2_plant.core.stream import Stream

class TestCoalescerRobustness(unittest.TestCase):
    def setUp(self):
        self.coalescer = Coalescer()
        self.coalescer.initialize(1.0, None)

    def test_constructor_validation(self):
        """Test that invalid dimensions raise ValueError."""
        with self.assertRaises(ValueError):
            Coalescer(d_shell=0.0)
        with self.assertRaises(ValueError):
            Coalescer(d_shell=-1.0)
        with self.assertRaises(ValueError):
            Coalescer(l_elem=0.0)

    def test_stale_state_on_invalid_density(self):
        """Test that outputs are reset when density is invalid."""
        # Step 1: Valid flow
        valid_stream = Stream(
            mass_flow_kg_h=100.0,
            temperature_k=300.0,
            pressure_pa=30e5,
            composition={'H2': 1.0},
            phase='gas'
        )
        self.coalescer.receive_input('inlet', valid_stream)
        self.assertIsNotNone(self.coalescer.output_stream)
        self.assertGreater(self.coalescer.output_stream.mass_flow_kg_h, 0.0)

        # Step 2: Stream with invalid density (simulated by mocking or zero pressure/temp edge case)
        # Note: Stream property density_kg_m3 relies on pressure/temp. 
        # If we pass P=0, rho -> 0.
        invalid_stream = Stream(
            mass_flow_kg_h=100.0, # Flow exists
            temperature_k=300.0,
            pressure_pa=0.0, # Causes zero density
            composition={'H2': 1.0}
        )
        
        self.coalescer.receive_input('inlet', invalid_stream)
        
        # Expect output to be cleared/reset, not stale
        # Current BUG: output_stream likely remains from Step 1
        msg = "Output stream should be None or empty on invalid density"
        if self.coalescer.output_stream is not None:
             self.assertEqual(self.coalescer.output_stream.mass_flow_kg_h, 0.0, msg)
        
        state = self.coalescer.get_state()
        self.assertEqual(state['delta_p_bar'], 0.0, "Delta P should be 0 on error")

if __name__ == '__main__':
    unittest.main()
