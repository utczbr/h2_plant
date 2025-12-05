
import unittest
from h2_plant.control.dispatch import SoecOnlyStrategy, DispatchInput, DispatchState

class TestSoecOnlyStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = SoecOnlyStrategy()
        self.state = DispatchState(P_soec_prev=10.0, force_sell=False)
        
    def test_force_sell_trigger(self):
        # Minute 0, Ramp Up (Offer 12 > Prev 10)
        # Price High (400 > 306)
        # Sale Profit: (12-10)*0.25 * (400-50) = 2 * 0.25 * 350 = 175 EUR
        # H2 Profit: (12-10)*0.25 * (1000/37.5) * 9.6 = 0.5 * 26.66 * 9.6 = 128 EUR
        # Should Trigger Force Sell
        
        inputs = DispatchInput(
            minute=0,
            P_offer=12.0,
            P_future_offer=12.0,
            current_price=400.0,
            soec_capacity_mw=12.0,
            pem_max_power_mw=0.0,
            soec_h2_kwh_kg=37.5,
            pem_h2_kwh_kg=50.0
        )
        
        result = self.strategy.decide(inputs, self.state)
        
        self.assertTrue(result.state_update['force_sell'])
        self.assertEqual(result.P_soec, 10.0) # Locked at prev
        self.assertEqual(result.P_sold, 2.0) # Surplus sold
        
    def test_normal_operation(self):
        # Minute 10, Price Low
        inputs = DispatchInput(
            minute=10,
            P_offer=12.0,
            P_future_offer=12.0,
            current_price=50.0,
            soec_capacity_mw=12.0,
            pem_max_power_mw=0.0,
            soec_h2_kwh_kg=37.5,
            pem_h2_kwh_kg=50.0
        )
        
        result = self.strategy.decide(inputs, self.state)
        
        self.assertFalse(result.state_update['force_sell'])
        self.assertEqual(result.P_soec, 12.0) # Full power
        self.assertEqual(result.P_sold, 0.0)

if __name__ == '__main__':
    unittest.main()
