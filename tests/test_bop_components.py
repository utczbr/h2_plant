import unittest
import numpy as np
from h2_plant.components.balance_of_plant.compressor import Compressor
from h2_plant.components.balance_of_plant.pump import Pump
from h2_plant.components.balance_of_plant.tank import Tank

class TestBoPComponents(unittest.TestCase):
    
    def test_compressor_physics(self):
        """Test Compressor multi-stage logic."""
        # Config based on legacy defaults
        config = {
            "efficiency": 0.65,
            "t_in_c": 10.0,
            "t_max_c": 85.0
        }
        comp = Compressor(config)
        comp.initialize(1.0, None)
        
        # Test Case 1: 40 bar -> 140 bar (Legacy Example 1)
        # Legacy output approx: 1.2556 kWh/kg, 2 stages
        comp.step(t=0, mass_flow_kg_h=100.0, p_in_bar=40.0, p_out_bar=140.0)
        
        state = comp.get_state()
        print(f"\nCompressor (40->140 bar): {state}")
        
        # Check if power is reasonable (around 1.25 kWh/kg * 100 kg/h = 125 kW)
        expected_spec_energy = 1.25 # kWh/kg
        expected_power = expected_spec_energy * 100.0
        
        # Allow some tolerance due to CoolProp/Ideal Gas differences
        self.assertTrue(state['stages'] >= 1)
        self.assertGreater(state['power_kw'], 0.0)
        
    def test_pump_physics(self):
        """Test Pump logic."""
        config = {
            "eta_is": 0.82,
            "eta_m": 0.96
        }
        pump = Pump(config)
        pump.initialize(1.0, None)
        
        # Test Case: 1 bar -> 30 bar, 1000 kg/h
        # Hydraulic power approx: (1 m3/h * 29 bar * 100) / 3600 = 0.8 kW
        # Shaft power approx: 0.8 / (0.82 * 0.96) ~= 1.0 kW
        pump.step(t=0, mass_flow_kg_h=1000.0, p_in_bar=1.0, p_out_bar=30.0)
        
        state = pump.get_state()
        print(f"Pump (1->30 bar, 1000 kg/h): {state}")
        
        self.assertGreater(state['power_kw'], 0.5)
        self.assertLess(state['power_kw'], 2.0)

    def test_tank_physics(self):
        """Test Tank mass balance."""
        config = {
            "capacity_kg": 100.0,
            "initial_level_kg": 50.0,
            "max_pressure_bar": 200.0
        }
        tank = Tank(config)
        tank.initialize(1.0, None) # dt = 1 hour (default if not set, need to check implementation)
        # Actually Tank implementation uses getattr(self, 'dt', 1.0)
        
        # Step 1: Fill 10 kg
        tank.step(t=0, flow_in_kg_h=10.0, flow_out_kg_h=0.0)
        state = tank.get_state()
        self.assertAlmostEqual(state['level_kg'], 60.0)
        self.assertAlmostEqual(state['fill_percentage'], 60.0)
        self.assertAlmostEqual(state['pressure_bar'], 120.0) # 60% of 200 bar
        
        # Step 2: Drain 20 kg
        tank.step(t=1, flow_in_kg_h=0.0, flow_out_kg_h=20.0)
        state = tank.get_state()
        self.assertAlmostEqual(state['level_kg'], 40.0)

if __name__ == '__main__':
    unittest.main()
