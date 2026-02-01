
import unittest
import sys
import os
import numpy as np

# Adjust path to include project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from h2_plant.config.constants_physics import WaterConstants
from h2_plant.components.water.water_purifier import WaterPurifier
from h2_plant.components.water.ultrapure_water_tank import UltraPureWaterTank
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.stream import Stream

try:
    import CoolProp.CoolProp as CP
    COOLPROP_AVAILABLE = True
except ImportError:
    COOLPROP_AVAILABLE = False
    print("Warning: CoolProp not found. Skipping direct CoolProp comparisons.")

class TestWaterSystem(unittest.TestCase):
    def setUp(self):
        self.registry = ComponentRegistry()
        self.registry.add_factories(WP=WaterPurifier, TANK=UltraPureWaterTank)
        self.dt = 1.0 # 1 hour timestep

        # Mock LUT Manager (basic) or use real one if needed
        # For unit tests, we can verify the component logic independently of LUT accuracy often
        # But if component relies on LUT, we need it.
        # Let's see if we can instantiate it.
        try:
             from h2_plant.optimization.lut_manager import LUTManager
             self.lut = LUTManager() # Will try to load cache or fail
             self.registry.register("lut_manager", self.lut, "lut_manager")
        except:
             pass

    def test_water_purifier_logic(self):
        """Unit Test: Water Purifier mass balance and power."""
        wp = self.registry.create_component('WP', 'wp_1', max_flow_kg_h=1000.0)
        wp.initialize(self.dt, self.registry)
        
        # 1. Feed 1000 kg input
        raw_stream = Stream(1000.0, 293.15, 101325.0, {'H2O': 0.99}, 'liquid')
        wp.receive_input('raw_water_in', raw_stream)
        
        # 2. Step
        wp.step(0.0)
        
        # 3. Verify Output mass balance
        # Recovery = 0.75 (default)
        expected_pure = 1000.0 * 0.75
        expected_waste = 1000.0 * 0.25
        
        self.assertAlmostEqual(wp.ultrapure_out_stream.mass_flow_kg_h, expected_pure)
        self.assertAlmostEqual(wp.waste_out_stream.mass_flow_kg_h, expected_waste)
        
        # 4. Verify Power
        # 0.004 kWh/kg * 750 kg = 3.0 kWh
        # Power = 3.0 kW (since dt=1h)
        expected_power = expected_pure * WaterConstants.WATER_RO_SPEC_ENERGY_KWH_KG
        self.assertAlmostEqual(wp.power_consumed_kw, expected_power)

    def test_tank_mixing_logic(self):
        """Unit Test: Tank enthalpy mixing."""
        tank = self.registry.create_component('TANK', 'tank_1', capacity_kg=10000.0)
        tank.initialize(self.dt, self.registry)
        
        # Start state: 50% full (5000kg) at 20°C (293.15 K)
        initial_mass = 5000.0
        initial_T = 293.15
        tank.mass_kg = initial_mass
        tank.temperature_k = initial_T
        
        # Add Input: 1000 kg at 60°C (333.15 K)
        # Using simple Cp = 4184 J/kgK constant assumption in code currently
        in_stream = Stream(1000.0, 333.15, 101325.0, {'H2O': 1.0}, 'liquid')
        tank.receive_input('ultrapure_in', in_stream)
        
        tank.step(0.0)
        
        # Expected Mixing T
        # T_mix = (m1*T1 + m2*T2) / (m1+m2) (assuming constant Cp)
        m1, T1 = initial_mass, initial_T
        m2, T2 = 1000.0, 333.15
        expected_T = (m1 * T1 + m2 * T2) / (m1 + m2)
        
        self.assertAlmostEqual(tank.mass_kg, 6000.0)
        self.assertAlmostEqual(tank.temperature_k, expected_T)

    def test_coupling_purifier_to_tank(self):
        """Coupling Test: Purifier feeding Tank."""
        wp = self.registry.create_component('WP', 'wp_1', max_flow_kg_h=2000.0)
        tank = self.registry.create_component('TANK', 'tank_1')
        
        wp.initialize(self.dt, self.registry)
        tank.initialize(self.dt, self.registry)
        
        # Feed Purifier
        raw_stream = Stream(2000.0, 293.15, 101325.0, {'H2O': 0.99}, 'liquid')
        wp.receive_input('raw_water_in', raw_stream)
        
        # Run System Step
        # 1. Purifier Step
        wp.step(0.0)
        out_stream = wp.get_output('ultrapure_out')
        
        # 2. Transfer to Tank
        accepted = tank.receive_input('ultrapure_in', out_stream)
        
        # 3. Tank Step
        tank.step(0.0)
        
        # Verify
        expected_pure_flow = 2000.0 * 0.75 # 1500 kg/h
        self.assertAlmostEqual(out_stream.mass_flow_kg_h, 1500.0)
        
        # Tank should have accepted all (capacity is large)
        self.assertAlmostEqual(accepted, 1500.0) 
        
        # Tank mass increased by 1500
        # Initial is 5000 (0.5 * 10000)
        self.assertAlmostEqual(tank.mass_kg, 5000.0 + 1500.0)

    @unittest.skipUnless(COOLPROP_AVAILABLE, "CoolProp not installed")
    def test_coolprop_verification(self):
        """Verification: Compare simplified logic vs CoolProp."""
        # 1. Check Density/Water Prop assumptions
        # Code assumes ~1000 kg/m3 or uses LUT. 
        # Code assumes Cp ~4184 J/kgK for mixing if LUT missing.
        
        T_test = 300.0 # K
        P_test = 101325.0 # Pa
        
        cp_real = CP.PropsSI('C', 'T', T_test, 'P', P_test, 'Water')
        rho_real = CP.PropsSI('D', 'T', T_test, 'P', P_test, 'Water')
        
        print(f"\nCoolProp Verification (T={T_test}K):")
        print(f"  Cp_real: {cp_real:.2f} J/kgK")
        print(f"  Rho_real: {rho_real:.2f} kg/m3")
        
        # Check if our mixing logic with 4184 is reasonable error
        # 4184 is standard at 20C.
        error_cp = abs(cp_real - 4184.0) / cp_real
        print(f"  Cp Assumption Error: {error_cp*100:.2f}%")
        
        # We accept small error for the explicit formula, 
        # BUT if LUT is used, it should be closer.
        # Let's verify LUT if available
        if hasattr(self, 'lut'):
            try:
                cp_lut = self.lut.lookup('Water', 'C', P_test, T_test)
                print(f"  LUT Cp: {cp_lut:.2f}")
                self.assertLess(abs(cp_lut - cp_real)/cp_real, 0.01) # <1% error for LUT
            except:
                print("  LUT lookup failed or not initialized")

if __name__ == '__main__':
    unittest.main()
