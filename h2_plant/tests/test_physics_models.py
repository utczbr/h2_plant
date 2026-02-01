"""
Unit tests for core physics models (Thermal Inertia & Flow Dynamics)
"""

import unittest
import math
from h2_plant.models.thermal_inertia import ThermalInertiaModel
from h2_plant.models.flow_dynamics import PumpFlowDynamics, GasAccumulatorDynamics

class TestThermalInertiaModel(unittest.TestCase):
    def setUp(self):
        self.model = ThermalInertiaModel(
            C_thermal_J_K=1000.0,  # Small thermal mass for fast testing
            h_A_passive_W_K=10.0,
            T_initial_K=300.0,
            T_ambient_K=300.0
        )

    def test_initial_state(self):
        self.assertEqual(self.model.T_K, 300.0)
        self.assertEqual(self.model.heat_generated_W, 0.0)

    def test_heating_step(self):
        # Apply 100W heat for 10 seconds
        # Expected dT approx: (100 - 0) / 1000 * 10 = 1.0 K
        T_new = self.model.step(dt_s=10.0, heat_generated_W=100.0)
        self.assertAlmostEqual(T_new, 301.0, delta=0.1)
        self.assertEqual(self.model.T_K, T_new)

    def test_cooling_activation(self):
        # Set temp high enough to trigger cooling
        self.model.T_K = 350.0
        self.model.T_setpoint_K = 333.15
        
        # Step with no heat generation
        T_new = self.model.step(dt_s=1.0, heat_generated_W=0.0)
        
        # Should cool down
        self.assertLess(T_new, 350.0)
        self.assertGreater(self.model.heat_removed_W, 0.0)

    def test_equilibrium(self):
        # Run until equilibrium with constant heat
        # Q_in = Q_loss = hA * (T - T_amb)
        # 100 = 10 * (T - 300) -> T = 310
        
        # Disable active cooling for this test by setting high setpoint
        self.model.T_setpoint_K = 400.0
        
        for _ in range(1000):
            self.model.step(dt_s=10.0, heat_generated_W=100.0)
            
        self.assertAlmostEqual(self.model.T_K, 310.0, delta=0.1)


class TestPumpFlowDynamics(unittest.TestCase):
    def setUp(self):
        self.pump = PumpFlowDynamics(
            initial_flow_m3_h=0.0,
            fluid_inertance_kg_m4=1e10  # Larger inertance for stability
        )

    def test_initial_state(self):
        self.assertEqual(self.pump.Q_m3_h, 0.0)

    def test_ramp_up(self):
        # Step with full speed
        # Pump pressure > System pressure -> Acceleration
        Q_new = self.pump.step(dt_s=1.0, pump_speed_fraction=1.0)
        self.assertGreater(Q_new, 0.0)
        self.assertGreater(self.pump.dQ_dt_m3_h_s, 0.0)

    def test_steady_state(self):
        # Run until steady state
        for _ in range(1000):
            self.pump.step(dt_s=1.0, pump_speed_fraction=1.0)
            
        # At steady state, pump pressure should equal system pressure
        state = self.pump.get_state()
        self.assertAlmostEqual(state['pump_pressure_pa'], state['system_pressure_pa'], delta=100.0)


class TestGasAccumulatorDynamics(unittest.TestCase):
    def setUp(self):
        self.tank = GasAccumulatorDynamics(
            V_tank_m3=1.0,
            initial_pressure_pa=100000.0, # 1 bar
            T_tank_k=300.0
        )

    def test_pressure_increase(self):
        # Add mass
        # dP = (R*T/V) * dm
        # R=4124, T=300, V=1 -> coeff = 1,237,200
        # m_in = 0.001 kg/s, dt=1s -> dm=0.001
        # dP approx 1237 Pa
        
        P_new = self.tank.step(dt_s=1.0, m_dot_in_kg_s=0.001, m_dot_out_kg_s=0.0)
        expected_P = 100000.0 + (4124.0 * 300.0 / 1.0) * 0.001
        self.assertAlmostEqual(P_new, expected_P, delta=1.0)

    def test_pressure_decrease(self):
        P_initial = self.tank.P
        P_new = self.tank.step(dt_s=1.0, m_dot_in_kg_s=0.0, m_dot_out_kg_s=0.001)
        self.assertLess(P_new, P_initial)

if __name__ == '__main__':
    unittest.main()
