
import sys
import os
import unittest
# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from h2_plant.core.stream import Stream
from h2_plant.components.cooling.dry_cooler import DryCooler
from h2_plant.components.thermal.interchanger import Interchanger
from h2_plant.components.thermal.chiller import Chiller
from h2_plant.components.thermal.electric_boiler import ElectricBoiler

class TestMassBalance(unittest.TestCase):
    def setUp(self):
        # Create a "Wet" Stream
        # 100 kg/h Gas (H2) + 10 kg/h Liquid (H2O) Entrained
        self.input_stream = Stream(
            mass_flow_kg_h=100.0,
            temperature_k=350.0,
            pressure_pa=101325.0,
            composition={'H2': 1.0},
            extra={'m_dot_H2O_liq_accomp_kg_s': 10.0 / 3600.0}
        )
        self.expected_total_mass = 110.0

    def test_dry_cooler_mass_balance(self):
        dc = DryCooler(component_id="dc_test")
        dc.initialize(dt=1/60, registry=None)
        dc.receive_input("fluid_in", self.input_stream)
        dc.step(0.0)
        state = dc.get_state()
        print(f"DryCooler Outlet: {state['outlet_mass_flow_kg_h']:.4f}")
        self.assertAlmostEqual(state['outlet_mass_flow_kg_h'], self.expected_total_mass, places=2)

    def test_interchanger_mass_balance(self):
        # Needs hot and cold. We test hot side.
        ic = Interchanger(component_id="ic_test")
        ic.initialize(dt=1/60, registry=None)
        ic.receive_input("hot_in", self.input_stream)
        
        # Dummy cold stream
        cold_stream = Stream(100.0, temperature_k=300.0)
        ic.receive_input("cold_in", cold_stream)
        
        ic.step(0.0)
        state = ic.get_state()
        print(f"Interchanger Outlet: {state['outlet_mass_flow_kg_h']:.4f}")
        self.assertAlmostEqual(state['outlet_mass_flow_kg_h'], self.expected_total_mass, places=2)

    def test_chiller_mass_balance(self):
        ch = Chiller(component_id="ch_test")
        ch.initialize(dt=1/60, registry=None)
        ch.receive_input("fluid_in", self.input_stream)
        ch.step(0.0)
        state = ch.get_state()
        print(f"Chiller Outlet: {state['outlet_mass_flow_kg_h']:.4f}")
        self.assertAlmostEqual(state['outlet_mass_flow_kg_h'], self.expected_total_mass, places=2)

    def test_boiler_mass_balance(self):
        bo = ElectricBoiler(config={'max_power_kw': 100}, component_id="bo_test")
        bo.initialize(dt=1/60, registry=None) # Registry optional for basic test
        # Need registry for LUT? Boiler handles no registry gracefully (fallback Cp).
        bo.receive_input("fluid_in", self.input_stream)
        bo.step(0.0)
        state = bo.get_state()
        print(f"Boiler Outlet: {state['outlet_mass_flow_kg_h']:.4f}")
        self.assertAlmostEqual(state['outlet_mass_flow_kg_h'], self.expected_total_mass, places=2)

if __name__ == '__main__':
    unittest.main()
