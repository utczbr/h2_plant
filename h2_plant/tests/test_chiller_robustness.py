
import unittest
from h2_plant.components.thermal.chiller import Chiller
from h2_plant.core.stream import Stream

class TestChillerRobustness(unittest.TestCase):
    def setUp(self):
        self.chiller = Chiller(
            cooling_capacity_kw=100.0,
            target_temp_k=300.0,
            efficiency=1.0, # Simplifies math
            cop=4.0
        )
        self.chiller.initialize(1.0, None)

    def test_capacity_cap(self):
        """Test that cooling load is capped at capacity."""
        # Create a stream that requires huge cooling (e.g. 1000 kW) to reach target
        # H2 Cp ~ 14.3 kJ/kgK. 
        # m = 1 kg/s = 3600 kg/h.
        # dT = 100 K.
        # Q = 1 * 14.3 * 100 = 1430 kW.
        # Capacity is 100 kW.
        
        inlet = Stream(
            mass_flow_kg_h=3600.0, # 1 kg/s
            temperature_k=400.0,   # Target 300K -> dT=100K
            pressure_pa=1e5,
            composition={'H2': 1.0}
        )
        
        self.chiller.receive_input('fluid_in', inlet)
        self.chiller.step(1.0)
        
        # Should be capped at 100 kW
        # Current behavior: Uncapped -> ~1430 kW
        self.assertAlmostEqual(self.chiller.cooling_load_kw, 100.0, delta=1.0, 
                               msg=f"Cooling load {self.chiller.cooling_load_kw} exceeds capacity 100.0")

    def test_phantom_water_independence(self):
        """Test that cooling water output requires input or uses tracked state properly."""
        # Case 1: Propagated Inlet
        cw_in = Stream(mass_flow_kg_h=500.0, temperature_k=300.0, pressure_pa=2e5, composition={'H2O': 1.0})
        self.chiller.receive_input('cooling_water_in', cw_in, 'water')
        
        # Force some heat rejection
        # m=100 kh/h, dT=10K -> Q ~ 4 kW
        fluid_in = Stream(mass_flow_kg_h=100.0, temperature_k=310.0, pressure_pa=1e5, composition={'H2': 1.0})
        self.chiller.receive_input('fluid_in', fluid_in)
        self.chiller.step(1.0)
        
        cw_out = self.chiller.get_output('cooling_water_out')
        self.assertEqual(cw_out.pressure_pa, 2e5, "Output pressure should match inlet")
        self.assertGreater(cw_out.temperature_k, 300.0, "Output temp should increase")
        self.assertAlmostEqual(cw_out.mass_flow_kg_h, self.chiller.cooling_water_flow_kg_h, 5) 

if __name__ == '__main__':
    unittest.main()
