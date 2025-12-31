import unittest
import numpy as np
from h2_plant.components.electrolysis.pem_electrolyzer import DetailedPEMElectrolyzer
from h2_plant.config.constants_physics import PEMConstants
from h2_plant.core.component_registry import ComponentRegistry

class TestPEMPhysicsUpdate(unittest.TestCase):
    def setUp(self):
        self.config = {
            'max_power_mw': 5.0,
            'base_efficiency': 0.65,
            'use_polynomials': False,
            'component_id': 'PEM_1'
        }
        self.pem = DetailedPEMElectrolyzer(self.config)
        self.registry = ComponentRegistry()
        self.pem.initialize(dt=1.0, registry=self.registry) 

    def test_constants_loaded(self):
        CONST = PEMConstants()
        print(f"\n[CONSTANTS CHECK]")
        print(f"O2 Crossover (ppm): {CONST.o2_crossover_ppm_molar} (Expected: 200)")
        print(f"Anode H2 Crossover (ppm): {CONST.anode_h2_crossover_ppm_molar} (Expected: 4000)")
        print(f"Cathode Liquid Factor: {CONST.cathode_liquid_water_factor} (Expected: 5.0)")
        
        self.assertEqual(CONST.o2_crossover_ppm_molar, 200.0)
        self.assertEqual(CONST.anode_h2_crossover_ppm_molar, 4000.0)
        self.assertEqual(CONST.cathode_liquid_water_factor, 5.0)

    def test_output_properties(self):
        # Run at full power
        self.pem.set_power_input_mw(5.0)
        self.pem.water_buffer_kg = 10000.0 # Plenty of water
        self.pem.step(t=0.0)
        
        h2_stream = self.pem.get_output('h2_out')
        o2_stream = self.pem.get_output('oxygen_out')
        
        print(f"\n[H2 STREAM CHECK]")
        print(f"Pressure: {h2_stream.pressure_pa/1e5:.2f} bar")
        print(f"Temp: {h2_stream.temperature_k:.2f} K")
        print(f"Composition: {h2_stream.composition}")
        print(f"Liquid H2O Flow: {h2_stream.extra.get('m_dot_H2O_liq_accomp_kg_s', 0)*3600:.2f} kg/h")
        
        print(f"\n[O2 STREAM CHECK]")
        print(f"Pressure: {o2_stream.pressure_pa/1e5:.2f} bar")
        print(f"Temp: {o2_stream.temperature_k:.2f} K")
        print(f"Composition: {o2_stream.composition}")
        print(f"Liquid H2O Flow (Cooling): {o2_stream.extra.get('m_dot_H2O_liq_accomp_kg_s', 0)*3600:.2f} kg/h")
        
        # CHECKS
        # 1. O2 Pressure should be ~40 bar
        self.assertAlmostEqual(o2_stream.pressure_pa, 40e5, delta=1e4)
        
        # 2. H2 Impurity in O2 should be ~0.4% (molar) -> convert to mass to check existence
        self.assertIn('H2', o2_stream.composition)
        self.assertGreater(o2_stream.composition['H2'], 0.0)
        
        # 3. Liquid Water Check
        # Consumed water approx 800-900 kg/h for 5MW
        consumed_approx = self.pem.m_H2O_kg_s * 3600
        cathode_liq_expected = consumed_approx * 5.0
        cathode_liq_actual = h2_stream.extra.get('m_dot_H2O_liq_accomp_kg_s', 0) * 3600
        
        print(f"Consumed Water: {consumed_approx:.2f} kg/h")
        print(f"Expected Cathode Liq (5x): {cathode_liq_expected:.2f} kg/h")
        
        self.assertAlmostEqual(cathode_liq_actual, cathode_liq_expected, delta=1.0)
        
        # 4. Cooling Flow Check
        # Heat is approx 1.7 MW. Calc cooling flow.
        q_dot = self.pem.heat_output_kw # kW
        expected_cooling_kgs = q_dot / (4.18 * 5.0) # 5K delta T
        expected_cooling_kgh = expected_cooling_kgs * 3600
        
        anode_liq = o2_stream.extra.get('m_dot_H2O_liq_accomp_kg_s', 0) * 3600
        print(f"Heat Output: {q_dot:.2f} kW")
        print(f"Expected Cooling Total: {expected_cooling_kgh:.2f} kg/h")
        print(f"Actual Anode Liq: {anode_liq:.2f} kg/h")
        
        # Anode liq should be Total Cooling - Consumed - Cathode Drag
        expected_anode = expected_cooling_kgh - consumed_approx - cathode_liq_expected
        self.assertAlmostEqual(anode_liq, expected_anode, delta=100.0) # Allow small rounding diffs

if __name__ == '__main__':
    unittest.main()
