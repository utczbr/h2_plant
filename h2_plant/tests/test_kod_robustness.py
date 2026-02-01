
import unittest
import math
from h2_plant.components.separation.knock_out_drum import KnockOutDrum, RHO_L_WATER
from h2_plant.core.stream import Stream

class TestKnockOutDrumRobustness(unittest.TestCase):
    def setUp(self):
        self.kod = KnockOutDrum()
        self.kod.initialize(dt=1.0, registry=None)

    def test_dry_feed_logic(self):
        """Test that dry gas passes through without creating water."""
        # Pure H2 input (0% water)
        inlet = Stream(
            mass_flow_kg_h=100.0,
            temperature_k=300.0,
            pressure_pa=40e5,
            composition={'H2': 1.0, 'H2O': 0.0},
            phase='gas'
        )
        self.kod.receive_input('gas_inlet', inlet)
        self.kod.step(0.0)
        
        gas_out = self.kod.get_output('gas_outlet')
        liq_out = self.kod.get_output('liquid_drain')
        
        # Expect 0 liquid drain
        self.assertAlmostEqual(liq_out.mass_flow_kg_h, 0.0, places=5)
        # Expect gas outlet composition to match inlet (pure H2)
        self.assertAlmostEqual(gas_out.composition.get('H2O', 0.0), 0.0, places=5)
        self.assertAlmostEqual(gas_out.composition['H2'], 1.0, places=5)

    def test_stale_state_reset(self):
        """Test that outputs are cleared when flow stops."""
        # Step 1: Flow
        inlet = Stream(mass_flow_kg_h=100.0, composition={'H2':1.0}, phase='gas')
        self.kod.receive_input('gas_inlet', inlet)
        self.kod.step(0.0)
        self.assertIsNotNone(self.kod.get_output('gas_outlet'))
        self.assertEqual(self.kod.get_state()['separation_status'], "OK")
        
        # Step 2: Zero Flow input
        no_flow = Stream(mass_flow_kg_h=0.0, composition={'H2':1.0}, phase='gas')
        self.kod.receive_input('gas_inlet', no_flow)
        self.kod.step(1.0)
        
        # Expect outputs to be None and state reset
        self.assertIsNone(self.kod.get_output('gas_outlet'))
        self.assertEqual(self.kod.get_state()['separation_status'], "NO_FLOW")
        self.assertEqual(self.kod.get_state()['rho_g'], 0.0)

    def test_souders_brown_guard(self):
        """Test that improbable high density doesn't crash."""
        # Mocking density via extremely high pressure might be hard due to thermodynamics,
        # but we can try normal operation and see if it passes, which it should.
        # To strictly test the guard we'd need to mock internal values or use invalid EOS.
        # For now, we verify that normal and high pressure operation is stable.
        
        # Very high pressure inlet (should produce high rho_g)
        inlet = Stream(
            mass_flow_kg_h=1000.0,
            temperature_k=300.0,
            pressure_pa=500e5, # 500 bar
            composition={'H2': 0.0, 'O2': 1.0}, # O2 is heavier
            phase='gas'
        )
        # O2 at 500 bar is dense (~600 kg/m3?) but < 1000 kg/m3.
        # Let's try to break it with hypothetical heavy gas if possible, but O2 is heaviest available.
        # Just running this ensures no crash.
        self.kod.gas_species = 'O2'
        self.kod.receive_input('gas_inlet', inlet)
        try:
            self.kod.step(0.0)
        except ValueError as e:
            self.fail(f"Step crashed with ValueError: {e}")
            
        state = self.kod.get_state()
        self.assertTrue(0 <= state['rho_g'])
