
import unittest
from unittest.mock import MagicMock, patch
from h2_plant.components.compression.compressor import CompressorStorage
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.component_ids import ComponentID

class TestCompressorRobustness(unittest.TestCase):
    def setUp(self):
        self.compressor = CompressorStorage(
            max_flow_kg_h=100.0,
            inlet_pressure_bar=20.0,
            outlet_pressure_bar=100.0
        )
        self.registry = ComponentRegistry()
        self.compressor.initialize(1.0, self.registry)

    def test_fallback_energy_assignment(self):
        """Test that fallback logic actually calculates/assigns energy correctly."""
        # Force fallback by ensuring no LUT
        # currently registry is empty, so LUT is None.
        
        # Set conditions
        self.compressor.transfer_mass_kg = 50.0
        # step() logic:
        # 1. Calculates actual mass (50 < 100) -> 50
        # 2. Calls _calculate_compression_physics -> falls back -> calculates local energy vars
        # 3. step() overwrites energy_consumed_kwh with compression_work_kwh + chilling
        
        self.compressor.step(1.0)
        
        # Bug: compression_work_kwh is 0 in fallback, so energy_consumed_kwh becomes 0
        self.assertGreater(self.compressor.energy_consumed_kwh, 0.0, 
                          "Energy consumed should be > 0 even in fallback mode")

    def test_fallback_uses_correct_mass(self):
        """Test that fallback uses actual constrained mass, not requested mass."""
        # Request huge mass > max_flow
        self.compressor.transfer_mass_kg = 1000.0 # max is 100
        
        # Verify actual limitation logic (part of step)
        self.compressor.step(1.0)
        
        expected_mass = 100.0
        self.assertEqual(self.compressor.actual_mass_transferred_kg, expected_mass)
        
        # Check energy scaling
        # Fallback currently uses transfer_mass_kg (1000) instead of actual (100)
        # So energy will be 10x higher than it should.
        
        # Let's calculate specific energy roughly
        # 20->100 bar is 5x. Ideal gas work W ~ Cp*T*(r^k - 1)/eff
        # Just check consistency: Energy / Actual Mass should = Specific Energy
        specific = self.compressor.specific_energy_kwh_kg
        total = self.compressor.energy_consumed_kwh
        
        # If bug exists: total = specific * 1000. 
        # If fixed: total = specific * 100.
        
        if total > 0: # Only if previous test passes or we ignore it
             ratio = total / specific if specific > 0 else 0
             self.assertAlmostEqual(ratio, expected_mass, delta=1.0,
                                   msg=f"Energy calculated for mass {ratio}, expected {expected_mass}")

    def test_no_compression_short_circuit(self):
        """Test short circuit when P_out <= P_in."""
        self.compressor.outlet_pressure_bar = 10.0 # < Inlet 20.0
        self.compressor.initialize(1.0, self.registry) # Re-init stage calc
        
        self.compressor.transfer_mass_kg = 50.0
        self.compressor.step(1.0)
        
        self.assertEqual(self.compressor.energy_consumed_kwh, 0.0, 
                        "Should consume 0 energy if P_out <= P_in")

if __name__ == '__main__':
    unittest.main()
