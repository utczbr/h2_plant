import sys
import os
import unittest
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.component_ids import ComponentID
from h2_plant.components.compression.filling_compressor import FillingCompressor
from h2_plant.components.thermal.thermal_manager import ThermalManager

class TestCompressorHeat(unittest.TestCase):
    def setUp(self):
        self.registry = ComponentRegistry()
        self.tm = ThermalManager()
        self.registry.register(ComponentID.THERMAL_MANAGER, self.tm)
        
    def test_compressor_heat_collection(self):
        # Register Compressor
        comp = FillingCompressor(max_flow_kg_h=100.0, num_stages=3)
        self.registry.register("compressor_1", comp)
        
        # Initialize
        self.registry.initialize_all(dt=1.0)
        
        # Run compressor
        comp.transfer_mass_kg = 50.0
        comp.step(0.0)
        
        print(f"Compressor Energy: {comp.energy_consumed_kwh} kWh")
        print(f"Compressor Heat: {comp.heat_output_kw} kW")
        
        self.assertGreater(comp.heat_output_kw, 0.0)
        
        # Run Thermal Manager
        self.tm.step(0.0)
        
        print(f"TM Total Heat: {self.tm.total_heat_available_kw} kW")
        self.assertAlmostEqual(self.tm.total_heat_available_kw, comp.heat_output_kw, places=2)

if __name__ == '__main__':
    unittest.main()
