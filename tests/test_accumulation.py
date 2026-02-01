
import unittest
from h2_plant.components.compression.compressor import CompressorStorage as Compressor
from h2_plant.core.stream import Stream

class TestCompressorAccumulation(unittest.TestCase):
    def test_power_accumulation(self):
        """Verify that multiple step() calls in same timestep accumulate power."""
        comp = Compressor(max_flow_kg_h=100.0, inlet_pressure_bar=30.0, outlet_pressure_bar=200.0)
        dt = 1.0 # 1 hour
        comp.initialize(dt, None)
        
        # Test Case: 10 kg of H2 split into two 5kg chunks
        # Chunk 1
        stream1 = Stream(mass_flow_kg_h=5.0, temperature_k=300.0, pressure_pa=30e5)
        comp.receive_input("inlet", stream1, "hydrogen")
        comp.step(t=0.0)
        p1 = comp.power_kw
        e1 = comp.energy_consumed_kwh
        print(f"Pass 1: Power={p1:.2f} kW, Energy={e1:.2f} kWh")
        
        # Power should be > 0. Roughly 1.3 kWh/kg * 5 kg = 6.5 kWh. Power = 6.5 kW. 
        self.assertGreater(p1, 0.0)
        
        # Chunk 2 (same timestep t=0.0)
        stream2 = Stream(mass_flow_kg_h=5.0, temperature_k=300.0, pressure_pa=30e5)
        comp.receive_input("inlet", stream2, "hydrogen")
        comp.step(t=0.0)
        p2 = comp.power_kw
        e2 = comp.timestep_energy_kwh # CHECK ACCUMULATOR
        print(f"Pass 2: Power={p2:.2f} kW, Energy={e2:.2f} kWh (Accumulated)")
        
        # Power should roughly double
        self.assertAlmostEqual(p2, p1 * 2, delta=0.5)
        self.assertAlmostEqual(e2, e1 * 2, delta=0.5)
        
        # Move to next timestep
        comp.receive_input("inlet", stream1, "hydrogen")
        comp.step(t=1.0)
        p3 = comp.power_kw
        print(f"Pass 3 (New Step): Power={p3:.2f} kW")
        
        # Should match Pass 1 (reset)
        self.assertAlmostEqual(p3, p1, delta=0.1)

if __name__ == '__main__':
    unittest.main()
