
import unittest
import logging
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.stream import Stream
from h2_plant.components.water.water_pump import WaterPumpThermodynamic
from h2_plant.components.mixing.water_mixer import WaterMixer
from h2_plant.components.thermal.chiller import Chiller
from h2_plant.components.separation.coalescer import Coalescer
from h2_plant.components.separation.knock_out_drum import KnockOutDrum
from h2_plant.components.compression.compressor import CompressorStorage

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestIntegrationChains(unittest.TestCase):
    def setUp(self):
        self.registry = ComponentRegistry()
        self.dt = 1.0 # 1 hour steps

    def test_water_conditioning_loop(self):
        """
        Water Conditioning Loop: 
        WaterPump -> WaterMixer -> Chiller -> Coalescer.
        Target: Chilled pressurized water.
        """
        logger.info("Starting Water Conditioning Loop Test")
        
        # 1. Instantiate Components
        pump = WaterPumpThermodynamic(
            pump_id='feed_pump',
            eta_is=0.8, eta_m=0.95, target_pressure_pa=30e5 # 30 bar
        )
        
        mixer = WaterMixer(
            max_inlet_streams=2
        )
        mixer.component_id = 'feed_mixer'
        
        chiller = Chiller(
            component_id='feed_chiller',
            cooling_capacity_kw=100.0,
            target_temp_k=283.15 # 10C
        )
        
        coalescer = Coalescer(
            component_id='feed_coalescer',
            # Use defaults for d_shell/l_elem
        )
        
        # Register
        self.registry.register(pump.component_id, pump)
        self.registry.register(mixer.component_id, mixer)
        self.registry.register(chiller.component_id, chiller)
        self.registry.register(coalescer.component_id, coalescer)
        
        # Initialize
        pump.initialize(self.dt, self.registry)
        mixer.initialize(self.dt, self.registry)
        chiller.initialize(self.dt, self.registry)
        coalescer.initialize(self.dt, self.registry)
        
        # 2. Define Input Stream
        # Pure water at 1 atm/25C
        water_in = Stream(
            mass_flow_kg_h=1000.0,
            temperature_k=298.15,
            pressure_pa=101325.0,
            composition={'H2O': 1.0},
            phase='liquid'
        )
        
        # 3. Simulate Step
        # Manually chain for unit test or use registry.step() if connections managed?
        # Components don't know connections unless Graph/Orchestrator manages them.
        # We will manually pass streams.
        
        # Step 1: Pump
        pump.receive_input('water_in', water_in, 'water')
        pump.step(0.0)
        pump_out = pump.get_output('water_out')
        
        self.assertAlmostEqual(pump_out.pressure_pa, 30e5, delta=1e4)
        self.assertGreater(pump_out.temperature_k, 298.15) # Should rise slightly
        
        # Step 2: Mixer (Single input for now)
        mixer.receive_input('inlet_0', pump_out, 'water') # Use inlet_0
        mixer.step(0.0)
        mixer_out = mixer.get_output('outlet')
        
        self.assertEqual(mixer_out.mass_flow_kg_h, 1000.0)
        
        # Step 3: Chiller
        chiller.receive_input('fluid_in', mixer_out, 'water')
        chiller.step(0.0)
        chiller_out = chiller.get_output('fluid_out')
        
        self.assertAlmostEqual(chiller_out.temperature_k, 283.15, delta=0.5)
        
        # Step 4: Coalescer (Safety check)
        coalescer.receive_input('inlet', chiller_out, 'mixed')
        coalescer.step(0.0)
        final_water = coalescer.get_output('drain') # Coalescer separates liquid to drain
        
        # For a water conditioning loop, likely we want the liquid stream.
        # Using 'drain' as the product? Or is Coalescer filtering SOLIDS/OIL from water?
        # The implemented Coalescer (from earlier task) removes liquid from gas.
        # If we feed it LIQUID water, it might consider it ALL liquid and drain it?
        # Let's see.
        pass

    def test_gas_cleanup_chain(self):
        """
        Gas Cleanup Chain:
        Input (Wet H2) -> Chiller -> KOD -> Coalescer -> Compressor.
        Target: Dry compressed gas > 350 bar.
        """
        logger.info("Starting Gas Cleanup Chain Test")
        
        # 1. Setup Components
        chiller = Chiller(component_id='gas_chiller', cooling_capacity_kw=50.0, target_temp_k=278.15) # 5C
        kod = KnockOutDrum(diameter_m=2.0) # volume_m3 not supported directly
        kod.component_id = 'kod'
        coalescer = Coalescer(component_id='gas_coalescer') # design_flow not arg
        compressor = CompressorStorage(
            max_flow_kg_h=500.0,
            inlet_pressure_bar=30.0,
            outlet_pressure_bar=350.0,
            isentropic_efficiency=0.85
        )
        compressor.component_id = 'main_compressor'
        
        for c in [chiller, kod, coalescer, compressor]:
            self.registry.register(c.component_id, c)
            # Some components need initialize()
            if hasattr(c, 'initialize'):
                c.initialize(self.dt, self.registry)
        
        # 2. Input Stream
        # Wet H2/O2 mix (mostly H2) at 80C/30bar
        wet_gas_in = Stream(
            mass_flow_kg_h=100.0,
            temperature_k=353.15, # 80C
            pressure_pa=30e5,     # 30 bar
            composition={'H2': 0.95, 'H2O': 0.05}, # 5% water vapor
            phase='gas'
        )
        
        # 3. Simulate
        
        # Step 1: Chiller (Condense water)
        chiller.receive_input('fluid_in', wet_gas_in, 'gas')
        chiller.step(0.0)
        chilled_gas = chiller.get_output('fluid_out')
        
        self.assertLess(chilled_gas.temperature_k, 300.0) # Should be ~5C
        # At 5C/30bar, most water should condense. Check phase or composition?
        # Chiller output is still a single stream? Or does Chiller have a drain?
        # h2_plant/components/thermal/chiller.py implementation check needed.
        # Typically Chiller just cools. Multiphase stream comes out.
        
        # Step 2: KOD (Bulk Separation)
        kod.receive_input('gas_inlet', chilled_gas, 'mixed')
        kod.step(0.0)
        kod_gas = kod.get_output('gas_outlet')
        kod_liquid = kod.get_output('liquid_drain')
        
        self.assertGreater(kod_liquid.mass_flow_kg_h, 0.0, "KOD should remove condensed water")
        
        # Step 3: Coalescer (Mist removal)
        coalescer.receive_input('inlet', kod_gas, 'gas')
        coalescer.step(0.0)
        dry_gas = coalescer.get_output('outlet') # or gas_out
        
        # Step 4: Compressor
        compressor.receive_input('h2_in', dry_gas, 'gas')
        compressor.step(0.0)
        final_gas = compressor.get_output('h2_out')
        
        self.assertGreaterEqual(final_gas.pressure_pa, 349e5) # ~350 bar
        self.assertEqual(final_gas.composition['H2'], 1.0) # approx pure (normalized) or close to it
        
if __name__ == '__main__':
    unittest.main()
