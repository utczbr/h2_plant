
import logging
import sys
from h2_plant.core.graph_builder import PlantGraphBuilder
from h2_plant.config.models import SimulationContext, SimulationConfig, PhysicsConfig
from h2_plant.config.loader import ConfigLoader

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def verify_atr_drain_pumps():
    logger.info("Starting topology verification for ATR Drain Pumps...")
    
    try:
        # Load configuration
        loader = ConfigLoader("scenarios")
        context = loader.load_context("plant_topology.yaml")
        
        # Build graph
        builder = PlantGraphBuilder(context)
        components = builder.build()
        
        # 1. Verify Pumps Exist
        assert "ATR_Drain_Pump_1" in components, "ATR_Drain_Pump_1 not found in components!"
        assert "ATR_Drain_Pump_2" in components, "ATR_Drain_Pump_2 not found in components!"
        logger.info("✓ Pumps found in component registry.")
        
        # 2. Verify Pump Types
        pump1 = components["ATR_Drain_Pump_1"]
        pump2 = components["ATR_Drain_Pump_2"]
        from h2_plant.components.water.water_pump import WaterPumpThermodynamic
        assert isinstance(pump1, WaterPumpThermodynamic), "ATR_Drain_Pump_1 is not WaterPumpThermodynamic!"
        assert isinstance(pump2, WaterPumpThermodynamic), "ATR_Drain_Pump_2 is not WaterPumpThermodynamic!"
        logger.info("✓ Pump types verified.")
        
        # 3. Verify Pump Configuration
        assert pump1.target_pressure_pa == 1500000.0, f"Pump 1 target pressure wrong: {pump1.target_pressure_pa}"
        assert pump2.target_pressure_pa == 1500000.0, f"Pump 2 target pressure wrong: {pump2.target_pressure_pa}"
        logger.info("✓ Pump target pressures verified.")
        
        # 4. Verify Mixer Source IDs
        mixer = components["ATR_Drain_Mixer"]
        from h2_plant.components.water.drain_recorder_mixer import DrainRecorderMixer
        assert isinstance(mixer, DrainRecorderMixer), "ATR_Drain_Mixer is not DrainRecorderMixer!"
        expected_sources = ["ATR_Drain_Pump_1", "ATR_Drain_Pump_2"]
        assert sorted(mixer.source_ids) == sorted(expected_sources), f"Mixer source_ids mismatch! Got: {mixer.source_ids}"
        logger.info("✓ Mixer source inputs verified.")
        
        # 5. Verify Cyclone Connections (Static Check from Topology)
        # We need to look at the raw topology nodes for this, or check the graph logic if we had a full graph traverser.
        # Here we just rely on component existence and mixer configuration as a proxy for successful loading.
        # But we can check if the components have valid ports.
        assert "water_in" in pump1.get_ports(), "Pump 1 missing 'water_in' port"
        assert "water_out" in pump1.get_ports(), "Pump 1 missing 'water_out' port"
        logger.info("✓ Pump ports verified.")

        logger.info("SUCCESS: All topology checks passed.")
        return True
        
    except Exception as e:
        logger.error(f"Verification FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = verify_atr_drain_pumps()
    sys.exit(0 if success else 1)
