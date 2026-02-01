
import logging
import sys
import numpy as np
import pandas as pd
from h2_plant.core.graph_builder import PlantGraphBuilder
from h2_plant.config.models import SimulationContext
from h2_plant.config.loader import ConfigLoader
from h2_plant.control.engine_dispatch import HybridArbitrageEngineStrategy

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def verify_pump_power_recording():
    logger.info("Starting verification of Pump Power Recording...")
    
    try:
        # 1. Load context and build plant
        loader = ConfigLoader("scenarios")
        context = loader.load_context("plant_topology.yaml")
        
        builder = PlantGraphBuilder(context)
        # 2. Build graph (returns dict of components)
        components = builder.build()
        
        # 3. Create Registry wrapper expected by Engine
        from h2_plant.core.component_registry import ComponentRegistry
        component_registry = ComponentRegistry()
        for cid, comp in components.items():
            component_registry.register(cid, comp)
            
        logger.info(f"✓ Built registry with {len(components)} components")
        
        # 2. Get Pump Reference
        pump_id = "ATR_Drain_Pump_1"
        pump = component_registry.get(pump_id)
        assert pump is not None, f"Pump {pump_id} not found!"
        logger.info(f"✓ Found pump {pump_id}")
        
        # 3. initialize Engine
        total_steps = 10
        engine = HybridArbitrageEngineStrategy()
        engine.initialize(component_registry, context, total_steps)
        
        # 4. Check if pump has a history slot allocated
        history = engine._history
        # Direct lookup or via recorders?
        # Check if column exists
        power_col = f"{pump_id}_power_kw"
        
        # Note: HistoryDictProxy might hide keys until accessed if using chunked, 
        # but here we use default unless configured otherwise.
        # Let's inspect _recorders to see if it was picked up.
        pump_recorder = None
        for rec in engine._recorders:
            if rec.component.component_id == pump_id:
                pump_recorder = rec
                break
        
        assert pump_recorder is not None, f"Pump {pump_id} was NOT registered by the engine!"
        logger.info(f"✓ Pump {pump_id} registered in engine recorders.")
        
        # 5. Simulate One Step
        # Mock inputs
        prices = np.zeros(total_steps)
        wind = np.zeros(total_steps)
        t = 0.0
        
        # Run step
        pump.initialize(dt=1/3600, registry=component_registry)
        engine.decide_and_apply(t, prices, wind)
        
        # Apply strict flow to pump to force power consumption
        from h2_plant.core.stream import Stream
        input_stream = Stream(
            mass_flow_kg_h=1000.0,
            temperature_k=300.0,
            pressure_pa=100000.0,
            composition={'H2O': 1.0},
            phase='liquid'
        )
        pump.receive_input('water_in', input_stream, 'water')
        
        # Mock step execution
        pump.step(t)
        
        # Post-step record
        engine.record_post_step()
        
        # 6. Verify Data
        history_data = engine.get_history()
        recorded_power = history_data[f"{pump_id}_power_kw"][0]
        logger.info(f"Recorded Power for {pump_id}: {recorded_power} kW")
        
        assert recorded_power > 0.0, "Recorded power is ZERO! Expected > 0."
        
        logger.info("SUCCESS: Pump power is being recorded.")
        return True

    except Exception as e:
        logger.error(f"Verification FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = verify_pump_power_recording()
    sys.exit(0 if success else 1)
