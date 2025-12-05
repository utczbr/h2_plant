"""
Test script for configurable component connections.
"""

import logging
import sys
import os
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from h2_plant.simulation.engine import SimulationEngine
from h2_plant.config.plant_config import PlantConfig
from h2_plant.core.stream import Stream

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from h2_plant.config.plant_builder import PlantBuilder

def test_flow_network():
    logger.info("Starting FlowNetwork verification...")
    
    config_path = os.path.join(project_root, "h2plant.yaml")
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        return

    # Use PlantBuilder to load config and build registry
    builder = PlantBuilder.from_file(config_path)
    
    # Initialize Engine with registry and config from builder
    engine = SimulationEngine(builder.registry, builder.config)
    engine.initialize()
    
    logger.info("Engine initialized. Running simulation steps...")
    
    # 2. Run a few steps
    # We need to ensure inputs are present.
    # Grid provides electricity (infinite).
    # Water supply provides water (infinite).
    
    # Check initial state
    pem = engine.registry.get("pem_stacks")
    tank = engine.registry.get("lp_storage")
    
    logger.info(f"Initial Tank Level: {tank.get_total_mass()} kg")
    
    # Run step 1
    engine.step(0)
    
    # Check if Electrolyzer produced H2
    # And if it flowed to Tank (via Chiller -> Separation -> PSA -> Compressor -> Chiller -> Tank)
    # This is a long chain.
    
    logger.info(f"Step 0 Complete.")
    logger.info(f"PEM H2 Output: {pem.h2_stream.mass_flow_kg_h} kg/h")
    logger.info(f"Tank Level: {tank.get_total_mass()} kg")
    
    # Run more steps
    for i in range(1, 5):
        engine.step(i)
        logger.info(f"Step {i} Complete. Tank Level: {tank.get_total_mass()} kg")
        
    if tank.get_total_mass() > 0:
        logger.info("SUCCESS: Mass reached storage tank!")
    else:
        logger.error("FAILURE: Tank remains empty.")

if __name__ == "__main__":
    test_flow_network()
