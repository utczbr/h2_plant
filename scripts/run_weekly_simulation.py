import logging
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock visualization if needed (though engine tries to import it)
# We want the real dashboard generator if possible.
# Assuming h2_plant package is installed or in path.

from h2_plant.config.plant_builder import PlantBuilder
from h2_plant.simulation.engine import SimulationEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("simulation_weekly.log")
    ]
)
logger = logging.getLogger(__name__)

def run_weekly_simulation():
    config_path = Path("configs/h2plant_detailed.yaml")
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
        
    logger.info(f"Loading config from {config_path}")
    
    try:
        # Build plant
        builder = PlantBuilder.from_file(config_path)
        
        # Override duration for 1 week (168 hours)
        builder.config.simulation.duration_hours = 168
        logger.info("Overridden simulation duration to 168 hours (1 week)")
        
        # Build engine
        engine = SimulationEngine(
            registry=builder.registry,
            config=builder.config,
            output_dir=Path("simulation_output_weekly"),
            topology=builder.config.topology,
            indexed_topology=builder.config.indexed_topology
        )
        
        # Initialize
        engine.initialize()
        
        # Run
        logger.info("Starting 1-week simulation (10080 minutes)...")
        results = engine.run()
        
        logger.info("Simulation completed successfully.")
        logger.info(f"Results saved to: {engine.output_dir}")
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}", exc_info=True)
        sys.exit(1)
    
    # Export raw metrics for advanced visualization
    logger.info("Exporting raw metrics...")
    engine.monitoring.export_raw_metrics()
    
    # Generate advanced graphs
    logger.info("Generating advanced graphs...")
    import subprocess
    subprocess.run(["python3", "generate_advanced_graphs.py"], check=True)
    
    logger.info("Weekly simulation completed successfully.")

if __name__ == "__main__":
    run_weekly_simulation()
