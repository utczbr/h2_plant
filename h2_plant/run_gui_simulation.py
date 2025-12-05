"""
GUI Simulation Runner

Standalone script to run H2 Plant simulations from a configuration file.
This script is called by the GUI when the user clicks "Run Simulation".

Usage:
    python run_gui_simulation.py <config_file_path>
"""

import sys
import argparse
import logging
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from h2_plant.config.plant_builder import PlantBuilder
from h2_plant.config.loaders import load_plant_config
from h2_plant.core.component_ids import ComponentID

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("SimulationRunner")

def run_simulation(config_path: str):
    """Run simulation from config file."""
    logger.info(f"Loading configuration from {config_path}...")
    
    try:
        # 1. Load Configuration
        config = load_plant_config(config_path)
        logger.info(f"Configuration loaded: {config.name}")
        
        # 2. Build Plant
        logger.info("Building plant...")
        builder = PlantBuilder(config)
        builder.build()
        registry = builder.registry
        
        # 3. Initialize
        logger.info("Initializing components...")
        dt_hours = config.simulation.timestep_hours
        registry.initialize_all(dt_hours)
        
        # 4. Run Simulation Loop
        duration_hours = config.simulation.duration_hours
        steps = int(duration_hours / dt_hours)
        
        logger.info(f"Starting simulation: {duration_hours} hours ({steps} steps)")
        
        start_time = time.time()
        
        # Track key metrics
        total_h2_produced = 0.0
        total_energy_consumed = 0.0
        
        # Get key components for monitoring
        electrolyzers = registry.get_by_type('production')
        tanks = registry.get_by_type('storage')
        
        print("\n" + "="*60)
        print(f"🚀 SIMULATION PROGRESS: {config.name}")
        print("="*60)
        
        for i in range(steps):
            t = i * dt_hours
            
            # Execute step for all components using registry's orchestration
            registry.step_all(t)
            
            # Update total from electrolyzer cumulative
            if i % (steps // 10) == 0:  # Progress every 10%
                progress = (i / steps) * 100
                # Get latest total from first electrolyzer
                if electrolyzers:
                    state = electrolyzers[0].get_state()
                    total_h2_produced = state.get('cumulative_h2_kg', 0.0)
                print(f"[{progress:3.0f}%] Hour {t:.1f}: H2 Produced = {total_h2_produced:.1f} kg")
        
        elapsed = time.time() - start_time
        
        # Get final total from electrolyzer
        if electrolyzers:
            final_state = electrolyzers[0].get_state()
            total_h2_produced = final_state.get('cumulative_h2_kg', 0.0)
        
        print("\n" + "="*60)
        print("✅ SIMULATION COMPLETE")
        print("="*60)
        print(f"Duration: {elapsed:.2f} seconds")
        print(f"Total H2 Produced: {total_h2_produced:.2f} kg")
        
        # Print Storage Levels
        for tank in tanks:
            state = tank.get_state()
            print(f"Storage ({tank.component_id}): {state.get('fill_level_kg', 0):.1f} kg / {state.get('capacity_kg', 0):.1f} kg")
            
    except Exception as e:
        logger.error(f"Simulation failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_gui_simulation.py <config_file>")
        sys.exit(1)
        
    config_file = sys.argv[1]
    run_simulation(config_file)
