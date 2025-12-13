"""
GUI Simulation Runner.

Standalone script to run H2 Plant simulations from a configuration file.
This script is called by the GUI when the user clicks "Run Simulation".

Execution Flow:
    1. Load configuration from YAML/JSON file.
    2. Build plant component graph via PlantBuilder.
    3. Initialize all components with timestep.
    4. Execute simulation loop with progress reporting.
    5. Display final summary with production metrics.

Usage:
    python run_gui_simulation.py <config_file_path>
"""

import sys
import argparse
import logging
from pathlib import Path
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from h2_plant.config.plant_builder import PlantBuilder
from h2_plant.config.loaders import load_plant_config
from h2_plant.core.component_ids import ComponentID

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("SimulationRunner")


def run_simulation(config_path: str) -> None:
    """
    Run simulation from configuration file with progress reporting.

    Executes a complete simulation loop using the registry's step_all()
    method, providing periodic progress updates to stdout.

    Args:
        config_path (str): Path to plant configuration YAML/JSON file.

    Raises:
        SystemExit: On simulation failure (exit code 1).
    """
    logger.info(f"Loading configuration from {config_path}...")

    try:
        # Load and validate configuration
        config = load_plant_config(config_path)
        logger.info(f"Configuration loaded: {config.name}")

        # Build component graph
        logger.info("Building plant...")
        builder = PlantBuilder(config)
        builder.build()
        registry = builder.registry

        # Initialize all components with configured timestep
        logger.info("Initializing components...")
        dt_hours = config.simulation.timestep_hours
        registry.initialize_all(dt_hours)

        # Calculate simulation parameters
        duration_hours = config.simulation.duration_hours
        steps = int(duration_hours / dt_hours)

        logger.info(f"Starting simulation: {duration_hours} hours ({steps} steps)")

        start_time = time.time()

        total_h2_produced = 0.0
        total_energy_consumed = 0.0

        # Get key components for progress monitoring
        electrolyzers = registry.get_by_type('production')
        tanks = registry.get_by_type('storage')

        print("\n" + "="*60)
        print(f"🚀 SIMULATION PROGRESS: {config.name}")
        print("="*60)

        # Main simulation loop
        for i in range(steps):
            t = i * dt_hours

            # Execute timestep for all components
            registry.step_all(t)

            # Progress reporting every 10%
            if i % (steps // 10) == 0:
                progress = (i / steps) * 100
                if electrolyzers:
                    state = electrolyzers[0].get_state()
                    total_h2_produced = state.get('cumulative_h2_kg', 0.0)
                print(f"[{progress:3.0f}%] Hour {t:.1f}: H2 Produced = {total_h2_produced:.1f} kg")

        elapsed = time.time() - start_time

        # Get final production total
        if electrolyzers:
            final_state = electrolyzers[0].get_state()
            total_h2_produced = final_state.get('cumulative_h2_kg', 0.0)

        # Print summary
        print("\n" + "="*60)
        print("✅ SIMULATION COMPLETE")
        print("="*60)
        print(f"Duration: {elapsed:.2f} seconds")
        print(f"Total H2 Produced: {total_h2_produced:.2f} kg")

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
