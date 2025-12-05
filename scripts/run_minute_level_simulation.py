#!/usr/bin/env python3
"""
Run minute-level simulation for validation and production use.

This script runs the h2_plant system with minute-level timesteps
to match the reference manager.py implementation exactly.
"""

import logging
import time
import sys
from pathlib import Path

# Add h2_plant to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from h2_plant.config.plant_builder import PlantBuilder
from h2_plant.simulation.engine import SimulationEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('simulation_minute_level.log')
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Run minute-level simulation."""
    
    config_path = "configs/plant_pem_soec_minute_level.yaml"
    
    print("="*70)
    print("H2 PLANT - MINUTE-LEVEL SIMULATION")
    print("="*70)
    print(f"Configuration: {config_path}")
    print(f"Timestep: 1 minute (dt = 1/60 hour)")
    print(f"Duration: 1 year = 8760 hours = 525,600 minutes")
    print("="*70)
    print("\nThis simulation matches the reference manager.py implementation.")
    print("Expected runtime: 10-20 minutes for full year\n")
    
    # Validate config exists
    if not Path(config_path).exists():
        logger.error(f"Config file not found: {config_path}")
        return 1
    
    try:
        # Build plant
        print("\n[1/3] Building plant components...")
        builder = PlantBuilder.from_file(config_path)
        # Note: from_file already calls build() internally
        registry = builder.registry
        
        print(f"      ✓ {registry.get_component_count()} components registered")
        
        # Create simulation engine
        print("\n[2/3] Initializing simulation engine...")
        logger.info("Creating SimulationEngine")
        engine = SimulationEngine(
            registry=builder.registry,
            config=builder.config.simulation,
            topology=getattr(builder.config, 'topology', []),
            indexed_topology=getattr(builder.config, 'indexed_topology', [])
        )
        
        total_hours = builder.config.simulation.duration_hours
        total_steps = int(total_hours * 60)  # minutes
        dt_hours = builder.config.simulation.timestep_hours
        
        print(f"      - Total steps: {total_steps:,}")
        print(f"      - Timestep: {dt_hours:.6f} hours ({dt_hours*60:.2f} minutes)")
        print(f"      - Duration: {total_hours:.0f} hours")
        
        # Run simulation
        print("\n[3/3] Running simulation...")
        print("Progress will be reported by the engine...")
        print("(This may take 10-20 minutes for a full year)\n")
        
        start_time = time.time()
        results = engine.run()
        elapsed_time = time.time() - start_time
        
        # Summary
        print("\n" + "="*70)
        print("✅ SIMULATION COMPLETE")
        print("="*70)
        print(f"Runtime: {elapsed_time/60:.1f} minutes ({elapsed_time:.0f} seconds)")
        print(f"Steps completed: {total_steps:,}")
        print(f"Avg speed: {total_steps/elapsed_time:.0f} steps/second")
        print("="*70)
        
        # Results already saved by engine
        print("\n✓ Results saved by simulation engine")
        
        print("\n" + "="*70)
        print("ALL DONE!")
        print("="*70)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠ Simulation interrupted by user")
        logger.warning("Simulation interrupted")
        return 130
        
    except Exception as e:
        print(f"\n\n❌ Simulation failed: {e}")
        logger.exception("Simulation failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
