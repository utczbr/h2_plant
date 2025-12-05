"""
Example simulation using the new modular architecture.

This demonstrates how to set up and run a simulation using the 
new consolidated architecture.
"""

from h2_plant.config.plant_builder import PlantBuilder
from h2_plant.simulation.engine import SimulationEngine


def main():
    """
    Main example demonstrating the new architecture.
    """
    print("Setting up hydrogen production simulation...")
    
    # Load plant configuration (using default configuration)
    plant = PlantBuilder.from_file("configs/plant_baseline.yaml")
    registry = plant.registry
    
    # Create simulation engine
    engine = SimulationEngine(
        registry=registry,
        config=plant.config.simulation
    )
    
    print("Starting simulation...")
    
    # Run simulation (first 24 hours for demo)
    results = engine.run(start_hour=0, end_hour=24)
    
    print("Simulation complete!")
    print(f"Results: {results}")


if __name__ == "__main__":
    main()