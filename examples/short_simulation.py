
import logging
from pathlib import Path
from h2_plant.config.plant_builder import PlantBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    print("Setting up short hydrogen production simulation...")
    
    # Load plant from baseline configuration
    plant = PlantBuilder.from_file("configs/plant_baseline.yaml")
    
    from h2_plant.simulation.engine import SimulationEngine
    
    # Create simulation engine
    engine = SimulationEngine(plant.registry, plant.config.simulation)
    
    print("Starting simulation (24 hours)...")
    # Run simulation for only 24 hours
    results = engine.run(start_hour=0, end_hour=24)
    
    print("Simulation complete!")
    print(f"Total H2 Production: {results['metrics']['total_production_kg']:.2f} kg")
    print(f"Total Cost: ${results['metrics']['total_cost']:.2f}")

if __name__ == "__main__":
    main()
