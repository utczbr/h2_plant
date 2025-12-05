import os
import sys
import logging

# Add project root to path
sys.path.append(os.getcwd())

from h2_plant.orchestrator import Orchestrator

# Configure logging
logging.basicConfig(level=logging.INFO)

def test_full_simulation():
    print("\nStarting Full Simulation Test...")
    
    scenarios_dir = os.path.abspath("scenarios")
    orchestrator = Orchestrator(scenarios_dir)
    
    # Initialize components
    orchestrator.initialize_components()
    
    # Run simulation for 24 hours
    history = orchestrator.run_simulation(hours=24)
    
    if history is None:
        print("Simulation failed to return history.")
        exit(1)
        
    print(f"Simulation completed. Steps: {len(history['minute'])}")
    
    # Basic checks
    assert len(history['minute']) > 0
    assert len(history['h2_kg']) == len(history['minute'])
    
    total_h2 = sum(history['h2_kg'])
    print(f"Total H2 Produced: {total_h2:.2f} kg")
    
    # Check if we have any production (assuming wind > 0)
    if total_h2 > 0:
        print("Production verified.")
    else:
        print("Warning: No H2 produced (check wind data or logic).")

if __name__ == "__main__":
    test_full_simulation()
