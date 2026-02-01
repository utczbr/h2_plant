
import logging
import sys
import csv
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.pathways.integrated_plant import IntegratedPlant

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def run_verification():
    print("Starting Integrated Plant 24-Hour Test...")
    
    # 1. Setup
    registry = ComponentRegistry()
    plant = IntegratedPlant()
    plant.set_component_id("integrated_plant")
    
    # 2. Initialize
    dt = 1.0 # 1 hour
    plant.initialize(dt, registry)
    print("Initialization successful.")
    
    # 3. Run Simulation
    print("\nRunning 24-hour simulation...")
    
    results = []
    
    for t in range(24):
        # Simulate some dynamic behavior
        # Hour 0-6: Low demand, full production
        # Hour 6-18: High demand
        # Hour 18-24: Low demand
        
        if 6 <= t < 18:
            plant.h2_demand_kg_h = 400.0 # High demand
        else:
            plant.h2_demand_kg_h = 100.0 # Low demand
            
        plant.step(float(t))
        state = plant.get_state()
        
        # Collect metrics
        metrics = {
            'Hour': t,
            'Water_Stored_kg': state['subsystems']['water_system']['summary']['stored_water_kg'],
            'H2_Prod_Rate_kg_h': state['production_rate_kg_h'],
            'H2_Demand_kg_h': plant.h2_demand_kg_h,
            'H2_Stored_Total_kg': state['total_h2_stored_kg'],
            'CO2_Stored_kg': state['co2_stored_kg'],
            'PEM_Prod_kg_h': state['subsystems']['pem_system']['summary']['h2_product_kg_h'],
            'SOEC_Prod_kg_h': state['subsystems']['soec_system']['summary']['h2_product_kg_h'],
            'ATR_Prod_kg_h': state['subsystems']['atr_system']['summary']['h2_product_kg_h']
        }
        results.append(metrics)
        
        # Print status every 6 hours
        if t % 6 == 0:
            print(f"Hour {t}: H2 Stored={metrics['H2_Stored_Total_kg']:.1f} kg, Demand={metrics['H2_Demand_kg_h']} kg/h")

    # 4. Save results
    csv_filename = 'simulation_results_24h.csv'
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
        
    print(f"\nSimulation complete. Results saved to {csv_filename}")
    
    # 5. Final Summary
    final = results[-1]
    print("\nFinal State (Hour 23):")
    print(f"  Total H2 Stored: {final['H2_Stored_Total_kg']:.2f} kg")
    print(f"  Total CO2 Captured: {final['CO2_Stored_kg']:.2f} kg")
    print(f"  Water Remaining: {final['Water_Stored_kg']:.2f} kg")

if __name__ == "__main__":
    run_verification()
