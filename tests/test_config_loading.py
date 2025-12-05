import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from h2_plant.config.loader import ConfigLoader

def test_config_loading():
    print("Testing Config Loading...")
    loader = ConfigLoader("scenarios")
    
    try:
        context = loader.load_context()
        print("✅ Context loaded successfully!")
        print(f"Physics PEM Max Power: {context.physics.pem_system.max_power_mw} MW")
        print(f"Topology Nodes: {len(context.topology.nodes)}")
        print(f"Simulation Duration: {context.simulation.duration_hours} hours")
        print(f"Economics H2 Price: {context.economics.h2_price_eur_kg} EUR/kg")
        
    except Exception as e:
        print(f"❌ Failed to load context: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_config_loading()
