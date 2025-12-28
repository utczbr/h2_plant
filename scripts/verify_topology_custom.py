
import sys
import os

# Set paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(PROJECT_ROOT)

from h2_plant.core.graph_builder import PlantGraphBuilder
from h2_plant.config.loader import ConfigLoader
from h2_plant.core.component_registry import ComponentRegistry

def verify_yaml(yaml_path, scenarios_dir):
    print(f"Verifying YAML: {yaml_path}")
    print(f"Scenarios Dir: {scenarios_dir}")

    if not os.path.exists(yaml_path):
        print(f"Error: File not found at {yaml_path}")
        return False
    
    try:
        # Load Context
        loader = ConfigLoader(scenarios_dir)
        context = loader.load_context(yaml_path)
        print("Configuration Loaded Successfully.")

        # Build Graph
        registry = ComponentRegistry()
        # Note: PlantGraphBuilder takes context, not registry in __init__? 
        # Checking source: __init__(self, context: SimulationContext)
        # It doesn't seem to take registry. Registry might be handled differently or inside components.
        # But for graph building verification, we just need to ensure it instantiates components.
        
        builder = PlantGraphBuilder(context)
        components = builder.build()
        
        print(f"SUCCESS: Graph built with {len(components)} components.")
        
        # Verify specific nodes exist
        required_nodes = ["SOEC_Cluster", "O2_DryCooler", "O2_Chiller", "O2_KOD_2", "O2_Coalescer", "O2_Vent_Valve", "Drain_Mixer"]
        missing = []
        for node_id in required_nodes:
            if node_id not in components:
                missing.append(node_id)
        
        if missing:
            print(f"ERROR: Missing nodes in graph: {missing}")
            return False

        # Check Drain_Mixer connection count/logic if possible
        drain_mixer = components.get("Drain_Mixer")
        if drain_mixer:
             print(f"Drain Mixer found: {type(drain_mixer)}")
             # Logic verification would require inspecting internals, but existence is good first step.
        
        print("All required nodes present and instantiated.")
        return True

    except Exception as e:
        print(f"FAILURE: Exception during verification: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    yaml_file = "/home/stuart/Documentos/Planta Hidrogenio/scenarios/plant_topology.yaml"
    scenarios_dir = "/home/stuart/Documentos/Planta Hidrogenio/scenarios"
    success = verify_yaml(yaml_file, scenarios_dir)
    sys.exit(0 if success else 1)
