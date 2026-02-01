import sys
import os
import logging
from h2_plant.config.loader import ConfigLoader

# Setup logging
logging.basicConfig(level=logging.DEBUG)

print("Loading configuration...")
scenarios_dir = os.path.join(os.getcwd(), 'scenarios')
config_loader = ConfigLoader(scenarios_dir)
context = config_loader.load_context()
plant_config = context.topology.dict() if hasattr(context.topology, 'dict') else context.topology.__dict__

print("\n--- Checking YAML Config ---")
from h2_plant.core.graph_builder import PlantGraphBuilder

print("Config Keys:", plant_config.keys())
nodes_list = plant_config.get('nodes', [])

print("\n--- Checking YAML Config ---")
for node in nodes_list:
    if node['id'] == 'ATR_H2O_DryCooler':
        print(f"YAML Params: {node.get('params')}")

print("\n--- Building Graph ---")
builder = PlantGraphBuilder(context)
components = builder.build()

print("\n--- Checking Instantiated Component ---")
comp = components.get("ATR_H2O_DryCooler")
if comp:
    print(f"Component: {comp.component_id}")
    print(f"Target Temp (Attribute): {getattr(comp, 'target_outlet_temp_c', 'Not Found')}")
else:
    print("Component ATR_H2O_DryCooler NOT FOUND in components dict")
