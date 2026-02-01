
import sys
import os
from pathlib import Path
import logging

# Add repo root to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from h2_plant.config.loader import ConfigLoader
from h2_plant.core.graph_builder import PlantGraphBuilder
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.config.plant_config import ConnectionConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("validate_topology")

def validate_topology(scenarios_dir: str):
    print(f"Validating topology in: {scenarios_dir}")
    
    # 1. Load Configuration
    try:
        loader = ConfigLoader(scenarios_dir)
        context = loader.load_context()
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return False

    # 2. Build Components
    try:
        builder = PlantGraphBuilder(context)
        components = builder.build()
        print(f"✅ Components built successfully ({len(components)} components)")
    except Exception as e:
        logger.error(f"Failed to build components: {e}")
        return False

    # 3. Register Components
    registry = ComponentRegistry()
    for cid, comp in components.items():
        registry.register(cid, comp)

    # 4. Validate Connections
    errors = []
    warnings = []
    
    # Map of Component ID -> Ports
    comp_ports = {}
    for cid, comp in components.items():
        comp_ports[cid] = comp.get_ports()

    # Iterate Connections
    for node in context.topology.nodes:
        source_id = node.id
        
        # Check Source Existence
        if source_id not in components:
            errors.append(f"Node definition for '{source_id}' found, but component not built.")
            continue
            
        source_ports = comp_ports.get(source_id, {})

        for conn in node.connections:
            target_id = conn.target_name
            target_port = conn.target_port
            source_port = conn.source_port
            
            # Check Target Existence
            if target_id not in components:
                errors.append(f"❌ Broken Link: '{source_id}' -> '{target_id}' (Target '{target_id}' does not exist)")
                continue

            target_ports = comp_ports.get(target_id, {})
            
            # Check Valid Source Port
            if source_port not in source_ports:
                errors.append(f"❌ Invalid Source Port: '{source_id}'.{source_port} does not exist. Available: {list(source_ports.keys())}")
            elif source_ports[source_port].get('type') != 'output':
                 errors.append(f"❌ Wrong Direction: '{source_id}'.{source_port} is not an OUTPUT port.")

            # Check Valid Target Port
            if target_port not in target_ports:
                # Fuzzy matching hint
                import difflib
                matches = difflib.get_close_matches(target_port, target_ports.keys())
                hint = f" Did you mean '{matches[0]}'?" if matches else f" Available: {list(target_ports.keys())}"
                errors.append(f"❌ Invalid Target Port: '{target_id}'.{target_port} does not exist.{hint}")
            elif target_ports[target_port].get('type') != 'input':
                 errors.append(f"❌ Wrong Direction: '{target_id}'.{target_port} is not an INPUT port.")

    # 5. Report Results
    print("\nValidation Results:")
    if not errors:
        print("✅ No topology errors found.")
        return True
    else:
        for err in errors:
            print(err)
        print(f"\nFound {len(errors)} errors.")
        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("scenarios_dir", help="Path to scenarios directory")
    args = parser.parse_args()
    
    success = validate_topology(args.scenarios_dir)
    sys.exit(0 if success else 1)
