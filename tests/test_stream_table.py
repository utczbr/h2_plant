
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from h2_plant.config.loader import ConfigLoader
from h2_plant.core.graph_builder import PlantGraphBuilder
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.reporting.stream_table import print_stream_summary_table
from h2_plant.core.stream import Stream

def main():
    base_path = Path("/home/stuart/Documentos/Planta Hidrogenio")
    print("Loading configuration...")
    loader = ConfigLoader(str(base_path / "scenarios"))
    context = loader.load_context(
        topology_file="plant_topology.yaml"
    )
    
    print("Building plant graph...")
    builder = PlantGraphBuilder(context)
    components = builder.build()
    
    # Initialize components to ensure they have streams (even if empty)
    registry = ComponentRegistry()
    for comp_id, comp in components.items():
        registry.register(comp_id, comp)
        try:
            comp.initialize(1.0/60.0, registry)
            # Create dummy output streams for the components we care about
            # so they appear in the table
            if comp_id in ['SOEC_H2_Boiler', 'PEM_H2_Cyclone_3', 'PEM_H2_Coalescer_2']:
                dummy_stream = Stream(
                    mass_flow_kg_h=100.0,
                    temperature_k=298.15,
                    pressure_pa=3000000.0,
                    composition={'H2': 1.0},
                    phase='gas'
                )
                # Manually inject into output buffer based on component internals
                if hasattr(comp, '_outlet_stream'): # HydrogenMultiCyclone
                     comp._outlet_stream = dummy_stream
                elif hasattr(comp, 'output_stream'): # Coalescer
                     comp.output_stream = dummy_stream
                elif hasattr(comp, '_output_stream'): # ElectricBoiler
                     comp._output_stream = dummy_stream
                elif hasattr(comp, 'outlet_stream'):
                     comp.outlet_stream = dummy_stream
                
                # Also generic "step" might produce it if we feed input
                # But manual injection is faster for table test
        except:
            pass
            
    # Mock topo order (just sort keys)
    topo_order = list(components.keys())
    
    print("\n--- Generating Stream Table ---")
    print_stream_summary_table(components, topo_order)

if __name__ == "__main__":
    main()
