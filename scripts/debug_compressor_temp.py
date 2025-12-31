
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
from h2_plant.core.stream import Stream

# Configure logging
logging.basicConfig(level=logging.WARNING)

from h2_plant.simulation.engine import SimulationEngine
from h2_plant.config.plant_config import SimulationConfig

def debug_compressor():
    scenarios_dir = "/home/stuart/Documentos/Planta Hidrogenio/scenarios"
    
    # 1. Load Topology
    print(f"Loading topology from {scenarios_dir}...")
    loader = ConfigLoader(scenarios_dir)
    context = loader.load_context()
    
    # 2. Build Components
    builder = PlantGraphBuilder(context)
    components = builder.build()
    
    # 3. Create Engine
    print("Initializing Engine...")
    registry = ComponentRegistry()
    for cid, comp in components.items():
        registry.register(cid, comp)
        
    config = SimulationConfig(
        start_hour=0,
        duration_hours=15/60, # 15 minutes to allow full propagation
        timestep_hours=1/60,
        checkpoint_interval_hours=0
    )
    
    # Get topology list for FlowNetwork
    # We can reconstruct generic connection list or ask builder
    # Builder doesn't return connection list easily in current version, 
    # but FlowNetwork needs it.
    # Actually PlantGraphBuilder builds the graph but doesn't return specific ConnectionConfig list easily
    # unless we parse context again.
    # However, we can use context.topology_config if available.
    
    # Wait, PlantGraphBuilder.build() configures the components but FlowNetwork needs
    # the list of connections to know what to transfer.
    # We must extract connections from context.
    
    topology_connections = []
    if hasattr(context, 'topology_config'):
        topology_connections = context.topology_config
    else:
        # Fallback: Parse from context.definitions['connections'] or similar?
        # Actually ConfigLoader.load_context() returns PlantContext.
        # PlantContext has topology field?
        pass

    # Since we don't have easy access to connections list without re-parsing,
    # let's try to pass the connections from the YAML directly if possible.
    # OR rely on FlowNetwork to auto-discover? No, FlowNetwork needs explicit list.
    
    # Let's read the YAML manually for connections to be safe
    import yaml
    with open(f"{scenarios_dir}/plant_topology.yaml") as f:
        raw = yaml.safe_load(f)
    
    from h2_plant.config.plant_config import ConnectionConfig
            
    conn_list = []
    for node in raw.get('nodes', []):
        src_id = node['id']
        for conn in node.get('connections', []):
            conn_list.append(ConnectionConfig(
                source_id=src_id,
                source_port=conn['source_port'],
                target_id=conn['target_name'],
                target_port=conn['target_port'],
                resource_type=conn.get('resource_type', 'stream')
            ))
            
    engine = SimulationEngine(
        registry=registry,
        config=config,
        topology=conn_list
    )
    
    engine.initialize()
    
    # 4. Inject Feed & Power (Pre-step callback to ensure continuous feed)
    def pre_step_injector(t):
        # Inject water to Makeup_Mixer_1
        mixer = registry.get('Makeup_Mixer_1')
        if mixer:
            water_in = Stream(
                mass_flow_kg_h=2000.0,
                temperature_k=298.15,
                pressure_pa=101325,
                composition={'H2O': 0.0, 'H2O_liq': 1.0},
                phase='liquid'
            )
            # Inject into a 'makeup' port or override input
            # MakeupMixer usually has 'water_in' or 'makeup_water'
            # If we inject to 'water_in', it might be from drain.
            # Let's inject to 'water_in' as if drain is providing it.
            mixer.receive_input('water_in', water_in, 'stream')
            
        # Inject Power to SOEC
        soec = registry.get('SOEC_Cluster')
        if soec:
             soec.receive_input('power_in', 2.4e6, 'electricity')
             
    engine.set_callbacks(pre_step=pre_step_injector)

    # 5. Run
    print("Running Simulation...")
    engine.run()

    # 6. Check Upstream States
    print("\n" + "="*50)
    for cid in ["Makeup_Mixer_1", "Feed_Pump", "SOEC_Steam_Generator", "SOEC_Cluster", "SOEC_H2_Compressor_S1"]:
        comp = components.get(cid)
        if comp:
            print(f"\n=== {cid} State ===")
            st = comp.get_state()
            if cid == "SOEC_Cluster":
                print(st) # Print all for SOEC
            else:
                for k in ['mode', 'power_kw', 'transfer_mass_kg', 'actual_mass_transferred_kg', 
                        'outlet_pressure_bar', 'outlet_temperature_c', 'mass_flow_kg_h', 'current_flow_kg_h']:
                    if k in st:
                        print(f"{k}: {st[k]}")
            
            try:
                out = comp.get_output('outlet') or comp.get_output('h2_out') or comp.get_output('fluid_out') or comp.get_output('water_out')
                if out:
                    print(f"Output Stream Flow: {out.mass_flow_kg_h:.4f} kg/h")
                    print(f"Output Stream Temp: {out.temperature_k - 273.15:.2f} C")
            except:
                pass

if __name__ == "__main__":
    debug_compressor()
