
import sys
import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from h2_plant.config.loader import ConfigLoader
from h2_plant.core.graph_builder import PlantGraphBuilder
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.simulation.engine import SimulationEngine
from h2_plant.control.engine_dispatch import HybridArbitrageEngineStrategy
from h2_plant.data.price_loader import EnergyPriceLoader

# Setup Logging
logging.basicConfig(level=logging.WARNING, format='%(message)s')
logger = logging.getLogger("ZeroPowerDiag")

def diagnose_component(comp):
    print(f"\n--- DIAGNOSIS: {comp.component_id} ({comp.__class__.__name__}) ---")
    
    state = comp.get_state()
    
    # Power
    power_kw = state.get('power_kw', 0.0)
    elec_kw = state.get('electrical_power_kw', 0.0)
    print(f"Power: {power_kw:.2f} kW (Elec: {elec_kw:.2f} kW)")
    
    # Flow In
    inlet = getattr(comp, 'inlet_stream', None) or getattr(comp, '_inlet_stream', None) or getattr(comp, 'water_inlet', None)
    
    if inlet:
        print(f"  Inlet Stream:")
        print(f"    Flow: {inlet.mass_flow_kg_h:.2f} kg/h")
        print(f"    Press: {inlet.pressure_pa/1e5:.2f} bar")
    else:
        print("  Inlet Stream: None")

    # Flow Out
    if hasattr(comp, 'outlet_stream') and comp.outlet_stream:
        print(f"  Outlet Stream:")
        print(f"    Flow: {comp.outlet_stream.mass_flow_kg_h:.2f} kg/h")
        print(f"    Press: {comp.outlet_stream.pressure_pa/1e5:.2f} bar")
    
    return power_kw

def run_diagnosis():
    # 1. Load Configuration
    scenarios_dir = "scenarios"
    topology_file = "plant_topology.yaml"
    
    logger.info("Loading plant topology...")
    loader = ConfigLoader(scenarios_dir)
    context = loader.load_context(topology_file=topology_file)
    
    builder = PlantGraphBuilder(context)
    components = builder.build()
    
    registry = ComponentRegistry()
    for cid, comp in components.items():
        registry.register(cid, comp)
        
    # 2. Setup Engine
    data_loader = EnergyPriceLoader(scenarios_dir)
    prices, wind = data_loader.load_data(
        context.simulation.energy_price_file,
        context.simulation.wind_data_file,
        context.simulation.duration_hours,
        context.simulation.timestep_hours
    )
    
    # Connection Config derivation (Simplified for script)
    from h2_plant.config.plant_config import ConnectionConfig
    topology_connections = []
    if context.topology and context.topology.nodes:
        for node in context.topology.nodes:
            for conn in node.connections:
                topology_connections.append(ConnectionConfig(
                    source_id=node.id,
                    source_port=conn.source_port,
                    target_id=conn.target_name,
                    target_port=conn.target_port,
                    resource_type=conn.resource_type
                ))

    engine = SimulationEngine(
        registry=registry,
        config=context.simulation,
        output_dir=Path("simulation_output/debug_zero_power"),
        topology=topology_connections,
        indexed_topology=[],
        dispatch_strategy=HybridArbitrageEngineStrategy()
    )
    engine.set_dispatch_data(prices, wind)

    # Initialize Dispatch Strategy
    total_steps = int(context.simulation.duration_hours / context.simulation.timestep_hours)
    engine.initialize_dispatch_strategy(context, total_steps=total_steps)

    engine.initialize()
    
    # 3. Run for 1 hour (60 steps) to establish flow
    logging.info("Running simulation for 1 hour...")
    engine.run(start_hour=0, end_hour=1)
    
    # Check targets
    targets = [
        "ATR_Feed_Pump",
        "SOEC_Drain_Pump",
        "SOEC_Steam_Compressor_2",
        "ATR_O2_Compressor",
        "O2_Backup_Supply",
        "SOEC_Steam_Drycooler"
    ]
    
    for cid in targets:
        comp = components.get(cid)
        if comp:
            diagnose_component(comp)
        else:
            print(f"Component {cid} NOT FOUND")

if __name__ == "__main__":
    run_diagnosis()
