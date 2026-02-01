
import os
import sys
import json
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
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PowerExtractor")
logger.setLevel(logging.INFO)

def run_and_extract():
    # 1. Load Configuration (Topology + Simulation)
    scenarios_dir = "scenarios"
    topology_file = "plant_topology.yaml"
    
    logger.info(f"Loading topology: {topology_file}")
    loader = ConfigLoader(scenarios_dir)
    context = loader.load_context(topology_file=topology_file)
    
    # 2. Build Component Graph
    builder = PlantGraphBuilder(context)
    components = builder.build()
    
    # 3. Registry
    registry = ComponentRegistry()
    for cid, comp in components.items():
        registry.register(cid, comp)
        
    # 4. Load Data
    data_loader = EnergyPriceLoader(scenarios_dir)
    prices, wind = data_loader.load_data(
        context.simulation.energy_price_file,
        context.simulation.wind_data_file,
        context.simulation.duration_hours,
        context.simulation.timestep_hours
    )
    total_steps = len(prices)
    
    # 5. Engine Strategy
    dispatch_strategy = HybridArbitrageEngineStrategy()
    
    # 6. Initialize Engine (Minimal for running dispatch)
    # We can use the strategy directly if we don't need the full engine wrapper loop
    # But using full engine ensures correct data flow if topology connections matter (they do for simple loop)
    # Actually, HybridArbitrageEngineStrategy.decide_and_apply does a lot, but the standard SimulationEngine 
    # handles the component step() order. Let's use SimulationEngine to be safe.
    
    output_path = Path("simulation_output/extract_power_run")
    
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
        output_dir=output_path,
        topology=topology_connections,
        indexed_topology=[],
        dispatch_strategy=dispatch_strategy
    )
    
    engine.set_dispatch_data(prices, wind)
    engine.initialize()
    engine.initialize_dispatch_strategy(context=context, total_steps=total_steps, use_chunked_history=False)
    
    logger.info(f"Running simulation for {total_steps} steps...")
    engine.run()
    
    # 7. Extract Data
    history = dispatch_strategy.get_history()
    
    max_power_map = {}
    
    # Identify relevant keys
    for key, data in history.items():
        # Check for Power metrics
        # Standard: *_power_kw
        # DryCooler: *_fan_power_kw
        # Chiller: *_electrical_power_kw
        
        cid = None
        power_kw = 0.0
        
        if key.endswith('_power_kw'):
            cid = key.replace('_power_kw', '')
        elif key.endswith('_fan_power_kw'):
            cid = key.replace('_fan_power_kw', '')
        elif key.endswith('_electrical_power_kw'):
            cid = key.replace('_electrical_power_kw', '')
        
        if cid:
            # Check if this looks like a component ID (simple heuristic)
            # Or if it's aggregate like 'compressor_power_kw' (ignore aggregates?)
            # User said "ALL components", so specific IDs.
            # 'compressor_power_kw' is aggregate. Components are like 'HP_Compressor_S1'.
            
            # Filter out known aggregates or non-components
            if cid in ['compressor', 'bop']: continue
            
            max_val = np.max(data)
            max_power_map[cid] = max_val * 1000.0 # Convert kW to W
            
    # Add Special Components (SOEC / PEM)
    if 'P_soec_actual' in history:
        # P_soec_actual is in MW
        max_mw = np.max(history['P_soec_actual'])
        max_power_map['SOEC_System'] = max_mw * 1e6 # MW -> W
        
    if 'P_pem' in history:
        # P_pem is in MW
        max_mw = np.max(history['P_pem'])
        max_power_map['PEM_System'] = max_mw * 1e6 # MW -> W
        
    # Sort by Power
    sorted_map = dict(sorted(max_power_map.items(), key=lambda item: item[1], reverse=True))
    
    # 8. Save
    output_dir = Path("scenarios/Economics")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "max_power_consumption.json"
    
    with open(json_path, 'w') as f:
        json.dump(sorted_map, f, indent=4)
        
    logger.info(f"Saved max power data to {json_path}")
    
    # Print top 5
    print("\nTop 5 Consumers:")
    for k, v in list(sorted_map.items())[:5]:
        print(f"  {k}: {v:.2f} W")

if __name__ == "__main__":
    run_and_extract()
