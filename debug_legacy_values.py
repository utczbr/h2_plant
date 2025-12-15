
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import shutil

# Add project root to path
sys.path.append(os.path.abspath(os.curdir))

from h2_plant.config.loader import ConfigLoader
from h2_plant.core.graph_builder import PlantGraphBuilder
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.simulation.engine import SimulationEngine
from h2_plant.control.engine_dispatch import HybridArbitrageEngineStrategy
from h2_plant.data.price_loader import EnergyPriceLoader
from h2_plant.gui.plotting.legacy_adapter import LegacyDataAdapter
from h2_plant.config.plant_config import ConnectionConfig

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def run_debug_simulation():
    print("\n=== Running Debug Simulation (1 Hour) ===")
    
    topology_file = "scenarios/topologies/topology_h2_compression_test.yaml"
    target_topology = "scenarios/plant_topology.yaml"
    shutil.copy(topology_file, target_topology)
    
    loader = ConfigLoader("scenarios")
    context = loader.load_context()
    
    # OVERRIDE duration for debug
    context.simulation.duration_hours = 1
    hours = 1
    
    # Build Graph
    builder = PlantGraphBuilder(context)
    components = builder.build()
    
    # Registry
    registry = ComponentRegistry()
    for cid, comp in components.items():
        registry.register(cid, comp)
        
    # Dispatch Data
    data_loader = EnergyPriceLoader("scenarios")
    prices, wind = data_loader.load_data(
        context.simulation.energy_price_file,
        context.simulation.wind_data_file,
        hours,
        context.simulation.timestep_hours
    )
    
    # Strategy
    dispatch_strategy = HybridArbitrageEngineStrategy()
    
    # Connections
    connections = []
    if context.topology and context.topology.nodes:
        for node in context.topology.nodes:
            source_id = node.id
            for conn in node.connections:
                connections.append(ConnectionConfig(
                    source_id=source_id,
                    source_port=conn.source_port,
                    target_id=conn.target_name,
                    target_port=conn.target_port,
                    resource_type=conn.resource_type
                ))

    # Engine
    out_dir = Path("simulation_output/DEBUG")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    engine = SimulationEngine(
        registry=registry,
        config=context.simulation,
        output_dir=out_dir,
        topology=connections,
        dispatch_strategy=dispatch_strategy
    )
    
    # Run
    engine.initialize()
    engine.set_dispatch_data(prices, wind)
    engine.initialize_dispatch_strategy(context, len(prices))
    engine.run(end_hour=hours)
    
    # Generate Legacy Data
    adapter = LegacyDataAdapter(registry)
    df_h2, df_o2 = adapter.generate_dataframes()
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.options.display.float_format = '{:,.8f}'.format

    cols_h2 = ['Componente', 'T_C', 'P_bar', 'w_H2O', 'H_mix_J_kg']  # Core properties
    cols_energy = ['Componente', 'Q_dot_fluxo_W', 'W_dot_comp_W']   # Energy
    cols_removal = ['Componente', 'Agua_Condensada_kg_s']           # Water Removal

    print("\n=== Legacy Data Inspection ===")
    
    if not df_h2.empty:
        print("\n--- H2 Stream Data (Last Step) ---")
        
        print("\n[Water Removal Data]")
        print(df_h2[cols_removal])
        
        print("\n[Energy Flux Data]")
        print(df_h2[cols_energy])
        
        print("\n[Properties Data]")
        print(df_h2[cols_h2])
    else:
        print("df_h2 is empty!")

    if not df_o2.empty:
        print("\n--- O2 Stream Data (Last Step) ---")
        print(df_o2)
    else:
        print("\n--- O2 Stream Data (Last Step) ---")
        print("Empty DataFrame")
        print(f"Columns: {df_o2.columns.tolist()}")
        print(f"Index: {df_o2.index.tolist()}")

if __name__ == "__main__":
    run_debug_simulation()
