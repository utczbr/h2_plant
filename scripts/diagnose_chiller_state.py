
import os
import sys
import logging
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
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("ChillerDiagnosis")

def print_stream(name, stream):
    if not stream:
        print(f"  {name}: <None>")
        return
        
    print(f"  {name}:")
    print(f"    Temp: {stream.temperature_k:.2f} K ({stream.temperature_k - 273.15:.2f} C)")
    print(f"    Press: {stream.pressure_pa/1e5:.2f} bar")
    print(f"    Flow: {stream.mass_flow_kg_h:.2f} kg/h")
    
    # Composition
    print(f"    Composition (Mass Frac):")
    for k, v in stream.composition.items():
        if v > 1e-4:
            print(f"      {k}: {v:.4f}")
    
    # Extra liquid?
    if hasattr(stream, 'extra') and stream.extra:
        print(f"    Extra: {stream.extra}")

def diagnose():
    scenarios_dir = "scenarios"
    topology_file = "plant_topology.yaml"
    
    loader = ConfigLoader(scenarios_dir)
    context = loader.load_context(topology_file=topology_file)
    
    builder = PlantGraphBuilder(context)
    components = builder.build()
    
    registry = ComponentRegistry()
    for cid, comp in components.items():
        registry.register(cid, comp)
        
    # Load Data (Minimal)
    data_loader = EnergyPriceLoader(scenarios_dir)
    prices, wind = data_loader.load_data(
        context.simulation.energy_price_file,
        context.simulation.wind_data_file,
        24, # Run for 24 hours to let dynamics settle if any
        context.simulation.timestep_hours
    )
    
    dispatch_strategy = HybridArbitrageEngineStrategy()
    
    # Setup Engine
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
        output_dir=Path("simulation_output/debug_run"),
        topology=topology_connections,
        indexed_topology=[],
        dispatch_strategy=dispatch_strategy
    )
    
    engine.set_dispatch_data(prices, wind)
    engine.initialize()
    engine.initialize_dispatch_strategy(context=context, total_steps=len(prices), use_chunked_history=False)
    
    print("\n=== Running Simulation (12 Hours) ===")
    dt = context.simulation.timestep_hours
    # Run for 12 hours
    engine.run(start_hour=0.0, end_hour=12.0)
    
    # Inspect Target Chillers
    targets = ['SOEC_H2_Chiller_1', 'ATR_Chiller_1', 'SOEC_H2_Chiller_2']
    
    for cid in targets:
        comp = components.get(cid)
        if not comp:
            print(f"\n--- {cid} NOT FOUND ---")
            continue
            
        print(f"\n--- DIAGNOSIS: {cid} ---")
        state = comp.get_state()
        print(f"Power: {state['electrical_power_kw']:.2f} kW")
        print(f"Cooling Load: {state['cooling_load_kw']:.2f} kW")
        print(f"Sensible: {state.get('sensible_heat_kw', 0):.2f} kW")
        print(f"Latent: {state.get('latent_heat_kw', 0):.2f} kW")
        print(f"COP: {state.get('cop', 0):.2f}")
        
        print_stream("Inlet Stream", comp.inlet_stream)
        print_stream("Outlet Stream", comp.outlet_stream)
        
        if cid == 'SOEC_H2_Chiller_1':
            # Check SOEC Outlet
            soec = components.get('SOEC_Cluster')
            if soec:
                print(f"\n  SOEC_Cluster Outlet (h2_out):")
                print_stream("    Output", soec.get_output('h2_out'))

            upstream_id = 'SOEC_H2_KOD_1'
            upstream = components.get(upstream_id)
            if upstream:
                print(f"\n  Upstream ({upstream_id}) Outlet:")
                # KOD has 'gas_outlet'
                # But we can check internal state or just look at Chiller Input (which IS KOD output)
                # Let's check the KOD DRAIN
                print(f"  {upstream_id} Water Removed: {getattr(upstream, 'water_removed_kg_h', 'N/A')} kg/h")

if __name__ == "__main__":
    diagnose()
