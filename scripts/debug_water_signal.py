"""
Debug script to trace signal flow in the water supply chain.
Run with: python3 scripts/debug_water_signal.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

# Set up logging before imports
import logging
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')

from h2_plant.config.loader import ConfigLoader
from h2_plant.core.graph_builder import PlantGraphBuilder
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.simulation.engine import SimulationEngine


def main():
    scenarios_dir = Path("scenarios")
    
    print("=== Loading Configuration ===")
    loader = ConfigLoader(scenarios_dir)
    context = loader.load_context()
    
    print("=== Building Components ===")
    builder = PlantGraphBuilder(context)
    components = builder.build()
    
    # Create registry
    registry = ComponentRegistry()
    for comp_id, comp in components.items():
        registry.register(comp_id, comp)
    
    # Get our key components
    tank = registry.get("UltraPure_Tank")
    source = registry.get("Water_Source")
    purifier = registry.get("Water_Purifier")
    
    print("\n=== Component Status ===")
    print(f"Tank capacity: {tank.capacity_kg} kg")
    print(f"Tank initial mass: {tank.mass_kg} kg ({tank.fill_level*100:.1f}%)")
    print(f"Tank Zone: {tank.control_zone}")
    print(f"Source mode: {source.mode}")
    print(f"Source max flow: {source.flow_rate_kg_h} kg/h")
    
    # Check if signal connection exists in topology
    print("\n=== Checking Topology Connections ===")
    from h2_plant.config.plant_config import ConnectionConfig
    
    # Get topology from context
    topology = []
    for node in context.topology.nodes:
        for conn in node.connections:
            topology.append(ConnectionConfig(
                source_id=node.id,
                source_port=conn.source_port,
                target_id=conn.target_name,
                target_port=conn.target_port,
                resource_type=conn.resource_type
            ))
    
    signal_connections = [c for c in topology if c.resource_type == 'signal' and 'control' in c.source_port]
    print(f"Found {len(signal_connections)} control signal connections:")
    for conn in signal_connections:
        print(f"  {conn.source_id}:{conn.source_port} -> {conn.target_id}:{conn.target_port}")
    
    # Initialize and run a few steps
    print("\n=== Initializing Engine ===")
    from h2_plant.config.plant_config import SimulationConfig
    config = SimulationConfig(
        timestep_hours=1/60,  # 1 minute
        duration_hours=1,
        start_hour=0,
        checkpoint_interval_hours=0
    )
    
    engine = SimulationEngine(
        registry=registry,
        config=config,
        topology=topology
    )
    engine.initialize()
    
    print("\n=== Running 5 Steps with Debug Output ===")
    dt = 1/60
    registry.initialize_all(dt=dt)
    
    # Force tank to Zone C
    print("\n--- Forcing tank to Zone C (95% full) ---")
    tank.mass_kg = tank.capacity_kg * 0.95
    tank.fill_level = 0.95
    tank.control_zone = 'C'  # Force Zone C
    tank.requested_production_kg_h = 0.0  # Should be 0 in Zone C
    
    for step in range(5):
        t = step * dt
        print(f"\n=== Step {step} (t={t:.4f}h) ===")
        
        # Before step
        print(f"  [Before Step]")
        print(f"    Tank mass: {tank.mass_kg:.1f} kg, Zone: {tank.control_zone}, Request: {tank.requested_production_kg_h:.1f}")
        print(f"    Source signal buffer: {source._signal_request_kg_h}")
        print(f"    Source current flow: {source.current_flow_kg_h:.1f} kg/h")
        
        # Run step on key components
        source.step(t)
        purifier.step(t)
        tank.step(t)
        
        print(f"  [After Step]")
        print(f"    Tank mass: {tank.mass_kg:.1f} kg, Zone: {tank.control_zone}, Request: {tank.requested_production_kg_h:.1f}")
        print(f"    Source flow after step: {source.current_flow_kg_h:.1f} kg/h")
        
        # Get signals
        signal = tank.get_output('control_signal')
        print(f"    Tank control_signal output: {signal.mass_flow_kg_h:.1f} kg/h")
        
        # Simulate FlowNetwork transfer
        print(f"  [Flow Transfer]")
        accepted = source.receive_input('control_signal', signal, 'signal')
        print(f"    Signal transferred to source, new buffer: {source._signal_request_kg_h}")
        
        # Check purifier
        if purifier.ultrapure_out_stream:
            print(f"    Purifier output: {purifier.ultrapure_out_stream.mass_flow_kg_h:.1f} kg/h")
        else:
            print(f"    Purifier output: None")


if __name__ == '__main__':
    main()
