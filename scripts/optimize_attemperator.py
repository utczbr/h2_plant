
import sys
import os
import numpy as np
from scipy.optimize import brentq

# Add project root to path
project_root = "/home/stuart/Documentos/Planta Hidrogenio"
sys.path.append(project_root)

from h2_plant.core.graph_builder import PlantGraphBuilder
from h2_plant.simulation.engine import SimulationEngine
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.optimization.lut_manager import LUTManager
from h2_plant.config.loader import ConfigLoader
from h2_plant.config.plant_config import ConnectionConfig

def run_optimization():
    # 1. Setup Simulation Environment
    print("Setting up simulation environment...")
    
    # Initialize LUT Manager (heavy, do once)
    lut_manager = LUTManager()
    lut_manager.initialize()
    
    # Load Context using ConfigLoader
    scenarios_dir = os.path.join(project_root, "scenarios")
    loader = ConfigLoader(scenarios_dir)
    context = loader.load_context()
    
    # Extract Topology Connections (Static)
    topology_connections = []
    for node in context.topology.nodes:
        source_id = node.id
        for conn in node.connections:
            topology_connections.append(ConnectionConfig(
                source_id=source_id,
                source_port=conn.source_port,
                target_id=conn.target_name,
                target_port=conn.target_port,
                resource_type=conn.resource_type
            ))
            
    # Initialize Builder
    builder = PlantGraphBuilder(context)
    
    # 2. Define Evaluation Function
    def objective_function(flow_1):
        flow_2 = 3600.0 - flow_1
        
        # Modify Context Parameters directly
        ws1_node = next(n for n in context.topology.nodes if n.id == 'watersource_1')
        ws2_node = next(n for n in context.topology.nodes if n.id == 'watersource_2')
        
        # Update params
        ws1_node.params['flow_rate_kg_h'] = flow_1
        ws2_node.params['flow_rate_kg_h'] = flow_2
        ws2_node.params['pressure_bar'] = 10.0 # Force sufficient pressure for injection
        
        # Fix DryCooler: Prevent condensation (Default was cooling to 25C)
        try:
            dc_node = next(n for n in context.topology.nodes if n.id == 'drycooler')
            dc_node.params['target_outlet_temp_c'] = 140.0 
        except StopIteration:
            pass
        
        # Create Fresh Registry
        registry = ComponentRegistry()
        registry.register('lut_manager', lut_manager) # Reuse initialized logic
        
        # Build Graph Components
        components = builder.build()
        for cid, comp in components.items():
            if cid != 'lut_manager': # Don't double register
                try:
                    registry.register(cid, comp)
                except Exception:
                    pass # Ignore if already registered (e.g. builder side effect)
            
        # Initialize Engine with Fresh Registry
        engine = SimulationEngine(
            registry=registry, 
            config=context.simulation,
            topology=topology_connections
        )
        
        # Initialize
        engine.initialize()
        
        # Run to Steady State
        engine.run(end_hour=1)
        
        # Check Result
        att = registry.get('atemperador')
        
        drain_flow = 0.0
        if att.drain_stream:
            drain_flow = att.drain_stream.mass_flow_kg_h
            
        t_out_c = att.output_stream.temperature_k - 273.15 if att.output_stream else 0.0
        p_out = att.output_stream.pressure_pa if att.output_stream else 0.0
        
        target_c = 152.0
        
        # Metric:
        # High Flow2 (Low Flow1) => Excess water => Drain > 0.  Result: Positive.
        # Low Flow2 (High Flow1) => Deficit water => Temp > 152. Result: Negative.
        
        if drain_flow > 1e-4:
            return drain_flow
        else:
            temp_diff = t_out_c - target_c
            # Check for negative diff (Too cold implies excess water -> Positive result)
            return -temp_diff * 10.0

    print("Starting optimization loop...")
    
    try:
        # Initial guess brackets
        # If flow_1 is low (100), flow_2 is high (3500) -> Excess water -> Positive result
        # If flow_1 is high (3500), flow_2 is low (100) -> Deficit water -> Negative result
        # brentq expects f(a) and f(b) to have different signs.
        
        val_low = objective_function(100.0)
        val_high = objective_function(3500.0)
        
        print(f"Brackets: f(100)={val_low:.4f}, f(3500)={val_high:.4f}")
        
        if val_low * val_high > 0:
             print(f"Warning: Root not bracketed! Both results same sign.")
             # Optimization fallback
             if val_low > 0:
                 # Even at 3500 kg/h steam (100 kg/h water), we have excess water? Unlikely.
                 # Actually, if steam is 3500, we need lots of water. Supply is 100. 
                 # So we should be in Deficit (Negative).
                 pass
             
        optimal_flow_1 = brentq(objective_function, 100.0, 3500.0, xtol=0.01)
        optimal_flow_2 = 3600.0 - optimal_flow_1
        
        print("\n" + "="*40)
        print("OPTIMIZATION SUCCESSFUL")
        print("="*40)
        print(f"Optimal Flow 1 (Water Source 1): {optimal_flow_1:.4f} kg/h")
        print(f"Optimal Flow 2 (Water Source 2): {optimal_flow_2:.4f} kg/h")
        print(f"Total Flow: {optimal_flow_1 + optimal_flow_2:.4f} kg/h")
        
        # Verify (Run one last time to print properties)
        print("\nVerifying Solution Properties:")
        objective_function(optimal_flow_1) 
        
        # Need to access registry from last run? objective_function does not return it.
        # But we can just use the print values from a manual run here.
        
        # Manual Re-Run for print
        ws1_node = next(n for n in context.topology.nodes if n.id == 'watersource_1')
        ws2_node = next(n for n in context.topology.nodes if n.id == 'watersource_2')
        ws1_node.params['flow_rate_kg_h'] = optimal_flow_1
        ws2_node.params['flow_rate_kg_h'] = optimal_flow_2
        ws2_node.params['pressure_bar'] = 10.0
        
        registry = ComponentRegistry()
        registry.register('lut_manager', lut_manager)
        components = builder.build()
        for cid, comp in components.items():
            if cid != 'lut_manager': registry.register(cid, comp)
            
        engine = SimulationEngine(registry=registry, config=context.simulation, topology=topology_connections)
        engine.initialize()
        engine.run(end_hour=1)
        
        att = registry.get('atemperador')
        t_out_c = att.output_stream.temperature_k - 273.15
        p_out_bar = att.output_stream.pressure_pa / 1e5
        drain = att.drain_stream.mass_flow_kg_h if att.drain_stream else 0.0
        
        print(f"Attemperator Outlet Temp: {t_out_c:.4f} C")
        print(f"Attemperator Outlet Pressure: {p_out_bar:.4f} bar")
        print(f"Attemperator Drain Flow: {drain:.4f} kg/h")
        print("="*40)
        
    except ValueError as ve:
        print(f"Optimization failed: {ve}")

if __name__ == "__main__":
    run_optimization()
