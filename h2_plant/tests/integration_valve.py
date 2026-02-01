
import sys
import os
from pathlib import Path
import logging

# Add project root
sys.path.append(str(Path(__file__).parents[1]))

from h2_plant.config.models import SimulationContext, TopologyConfig, ComponentNode, NodeConnection
from h2_plant.gui.core.graph_adapter import GraphToConfigAdapter, GraphNode, GraphEdge, FlowType, Port
from h2_plant.orchestrator import Orchestrator
from h2_plant.core.component_ids import ComponentID
from h2_plant.optimization.lut_manager import LUTManager, LUTConfig
from h2_plant.legacy.Valvula.Valvula import modelo_valvula_isoentalpica

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IntegrationTest")

def create_mock_context():
    # 1. Create Nodes
    # Source (We'll use a CompressedH2Source for simplicity or just a Tank starting full)
    # Since we need high pressure, let's use a HPTank as source (350 bar)
    
    # Actually, orchestrator needs a graph.
    # Let's use GraphAdapter to build it properly, mocking the GUI nodes.
    
    adapter = GraphToConfigAdapter()
    
    # Node 1: High Pressure Source (HPTank A)
    node1 = GraphNode(
        id="SourceTank",
        type="HPTankNode",
        display_name="HP Source",
        x=0, y=0,
        properties={
            "tank_count": 1,
            "capacity_per_tank_kg": 1000.0,
            "operating_pressure_bar": 350.0,
            "initial_level_kg": 1000.0, # Full
            "name": "HP Source"
        },
        ports=[]
    )
    # Add ports manually as Adapter expects them
    node1.ports.append(Port("exhaust", FlowType.HYDROGEN, "output")) # Hack: Tank output
    
    # Node 2: Valve
    node2 = GraphNode(
        id="Valve1",
        type="ValveNode",
        display_name="Throttling Valve",
        x=200, y=0,
        properties={
            "outlet_pressure_bar": 30.0,
            "fluid_type": "H2"
        },
        ports=[]
    )
    
    # Node 3: Low Pressure Destination (LPTank B)
    node3 = GraphNode(
        id="DestTank",
        type="LPTankNode",
        display_name="LP Dest",
        x=400, y=0,
        properties={
            "tank_count": 1, 
            "capacity_per_tank_kg": 1000.0,
            "operating_pressure_bar": 30.0
        },
        ports=[]
    )
    
    adapter.add_node(node1)
    adapter.add_node(node2)
    adapter.add_node(node3)
    
    # Edges
    # Source -> Valve
    edge1 = GraphEdge("SourceTank", "exhaust", "Valve1", "inlet", FlowType.HYDROGEN)
    # Valve -> Dest
    edge2 = GraphEdge("Valve1", "outlet", "DestTank", "intake", FlowType.HYDROGEN)
    
    adapter.add_edge(edge1)
    adapter.add_edge(edge2)
    
    # Convert
    # Note: Adapter expects valid connections. Tank nodes in GUI usually have 'h2_out', 'h2_in'.
    # We need to match what Adapter expects or what ComponentBuilder expects.
    # Tank component usually just has mass balance.
    # But wait, Tank is not an active source that pushes flow!
    # A Pump or Compressor moves flow.
    # A Valve is a PASSIVE restriction. It doesn't PUSH flow.
    # If we connect Tank -> Valve -> Tank, who drives the flow?
    # In this simulation engine (push-based), usually a Source or Compressor pushes.
    # Or strict pressure-difference flow?
    
    # Reviewing `tank.py` or system architecture: 
    # Flows are typically determined by Components step().
    # Tank doesn't push.
    
    # To test Valve, we should put it after a Compressor or Source that pushes flow.
    # Let's use `FillingCompressorNode` -> `Valve` -> `Tank`.
    # Or just `Compressor` -> `Valve`.
    # Compressor Pushes constant flow.
    
    # Better: Use a simple Mock Source component if available, OR reuse Compressor.
    
    # Let's rebuild the graph:
    # Compressor (350 bar out, Flow=10kg/h) -> Valve (30 bar throttling) -> Tank.
    # Wait, Compressor creates 350 bar. Valve throttles to 30.
    
    adapter = GraphToConfigAdapter() # Reset
    
    # Node 1: Compressor (Source of flow and pressure)
    c_props = {
        "max_flow_kg_h": 10.0,
        "efficiency": 0.8,
        "inlet_pressure_bar": 1.0, 
        "outlet_pressure_bar": 350.0, # Compressor Output P
        "name": "Comp1"
    }
    node_c = GraphNode("Comp1", "FillingCompressorNode", "Comp1", 0,0, c_props, [Port("h2_out", FlowType.HYDROGEN, "output")])
    
    # Node 2: Valve
    v_props = {"outlet_pressure_bar": 30.0, "fluid_type": "H2", "name": "Valv1"}
    node_v = GraphNode("Valv1", "ValveNode", "Valv1", 200,0, v_props, 
                       [Port("inlet", FlowType.HYDROGEN, "input"), Port("outlet", FlowType.HYDROGEN, "output")])
    
    # Node 3: Tank (Sink)
    t_props = {"tank_count": 1, "capacity_per_tank_kg": 100, "operating_pressure_bar": 30.0, "name": "Tank1"}
    node_t = GraphNode("Tank1", "LPTankNode", "Tank1", 400,0, t_props, [Port("h2_in", FlowType.HYDROGEN, "input")])
    
    adapter.add_node(node_c)
    adapter.add_node(node_v)
    adapter.add_node(node_t)
    
    adapter.add_edge(GraphEdge("Comp1", "h2_out", "Valv1", "inlet", FlowType.HYDROGEN))
    adapter.add_edge(GraphEdge("Valv1", "outlet", "Tank1", "h2_in", FlowType.HYDROGEN))
    
    context = adapter.to_simulation_context()
    return context

def run_integration_test():
    print("=== Running Integration Test ===")
    
    # 1. Setup Context & Orchestrator
    context = create_mock_context()
    orchestrator = Orchestrator(".", context=context)
    
    # 2. Inject LUT Manager manually into registry (since Orchestrator usually does defaults)
    # We need to ensure LUT is available. Orchestrator relies on GraphBuilder.
    # GraphBuilder creates components.
    # LUTManager is a "Manager". Orchestrator might load standard managers?
    # Checking Orchestrator...
    # It seems we might need to manually ensure LUTManager is there if not default.
    # But let's assume standard Orchestrator flow works or we patch it.
    
    # PATCH: Add LUTManager to components list before init
    from h2_plant.optimization.lut_manager import LUTManager, LUTConfig
    # Use Fast Config
    fast_config = LUTConfig(pressure_points=50, temperature_points=50, fluids=('H2',), cache_dir=Path("/tmp/h2_int_test"))
    lut_mgr = LUTManager(fast_config)
    
    # We need to inject this into the registry that Orchestrator builds.
    # Orchestrator.initialize_components() creates a NEW registry.
    # We can perform initialization, THEN inject, THEN rebuild logic?
    # Or just subclass/mock.
    
    # Simpler: Initialize, then inject LUT manually into registry, then re-initialize components?
    orchestrator.initialize_components()
    orchestrator.registry.register(ComponentID.LUT_MANAGER.value, lut_mgr)
    lut_mgr.initialize() # Check if already initialized
    
    # Because components might have grabbed None reference during their init, 
    # we might need to re-init them or they look up registry in Step?
    # Valve looks up registry in initialize().
    # So if LUT wasn't there during orchestrator.initialize_components(), Valve has None.
    
    # Fix: Register LUT *before* other components init?
    # Orchestrator.initialize_components code:
    #   self.registry = ComponentRegistry()
    #   for cid, comp in self.components.items(): registry.register(..., comp)
    #   registry.initialize_all(dt)
    
    # We can add LUTManager to orchestrator.components before initialize_components() is called!
    orchestrator.components[ComponentID.LUT_MANAGER.value] = lut_mgr
    # Now call init
    orchestrator.initialize_components()
    
    # 3. Reference Calculation (Valvula.py)
    # Compressor Output: 350 bar. (Assuming perfect compressor, T out depends on efficiency)
    # Comp Efficiency = 0.8.
    # T_in = 298.15 K (Ambient).
    # P_in_comp = 1 bar. P_out_comp = 350 bar.
    # We need to know what T the compressor calculates to predict Valve Inlet T.
    
    # Let's run 1 step to see inputs.
    dt = context.simulation.timestep_hours
    
    # Inject Input to Compressor (since it has no source)
    comp = orchestrator.registry.get("Comp1")
    if comp:
        from h2_plant.core.stream import Stream
        # 10 kg/h at 1 bar, 298 K
        s_in = Stream(10.0, 298.15, 101325.0, {'H2': 1.0})
        comp.receive_input('h2_in', s_in, 'hydrogen')
        # Also give it power strictly speaking, but usually it calculates demand?
        comp.receive_input('power', 1000.0, 'electricity') # Plenty of power
        
    
    # MANUAL STEPPING to ensure flow propagation
    # 1. Step Compressor
    comp = orchestrator.registry.get("Comp1")
    comp.step(dt)
    
    # 2. Transfer Comp -> Valve
    h2_stream = comp.get_output("h2_out")
    if h2_stream:
        valve = orchestrator.registry.get("Valv1")
        valve.receive_input("inlet", h2_stream, "hydrogen")
        valve.step(dt)
        
        # 3. Transfer Valve -> Tank
        v_out = valve.get_output("outlet")
        if v_out:
            tank = orchestrator.registry.get("Tank1")
            tank.receive_input("h2_in", v_out, "hydrogen")
            tank.step(dt)
            
    # Inspect Valve
    valve = orchestrator.registry.get("Valv1")
    outlet_stream = valve.get_output("outlet")
        
    # Inspect state
    inlet_stream = valve.inlet_stream
    if inlet_stream:
        print(f"Valve Inlet: P={inlet_stream.pressure_pa/1e5:.2f} bar, T={inlet_stream.temperature_k:.2f} K")
        print(f"Valve Outlet: P={outlet_stream.pressure_pa/1e5:.2f} bar, T={outlet_stream.temperature_k:.2f} K")
        print(f"Simulated Delta T: {outlet_stream.temperature_k - inlet_stream.temperature_k:.4f} K")
        
        # COMPARE with Reference
        print("\n--- Reference Calculation (CoolProp) ---")
        ref_data = modelo_valvula_isoentalpica(
            "hydrogen", 
            inlet_stream.temperature_k, 
            inlet_stream.pressure_pa, 
            outlet_stream.pressure_pa
        )
        
        if ref_data:
            ref_T_out = ref_data['SAIDA']['T_K']
            print(f"Reference T_out: {ref_T_out:.2f} K")
            error = abs(outlet_stream.temperature_k - ref_T_out)
            print(f"Error: {error:.4f} K")
            
            if error < 1.0:
                print("SUCCESS: Simulation matches Reference within 1K.")
            else:
                print("FAILURE: Divergence > 1K.")
                
    else:
        print("Flow failed to reach valve.")

if __name__ == "__main__":
    run_integration_test()
