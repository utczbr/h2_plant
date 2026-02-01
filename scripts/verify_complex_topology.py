
import sys
import os
import logging
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from h2_plant.gui.core.graph_adapter import GraphToConfigAdapter, GraphNode, GraphEdge, Port, FlowType
# from h2_plant.gui.core.worker import SimulationWorker # Avoid PySide6 dependency
from h2_plant.simulation.engine import SimulationEngine
from h2_plant.control.engine_dispatch import HybridArbitrageEngineStrategy
from h2_plant.core.graph_builder import PlantGraphBuilder
from h2_plant.core.component_registry import ComponentRegistry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ComplexTopologyVerifier")

def create_port(name, type, direction):
    return Port(name, type, direction)

def run_test():
    logger.info("Building Complex Topology Graph...")
    adapter = GraphToConfigAdapter()
    
    # === 1. Water System ===
    # Water Supply
    adapter.add_node(GraphNode("water_supply", "WaterSupplyNode", "Water Supply", 0, 0, {"flow_rate_kg_h": 2000.0}, [
        create_port("out", FlowType.WATER, "output")
    ]))
    
    # Water Purifier
    adapter.add_node(GraphNode("purifier", "WaterPurifierNode", "Purifier", 100, 0, {"efficiency": 0.99, "max_flow_kg_h": 2000.0}, [
        create_port("in", FlowType.WATER, "input"),
        create_port("out", FlowType.WATER, "output")
    ]))
    
    # Ultra Pure Tank
    adapter.add_node(GraphNode("upw_tank", "UltraPureWaterTankNode", "Ultrapure Tank", 200, 0, {"volume_m3": 50}, [
        create_port("in", FlowType.WATER, "input"),
        create_port("out_pem", FlowType.WATER, "output"),
        create_port("out_soec", FlowType.WATER, "output")
    ]))
    
    # === 2. PEM Train ===
    # Pump P-1 from Tank to PEM
    adapter.add_node(GraphNode("pump_pem", "PumpNode", "PEM Pump", 300, -100, {"max_flow_kg_h": 1000, "target_pressure_bar": 30.0}, [
        create_port("in", FlowType.WATER, "input"),
        create_port("out", FlowType.WATER, "output")
    ]))
    
    # PEM Stack
    adapter.add_node(GraphNode("pem", "PEMStackNode", "PEM System", 400, -100, {"max_power_mw": 5.0}, [
        create_port("water_in", FlowType.WATER, "input"),
        create_port("power_in", FlowType.ELECTRICITY, "input"),
        create_port("h2_out", FlowType.HYDROGEN, "output"), 
        create_port("o2_out", FlowType.OXYGEN, "output")
    ]))
    
    # === 3. SOEC Train ===
    # Pump P-3 from Tank to SOEC
    adapter.add_node(GraphNode("pump_soec", "PumpNode", "SOEC Pump", 300, 100, {"max_flow_kg_h": 500, "target_pressure_bar": 4.0}, [
        create_port("in", FlowType.WATER, "input"),
        create_port("out", FlowType.WATER, "output")
    ]))
    
    # SOEC Stack
    adapter.add_node(GraphNode("soec", "SOECStackNode", "SOEC System", 400, 100, {"max_power_nominal_mw": 2.4}, [
        create_port("water_in", FlowType.WATER, "input"),
        create_port("power_in", FlowType.ELECTRICITY, "input"),
        create_port("h2_out", FlowType.HYDROGEN, "output")
    ]))
    
    # === 4. Power & Logic ===
    # Arbitrage Controller
    adapter.add_node(GraphNode("arbitrage", "ArbitrageNode", "Arbitrage", 400, 0, {"mode": "hybrid", "lookahead_hours": 1}, [
        create_port("pem_power", FlowType.ELECTRICITY, "output"),
        create_port("soec_power", FlowType.ELECTRICITY, "output")
    ]))
    
    # === Connections ===
    # Water: Supply -> Purifier -> Tank
    adapter.add_edge(GraphEdge("water_supply", "out", "purifier", "in", FlowType.WATER))
    adapter.add_edge(GraphEdge("purifier", "out", "upw_tank", "in", FlowType.WATER))
    
    # Tank -> PEM Pump -> PEM
    adapter.add_edge(GraphEdge("upw_tank", "out_pem", "pump_pem", "in", FlowType.WATER))
    adapter.add_edge(GraphEdge("pump_pem", "out", "pem", "water_in", FlowType.WATER))
    
    # Tank -> SOEC Pump -> SOEC
    adapter.add_edge(GraphEdge("upw_tank", "out_soec", "pump_soec", "in", FlowType.WATER))
    adapter.add_edge(GraphEdge("pump_soec", "out", "soec", "water_in", FlowType.WATER))
    
    # Power
    adapter.add_edge(GraphEdge("arbitrage", "pem_power", "pem", "power_in", FlowType.ELECTRICITY))
    adapter.add_edge(GraphEdge("arbitrage", "soec_power", "soec", "power_in", FlowType.ELECTRICITY))
    
    # === Convert to Context ===
    logger.info("Converting graph to SimulationContext...")
    valid, errors = adapter.validate()
    if not valid:
        logger.error(f"Validation failed: {errors}")
        return
        
    context = adapter.to_simulation_context()
    logger.info(f"Context created with {len(context.topology.nodes)} nodes")
    
    # === Build & Run Simulation ===
    logger.info("Initializing SimulationEngine...")
    
    # 1. Build Component Graph
    builder = PlantGraphBuilder(context)
    components = builder.build()
    
    # 2. Registry
    registry = ComponentRegistry()
    for comp_id, comp in components.items():
        registry.register(comp_id, comp)
        
    # 3. Strategy
    strategy = HybridArbitrageEngineStrategy()
    
    # 4. Engine
    from h2_plant.config.models import SimulationConfig
    sim_config = SimulationConfig(
        timestep_hours=1.0/60.0,
        duration_hours=24, # Run for 24 hours
        start_hour=0,
        energy_price_file=context.simulation.energy_price_file,
        wind_data_file=context.simulation.wind_data_file
    )
    
    from h2_plant.simulation.engine import SimulationEngine
    engine = SimulationEngine(
        registry=registry,
        config=sim_config,
        dispatch_strategy=strategy
    )
    
    # 5. Load Data (Mock if files missing, or verify paths)
    # We'll use random data if files verify fails for robustness of this test
    import numpy as np
    steps = 24 * 60
    prices = np.random.uniform(50, 150, steps) # EUR/MWh
    wind = np.random.uniform(0, 10, steps) # MW
    
    engine.set_dispatch_data(prices, wind)
    engine.initialize()
    engine.initialize_dispatch_strategy(context, steps)
    
    logger.info("Running simulation...")
    results_raw = engine.run()
    
    logger.info("Simulation complete.")
    history = engine.get_dispatch_history()
    
    if history:
        total_h2 = np.sum(history['h2_kg'])
        logger.info(f"Total H2 Produced: {total_h2:.2f} kg")
        if total_h2 > 0:
            logger.info("SUCCESS: Topology produced hydrogen!")
        else:
            logger.warning("Topology connected but produced 0 H2 (check wind/prices or logic)")
    else:
        logger.error("No history returned!")

if __name__ == "__main__":
    run_test()
