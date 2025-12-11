import sys
import os
import logging

# Add project root to path
# Add project root to path
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from h2_plant.gui.core.graph_adapter import GraphToConfigAdapter, GraphNode, FlowType, GraphEdge
# Mock PySide6
import sys
from unittest.mock import MagicMock

# Mock QThread and Signal
class MockQThread:
    def __init__(self):
        pass
    def start(self):
        self.run()
    def run(self):
        pass

class MockSignal:
    def __init__(self, *args):
        pass
    def emit(self, *args):
        pass

mock_pyside = MagicMock()
mock_pyside.QtCore.QThread = MockQThread
mock_pyside.QtCore.Signal = MockSignal
sys.modules["PySide6"] = mock_pyside
sys.modules["PySide6.QtCore"] = mock_pyside.QtCore

from h2_plant.gui.core.worker import SimulationWorker
from h2_plant.config.models import SimulationContext

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Verification")

def test_integration():
    logger.info("Starting integration test...")
    
    # 1. Create Adapter and Mock Graph
    adapter = GraphToConfigAdapter()
    
    # Create PEM Node
    pem_node = GraphNode(
        id="pem_1",
        type="h2_plant.nodes.PEMStackNode",
        display_name="PEM Electrolyzer",
        x=0, y=0,
        properties={
            "rated_power_kw": 5000.0,
            "efficiency_rated": 0.65
        },
        ports=[]
    )
    adapter.add_node(pem_node)
    
    # Create Tank Node
    tank_node = GraphNode(
        id="tank_1",
        type="h2_plant.nodes.LPTankNode",
        display_name="LP Tank",
        x=200, y=0,
        properties={
            "tank_count": 4,
            "capacity_per_tank_kg": 50.0,
            "operating_pressure_bar": 30.0
        },
        ports=[]
    )
    adapter.add_node(tank_node)
    
    # Create Connection
    edge = GraphEdge(
        source_node_id="pem_1",
        source_port="h2_out",
        target_node_id="tank_1",
        target_port="h2_in",
        flow_type=FlowType.HYDROGEN
    )
    adapter.add_edge(edge)
    
    # 2. Generate SimulationContext
    logger.info("Generating SimulationContext...")
    context = adapter.to_simulation_context()
    
    if not isinstance(context, SimulationContext):
        logger.error("Failed to generate SimulationContext")
        return False
        
    logger.info("SimulationContext generated successfully.")
    logger.info(f"Topology Nodes: {len(context.topology.nodes)}")
    
    # 3. Initialize Worker
    logger.info("Initializing SimulationWorker...")
    worker = SimulationWorker(context)
    
    # 4. Run Simulation (Synchronously for test)
    # We can't easily run QThread synchronously without an event loop, 
    # so we'll just call the logic that would be in run() or verify Orchestrator init.
    
    from h2_plant.orchestrator import Orchestrator
    orchestrator = Orchestrator(scenarios_dir=".", context=context)
    
    logger.info("Orchestrator initialized.")
    logger.info(f"Components initialized: {len(orchestrator.components)}")
    
    # Run a short simulation
    logger.info("Running short simulation...")
    history = orchestrator.run_simulation(hours=24)
    
    if history:
        logger.info("Simulation completed successfully.")
        logger.info(f"History keys: {list(history.keys())}")
        return True
    else:
        logger.error("Simulation returned no history.")
        return False

if __name__ == "__main__":
    try:
        success = test_integration()
        if success:
            print("✅ Integration Test Passed")
            sys.exit(0)
        else:
            print("❌ Integration Test Failed")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Integration Test Failed with Exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
