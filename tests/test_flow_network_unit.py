"""
Unit tests for FlowNetwork functionality.
"""
import unittest
import unittest
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock

project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from h2_plant.simulation.flow_network import FlowNetwork
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.config.plant_config import ConnectionConfig
from h2_plant.core.stream import Stream

logging.basicConfig(level=logging.INFO)

from h2_plant.core.component import Component

class MockComponent(Component):
    def __init__(self, component_id):
        super().__init__()
        self.component_id = component_id
        self.inputs = {}
        self.outputs = {}
        self.extracted = {}
        self.dt = 1.0

    def initialize(self, dt, registry):
        super().initialize(dt, registry)

    def step(self, t):
        pass

    def get_state(self):
        return {}

    def get_output(self, port_name):
        return self.outputs.get(port_name, 0.0)

    def receive_input(self, port_name, value, resource_type):
        self.inputs[port_name] = value
        return value if isinstance(value, (int, float)) else value.mass_flow_kg_h

    def extract_output(self, port_name, amount, resource_type):
        self.extracted[port_name] = amount

    def get_ports(self):
        return {}

class TestFlowNetwork(unittest.TestCase):
    def setUp(self):
        self.registry = ComponentRegistry()
        self.source = MockComponent("source")
        self.target = MockComponent("target")
        self.registry.register("source", self.source)
        self.registry.register("target", self.target)

    def test_single_connection_transfer(self):
        """Test simple value transfer between components."""
        topology = [
            ConnectionConfig("source", "out", "target", "in", "water")
        ]
        network = FlowNetwork(self.registry, topology)
        network.initialize()

        # Setup source output
        self.source.outputs["out"] = 100.0

        # Execute
        network.execute_flows(0.0)

        # Verify
        self.assertEqual(self.target.inputs["in"], 100.0)
        self.assertEqual(self.source.extracted["out"], 100.0)

    def test_stream_transfer(self):
        """Test Stream object transfer."""
        topology = [
            ConnectionConfig("source", "h2_out", "target", "h2_in", "hydrogen")
        ]
        network = FlowNetwork(self.registry, topology)
        network.initialize()

        # Setup source output as Stream
        stream = Stream(mass_flow_kg_h=50.0, temperature_k=300, pressure_pa=1e5, composition={'H2': 1.0}, phase='gas')
        self.source.outputs["h2_out"] = stream

        # Execute
        network.execute_flows(0.0)

        # Verify
        self.assertIn("h2_in", self.target.inputs)
        received = self.target.inputs["h2_in"]
        self.assertIsInstance(received, Stream)
        self.assertEqual(received.mass_flow_kg_h, 50.0)
        # Verify extraction called with mass flow
        self.assertEqual(self.source.extracted["h2_out"], 50.0)

    def test_multi_target_splitting(self):
        """Test splitting output to multiple targets (if supported) or sequential."""
        # Note: Current FlowNetwork implementation might not support splitting automatically 
        # unless the component handles it. 
        # Standard behavior: First connection takes what it needs/can, second takes remainder?
        # Let's verify current behavior.
        
        target2 = MockComponent("target2")
        self.registry.register("target2", target2)
        
        topology = [
            ConnectionConfig("source", "out", "target", "in", "water"),
            ConnectionConfig("source", "out", "target2", "in", "water")
        ]
        network = FlowNetwork(self.registry, topology)
        network.initialize()

        self.source.outputs["out"] = 100.0
        
        # Execute
        network.execute_flows(0.0)
        
        # With current logic:
        # 1. Get output (100)
        # 2. Send to target -> returns 100 (accepted)
        # 3. Extract 100 from source
        # 4. Next connection: Get output (now 0 if extracted correctly? Or does get_output return remaining?)
        # The MockComponent doesn't update output based on extraction automatically.
        # But FlowNetwork calls get_output for EACH connection.
        
        # If MockComponent doesn't decrement output, both get 100.
        # This highlights importance of component implementation (output buffering).
        
        self.assertEqual(self.target.inputs["in"], 100.0)
        self.assertEqual(target2.inputs["in"], 100.0)
        
        # This test confirms that FlowNetwork is stateless regarding source availability 
        # within a timestep - it relies on the component's get_output to reflect current state.
        # If component doesn't update, mass duplication occurs.
        
if __name__ == "__main__":
    unittest.main()
