"""
Unit tests for TopologyValidator.
"""
import unittest
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock

project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from h2_plant.config.validator import TopologyValidator
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.config.plant_config import PlantConfig, ConnectionConfig
from h2_plant.core.component import Component
from h2_plant.core.exceptions import ConfigurationError

logging.basicConfig(level=logging.INFO)

class MockComponent(Component):
    def __init__(self, component_id, ports=None):
        super().__init__()
        self.component_id = component_id
        self._ports = ports or {}

    def initialize(self, dt, registry): pass
    def step(self, t): pass
    def get_state(self): return {}
    def get_output(self, port_name): return 0.0
    def receive_input(self, port_name, value, resource_type): return 0.0
    def extract_output(self, port_name, amount, resource_type): pass
    
    def get_ports(self):
        return self._ports

class TestTopologyValidator(unittest.TestCase):
    def setUp(self):
        self.registry = ComponentRegistry()
        
        # Setup components with ports
        self.source = MockComponent("source", {
            "out": {"type": "output", "resource_type": "water", "units": "kg/h"}
        })
        self.target = MockComponent("target", {
            "in": {"type": "input", "resource_type": "water", "units": "kg/h"}
        })
        
        self.registry.register("source", self.source)
        self.registry.register("target", self.target)
        
        # Mock PlantConfig
        self.config = MagicMock(spec=PlantConfig)
        self.config.topology = []

    def test_valid_topology(self):
        """Test a valid connection."""
        self.config.topology = [
            ConnectionConfig("source", "out", "target", "in", "water")
        ]
        validator = TopologyValidator(self.config, self.registry)
        # Should not raise
        validator.validate()

    def test_invalid_source_id(self):
        """Test connection from non-existent source."""
        self.config.topology = [
            ConnectionConfig("nonexistent", "out", "target", "in", "water")
        ]
        validator = TopologyValidator(self.config, self.registry)
        with self.assertRaises(ConfigurationError):
            validator.validate()

    def test_invalid_target_id(self):
        """Test connection to non-existent target."""
        self.config.topology = [
            ConnectionConfig("source", "out", "nonexistent", "in", "water")
        ]
        validator = TopologyValidator(self.config, self.registry)
        with self.assertRaises(ConfigurationError):
            validator.validate()

    def test_port_direction_mismatch(self):
        """Test connecting input to input."""
        # Add another input port to source for testing
        self.source._ports["in_extra"] = {"type": "input", "resource_type": "water"}
        
        self.config.topology = [
            ConnectionConfig("source", "in_extra", "target", "in", "water")
        ]
        validator = TopologyValidator(self.config, self.registry)
        with self.assertRaisesRegex(ConfigurationError, "is not an output"):
            validator.validate()

    def test_resource_mismatch_warning(self):
        """Test resource type mismatch (should log warning, not error currently)."""
        self.config.topology = [
            ConnectionConfig("source", "out", "target", "in", "hydrogen") # Mismatch: water vs hydrogen
        ]
        validator = TopologyValidator(self.config, self.registry)
        
        with self.assertLogs(level='WARNING') as cm:
            validator.validate()
            self.assertTrue(any("does not match" in o for o in cm.output))

if __name__ == "__main__":
    unittest.main()
