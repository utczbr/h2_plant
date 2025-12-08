
import pytest
# Skip GUI node import to avoid Qt/Matplotlib dependencies
# from h2_plant.gui.nodes.thermal import DryCoolerNode 
from h2_plant.config.plant_config import PlantConfig, ThermalComponentsConfig
from h2_plant.config.plant_builder import PlantBuilder
from h2_plant.gui.core.graph_adapter import GraphToConfigAdapter, GraphNode, FlowType
from h2_plant.components.cooling.dry_cooler import DryCooler

def test_plant_config_parsing():
    """Test that PlantConfig accepts dry_coolers count."""
    config_dict = {
        "production": {},
        "storage": {},
        "thermal_components": {
            "chillers": 5,
            "dry_coolers": 3,
            "heat_exchangers": 2,
            "steam_generators": 1
        }
    }
    config = PlantConfig.from_dict(config_dict)
    assert config.thermal_components.dry_coolers == 3

def test_plant_builder_instantiation():
    """Test that PlantBuilder builds DryCooler components."""
    config_dict = {
        "production": {},
        "storage": {},
        "thermal_components": {
            "chillers": 0,
            "dry_coolers": 2,
            "heat_exchangers": 0,
            "steam_generators": 0
        }
    }
    config = PlantConfig.from_dict(config_dict)
    builder = PlantBuilder(config)
    plant = builder.build()
    
    # Check if registered
    registry = builder.registry
    assert registry.get("dry_cooler_0") is not None
    assert isinstance(registry.get("dry_cooler_0"), DryCooler)
    assert registry.get("dry_cooler_1") is not None
    
    # Check fan power was set (default in builder)
    dc = registry.get("dry_cooler_0")
    assert dc.fan_power_kw == 10.0

def test_graph_adapter_counting():
    """Test that GraphToConfigAdapter correctly counts DryCooler nodes."""
    adapter = GraphToConfigAdapter()
    
    # Add 2 Dry Cooler nodes (Simulating GraphNode objects created by GUI)
    node1 = GraphNode(
        id="node1",
        type="h2_plant.nodes.thermal.DryCoolerNode", # Full type string from GUI
        display_name="DC 1",
        x=0, y=0,
        properties={"fan_power_kw": 15.0},
        ports=[]
    )
    node2 = GraphNode(
        id="node2",
        type="h2_plant.nodes.thermal.DryCoolerNode",
        display_name="DC 2",
        x=100, y=0,
        properties={"fan_power_kw": 20.0},
        ports=[]
    )
    
    adapter.add_node(node1)
    adapter.add_node(node2)
    
    config_dict = adapter.to_config_dict()
    
    assert "thermal_components" in config_dict
    assert config_dict["thermal_components"]["dry_coolers"] == 2
    
def test_graph_adapter_mapping():
    """Test that DryCoolerNode maps to DryCooler component type."""
    adapter = GraphToConfigAdapter()
    backend_type = adapter._map_node_type("DryCoolerNode")
    assert backend_type == "DryCooler"
