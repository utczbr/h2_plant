"""
Test Phase 1 GUI Components.
"""
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from h2_plant.gui.core.graph_adapter import GraphToConfigAdapter, GraphNode, GraphEdge, FlowType
from h2_plant.config.plant_builder import PlantBuilder

def test_adapter_basic():
    print("Testing GraphToConfigAdapter...")
    adapter = GraphToConfigAdapter()
    
    # Create nodes
    elec_node = GraphNode(
        id="elec_1",
        type="ElectrolyzerNode",
        display_name="Electrolyzer",
        x=0, y=0,
        properties={"max_power_mw": 5.0, "base_efficiency": 0.68},
        ports=[]
    )
    
    tank_node = GraphNode(
        id="tank_1",
        type="LPTankNode",
        display_name="LP Tank",
        x=100, y=0,
        properties={"count": 4, "capacity_kg": 50.0, "pressure_bar": 30.0},
        ports=[]
    )
    
    adapter.add_node(elec_node)
    adapter.add_node(tank_node)
    
    # Create connection
    edge = GraphEdge(
        source_node_id="elec_1",
        source_port="h2_out",
        target_node_id="tank_1",
        target_port="h2_in",
        flow_type=FlowType.HYDROGEN
    )
    adapter.add_edge(edge)
    
    # Convert to config
    config_dict = adapter.to_config_dict()
    print("Generated Config:", config_dict)
    
    # Verify structure
    assert config_dict["production"]["electrolyzer"]["max_power_mw"] == 5.0
    assert config_dict["storage"]["lp_tanks"]["count"] == 4
    assert config_dict["storage"]["source_isolated"] is False
    
    print("Adapter test passed!")
    return config_dict

def test_plant_builder(config_dict):
    print("\nTesting PlantBuilder.from_dict...")
    try:
        plant = PlantBuilder.from_dict(config_dict)
        print("Plant built successfully!")
        print(f"Registry contains: {plant.registry.get_all_ids()}")
    except Exception as e:
        print(f"PlantBuilder failed: {e}")
        raise

if __name__ == "__main__":
    try:
        config = test_adapter_basic()
        test_plant_builder(config)
        print("\nALL TESTS PASSED")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        sys.exit(1)
