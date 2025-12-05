import pytest
from h2_plant.simulation.flow_tracker import FlowTracker, Flow, FlowType

@pytest.fixture
def flow_tracker():
    """Fixture for a new FlowTracker instance."""
    return FlowTracker()

def test_flow_tracker_init(flow_tracker):
    """Test FlowTracker initializes correctly."""
    assert len(flow_tracker.flows) == 0
    assert flow_tracker.current_hour == 0

def test_record_flow(flow_tracker):
    """Test recording a single valid flow."""
    flow_tracker.set_current_hour(10)
    flow_tracker.record_flow(
        source_component="electrolyzer",
        destination_component="lp_storage",
        flow_type="HYDROGEN_MASS",
        amount=50.0,
        unit="kg"
    )
    
    assert len(flow_tracker.flows) == 1
    flow = flow_tracker.flows[0]
    assert flow.hour == 10
    assert flow.flow_type == FlowType.HYDROGEN_MASS
    assert flow.source_component == "electrolyzer"
    assert flow.destination_component == "lp_storage"
    assert flow.amount == 50.0
    assert flow.unit == "kg"

def test_record_invalid_flow_type(flow_tracker):
    """Test that an invalid flow type is ignored."""
    flow_tracker.record_flow(
        source_component="a",
        destination_component="b",
        flow_type="INVALID_TYPE",
        amount=10.0,
        unit="-"
    )
    assert len(flow_tracker.flows) == 0

def test_get_summary_statistics(flow_tracker):
    """Test the summary statistics aggregation."""
    flow_tracker.set_current_hour(1)
    flow_tracker.record_flow("A", "B", "HYDROGEN_MASS", 10, "kg")
    flow_tracker.record_flow("A", "B", "HYDROGEN_MASS", 15, "kg")
    flow_tracker.record_flow("B", "C", "ELECTRICAL_ENERGY", 100, "kWh")

    summary = flow_tracker.get_summary_statistics()
    
    assert "A_to_B_HYDROGEN_MASS" in summary
    assert "B_to_C_ELECTRICAL_ENERGY" in summary
    
    h2_flow = summary["A_to_B_HYDROGEN_MASS"]
    assert h2_flow['total_amount'] == 25.0
    assert h2_flow['count'] == 2
    assert h2_flow['unit'] == "kg"

    energy_flow = summary["B_to_C_ELECTRICAL_ENERGY"]
    assert energy_flow['total_amount'] == 100.0
    assert energy_flow['count'] == 1
    assert energy_flow['unit'] == "kWh"

def test_get_sankey_data(flow_tracker):
    """Test the generation of data for Sankey diagrams."""
    flow_tracker.set_current_hour(1)
    flow_tracker.record_flow("Grid", "Electrolyzer", "ELECTRICAL_ENERGY", 200, "kWh")
    flow_tracker.record_flow("Electrolyzer", "Storage", "HYDROGEN_MASS", 10, "kg")
    flow_tracker.record_flow("Electrolyzer", "Vent", "OXYGEN_MASS", 80, "kg")
    
    sankey_data = flow_tracker.get_sankey_data()
    
    assert "nodes" in sankey_data
    assert "links" in sankey_data
    
    nodes = sankey_data["nodes"]
    links = sankey_data["links"]

    # Expected nodes: Grid, Electrolyzer, Storage, Vent
    assert len(nodes) == 4
    node_names = [n['name'] for n in nodes]
    assert "Grid" in node_names
    assert "Electrolyzer" in node_names
    assert "Storage" in node_names
    assert "Vent" in node_names

    # Expected links
    assert len(links) == 3
    
    # Check one link in detail
    grid_to_elec = next(l for l in links if l['source'] == node_names.index("Grid"))
    assert grid_to_elec['target'] == node_names.index("Electrolyzer")
    assert grid_to_elec['value'] == 200
    assert grid_to_elec['label'] == "Electrical Energy"
