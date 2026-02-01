import pytest
from pathlib import Path
import json
from h2_plant.simulation.state_manager import StateManager

@pytest.fixture
def state_manager(tmp_path):
    """Fixture for StateManager."""
    return StateManager(output_dir=tmp_path)

def test_save_and_load_json_checkpoint(state_manager):
    """Test saving and loading a checkpoint in JSON format."""
    hour = 100
    component_states = {
        "comp1": {"param1": 10, "param2": "value"},
        "comp2": {"mass": [1.0, 2.0, 3.0]}
    }
    metadata = {"sim_name": "test_run"}

    # Save
    checkpoint_path = state_manager.save_checkpoint(hour, component_states, metadata, format="json")
    assert checkpoint_path.exists()
    assert checkpoint_path.name == "checkpoint_hour_100.json"

    # Load
    loaded_data = state_manager.load_checkpoint(checkpoint_path)

    assert loaded_data["hour"] == hour
    assert loaded_data["metadata"] == metadata
    assert loaded_data["component_states"] == component_states

def test_list_checkpoints(state_manager):
    """Test listing available checkpoints."""
    state_manager.save_checkpoint(1, {}, format="json")
    state_manager.save_checkpoint(2, {}, format="json")
    
    checkpoints = state_manager.list_checkpoints()
    assert len(checkpoints) == 2
    assert "checkpoint_hour_1.json" in str(checkpoints[0])
    assert "checkpoint_hour_2.json" in str(checkpoints[1])

def test_save_results(state_manager):
    """Test saving final simulation results."""
    results = {"metrics": {"total_prod": 1000}, "final_states": {}}
    results_path = state_manager.output_dir / "results.json"
    
    state_manager.save_results(results, results_path)
    assert results_path.exists()

    with open(results_path, 'r') as f:
        loaded_results = json.load(f)
    
    assert loaded_results == results
