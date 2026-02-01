from pathlib import Path
from h2_plant.config.loaders import load_plant_config
from h2_plant.core.exceptions import ConfigurationError
import pytest


def test_load_yaml_configuration(tmp_path):
    """Test loading YAML configuration."""
    
    yaml_content = """
name: "Test Plant"
version: "1.0"
production:
  electrolyzer:
    max_power_mw: 2.5
    base_efficiency: 0.65
storage:
  lp_tanks:
    count: 4
    capacity_kg: 50.0
    pressure_bar: 30.0
    temperature_k: 298.15
  hp_tanks:
    count: 8
    capacity_kg: 200.0
    pressure_bar: 350.0
    temperature_k: 298.15
compression:
  filling_compressor:
    max_flow_kg_h: 100.0
    inlet_pressure_bar: 30.0
    outlet_pressure_bar: 350.0
    num_stages: 3
    efficiency: 0.75
  outgoing_compressor:
    max_flow_kg_h: 200.0
    inlet_pressure_bar: 350.0
    outlet_pressure_bar: 900.0
    efficiency: 0.75
demand:
  pattern: "constant"
  base_demand_kg_h: 50.0
energy_price:
  source: "constant"
  constant_price_per_mwh: 60.0
pathway:
  allocation_strategy: "COST_OPTIMAL"
  priority_source: null
simulation:
  timestep_hours: 1.0
  duration_hours: 8760
  start_hour: 0
  checkpoint_interval_hours: 168
"""
    
    # Write temporary config file
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(yaml_content)
    
    # Load configuration
    config = load_plant_config(config_file)
    
    assert config.name == "Test Plant"
    assert config.production.electrolyzer.max_power_mw == 2.5
    assert config.storage.hp_tanks.count == 8

def test_load_nonexistent_file():
    """Test loading a nonexistent configuration file."""
    with pytest.raises(ConfigurationError, match="not found"):
        load_plant_config("nonexistent_config.yaml")

def test_load_invalid_yaml(tmp_path):
    """Test loading an invalid YAML file."""
    invalid_yaml_content = """
production:
  electrolyzer:
    max_power_mw: 2.5
    base_efficiency: invalid_value
"""
    config_file = tmp_path / "invalid_config.yaml"
    config_file.write_text(invalid_yaml_content)

    with pytest.raises(ConfigurationError):
        load_plant_config(config_file)

def test_load_json_configuration(tmp_path):
    """Test loading JSON configuration."""
    json_content = """
{
  "name": "Test JSON Plant",
  "version": "1.0",
  "production": {
    "electrolyzer": {
      "max_power_mw": 2.0,
      "base_efficiency": 0.6
    }
  },
  "storage": {
    "lp_tanks": {
      "count": 2,
      "capacity_kg": 25.0,
      "pressure_bar": 20.0,
      "temperature_k": 298.15
    },
    "hp_tanks": {
      "count": 4,
      "capacity_kg": 100.0,
      "pressure_bar": 300.0,
      "temperature_k": 298.15
    }
  },
  "compression": {
    "filling_compressor": {
      "max_flow_kg_h": 50.0,
      "inlet_pressure_bar": 20.0,
      "outlet_pressure_bar": 300.0,
      "num_stages": 3,
      "efficiency": 0.7
    },
    "outgoing_compressor": {
      "max_flow_kg_h": 100.0,
      "inlet_pressure_bar": 300.0,
      "outlet_pressure_bar": 800.0,
      "efficiency": 0.7
    }
  },
  "demand": {
    "pattern": "constant",
    "base_demand_kg_h": 25.0
  },
  "energy_price": {
    "source": "constant",
    "constant_price_per_mwh": 50.0
  },
  "pathway": {
    "allocation_strategy": "BALANCED"
  },
  "simulation": {
    "timestep_hours": 1.0,
    "duration_hours": 100,
    "start_hour": 0,
    "checkpoint_interval_hours": 24
  }
}
"""
    config_file = tmp_path / "test_config.json"
    config_file.write_text(json_content)

    config = load_plant_config(config_file)
    assert config.name == "Test JSON Plant"
    assert config.production.electrolyzer.max_power_mw == 2.0
