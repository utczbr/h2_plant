from h2_plant.config.plant_builder import PlantBuilder
from h2_plant.config.plant_config import (
    PlantConfig, ProductionConfig, ElectrolyzerConfig,
    StorageConfig, TankArrayConfig, CompressionConfig, CompressorConfig, OutgoingCompressorConfig,
    DemandConfig, EnergyPriceConfig, PathwayConfig, SimulationConfig,
    SourceIsolatedStorageConfig, ATRConfig
)
from h2_plant.core.enums import AllocationStrategy
from h2_plant.core.component_registry import ComponentRegistry
import pytest


def test_plant_builder_from_config():
    """Test PlantBuilder constructs plant correctly from a PlantConfig object."""
    
    config = PlantConfig(
        name="Test Plant From Config",
        production=ProductionConfig(
            electrolyzer=ElectrolyzerConfig(max_power_mw=2.5, base_efficiency=0.65)
        ),
        storage=StorageConfig(
            lp_tanks=TankArrayConfig(count=2, capacity_kg=50.0, pressure_bar=30.0),
            hp_tanks=TankArrayConfig(count=4, capacity_kg=200.0, pressure_bar=350.0)
        ),
        compression=CompressionConfig(
            filling_compressor=CompressorConfig(max_flow_kg_h=100.0, inlet_pressure_bar=30.0, outlet_pressure_bar=350.0),
            outgoing_compressor=OutgoingCompressorConfig(max_flow_kg_h=200.0, inlet_pressure_bar=350.0, outlet_pressure_bar=900.0)
        ),
        demand=DemandConfig(pattern="constant", base_demand_kg_h=50.0),
        energy_price=EnergyPriceConfig(source="constant", constant_price_per_mwh=50.0),
        pathway=PathwayConfig(allocation_strategy=AllocationStrategy.BALANCED),
        simulation=SimulationConfig(duration_hours=100)
    )
    
    plant = PlantBuilder.from_config(config)
    
    # Verify components registered
    assert plant.registry.has('electrolyzer')
    assert plant.registry.has('lut_manager')
    assert plant.registry.has('lp_tanks')
    assert plant.registry.has('hp_tanks')
    assert plant.registry.has('filling_compressor')
    assert plant.registry.has('outgoing_compressor')
    assert plant.registry.has('demand_scheduler')
    assert plant.registry.has('energy_price_tracker')
    assert plant.registry.get_component_count() > 0


def test_plant_builder_from_file(tmp_path):
    """Test PlantBuilder constructs plant correctly from a configuration file."""
    
    yaml_content = """
name: "File Test Plant"
version: "1.0"
production:
  electrolyzer:
    enabled: true
    max_power_mw: 1.0
    base_efficiency: 0.6
storage:
  lp_tanks:
    count: 1
    capacity_kg: 10.0
    pressure_bar: 20.0
    temperature_k: 298.15
  hp_tanks:
    count: 2
    capacity_kg: 50.0
    pressure_bar: 300.0
    temperature_k: 298.15
compression:
  filling_compressor:
    max_flow_kg_h: 10.0
    inlet_pressure_bar: 20.0
    outlet_pressure_bar: 300.0
    num_stages: 2
    efficiency: 0.7
  outgoing_compressor:
    max_flow_kg_h: 20.0
    inlet_pressure_bar: 300.0
    outlet_pressure_bar: 800.0
    efficiency: 0.7
demand:
  pattern: "constant"
  base_demand_kg_h: 10.0
energy_price:
  source: "constant"
  constant_price_per_mwh: 40.0
pathway:
  allocation_strategy: "PRIORITY_ATR"
  priority_source: "atr"
simulation:
  timestep_hours: 1.0
  duration_hours: 50
  start_hour: 0
  checkpoint_interval_hours: 10
"""
    config_file = tmp_path / "file_test_plant.yaml"
    config_file.write_text(yaml_content)

    plant = PlantBuilder.from_file(config_file)
    
    assert plant.config.name == "File Test Plant"
    assert plant.registry.has('electrolyzer')
    assert plant.registry.has('lp_tanks')
    assert plant.registry.get_component_count() > 0


def test_plant_builder_initializes_components():
    """Test PlantBuilder components can be initialized."""
    
    # Create a minimal config for initialization test
    config = PlantConfig(
        name="Init Test Plant",
        production=ProductionConfig(
            electrolyzer=ElectrolyzerConfig(max_power_mw=0.5, base_efficiency=0.6)
        ),
        storage=StorageConfig(
            lp_tanks=TankArrayConfig(count=1, capacity_kg=10.0, pressure_bar=20.0),
            hp_tanks=TankArrayConfig(count=1, capacity_kg=20.0, pressure_bar=200.0)
        ),
        compression=CompressionConfig(
            filling_compressor=CompressorConfig(max_flow_kg_h=5.0, inlet_pressure_bar=20.0, outlet_pressure_bar=200.0),
            outgoing_compressor=OutgoingCompressorConfig(max_flow_kg_h=10.0, inlet_pressure_bar=200.0, outlet_pressure_bar=800.0)
        ),
        demand=DemandConfig(pattern="constant", base_demand_kg_h=5.0),
        energy_price=EnergyPriceConfig(source="constant", constant_price_per_mwh=30.0),
        pathway=PathwayConfig(allocation_strategy=AllocationStrategy.COST_OPTIMAL),
        simulation=SimulationConfig(duration_hours=10)
    )
    
    plant = PlantBuilder.from_config(config)
    
    # Should initialize without errors
    plant.registry.initialize_all(dt=1.0)
    
    # Verify initialization
    electrolyzer = plant.registry.get('electrolyzer')
    assert electrolyzer._initialized
    
    lp_tanks = plant.registry.get('lp_tanks')
    assert lp_tanks._initialized
