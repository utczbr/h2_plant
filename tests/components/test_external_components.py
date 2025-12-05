import pytest
from typing import Dict, Any
from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.components.external.oxygen_source import ExternalOxygenSource
from h2_plant.components.external.heat_source import ExternalHeatSource
from h2_plant.components.mixing.oxygen_mixer import OxygenMixer
from h2_plant.components.storage.battery_storage import BatteryStorage, BatteryMode

# --- Tests for ExternalOxygenSource ---

def test_ext_o2_source_fixed_flow():
    comp = ExternalOxygenSource(mode="fixed_flow", flow_rate_kg_h=100)
    comp.initialize(dt=1.0, registry=ComponentRegistry())
    comp.step(t=0)
    assert comp.o2_output_kg == 100.0

def test_ext_o2_source_cumulative():
    comp = ExternalOxygenSource(mode="fixed_flow", flow_rate_kg_h=50, cost_per_kg=0.1)
    comp.initialize(dt=1.0, registry=ComponentRegistry())
    comp.step(t=0)
    comp.step(t=1)
    assert comp.cumulative_o2_kg == 100.0
    assert comp.cumulative_cost == 10.0

# --- Tests for ExternalHeatSource ---

def test_ext_heat_source_demand():
    comp = ExternalHeatSource(thermal_power_kw=1000)
    comp.initialize(dt=1.0, registry=ComponentRegistry())
    comp.set_demand(500)
    comp.step(t=0)
    assert comp.current_power_kw == 500.0
    assert comp.heat_output_kwh == 500.0

def test_ext_heat_source_availability():
    # Set availability to 0, should produce no heat
    comp = ExternalHeatSource(availability_factor=0.0)
    comp.initialize(dt=1.0, registry=ComponentRegistry())
    comp.set_demand(500)
    comp.step(t=0)
    assert comp.current_power_kw == 0.0

# --- Tests for OxygenMixer ---

def test_mixer_init():
    comp = OxygenMixer(input_source_ids=["o2_source_1"])
    assert "o2_source_1" in comp.input_source_ids

# Define a mock component that inherits from the Component ABC
class MockOxygenSource(Component):
    def __init__(self, o2_output=0.0):
        super().__init__()
        self._o2_output = o2_output

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        super().initialize(dt, registry)

    def step(self, t: float) -> None:
        super().step(t)

    def get_state(self) -> Dict[str, Any]:
        return {**super().get_state(), 'o2_output_kg': self._o2_output}


def test_mixer_collect_flow():
    """Test that the mixer correctly collects flow from a valid component."""
    registry = ComponentRegistry()
    
    # Use the mock component that respects the ABC
    mock_source = MockOxygenSource(o2_output=10.0)
    registry.register("o2_source_1", mock_source)
    
    mixer = OxygenMixer(input_source_ids=["o2_source_1"])
    registry.register("mixer", mixer)
    
    # Initialize all components in the registry
    registry.initialize_all(dt=1.0)
    
    # In a real run, the engine's step_all would handle this. Here we do it manually.
    mock_source.step(t=0)
    mixer.step(t=0)
    
    assert mixer.mass_kg == 10.0
    assert mixer.cumulative_input_kg == 10.0


# --- Tests for BatteryStorage ---

def test_battery_initial_soc():
    comp = BatteryStorage(capacity_kwh=1000, initial_soc=0.6)
    assert comp.energy_kwh == 600
    assert comp.soc == 0.6

def test_battery_charging():
    comp = BatteryStorage(capacity_kwh=1000, max_charge_power_kw=100, initial_soc=0.5)
    comp.initialize(dt=1.0, registry=ComponentRegistry())
    
    comp.set_grid_status(available=True, power_kw=100)
    comp.set_load_demand(0)
    
    comp.step(t=0)
    
    assert comp.mode == BatteryMode.CHARGING
    assert comp.charge_power_kw == 100.0
    # energy_stored = 100 * 1.0 * 0.95 (charge_efficiency)
    assert comp.energy_kwh == 500 + (100 * 0.95)

def test_battery_discharging():
    comp = BatteryStorage(capacity_kwh=1000, max_discharge_power_kw=200, initial_soc=0.8)
    comp.initialize(dt=1.0, registry=ComponentRegistry())

    comp.set_grid_status(available=False, power_kw=0)
    comp.set_load_demand(100) # 100 kW load
    
    comp.step(t=0)

    assert comp.mode == BatteryMode.DISCHARGING
    # energy_delivered = 100 * 1.0
    assert abs(comp.discharge_power_kw - 100.0) < 1e-6
    # energy_drawn = 100 / 0.95 (discharge_efficiency)
    assert comp.energy_kwh == 800 - (100 / 0.95)
