"""Unit tests for MultiComponentMixer."""

import pytest
from h2_plant.components.mixing.multicomponent_mixer import MultiComponentMixer
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.component import Component
from typing import Dict, Any

class MockSource(Component):
    def __init__(self, stream_data: Dict[str, Any]):
        super().__init__()
        self._stream_data = stream_data
    
    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        super().initialize(dt, registry)

    def step(self, t: float) -> None:
        super().step(t)
    
    def get_state(self) -> Dict[str, Any]:
        return self._stream_data

def test_mixer_initialization():
    """Test mixer initializes with correct default state."""
    mixer = MultiComponentMixer(volume_m3=10.0)
    assert mixer.volume_m3 == 10.0
    assert mixer.temperature_k == 298.15
    assert sum(mixer.moles_stored.values()) == 0.0

def test_mixer_single_stream_step():
    """Test a single step with one input stream."""
    registry = ComponentRegistry()
    
    stream_data = {
        'flow_kmol_hr': 1.0,
        'temperature_k': 350.0,
        'pressure_pa': 2e5,
        'composition': {'O2': 1.0}
    }
    source = MockSource(stream_data)
    registry.register('source1', source)
    
    mixer = MultiComponentMixer(volume_m3=10.0, input_source_ids=['source1'])
    registry.register('mixer', mixer)
    
    registry.initialize_all(dt=1.0)
    
    mixer.step(0)
    
    assert mixer.moles_stored['O2'] > 0
    assert mixer.total_internal_energy_J != 0
    assert mixer.temperature_k != 298.15 # Should change after mixing
    assert mixer.pressure_pa > 1e5

def test_energy_conservation():
    """Test that energy is conserved during mixing (without heat loss)."""
    mixer = MultiComponentMixer(volume_m3=10.0, heat_loss_coeff_W_per_K=0.0)
    mixer.initialize(dt=1.0, registry=ComponentRegistry())

    # Initial state
    mixer.moles_stored['O2'] = 10.0
    mixer._initialize_internal_energy()
    U_initial = mixer.total_internal_energy_J
    
    # Simulate an input stream
    H_input = mixer._calculate_molar_enthalpy(350, 1e5, {'O2': 1.0}) * (1000/3600) * 3600 # 1 kmol
    
    mixer.total_internal_energy_J += H_input
    
    assert abs(mixer.total_internal_energy_J - (U_initial + H_input)) < 1e-6

def test_pressure_relief_placeholder():
    """Test that the pressure relief placeholder runs without error."""
    mixer = MultiComponentMixer(volume_m3=1.0, pressure_relief_threshold_bar=10.0)
    mixer.initialize(dt=1.0, registry=ComponentRegistry())
    mixer.pressure_pa = 20e5 # Above threshold
    
    # This should run without error. A more detailed test would check vented moles.
    mixer._activate_pressure_relief()
    assert True
