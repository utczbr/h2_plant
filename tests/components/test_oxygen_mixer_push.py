
import pytest
from h2_plant.components.mixing.oxygen_mixer import OxygenMixer
from h2_plant.core.stream import Stream
from h2_plant.core.component_registry import ComponentRegistry

def test_oxygen_mixer_push_logic():
    # Setup
    mixer = OxygenMixer(capacity_kg=100.0, target_pressure_bar=10.0, target_temperature_c=25.0)
    registry = ComponentRegistry()
    mixer.initialize(dt=1.0, registry=registry)
    
    # Create Input Streams
    # Stream 1: Hot (80C), 10 kg/h
    s1 = Stream(mass_flow_kg_h=10.0, temperature_k=353.15, pressure_pa=10e5, composition={'O2': 1.0})
    
    # Stream 2: Cold (20C), 10 kg/h
    s2 = Stream(mass_flow_kg_h=10.0, temperature_k=293.15, pressure_pa=10e5, composition={'O2': 1.0})
    
    # Push inputs
    accepted1 = mixer.receive_input("oxygen_in", s1, "oxygen")
    accepted2 = mixer.receive_input("oxygen_in", s2, "oxygen")
    
    assert accepted1 == 10.0
    assert accepted2 == 10.0
    
    # Verify buffer
    assert len(mixer._input_buffer) == 2
    
    # Execute Step
    mixer.step(0.0)
    
    # Verify State after mixing
    # Mass should be 20 kg
    assert mixer.mass_kg == 20.0
    
    # Temperature should be average (approx 50C -> 323.15 K) since mass is equal and Cp constant-ish
    # Tolerance 1K
    assert abs(mixer.temperature_k - 323.15) < 1.0
    
    # Verify buffer cleared
    assert len(mixer._input_buffer) == 0

def test_mixer_overflow():
    mixer = OxygenMixer(capacity_kg=10.0) # Small capacity
    registry = ComponentRegistry()
    mixer.initialize(dt=1.0, registry=registry)
    
    s1 = Stream(mass_flow_kg_h=20.0, temperature_k=298.15)
    mixer.receive_input("oxygen_in", s1)
    
    mixer.step(0.0)
    
    assert mixer.mass_kg == 10.0 # Capped
    assert mixer.cumulative_vented_kg == 10.0
