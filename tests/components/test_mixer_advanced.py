
import pytest
from h2_plant.components.mixing.oxygen_mixer import OxygenMixer
from h2_plant.components.mixing.water_mixer import WaterMixer
from h2_plant.core.stream import Stream
from h2_plant.core.component_registry import ComponentRegistry
from unittest.mock import patch

def test_water_mixer_validation_and_cleanup():
    mixer = WaterMixer()
    registry = ComponentRegistry()
    mixer.initialize(dt=1.0, registry=registry)
    
    # 1. Test Input Validation (Reject Cryogenic Water)
    s_invalid = Stream(mass_flow_kg_h=100.0, temperature_k=100.0, pressure_pa=1e5, composition={'H2O': 1.0}, phase='liquid')
    accepted = mixer.receive_input("in_1", s_invalid, "water")
    assert accepted == 0.0
    assert len(mixer.inlet_streams) == 0
    
    # 2. Test Stale Stream Cleanup
    s_valid = Stream(mass_flow_kg_h=100.0, temperature_k=300.0, pressure_pa=1e5, composition={'H2O': 1.0}, phase='liquid')
    mixer.receive_input("in_valid", s_valid, "water")
    
    # Simulate source stopped sending (stream stays in dict but has 0 flow if updated, 
    # OR if we manually inject a zero flow stream to simulate 'end of flow')
    # The component checks `value.mass_flow_kg_h > 0` in cleanup.
    # We must simulate the source sending a 0-flow update OR the component cleaning up old ones?
    # Actually the code `self.inlet_streams[port_name] = value` overwrites. 
    # If a source sends 0 flow, it gets stored?
    # Let's check: receive_input overwrites. Cleaning happens in step.
    
    s_zero = Stream(mass_flow_kg_h=0.0, temperature_k=300.0, pressure_pa=1e5, composition={'H2O': 1.0})
    mixer.receive_input("in_valid", s_zero, "water") # Update to zero
    
    # Step should remove it
    mixer.step(0.0)
    assert len(mixer.inlet_streams) == 0


def test_oxygen_mixer_damping_and_capacity():
    # 1. Capacity Init
    # Vol=10m3, P=10bar. 
    # Expected Capacity ~ 10 * 1.429 * (10/1) ~ 142.9 kg
    mixer = OxygenMixer(volume_m3=10.0, target_pressure_bar=10.0)
    assert mixer.capacity_kg > 100.0
    
    # 2. Damping
    # Add a lot of mass at high temp.
    # Initial T = 25C (298K). Input T = 1000K.
    # Tau=5s. dt=1s. alpha = 1/5 = 0.2.
    # T_new = 0.2 * 1000 + 0.8 * 298 = 200 + 238.4 = 438.4 K
    
    mixer.dt = 1.0/3600.0 # 1 second
    registry = ComponentRegistry()
    mixer.initialize(1.0/3600.0, registry)
    
    # Start with some mass so damping applies
    mixer.mass_kg = 10.0 
    
    s_hot = Stream(mass_flow_kg_h=3600.0, temperature_k=1000.0, pressure_pa=10e5, composition={'O2': 1.0})
    # 1 sec flow = 1 kg. Input mass < Stored mass (10kg). Damping applies.
    mixer.receive_input("oxygen_in", s_hot, "oxygen")
    
    mixer.step(0.0)
    
    # Check if T is damped (should be much less than 1000K)
    assert mixer.temperature_k < 800.0
    assert mixer.temperature_k > 300.0

