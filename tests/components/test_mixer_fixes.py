
import pytest
from h2_plant.components.mixing.multicomponent_mixer import MultiComponentMixer
from h2_plant.components.mixing.water_mixer import WaterMixer
from h2_plant.components.mixing.oxygen_mixer import OxygenMixer
from h2_plant.core.stream import Stream
from h2_plant.core.component_registry import ComponentRegistry
from unittest.mock import MagicMock, patch

def test_multicomponent_mixer_extreme_temps():
    # Test that UV-Flash converges for very low/high temperatures now
    mixer = MultiComponentMixer(volume_m3=1.0, initial_temperature_k=300.0)
    registry = ComponentRegistry()
    registry.register("mixer", mixer)
    mixer.initialize(dt=1.0, registry=registry)
    
    # 1. Very Low temp (Cryogenic Hydrogen) - 30K
    s_cryo = Stream(mass_flow_kg_h=1000.0, temperature_k=30.0, pressure_pa=1e5, composition={'H2': 1.0})
    mixer.receive_input("inlet", s_cryo)
    mixer.step(0.0)
    assert mixer.temperature_k < 100.0 # Should converge to low temp
    
    # 2. Very High temp (Combustion) - 1500K
    mixer.moles_stored = {k: 0.0 for k in mixer.moles_stored} # Reset
    mixer.total_internal_energy_J = 0.0
    
    s_hot = Stream(mass_flow_kg_h=20.0, temperature_k=1500.0, pressure_pa=1e5, composition={'H2': 1.0})
    mixer.receive_input("inlet", s_hot)
    mixer.step(1.0)
    assert mixer.temperature_k > 1000.0 # Should converge to high temp

def test_water_mixer_error_handling():
    # Verify it returns a defined stream instead of None or crashing on error
    mixer = WaterMixer()
    registry = ComponentRegistry()
    mixer.initialize(dt=1.0, registry=registry)
    
    # Create a stream with invalid properties for CoolProp (e.g. extremely negative pressure or NaN) defined indirectly
    # Easier way: Mock PropsSI to raise exception
    
    with patch('CoolProp.CoolProp.PropsSI', side_effect=ValueError("CoolProp error")):
        s_in = Stream(mass_flow_kg_h=100.0, temperature_k=300.0, pressure_pa=101325, composition={'H2O': 1.0}, phase='liquid')
        mixer.receive_input("inlet_0", s_in, "water")
        
        mixer.step(0.0)
        
        out = mixer.get_output("outlet")
        assert out is not None
        assert out.mass_flow_kg_h == 0.0 # Should be fallback stream

def test_oxygen_mixer_physics():
    # Verify P = nRT/V
    vol = 2.0
    mixer = OxygenMixer(volume_m3=vol, target_pressure_bar=10.0)
    registry = ComponentRegistry()
    mixer.initialize(dt=1.0, registry=registry)
    
    # Add Oxygen
    # 32 kg of O2 = 1000 moles
    s_in = Stream(mass_flow_kg_h=32.0, temperature_k=300.0, pressure_pa=1e5, composition={'O2': 1.0})
    mixer.receive_input("oxygen_in", s_in)
    
    mixer.step(0.0) # 1 hour = 32 kg added
    
    assert mixer.mass_kg == 32.0
    
    # Expected Pressure:
    # P = nRT/V = 1000 * 8.314 * 300 / 2.0 = 2,494,200 / 2 = 1,247,100 Pa
    expected_P = (1000.0 * 8.314 * 300.0) / 2.0
    
    assert abs(mixer.pressure_pa - expected_P) < 1000.0
