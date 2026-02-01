
import pytest
from h2_plant.components.mixing.multicomponent_mixer import MultiComponentMixer
from h2_plant.core.stream import Stream
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.constants import GasConstants

def test_multicomponent_mixer_mixing():
    # Setup
    # 1 m3 volume
    mixer = MultiComponentMixer(volume_m3=1.0, initial_temperature_k=300.0)
    registry = ComponentRegistry()
    registry.register("mixer", mixer)
    mixer.initialize(dt=1.0, registry=registry)
    
    # Create Input Streams
    # Stream 1: Pure H2, 2 kg/h at 300K
    s1 = Stream(mass_flow_kg_h=2.0, temperature_k=300.0, pressure_pa=1e5, composition={'H2': 1.0})
    
    # Stream 2: Pure N2, 28 kg/h at 300K
    s2 = Stream(mass_flow_kg_h=28.0, temperature_k=300.0, pressure_pa=1e5, composition={'N2': 1.0})
    
    # Push inputs
    mixer.receive_input("inlet", s1)
    mixer.receive_input("inlet", s2)
    
    # Step (1 hour)
    mixer.step(0.0)
    
    # Verification
    # H2: 2 kg / 2.016 g/mol 
    h2_mw = GasConstants.SPECIES_DATA['H2']['molecular_weight'] / 1000.0
    expected_h2_moles = 2.0 / h2_mw
    
    # N2: 28 kg / 28.014 g/mol
    n2_mw = GasConstants.SPECIES_DATA['N2']['molecular_weight'] / 1000.0
    expected_n2_moles = 28.0 / n2_mw
    
    # Total moles
    expected_total = expected_h2_moles + expected_n2_moles
    
    assert abs(mixer.moles_stored['H2'] - expected_h2_moles) < 1.0
    assert abs(mixer.moles_stored['N2'] - expected_n2_moles) < 1.0
    
    total_moles = sum(mixer.moles_stored.values())
    assert abs(total_moles - expected_total) < 2.0
    
    # Temperature should increase due to adiabatic compression (filling empty tank)
    # T_final = gamma * T_in approximately. 
    # For H2/N2 mix, gamma is roughly 1.4.
    # Input T = 300K. Expected T ~ 420K.
    # Let's just assert it is > 300K (heating occurred) and < 600K.
    assert mixer.temperature_k > 300.0
    assert mixer.temperature_k < 600.0
    
    # Pressure check
    # P = nRT/V
    expected_P = (total_moles * 8.314 * mixer.temperature_k) / 1.0
    assert abs(mixer.pressure_pa - expected_P) < 1000.0

def test_multicomponent_mixer_energy_balance():
    # Mix Hot H2 and Cold H2
    mixer = MultiComponentMixer(volume_m3=1.0, initial_temperature_k=300.0)
    registry = ComponentRegistry()
    registry.register("mixer", mixer)
    mixer.initialize(dt=1.0, registry=registry)
    
    # Hot H2: 2 kg/h at 400K
    s_hot = Stream(mass_flow_kg_h=2.0, temperature_k=400.0, composition={'H2': 1.0})
    
    # Cold H2: 2 kg/h at 300K
    s_cold = Stream(mass_flow_kg_h=2.0, temperature_k=300.0, composition={'H2': 1.0})
    
    mixer.receive_input("inlet", s_hot)
    mixer.receive_input("inlet", s_cold)
    
    mixer.step(0.0)
    
    # Average Input T = 350K.
    # Adiabatic compression means T_final ~ 1.4 * 350 = 490K
    assert mixer.temperature_k > 400.0
    assert mixer.temperature_k < 550.0
