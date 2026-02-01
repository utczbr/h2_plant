import pytest
import numpy as np
from h2_plant.components.production.electrolyzer_source import ElectrolyzerProductionSource
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.enums import ProductionState


def test_electrolyzer_initialization():
    """Test electrolyzer initializes correctly."""
    elec = ElectrolyzerProductionSource(max_power_mw=2.5, base_efficiency=0.65)
    registry = ComponentRegistry()
    
    elec.initialize(dt=1.0, registry=registry)
    
    assert elec._initialized
    assert elec.state == ProductionState.OFFLINE


def test_electrolyzer_production_calculation():
    """Test hydrogen production calculation."""
    elec = ElectrolyzerProductionSource(max_power_mw=2.5, base_efficiency=0.65)
    registry = ComponentRegistry()
    elec.initialize(dt=1.0, registry=registry)
    
    # Set 2 MW input (80% load)
    elec.power_input_mw = 2.0
    elec.step(0.0)
    
    # Check production
    assert elec.h2_output_kg > 0
    assert elec.state == ProductionState.RUNNING
    
    # Check oxygen byproduct
    expected_o2 = elec.h2_output_kg * 7.94
    assert abs(elec.o2_output_kg - expected_o2) < 0.01


def test_electrolyzer_below_min_load():
    """Test electrolyzer shuts down below minimum load."""
    elec = ElectrolyzerProductionSource(
        max_power_mw=2.5,
        min_load_factor=0.20
    )
    registry = ComponentRegistry()
    elec.initialize(dt=1.0, registry=registry)
    
    # Set below min load (10% = 0.25 MW < 20% min)
    elec.power_input_mw = 0.25
    elec.step(0.0)
    
    assert elec.state == ProductionState.OFFLINE
    assert elec.h2_output_kg == 0.0


def test_electrolyzer_state_serialization():
    """Test get_state returns complete state."""
    elec = ElectrolyzerProductionSource(max_power_mw=2.5)
    registry = ComponentRegistry()
    elec.initialize(dt=1.0, registry=registry)
    
    elec.power_input_mw = 2.0
    elec.step(0.0)
    
    state = elec.get_state()
    
    assert 'h2_output_kg' in state
    assert 'efficiency' in state
    assert 'state' in state
    assert 'cumulative_h2_kg' in state
