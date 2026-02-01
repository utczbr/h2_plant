import pytest
from h2_plant.components.storage.source_isolated_tanks import SourceIsolatedTanks, SourceTag

@pytest.fixture
def isolated_storage():
    sources = {
        'electrolyzer': SourceTag('elec_1', 'electrolyzer', 0.0),
        'atr': SourceTag('atr_1', 'atr', 10.5)
    }
    # Small tanks for testing: 4 tanks, 10kg each = 40kg per source
    return SourceIsolatedTanks(sources, tanks_per_source=4, capacity_kg=10.0, pressure_bar=350)

def test_fill_separation(isolated_storage):
    """Test that filling one source does not affect the other."""
    isolated_storage.initialize(1.0, None)
    
    # Fill Electrolyzer
    stored, _ = isolated_storage.fill('electrolyzer', 20.0)
    assert stored == 20.0
    
    # Check states
    assert isolated_storage.get_mass_by_source('electrolyzer') == 20.0
    assert isolated_storage.get_mass_by_source('atr') == 0.0
    
    # Fill ATR
    isolated_storage.fill('atr', 15.0)
    assert isolated_storage.get_mass_by_source('atr') == 15.0

def test_discharge_priority(isolated_storage):
    """Test discharge priority (lowest emissions first)."""
    isolated_storage.initialize(1.0, None)
    
    # Fill both: 20kg Green (0.0 CO2), 20kg Blue (10.5 CO2)
    isolated_storage.fill('electrolyzer', 20.0)
    isolated_storage.fill('atr', 20.0)
    
    # Discharge 10kg - Should come from Green
    discharged, source = isolated_storage.discharge(10.0)
    assert discharged == 10.0
    assert source == 'electrolyzer'
    assert isolated_storage.get_mass_by_source('electrolyzer') == 10.0
    
    # Discharge 20kg - Should exhaust Green (10kg) then take Blue (10kg)
    discharged, source = isolated_storage.discharge(20.0)
    assert discharged == 20.0
    # The primary source returned is the one requested or prioritized, 
    # but internal state should reflect split
    assert isolated_storage.get_mass_by_source('electrolyzer') == 0.0
    assert isolated_storage.get_mass_by_source('atr') == 10.0

def test_weighted_emissions_factor(isolated_storage):
    """Test mass-weighted emissions calculation."""
    isolated_storage.initialize(1.0, None)
    
    # Fill 30kg Green (0.0) and 10kg Blue (10.5)
    isolated_storage.fill('electrolyzer', 30.0)
    isolated_storage.fill('atr', 10.0)
    
    # Weighted average = (30*0.0 + 10*10.5) / 40 = 2.625
    weighted_ef = isolated_storage.get_weighted_emissions_factor()
    assert abs(weighted_ef - 2.625) < 0.01

def test_unknown_source_error(isolated_storage):
    """Test that filling unknown source raises error."""
    isolated_storage.initialize(1.0, None)
    
    with pytest.raises(ValueError, match="Unknown source"):
        isolated_storage.fill('unknown_source', 10.0)
