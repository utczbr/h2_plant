import pytest
from h2_plant.components.mixing.h2_distribution import H2Distribution

class MockRegistry:
    pass

@pytest.fixture
def distribution():
    dist = H2Distribution(num_inputs=3)
    dist.initialize(dt=1.0, registry=MockRegistry())
    return dist

def test_dual_pathway_separation(distribution):
    """Test that electrolysis and ATR flows are tracked separately."""
    # Inputs: PEM=10, SOEC=20, ATR=50
    distribution.add_inlet_flow(10.0, 0)
    distribution.add_inlet_flow(20.0, 1)
    distribution.add_inlet_flow(50.0, 2)
    
    distribution.step(0.0)
    
    state = distribution.get_state()
    
    # Electrolysis should be PEM + SOEC
    assert state['electrolysis_flow_kg_h'] == 30.0
    # ATR should be separate
    assert state['atr_flow_kg_h'] == 50.0
    # Total should be sum of all
    assert state['total_h2_output_kg_h'] == 80.0

def test_cumulative_tracking(distribution):
    """Test cumulative mass integration over time."""
    # Step 1: 1 hour at 10kg/h PEM
    distribution.add_inlet_flow(10.0, 0)
    distribution.step(0.0)
    
    assert distribution.cumulative_electrolysis_kg == 10.0
    
    # Step 2: 1 hour at 10kg/h PEM + 50kg/h ATR
    distribution.add_inlet_flow(50.0, 2)
    distribution.step(1.0)
    
    assert distribution.cumulative_electrolysis_kg == 20.0  # 10 + 10
    assert distribution.cumulative_atr_kg == 50.0           # 0 + 50
    assert distribution.cumulative_h2_kg == 70.0            # 10 + 60

def test_atr_disabled(distribution):
    """Test behavior when ATR input is zero."""
    distribution.add_inlet_flow(50.0, 0) # PEM only
    distribution.step(0.0)
    
    assert distribution.electrolysis_flow_kg_h == 50.0
    assert distribution.atr_flow_kg_h == 0.0

def test_state_serialization(distribution):
    """Test that state dictionary contains all required fields."""
    distribution.add_inlet_flow(10.0, 0)
    distribution.add_inlet_flow(20.0, 1)
    distribution.add_inlet_flow(30.0, 2)
    distribution.step(0.0)
    
    state = distribution.get_state()
    
    # Check all required fields exist
    assert 'inlet_flows_kg_h' in state
    assert 'total_h2_output_kg_h' in state
    assert 'electrolysis_flow_kg_h' in state
    assert 'atr_flow_kg_h' in state
    assert 'cumulative_h2_kg' in state
    assert 'cumulative_electrolysis_kg' in state
    assert 'cumulative_atr_kg' in state
    assert 'emissions_metadata' in state
    
    # Check emissions metadata structure
    metadata = state['emissions_metadata']
    assert 'electrolysis' in metadata
    assert 'atr' in metadata
    assert metadata['electrolysis']['co2_factor'] == 0.0
    assert metadata['atr']['co2_factor'] == 10.5
