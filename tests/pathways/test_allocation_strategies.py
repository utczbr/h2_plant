import pytest
from h2_plant.pathways.allocation_strategies import CostOptimalStrategy, PriorityGridStrategy, PriorityATRStrategy, BalancedStrategy, EmissionsOptimalStrategy
from h2_plant.core.enums import AllocationStrategy

def test_cost_optimal_allocation():
    """Test cost-optimal strategy chooses cheapest pathway."""
    strategy = CostOptimalStrategy()
    
    pathway_states = {
        'electrolyzer': {
            'max_production_kg': 50.0,
            'specific_cost_per_kg': 5.00  # More expensive
        },
        'atr': {
            'max_production_kg': 75.0,
            'specific_cost_per_kg': 2.00  # Cheaper
        }
    }
    
    allocation = strategy.allocate(
        total_demand_kg=100.0,
        pathway_states=pathway_states,
        current_hour=0
    )
    
    # Should allocate to ATR first (cheaper)
    assert allocation['atr'] == 75.0  # Maxed out ATR
    assert allocation['electrolyzer'] == 25.0  # Remainder to electrolyzer

def test_priority_grid_allocation():
    """Test priority-grid strategy maximizes electrolyzer usage."""
    strategy = PriorityGridStrategy()
    
    pathway_states = {
        'electrolyzer': {'max_production_kg': 50.0},
        'atr': {'max_production_kg': 75.0}
    }
    
    allocation = strategy.allocate(
        total_demand_kg=100.0,
        pathway_states=pathway_states,
        current_hour=0
    )
    
    # Should allocate to electrolyzer first
    assert allocation['electrolyzer'] == 50.0  # Maxed out electrolyzer
    assert allocation['atr'] == 50.0  # Remainder to ATR

def test_priority_atr_allocation():
    """Test priority-atr strategy maximizes ATR usage."""
    strategy = PriorityATRStrategy()
    
    pathway_states = {
        'electrolyzer': {'max_production_kg': 50.0},
        'atr': {'max_production_kg': 75.0}
    }
    
    allocation = strategy.allocate(
        total_demand_kg=100.0,
        pathway_states=pathway_states,
        current_hour=0
    )
    
    # Should allocate to ATR first
    assert allocation['atr'] == 75.0  # Maxed out ATR
    assert allocation['electrolyzer'] == 25.0  # Remainder to electrolyzer

def test_balanced_allocation():
    """Test balanced strategy splits demand."""
    strategy = BalancedStrategy()
    
    pathway_states = {
        'electrolyzer': {'max_production_kg': 60.0},
        'atr': {'max_production_kg': 60.0}
    }
    
    allocation = strategy.allocate(
        total_demand_kg=100.0,
        pathway_states=pathway_states,
        current_hour=0
    )
    
    # Should split 50/50
    assert allocation['electrolyzer'] == 50.0
    assert allocation['atr'] == 50.0

def test_emissions_optimal_allocation():
    """Test emissions-optimal strategy chooses greenest pathway."""
    strategy = EmissionsOptimalStrategy()
    
    pathway_states = {
        'electrolyzer': {
            'max_production_kg': 50.0,
            'emissions_factor': 0.0 # Green H2
        },
        'atr': {
            'max_production_kg': 75.0,
            'emissions_factor': 10.5 # Grey H2
        }
    }
    
    allocation = strategy.allocate(
        total_demand_kg=100.0,
        pathway_states=pathway_states,
        current_hour=0
    )
    
    # Should allocate to electrolyzer first (lower emissions)
    assert allocation['electrolyzer'] == 50.0  # Maxed out electrolyzer
    assert allocation['atr'] == 50.0  # Remainder to ATR

def test_get_allocation_strategy_factory():
    """Test the get_allocation_strategy factory function."""
    from h2_plant.pathways.allocation_strategies import get_allocation_strategy
    
    strategy = get_allocation_strategy(AllocationStrategy.COST_OPTIMAL)
    assert isinstance(strategy, CostOptimalStrategy)

    strategy = get_allocation_strategy(AllocationStrategy.BALANCED)
    assert isinstance(strategy, BalancedStrategy)

    with pytest.raises(ValueError):
        get_allocation_strategy(999) # Invalid strategy
