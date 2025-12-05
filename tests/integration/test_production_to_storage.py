from h2_plant.components.production.electrolyzer_source import ElectrolyzerProductionSource
from h2_plant.components.storage.tank_array import TankArray
from h2_plant.core.component_registry import ComponentRegistry

def test_electrolyzer_to_tank_array():
    """Test electrolyzer filling tank array."""
    
    # Setup
    registry = ComponentRegistry()
    
    electrolyzer = ElectrolyzerProductionSource(max_power_mw=2.5)
    tanks = TankArray(n_tanks=4, capacity_kg=200.0, pressure_bar=350)
    
    registry.register('electrolyzer', electrolyzer, component_type='production')
    registry.register('tanks', tanks, component_type='storage')
    
    registry.initialize_all(dt=1.0)
    
    # Simulate 10 hours of production
    for hour in range(10):
        # Produce hydrogen
        electrolyzer.power_input_mw = 2.0
        electrolyzer.step(hour)
        
        # Store in tanks
        h2_produced = electrolyzer.h2_output_kg
        stored, overflow = tanks.fill(h2_produced)
        
        # Verify no overflow
        assert overflow == 0.0
        
        # Step tanks
        tanks.step(hour)
    
    # Verify mass balance
    total_produced = electrolyzer.cumulative_h2_kg
    total_stored = tanks.get_total_mass()
    
    assert abs(total_produced - total_stored) < 0.01  # Mass conserved
