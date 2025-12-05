"""
Test individual component pairs to isolate where mass is duplicated
"""

import logging
import sys
from pathlib import Path

project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.simulation.flow_network import FlowNetwork
from h2_plant.config.plant_config import ConnectionConfig
from h2_plant.components.storage.tank_array import TankArray
from h2_plant.components.production.electrolyzer_source import ElectrolyzerProductionSource

logging.basicConfig(level=logging.ERROR)

def test_electrolyzer_to_tank():
    """Test electrolyzer → tank connection only"""
    print("\n" + "="*60)
    print("TEST 1: Electrolyzer → Tank")
    print("="*60)
    
    registry = ComponentRegistry()
    
    electrolyzer = ElectrolyzerProductionSource(max_power_mw=2.5)
    tank = TankArray(n_tanks=2, capacity_kg=100.0, pressure_bar=30.0)
    
    registry.register('electrolyzer', electrolyzer, 'production')
    registry.register('tank', tank, 'storage')
    
    registry.initialize_all(dt=1.0)
    
    electrolyzer.power_input_mw = 1.25
    
    topology = [ConnectionConfig('electrolyzer', 'h2_out', 'tank', 'h2_in', 'hydrogen')]
    flow_network = FlowNetwork(registry, topology)
    flow_network.initialize()
    
    # Run 10 steps
    for hour in range(10):
        registry.step_all(hour)
        flow_network.execute_flows(hour)
    
    produced = electrolyzer.cumulative_h2_kg
    stored = tank.get_total_mass()
    error = stored - produced
    
    print(f"Produced: {produced:.2f} kg")
    print(f"Stored: {stored:.2f} kg")
    print(f"Error: {error:.2f} kg ({error/produced*100:.1f}%)")
    
    if abs(error) < 0.01:
        print("✓ PERFECT BALANCE")
        return True
    else:
        print(f"✗ MASS IMBALANCE")
        return False

if __name__ == "__main__":
    test_electrolyzer_to_tank()
