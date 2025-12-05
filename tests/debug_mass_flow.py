"""
Debug script to trace mass flow step-by-step
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
from h2_plant.components.compression.filling_compressor import FillingCompressor
from h2_plant.components.production.electrolyzer_source import ElectrolyzerProductionSource

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_mass_flow():
    registry = ComponentRegistry()
    
    # Create components
    electrolyzer = ElectrolyzerProductionSource(max_power_mw=2.5)
    lp_tank = TankArray(n_tanks=2, capacity_kg=100.0, pressure_bar=30.0)
    compressor = FillingCompressor(max_flow_kg_h=50.0)
    hp_tank = TankArray(n_tanks=2, capacity_kg=100.0, pressure_bar=350.0)
    
    registry.register('electrolyzer', electrolyzer, 'production')
    registry.register('lp_storage', lp_tank, 'storage')
    registry.register('compressor', compressor, 'compression')
    registry.register('hp_storage', hp_tank, 'storage')
    
    registry.initialize_all(dt=1.0)
    
    # Set power
    electrolyzer.power_input_mw = 1.25
    
    # Create topology
    topology = [
        ConnectionConfig('electrolyzer', 'h2_out', 'lp_storage', 'h2_in', 'hydrogen'),
        ConnectionConfig('lp_storage', 'h2_out', 'compressor', 'h2_in', 'hydrogen'),
        ConnectionConfig('compressor', 'h2_out', 'hp_storage', 'h2_in', 'hydrogen')
    ]
    
    flow_network = FlowNetwork(registry, topology)
    flow_network.initialize()
    
    # Run 3 steps with detailed logging
    for hour in range(3):
        print("\\n" + "="*60)
        print(f"HOUR {hour}")
        print("="*60)
        
        # Step components first
        print("\\nCalling step() on all components...")
        registry.step_all(hour)
        
        print(f"After step:")
        print(f"  Electrolyzer H2 rate: {electrolyzer.h2_stream.mass_flow_kg_h if electrolyzer.h2_stream else 0} kg/h")
        print(f"  LP mass: {lp_tank.get_total_mass()} kg")
        print(f"  Compressor transfer: {compressor.transfer_mass_kg} kg, actual: {compressor.actual_mass_transferred_kg} kg")
        print(f"  HP mass: {hp_tank.get_total_mass()} kg")
        
        # Now execute flows
        print("\\nExecuting flows...")
        flow_network.execute_flows(hour)
        
        print(f"After flows:")
        print(f"  LP mass: {lp_tank.get_total_mass()} kg")
        print(f"  Compressor transfer: {compressor.transfer_mass_kg} kg")
        print(f"  HP mass: {hp_tank.get_total_mass()} kg")
        
        print(f"\\nCumulative:")
        print(f"  Electrolyzer produced: {electrolyzer.cumulative_h2_kg} kg")
        print(f"  HP stored: {hp_tank.get_total_mass()} kg")

if __name__ == "__main__":
    test_mass_flow()
