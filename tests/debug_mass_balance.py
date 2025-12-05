"""
Comprehensive mass balance investigation
Tracks every mass transfer to find duplication source
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

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class MassTracker:
    """Track all mass transfers"""
    def __init__(self):
        self.transfers = []
        self.component_states = {}
    
    def log_state(self, hour, phase, components):
        """Log current state of all components"""
        state = {
            'hour': hour,
            'phase': phase,
            'electrolyzer_cumulative': components['electrolyzer'].cumulative_h2_kg,
            'lp_mass': components['lp_storage'].get_total_mass(),
            'compressor_transfer': components['compressor'].transfer_mass_kg,
            'compressor_actual': components['compressor'].actual_mass_transferred_kg,
            'compressor_cumulative': components['compressor'].cumulative_mass_kg,
            'hp_mass': components['hp_storage'].get_total_mass(),
        }
        self.component_states[f"h{hour}_{phase}"] = state
        return state
    
    def analyze(self):
        """Analyze where mass is being duplicated"""
        print("\n" + "="*80)
        print("MASS BALANCE ANALYSIS")
        print("="*80)
        
        for key in sorted(self.component_states.keys()):
            state = self.component_states[key]
            print(f"\n{key}:")
            print(f"  Electrolyzer total: {state['electrolyzer_cumulative']:.2f} kg")
            print(f"  LP storage: {state['lp_mass']:.2f} kg")
            print(f"  Compressor transfer: {state['compressor_transfer']:.2f} kg")
            print(f"  Compressor actual: {state['compressor_actual']:.2f} kg")
            print(f"  Compressor cumulative: {state['compressor_cumulative']:.2f} kg")
            print(f"  HP storage: {state['hp_mass']:.2f} kg")
            
            # Check balance
            total_in_system = (state['lp_mass'] + state['hp_mass'])
            error = total_in_system - state['electrolyzer_cumulative']
            if abs(error) > 0.01:
                print(f"  ⚠️  BALANCE ERROR: {error:.2f} kg ({error/state['electrolyzer_cumulative']*100:.1f}% if state['electrolyzer_cumulative'] > 0 else 'N/A')")

def test_with_logging():
    tracker = MassTracker()
    
    # Create components
    registry = ComponentRegistry()
    
    electrolyzer = ElectrolyzerProductionSource(max_power_mw=2.5)
    lp_tank = TankArray(n_tanks=2, capacity_kg=100.0, pressure_bar=30.0)
    compressor = FillingCompressor(max_flow_kg_h=50.0)
    hp_tank = TankArray(n_tanks=2, capacity_kg=200.0, pressure_bar=350.0)
    
    registry.register('electrolyzer', electrolyzer, 'production')
    registry.register('lp_storage', lp_tank, 'storage')
    registry.register('compressor', compressor, 'compression')
    registry.register('hp_storage', hp_tank, 'storage')
    
    registry.initialize_all(dt=1.0)
    
    electrolyzer.power_input_mw = 1.25
    
    # Create topology
    topology = [
        ConnectionConfig('electrolyzer', 'h2_out', 'lp_storage', 'h2_in', 'hydrogen'),
        ConnectionConfig('lp_storage', 'h2_out', 'compressor', 'h2_in', 'hydrogen'),
        ConnectionConfig('compressor', 'h2_out', 'hp_storage', 'h2_in', 'hydrogen')
    ]
    
    flow_network = FlowNetwork(registry, topology)
    flow_network.initialize()
    
    components = {
        'electrolyzer': electrolyzer,
        'lp_storage': lp_tank,
        'compressor': compressor,
        'hp_storage': hp_tank
    }
    
    # Monkey-patch FlowNetwork to log transfers
    original_execute = flow_network._execute_single_flow
    
    def logged_execute(conn, t):
        source = registry.get(conn.source_id)
        target = registry.get(conn.target_id)
        
        # Get state before
        if hasattr(source, 'get_total_mass'):
            source_before = source.get_total_mass()
        else:
            source_before = getattr(source, 'cumulative_h2_kg', 0)
            
        if hasattr(target, 'get_total_mass'):
            target_before = target.get_total_mass()
        else:
            target_before = getattr(target, 'actual_mass_transferred_kg', 0)
        
        # Execute the flow
        result = original_execute(conn, t)
        
        # Get state after
        if hasattr(source, 'get_total_mass'):
            source_after = source.get_total_mass()
        else:
            source_after = getattr(source, 'cumulative_h2_kg', 0)
            
        if hasattr(target, 'get_total_mass'):
            target_after = target.get_total_mass()
        else:
            target_after = getattr(target, 'actual_mass_transferred_kg', 0)
        
        # Log the transfer
        source_delta = source_after - source_before
        target_delta = target_after - target_before
        
        if abs(target_delta) > 0.001:
            print(f"    {conn.source_id} → {conn.target_id}: "
                  f"source Δ={source_delta:+.3f}, target Δ={target_delta:+.3f}")
        
        return result
    
    flow_network._execute_single_flow = logged_execute
    
    # Run 5 steps
    for hour in range(5):
        print(f"\n{'='*80}\nHOUR {hour}\n{'='*80}")
        
        tracker.log_state(hour, 'start', components)
        
        print(f"\n  STEP():")
        registry.step_all(hour)
        tracker.log_state(hour, 'after_step', components)
        
        print(f"\n  EXECUTE_FLOWS():")
        flow_network.execute_flows(hour)
        tracker.log_state(hour, 'after_flows', components)
    
    # Final analysis
    tracker.analyze()
    
    # Summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"Electrolyzer produced: {electrolyzer.cumulative_h2_kg:.2f} kg")
    print(f"LP storage: {lp_tank.get_total_mass():.2f} kg")
    print(f"HP storage: {hp_tank.get_total_mass():.2f} kg")
    print(f"Total stored: {lp_tank.get_total_mass() + hp_tank.get_total_mass():.2f} kg")
    print(f"Compressor cumulative: {compressor.cumulative_mass_kg:.2f} kg")
    
    error = (lp_tank.get_total_mass() + hp_tank.get_total_mass()) - electrolyzer.cumulative_h2_kg
    print(f"\nMass balance error: {error:.2f} kg ({error/electrolyzer.cumulative_h2_kg*100:.1f}%)")

if __name__ == "__main__":
    test_with_logging()
