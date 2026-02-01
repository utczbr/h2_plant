
import unittest
import numpy as np
import logging
from unittest.mock import MagicMock
from h2_plant.control.engine_dispatch import HybridArbitrageEngineStrategy
from h2_plant.components.delivery.discharge_station import DischargeStation
from h2_plant.core.component_registry import ComponentRegistry

# Setup logging
logging.basicConfig(level=logging.INFO)



from h2_plant.core.component import Component

class MockComponent(Component):
    def __init__(self, component_id, **kwargs):
        super().__init__(**kwargs)
        self.set_component_id(component_id)
        self.efficiency = 1.0

    def initialize(self, dt, registry):
        super().initialize(dt, registry)
        
    def step(self, t):
        super().step(t)
        
    def get_state(self):
        return {}
        
    def receive_input(self, port, value, type):
        pass

class TestDemandRecording(unittest.TestCase):
    def test_demand_signal_recorded(self):
        # 1. Setup Registry with DischargeStation
        registry = ComponentRegistry()
        
        # Create station
        config = {
            'n_stations': 5,
            'truck_capacity_kg': 280.0,
            'delivery_pressure_bar': 500.0,
            'max_fill_rate_kg_min': 1.5,
            'h_in_day_max': 16.0, # Scheduled mode
            'station_id': 1,
            'min_fill_rate_kg_min': 0.583 # Add missing param
        }
        station = DischargeStation(component_id='Truck_Station_1', **config)
        station.initialize(dt=1.0/60.0, registry=registry) # 1 min dt
        registry.register('Truck_Station_1', station)
        

        # Mock Transformers (required by Strategy)
        registry.register('SOEC_Transformer', MockComponent('SOEC_Transformer'))
        registry.register('PEM_Transformer', MockComponent('PEM_Transformer'))
        registry.register('BOP_Transformer', MockComponent('BOP_Transformer'))
        registry.register('cooling_manager', MockComponent('cooling_manager'))
        
        # 2. Setup Strategy
        strategy = HybridArbitrageEngineStrategy()
        
        # Mock Context
        context = MagicMock()

        context.simulation.timestep_hours = 1.0/60.0
        context.physics.soec_cluster.kwh_per_kg = 50.0 
        context.physics.pem_system.kwh_per_kg = 50.0
        context.economics.h2_price_eur_kg = 1.0
        context.economics.guaranteed_power_mw = 0.0
        context.economics.ppa_contract_price_eur_mwh = 50.0
        context.economics.ppa_variable_price_eur_mwh = 20.0
        context.economics.h2_non_rfnbo_price_eur_kg = 2.0
        context.economics.p_grid_max_mw = 100.0
        
        # Initialize Strategy
        total_steps = 10
        strategy.initialize(registry, context, total_steps=total_steps, use_chunked_history=False)
        
        # 3. Run a few steps
        # We need to manually trigger record_post_step
        # Strategy expects _state.step_idx to be updated via decide_and_apply
        
        prices = np.zeros(total_steps)
        wind = np.zeros(total_steps)
        
        for i in range(5):
            station.step(i/60.0) # Update station state (so demand signal is calculated)
            
            # Run decide (updates step_idx)
            strategy.decide_and_apply(i/60.0, prices, wind)
            
            # Run record
            strategy.record_post_step()
            
        # 4. Check History
        history = strategy.get_history()
        
        # Verify keys
        print("History Keys:", list(history.keys()))
        
        # We expect a key related to demand
        # Based on my plan, it should be 'Truck_Station_1_truck_demand_kg_h'
        # But for now, just checking if ANY discharge specific key exists
        
        discharge_keys = [k for k in history.keys() if 'Truck_Station_1' in k]
        print("Discharge Keys:", discharge_keys)
        
        # Assertion: Check for demand signal
        # Use regex or substring matching
        found = any('truck_demand_kg_h' in k for k in discharge_keys)
        
        if not found:
            print("FAIL: 'truck_demand_kg_h' not found in history.")
        else:
            print("SUCCESS: 'truck_demand_kg_h' found in history.")
            
        self.assertTrue(found, "DischargeStation demand signal should be recorded in history")

if __name__ == '__main__':
    unittest.main()
