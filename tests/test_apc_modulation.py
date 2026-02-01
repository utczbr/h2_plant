
import unittest
import numpy as np
from unittest.mock import MagicMock
from h2_plant.control.engine_dispatch import HybridArbitrageEngineStrategy
from h2_plant.control.dispatch import DispatchResult

class TestAPCModulation(unittest.TestCase):
    def test_apc_power_reduction(self):
        # 1. Setup Strategy
        strategy = HybridArbitrageEngineStrategy()
        
        # Mock Context & Components
        strategy._context = MagicMock()
        strategy._context.simulation.timestep_hours = 1.0 
        strategy._context.economics.guaranteed_power_mw = 0.0
        # Initialize physics constants to avoid MagicMock comparison errors
        strategy._context.physics.soec_cluster.kwh_per_kg = 37.5
        strategy._context.physics.pem_system.kwh_per_kg = 50.0
        
        # Mock Inner Strategy to always return 10 MW
        strategy._inner_strategy = MagicMock()
        strategy._inner_strategy.decide.return_value = DispatchResult(
            P_soec=10.0, P_pem=0.0, P_sold=0.0, state_update={}
        )
        
        # Mock SOEC & PEM Access based on how 'decide_and_apply' uses them
        # It checks if self._soec is truthy
        strategy._soec = MagicMock()
        strategy._pem = None
        
        # Mock Storage Info
        # Capacity = 1000 kg. 10 MW for 1 hour approx 250 kg H2 production (at 40 kWh/kg)
        strategy._storage_total_capacity_kg = 10000.0 
        
        # We need to hook into _get_aggregate_soc to return our simulated mass
        current_mass = 5000.0 # Start at 50% SOC
        
        # Arrays for plotting check
        soc_history = []
        power_history = []
        factor_history = []
        
        # We also need to init the strategy arrays usually done in initialize()
        strategy._total_steps = 100
        strategy._history = {
            'minute': np.zeros(100), 'P_offer': np.zeros(100),
            'storage_soc': np.zeros(100), 'storage_dsoc_per_h': np.zeros(100),
            'storage_zone': np.zeros(100), 'storage_action_factor': np.zeros(100),
            'storage_time_to_full_h': np.zeros(100), 'spot_price': np.zeros(100),
            'ppa_price_effective_eur_mwh': np.zeros(100),
            'h2_rfnbo_kg': np.zeros(100), 'h2_non_rfnbo_kg': np.zeros(100),
            'spot_purchased_mw': np.zeros(100), 'spot_threshold_eur_mwh': np.zeros(100),
            'cumulative_h2_rfnbo_kg': np.zeros(100), 'cumulative_h2_non_rfnbo_kg': np.zeros(100)
        }
        
        # Dummy prices/wind
        prices = np.zeros(100)
        wind = np.full(100, 20.0) # 20 MW wind available
        
        # Initialize Controller State (usually done in initialize)
        strategy._ctrl_params = {
            'SOC_LOW': 0.60, 'SOC_HIGH': 0.80, 'SOC_CRITICAL': 0.95,
            'HYSTERESIS': 0.02, 'MAX_RATE_H': 0.20, 'MIN_ACTION_FACTOR': 0.1
        }
        strategy._ctrl_state = {
            'prev_soc': 0.5, 'current_zone': 0, 'time_to_full_h': 99.0
        }
        
        print("\n--- Starting APC Modulation Test ---")
        print(f"{'Step':<5} | {'SOC (%)':<10} | {'Zone':<5} | {'Factor':<10} | {'Power (MW)':<10}")
        print("-" * 55)

        for i in range(50): # Simulate 50 steps
            strategy._state.step_idx = i
            
            # Monkey-patch _get_aggregate_soc to return our simulation state
            # (Since decide_and_apply calls it internally)
            strategy._get_aggregate_soc = MagicMock(return_value=(current_mass / 10000.0, current_mass))
            
            # Execute logic
            strategy.decide_and_apply(t=float(i), prices=prices, wind=wind)
            
            # Retrieve factor and power from history/mock
            factor = strategy._history['storage_action_factor'][i]
            # Since P_soec_final is not stored directly in history (only P_offer which is wind/grid),
            # we check the call to _soec.receive_input
            
            # Get the argument passed to receive_input('power_in', VALUE, ...)
            # Mock records all calls. We want the last one.
            call_args = strategy._soec.receive_input.call_args
            # args[0] is port, args[1] is value
            actual_power = call_args[0][1] 
            
            soc = current_mass / 10000.0
            zone = strategy._history['storage_zone'][i]
            
            soc_history.append(soc)
            power_history.append(actual_power)
            factor_history.append(factor)
            
            print(f"{i:<5} | {soc*100:<10.1f} | {int(zone):<5} | {factor:<10.4f} | {actual_power:<10.2f}")
            
            # Assertions based on SOC
            if soc < 0.60:
                self.assertAlmostEqual(actual_power, 10.0, delta=0.01) # Zone 0: Full Power
            elif 0.60 <= soc < 0.80:
                if soc == 0.60:
                     self.assertAlmostEqual(actual_power, 10.0, delta=0.01)
                else:
                     self.assertLess(actual_power, 10.0) # Zone 1: Reduced
                self.assertGreater(actual_power, 6.9) 
            elif 0.80 <= soc < 0.95:
                 self.assertLess(actual_power, 7.0) # Zone 2: Aggressive reduced
            elif soc >= 0.95:
                 self.assertAlmostEqual(actual_power, 0.0, delta=0.01) # Zone 3: Stop
                 
            # Simulate Fill: 10 MW -> 250 kg H2
            h2_produced = (actual_power * 1000.0) / 40.0 # approx 40 kWh/kg
            current_mass += h2_produced
            
            if current_mass > 10000.0:
                current_mass = 10000.0 # Clamp physics

        print("Test Completed Successfully")

if __name__ == "__main__":
    unittest.main()
