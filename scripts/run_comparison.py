import logging
import numpy as np
from typing import Dict, Any, List
from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
print(f"DEBUG: sys.path: {sys.path}")
import h2_plant.pathways.dual_path_coordinator
print(f"DEBUG: DualPathCoordinator file: {h2_plant.pathways.dual_path_coordinator.__file__}")

from h2_plant.config.plant_builder import PlantBuilder
from h2_plant.simulation.engine import SimulationEngine
from h2_plant.config.plant_config import SimulationConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- DATA FROM MANAGER.PY ---
SPOT_PRICE_HOUR_BY_HOUR = [
    [400.0, 420.0, 450.0, 40.0],
    [35.0, 35.0, 65.0, 55.0],
    [35.0, 38.0, 40.0, 42.0],
    [50.0, 52.0, 55.0, 58.0],
    [45.0, 48.0, 50.0, 52.0],
    [55.0, 58.0, 60.0, 62.0],
    [40.0, 455.0, 425.0, 48.0], # Modified by user
    [30.0, 32.0, 35.0, 38.0]
]

HOUR_OFFER = [3.0, 5.0, 13.0, 18.0, 15.0, 9.0, 18.0, 0.0]

# Expand to minute-by-minute arrays
SPOT_PRICE_PROFILE = []
for hour in SPOT_PRICE_HOUR_BY_HOUR:
    for price_15min in hour:
        SPOT_PRICE_PROFILE.extend([price_15min] * 15)

OFFERED_POWER_PROFILE = []
for pot in HOUR_OFFER:
    OFFERED_POWER_PROFILE.extend([pot] * 60)

class MockEnvironmentManager(Component):
    """Mock environment manager that serves fixed profiles."""
    def __init__(self):
        super().__init__()
        self.current_wind_power_mw = 0.0
        self.current_energy_price_eur_mwh = 0.0
        self.component_id = "environment_manager"
        self.current_minute = 0

    def step(self, t: float) -> None:
        minute = int(t * 60)
        self.current_minute = minute
        if minute < len(OFFERED_POWER_PROFILE):
            self.current_wind_power_mw = OFFERED_POWER_PROFILE[minute]
            self.current_energy_price_eur_mwh = SPOT_PRICE_PROFILE[minute]
        else:
            self.current_wind_power_mw = 0.0
            self.current_energy_price_eur_mwh = 0.0

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        super().initialize(dt, registry)

    def get_state(self) -> Dict[str, Any]:
        return {
            "current_wind_power_mw": self.current_wind_power_mw,
            "current_energy_price_eur_mwh": self.current_energy_price_eur_mwh
        }

    def get_future_power(self, minutes_ahead: int) -> float:
        """Get forecast for future power."""
        target_minute = self.current_minute + minutes_ahead
        if target_minute < len(OFFERED_POWER_PROFILE):
            return OFFERED_POWER_PROFILE[target_minute]
        return 0.0

def run_comparison():
    print("--- Starting H2 Plant Comparison Simulation (8 Hours) ---")
    
    # 1. Build Plant
    builder = PlantBuilder.from_file("configs/h2plant_detailed.yaml")
    # builder.load_config("h2plant_detailed.yaml") # Removed incorrect call
    
    registry = builder.registry
    
    # 2. Replace Environment Manager with Mock
    mock_env = MockEnvironmentManager()
    # We need to manually register it, overwriting the existing one
    # ComponentRegistry doesn't have a direct 'replace' but 'register' overwrites if ID exists?
    # Let's check registry implementation or just set it.
    # Registry uses a dict, so we can overwrite.
    # Force replace environment manager
    if "environment_manager" in registry._components:
        del registry._components["environment_manager"]
    registry.register(mock_env.component_id, mock_env, component_type='environment')
    
    # 3. Configure Simulation
    # Use config from builder
    plant_config = builder.config
    plant_config.simulation.start_hour = 0
    plant_config.simulation.duration_hours = 8
    plant_config.simulation.timestep_hours = 1.0 / 60.0 # 1 minute
    
    engine = SimulationEngine(registry, plant_config)
    engine.initialize()
    
    # 4. Run Step-by-Step and Print Table
    print(" | Min | H | P. Offer | P. Set SOEC | P. SOEC Act | P. PEM | P. Sold | P. Spot | Decision | H2 SOEC (kg/min) | Steam SOEC (kg/min) |")
    print("----------------------------------------------------------------------------------------------------------------------")
    
    # We need to access components to get their state
    coordinator = registry.get("dual_path_coordinator")
    soec = registry.get("soec_cluster")

    history = {
        'P_offer': [],
        'P_soec_set': [],
        'P_soec_actual': [],
        'P_pem': [],
        'P_sold': [],
        'spot_price': [],
        'sell_decision': [],
        'H2_soec_kg': [],
        'steam_soec_kg': []
    }
    
    for minute in range(480):
        hour_fraction = minute / 60.0
        engine._execute_timestep(hour_fraction)
        
        # Get values for table
        # Coordinator state
        coord_state = coordinator.get_state()
        P_offer = mock_env.current_wind_power_mw
        P_soec_set = coord_state.get('soec_setpoint_mw', 0.0)
        P_pem = coord_state.get('pem_setpoint_mw', 0.0)
        P_sold = coord_state.get('sold_power_mw', 0.0)
        spot_price = mock_env.current_energy_price_eur_mwh
        sell_decision = coord_state.get('sell_decision', 0)
        
        # SOEC state
        soec_state = soec.get_state()
        P_soec_actual = soec_state.get('power_consumption_mw', 0.0)
        h2_soec_kg = soec_state.get('h2_production_kg_h', 0.0) / 60.0 # kg/min
        steam_soec_kg = soec_state.get('steam_consumption_kg_h', 0.0) / 60.0 # kg/min
        
        if minute % 15 == 0:
             print(
                f" | {minute:03d} | {minute//60 + 1} | {P_offer:9.2f} | {P_soec_set:11.2f} | {P_soec_actual:11.2f} | {P_pem:6.2f} | {P_sold:7.2f} | {spot_price:7.2f} | {('SELL' if sell_decision == 1 else 'H2'):8s} | {h2_soec_kg:16.4f} | {steam_soec_kg:19.4f} |"
            )
            
        # Append to history
        history['P_offer'].append(P_offer)
        history['P_soec_set'].append(P_soec_set)
        history['P_soec_actual'].append(P_soec_actual)
        history['P_pem'].append(P_pem)
        history['P_sold'].append(P_sold)
        history['spot_price'].append(spot_price)
        history['sell_decision'].append(sell_decision)
        history['H2_soec_kg'].append(h2_soec_kg)
        history['steam_soec_kg'].append(steam_soec_kg)

    print("----------------------------------------------------------------------------------------------------------------------")
    print("\n--- End of Comparison Simulation ---")

    # Calculate and print summary
    print("\n## Simulation Summary (Total/Average Values)")
    
    # Constants for calculation
    PEM_KWH_KG = 33.3 # From manager.py MW_TON_H2
    
    E_total_offer = sum(history['P_offer']) / 60.0
    E_soec = sum(history['P_soec_actual']) / 60.0
    E_pem = sum(history['P_pem']) / 60.0
    E_sold = sum(history['P_sold']) / 60.0
    
    H2_soec_total = sum(history['H2_soec_kg'])
    Steam_soec_total = sum(history['steam_soec_kg'])
    
    # Calculate PEM H2 (Estimate based on energy)
    # H2 (kg) = Energy (MWh) * 1000 (kWh/MWh) / Efficiency (kWh/kg)
    H2_pem_total = (E_pem * 1000.0) / PEM_KWH_KG
    
    H2_total = H2_soec_total + H2_pem_total
    
    print(f"* Total Offered Energy: {E_total_offer:.2f} MWh")
    print(f"* Energy Supplied to SOEC (H2/Steam): {E_soec:.2f} MWh")
    print(f"* **Total SOEC Hydrogen Production**: {H2_soec_total:.2f} kg")
    print(f"* **Total SOEC Steam Consumption**: {Steam_soec_total:.2f} kg")
    print(f"* Energy Supplied to PEM (H2): {E_pem:.2f} MWh")
    print(f"* **Total PEM Hydrogen Production (Est. @ 33.3 kWh/kg)**: {H2_pem_total:.2f} kg")
    print(f"* **TOTAL HYDROGEN PRODUCTION**: {H2_total:.2f} kg")
    print(f"* Energy Sold to the Market: {E_sold:.2f} MWh")
    print(f"* Offer Deviation (Error Margin): {E_total_offer - (E_soec + E_pem + E_sold):.4f} MWh")
    print("-------------------------------------------------------------------")

if __name__ == "__main__":
    run_comparison()
