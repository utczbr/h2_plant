
import sys
import os
import logging

# Ensure project root is in path
sys.path.append(os.getcwd())

from h2_plant.config.plant_config import (
    PlantConfig, ProductionConfig, StorageConfig, 
    CompressionConfig, DemandConfig, EnergyPriceConfig,
    SimulationConfig, PathwayConfig,
    SOECConfig, ElectrolyzerConfig, ExternalInputsConfig,
    OxygenManagementConfig, WaterTreatmentConfig, BatteryConfig
)
from h2_plant.core.enums import AllocationStrategy
# from h2_plant.config.models import EconomicsConfig
# Pydantic missing, mocking simple class
class EconomicsConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
from h2_plant.config.plant_builder import PlantBuilder
from h2_plant.core.component_ids import ComponentID

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def run_test():
    print("--- Starting Reproduction Test: SOEC + Arbitrage ---")

    # 1. Mock the Config produced by GraphToConfigAdapter
    # "Wind -> Arbitrage -> SOEC"
    # Wind: handled via energy_price (defaults used if node not mapped, which is fine for now)
    # Arbitrage: Sets pathway params
    # SOEC: Sets production.soec params

    config = PlantConfig(
        name="Reproduction Plant",
        production=ProductionConfig(
            soec=SOECConfig(
                enabled=True,
                max_power_nominal_mw=2.4, # Was max_power_mw
                num_modules=6
            )
        ),
        storage=StorageConfig(),
        pathway=PathwayConfig(
            allocation_strategy=AllocationStrategy.COST_OPTIMAL,
            h2_price_eur_kg=5.0,     # Profitable H2 price
            ppa_price_eur_mwh=50.0   # Standard PPA
        ),
        energy_price=EnergyPriceConfig(
            source='file',
            price_file='h2_plant/data/NL_Prices_2024_15min.csv',
            wind_data_file='h2_plant/data/producao_horaria_2_turbinas.csv'
        ),
        simulation=SimulationConfig(
            duration_hours=24, # Short run
            timestep_hours=0.25
        ),
        external_inputs=ExternalInputsConfig(),
        oxygen_management=OxygenManagementConfig(),
        water_treatment=WaterTreatmentConfig(),
        battery=BatteryConfig(),
        thermal_components=None,
        atr_system=None,
        pem_system=None,
        soec_cluster=None,
        logistics=None
    )
    
    # 2. Build the Plant
    print("Building Plant...")
    builder = PlantBuilder.from_config(config)
    registry = builder.registry
    
    # 3. Check Components
    print(f"Registry keys: {list(registry._components.keys())}")
    
    soec = registry.get(ComponentID.SOEC_CLUSTER)
    coordinator = registry.get(ComponentID.DUAL_PATH_COORDINATOR)
    
    if not soec:
        print("FAIL: SOEC Cluster not found!")
        return
    if not coordinator:
        print("FAIL: DualPathCoordinator not found!")
        return
        
    print(f"Coordinator: {coordinator}")
    print(f"Coordinator H2 Price: {getattr(coordinator, 'H2_PRICE_KG', 'N/A')}")
    print(f"Coordinator PPA Price: {getattr(coordinator, 'PPA_PRICE', 'N/A')}")

    # 4. Run Simulation Step
    print("Initializing...")
    registry.initialize_all(dt=0.25)
    
    print("Stepping simulation...")
    # Step 10 times to get past initialization ramps
    for i in range(10):
        t = i * 0.25
        registry.step_all(t)
        
        # Check Coordinator State
        state = coordinator.get_state()
        print(f"Step {i}: Offer={state.get('P_offer_mw'):.2f}, SOEC_Set={state.get('soec_setpoint_mw'):.2f}, Sold={state.get('sold_power_mw'):.2f}")
    
    print("--- Test Complete ---")

if __name__ == "__main__":
    run_test()
