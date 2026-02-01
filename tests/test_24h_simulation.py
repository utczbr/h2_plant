"""
24-Hour Simulation Test
Tests the complete H2 plant system with configurable connections.
"""

import logging
import sys
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.simulation.flow_network import FlowNetwork
from h2_plant.simulation.engine import SimulationEngine
from h2_plant.config.plant_config import ConnectionConfig, SimulationConfig
from h2_plant.components.storage.tank_array import TankArray
from h2_plant.components.compression.filling_compressor import FillingCompressor
from h2_plant.components.production.electrolyzer_source import ElectrolyzerProductionSource

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def build_simple_plant(registry: ComponentRegistry):
    """Build a simple plant for testing."""
    logger.info("Building simple test plant...")
    
    # Production
    electrolyzer = ElectrolyzerProductionSource(
        max_power_mw=2.5,
        base_efficiency=0.65,
        min_load_factor=0.20,
        startup_time_hours=0.1
    )
    registry.register('electrolyzer', electrolyzer, component_type='production')
    
    # Storage
    lp_tanks = TankArray(n_tanks=4, capacity_kg=50.0, pressure_bar=30.0)
    hp_tanks = TankArray(n_tanks=8, capacity_kg=200.0, pressure_bar=350.0)
    registry.register('lp_storage', lp_tanks, component_type='storage')
    registry.register('hp_storage', hp_tanks, component_type='storage')
    
    # Compression
    compressor = FillingCompressor(
        max_flow_kg_h=100.0,
        inlet_pressure_bar=30.0,
        outlet_pressure_bar=350.0,
        num_stages=3
    )
    registry.register('compressor', compressor, component_type='compression')
    
    logger.info(f"Built plant with {registry.get_component_count()} components")
    return registry

def create_topology():
    """Create simple topology for testing."""
    return [
        ConnectionConfig(
            source_id='electrolyzer',
            source_port='h2_out',
            target_id='lp_storage',
            target_port='h2_in',
            resource_type='hydrogen'
        ),
        ConnectionConfig(
            source_id='lp_storage',
            source_port='h2_out',
            target_id='compressor',
            target_port='h2_in',
            resource_type='hydrogen'
        ),
        ConnectionConfig(
            source_id='compressor',
            source_port='h2_out',
            target_id='hp_storage',
            target_port='h2_in',
            resource_type='hydrogen'
        )
    ]

def run_24h_simulation():
    """Run a 24-hour simulation test."""
    logger.info("=" * 60)
    logger.info("24-HOUR SIMULATION TEST")
    logger.info("=" * 60)
    
    # Create registry and build plant
    registry = ComponentRegistry()
    build_simple_plant(registry)
    
    # Create simulation config
    from h2_plant.config.plant_config import SimulationConfig
    sim_config = SimulationConfig(
        timestep_hours=1.0,
        duration_hours=24,
        start_hour=0,
        checkpoint_interval_hours=0
    )
    
    # Add topology to registry's config manually
    sim_config.topology = create_topology()
    
    # Create engine
    engine = SimulationEngine(registry, sim_config)
    
    # Set electrolyzer to run at 50% capacity
    electrolyzer = registry.get('electrolyzer')
    electrolyzer.power_input_mw = 1.25  # 50% of 2.5 MW
    
    logger.info("\nStarting simulation...")
    logger.info(f"Duration: {sim_config.duration_hours} hours")
    logger.info(f"Timestep: {sim_config.timestep_hours} hour(s)")
    logger.info(f"Electrolyzer power: {electrolyzer.power_input_mw} MW")
    
    # Run simulation
    results = engine.run()
    
    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("SIMULATION RESULTS")
    logger.info("=" * 60)
    
    lp_storage = registry.get('lp_storage')
    hp_storage = registry.get('hp_storage')
    compressor = registry.get('compressor')
    
    logger.info(f"\nProduction:")
    logger.info(f"  Electrolyzer H2 produced: {electrolyzer.cumulative_h2_kg:.2f} kg")
    logger.info(f"  Electrolyzer energy used: {electrolyzer.cumulative_energy_kwh:.2f} kWh")
    logger.info(f"  H2 production rate: {electrolyzer.cumulative_h2_kg / 24:.2f} kg/h")
    
    logger.info(f"\nStorage:")
    logger.info(f"  LP storage: {lp_storage.get_total_mass():.2f} kg")
    logger.info(f"  HP storage: {hp_storage.get_total_mass():.2f} kg")
    logger.info(f"  Total stored: {lp_storage.get_total_mass() + hp_storage.get_total_mass():.2f} kg")
    
    logger.info(f"\nCompression:")
    logger.info(f"  Mass compressed: {compressor.cumulative_mass_kg:.2f} kg")
    logger.info(f"  Energy consumed: {compressor.cumulative_energy_kwh:.2f} kWh")
    if compressor.cumulative_mass_kg > 0:
        logger.info(f"  Specific energy: {compressor.cumulative_energy_kwh / compressor.cumulative_mass_kg:.2f} kWh/kg")
    
    logger.info(f"\nMass Balance:")
    total_h2 = (lp_storage.get_total_mass() + 
                hp_storage.get_total_mass())
    produced_h2 = electrolyzer.cumulative_h2_kg
    balance_error = abs(total_h2 - produced_h2)
    logger.info(f"  Produced: {produced_h2:.2f} kg")
    logger.info(f"  Stored: {total_h2:.2f} kg")
    logger.info(f"  Balance error: {balance_error:.4f} kg ({balance_error/produced_h2*100:.2f}% if produced_h2 > 0 else 'N/A')")
    
    logger.info(f"\nSimulation time: {results['simulation']['execution_time_seconds']:.2f} seconds")
    logger.info("=" * 60)
    
    # Return success if mass balance is good
    if produced_h2 > 0 and balance_error / produced_h2 < 0.01:  # < 1% error
        logger.info("\n✓ SUCCESS: Simulation completed with good mass balance!")
        return True
    elif produced_h2 == 0:
        logger.warning("\n⚠ WARNING: No hydrogen produced")
        return False
    else:
        logger.error(f"\n✗ FAILURE: Mass balance error too high ({balance_error/produced_h2*100:.2f}%)")
        return False

if __name__ == "__main__":
    try:
        success = run_24h_simulation()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.exception("Simulation failed with exception:")
        sys.exit(1)
