"""
Simplified test for FlowNetwork functionality.
Tests the flow network directly with manually created components.
"""

import logging
import sys
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.simulation.flow_network import FlowNetwork
from h2_plant.config.plant_config import ConnectionConfig
from h2_plant.components.storage.tank_array import TankArray
from h2_plant.components.compression.filling_compressor import FillingCompressor
from h2_plant.core.stream import Stream

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_simple_flow():
    logger.info("Starting simplified FlowNetwork test...")
    
    # Create registry
    registry = ComponentRegistry()
    
    # Create components
    lp_tank = TankArray(n_tanks=2, capacity_kg=100.0, pressure_bar=30.0)
    hp_tank = TankArray(n_tanks=2, capacity_kg=100.0, pressure_bar=350.0)
    compressor = FillingCompressor(max_flow_kg_h=50.0, inlet_pressure_bar=30.0, outlet_pressure_bar=350.0)
    
    # Register components
    registry.register('lp_storage', lp_tank, component_type='storage')
    registry.register('hp_storage', hp_tank, component_type='storage')
    registry.register('compressor', compressor, component_type='compression')
    
    # Initialize components
    registry.initialize_all(dt=1.0)
    
    # Fill LP tank with some H2
    logger.info("Filling LP tank with 50 kg H2...")
    lp_tank.fill(50.0)
    logger.info(f"LP Tank mass: {lp_tank.get_total_mass()} kg")
    
    # Define topology: LP -> Compressor -> HP
    topology = [
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
    
    # Create and initialize FlowNetwork
    flow_network = FlowNetwork(registry, topology)
    flow_network.initialize()
    
    logger.info("\\nExecuting flow for hour 0...")
    logger.info(f"Before flow - Compressor transfer_mass_kg: {compressor.transfer_mass_kg}")
    flow_network.execute_flows(0.0)
    logger.info(f"After flow - Compressor transfer_mass_kg: {compressor.transfer_mass_kg}")
    
    # Call step to process the input (this is normally done by Engine)
    logger.info("\\nCalling compressor.step() to process input...")
    compressor.step(0.0)
    logger.info(f"After step - Compressor actual_mass_transferred_kg: {compressor.actual_mass_transferred_kg}")
    
    # Now execute flows again to move mass from compressor -> HP
    logger.info("\\nExecuting second flow round (compressor -> HP)...")
    flow_network.execute_flows(0.0)
    
    # Check results
    logger.info(f"\\nAfter flow execution:")
    logger.info(f"  LP Tank mass: {lp_tank.get_total_mass():.2f} kg")
    logger.info(f"  HP Tank mass: {hp_tank.get_total_mass():.2f} kg")
    logger.info(f"  Compressor energy: {compressor.energy_consumed_kwh:.2f} kWh")
    
    if hp_tank.get_total_mass() > 0:
        logger.info("\\n✓ SUCCESS:  Mass flowed through the system!")
        logger.info(f"  {hp_tank.get_total_mass():.2f} kg transferred from LP to HP storage")
        return True
    else:
        logger.error("\\n✗ FAILURE: No mass in HP tank")
        return False

if __name__ == "__main__":
    success = test_simple_flow()
    sys.exit(0 if success else 1)
