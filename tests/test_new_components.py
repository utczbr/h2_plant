"""
Verification test for newly implemented component flow interfaces:
- ATRReactor
- SOECStack  
- BatteryStorage
"""

import logging
import sys
from pathlib import Path

project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.simulation.flow_network import FlowNetwork
from h2_plant.config.plant_config import ConnectionConfig
from h2_plant.components.reforming.atr_reactor import ATRReactor
from h2_plant.components.electrolysis.soec_stack import SOECStack
from h2_plant.components.storage.battery_storage import BatteryStorage
from h2_plant.components.storage.tank_array import TankArray

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_atr_reactor():
    """Test ATRReactor flow interface."""
    logger.info("\n" + "="*60)
    logger.info("TEST: ATRReactor")
    logger.info("="*60)
    
    registry = ComponentRegistry()
    
    # Create ATR reactor (note: will fail without model file, but interface should work)
    atr = ATRReactor(component_id='atr1', max_flow_kg_h=100.0, model_path='nonexistent.pkl')
    h2_tank = TankArray(n_tanks=2, capacity_kg=100.0, pressure_bar=30.0)
    
    registry.register('atr', atr, 'reforming')
    registry.register('h2_storage', h2_tank, 'storage')
    
    try:
        registry.initialize_all(dt=1.0)
    except:
        logger.warning("ATR model file not found - expected for this test")
    
    # Check ports
    ports = atr.get_ports()
    logger.info(f"ATR Ports: {list(ports.keys())}")
    assert 'o2_in' in ports
    assert 'h2_out' in ports
    assert ports['h2_out']['type'] == 'output'
    
    logger.info("✓ ATR ports defined correctly")
    return True

def test_soec_stack():
    """Test SOECStack flow interface."""
    logger.info("\n" + "="*60)
    logger.info("TEST: SOECStack")
    logger.info("="*60)
    
    registry = ComponentRegistry()
    
    soec = SOECStack(max_power_kw=1000.0)
    h2_tank = TankArray(n_tanks=2, capacity_kg=50.0, pressure_bar=30.0)
    
    registry.register('soec', soec, 'electrolysis')
    registry.register('h2_storage', h2_tank, 'storage')
    
    topology = [
        ConnectionConfig('soec', 'h2_out', 'h2_storage', 'h2_in', 'hydrogen')
    ]
    
    registry.initialize_all(dt=1.0)
    flow_network = FlowNetwork(registry, topology)
    flow_network.initialize()
    
    # Simulate power input
    soec.receive_input('electricity_in', 500.0, 'electricity')
    
    # Run one step
    registry.step_all(0)
    flow_network.execute_flows(0)
    
    # Check results
    h2_produced = h2_tank.get_total_mass()
    logger.info(f"H2 produced and stored: {h2_produced:.4f} kg")
    
    if h2_produced > 0:
        logger.info("✓ SOEC producing H2 correctly")
        return True
    else:
        logger.error("✗ No H2 production")
        return False

def test_battery_storage():
    """Test BatteryStorage flow interface."""
    logger.info("\n" + "="*60)
    logger.info("TEST: BatteryStorage")
    logger.info("="*60)
    
    registry = ComponentRegistry()
    
    battery = BatteryStorage(
        capacity_kwh=1000.0,
        max_charge_power_kw=500.0,
        max_discharge_power_kw=500.0,
        initial_soc=0.5
    )
    
    registry.register('battery', battery, 'storage')
    registry.initialize_all(dt=1.0)
    
    # Check ports
    ports = battery.get_ports()
    logger.info(f"Battery Ports: {list(ports.keys())}")
    assert 'electricity_in' in ports
    assert 'electricity_out' in ports
    
    # Test charge
    initial_energy = battery.energy_kwh
    battery.receive_input('electricity_in', 200.0, 'electricity')
    registry.step_all(0)
    
    logger.info(f"Initial energy: {initial_energy:.2f} kWh")
    logger.info(f"After charge: {battery.energy_kwh:.2f} kWh")
    
    # Test discharge capability
    available_power = battery.get_output('electricity_out')
    logger.info(f"Available discharge power: {available_power:.2f} kW")
    
    if available_power > 0:
        logger.info("✓ Battery flow interface working")
        return True
    else:
        logger.error("✗ Battery not providing power")
        return False

def main():
    logger.info("\n" + "="*60)
    logger.info("VERIFYING NEW COMPONENT FLOW INTERFACES")
    logger.info("="*60)
    
    results = []
    
    # Test each component
    try:
        results.append(("ATR", test_atr_reactor()))
    except Exception as e:
        logger.error(f"ATR test failed: {e}")
        results.append(("ATR", False))
    
    try:
        results.append(("SOEC", test_soec_stack()))
    except Exception as e:
        logger.error(f"SOEC test failed: {e}")
        results.append(("SOEC", False))
    
    try:
        results.append(("Battery", test_battery_storage()))
    except Exception as e:
        logger.error(f"Battery test failed: {e}")
        results.append(("Battery", False))
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("VERIFICATION SUMMARY")
    logger.info("="*60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{name}: {status}")
    
    all_passed = all(passed for _, passed in results)
    if all_passed:
        logger.info("\n✓ ALL TESTS PASSED")
        return 0
    else:
        logger.info("\n✗ SOME TESTS FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())
