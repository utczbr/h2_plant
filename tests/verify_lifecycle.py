
import logging
import numpy as np
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.components.electrolysis.soec_operator import SOECOperator
from h2_plant.components.electrolysis.pem_electrolyzer import DetailedPEMElectrolyzer
from h2_plant.config.plant_config import SimulationConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_soec_lifecycle():
    logger.info("=== Verifying SOEC Lifecycle & Reset ===")
    
    # 1. Setup: Short lifecycle for testing (e.g., 5 years = 43800h, but we can jump ahead)
    lifecycle_h = 43800.0 # 5 years
    config = {
        'num_modules': 6,
        'max_power_nominal_mw': 2.4,
        'lifecycle': lifecycle_h,
        'degradation_year': 0.0 # Start at Year 0
    }
    
    registry = ComponentRegistry()
    soec = SOECOperator(config)
    soec.initialize(dt=1.0, registry=registry) # 1 hour steps
    
    # 2. Check BOL (Beginning of Life)
    soec.step(t=0.0)
    eff_bol = soec.current_efficiency_kwh_kg
    cap_bol = soec.current_capacity_factor
    logger.info(f"[BOL t=0h] Efficiency: {eff_bol:.2f} kWh/kg (Lower is better), Capacity: {cap_bol:.2f}")
    
    # 3. Fast Forward to End of Life (Year 5 - 1 hour)
    # We can manipulate accumulated_hours directly to save simulation time
    logger.info(">> Fast forwarding to End of Life...")
    soec.accumulated_hours = lifecycle_h - 10.0 # Leave some buffer
    soec.step(t=0.0) # Step to trigger degradation update
    
    eff_eol = soec.current_efficiency_kwh_kg
    cap_eol = soec.current_capacity_factor
    logger.info(f"[EOL t={soec.accumulated_hours}h] Efficiency: {eff_eol:.2f} kWh/kg, Capacity: {cap_eol:.2f}")
    
    # Assert Degradation occurred
    if eff_eol <= eff_bol:
        logger.error("FAILURE: Efficiency did not degrade (increase) over time.")
    if cap_eol >= cap_bol:
        logger.error("FAILURE: Capacity did not degrade (decrease) over time.")
        
    # 4. Trigger Reset (Cross lifecycle boundary)
    logger.info(">> Stepping across lifecycle boundary (Reset)...")
    soec.accumulated_hours = lifecycle_h + 10.0 # Push past limit clearly
    soec.step(t=0.0) # Trigger update
    
    eff_reset = soec.current_efficiency_kwh_kg
    cap_reset = soec.current_capacity_factor
    logger.info(f"[RESET t={soec.accumulated_hours}h] Efficiency: {eff_reset:.2f} kWh/kg, Capacity: {cap_reset:.2f}")
    
    # Assert Reset
    # Should be close to BOL values
    if abs(eff_reset - eff_bol) > 0.1:
         logger.error(f"FAILURE: Efficiency did not reset properly. Got {eff_reset}, Expected ~{eff_bol}")
    else:
         logger.info("SUCCESS: Efficiency reset confirmed.")

    if abs(cap_reset - cap_bol) > 0.01:
         logger.error(f"FAILURE: Capacity did not reset properly. Got {cap_reset}, Expected ~{cap_bol}")
    else:
         logger.info("SUCCESS: Capacity reset confirmed.")
         

def verify_pem_lifecycle():
    logger.info("\n=== Verifying PEM Lifecycle & Reset ===")
    
    # 1. Setup
    lifecycle_h = 43800.0 # 5 years
    config = {
        'max_power_mw': 5.0,
        'base_efficiency': 0.65,
        'lifecycle': lifecycle_h
    }
    
    registry = ComponentRegistry()
    pem = DetailedPEMElectrolyzer(config)
    pem.initialize(dt=1.0, registry=registry)
    
    # Helper to check voltage at fixed power
    def check_voltage(pem_component):
        pem_component.set_power_input_mw(2.5) # 50% load
        # Ensure water buffer
        pem_component.water_buffer_kg = 1000.0 
        pem_component.available_water_kg_h = 1000.0
        pem_component.step(t=0.0)
        return pem_component.V_cell
        
    # 2. Check BOL
    v_bol = check_voltage(pem)
    logger.info(f"[BOL t=0h] Voltage: {v_bol:.4f} V (Lower is better)")
    
    # 3. Fast Forward
    logger.info(">> Fast forwarding to End of Life...")
    pem.t_op_h = lifecycle_h - 10.0
    v_eol = check_voltage(pem)
    logger.info(f"[EOL t={pem.t_op_h}h] Voltage: {v_eol:.4f} V")
    
    # Assert Degradation
    if v_eol <= v_bol:
        logger.error("FAILURE: PEM Voltage did not increase (degrade).")
    else:
        logger.info(f"Degradation confirmed: +{(v_eol - v_bol)*1000:.1f} mV")

    # 4. Trigger Reset
    logger.info(">> Stepping across lifecycle boundary (Reset)...")
    pem.t_op_h = lifecycle_h + 10.0
    v_reset = check_voltage(pem)
    logger.info(f"[RESET t={pem.t_op_h}h] Voltage: {v_reset:.4f} V")
    
    # Assert Reset
    if abs(v_reset - v_bol) > 0.01: # allow small tolerance
         logger.error(f"FAILURE: PEM Voltage did not reset. Got {v_reset}, Expected ~{v_bol}")
    else:
         logger.info("SUCCESS: PEM Voltage reset confirmed.")

if __name__ == "__main__":
    verify_soec_lifecycle()
    verify_pem_lifecycle()
