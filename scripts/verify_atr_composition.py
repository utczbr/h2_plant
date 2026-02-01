import sys
import os
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from h2_plant.components.reforming.atr_reactor import ATRReactor
from h2_plant.core.component_registry import ComponentRegistry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_atr_composition():
    logger.info("Initializing ATR Reactor for Composition Verification...")
    
    # create registry
    registry = ComponentRegistry()
    
    # Instantiate Reactor
    atr = ATRReactor(component_id="Test_ATR", max_flow_kg_h=5000.0)
    atr.initialize(dt=3600.0, registry=registry)
    
    # Provide inputs to trigger operation
    # Based on observation, Oxygen flow drives the lookup
    # Let's provide enough O2 to be in a valid range (e.g. 10 kmol/h)
    atr.buffer_oxygen_kmol = 10.0 * 3600.0 # 10 kmol/h * 1h
    atr.buffer_biogas_kmol = 100.0 * 3600.0 # Excess
    atr.buffer_steam_kmol = 100.0 * 3600.0 # Excess
    
    # Step
    logger.info("Stepping reactor...")
    atr.step(0.0)
    
    # Check output
    output_stream = atr.get_output('syngas_out')
    
    if output_stream:
        comp = output_stream.composition
        logger.info(f"Output Composition: {comp}")
        
        # Verify N2 is 0
        if comp.get('N2', 0.0) == 0.0:
            logger.info("PASS: N2 is 0.0")
        else:
            logger.error(f"FAIL: N2 is {comp.get('N2')}")
            
        # Verify CO is 0.01 (or close after normalization)
        # Verify H2, CO2, CH4 are present
        
        logger.info(f"H2: {comp.get('H2'):.4f}")
        logger.info(f"CO2: {comp.get('CO2'):.4f}")
        logger.info(f"CH4: {comp.get('CH4'):.4f}")
        logger.info(f"CO: {comp.get('CO'):.4f}")
        logger.info(f"H2O: {comp.get('H2O'):.4f}")
        
        total = sum(comp.values())
        logger.info(f"Total Sum: {total:.6f}")
        
    else:
        logger.error("FAIL: No output stream produced.")

if __name__ == "__main__":
    verify_atr_composition()
