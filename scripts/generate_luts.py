
import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.optimization.lut_manager import LUTManager, LUTConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("LUT_Generator")

def generate_high_res_luts():
    """
    Generates high-resolution Lookup Tables for the H2 Plant Simulation.
    Saves results to ~/.h2_plant/lut_cache
    """
    logger.info("Starting High-Resolution LUT Generation...")
    
    # 1. Define High-Resolution Config
    # User requested "maximum number of points".
    # Standard was 200/100. We go for 500/500 for high precision.
    config = LUTConfig(
        # Pressure: 1 bar to 900 bar (covers Storage @ 500 bar)
        pressure_min=1e5,
        pressure_max=900e5,
        pressure_points=500,
        
        # Temperature: 0°C to ~927°C (Covers SOEC ~850°C and Compression heat)
        temperature_min=273.15,
        temperature_max=1200.0,
        temperature_points=500,
        
        # Entropy: Covers liquid water to hot gas
        entropy_min=0.0,
        entropy_max=100000.0,
        entropy_points=500,
        
        # Fluids to generate
        fluids=['Hydrogen', 'Water', 'Oxygen']
    )
    
    logger.info(f"Configuration:")
    logger.info(f"  Pressure: {config.pressure_min/1e5:.1f}-{config.pressure_max/1e5:.1f} bar ({config.pressure_points} pts)")
    logger.info(f"  Temperature: {config.temperature_min:.1f}-{config.temperature_max:.1f} K ({config.temperature_points} pts)")
    
    # 2. Initialize Manager (Triggers Generation)
    registry = ComponentRegistry()
    manager = LUTManager(config)
    
    # Force regeneration by clearing cache if needed?
    # LUTManager checks if config matches cache. Since we changed resolution, 
    # it WILL mismatch and regenerate automatically.
    
    logger.info("Initializing Manager (Generates tables)...")
    manager.initialize(dt=1.0, registry=registry)
    
    # 3. Verify
    logger.info("Verifying Lookups...")
    
    # Check H2 at key points
    h_h2 = manager.lookup('Hydrogen', 'H', 30e5, 353.15) # PEM Outlet
    logger.info(f"  H2 @ 30bar, 80°C: {h_h2/1e6:.3f} MJ/kg")
    
    # Check SOEC point (requires > 647K support)
    try:
        h_h2_soec = manager.lookup('Hydrogen', 'H', 1e5, 1073.15) # SOEC 800°C
        logger.info(f"  H2 @ 1bar, 800°C: {h_h2_soec/1e6:.3f} MJ/kg (SUCCESS - Valid SOEC Range)")
    except Exception as e:
        logger.error(f"  SOEC Lookup Failed: {e}")
        
    logger.info(f"LUT Generation Complete. Files saved to {config.cache_dir}")

if __name__ == "__main__":
    generate_high_res_luts()
