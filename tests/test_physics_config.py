import sys
import os
import yaml
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from h2_plant.config.constants_physics import PEMConstants, SOECConstants
from h2_plant.config.physics_loader import load_physics_parameters

def test_physics_loading():
    print("Testing Physics Parameter Loading...")
    
    # 1. Check default values (loaded from file)
    pem = PEMConstants()
    soec = SOECConstants()
    
    print(f"PEM Membrane Thickness: {pem.delta_mem_m} m")
    print(f"PEM Conductivity: {pem.sigma_base} S/m")
    print(f"SOEC Specific Energy: {soec.SPECIFIC_ENERGY_KWH_KG} kWh/kg")
    
    # Verify correction
    if abs(pem.delta_mem_m - 100e-6) < 1e-9:
        print("✅ PEM Membrane Thickness is correct (100 microns)")
    else:
        print(f"❌ PEM Membrane Thickness incorrect: {pem.delta_mem_m}")
        
    if abs(pem.sigma_base - 10.0) < 1e-9:
        print("✅ PEM Conductivity is correct (10 S/m)")
    else:
        print(f"❌ PEM Conductivity incorrect: {pem.sigma_base}")

    if abs(soec.SPECIFIC_ENERGY_KWH_KG - 37.5) < 1e-9:
        print("✅ SOEC Specific Energy is correct (37.5 kWh/kg)")
    else:
        print(f"❌ SOEC Specific Energy incorrect: {soec.SPECIFIC_ENERGY_KWH_KG}")

if __name__ == "__main__":
    test_physics_loading()
