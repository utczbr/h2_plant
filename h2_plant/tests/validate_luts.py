
import sys
import os
import logging
from pathlib import Path
import numpy as np
import CoolProp.CoolProp as CP

# Add project root
sys.path.append(str(Path(__file__).parents[1]))

from h2_plant.optimization.lut_manager import LUTManager, LUTConfig

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("LUTValidator")

def validate_all_luts():
    print("=== LUT Cache Validation Tool ===")
    
    # Initialize Manager
    # We use default config to point to standard cache dir
    mgr = LUTManager()
    
    # Fluids to check
    fluids = ['H2', 'O2', 'N2', 'CO2', 'CH4', 'H2O']
    
    corrupted = []
    missing = []
    valid = []
    
    for fluid in fluids:
        print(f"\nChecking {fluid}...")
        
        # Check if file exists
        cache_path = mgr._get_cache_path(fluid)
        if not cache_path.exists():
            print(f"  [MISSING] Cache file not found: {cache_path.name}")
            missing.append(fluid)
            continue
            
        # Patch Config to prevent generating other fluids
        mgr.config.fluids = (fluid,)
        
        try:
            # Generate Report
            # This loads the LUT and runs sampling
            report = mgr.get_accuracy_report(fluid, num_samples=500)
            
            # Reset initialized flag to force reload for next fluid if needed (mgr handles one fluid at a time now)
            mgr._initialized = False
            
            # Analyze
            is_bad = False
            for prop, stats in report.items():
                max_err = stats['max_rel_error_pct']
                mean_err = stats['mean_rel_error_pct']
                
                print(f"  - {prop}: Mean Err={mean_err:.4f}%, Max Err={max_err:.4f}%")
                
                # Thresholds: 1% Max Error target (User mentioned <0.5% in docstring)
                if max_err > 1.0:
                    print(f"    !!! FAIL: {prop} errors exceed 1%")
                    is_bad = True
                if mean_err > 0.1:
                    print(f"    !!! WARN: {prop} mean error > 0.1%")
            
            if is_bad:
                print(f"  [CORRUPTED] {fluid} has significant errors.")
                corrupted.append(fluid)
            else:
                print(f"  [OK] {fluid} looks healthy.")
                valid.append(fluid)
                
        except Exception as e:
            print(f"  [ERROR] Failed to validate {fluid}: {e}")
            corrupted.append(fluid)

    print("\n=== Validation Summary ===")
    print(f"Valid:     {len(valid)} {valid}")
    print(f"Missing:   {len(missing)} {missing}")
    print(f"Corrupted: {len(corrupted)} {corrupted}")
    
    if corrupted:
        print("\nRecommendation: Delete corrupted cache files to force regeneration.")
        # Print paths
        for f in corrupted:
            print(f"  rm {mgr._get_cache_path(f)}")

if __name__ == "__main__":
    validate_all_luts()
