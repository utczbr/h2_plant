import sys
import numpy as np
from unittest.mock import MagicMock

# Mock scipy BEFORE importing reference modules
sys.modules['scipy'] = MagicMock()
sys.modules['scipy'].__version__ = '1.10.0' # Fake version for Numba
sys.modules['scipy.optimize'] = MagicMock()
sys.modules['scipy.interpolate'] = MagicMock()

# Implement simple mock for fsolve
def mock_fsolve(func, x0, full_output=False, xtol=1e-4):
    # Simple Newton-Raphson
    x = float(x0)
    for _ in range(20):
        fx = func(x)
        if abs(fx) < 1e-4:
            break
        delta = 1e-5
        dfx = (func(x + delta) - fx) / delta
        if dfx == 0:
            break
        x = x - fx / dfx
    
    if full_output:
        return np.array([x]), {}, 1, "Mock Success"
    return np.array([x])

sys.modules['scipy.optimize'].fsolve = mock_fsolve

# Implement simple mock for interp1d
class MockInterp1d:
    def __init__(self, x, y, kind='linear', fill_value=None, bounds_error=False):
        self.x = x
        self.y = y
    def __call__(self, x_new):
        return np.interp(x_new, self.x, self.y)

sys.modules['scipy.interpolate'].interp1d = MockInterp1d

import os
# Add project root to path
sys.path.append(os.getcwd())

from h2_plant.components.electrolysis.pem_electrolyzer import DetailedPEMElectrolyzer as PEMElectrolyzer
from h2_plant.legacy.pem_soec_reference.ALL_Reference import pem_operator

def test_pem_parity():
    print("\nStarting PEM Parity Test...")
    
    # 1. Initialize Reference PEM
    ref_state = pem_operator.initialize_pem_simulation()
    
    # 2. Initialize New PEM Component
    # We need to mock the config
    config = {
        'max_power_mw': 5.0,
        'base_efficiency': 0.65,
        'use_polynomials': False # Force analytical for strict parity check first
    }
    new_pem = PEMElectrolyzer(config)
    # Manually set use_polynomials to False to match reference default fallback
    new_pem.use_polynomials = False 
    
    # Initialize with dt
    new_pem.initialize(dt=1.0/60.0, registry=MagicMock())

    # Test Points (MW)
    setpoints = [0.5, 1.0, 2.5, 4.0, 5.0]
    
    for sp_mw in setpoints:
        sp_kw = sp_mw * 1000.0
        
        # Reference Run
        ref_h2, ref_o2, ref_h2o, ref_state = pem_operator.run_pem_step(sp_kw, ref_state)
        
        # New Component Run
        new_pem.set_power_input_mw(sp_mw)
        new_pem.step(t=0)
        
        new_h2 = new_pem.h2_output_kg
        new_o2 = new_pem.o2_output_kg
        new_h2o = new_pem.m_H2O_kg_s * (1.0/60.0 * 3600.0)
        
        # Compare
        diff_h2 = abs(new_h2 - ref_h2)
        
        print(f"SP={sp_mw} MW | New H2={new_h2:.4f} | Ref H2={ref_h2:.4f} | Diff={diff_h2:.6f}")
        
        assert diff_h2 < 1e-3, f"Mismatch at {sp_mw} MW: New={new_h2}, Ref={ref_h2}"
        
    print("PEM Parity Test Passed!")

if __name__ == "__main__":
    test_pem_parity()
