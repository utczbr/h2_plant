import sys
import os
import logging

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from h2_plant.economics.capex_generator import CapexGenerator
from h2_plant.core.component import Component

# Mock Classes for Testing
class MockCoalescer:
    def __init__(self, volume_m3):
        self.volume_m3 = volume_m3
        self.modular_design = True 

class MockCyclone:
    def __init__(self, volume):
        self._volume = volume
    
    @property
    def cross_sectional_area_m2(self):
         # Mocking the property from HydrogenMultiCyclone
         return 10.0

def test_updates():
    print("--- Verifying CAPEX Updates ---")
    gen = CapexGenerator()
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../scenarios/Economics/equipment_mappings.yaml'))
    gen.load_config(config_path)
    
    # 1. Verify SYNGAS-CLR Coefficients
    print("\n[1] Checking SYNGAS-CLR Coefficients...")
    mapping_clr = next((m for m in gen.mappings if m.tag == "SYNGAS-CLR"), None)
    if mapping_clr and mapping_clr.coefficients:
        c = mapping_clr.coefficients
        print(f"K1={c.K1}, K2={c.K2}, K3={c.K3}")
        print(f"F_m={c.F_m}, F_p={c.F_p}")
        
        if c.K1 == 4.4646 and c.K2 == -0.530 and c.F_m == 2.7:
             print("✅ PASS: Coefficients match user request.")
        else:
             print("❌ FAIL: Coefficients mismatch.")
    else:
        print("❌ FAIL: SYNGAS-CLR mapping not found.")

    # 2. Verify Modular Logic (COA-ATR)
    print("\n[2] Checking Modular Logic (COA-ATR)...")
    mapping_coa = next((m for m in gen.mappings if m.tag == "COA-ATR"), None)
    
    if mapping_coa:
        # Check mapping flags
        is_modular = getattr(mapping_coa, 'modular_design', False)
        module_d = getattr(mapping_coa, 'module_d_shell', 0.0)
        print(f"Mapping modular_design: {is_modular}")
        print(f"Mapping module_d_shell: {module_d}")
        
        if is_modular and module_d == 0.3:
             print("✅ PASS: Mapping configured for modularity.")
        else:
             print("❌ FAIL: Mapping configuration incorrect.")
             
        # Mock calculation logic
        # Ideally we'd run gen.generate() but that needs a registry. 
        # Let's verify the logic by creating a dummy registry and mocking.
        class MockRegistry:
            def has(self, id): return True
            def get(self, id): return MockCoalescer(volume_m3=10.0)
            
        cap, num, src, notes = gen._extract_capacity(
            ['ATR_Coalescer_1'], 'volume_m3', 'sum', MockRegistry(), None
        )
        # Note: _extract_capacity returns total capacity and num=1. 
        # The splitting logic happens in generate(), let's mimic that snippet here.
        
        L_mod = 5.0 * module_d
        module_vol = (3.14159 * (module_d**2) / 4) * L_mod
        print(f"Calculated Module Volume (D=0.3, L=1.5): {module_vol:.4f} m3")
        
        import numpy as np
        total_vol = 10.0
        expected_N = int(np.ceil(total_vol / module_vol))
        print(f"For 10 m3 total, expected N = ceil(10 / {module_vol:.4f}) = {expected_N}")
        
    else:
        print("❌ FAIL: COA-ATR mapping not found.")

    # 3. Verify Cyclone Area Extraction
    print("\n[3] Checking Cyclone Area Extraction...")
    class MockRegistryCyc:
        def has(self, id): return True
        def get(self, id): return MockCyclone(volume=5.0)
        
    cap, num, src, notes = gen._extract_capacity(
        ['MockCyc'], 'cross_sectional_area_m2', 'sum', MockRegistryCyc(), None
    )
    print(f"Extracted Capacity: {cap}")
    print(f"Source: {src}")
    
    if cap == 10.0:
        print("✅ PASS: Correctly extracted cross_sectional_area_m2 property.")
    else:
        print(f"❌ FAIL: Expected 10.0, got {cap}")

if __name__ == "__main__":
    test_updates()
