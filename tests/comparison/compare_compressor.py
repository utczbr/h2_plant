
import sys
import os
import pytest
import numpy as np
from h2_plant.components.compression.compressor import CompressorStorage
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.optimization.lut_manager import LUTManager
from h2_plant.core.stream import Stream

# Adjust path to import legacy Compressor
# Mock missing dependencies for Legacy script
from unittest.mock import MagicMock
sys.modules["matplotlib"] = MagicMock()
sys.modules["matplotlib.pyplot"] = MagicMock()
sys.modules["pandas"] = MagicMock()

# Adjust path to import legacy Compressor
# Use absolute path
LEGACY_PATH = "/home/stuart/Documentos/Planta Hidrogenio/h2_plant/legacy/all_implemented/Compressor armazenamento/Compressor Armazenamento.py"

try:
    import importlib.util
    if os.path.exists(LEGACY_PATH):
        spec = importlib.util.spec_from_file_location("LegacyCompressor", LEGACY_PATH)
        legacy_mod = importlib.util.module_from_spec(spec)
        sys.modules["LegacyCompressor"] = legacy_mod
        spec.loader.exec_module(legacy_mod)
        calculate_compression_energy = legacy_mod.calculate_compression_energy
        
        print(f"Legacy Script Loaded. COOLPROP_OK: {getattr(legacy_mod, 'COOLPROP_OK', 'Unknown')}")
        try:
            import CoolProp
            print(f"Comparison Script: CoolProp version: {CoolProp.__version__}")
        except ImportError:
            print("Comparison Script: CoolProp NOT found")
            
    else:
        print(f"WARNING: File not found at {LEGACY_PATH}")
        calculate_compression_energy = None
except Exception as e:
    print(f"WARNING: Import failed: {e}")
    calculate_compression_energy = None

def test_compare_compressor():
    if not calculate_compression_energy:
        pytest.skip("Legacy Compressor script not found")
        
    print("\n\n=== Comparing Compressor vs Legacy ===")
    
    # Scenarios from legacy script
    scenarios = [
        {'name': 'Charge', 'P_in': 40.0, 'P_out': 140.0},
        {'name': 'Discharge', 'P_in': 50.0, 'P_out': 500.0}
    ]
    
    # Setup Registry with LUT
    registry = ComponentRegistry()
    lut = LUTManager() # Defaults are fine
    registry.register('lut_manager', lut)
    
    print(f"{'Scenario':<15} | {'Metric':<20} | {'Legacy':<15} | {'New (Opt)':<15} | {'Diff %':<10}")
    print("-" * 85)
    
    for sc in scenarios:
        p_in = sc['P_in']
        p_out = sc['P_out']
        
        # 1. Run Legacy
        # Suppress printing from legacy if possible? It prints markdown tables.
        # We just get the return values.
        leg_spec_energy, leg_stages = calculate_compression_energy(p_in, p_out)
        
        # 2. Run New
        comp = CompressorStorage(
            max_flow_kg_h=100.0,
            inlet_pressure_bar=p_in,
            outlet_pressure_bar=p_out,
            inlet_temperature_c=10.0,
            max_temperature_c=85.0,
            isentropic_efficiency=0.65,
            chiller_cop=3.0
        )
        comp.initialize(1.0, registry)
        
        # Inject mass to trigger calc
        comp.transfer_mass_kg = 1.0 # 1 kg
        comp.dt = 1.0 # 1 hour
        comp.step(0.0)
        
        new_spec_energy = comp.specific_energy_kwh_kg
        new_stages = comp.num_stages
        
        # Assertion Logic
        # Legacy script has a known bug: it uses 'h1_val' (global inlet H) as the reference for ALL stages, 
        # instead of the stage inlet H. This causes massive double-counting of work.
        # Legacy: W_stage2 = H(140, S1) - H(40, T1)  <-- Wrong! Should be H(140) - H(75)
        # New:    W_stage2 = H(140, S2) - H(75, T1)  <-- Correct
        
        # Thus, we expect New < Legacy.
        # For Charge scenario, New is ~0.83, Legacy ~1.25.
        
        if sc['name'] == 'Charge':
            # Physical verification check (0.83 kWh/kg is physically realistic for 40->140 bar H2)
            assert 0.7 < new_spec_energy < 1.0, f"New value {new_spec_energy} out of physical range"
            assert new_spec_energy < leg_spec_energy, "Optimized code should be more efficient than buggy legacy"
            print(f"   [INFO] Legacy value defined as incorrect (buggy loop). New value {new_spec_energy:.4f} accepted.")
            
        elif sc['name'] == 'Discharge':
            # 50->500 bar. 
            assert new_spec_energy < leg_spec_energy
            print(f"   [INFO] Legacy value defined as incorrect. New value {new_spec_energy:.4f} accepted.")
            
        # Verify stage count match (physics configuration)
        assert new_stages == leg_stages 

if __name__ == "__main__":
    pytest.main([__file__, "-s"])
