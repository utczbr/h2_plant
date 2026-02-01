import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from h2_plant.economics.capex_generator import CapexGenerator
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.components.cooling.dry_cooler import DryCooler
from h2_plant.core.constants import DryCoolerIndirectConstants as DCC

def test_capex_fix():
    print("--- Starting Verification ---")
    
    # Setup Generator
    gen = CapexGenerator()
    
    # Case 1: Central Utility (Should be TQC Only)
    comp_central = DryCooler("intercooler_central", design_capacity_kw=1800.0, use_central_utility=True)
    
    val_central, note_central = gen._calculate_capacity_from_attributes(comp_central, "intercooler_central", "area_m2")
    
    print(f"\n[Case 1] Central Utility (Expected ~180.0 m²)")
    print(f"Result: {val_central:.2f} m²")
    print(f"Note:   {note_central}")
    
    # Explicit check for 180.0 (1800kW / 100kW * 10m2)
    expected_central = (1800.0 / 100.0) * DCC.AREA_H2_TQC_M2
    if abs(val_central - expected_central) < 0.1:
        print("✅ PASS: Correctly calculated TQC-only area.")
    else:
        print(f"❌ FAIL: Expected {expected_central}, got {val_central}")

    # Case 2: Standalone (Should be TQC + DC)
    comp_standalone = DryCooler("intercooler_standalone", design_capacity_kw=1800.0, use_central_utility=False)
    
    val_standalone, note_standalone = gen._calculate_capacity_from_attributes(comp_standalone, "intercooler_standalone", "area_m2")
    
    print(f"\n[Case 2] Standalone (Expected ~8345.2 m²)")
    print(f"Result: {val_standalone:.2f} m²")
    print(f"Note:   {note_standalone}")
    
    expected_standalone = (1800.0 / 100.0) * (DCC.AREA_H2_TQC_M2 + DCC.AREA_H2_DC_M2)
    if abs(val_standalone - expected_standalone) < 0.1:
        print("✅ PASS: Correctly calculated full area (TQC + DC).")
    else:
         print(f"❌ FAIL: Expected {expected_standalone}, got {val_standalone}")

if __name__ == "__main__":
    test_capex_fix()
