
import sys
import os
import logging

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from h2_plant.economics.capex_generator import CapexGenerator
from h2_plant.economics.models import EquipmentMapping, CEPCIData

def investigate_inflation():
    print("--- Investigating Inflation on Direct Quotes ---")
    gen = CapexGenerator()
    
    # Set CEPCI to simulate inflation
    # Base = 2001 (397), Current = 2026 (820)
    # Inflation Factor = 820 / 397 = 2.065
    gen.cepci = CEPCIData(current_year=2026, current_index=820.0)
    
    print(f"Current Year: {gen.cepci.current_year}, Index: {gen.cepci.current_index}")

    # Case 1: cost_source = "fixed"
    print("\n[Case 1] cost_source = 'fixed'")
    mapping_fixed = EquipmentMapping(
        tag="FIX-1", name="Fixed Cost Item",
        topology_ids=["Fixed1"], component_type="General",
        cost_source="fixed",
        vendor_quote_usd=1000.0
    )
    
    cp0, cbm, formula, cls, bounds = gen._calculate_cost(100.0, mapping_fixed)
    print(f"Quote: $1000.0")
    print(f"Calculated C_BM: ${cbm:,.2f}")
    if cbm == 1000.0:
        print("RESULT: Inflation NOT applied.")
    else:
        print(f"RESULT: Inflation APPLIED (Factor: {cbm/1000.0:.4f})")

    # Case 2: cost_source = "vendor_quote" (Default behavior)
    print("\n[Case 2] cost_source = 'vendor_quote' (Defaults)")
    mapping_vq = EquipmentMapping(
        tag="VQ-1", name="Vendor Quote Item",
        topology_ids=["VQ1"], component_type="General",
        cost_source="vendor_quote",
        vendor_quote_usd=1000.0
    )
    
    # VendorQuoteStrategy default reference year is 2024 (CEPCI 800)
    # Expected Inflation: 820 / 800 = 1.025
    cp0, cbm, formula, cls, bounds = gen._calculate_cost(100.0, mapping_vq)
    print(f"Quote: $1000.0")
    print(f"Calculated C_BM: ${cbm:,.2f}")
    print(f"Formula: {formula}")
    
    expected_cbm = 1000.0 * (820.0 / 800.0)
    if abs(cbm - expected_cbm) < 0.1:
        print(f"RESULT: Default Inflation APPLIED (Factor 1.025). matches 2024->2026")
    elif cbm == 1000.0:
        print("RESULT: Inflation NOT applied.")
    else:
        print(f"RESULT: Unexpected Inflation (Factor: {cbm/1000.0:.4f})")

if __name__ == "__main__":
    investigate_inflation()
