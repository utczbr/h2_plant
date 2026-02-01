
import sys
import os
import math
import numpy as np

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from h2_plant.economics.capex_generator import CapexGenerator
from h2_plant.economics.models import EquipmentMapping, CEPCIData, CostCoefficients

def verify_pump_costing():
    print("--- Verifying Composite Pump Costing (Pump + Motor) ---")
    gen = CapexGenerator()
    
    # 1. Setup Environment
    # Use Base Index 2001=397.0, Current 2026=820.0
    # Inflation Factor = 820/397 = 2.06549
    gen.cepci = CEPCIData(current_year=2026, current_index=820.0, base_index=397.0)
    inflation_factor = 820.0 / 397.0
    print(f"Inflation Factor: {inflation_factor:.4f}")

    # 2. Define Test Component (Pump)
    # Shaft Power = 50 kW
    pump_power_kw = 50.0
    
    mapping = EquipmentMapping(
        tag="PMP-101",
        name="Test Pump (50kW)",
        topology_ids=["PMP1"],
        component_type="Centrifugal Pump",
        capacity_unit="kW",
        # Explicit Coefficients for Pump (from Default)
        # K1: 3.3892, K2: 0.0536, K3: 0.1538
        # F_m: 2.0, F_BM: 3.30
        coefficients=CostCoefficients(
             K1=3.3892, K2=0.0536, K3=0.1538,
             F_m=2.0, F_BM=3.30
        )
    )
    
    # 3. Expected Calculation (Mental Check)
    # A) Pump Head Cost
    log_p = math.log10(pump_power_kw)
    log_cp0_pump = 3.3892 + 0.0536*log_p + 0.1538*(log_p**2)
    cp0_pump_base = 10**log_cp0_pump # USD 2001
    cp0_pump_curr = cp0_pump_base * inflation_factor
    cbm_pump = cp0_pump_curr * 3.30 * 2.0 # F_BM * F_m
    
    print(f"\n[Expected Pump]")
    print(f"Shaft Power: {pump_power_kw} kW")
    print(f"Log10 Cost: {log_cp0_pump:.4f}")
    print(f"Cp0 (Base 2001): ${cp0_pump_base:.2f}")
    print(f"Cp0 (Current): ${cp0_pump_curr:.2f}")
    print(f"C_BM (Pump Only): ${cbm_pump:.2f}")
    
    # B) Motor Cost (Tier 1: <= 75 kW)
    # Efficiency = 0.90
    motor_power = pump_power_kw / 0.90 # 55.55 kW
    # K1: 3.3432, K2: 0.2761, K3: 0.0543
    log_pm = math.log10(motor_power)
    log_cp0_motor = 3.3432 + 0.2761*log_pm + 0.0543*(log_pm**2)
    cp0_motor_base = 10**log_cp0_motor
    cbm_motor = cp0_motor_base * inflation_factor # Treated as bare purchase + included installation? 
    # Logic implementation said: inflated_cost = cp0_usd * inflation
    
    print(f"\n[Expected Motor]")
    print(f"Motor Power: {motor_power:.2f} kW")
    print(f"Tier: Small (<=75)")
    print(f"Log10 Cost: {log_cp0_motor:.4f}")
    print(f"Cp0 (Base 2001): ${cp0_motor_base:.2f}")
    print(f"C_BM (Motor): ${cbm_motor:.2f}")
    
    expected_total_cbm = cbm_pump + cbm_motor
    print(f"\nExpected Total C_BM: ${expected_total_cbm:,.2f}")

    # 4. Run Actual Code
    cp0, cbm, formula, cls, bounds = gen._calculate_cost(pump_power_kw, mapping)
    
    print(f"\n[Actual Result]")
    print(f"Total C_BM: ${cbm:,.2f}")
    print(f"Formula: {formula}")
    
    # 5. Verify
    if abs(cbm - expected_total_cbm) < 1.0: # 1 USD tolerance
        print("\n✅ PASS: Calculation matches expected values.")
    else:
        print(f"\n❌ FAIL: Diff = {cbm - expected_total_cbm:.2f}")

if __name__ == "__main__":
    verify_pump_costing()
