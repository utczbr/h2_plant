
import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.getcwd())

from h2_plant.components.reforming.atr_recovery import ATRSyngasCooler
from h2_plant.core.stream import Stream
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.components.reforming.atr_data_manager import ATRDataManager

def test_atr_recovery():
    print("Initializing ATRSyngasCooler...")
    
    # 1. Setup Registry and Loader
    registry = ComponentRegistry()
    # Ensure Data Manager loads
    mgr = ATRDataManager()
    mgr.load_data()
    
    # 2. Extract Max Point values for reference
    # x = 23.75 kmol/h O2 (Max point in most CSVs is usually lower, let's check range)
    # The regression range was 7.125 to 23.75.
    # Max Flow (from previous task): O2=757.54, Bio=1959.59, Steam=2331.95. Total = 5049.08.
    # Corresponds to F_O2 = 757.54 / 32 = 23.67 kmol/h.
    
    # Let's target this max point.
    target_f_o2 = 23.673 # approx
    total_mass_in = 5049.08
    
    # Lookup Expected Target Temp (Tin_H05)
    expected_temp_c = mgr.lookup('Tin_H05_func', target_f_o2)
    print(f"Reference: For Mass={total_mass_in:.2f} kg/h (F_O2~{target_f_o2:.2f}), Expected Tin_H05 = {expected_temp_c:.2f} C")
    
    # 3. Instantiate Component
    cooler = ATRSyngasCooler('ATR_Recovery_Test')
    cooler.initialize(1.0, registry)
    
    # Verify Inverse Table
    print("\nInverse Lookup Verification:")
    inferred_fo2 = cooler._infer_f_o2(total_mass_in)
    print(f"Input Mass: {total_mass_in:.2f} -> Inferred F_O2: {inferred_fo2:.4f} kmol/h")
    
    if abs(inferred_fo2 - target_f_o2) > 0.1:
        print("WARNING: Inference significantly different from manual calc.")
    
    # 4. Create Streams
    # Syngas In (Hot) - H2, CO, etc. (Approx composition from walkthrogh)
    # 66% H2, 10% CO, 10% CO2 ... 
    # Let's allow simple composition. Main factor is Enthalpy.
    # T_in > T_target. Reference says LTWGS out (Tin_H05 upstream).
    # Wait, Reference says: LTWGS -> H05.
    # So T_in should correspond to Tout_LTWGS.
    t_in_c = mgr.lookup('Tout_H09_func', target_f_o2) # H09 is LTWGS
    print(f"Inlet Temp (from H09): {t_in_c:.2f} C")
    
    syngas = Stream(
        mass_flow_kg_h=total_mass_in,
        temperature_k=t_in_c + 273.15,
        pressure_pa=14.5 * 100000,
        composition={'H2': 0.66, 'CO': 0.10, 'CO2': 0.10, 'H2O': 0.10, 'CH4': 0.01, 'N2': 0.03},
        phase='gas'
    )
    
    water = Stream(
        mass_flow_kg_h=10000.0,
        temperature_k=25.0 + 273.15, # 25C
        pressure_pa=3.0 * 100000,
        composition={'H2O': 1.0}, # Pure Water (Liquid implicitly)
        phase='liquid'
    )
    water.composition['H2O_liq'] = 1.0 # Explicitly mark liquid
    
    # 5. Run Step
    cooler.receive_input('syngas_in', syngas)
    cooler.receive_input('water_in', water)
    cooler.step(0.0)
    
    # 6. Check Outputs
    out_syngas = cooler.get_output('syngas_out')
    out_water = cooler.get_output('water_out')
    
    print("\nResults:")
    print(f"Syngas Out Temp: {out_syngas.temperature_k - 273.15:.2f} C (Target: {expected_temp_c:.2f} C)")
    print(f"Water Out Temp: {out_water.temperature_k - 273.15:.2f} C (In: 25.00 C)")
    print(f"Q Transferred: {cooler.q_transferred_kw:.2f} kW")
    
    if abs((out_syngas.temperature_k - 273.15) - expected_temp_c) < 1.0:
        print("\nSUCCESS: Syngas temperature matches regression target.")
    else:
        print("\nFAILURE: Temperature mismatch.")
        
test_atr_recovery()
