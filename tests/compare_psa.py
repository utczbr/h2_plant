import sys
import os
import numpy as np

# Adjust path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from h2_plant.core.stream import Stream
from h2_plant.components.separation.psa import PSA
from h2_plant.legacy.NEW.PEM.modulos.modelo_psa import modelar_psa

def compare_psa():
    print("\n=== PSA COMPARISON TEST ===")
    
    # Inputs
    m_dot_h = 100.0  # kg/h
    m_dot_s = m_dot_h / 3600.0
    T_in_C = 25.0
    T_in_K = T_in_C + 273.15
    P_in_bar = 30.0
    P_in_Pa = P_in_bar * 1e5
    
    # Composition: H2 saturated with some H2O, trace O2
    # Legacy Function takes: y_H2O_in, y_O2_in (Mole Fractions)
    # PSA Component takes: Stream (Mass Fractions)
    
    # Target Mole Fractions
    y_H2O = 0.01 # 1% water
    y_O2 = 0.001 # 0.1% O2
    y_H2 = 1.0 - y_H2O - y_O2
    
    # Convert to Mass Fractions for PSA Component
    MW_H2 = 2.016e-3
    MW_O2 = 32.0e-3
    MW_H2O = 18.015e-3
    
    MW_mix = y_H2 * MW_H2 + y_O2 * MW_O2 + y_H2O * MW_H2O
    
    x_H2 = (y_H2 * MW_H2) / MW_mix
    x_O2 = (y_O2 * MW_O2) / MW_mix
    x_H2O = (y_H2O * MW_H2O) / MW_mix
    
    print(f"Input Conf: Flow={m_dot_h} kg/h, P={P_in_bar} bar")
    print(f"Mole Fracs: H2={y_H2:.4f}, O2={y_O2:.4f}, H2O={y_H2O:.4f}")
    
    # --- 1. Run Legacy Model ---
    print("\n[Legacy Model Run]")
    legacy_res = modelar_psa(
        m_dot_g_kg_s=m_dot_s,
        P_in_bar=P_in_bar,
        T_in_C=T_in_C,
        y_H2O_in=y_H2O,
        y_O2_in=y_O2 
    )
    
    # Legacy outputs
    # It returns m_dot_gas_out_kg_s (H2 product + O2 pass through)
    # It returns H2_Perdido_kg_s
    # It returns Agua_Removida_kg_s
    
    leg_out_flow = legacy_res['m_dot_gas_out_kg_s'] * 3600
    leg_p_out = legacy_res['P_out_bar']
    leg_power = legacy_res['W_dot_comp_W'] / 1000.0 # kW
    leg_dp = legacy_res['DELTA_P_PSA_bar']
    
    print(f"Product Flow: {leg_out_flow:.4f} kg/h")
    print(f"Pressure Drop: {leg_dp:.5f} bar")
    print(f"P_out: {leg_p_out:.4f} bar")
    print(f"Power: {leg_power:.4f} kW")
    print(f"H2O Out (Legacy y_vap): {legacy_res['y_H2O_out']:.2e}")
    
    # --- 2. Run New Component ---
    print("\n[New Component Run]")
    # Legacy constants: ETA_REC_H2 = 0.90
    psa = PSA(
        recovery_rate=0.90,
        cycle_time_min=5.0, # Match logic assumption
        power_consumption_kw=1.0 # Base load from logic
    )
    
    stream = Stream(
        mass_flow_kg_h=m_dot_h,
        temperature_k=T_in_K,
        pressure_pa=P_in_Pa,
        composition={'H2': x_H2, 'O2': x_O2, 'H2O': x_H2O}
    )
    
    psa.initialize(1/60, None)
    psa.receive_input('gas_in', stream)
    psa.step(0)
    
    out_stream = psa.get_output('purified_gas_out')
    
    new_out_flow = out_stream.mass_flow_kg_h
    new_p_out = out_stream.pressure_pa / 1e5
    state = psa.get_state()
    new_power = state['power_consumption_kw']
    new_dp = (P_in_Pa - out_stream.pressure_pa) / 1e5
    
    print(f"Product Flow: {new_out_flow:.4f} kg/h")
    print(f"Pressure Drop: {new_dp:.5f} bar")
    print(f"P_out: {new_p_out:.4f} bar")
    print(f"Power: {new_power:.4f} kW")
    
    # --- Composition Check ---
    # Legacy passes O2. Component removes it?
    # Component "purity_target" = 0.9999 H2. 
    # Impurities H2O, O2 shared remaining 0.0001
    
    new_comp = out_stream.composition
    print(f"New Comp Mass: {new_comp}")
    
    # --- Analysis ---
    # Flow comparison validation
    # Legacy Product = H2_in * 0.9 + O2_in
    # New Product = Total_in * 0.9 ? No.
    # New code: product_flow = inlet_flow_kg_h * self.recovery_rate
    # This is TOTAL MASS RECOVERY of 90%.
    # Legacy is H2 SPECIFIC RECOVERY of 90%.
    # This is a Logic Gap.
    
    diff_flow = abs(leg_out_flow - new_out_flow)
    diff_p = abs(leg_p_out - new_p_out)
    diff_power = abs(leg_power - new_power)
    
    print(f"\nDiff Flow: {diff_flow:.4f} kg/h")
    print(f"Diff Pressure: {diff_p:.5f} bar")
    print(f"Diff Power: {diff_power:.4f} kW")

if __name__ == "__main__":
    compare_psa()
