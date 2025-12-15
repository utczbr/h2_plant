import sys
import os
import numpy as np

# Adjust path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from h2_plant.core.stream import Stream
from h2_plant.components.thermal.chiller import Chiller
from h2_plant.legacy.NEW.PEM.modulos.modelo_chiller import modelar_chiller_gas

def compare_chiller():
    print("\n=== CHILLER COMPARISON TEST ===")
    
    # Inputs
    m_dot_h = 100.0  # kg/h
    m_dot_s = m_dot_h / 3600.0
    T_in_C = 80.0
    T_in_K = T_in_C + 273.15
    P_in_bar = 30.0
    P_in_Pa = P_in_bar * 1e5
    T_target_C = 25.0
    T_target_K = T_target_C + 273.15
    COP = 4.0
    DP_bar = 0.2
    
    gas_type = 'H2'
    # Composition Mass Fractions
    comp = {'H2': 1.0}
    
    # --- 1. Runs Legacy Model ---
    # Legacy requires Enthalpies OR uses fallback. 
    # To test parity of fundamental equations (fallback vs fallback), we pass H=0.
    # To test parity of Enthalpy, we need consistent H calculations.
    # The new component calculates H internally. Legacy expects it injected.
    # We will let Legacy use its fallback (Cp=14300) and check if New component's default H calculation differs.
    
    print("\n[Legacy Model Run]")
    legacy_res = modelar_chiller_gas(
        gas_fluido=gas_type,
        m_dot_mix_kg_s=m_dot_s,
        P_in_bar=P_in_bar,
        T_in_C=T_in_C,
        T_out_C_desejada=T_target_C,
        COP_chiller=COP,
        Delta_P_estimado=DP_bar,
        H_in_J_kg=0.0, # Trigger fallback
        H_out_J_kg=0.0
    )
    
    print(f"Q_cool (Kw): {legacy_res['Q_dot_fluxo_W']/1000:.4f}")
    print(f"W_elec (Kw): {legacy_res['W_dot_comp_W']/1000:.4f}")
    print(f"P_out (bar): {legacy_res['P_bar']:.4f}")
    
    # --- 2. Run New Component ---
    print("\n[New Component Run]")
    chiller = Chiller(
        cooling_capacity_kw=1000.0, # Unconstrained
        cop=COP,
        target_temp_k=T_target_K,
        pressure_drop_bar=DP_bar,
        enable_dynamics=False
    )
    
    stream = Stream(
        mass_flow_kg_h=m_dot_h,
        temperature_k=T_in_K,
        pressure_pa=P_in_Pa,
        composition=comp
    )
    
    chiller.initialize(1/60, None)
    chiller.receive_input('fluid_in', stream)
    chiller.step(0)
    
    state = chiller.get_state()
    
    print(f"Q_cool (Kw): {-state['cooling_load_kw']:.4f}") # Legacy uses negative sign for cooling? 
    # Legacy: Q = -m * Cp * dT. With dT positive (80-25), Q is negative.
    # New: cooling_load_kw is magnitude (positive definition in docstring?) or algebra?
    # Docstring: Q = m*(h_in - h_target). If h_in > h_target (cooling), Q is Positive.
    # But Legacy returns Negative for cooling.
    # We will compare magnitudes.
    
    print(f"W_elec (Kw): {state['electrical_power_kw']:.4f}")
    print(f"P_out (bar): {state['outlet_pressure_bar']:.4f}")
    
    # --- Analysis ---
    q_legacy = abs(legacy_res['Q_dot_fluxo_W']) / 1000.0
    q_new = abs(state['cooling_load_kw'])
    
    diff_q = abs(q_legacy - q_new) / q_legacy * 100
    
    print(f"\nDiscrepancy Q_cool: {diff_q:.4f}%")
    
    # Check if discrepancy is due to Cp 14300 vs Real Enthalpy
    # Stream.specific_enthalpy uses Shomate.
    # Legacy Fallback uses 14300.
    # Let's see how close Shomate is to 14300 for H2.
    
    if diff_q > 1.0:
        print("Note: Discrepancy likely due to Shomate (Real Gas) vs Constant Cp (14300).")
        print("Legacy uses Fallback if H unprovided. New uses Shomate by default.")

if __name__ == "__main__":
    compare_chiller()
