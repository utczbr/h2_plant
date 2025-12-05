import CoolProp.CoolProp as CP
import numpy as np

# Clean imports - no more importlib complexity
from framework_mocks import Stream, ComponentRegistry
from water_pump_component import WaterPump

# --- DATA FROM water_pump_model.py ---
EXEMPLO_ENTRADA = {
    'P1': 101.325, # kPa
    'T1': 20.0,    # °C
    'P_final': 500.0, # kPa
    'Vazao_m': 10.0, # kg/s
    'Eta_is': 0.82,
    'Eta_m': 0.96
}

EXEMPLO_SAIDA = {
    'P2': 500.0,   # kPa
    'T2': 20.05,   # °C
    'P_final': 101.325, # kPa (Target Inlet)
    'Vazao_m': 10.0,
    'Eta_is': 0.82,
    'Eta_m': 0.96
}

def calculate_reference_legacy(data, mode):
    """
    Exact copy of logic from water_pump_model.py
    Returns dictionary of calculated values.
    """
    fluido = 'Water'
    T_kelvin_offset = 273.15
    P_pascal_to_kPa = 1000.0
    
    Vazao_m = data['Vazao_m']
    Eta_is = data['Eta_is']
    Eta_m = data['Eta_m']
    P_final = data['P_final']

    results = {}

    if mode == 'forward':
        P1 = data['P1']
        T1 = data['T1']
        
        # LEGACY LOGIC START
        T1_K = T1 + T_kelvin_offset
        P1_Pa = P1 * P_pascal_to_kPa
        P2_Pa = P_final * P_pascal_to_kPa
        
        h1 = CP.PropsSI('H', 'P', P1_Pa, 'T', T1_K, fluido) / 1000.0
        s1 = CP.PropsSI('S', 'P', P1_Pa, 'T', T1_K, fluido) / 1000.0
        h2s = CP.PropsSI('H', 'P', P2_Pa, 'S', s1 * 1000.0, fluido) / 1000.0
        
        Trabalho_is = h2s - h1
        Trabalho_real = Trabalho_is / Eta_is
        h2 = h1 + Trabalho_real
        
        T2_K = CP.PropsSI('T', 'P', P2_Pa, 'H', h2 * 1000.0, fluido)
        T2 = T2_K - T_kelvin_offset
        # LEGACY LOGIC END
        
        results['T_final_c'] = T2
        results['work_real'] = Trabalho_real
        results['work_is'] = Trabalho_is
    
    elif mode == 'reverse':
        P2 = data['P2']
        T2 = data['T2']
        
        # LEGACY LOGIC START
        T2_K = T2 + T_kelvin_offset
        P2_Pa = P2 * P_pascal_to_kPa
        P1_Pa = P_final * P_pascal_to_kPa
        
        h2 = CP.PropsSI('H', 'P', P2_Pa, 'T', T2_K, fluido) / 1000.0
        rho_2 = CP.PropsSI('D', 'P', P2_Pa, 'T', T2_K, fluido)
        v_avg = 1.0 / rho_2
        
        P_diff = P2_Pa - P1_Pa
        w_is_kj = (v_avg * P_diff) / 1000.0
        w_real_kj = w_is_kj / Eta_is
        h1 = h2 - w_real_kj
        
        T1_K = CP.PropsSI('T', 'P', P1_Pa, 'H', h1 * 1000.0, fluido)
        T1 = T1_K - T_kelvin_offset
        # LEGACY LOGIC END

        results['T_final_c'] = T1
        results['work_real'] = w_real_kj
        results['work_is'] = w_is_kj

    results['power_fluid'] = Vazao_m * results['work_real']
    results['power_shaft'] = results['power_fluid'] / Eta_m
    
    return results

def run_comparison():
    print("="*70)
    print("WATER PUMP VALIDATION: New Implementation vs Legacy")
    print("="*70)
    print()
    
    registry = ComponentRegistry()

    # --- CASE 1: FORWARD ---
    print("Case 1: Forward (Inlet Known) - 101 kPa → 500 kPa")
    print("-" * 70)
    ref_res = calculate_reference_legacy(EXEMPLO_ENTRADA, 'forward')
    
    # Setup New Component
    pump_fwd = WaterPump(
        pump_id="pump_fwd",
        eta_is=EXEMPLO_ENTRADA['Eta_is'],
        eta_m=EXEMPLO_ENTRADA['Eta_m'],
        target_pressure_pa=EXEMPLO_ENTRADA['P_final'] * 1000.0
    )
    
    # Create Input Stream
    inlet_stream = Stream(
        mass_flow_kg_h=EXEMPLO_ENTRADA['Vazao_m'] * 3600.0,
        temperature_k=EXEMPLO_ENTRADA['T1'] + 273.15,
        pressure_pa=EXEMPLO_ENTRADA['P1'] * 1000.0,
        composition={'H2O': 1.0},
        phase='liquid'
    )
    
    pump_fwd.receive_input('water_in', inlet_stream, 'water')
    pump_fwd.step(0)
    new_state = pump_fwd.get_state()

    compare("Temperature (°C)", ref_res['T_final_c'], new_state['calculated_T_c'])
    compare("Work Real (kJ/kg)", ref_res['work_real'], new_state['work_real_kj_kg'])
    compare("Power Shaft (kW)", ref_res['power_shaft'], new_state['power_shaft_kw'])
    
    print()

    # --- CASE 2: REVERSE ---
    print("Case 2: Reverse (Outlet Known) - 500 kPa → 101 kPa")
    print("-" * 70)
    ref_res_rev = calculate_reference_legacy(EXEMPLO_SAIDA, 'reverse')
    
    # Setup New Component
    pump_rev = WaterPump(
        pump_id="pump_rev",
        eta_is=EXEMPLO_SAIDA['Eta_is'],
        eta_m=EXEMPLO_SAIDA['Eta_m'],
        target_pressure_pa=EXEMPLO_SAIDA['P_final'] * 1000.0
    )
    
    # Create Outlet Stream
    outlet_stream = Stream(
        mass_flow_kg_h=EXEMPLO_SAIDA['Vazao_m'] * 3600.0,
        temperature_k=EXEMPLO_SAIDA['T2'] + 273.15,
        pressure_pa=EXEMPLO_SAIDA['P2'] * 1000.0,
        composition={'H2O': 1.0},
        phase='liquid'
    )
    
    pump_rev.receive_input('water_out_reverse', outlet_stream, 'water')
    pump_rev.step(0)
    new_state_rev = pump_rev.get_state()
    
    compare("Temperature (°C)", ref_res_rev['T_final_c'], new_state_rev['calculated_T_c'])
    compare("Work Real (kJ/kg)", ref_res_rev['work_real'], new_state_rev['work_real_kj_kg'])
    compare("Power Shaft (kW)", ref_res_rev['power_shaft'], new_state_rev['power_shaft_kw'])
    
    print()
    print("="*70)
    print("VALIDATION COMPLETE")
    print("="*70)

def compare(label, val_ref, val_new):
    diff = abs(val_ref - val_new)
    match = diff < 1e-9
    status = "✅ PASS" if match else "❌ FAIL"
    rel_diff_pct = (diff / val_ref * 100) if val_ref != 0 else 0
    print(f"{label:20} | Ref: {val_ref:10.5f} | New: {val_new:10.5f} | Diff: {diff:.2e} ({rel_diff_pct:.4f}%) {status}")

if __name__ == "__main__":
    run_comparison()