
import sys
import os
import pytest
import CoolProp.CoolProp as CP
from h2_plant.components.balance_of_plant.pump import Pump
from h2_plant.components.water.water_pump import WaterPumpThermodynamic
from h2_plant.components.mixing.water_mixer import WaterMixer
from h2_plant.core.stream import Stream
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.optimization.lut_manager import LUTManager

# Adjust path to import legacy Mixer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../h2_plant/legacy/all_implemented/Modelo Mixer')))
try:
    from Mixer import mixer_model as legacy_mixer_model
except ImportError:
    print("WARNING: Could not import legacy Mixer.py")
    legacy_mixer_model = None

# Legacy Pump Model Logic (Replicated exactly from water_pump_model.py for comparison)
# The file water_pump_model.py has everything in one big function, hard to import cleanly without I/O.
def legacy_pump_forward(P1_kPa, T1_C, P2_kPa, m_dot_kg_s, eta_is, eta_m):
    fluido = 'Water'
    # Unit conversions
    P1_Pa = P1_kPa * 1000.0
    T1_K = T1_C + 273.15
    P2_Pa = P2_kPa * 1000.0
    
    h1 = CP.PropsSI('H', 'P', P1_Pa, 'T', T1_K, fluido) / 1000.0
    s1 = CP.PropsSI('S', 'P', P1_Pa, 'T', T1_K, fluido) / 1000.0
    
    h2s = CP.PropsSI('H', 'P', P2_Pa, 'S', s1*1000.0, fluido) / 1000.0
    w_is = h2s - h1
    w_real = w_is / eta_is
    h2 = h1 + w_real
    
    T2_K = CP.PropsSI('T', 'P', P2_Pa, 'H', h2*1000.0, fluido)
    T2_C = T2_K - 273.15
    
    P_fluid = m_dot_kg_s * w_real
    P_shaft = P_fluid / eta_m
    
    return {
        'h2': h2,
        'T2': T2_C,
        'Power_Shaft_kW': P_shaft
    }

def legacy_pump_reverse(P2_kPa, T2_C, P1_kPa, m_dot_kg_s, eta_is, eta_m):
    """Legacy Reverse logic uses incompressible approximation."""
    fluido = 'Water'
    P2_Pa = P2_kPa * 1000.0
    T2_K = T2_C + 273.15
    P1_Pa = P1_kPa * 1000.0
    
    h2 = CP.PropsSI('H', 'P', P2_Pa, 'T', T2_K, fluido) / 1000.0
    rho_2 = CP.PropsSI('D', 'P', P2_Pa, 'T', T2_K, fluido)
    v_avg = 1.0 / rho_2
    
    P_diff = P2_Pa - P1_Pa
    w_is = (v_avg * P_diff) / 1000.0
    w_real = w_is / eta_is
    
    h1 = h2 - w_real
    
    T1_K = CP.PropsSI('T', 'P', P1_Pa, 'H', h1*1000.0, fluido)
    T1_C = T1_K - 273.15
    
    P_fluid = m_dot_kg_s * w_real
    P_shaft = P_fluid / eta_m
    
    return {
        'h1': h1,
        'T1': T1_C,
        'Power_Shaft_kW': P_shaft
    }


def test_compare_water_mixer():
    if not legacy_mixer_model:
        pytest.skip("Legacy Mixer not found")
        
    print("\n\n=== Comparing WaterMixer vs Legacy Mixer ===")
    
    # Inputs
    inputs = [
        {'m_dot': 0.5, 'T': 15.0, 'P': 200.0},
        {'m_dot': 0.3, 'T': 80.0, 'P': 220.0},
        {'m_dot': 0.2, 'T': 50.0, 'P': 210.0}
    ]
    P_out_kPa = 200.0
    
    # Run Legacy
    print("Running Legacy Mixer...")
    # Adapt inputs to legacy dictionary format
    legacy_inputs = inputs
    _, legacy_out = legacy_mixer_model(legacy_inputs, P_out_kPa)
    
    # Run New
    print("Running New WaterMixer...")
    registry = ComponentRegistry()
    mixer = WaterMixer(outlet_pressure_kpa=P_out_kPa, max_inlet_streams=3)
    mixer.initialize(1.0, registry)
    
    for i, inp in enumerate(inputs):
        s = Stream(
            mass_flow_kg_h=inp['m_dot'] * 3600.0,
            temperature_k=inp['T'] + 273.15,
            pressure_pa=inp['P'] * 1000.0,
            composition={'H2O': 1.0}
        )
        mixer.receive_input(f"in_{i}", s, "water")
        
    mixer.step(0.0)
    new_out = mixer.get_output("outlet")
    
    # Comparison
    print(f"\nResults:")
    print(f"{'Metric':<15} | {'Legacy':<15} | {'New':<15} | {'Diff %':<10}")
    print("-" * 60)
    
    # Mass Flow
    l_m = legacy_out['Output Mass Flow Rate (kg/s)']
    n_m = new_out.mass_flow_kg_h / 3600.0
    print(f"{'Mass (kg/s)':<15} | {l_m:<15.4f} | {n_m:<15.4f} | {abs(l_m-n_m)/l_m*100:<10.2e}")
    assert abs(l_m - n_m) < 1e-6
    
    # Enthalpy
    l_h = legacy_out['Output Specific Enthalpy (kJ/kg)']
    n_h = CP.PropsSI('H', 'T', new_out.temperature_k, 'P', new_out.pressure_pa, 'Water') / 1000.0
    # Note: New component outputs T/P, enthalpy is implicit for Stream. We calculate from Stream T/P to compare.
    print(f"{'H (kJ/kg)':<15} | {l_h:<15.4f} | {n_h:<15.4f} | {abs(l_h-n_h)/abs(l_h)*100:<10.2e}")
    assert abs(l_h - n_h) < 1e-2 # 0.01 kJ/kg tolerance
    
    # Temperature
    l_T = legacy_out['Output Temperature (°C)']
    n_T = new_out.temperature_k - 273.15
    print(f"{'T (°C)':<15} | {l_T:<15.4f} | {n_T:<15.4f} | {abs(l_T-n_T)/abs(l_T)*100:<10.2e}")
    assert abs(l_T - n_T) < 1e-2


def test_compare_pump_forward():
    print("\n\n=== Comparing Pump (Forward) vs Legacy Logic ===")
    
    # Inputs
    P1_kPa = 101.325
    T1_C = 20.0
    P2_kPa = 500.0
    m_dot_kg_s = 10.0
    eta_is = 0.82
    eta_m = 0.96
    
    # Run Legacy
    legacy_res = legacy_pump_forward(P1_kPa, T1_C, P2_kPa, m_dot_kg_s, eta_is, eta_m)
    
    # Run New (WaterPumpThermodynamic)
    registry = ComponentRegistry()
    pump = WaterPumpThermodynamic(pump_id="p1", eta_is=eta_is, eta_m=eta_m, target_pressure_pa=P2_kPa*1000.0)
    pump.initialize(1.0, registry)
    
    s_in = Stream(
        mass_flow_kg_h=m_dot_kg_s * 3600.0,
        temperature_k=T1_C + 273.15,
        pressure_pa=P1_kPa * 1000.0,
        composition={'H2O': 1.0}
    )
    pump.receive_input("water_in", s_in, "water")
    pump.step(0.0)
    
    # Comparison
    print(f"\nResults:")
    print(f"{'Metric':<15} | {'Legacy':<15} | {'New':<15} | {'Diff %':<10}")
    print("-" * 60)
    
    # Power
    l_pow = legacy_res['Power_Shaft_kW']
    n_pow = pump.power_shaft_kw
    print(f"{'Power (kW)':<15} | {l_pow:<15.4f} | {n_pow:<15.4f} | {abs(l_pow-n_pow)/l_pow*100:<10.2e}")
    assert abs(l_pow - n_pow) < 1e-3
    
    # Temp Out
    l_T = legacy_res['T2']
    n_T = pump.outlet_stream.temperature_k - 273.15
    print(f"{'T_out (°C)':<15} | {l_T:<15.4f} | {n_T:<15.4f} | {abs(l_T-n_T)/abs(l_T)*100:<10.2e}")
    assert abs(l_T - n_T) < 1e-3
    
def test_compare_pump_reverse():
    print("\n\n=== Comparing Pump (Reverse) vs Legacy Logic ===")
    # Inputs
    P2_kPa = 500.0
    T2_C = 20.05
    P1_kPa = 101.325
    m_dot_kg_s = 10.0
    eta_is = 0.82
    eta_m = 0.96
    
    # Run Legacy
    legacy_res = legacy_pump_reverse(P2_kPa, T2_C, P1_kPa, m_dot_kg_s, eta_is, eta_m)
    
    # Run New (WaterPumpThermodynamic)
    registry = ComponentRegistry()
    pump = WaterPumpThermodynamic(pump_id="p1", eta_is=eta_is, eta_m=eta_m, target_pressure_pa=P1_kPa*1000.0)
    pump.initialize(1.0, registry)
    
    s_out = Stream(
        mass_flow_kg_h=m_dot_kg_s * 3600.0,
        temperature_k=T2_C + 273.15,
        pressure_pa=P2_kPa * 1000.0,
        composition={'H2O': 1.0}
    )
    pump.receive_input("water_out_reverse", s_out, "water")
    pump.step(0.0)
    
    # Comparison
    print(f"\nResults:")
    print(f"{'Metric':<15} | {'Legacy':<15} | {'New':<15} | {'Diff %':<10}")
    print("-" * 60)
    
    # Power
    l_pow = legacy_res['Power_Shaft_kW']
    n_pow = pump.power_shaft_kw
    print(f"{'Power (kW)':<15} | {l_pow:<15.4f} | {n_pow:<15.4f} | {abs(l_pow-n_pow)/l_pow*100:<10.2e}")
    assert abs(l_pow - n_pow) < 1e-3
    
    # Temp In
    l_T = legacy_res['T1']
    n_T = pump.inlet_stream.temperature_k - 273.15
    print(f"{'T_in (°C)':<15} | {l_T:<15.4f} | {n_T:<15.4f} | {abs(l_T-n_T)/abs(l_T)*100:<10.2e}")
    assert abs(l_T - n_T) < 1e-3

if __name__ == "__main__":
    # Allow running as script
    pytest.main([__file__, "-s"])
