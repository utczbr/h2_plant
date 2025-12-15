"""
Test script for CompressorSingle vs Legacy modelo_compressor.

Verifies that the new single-stage component matches legacy physics exactly.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from h2_plant.components.compression.compressor_single import CompressorSingle
from h2_plant.core.stream import Stream
from h2_plant.legacy.NEW.SOEC.modelo_compressor import modelo_compressor_ideal

def test_single_stage():
    print("\n=== SINGLE-STAGE COMPRESSOR COMPARISON ===")
    
    # Test conditions
    P_in_bar = 1.0
    P_out_bar = 2.0
    T_in_C = 25.0
    m_dot_kg_h = 150.0
    m_dot_kg_s = m_dot_kg_h / 3600.0
    
    ETA_IS = 0.65
    ETA_M = 0.96
    ETA_EL = 0.93
    
    print(f"Conditions: P={P_in_bar}->{P_out_bar} bar, T={T_in_C}°C, Flow={m_dot_kg_h} kg/h")
    
    # --- Legacy Model ---
    print("\n[Legacy Model]")
    legacy_res = modelo_compressor_ideal(
        fluido_nome='hydrogen',
        T_in_C=T_in_C,
        P_in_Pa=P_in_bar * 1e5,
        P_out_Pa=P_out_bar * 1e5,
        m_dot_mix_kg_s=m_dot_kg_s,
        m_dot_gas_kg_s=m_dot_kg_s,
        Eta_is=ETA_IS,
        Eta_m=ETA_M,
        Eta_el=ETA_EL
    )
    
    leg_power_kw = legacy_res['W_dot_comp_W'] / 1000.0
    leg_t_out = legacy_res['T_C']
    leg_t_is = legacy_res['T_out_isentropic_C']
    
    print(f"Shaft Power: {leg_power_kw:.4f} kW")
    print(f"T_out (Actual): {leg_t_out:.2f} °C")
    print(f"T_out (Isentropic): {leg_t_is:.2f} °C")
    
    # --- New Component ---
    print("\n[CompressorSingle]")
    comp = CompressorSingle(
        max_flow_kg_h=200.0,
        inlet_pressure_bar=P_in_bar,
        outlet_pressure_bar=P_out_bar,
        inlet_temperature_c=T_in_C,
        isentropic_efficiency=ETA_IS,
        mechanical_efficiency=ETA_M,
        electrical_efficiency=ETA_EL
    )
    
    class MockRegistry:
        def get(self, cid):
            return None
        def has(self, cid):
            return False
    
    comp.initialize(dt=1/60, registry=MockRegistry())
    comp.transfer_mass_kg = m_dot_kg_h * (1/60)
    comp.step(t=0)
    
    state = comp.get_state()
    
    new_power_kw = state['energy_consumed_kwh'] / (1/60)
    new_t_out = state['outlet_temperature_c']
    new_t_is = state['outlet_temperature_isentropic_c']
    
    print(f"Electrical Power: {new_power_kw:.4f} kW")
    print(f"T_out (Actual): {new_t_out:.2f} °C")
    print(f"T_out (Isentropic): {new_t_is:.2f} °C")
    
    # --- Comparison ---
    print("\n=== COMPARISON ===")
    # Note: Legacy returns Shaft Power (before η_el), New returns Electrical Power
    # For fair comparison, convert New to shaft: new_shaft = new_elec * η_el
    new_shaft_kw = new_power_kw * ETA_EL
    
    print(f"Legacy Shaft Power: {leg_power_kw:.4f} kW")
    print(f"New Shaft Power (Equiv): {new_shaft_kw:.4f} kW")
    
    power_diff = abs(leg_power_kw - new_shaft_kw) / leg_power_kw * 100 if leg_power_kw > 0 else 0
    temp_diff = abs(leg_t_out - new_t_out)
    temp_is_diff = abs(leg_t_is - new_t_is)
    
    print(f"\nPower Difference: {power_diff:.4f}%")
    print(f"T_out Difference: {temp_diff:.4f} °C")
    print(f"T_is Difference: {temp_is_diff:.4f} °C")
    
    if power_diff < 0.01 and temp_diff < 0.1:
        print("\n✓ PARITY ACHIEVED")
    else:
        print("\n✗ DISCREPANCY DETECTED")

if __name__ == "__main__":
    test_single_stage()
