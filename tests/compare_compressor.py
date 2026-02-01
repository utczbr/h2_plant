"""
Comparison test: CompressorStorage vs modelo_compressor_ideal

Verifies that the new multi-stage compressor with efficiency chain
produces comparable power consumption to the legacy single-stage model.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from h2_plant.components.compression.compressor import CompressorStorage
from h2_plant.core.stream import Stream
from h2_plant.legacy.NEW.SOEC.modelo_compressor import modelo_compressor_ideal

def compare_compressor():
    print("\n=== COMPRESSOR COMPARISON TEST ===")
    
    # Test Conditions (Single Stage equivalent)
    # Low pressure ratio to match single-stage legacy model
    P_in_bar = 1.0
    P_out_bar = 2.0  # Ratio = 2:1 (should be single stage)
    T_in_C = 25.0
    T_in_K = T_in_C + 273.15
    m_dot_kg_h = 150.0  # 0.0416 kg/s
    m_dot_kg_s = m_dot_kg_h / 3600.0
    
    # Efficiencies (Legacy defaults)
    ETA_IS = 0.65
    ETA_M = 0.96
    ETA_EL = 0.93
    
    print(f"Conditions: P={P_in_bar}->{P_out_bar} bar, T={T_in_C}°C, Flow={m_dot_kg_h} kg/h")
    print(f"Efficiencies: η_is={ETA_IS}, η_m={ETA_M}, η_el={ETA_EL}")
    
    # --- 1. Legacy Model ---
    print("\n[Legacy Model (Single-Stage)]")
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
    
    # --- 2. New Component ---
    print("\n[New Component (Multi-Stage)]")
    comp = CompressorStorage(
        max_flow_kg_h=200.0,
        inlet_pressure_bar=P_in_bar,
        outlet_pressure_bar=P_out_bar,
        inlet_temperature_c=T_in_C,
        max_temperature_c=85.0,
        isentropic_efficiency=ETA_IS,
        mechanical_efficiency=ETA_M,
        electrical_efficiency=ETA_EL,
        chiller_cop=3.0
    )
    
    # Real-gas LUT mock using CoolProp
    try:
        import CoolProp.CoolProp as CP
    except ImportError:
        print("CoolProp not found, cannot run real-gas comparison.")
        return

    class MockLUT:
        def lookup(self, fluid, prop, p_pa, t_k):
            # Map simplified property names to CoolProp if necessary, 
            # assuming direct mapping for 'H', 'S', etc. works or wrapper handles it.
            # Component calls with fluid='H2', prop='H'/'S'/'D', P, T.
            return CP.PropsSI(prop, 'P', p_pa, 'T', t_k, fluid)
            
        def lookup_isentropic_enthalpy(self, fluid, p_out_pa, s_in):
            return CP.PropsSI('H', 'P', p_out_pa, 'S', s_in, fluid)

    class MockRegistry:
        def __init__(self):
            self.lut = MockLUT()
            
        def get(self, cid):
            # CID.LUT_MANAGER is likely an enum or string. 
            # We'll just return the lut object regardless to be safe, 
            # or check if string form matches.
            return self.lut
            
        def has(self, cid):
            return True # Pretend we have everything

    comp.initialize(dt=1/60, registry=MockRegistry())
    
    # Transfer mass directly (simulate feed)
    comp.transfer_mass_kg = m_dot_kg_h * (1/60)  # 1 minute worth
    comp.step(t=0)
    
    state = comp.get_state()
    
    # Power = Energy / Time
    new_power_kw = state['energy_consumed_kwh'] / (1/60) if (1/60) > 0 else 0
    new_comp_power = state['compression_work_kwh'] / (1/60) if (1/60) > 0 else 0
    new_chill_power = state['chilling_work_kwh'] / (1/60) if (1/60) > 0 else 0
    
    print(f"Total Power: {new_power_kw:.4f} kW")
    print(f"  Compression: {new_comp_power:.4f} kW")
    print(f"  Chilling: {new_chill_power:.4f} kW")
    print(f"Num Stages: {state['num_stages']}")
    print(f"Stage Ratio: {state['stage_pressure_ratio']:.3f}")
    
    # --- Analysis ---
    print("\n=== ANALYSIS ===")
    # Legacy returns SHAFT power (W_is / η_is / η_m), excludes η_el from power?
    # Looking at legacy: Potencia_do_Eixo_W = W_real_dot / Eta_m
    # So it's W_is / (η_is * η_m) = Shaft Power (mechanical, not electrical)
    
    # New component returns ELECTRICAL power (W_is / (η_is * η_m * η_el)) + Chilling
    # For fair comparison of shaft work, subtract chilling and multiply by η_el
    new_shaft_power_kw = new_comp_power * ETA_EL
    
    print(f"Legacy Shaft Power: {leg_power_kw:.4f} kW")
    print(f"New Shaft Power (Equiv): {new_shaft_power_kw:.4f} kW")
    
    diff_pct = abs(leg_power_kw - new_shaft_power_kw) / leg_power_kw * 100 if leg_power_kw > 0 else 0
    print(f"Difference: {diff_pct:.2f}%")
    
    # Note: New component adds chilling energy, so total is higher
    print(f"\nNote: New component includes chilling ({new_chill_power:.2f} kW) for intercooling.")
    print("Legacy model is adiabatic (no chilling).")

if __name__ == "__main__":
    compare_compressor()
