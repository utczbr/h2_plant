#!/usr/bin/env python3
"""
Direct comparison of Legacy vs New KnockOutDrum implementations.

This script feeds identical inlet conditions to both implementations
and compares output pressure, water vapor fraction, gas flow, and status.
"""
import sys
sys.path.insert(0, '/home/stuart/Documentos/Planta Hidrogenio')
sys.path.insert(0, '/home/stuart/Documentos/Planta Hidrogenio/h2_plant/legacy/NEW/PEM')
sys.path.insert(0, '/home/stuart/Documentos/Planta Hidrogenio/h2_plant/legacy/NEW/PEM/modulos')

# Import Legacy model
from modelo_kod import modelar_knock_out_drum

# Import New component
from h2_plant.components.separation.knock_out_drum import KnockOutDrum
from h2_plant.core.stream import Stream

# ==============================================================
# INLET CONDITIONS (Matching Legacy simulation after Chiller)
# ==============================================================
gas_fluido = 'H2'
T_in_C = 4.0       # After chiller
P_in_bar = 39.90
m_dot_g_kg_s = 0.02472  # Main gas flow
y_H2O_in = 2.04e-4      # ~204 ppm molar
m_dot_H2O_liq_in_kg_s = 0.0  # No liquid carryover from chiller

# ==============================================================
# 1. LEGACY MODEL
# ==============================================================
print("=" * 70)
print("LEGACY MODEL (modelo_kod.py)")
print("=" * 70)

legacy_result = modelar_knock_out_drum(
    gas_fluido=gas_fluido,
    m_dot_g_kg_s=m_dot_g_kg_s,
    P_in_bar=P_in_bar,
    T_in_C=T_in_C,
    y_H2O_in=y_H2O_in,
    m_dot_H2O_liq_in_kg_s=m_dot_H2O_liq_in_kg_s
)

if 'erro' in legacy_result:
    print(f"ERROR: {legacy_result['erro']}")
else:
    print(f"Inlet: T={T_in_C}°C, P={P_in_bar} bar, y_H2O={y_H2O_in*1e6:.1f} ppm")
    print(f"Outlet P: {legacy_result['P_bar']:.2f} bar")
    print(f"Outlet y_H2O: {legacy_result['y_H2O_out_vap']*1e6:.2f} ppm")
    print(f"Gas Out: {legacy_result['m_dot_gas_out_kg_s']*3600:.4f} kg/h")
    print(f"Water Removed: {legacy_result['Agua_Condensada_removida_kg_s']*3600:.4f} kg/h")
    print(f"Dissolved Gas: {legacy_result['Gas_Dissolvido_removido_kg_s']*1e6*3600:.4f} mg/h")
    print(f"Status: {legacy_result['Status_KOD']}")

# ==============================================================
# 2. NEW COMPONENT
# ==============================================================
print("\n" + "=" * 70)
print("NEW COMPONENT (KnockOutDrum)")
print("=" * 70)

# Create component
kod = KnockOutDrum(
    diameter_m=1.0,
    delta_p_bar=0.05,
    gas_species='H2'
)

# Create inlet stream
# Convert inputs for Stream: mole fraction -> mass fraction approximation
# For H2-dominated stream, mass frac ≈ mole frac × (M_H2O/M_H2)
MW_H2O = 18.015
MW_H2 = 2.016
x_H2O_approx = y_H2O_in * (MW_H2O / MW_H2)  # Approx mass fraction

inlet_stream = Stream(
    mass_flow_kg_h=m_dot_g_kg_s * 3600,
    temperature_k=T_in_C + 273.15,
    pressure_pa=P_in_bar * 1e5,
    composition={'H2': 1.0 - x_H2O_approx, 'H2O': x_H2O_approx},
    phase='gas'
)

# Initialize component properly
kod._component_id = 'KOD_Test'
kod._initialized = True
kod.dt = 1/60
kod._input_stream = inlet_stream
kod._current_time = 0.0

# Run step
kod.step(0.0)

# Get results
gas_out = kod._gas_outlet_stream
liquid_out = kod._liquid_drain_stream

print(f"Inlet: T={T_in_C}°C, P={P_in_bar} bar, x_H2O={x_H2O_approx*1e6:.1f} ppm (mass)")
print(f"Outlet P: {gas_out.pressure_pa/1e5:.2f} bar")
print(f"Outlet x_H2O: {gas_out.composition.get('H2O', 0)*1e6:.2f} ppm (mass)")
print(f"Gas Out: {gas_out.mass_flow_kg_h:.4f} kg/h")
if liquid_out:
    print(f"Water Removed: {liquid_out.mass_flow_kg_h:.4f} kg/h")
else:
    print(f"Water Removed: 0.0000 kg/h")
print(f"Dissolved Gas: {kod._dissolved_gas_kg_h*1e6:.4f} mg/h")
print(f"Status: {kod._separation_status}")
print(f"V_real: {kod._v_real:.6f} m/s, V_max: {kod._v_max:.4f} m/s")

# ==============================================================
# 3. COMPARISON SUMMARY
# ==============================================================
print("\n" + "=" * 70)
print("COMPARISON SUMMARY")
print("=" * 70)
print(f"{'Metric':<25} | {'Legacy':>15} | {'New':>15} | {'Match?':>8}")
print("-" * 70)
print(f"{'Outlet P (bar)':<25} | {legacy_result['P_bar']:>15.2f} | {gas_out.pressure_pa/1e5:>15.2f} | {'✓' if abs(legacy_result['P_bar'] - gas_out.pressure_pa/1e5) < 0.01 else '✗':>8}")
print(f"{'Gas Out (kg/h)':<25} | {legacy_result['m_dot_gas_out_kg_s']*3600:>15.4f} | {gas_out.mass_flow_kg_h:>15.4f} | {'-':>8}")
print(f"{'Status':<25} | {legacy_result['Status_KOD'][:15]:>15} | {kod._separation_status:>15} | {'✓' if 'OK' in legacy_result['Status_KOD'] and kod._separation_status == 'OK' else '?':>8}")
