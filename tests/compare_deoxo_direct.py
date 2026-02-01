#!/usr/bin/env python3
"""
Direct comparison of Legacy vs New Deoxo reactor implementations.

This script feeds identical inlet conditions to both implementations
and compares conversion, temperature rise, and outlet composition.
"""
import sys
sys.path.insert(0, '/home/stuart/Documentos/Planta Hidrogenio')

import numpy as np

# Import Legacy model
from h2_plant.legacy.NEW.PEM.modulos.modelo_deoxo import modelar_deoxo

# Import New component dependencies
from h2_plant.core.constants import DeoxoConstants, GasConstants
from h2_plant.optimization import numba_ops

# ==============================================================
# INLET CONDITIONS (Matching Legacy "Aquecedor Imaginário" output)
# ==============================================================
T_in_C = 40.0  # After heating to 40°C
P_in_bar = 39.70
m_dot_gas_kg_s = 0.02472
y_H2O_in = 2.04e-4  # 204 ppm
y_O2_in = 1.98e-4   # ~198 ppm

# Mass fractions (approx, assuming mostly H2)
# x_O2 ≈ y_O2 * (M_O2/M_mix) where M_mix ≈ M_H2 for high purity H2
M_H2 = 2.016e-3  # kg/mol
M_O2 = 32.0e-3
M_H2O = 18.015e-3
M_mix = M_H2  # Approximation for ~100% H2

# ==============================================================
# 1. LEGACY MODEL
# ==============================================================
print("=" * 70)
print("LEGACY MODEL (modelo_deoxo.py)")
print("=" * 70)

# Legacy uses L_M_input as a parameter
L_legacy = 0.800  # m (as shown in Legacy output)

legacy_result = modelar_deoxo(
    m_dot_g_kg_s=m_dot_gas_kg_s,
    P_in_bar=P_in_bar,
    T_in_C=T_in_C,
    y_H2O_in=y_H2O_in,
    y_O2_in=y_O2_in,
    L_M_input=L_legacy
)

print(f"Inlet: T={T_in_C}°C, P={P_in_bar} bar, y_O2={y_O2_in*1e6:.1f} ppm")
print(f"Reactor Length: {L_legacy} m")
print(f"Outlet T: {legacy_result['T_C']:.2f}°C")
print(f"Outlet y_O2: {legacy_result['y_O2_out']*1e6:.2f} ppm")
print(f"Conversion X_O2: {legacy_result['X_O2']*100:.2f}%")
print(f"T_max: {legacy_result['T_max_calc']:.2f}°C")
print(f"Delta T: {legacy_result['T_C'] - T_in_C:.2f}°C")

# ==============================================================
# 2. NEW COMPONENT (Numba JIT Solver)
# ==============================================================
print("\n" + "=" * 70)
print("NEW COMPONENT (numba_ops.solve_deoxo_pfr_step)")
print("=" * 70)

# Convert to SI units for New component
T_in_K = T_in_C + 273.15
P_in_Pa = P_in_bar * 1e5

# Calculate molar flow (New component expects mol/s)
# n_total = m_dot / M_mix
molar_flow_total = m_dot_gas_kg_s / M_mix

# Use New component constants
L_new = DeoxoConstants.L_REACTOR_M  # 1.334 m

print(f"Inlet: T={T_in_K:.2f}K, P={P_in_Pa:.0f} Pa, y_O2={y_O2_in*1e6:.1f} ppm")
print(f"Reactor Length: {L_new} m")
print(f"Molar Flow: {molar_flow_total:.4f} mol/s")

conversion, t_out, t_peak, L_prof, T_prof, X_prof = numba_ops.solve_deoxo_pfr_step(
    L_total=L_new,
    steps=50,
    T_in=T_in_K,
    P_in_pa=P_in_Pa,
    molar_flow_total=molar_flow_total,
    y_o2_in=y_O2_in,
    k0=DeoxoConstants.K0_VOL_S1,
    Ea=DeoxoConstants.EA_J_MOL,
    R=GasConstants.R_UNIVERSAL_J_PER_MOL_K,
    delta_H=DeoxoConstants.DELTA_H_RXN_J_MOL_O2,
    U_a=DeoxoConstants.U_A_W_M3_K,
    T_jacket=DeoxoConstants.T_JACKET_K,
    Area=DeoxoConstants.AREA_REACTOR_M2,
    Cp_mix=DeoxoConstants.CP_MIX_AVG_J_MOL_K,
    y_o2_target=DeoxoConstants.MAX_ALLOWED_O2_OUT_MOLE_FRAC
)

y_O2_out_new = y_O2_in * (1 - conversion)

print(f"Outlet T: {t_out - 273.15:.2f}°C")
print(f"Outlet y_O2: {y_O2_out_new*1e6:.2f} ppm")
print(f"Conversion X_O2: {conversion*100:.2f}%")
print(f"T_peak: {t_peak - 273.15:.2f}°C")
print(f"Delta T: {(t_out - T_in_K):.2f} K")

# ==============================================================
# 3. COMPARISON SUMMARY
# ==============================================================
print("\n" + "=" * 70)
print("COMPARISON SUMMARY")
print("=" * 70)
print(f"{'Metric':<20} | {'Legacy':>12} | {'New':>12} | {'Match?':>8}")
print("-" * 60)
print(f"{'Reactor Length (m)':<20} | {L_legacy:>12.3f} | {L_new:>12.3f} | {'NO' if L_legacy != L_new else 'YES':>8}")
print(f"{'Outlet T (°C)':<20} | {legacy_result['T_C']:>12.2f} | {t_out - 273.15:>12.2f} | {'-':>8}")
print(f"{'Outlet O2 (ppm)':<20} | {legacy_result['y_O2_out']*1e6:>12.2f} | {y_O2_out_new*1e6:>12.2f} | {'-':>8}")
print(f"{'Conversion (%)':<20} | {legacy_result['X_O2']*100:>12.2f} | {conversion*100:>12.2f} | {'-':>8}")
print(f"{'T_max (°C)':<20} | {legacy_result['T_max_calc']:>12.2f} | {t_peak - 273.15:>12.2f} | {'-':>8}")
