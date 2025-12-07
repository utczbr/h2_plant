"""
Debug: Identify source of 2% deviation between reference and new component
"""
import math
import sys
sys.path.insert(0, '/home/stuart/Documentos/Planta Hidrogenio')

from h2_plant.components.separation.coalescer import Coalescer
from h2_plant.core.stream import Stream
from h2_plant.core.constants import CoalescerConstants, ConversionFactors

# Reference constants
BAR_TO_PA = 1e5
M_H2 = 0.002016
M_H2O = 0.018015
R_J_K_mol = 8.31446
CELSIUS_TO_KELVIN = 273.15
P_NOMINAL_BAR = 39.70
T_NOMINAL_C = 4.00
VARIACAO_PERCENTUAL = 0.10
VISCOSIDADE_H2_REF = 9.0e-6
T_REF_K = 303.15
K_PERDA_EMPIRICO = 0.5e6

# Conditions
P_min_bar = P_NOMINAL_BAR * (1 - VARIACAO_PERCENTUAL)  # 35.73 bar
T_nominal_K = T_NOMINAL_C + CELSIUS_TO_KELVIN
T_max_op_K = T_nominal_K * (1 + VARIACAO_PERCENTUAL)  # ~304.87 K
P_total_Pa = P_min_bar * BAR_TO_PA
Q_M = 80.46  # kg/h

print("=" * 70)
print("DEBUG: Step-by-step comparison")
print("=" * 70)

# =========================================================================
# REFERENCE MODEL CALCULATION
# =========================================================================
print("\n--- REFERENCE MODEL ---")

# Saturation pressure and mixture
P_sat_bar = 0.0469
y_H2O = P_sat_bar / P_min_bar
y_gas = 1.0 - y_H2O
M_avg_ref = (y_gas * M_H2) + (y_H2O * M_H2O)
rho_mix_ref = (P_total_Pa * M_avg_ref) / (R_J_K_mol * T_max_op_K)

print(f"y_H2O = {y_H2O:.6f}")
print(f"M_avg = {M_avg_ref:.6f} kg/mol")
print(f"rho_mix = {rho_mix_ref:.6f} kg/m³")

# Viscosity
mu_g_ref = VISCOSIDADE_H2_REF * (T_max_op_K / T_REF_K) ** 0.7
print(f"mu_g = {mu_g_ref:.6e} Pa·s")

# Flow
Q_V_ref = Q_M / rho_mix_ref / 3600  # m³/s
A_shell = (math.pi / 4) * (0.32 ** 2)
U_sup_ref = Q_V_ref / A_shell
print(f"Q_V = {Q_V_ref:.6e} m³/s")
print(f"A_shell = {A_shell:.6f} m²")
print(f"U_sup = {U_sup_ref:.6e} m/s")

# Pressure drop
Delta_P_Pa_ref = K_PERDA_EMPIRICO * mu_g_ref * 1.0 * U_sup_ref
Delta_P_bar_ref = Delta_P_Pa_ref * 1e-5
print(f"Delta_P = {Delta_P_bar_ref:.6e} bar")
print(f"Power = {Q_V_ref * Delta_P_Pa_ref:.6e} W")

# =========================================================================
# NEW COMPONENT CALCULATION
# =========================================================================
print("\n--- NEW COMPONENT (via Stream) ---")

# Create Stream with pure H2 (no H2O_liq for density comparison)
inlet = Stream(
    mass_flow_kg_h=Q_M,
    temperature_k=T_max_op_K,
    pressure_pa=P_total_Pa,
    composition={'H2': 1.0}
)

print(f"Stream.density_kg_m3 = {inlet.density_kg_m3:.6f} kg/m³")
print(f"Expected (ref) rho   = {rho_mix_ref:.6f} kg/m³")

# The difference!
rho_diff = (inlet.density_kg_m3 - rho_mix_ref) / rho_mix_ref * 100
print(f"Density difference: {rho_diff:.2f}%")

# Now compute using Stream density
Q_V_new = Q_M / inlet.density_kg_m3 / 3600
U_sup_new = Q_V_new / A_shell
mu_g_new = CoalescerConstants.MU_REF_H2_PA_S * (T_max_op_K / CoalescerConstants.T_REF_K) ** 0.7

print(f"Q_V (new) = {Q_V_new:.6e} m³/s")
print(f"U_sup (new) = {U_sup_new:.6e} m/s")
print(f"mu_g (new) = {mu_g_new:.6e} Pa·s")

Delta_P_Pa_new = K_PERDA_EMPIRICO * mu_g_new * 1.0 * U_sup_new
Delta_P_bar_new = Delta_P_Pa_new * 1e-5
print(f"Delta_P (new) = {Delta_P_bar_new:.6e} bar")

# =========================================================================
# ROOT CAUSE
# =========================================================================
print("\n--- ROOT CAUSE ---")
print(f"Reference uses mixture density (H2 + H2O vapor): {rho_mix_ref:.6f} kg/m³")
print(f"Stream uses pure H2 density: {inlet.density_kg_m3:.6f} kg/m³")

# What if we match the reference density exactly?
print("\n--- IF WE USE REFERENCE DENSITY ---")
Q_V_fixed = Q_M / rho_mix_ref / 3600
U_sup_fixed = Q_V_fixed / A_shell
Delta_P_fixed = K_PERDA_EMPIRICO * mu_g_new * 1.0 * U_sup_fixed * 1e-5
ref_diff = abs(Delta_P_fixed - Delta_P_bar_ref) / Delta_P_bar_ref * 100
print(f"Delta_P would be: {Delta_P_fixed:.6e} bar")
print(f"Difference: {ref_diff:.4f}%")
