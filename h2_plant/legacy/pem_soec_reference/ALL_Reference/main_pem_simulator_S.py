import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve 
from numpy.polynomial import polynomial as P 
import random
import warnings
import pickle 
from scipy.interpolate import interp1d

# Ignore warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

# ==========================================================
# 1. GLOBAL FUNDAMENTAL CONSTANTS
# ==========================================================
F = 96485.33    # Faraday Constant
R = 8.314       
P_ref = 1.0e5   
z = 2           
MH2 = 2.016e-3  # Molar Mass of H2 (kg/mol)
MO2 = 31.998e-3 # Molar Mass of O2 (kg/mol) 
MH2O = 18.015e-3 # Molar Mass of H2O (kg/mol) 
LHVH2_kWh_kg = 33.33 

# --- 2. GENERAL CONFIGURATION AND PARAMETERS ---

P_nominal_sistema_kW = 5000  # 5 MW
P_nominal_sistema_W = P_nominal_sistema_kW * 1000

# Operational Variables (Nominal)
T = 333.15      
P_op = 40.0e5   

# --- 3. MECHANISTIC MODEL PARAMETERS (WHITE BOX) ---

# A. Geometry (Grouped)
N_stacks = 35           
N_cell_per_stack = 645   
A_cell = 300            
Area_Total = N_stacks * N_cell_per_stack * A_cell
j_nom = 2.91            

# B. Degradation and Simulation (Grouped)
k_deg = 14.0e-6         
H_MES = 730.0           
POLYNOMIAL_ORDER = 5    
H_SIM_YEARS = 4.5       
H_SIM_TOTAL_PRECALC = int(H_SIM_YEARS * 8760) 
V_CUTOFF = 2.2          # End-of-Life Cutoff Voltage (V)

# C. Stack Physics 
delta_mem = 100 * 1e-4  
sigma_base = 0.1        
j0 = 1.0e-6             
alpha = 0.5             
j_lim = 4.0             

# D. BoP and Limits
floss = 0.02
P_bop_fixo = 0.025 * P_nominal_sistema_W
k_bop_var = 0.04
P_min_op_W = 0.05 * P_nominal_sistema_W


# --- E. DEGRADATION PARAMETERS (BASED ON SPREADSHEET)
DEGRADATION_TABLE_YEARS = np.array([1.0, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 9.0, 10.0])
DEGRADATION_TABLE_V_STACK = np.array([1290, 1300, 1325, 1340, 1345, 1355, 1365, 1380, 1390, 1410, 1435, 1460, 1490])
T_OP_H_TABLE = DEGRADATION_TABLE_YEARS * 8760.0 
V_CELL_TABLE = DEGRADATION_TABLE_V_STACK / N_cell_per_stack


# --- 4. MECHANISTIC MODEL FUNCTIONS ---

def calculate_Urev(T, P_op):
    """ Reversible Voltage (corrected for T and P). """
    U_rev_T = 1.229 - 0.9e-3 * (T - 298.15)
    pressure_ratio = P_op / P_ref
    Nernst_correction = (R * T) / (z * F) * np.log(pressure_ratio**1.5)
    return U_rev_T + Nernst_correction

def calculate_Vcell_base(j, T, P_op):
    """Calculates Vcell without the degradation term (used for BOL reference)."""
    U_rev = calculate_Urev(T, P_op)
    eta_act = (R * T) / (alpha * z * F) * np.log(np.maximum(j, 1e-10) / j0)
    eta_ohm = j * (delta_mem / sigma_base)
    eta_conc = np.where(j >= j_lim, 100.0, (R * T) / (z * F) * np.log(j_lim / (j_lim - np.maximum(j, 1e-10))))
    return U_rev + eta_act + eta_ohm + eta_conc

V_CELL_BOL_NOM = calculate_Vcell_base(j_nom, T, P_op)

v_cell_degraded_interpolator = interp1d(
    T_OP_H_TABLE, 
    V_CELL_TABLE, 
    kind='linear', 
    fill_value=(V_CELL_TABLE[0], V_CELL_TABLE[-1]), 
    bounds_error=False
)

def calculate_U_deg_from_table(t_op_h):
    """ Accumulate Degradation Voltage (U_deg) based on spreadsheet interpolation. """
    V_cell_degraded = v_cell_degraded_interpolator(t_op_h)
    U_deg = np.maximum(0.0, V_cell_degraded - V_CELL_BOL_NOM) 
    return U_deg

def calculate_Vcell(j, T, P_op, t_op_h):
    """ Cell Voltage (White Box) - NOW USES INTERPOLATED DEGRADATION. """
    U_rev = calculate_Urev(T, P_op)
    eta_act = (R * T) / (alpha * z * F) * np.log(np.maximum(j, 1e-10) / j0)
    eta_ohm = j * (delta_mem / sigma_base)
    eta_conc = np.where(j >= j_lim, 100.0, (R * T) / (z * F) * np.log(j_lim / (j_lim - np.maximum(j, 1e-10))))
    U_deg = calculate_U_deg_from_table(t_op_h) 
    return U_rev + eta_act + eta_ohm + eta_conc + U_deg

def P_input_system(j, T, P_op, t_op_h):
    """ System Input Power (W). """
    I_total = j * Area_Total
    V_cell = calculate_Vcell(j, T, P_op, t_op_h)
    P_stack = I_total * V_cell
    P_BoP = P_bop_fixo + k_bop_var * P_stack
    return P_stack + P_BoP

def calculate_eta_F(j):
    """ Faraday Efficiency. """
    return np.maximum(j, 1e-6)**2 / (np.maximum(j, 1e-6)**2 + floss)

def calculate_H2_flow(I, eta_F):
    """ Mass Flow Rate of Hydrogen (kg/s). """
    return (I * eta_F * MH2) / (z * F)

# --- NOVAS FUNÇÕES DE CÁLCULO DE BALANÇO DE MASSA E ENERGIA ---

def calculate_O2_flow(I, eta_F):
    """ Mass Flow Rate of Oxygen (kg/s). """
    return (I * eta_F * (MO2 / 2.0)) / (z * F)

def calculate_H2O_demand(I, eta_F):
    """ Mass Flow Rate of Water consumed (kg/s). """
    return (I * eta_F * MH2O) / (z * F)

def calculate_energy_consumption(P_consumed_W, delta_t_h):
    """ Energy consumed in Wh (Watt-hours). """
    return P_consumed_W * delta_t_h

def calculate_kpis(P_target_kW, T, P_op, t_op_h):
    """ Solves for j_op using the Solver and calculates KPIs. """
    P_target_W = P_target_kW * 1000
    
    def equation_to_solve(j):
        P_calculated = P_input_system(j, T, P_op, t_op_h)
        return P_calculated - P_target_W

    j_initial_guess = 0.5 * j_nom
    
    j_solution, infodict, ier, msg = fsolve(
        equation_to_solve, 
        j_initial_guess, 
        full_output=True,
        maxfev=100
    )
    
    if ier == 1:
        j_op = np.clip(j_solution[0], 0.0, j_lim)
        I_total = j_op * Area_Total
        V_cell = calculate_Vcell(j_op, T, P_op, t_op_h)
        P_consumida_W = P_input_system(j_op, T, P_op, t_op_h)
        
        eta_F = calculate_eta_F(j_op)
        m_H2_dot = calculate_H2_flow(I_total, eta_F)
        P_chem_W = m_H2_dot * (LHVH2_kWh_kg * 3.6e6)
        
        eta_sys = P_chem_W / P_consumida_W
        SEC = LHVH2_kWh_kg / eta_sys
        
        return V_cell, eta_sys * 100, SEC
    return 0.0, 0.0, 0.0


# --- 5. LOAD PRE-CALCULATED DATA AND GENERATE PLOT ARRAYS ---
file_name = "degradation_polynomials.pkl"
try:
    with open(file_name, 'rb') as f:
        polynomial_list = pickle.load(f)
    print(f"✅ Polynomials loaded successfully: {len(polynomial_list)} tables.")
    print(f"⚠️ Ensure polynomials were generated by 'pre_calculator_pem_S.py'.")
except FileNotFoundError:
    print(f"❌ ERROR: File '{file_name}' not found. Run 'pre_calculator_pem_S.py' first.")
    exit()


# --- STATIC CALCULATIONS FOR PLOTTING ---

j_char_points = 200
j_char_range = np.linspace(0.001 * j_nom, 1.25 * j_nom, j_char_points)
I_total_char = j_char_range * Area_Total

# BOL Data (t=0)
P_input_char_W_BOL = np.array([P_input_system(j, T, P_op, 0.0) for j in j_char_range])
m_H2_dot_char_BOL = calculate_H2_flow(I_total_char, calculate_eta_F(j_char_range))

# Defining the Real EOL using fsolve for the new degradation model:
V_BOL_nominal = calculate_Vcell(j_nom, T, P_op, 0.0) 

def equation_for_T_EOL(t_h):
    """ Finds the time (h) where Vcell(j_nom, t_h) equals V_CUTOFF. """
    return calculate_Vcell(j_nom, T, P_op, t_h) - V_CUTOFF

# FIX: Initialize H_EOL_REAL here to prevent NameError
H_EOL_REAL = T_OP_H_TABLE[-1] 

try:
    t_eol_solution, infodict, ier, msg = fsolve(
        equation_for_T_EOL, 
        T_OP_H_TABLE[-1],
        full_output=True,
        maxfev=100
    )
    if ier == 1 and t_eol_solution[0] > 0 and t_eol_solution[0] < T_OP_H_TABLE[-1] * 2:
        H_EOL_REAL = t_eol_solution[0]
except Exception as e:
    pass 

H_EOL = H_EOL_REAL 
P_input_char_W_EOL = np.array([P_input_system(j, T, P_op, H_EOL) for j in j_char_range])
m_H2_dot_char_EOL = m_H2_dot_char_BOL 
V_cell_BOL = np.array([calculate_Vcell(j, T, P_op, 0.0) for j in j_char_range])
V_cell_EOL = np.array([calculate_Vcell(j, T, P_op, H_EOL) for j in j_char_range])


# --- 6. TIMESTEP REPORT (BOL vs EOL) ---

P_ref_kW = 0.95 * P_nominal_sistema_kW

V_bol, eta_bol, sec_bol = calculate_kpis(P_ref_kW, T, P_op, t_op_h=0)
V_eol, eta_eol, sec_eol = calculate_kpis(P_ref_kW, T, P_op, t_op_h=H_EOL)

print("="*60)
print(f"TIMESTEP REPORT (BOL vs EOL - {H_EOL/8760:.2f} YEARS)")
print(f"Reference Point: {P_ref_kW:.0f} kW (95% P_nom)")
print("-"*60)
print(f"{'Metric':<15} | {'Start (BOL)':<15} | {'End (EOL)':<15} | {'Degradation':<15}")
print("-"*60)
print(f"{'Voltage (V)':<15} | {V_bol:.4f} V{'':<11} | {V_eol:.4f} V{'':<11} | {V_eol-V_bol:.4f} V")
print(f"{'Efficiency (%)':<15} | {eta_bol:.2f} %{'':<11} | {eta_eol:.2f} %{'':<11} | {(eta_eol-eta_bol):.2f} pts")
print(f"{'SEC (kWh/kg)':<15} | {sec_bol:.2f}{'':<11} | {sec_eol:.2f}{'':<11} | {sec_eol-sec_bol:.2f}")
print("="*60)
print(f"WARNING: EOL is reached in {H_EOL/8760:.2f} years ({H_EOL:.0f} hours).")


# --- 7. CHARACTERISTIC CURVES VISUALIZATION (PLOTS 1-4 CONSOLIDATED) ---

# CALCULATIONS FOR CHARACTERIZATION PLOTS
load_factor_char_BOL = P_input_char_W_BOL / P_nominal_sistema_W
eta_sys_char_BOL = calculate_H2_flow(I_total_char, calculate_eta_F(j_char_range)) * (LHVH2_kWh_kg * 3.6e6) / P_input_char_W_BOL * 100

# Plot 1: Efficiency vs. Load Factor
plt.figure(figsize=(10, 6))
plt.plot(load_factor_char_BOL * 100, eta_sys_char_BOL, label='BOL Efficiency', color='blue')
plt.title('1. Efficiency Curve (BOL)')
plt.xlabel('Load Factor (% of Nominal Power)')
plt.ylabel('System Efficiency (% LHV)')
plt.grid(True)
plt.legend()
plt.show()

# Plot 2: SEC vs. Load Factor 
plt.figure(figsize=(10, 6))
SEC_char_BOL = LHVH2_kWh_kg / (eta_sys_char_BOL / 100)
plt.plot(load_factor_char_BOL * 100, SEC_char_BOL, label='SEC BOL', color='green')
plt.title('2. Specific Energy Consumption (SEC) (BOL)')
plt.xlabel('Load Factor (% of Nominal Power)')
plt.ylabel('SEC (kWh/kg H₂)')
plt.grid(True)
plt.legend()
plt.show()

# Plot 3: Polarization Curve (V-j) Comparative 
plt.figure(figsize=(10, 6))
plt.plot(j_char_range, V_cell_BOL, linewidth=3, color='blue', label='BOL (Year 0)')
plt.plot(j_char_range, V_cell_EOL, linewidth=3, linestyle='--', color='orange', label=f'EOL (Year {H_EOL/8760:.2f})') 
plt.title('3. Polarization Curve (V-j): Voltage Increase due to Degradation')
plt.xlabel('Current Density (A/cm²)')
plt.ylabel('Cell Voltage (V)')
plt.grid(True)
plt.legend()
plt.show()

# Plot 4: Dispatch Curve (Power vs. Production) Comparative
plt.figure(figsize=(10, 6))
plt.plot(P_input_char_W_BOL / 1000, m_H2_dot_char_BOL, linewidth=3, color='purple', label=f'BOL (Year 0)')
plt.plot(P_input_char_W_EOL / 1000, m_H2_dot_char_EOL, linewidth=3, linestyle='--', color='red', label=f'EOL (Year {H_EOL/8760:.2f})')
plt.title('4. Dispatch Curve: Impact of Degradation (BOL vs. EOL)')
plt.xlabel('System Input Power (kW)')
plt.ylabel('Hydrogen Production (kg/s)')
plt.axvline(x=P_nominal_sistema_kW, color='gray', linestyle=':', label='Nominal Power (5 MW)')
plt.grid(True)
plt.legend()
plt.show()


# Plot 5: PEM State of Health (SoH) - (PLOT RANGE EXTENDED)
V_BOL = V_cell_BOL[j_char_range.searchsorted(j_nom)] 
V_CUTOFF_REF = V_CUTOFF
T_EOL_H = H_EOL 

MAX_PLOT_YEARS = max(10.0, H_EOL_REAL/8760.0 * 1.05) 
t_vector_h = np.linspace(0, MAX_PLOT_YEARS * 8760.0, 100) 
t_vector_years = t_vector_h / 8760.0

V_t = np.array([calculate_Vcell(j_nom, T, P_op, t) for t in t_vector_h]) 

V_drift_total = V_CUTOFF_REF - V_BOL
SOH_percent = 100 * (V_CUTOFF_REF - V_t) / V_drift_total
SOH_percent = np.clip(SOH_percent, 0, 100) 

plt.figure(figsize=(10, 6))
plt.plot(t_vector_years, SOH_percent, linewidth=3, color='green', label=f'Estimated Life: {T_EOL_H/8760:.2f} Years')

plt.title('5. PEM State of Health (SoH) vs. Time (Table Degradation)')
plt.xlabel(f'Operating Time (Years)')
plt.ylabel('State of Health (%)')
plt.grid(True)
plt.legend()
plt.ylim(-10, 110)
plt.show()


# --- 8. 1-MONTH SIMULATION (MINUTE-BY-MINUTE) AND FINAL PLOTTING ---

TIME_SIMULATION_DAYS = 30 
TIME_SIMULATION_MIN = TIME_SIMULATION_DAYS * 24 * 60 
DELTA_T_H = 1.0 / 60.0 

t_op_h = 0.0 
P_available_profile_W = np.zeros(TIME_SIMULATION_MIN)
random.seed(44) 

for i in range(0, TIME_SIMULATION_MIN, 60): 
    P_random_kW = random.uniform(0.1, P_nominal_sistema_kW)
    P_random_W = P_random_kW * 1000
    P_available_profile_W[i:i+60] = P_random_W

m_H2_dot_kg_s_t = np.zeros(TIME_SIMULATION_MIN)
m_O2_dot_kg_s_t = np.zeros(TIME_SIMULATION_MIN)      
m_H2O_dot_kg_s_t = np.zeros(TIME_SIMULATION_MIN)     
P_consumida_W_t = np.zeros(TIME_SIMULATION_MIN)      
E_consumida_Wh_t = np.zeros(TIME_SIMULATION_MIN)     
eta_sys_percent_t = np.zeros(TIME_SIMULATION_MIN)
time_vector_min = np.arange(TIME_SIMULATION_MIN)

print(f"\nStarting Minute-by-Minute Simulation ({TIME_SIMULATION_MIN} minutes) - 1 MONTH...")

for t in range(TIME_SIMULATION_MIN):
    P_disponivel_W = P_available_profile_W[t]
    
    month_index = min(int(t_op_h / H_MES), len(polynomial_list) - 1)
    j_finder_poly = polynomial_list[month_index]
    
    if P_disponivel_W < P_min_op_W:
        m_H2_dot = 0.0
        m_O2_dot = 0.0
        m_H2O_dot = 0.0
        P_consumida_W = 0.0
        eta_sys = 0.0
    else:
        P_target_W = np.clip(P_disponivel_W, P_min_op_W, P_nominal_sistema_W) 
        j_op = j_finder_poly(P_target_W)
        j_op = np.clip(j_op, 0.001, j_lim) 
        
        V_cell_sim = calculate_Vcell(j_op, T, P_op, t_op_h) 
        I_total = j_op * Area_Total 

        if t_op_h > H_EOL: 
            m_H2_dot = 0.0
            m_O2_dot = 0.0
            m_H2O_dot = 0.0
            P_consumida_W = 0.0
            eta_sys = 0.0
        else:
            P_stack_W = I_total * V_cell_sim
            P_BoP_W = P_bop_fixo + k_bop_var * P_stack_W
            P_consumida_W = P_stack_W + P_BoP_W 
            
            eta_F = calculate_eta_F(j_op)
            
            m_H2_dot = calculate_H2_flow(I_total, eta_F)
            m_O2_dot = calculate_O2_flow(I_total, eta_F)
            m_H2O_dot = calculate_H2O_demand(I_total, eta_F)
            
            P_chem_W = m_H2_dot * (LHVH2_kWh_kg * 3.6e6)
            eta_sys = P_chem_W / P_consumida_W if P_consumida_W > 0 else 0.0
            
            t_op_h += DELTA_T_H 
        
    m_H2_dot_kg_s_t[t] = m_H2_dot
    m_O2_dot_kg_s_t[t] = m_O2_dot
    m_H2O_dot_kg_s_t[t] = m_H2O_dot
    P_consumida_W_t[t] = P_consumida_W
    E_consumida_Wh_t[t] = calculate_energy_consumption(P_consumida_W, DELTA_T_H) 
    eta_sys_percent_t[t] = eta_sys * 100

print("Minute-by-Minute Simulation concluded.")


# --- 9. NOVAS VISUALIZAÇÕES DE BALANÇO DE MASSA E ENERGIA ---

# Plot 9: Produção de Oxigênio Instantânea (1 Mês)
plt.figure(figsize=(12, 6))
plt.plot(time_vector_min, m_O2_dot_kg_s_t, color='blue', label='O₂ Production (kg/s)')
plt.title('9. Instantaneous Oxygen Production over 1 Month (1-Minute Step)')
plt.xlabel('Time (Minutes)')
plt.ylabel('O₂ Production (kg/s)')
plt.grid(True)
plt.legend()
plt.show()

# Plot 10: Demanda de Água Instantânea (1 Mês)
plt.figure(figsize=(12, 6))
plt.plot(time_vector_min, m_H2O_dot_kg_s_t, color='brown', label='H₂O Demand (kg/s)')
plt.title('10. Instantaneous Water Demand over 1 Month (1-Minute Step)')
plt.xlabel('Time (Minutes)')
plt.ylabel('H₂O Demand (kg/s)')
plt.grid(True)
plt.legend()
plt.show()

# Plot 11: Potência Consumida (1 Mês)
plt.figure(figsize=(12, 6))
plt.plot(time_vector_min, P_consumida_W_t / 1000, color='red', label='Power Consumption (kW)')
plt.title('11. Instantaneous Power Consumption over 1 Month (1-Minute Step)')
plt.xlabel('Time (Minutes)')
plt.ylabel('Power Consumption (kW)')
plt.grid(True)
plt.legend()
plt.show()


# Plot 12: Produção/Consumo Acumulados
m_O2_acumulada_kg = np.cumsum(m_O2_dot_kg_s_t * 60.0)
m_H2O_acumulada_kg = np.cumsum(m_H2O_dot_kg_s_t * 60.0)
E_consumida_acumulada_kWh = np.cumsum(E_consumida_Wh_t) / 1000 

plt.figure(figsize=(12, 6))
plt.plot(time_vector_min, m_H2_acumulada_kg, color='darkgreen', linewidth=2, label='H₂ Accumulated (kg)')
plt.plot(time_vector_min, m_O2_acumulada_kg, color='blue', linewidth=2, linestyle='--', label='O₂ Accumulated (kg)')
plt.plot(time_vector_min, m_H2O_acumulada_kg, color='brown', linewidth=2, linestyle=':', label='H₂O Consumed (kg)')
plt.plot(time_vector_min, E_consumida_acumulada_kWh, color='red', linewidth=2, label='Energy Consumed (kWh)')
plt.title('12. Accumulated Production and Consumption (1 Month)')
plt.xlabel('Time (Minutes)')
plt.ylabel('Accumulated Quantity (kg or kWh)')
plt.grid(True)
plt.legend()
plt.show()


# --- PLOTAGENS ORIGINAIS (6, 7, 8) ---
# Plotagem 6: Produção de Hidrogênio ao longo da Simulação (1 Mês)
plt.figure(figsize=(12, 6))
plt.plot(time_vector_min, m_H2_dot_kg_s_t, color='darkgreen', label='H₂ Production (kg/s)')
plt.title('6. Instantaneous Hydrogen Production over 1 Month (1-Minute Step)')
plt.xlabel('Time (Minutes)')
plt.ylabel('H₂ Production (kg/s)')
plt.grid(True)
plt.legend()
plt.show()

# Plotagem 7: Eficiência do Sistema ao longo da Operação (1 Mês)
plt.figure(figsize=(12, 6))
plt.plot(time_vector_min, eta_sys_percent_t, color='darkred', label='System Efficiency (% LHV)')
plt.title('7. System Efficiency over 1 Month (1-Minute Step)')
plt.xlabel('Time (Minutes)')
plt.ylabel('System Efficiency (% LHV)')
plt.grid(True)
plt.legend()
plt.ylim(ymin=0)
plt.show()

# Plotagem 8: Produção Acumulada de Hidrogênio
m_H2_acumulada_kg = np.cumsum(m_H2_dot_kg_s_t * 60.0)
plt.figure(figsize=(12, 6))
plt.plot(time_vector_min, m_H2_acumulada_kg, color='blue', linewidth=2, label='Accumulated Production')
plt.title('8. Accumulated Hydrogen Production (1 Month)')
plt.xlabel('Time (Minutes)')
plt.ylabel('Accumulated H₂ Mass (kg)')
plt.grid(True)
plt.legend()
plt.show()