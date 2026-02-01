
import pandas as pd
import numpy as np

# Load Data
df = pd.read_csv('h2_plant/data/ATR_linear_regressions.csv')

# Constants
MW_O2 = 32.0   # kg/kmol
MW_CH4 = 16.04 # kg/kmol
MW_H2O = 18.015 # kg/kmol

# Water side parameters
m_dot_water_kg_h = 3600.0
m_dot_water_kg_s = m_dot_water_kg_h / 3600.0
Cp_water = 4.186 # kJ/kg.K
T_water_in = 5.0

print(f"{'F_O2 (kmol/h)':<15} | {'Syngas Mass (kg/h)':<20} | {'H05 Duty (kW)':<15} | {'Water T_out (°C)':<15}")
print("-" * 75)

def process_row(row):
    # 1. Calculate Total Syngas Mass Flow (Input/Output are balanced in mass)
    # Mass In = Biogas + Steam + Oxygen
    m_bio = row['F_bio_func'] * MW_CH4
    m_steam = row['F_steam_func'] * MW_H2O
    m_o2 = row['x'] * MW_O2
    total_mass = m_bio + m_steam + m_o2
    
    # 2. Get Duty (kW)
    # H05_Q_func is typically negative (cooling). We need abs value.
    q_kw = abs(row['H05_Q_func'])
    
    # 3. Calculate Water dT
    # Q = m * Cp * dT  => dT = Q / (m * Cp)
    dt = q_kw / (m_dot_water_kg_s * Cp_water)
    t_out = T_water_in + dt
    
    print(f"{row['x']:<15.4f} | {total_mass:<20.4f} | {q_kw:<15.4f} | {t_out:<15.2f}")
    return total_mass, q_kw, t_out

df.apply(process_row, axis=1)

# Highlight interpolation for the simulation value
target_mass = 8015.82
# We can't easily interpolate 'apply' output, but we can interpolate the dataframe columns
df['Total_Mass'] = df['F_bio_func'] * MW_CH4 + df['F_steam_func'] * MW_H2O + df['x'] * MW_O2
q_target = np.interp(target_mass, df['Total_Mass'], df['H05_Q_func'].abs())
dt_target = q_target / (m_dot_water_kg_s * Cp_water)
t_out_target = T_water_in + dt_target

print("-" * 75)
print(f"TARGET INTERPOLATION:")
print(f"{'N/A':<15} | {target_mass:<20.4f} | {q_target:<15.4f} | {t_out_target:<15.2f}")
