
import pandas as pd
import glob
import os

output_dir = 'scenarios/simulation_output'
files = glob.glob(os.path.join(output_dir, '*.csv'))
files.sort(key=os.path.getmtime, reverse=True)
latest_file = files[0]
print(f"Reading: {latest_file}")

df = pd.read_csv(latest_file)

# Columns ending in _heat_rejected_kw or _latent_heat_kw
# EXCLUDING cooling_manager itself
comp_cols = [c for c in df.columns if ('heat_rejected_kw' in c or 'latent_heat_kw' in c) 
             and 'cooling_manager' not in c 
             and 'Chiller' not in c] # Chillers are separate? Usually Chiller rejects to water/air?
             # Check if Chiller is DryCooler type? No, Type=Chiller.
             # Chiller rejects Q_condenser.
             # If Chiller is air-cooled, it rejects.
             # If water-cooled, it rejects to cooling water (Manager).
             # Let's assume Chillers are electric.
             # Only sum DryCoolers/Intercoolers which are the issue.

print(f"Summing {len(comp_cols)} component columns...")
total_rejection = df[comp_cols].sum(axis=1).mean()
print(f"Total Component Rejection (Mean): {total_rejection:.2f} kW")

manager_duty = df['cooling_manager_glycol_duty_kw'].mean()
print(f"Manager Glycol Duty (Mean): {manager_duty:.2f} kW")

diff = total_rejection - manager_duty
print(f"Difference: {diff:.2f} kW")

# List non-zero contributors
for c in comp_cols:
    val = df[c].mean()
    if val > 1.0:
        print(f"  {c}: {val:.2f}")
