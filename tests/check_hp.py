
import pandas as pd
import glob
import os

# Find the latest simulation output directory (or default)
output_dir = 'scenarios/simulation_output'
files = glob.glob(os.path.join(output_dir, '*.csv'))
# Sort by modification time
files.sort(key=os.path.getmtime, reverse=True)

if not files:
    print("No CSV found in scenarios/simulation_output")
    # Try local root
    if os.path.exists("simulation_history.csv"):
        files = ["simulation_history.csv"]
    else:
        exit(1)

latest_file = files[0]
print(f"Reading: {latest_file}")

df = pd.read_csv(latest_file)
cols = [c for c in df.columns if 'HP_Intercooler' in c and 'heat' in c]
print(f"Found cols: {cols}")

for c in cols:
    mean_val = df[c].mean()
    print(f"{c}: {mean_val:.2f} kW")

# Also check manager load
mgr_cols = [c for c in df.columns if 'cooling_manager' in c]
for c in mgr_cols:
    print(f"{c}: {df[c].mean():.2f}")
