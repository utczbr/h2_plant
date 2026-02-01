
import pandas as pd
import json
import os
import glob

def analyze_boiler_power():
    # Directly look in simulation_output folder
    latest_dir = "scenarios/simulation_output"
    print(f"Analyzing results from: {latest_dir}")

    # Load history file (parquet or csv)
    history_file = os.path.join(latest_dir, "simulation_history.parquet")
    if not os.path.exists(history_file):
        history_file = os.path.join(latest_dir, "simulation_history.csv")
        
    if not os.path.exists(history_file):
        print("No history file found.")
        return

    if history_file.endswith(".parquet"):
        df = pd.read_parquet(history_file)
    else:
        df = pd.read_csv(history_file)

    # Filter for electric boiler power columns
    # Assuming columns are named like "{component_id}_power_kw" or similar
    # We'll look for columns containing "Boiler" and "power"
    
    boiler_cols = [c for c in df.columns if "Boiler" in c and "power_input_kw" in c]
    
    if not boiler_cols:
        # Fallback: try to find boiler IDs from topology and look for their power
        print("No explicit boiler power columns found. Checking for component states...")
        # (This part depends on how history is structured, simple column search is best first step)
        pass

    print("\n--- Electric Boiler Peak Power Analysis ---")
    print(f"{'Component ID':<30} | {'Peak Power (kW)':<15} | {'Mean Power (kW)':<15}")
    print("-" * 65)
    
    results = {}
    
    for col in boiler_cols:
        component_id = col.replace("_power_kw", "")
        peak_power = df[col].max()
        mean_power = df[col].mean()
        
        print(f"{component_id:<30} | {peak_power:<15.2f} | {mean_power:<15.2f}")
        results[component_id] = peak_power

    # Identify main boiler vs pre-heaters
    print("\n--- Classification ---")
    for comp, power in results.items():
        if power > 1000:
            print(f"{comp}: Main Boiler (Capacity > 1 MW)")
        else:
            print(f"{comp}: Small Heater / Pre-heater")

if __name__ == "__main__":
    analyze_boiler_power()
