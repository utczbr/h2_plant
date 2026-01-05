from h2_plant.data.price_loader import EnergyPriceLoader
import os
import pandas as pd
import numpy as np

# Mocking the path
scenarios_dir = "/home/stuart/Documentos/Planta Hidrogenio/scenarios"
price_file = "simulation_output/dummy_prices.csv" # Not used if we fail before? No, load_data needs both.
# But load_data takes filename relative to scenarios_dir.
# The user config pointed to "../h2_plant/data/..."
wind_file = "../h2_plant/data/producao_horaria_2_turbinas.csv"
price_file_config = "../h2_plant/data/NL_Prices_2024_15min.csv"

print(f"Testing EnergyPriceLoader with:")
print(f"  Scenarios: {scenarios_dir}")
print(f"  Wind: {wind_file}")
print(f"  Price: {price_file_config}")

loader = EnergyPriceLoader(scenarios_dir)

try:
    # Use 1 minute resolution (approx 0.0167 hours)
    dt = 1/60.0
    prices, wind = loader.load_data(price_file_config, wind_file, duration_hours=24, timestep_hours=dt)
    
    print("\n--- Wind Data Check at Hour Boundaries ---")
    print(f"Minute 0 (00:00): {wind[0]}")
    print(f"Minute 1 (00:01): {wind[1]}")
    print(f"Minute 59 (00:59): {wind[59]}")
    print(f"Minute 60 (01:00): {wind[60]}")
    print(f"Minute 61 (01:01): {wind[61]}")
    
    print("\n--- First 20 steps ---")
    print(wind[:20])
    
    print(f"\nTotal steps: {len(wind)}")
    print(f"Stats: Min={wind.min()}, Max={wind.max()}, Mean={wind.mean()}")

    
    unique_vals = np.unique(wind)
    print(f"Unique values count: {len(unique_vals)}")
    if len(unique_vals) < 5:
        print(f"Values: {unique_vals}")

    # Inspect the internal dataframe loading logic if needed (by copy-paste debug)
    
except Exception as e:
    print(f"CRASH: {e}")
