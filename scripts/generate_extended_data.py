import pandas as pd
import numpy as np

def generate_extended_data():
    # 1. Load 8-hour data
    prices_8h = pd.read_csv('prices_8hour_test.csv')
    power_8h = pd.read_csv('power_input_8hour_test.csv')
    
    print(f"Loaded 8h prices: {len(prices_8h)} rows")
    print(f"Loaded 8h power: {len(power_8h)} rows")
    
    # 2. Repeat 21 times (8h * 21 = 168h = 7 days)
    repeats = 21
    
    prices_7d = pd.concat([prices_8h] * repeats, ignore_index=True)
    power_7d = pd.concat([power_8h] * repeats, ignore_index=True)
    
    print(f"Generated 7d prices: {len(prices_7d)} rows")
    print(f"Generated 7d power: {len(power_7d)} rows")
    
    # 3. Save
    prices_7d.to_csv('prices_7day_test.csv', index=False)
    power_7d.to_csv('power_input_7day_test.csv', index=False)
    print("Saved prices_7day_test.csv and power_input_7day_test.csv")

if __name__ == "__main__":
    generate_extended_data()
