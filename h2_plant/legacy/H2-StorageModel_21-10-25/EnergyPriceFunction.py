import pandas as pd

def load_energy_price_data(
    filename="EnergyPriceAverage2023-24.xlsx",

    # Thresholds for low and high price quantities, represents the amount of data points above and below a certain percentage
    verylow_to_low = 0.05,
    low_to_medium_threshold=0.33,
    medium_to_high_threshold=0.66,
    veryhigh_to_high = 0.95,
    
):
    # Load Excel file and clean up column names
    df = pd.read_excel(filename)
    df.columns = df.columns.str.strip()

    # Convert to €/kWh
    df["Price_kWh"] = df["Price"] / 1000

    
    # Extract thresholds
    low_to_medium_threshold = df["Price_kWh"].quantile(low_to_medium_threshold)
    medium_to_high_threshold = df["Price_kWh"].quantile(medium_to_high_threshold)

    # Return the price series (as NumPy array), and thresholds
    return df["Price_kWh"].to_numpy(), low_to_medium_threshold, medium_to_high_threshold