
import pandas as pd
import argparse
import sys

def analyze_raw_prices(file_path):
    print(f"Analyzing {file_path}...")
    try:
        df = pd.read_csv(file_path)
        if 'price_eur_mwh' not in df.columns:
            print(f"Error: Column 'price_eur_mwh' not found. Columns: {df.columns.tolist()}")
            # Attempt to find a likely column
            for col in df.columns:
                if 'price' in col.lower():
                    print(f"Using alternative column: {col}")
                    df.rename(columns={col: 'price_eur_mwh'}, inplace=True)
                    break
            else:
                return

        prices = df['price_eur_mwh']
        total_points = len(prices)
        
        min_p = prices.min()
        max_p = prices.max()
        avg_p = prices.mean()
        
        neg_count = (prices <= 0).sum()
        below_40 = (prices < 40).sum()
        below_50 = (prices < 50).sum()
        below_80 = (prices < 80).sum()
        
        print("\n--- RAW PRICE FILE ANALYSIS ---")
        print(f"Total Data Points: {total_points}")
        print("-" * 30)
        print(f"Price Statistics (EUR/MWh):")
        print(f"  Min: {min_p:.2f}")
        print(f"  Max: {max_p:.2f}")
        print(f"  Avg: {avg_p:.2f}")
        print("-" * 30)
        print("Potential Purchase Opportunities (Frequency):")
        print(f"  Prices <= 0.00: {neg_count} ({neg_count/total_points*100:.2f}%)")
        print(f"  Prices < 40.00: {below_40} ({below_40/total_points*100:.2f}%)")
        print(f"  Prices < 50.00: {below_50} ({below_50/total_points*100:.2f}%)")
        print(f"  Prices < 80.00: {below_80} ({below_80/total_points*100:.2f}%)")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    analyze_raw_prices("h2_plant/data/NL_Prices_2024_15min.csv")
