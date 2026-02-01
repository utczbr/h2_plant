
import pandas as pd
from pathlib import Path
import argparse
import sys
import gc
import numpy as np

def analyze_prices(chunks_dir_path):
    chunks_dir = Path(chunks_dir_path)
    chunk_files = sorted(chunks_dir.glob("chunk_*.parquet"))
    
    # Fallback to subdirectory search
    if not chunk_files:
        subdir = chunks_dir / "history_chunks"
        if subdir.exists():
             chunk_files = sorted(subdir.glob("chunk_*.parquet"))
             
    if not chunk_files:
        print(f"No chunk files found in {chunks_dir}.")
        return

    print(f"Analyzing {len(chunk_files)} chunks for price conditions...")
    
    total_steps = 0
    purchase_opportunities = 0
    
    min_price = float('inf')
    max_price = float('-inf')
    avg_price_accum = 0.0
    
    threshold_accum = 0.0
    
    for chunk_file in chunk_files:
        try:
            # Load only price columns
            # We need spot_price and spot_threshold_eur_mwh
            # Using 'spot_threshold_eur_mwh' based on previous inspection
            cols = ['spot_price', 'spot_threshold_eur_mwh']
            
            # Check validation first (peek simplified)
            # Efficiently we just try to read.
            df = pd.read_parquet(chunk_file, columns=cols)
            
            # Basic stats
            steps = len(df)
            total_steps += steps
            
            prices = df['spot_price'].to_numpy()
            thresholds = df['spot_threshold_eur_mwh'].to_numpy()
            
            # Update global stats
            chunk_min = np.min(prices)
            chunk_max = np.max(prices)
            if chunk_min < min_price: min_price = chunk_min
            if chunk_max > max_price: max_price = chunk_max
            
            avg_price_accum += np.sum(prices)
            threshold_accum += np.sum(thresholds)
            
            # Count opportunities (Price < Threshold)
            # Assuming strictly less than, as usually equality doesn't trigger change or is edge case
            opportunities = np.sum(prices < thresholds)
            purchase_opportunities += opportunities
            
            # Explicit GC
            del df
            del prices
            del thresholds
            gc.collect()
            
        except Exception as e:
            print(f"Error reading {chunk_file.name}: {e}")
            # If column missing, maybe try 'purchase_threshold_eur_mwh'?
            # But let's stick to error reporting for now.

    if total_steps == 0:
        print("No data processed.")
        return

    avg_price = avg_price_accum / total_steps
    avg_threshold = threshold_accum / total_steps
    opportunity_pct = (purchase_opportunities / total_steps) * 100.0

    print("\n--- PRICE ANALYSIS RESULTS ---")
    print(f"Total Hours Analyzed:   {total_steps}")
    print(f"Purchase Opportunities: {purchase_opportunities} (Price < Threshold)")
    print(f"Opportunity Frequency:  {opportunity_pct:.2f}%")
    print("-" * 30)
    print(f"Spot Price (EUR/MWh):")
    print(f"  Min: {min_price:.2f}")
    print(f"  Max: {max_price:.2f}")
    print(f"  Avg: {avg_price:.2f}")
    print("-" * 30)
    print(f"Purchase Threshold (Avg): {avg_threshold:.2f} EUR/MWh")
    
    if purchase_opportunities == 0:
        print("\nCONCLUSION: The system NEVER detected a spot price lower than the threshold.")
        print("            This explains why Non-RFNBO H2 production is 0.00 kg.")
    else:
        print(f"\nCONCLUSION: The system detected {purchase_opportunities} hours where purchase provided an advantage.")
        print("            If Non-RFNBO H2 is still zero, check if 'safe mode' or capacity limits prevented purchasing.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", help="Directory containing history chunks")
    args = parser.parse_args()
    analyze_prices(args.dir)
