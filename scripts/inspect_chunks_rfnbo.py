
import pandas as pd
from pathlib import Path
import sys
import argparse
import gc

def inspect_chunks(chunks_dir_path):
    chunks_dir = Path(chunks_dir_path)
    if not chunks_dir.exists():
        print(f"Error: {chunks_dir} does not exist.")
        return

    # Check for direct parquet files (if user pointed to chunks dir)
    chunk_files = sorted(chunks_dir.glob("chunk_*.parquet"))
    
    # Check for 'history_chunks' subdirectory (if user pointed to sim root)
    if not chunk_files:
        subdir = chunks_dir / "history_chunks"
        if subdir.exists():
             chunk_files = sorted(subdir.glob("chunk_*.parquet"))
             
    if not chunk_files:
        print(f"No chunk files found in {chunks_dir} or {chunks_dir}/history_chunks.")
        return

    print(f"Found {len(chunk_files)} chunks in {chunks_dir}...")
    
    total_non_rfnbo = 0.0
    total_rfnbo = 0.0
    
    for chunk_file in chunk_files:
        try:
            # Only load necessary columns for speed
            df = pd.read_parquet(chunk_file, columns=['h2_rfnbo_kg', 'h2_non_rfnbo_kg'])
            # print(f"Scanning {chunk_file.name}...")
            
            if 'h2_non_rfnbo_kg' in df.columns:
                non_rfnbo_sum = df['h2_non_rfnbo_kg'].sum()
                rfnbo_sum = df['h2_rfnbo_kg'].sum()
                
                # print(f"  h2_non_rfnbo_kg: {non_rfnbo_sum:.2f}")
                
                total_non_rfnbo += non_rfnbo_sum
                total_rfnbo += rfnbo_sum
            else:
                print(f"  WARNING: h2_non_rfnbo_kg column missing in {chunk_file.name}!")
                
                
        except Exception as e:
            # Fallback if columns don't exist
            try:
                 df = pd.read_parquet(chunk_file)
                 if 'h2_non_rfnbo_kg' in df.columns:
                     total_non_rfnbo += df['h2_non_rfnbo_kg'].sum()
                     total_rfnbo += df['h2_rfnbo_kg'].sum()
            except Exception as e2:
                 print(f"  Error reading {chunk_file.name}: {e}")
        
        # Explicitly release memory
        del df
        gc.collect()

    print("\n--- Summary ---")
    print(f"Total Non-RFNBO H2: {total_non_rfnbo:.2f} kg")
    print(f"Total RFNBO H2:     {total_rfnbo:.2f} kg")
    total_h2 = total_rfnbo + total_non_rfnbo
    pct = (total_rfnbo / total_h2 * 100) if total_h2 > 0 else 0.0
    print(f"RFNBO Compliance:   {pct:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", help="Directory containing history chunks")
    args = parser.parse_args()
    inspect_chunks(args.dir)
