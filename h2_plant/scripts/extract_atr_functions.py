#!/usr/bin/env python3
"""
Extract all linear interpolation functions from ATR_model_functions.pkl
and save the underlying x,y data points to a CSV file.

The pickle file contains scipy.interpolate.interp1d objects (with kind='linear'),
each representing a relationship between an input variable (x) and an output variable (y).

Output CSV format:
- One column for the x values (shared input variable for all functions)
- One column per function containing the corresponding y values
"""

import pickle
import csv
import os
from pathlib import Path


def main():
    # Paths
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"
    pkl_path = data_dir / "ATR_model_functions.pkl"
    output_path = data_dir / "ATR_linear_regressions.csv"
    
    # Load the pickle file
    print(f"Loading: {pkl_path}")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    
    print(f"Found {len(data)} interpolation functions:")
    for key in sorted(data.keys()):
        print(f"  - {key}")
    
    # Get the first function to determine x values (all should share the same x)
    first_key = list(data.keys())[0]
    x_values = data[first_key].x
    
    # Verify all functions share the same x values
    for key, func in data.items():
        if len(func.x) != len(x_values):
            print(f"Warning: {key} has different x length ({len(func.x)} vs {len(x_values)})")
        elif not all(func.x == x_values):
            print(f"Warning: {key} has different x values")
    
    # Build the CSV data
    # Header: x, func1, func2, ...
    sorted_keys = sorted(data.keys())
    header = ["x"] + sorted_keys
    
    # Rows: x_value, y_value_for_func1, y_value_for_func2, ...
    rows = []
    for i, x in enumerate(x_values):
        row = [x]
        for key in sorted_keys:
            func = data[key]
            # Get the y value at this index
            if i < len(func.y):
                row.append(func.y[i])
            else:
                row.append("")  # Missing value
        rows.append(row)
    
    # Write to CSV
    print(f"\nWriting to: {output_path}")
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    
    print(f"Successfully exported {len(rows)} data points for {len(sorted_keys)} functions.")
    print(f"\nOutput file: {output_path}")


if __name__ == "__main__":
    main()
