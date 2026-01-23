#!/usr/bin/env python3
"""
Diagnostic script to trace H2O mass and mole fractions through the SOEC train.
Specific focus: SOEC_Cluster -> Interchanger -> DryCooler.
"""
import pandas as pd
import glob
import os
import numpy as np

# Load history
output_dir = "/home/stuart/Documentos/Planta Hidrogenio/scenarios/simulation_output/history_chunks"
parquet_pattern = os.path.join(output_dir, "chunk_*.parquet")
chunks = sorted(glob.glob(parquet_pattern))

if not chunks:
    print("No chunks found.")
    exit(1)

print(f"Loading {chunks[-1]}...")
df = pd.read_parquet(chunks[-1])

components = [
    'SOEC_Cluster',
    'SOEC_H2_Interchanger_1',
    'SOEC_H2_DryCooler_1' # Verify case sensitivity here? Script output showed DryCooler (capital C)
]

print(f"\n{'Component':<30} | {'Flow (kg/h)':<12} | {'H2O Mass Frac':<13} | {'H2O Mole Frac (calc)':<20} | {'Temp (C)':<10}")
print("-" * 100)

for cid in components:
    # Try different casing if needed, but diagnostic showed DryCooler (capital)
    # Check flow
    flow_col = f"{cid}_outlet_mass_flow_kg_h"
    flow = df[flow_col].mean() if flow_col in df.columns else np.nan
    
    # Check H2O mass frac (legacy)
    frac_col = f"{cid}_outlet_h2o_frac"
    if frac_col not in df.columns:
        # Try lowercase c
        frac_col_alt = f"{cid.replace('Cooler', 'cooler')}_outlet_h2o_frac"
        if frac_col_alt in df.columns:
            frac_col = frac_col_alt
            
    frac = df[frac_col].mean() if frac_col in df.columns else np.nan
    
    # Check Temp
    temp_col = f"{cid}_outlet_temp_c"
    temp = df[temp_col].mean() if temp_col in df.columns else np.nan
    
    # Calc Mole Frac from Mass Frac
    # MW H2O=18, H2=2 (approx)
    if pd.notnull(frac):
        if frac > 0:
            w = frac
            y = (w/18.015) / ((w/18.015) + ((1-w)/2.016))
            ppm = y * 1e6
        else:
            ppm = 0.0
    else:
        ppm = np.nan
        
    print(f"{cid:<30} | {flow:<12.2f} | {frac:<13.6f} | {ppm:<20.1f} | {temp:<10.1f}")

# Also check molf columns directly (should be zero in current history, but good to double check)
print("\nDirect Molf Columns:")
for cid in components:
    col = f"{cid}_outlet_H2O_molf"
    if col in df:
        print(f"{col}: {df[col].mean()}")
    else:
        print(f"{col}: Not found")
