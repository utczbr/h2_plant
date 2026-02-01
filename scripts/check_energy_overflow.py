
import logging
import pandas as pd
import numpy as np
from h2_plant.run_integrated_simulation import run_with_dispatch_strategy

# Suppress simulation logs
logging.getLogger('h2_plant').setLevel(logging.WARNING)

print("Running simulation to generate history...")
history = run_with_dispatch_strategy('scenarios', hours=1)
df = pd.DataFrame(history)

# Defined Columns
P_soec = df['P_soec_actual'] if 'P_soec_actual' in df.columns else df['P_soec']
P_pem = df['P_pem']
P_aux = df['auxiliary_power_kw'] / 1000.0 if 'auxiliary_power_kw' in df.columns else (df['compressor_power_kw']/1000.0)
P_sold = df['P_sold']
P_consumed_total = P_soec + P_pem + P_aux

# Available Power
# Use P_offer as the ground truth for what the dispatch engine THOUGHT was available
P_available = df['P_offer'] if 'P_offer' in df.columns else df['wind_coefficient'] * 20.0

# Calculate Overflow
# Tolerance for floating point (e.g. 1e-6 MW = 1 Watt)
TOLERANCE = 1e-6
df['overflow'] = P_consumed_total - P_available
overflow_mask = df['overflow'] > TOLERANCE

print("\n\n=== Energy Overflow Analysis ===")
if overflow_mask.any():
    n_overflow = overflow_mask.sum()
    pct_overflow = (n_overflow / len(df)) * 100
    max_overflow = df.loc[overflow_mask, 'overflow'].max()
    avg_overflow = df.loc[overflow_mask, 'overflow'].mean()
    
    print(f"⚠️  OVERFLOW DETECTED!")
    print(f"Count: {n_overflow} timesteps ({pct_overflow:.2f}%)")
    print(f"Max Overflow: {max_overflow:.6f} MW")
    print(f"Avg Overflow: {avg_overflow:.6f} MW")
    
    # Show worst case
    worst_idx = df['overflow'].idxmax()
    row = df.loc[worst_idx]
    print("\nWorst Case Snapshot (Minute {}):".format(row['minute']))
    print(f"  Available: {row['P_offer']:.6f} MW")
    print(f"  Consumed:  {P_consumed_total[worst_idx]:.6f} MW")
    print(f"    - SOEC:  {P_soec[worst_idx]:.6f} MW")
    print(f"    - PEM:   {P_pem[worst_idx]:.6f} MW")
    print(f"    - Aux:   {P_aux[worst_idx]:.6f} MW")
    print(f"  Diff:      {row['overflow']:.6f} MW")
    
else:
    print("✅ No energy overflow detected.")
    print(f"Max Consumption vs Available: {(P_consumed_total - P_available).max():.6f} MW (Negative means safe)")

print("\n=== Visualization Analysis (Downsampled) ===")

def downsample(data, max_points=500):
    if len(data) <= max_points: return data
    stride = max(1, len(data) // max_points)
    return data[::stride]

# Apply same downsampling as static_graphs.py
P_soec_ds = downsample(P_soec.values)
P_pem_ds = downsample(P_pem.values)
P_aux_ds = downsample(P_aux.values)
P_sold_ds = downsample(P_sold.values)
P_offer_ds = downsample(P_available.values)

P_stack_ds = P_soec_ds + P_pem_ds + P_aux_ds + P_sold_ds
Gap_vis_ds = P_offer_ds - P_stack_ds
    
print(f"Downsampled Points: {len(P_offer_ds)}")
print(f"Max Gap (Offer - Stack): {Gap_vis_ds.max():.6f} MW")
print(f"Min Gap (Offer - Stack): {Gap_vis_ds.min():.6f} MW")

# Check if Consumed > Offer in downsampled data
P_consumed_ds = P_soec_ds + P_pem_ds + P_aux_ds
Overflow_ds = P_consumed_ds - P_offer_ds

if (Overflow_ds > 1e-6).any():
    print(f"⚠️  VISUAL OVERFLOW DETECTED in downsampled data!")
    print(f"Max Overflow: {Overflow_ds.max():.6f} MW")
    print("This confirms the issue is a visual artifact caused by downsampling/slicing.")
else:
    print("✅ No overflow in downsampled data either.")
    print("Graph rendering (interpolation/stacking) is the likely culprit.")
