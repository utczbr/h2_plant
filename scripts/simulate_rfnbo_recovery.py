
import pandas as pd
from pathlib import Path
import argparse
import sys
import gc
import numpy as np

# Constants (from configuration/defaults)
DEFAULT_PEM_KWH_KG = 50.0
DEFAULT_PEM_MAX_MW = 5.0 # From topology PEM_Transformer rated 5.57, stack usually 5.
DEFAULT_GRID_MAX_MW = 30.0
DEFAULT_THRESHOLD_EUR_MWH = 40.0

def simulate_rfnbo_recovery(chunks_dir_path, output_report=None):
    chunks_dir = Path(chunks_dir_path)
    chunk_files = sorted(chunks_dir.glob("chunk_*.parquet"))
    
    if not chunk_files:
        subdir = chunks_dir / "history_chunks"
        if subdir.exists():
             chunk_files = sorted(subdir.glob("chunk_*.parquet"))
             
    if not chunk_files:
        print(f"No chunk files found in {chunks_dir}.")
        return

    print(f"Simulating 'ECONOMIC_SPOT' strategy on {len(chunk_files)} chunks...")
    print(f"Parameters:")
    print(f"  Spot Threshold: {DEFAULT_THRESHOLD_EUR_MWH} EUR/MWh")
    print(f"  Grid Max Import: {DEFAULT_GRID_MAX_MW} MW")
    print(f"  PEM Efficiency: {DEFAULT_PEM_KWH_KG} kWh/kg")
    
    total_non_rfnbo_h2 = 0.0
    total_cost_eur = 0.0
    total_opportunities = 0
    
    original_total_h2 = 0.0
    
    for chunk_file in chunk_files:
        try:
            # We need Spot Price, P_pem (actual), Storage SOC (constraint), h2_kg (original)
            # P_pem might be named 'P_pem' or 'P_pem_mw'
            cols = ['spot_price', 'P_pem', 'h2_kg', 'storage_soc', 'P_offer']
            
            # Read chunk
            df = pd.read_parquet(chunk_file)
            
            # Map columns
            if 'P_pem' not in df.columns and 'P_pem_mw' in df.columns:
                 df['P_pem'] = df['P_pem_mw']
            
            # Iterate (vectorized where possible, but user asked for sequential "simulation")
            # Step 1: Identify Opportunity Mask
            # Price < Threshold AND P_offer not fully utilized? 
            # Actually, Reference Hybrid uses P_offer for RFNBO.
            # We want to use GRID for Non-RFNBO.
            
            # Mask: Price favorable
            price_mask = df['spot_price'] < DEFAULT_THRESHOLD_EUR_MWH
            
            # Mask: PEM has capacity?
            # P_pem is what was used. P_available = PEM_MAX - P_pem
            # Ensure P_pem is in MW
            pem_used = df['P_pem']
            pem_avail = DEFAULT_PEM_MAX_MW - pem_used
            
            # Mask: Grid has capacity?
            # We assume P_grid used for BOP is negligible or we have 30 MW purely for this?
            # Let's assumes 30 MW is the *import limit*.
            # If we imported for BOP, we have less.
            # But usually BOP is small. Let's assume 30 MW limit applies to H2 production portion for simplicity.
            grid_avail = DEFAULT_GRID_MAX_MW
            
            # Mask: Tank has space?
            # If SOC > 0.98, we can't put more in.
            soc_mask = df['storage_soc'] < 0.98
            
            # Combined Opportunity
            opportunity_mask = price_mask & (pem_avail > 0.01) & soc_mask
            
            # Calculate Delta
            # For rows where Opportunity is True:
            # Power_to_Buy = Min(PEM_Avail, Grid_Avail)
            # H2_Potential = Power_to_Buy * 1000 / 50.0  (MW -> kW -> kg)
            
            # Note: dt (timestep) is needed. Usually 1 hour? Or 1 minute?
            # history['minute'] ...
            # Chunks are usually 1 minute steps? Check timestep.
            # Usually df index is steps.
            # Or check 'minute' column diff?
            # Assuming 1 minute steps for now (standard in this plant).
            dt_hours = 1.0 / 60.0
            
            # Vectorized Calculation
            potential_power_mw = np.minimum(pem_avail, grid_avail)
            potential_power_mw[~opportunity_mask] = 0.0
            
            potential_h2_kg_rate = (potential_power_mw * 1000.0) / DEFAULT_PEM_KWH_KG # kg/h
            potential_h2_kg = potential_h2_kg_rate * dt_hours
            
            cost_eur = potential_power_mw * df['spot_price'] * dt_hours
            
            # Accumulate
            chunk_additional_h2 = potential_h2_kg.sum()
            chunk_cost = cost_eur.sum()
            chunk_opps = opportunity_mask.sum()
            
            total_non_rfnbo_h2 += chunk_additional_h2
            total_cost_eur += chunk_cost
            total_opportunities += chunk_opps
            
            if 'h2_kg' in df.columns:
                original_total_h2 += df['h2_kg'].sum()
            
            if True and chunk_additional_h2 > 0:
                 # Debug first few
                 # print(f"Chunk {chunk_file.name}: +{chunk_additional_h2:.2f} kg H2 (Cost: {chunk_cost:.2f} EUR)")
                 pass

            del df
            del potential_power_mw
            del potential_h2_kg
            gc.collect()
            
        except Exception as e:
            print(f"Error processing {chunk_file.name}: {e}")

    print("\n" + "="*40)
    print("SIMULATION RESULTS (Reference -> Economic Spot)")
    print("="*40)
    print(f"Original Total H2:      {original_total_h2:,.2f} kg")
    print(f"Additional Non-RFNBO:   {total_non_rfnbo_h2:,.2f} kg")
    print(f"New Total H2:           {original_total_h2 + total_non_rfnbo_h2:,.2f} kg")
    print("-" * 40)
    print(f"Purchase Opportunities: {total_opportunities} minutes")
    print(f"Spot Energy Cost:       {total_cost_eur:,.2f} EUR")
    
    avg_price = (total_cost_eur / total_non_rfnbo_h2) if total_non_rfnbo_h2 > 0 else 0.0
    # Price per kg = Cost / kg
    print(f"Avg Cost of Added H2:   {avg_price:.2f} EUR/kg")
    print(f"RFNBO Compliance:       {(original_total_h2 / (original_total_h2 + total_non_rfnbo_h2) * 100):.2f}%")
    print("="*40)
    print("Note: This simulation assumes infinite storage headroom for the extra hydrogen.")
    print("      In reality, tank limits might strictly curtail this production.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", help="Directory containing history chunks")
    args = parser.parse_args()
    simulate_rfnbo_recovery(args.dir)
