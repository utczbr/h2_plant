
import json
import pandas as pd
from pathlib import Path
import argparse
import sys
import logging
import gc

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def patch_dashboard(simulation_dir: str):
    sim_path = Path(simulation_dir)
    chunks_dir = sim_path / "history_chunks"
    metrics_dir = sim_path / "metrics"
    dashboard_path = metrics_dir / "dashboard_data.json"
    
    if not chunks_dir.exists():
        logger.error(f"History chunks directory not found: {chunks_dir}")
        return
    
    if not dashboard_path.exists():
        logger.error(f"Dashboard data file not found: {dashboard_path}")
        return
        
    logger.info(f"Scanning history chunks in {chunks_dir}...")
    
    # helper to read chunks
    total_rfnbo = 0.0
    total_non_rfnbo = 0.0
    
    chunk_files = sorted(chunks_dir.glob("chunk_*.parquet"))
    if not chunk_files:
        logger.error("No chunk files found.")
        return
        
    for chunk_file in chunk_files:
        try:
            # Only read necessary columns
            columns = ['h2_rfnbo_kg', 'h2_non_rfnbo_kg']
            # Handle case where columns might not exist (though they should)
            df = pd.read_parquet(chunk_file)
            
            if 'h2_rfnbo_kg' in df.columns:
                total_rfnbo += df['h2_rfnbo_kg'].sum()
            
            if 'h2_non_rfnbo_kg' in df.columns:
                total_non_rfnbo += df['h2_non_rfnbo_kg'].sum()
                
        except Exception as e:
            logger.warning(f"Error reading {chunk_file.name}: {e}")
        
        # Explicit release
        del df
        gc.collect()
            
    logger.info(f"Calculated Totals from History:")
    logger.info(f"  RFNBO H2:     {total_rfnbo:.2f} kg")
    logger.info(f"  Non-RFNBO H2: {total_non_rfnbo:.2f} kg")
    
    # Load dashboard data
    with open(dashboard_path, 'r') as f:
        data = json.load(f)
        
    # Update KPIs
    if 'kpis' not in data:
        data['kpis'] = {}
        
    old_non_rfnbo = data['kpis'].get('h2_non_rfnbo_kg', 0.0)
    
    data['kpis']['h2_rfnbo_kg'] = float(total_rfnbo)
    data['kpis']['h2_non_rfnbo_kg'] = float(total_non_rfnbo)
    
    # Recalculate percentage
    total_h2 = total_rfnbo + total_non_rfnbo
    if total_h2 > 0:
        pct = (total_rfnbo / total_h2) * 100.0
    else:
        pct = 0.0 # Or 100 if we prefer optimistic default, but 0 makes sense for no production
        
    data['kpis']['rfnbo_compliance_pct'] = float(pct)
    
    logger.info(f"Updating dashboard data (Old Non-RFNBO: {old_non_rfnbo:.2f} -> New: {total_non_rfnbo:.2f})")
    
    # Save back
    # Backup first
    backup_path = dashboard_path.with_suffix('.json.bak')
    import shutil
    shutil.copy(dashboard_path, backup_path)
    logger.info(f"Backup created at {backup_path}")
    
    with open(dashboard_path, 'w') as f:
        json.dump(data, f, indent=2)
        
    logger.info("Dashboard data patched successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patch RFNBO metrics in dashboard_data.json from history chunks.")
    parser.add_argument("simulation_dir", help="Path to simulation output directory (e.g., scenarios/simulation_output/Sim_Name)")
    
    args = parser.parse_args()
    patch_dashboard(args.simulation_dir)
