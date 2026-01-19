#!/usr/bin/env python3
"""
Regenerate Graphs from Existing Simulation History.

This script loads an existing simulation_history.csv and regenerates ALL graphs
(both legacy GRAPH_MAP and orchestrated graphs) WITHOUT re-running the simulation.

Features:
- 60-second timeout per graph to skip frozen/hanging graphs
- Memory-efficient loading from parquet chunks (preferred)
- Same graphs as run_integrated_simulation.py (legacy + orchestrated)

Usage:
    python regenerate_graphs.py scenarios/simulation_output
    python regenerate_graphs.py /path/to/simulation_output --timeout 120
"""

import os
import sys
import argparse
import signal
import gc
import pandas as pd
import numpy as np
from pathlib import Path
from contextlib import contextmanager
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class TimeoutException(Exception):
    """Raised when a graph generation exceeds the timeout."""
    pass


@contextmanager
def time_limit(seconds: int, graph_name: str):
    """Context manager to limit execution time of a block."""
    def signal_handler(signum, frame):
        raise TimeoutException(f"Graph '{graph_name}' timed out after {seconds}s")
    
    old_handler = signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)



def regenerate_graphs_with_timeout(output_dir: Path, timeout_seconds: int = 60):
    """
    Regenerate graphs using UnifiedGraphExecutor with per-graph timeout protection.
    
    This replaces the fragmented PART 1 (Legacy) / PART 2 (Orchestrated) logic with
    a single unified execution pipeline.
    """
    import yaml
    
    graphs_dir = output_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    
    # Load visualization config
    config_path = PROJECT_ROOT / "scenarios" / "visualization_config.yaml"
    viz_config = {}
    if config_path.exists():
        try:
            with open(config_path) as f:
                viz_config = yaml.safe_load(f) or {}
            print(f"Loaded visualization config from {config_path.name}")
        except Exception as e:
            print(f"Warning: Could not load config: {e}")
    
    # Use UnifiedGraphExecutor
    try:
        from h2_plant.visualization.graph_catalog import GRAPH_REGISTRY
        from h2_plant.visualization.unified_executor import UnifiedGraphExecutor
        
        print("\n" + "=" * 60)
        print("UNIFIED GRAPH REGENERATION")
        print("=" * 60)
        
        executor = UnifiedGraphExecutor(GRAPH_REGISTRY, graphs_dir)
        executor.configure_from_yaml(viz_config)
        
        # Determine data source (chunks prefered, then CSV)
        chunks_dir = output_dir / "history_chunks"
        csv_path = output_dir / "simulation_history.csv"
        
        df = None
        if chunks_dir.exists():
            print(f"Loading from chunks: {chunks_dir}")
            df = executor.load_data(chunks_dir=chunks_dir)
        elif csv_path.exists():
            print(f"Loading from CSV: {csv_path}")
            df = executor.load_data(csv_path=csv_path)
        else:
            print("ERROR: No history data found (checked history_chunks/ and simulation_history.csv)")
            return
            
        # INJECT CONFIG into df.attrs so static_graphs can access it via get_config()
        if df is not None:
            if 'config' not in df.attrs:
                df.attrs['config'] = {}
            # Merge existing config with viz_config (especially plant_parameters)
            # This ensures keys like 'soec_capacity_mw' are available
            df.attrs['config'].update(viz_config.get('plant_parameters', {}))
            
            # Also attach the full viz config just in case
            df.attrs['viz_config'] = viz_config
            
        if df.empty:
            print("ERROR: Loaded DataFrame is empty.")
            return

        print(f"Loaded DataFrame: {df.shape[0]} rows x {df.shape[1]} columns")
        
        # Execute
        results = executor.execute(df, timeout_seconds=timeout_seconds)
        
        # Summary
        success_count = sum(1 for r in results.values() if r.status == 'success')
        failed_count = sum(1 for r in results.values() if r.status == 'failed')
        timeout_count = sum(1 for r in results.values() if r.status == 'timeout')
        skipped_count = sum(1 for r in results.values() if r.status == 'skipped')
        
        print("\n" + "=" * 60)
        print("SUMMARY")
        print(f"  Success:  {success_count}")
        print(f"  Timeout:  {timeout_count}")
        print(f"  Skipped:  {skipped_count}")
        print(f"  Failed:   {failed_count}")
        print(f"  Output:   {graphs_dir}")
        print("=" * 60)
        
    except ImportError as e:
        print(f"Error: UnifiedGraphExecutor not available: {e}")
    except Exception as e:
        print(f"Error during regeneration: {e}")
    
    # Run Daily H2 Production (Special Case)
    try:
        daily_config = viz_config.get('visualization', {}).get('orchestrated_graphs', {}).get('daily_h2_production_average', {})
        if daily_config.get('enabled', False):
            print("\nGenerating Daily H2 Production...")
            from scripts.plot_daily_h2_production import generate_daily_h2_production_graph
            
            csv_path = output_dir / "simulation_history.csv"
            output_path = graphs_dir / "daily_h2_production.png"
            
            if csv_path.exists():
                with time_limit(timeout_seconds, 'daily_h2_production'):
                    generate_daily_h2_production_graph(str(csv_path), str(output_path))
                    print("  OK: daily_h2_production.png")
            else:
                print("  SKIP: simulation_history.csv needed for daily graph")
    except Exception as e:
        print(f"  Failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Regenerate graphs from existing simulation history.'
    )
    parser.add_argument(
        'output_dir', type=str,
        help='Path to simulation output directory'
    )
    parser.add_argument(
        '--timeout', type=int, default=60,
        help='Timeout in seconds per graph (default: 60)'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir).resolve()
    if not output_dir.exists():
        print(f"ERROR: Directory not found: {output_dir}")
        sys.exit(1)
    
    regenerate_graphs_with_timeout(output_dir, args.timeout)


if __name__ == "__main__":
    main()
