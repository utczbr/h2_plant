#!/usr/bin/env python3
"""
Regenerate Graphs from Existing Simulation History (Memory-Safe Version).

This script loads an existing simulation_history.csv and regenerates ALL graphs
using a streaming architecture to prevent memory explosion.

Features:
- Streaming chunk processing (never loads all data at once)
- Configurable memory budget
- Batched graph execution with extensive garbage collection
- 60-second timeout per graph

Usage:
    python regenerate_graphs.py scenarios/simulation_output --max-memory-mb 2000
    python regenerate_graphs.py /path/to/output --batch-size 5
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


def regenerate_graphs_safe(
    output_dir: Path,
    timeout_seconds: int = 60,
    max_memory_mb: int = 4000,
    target_resolution: int = 60,
    batch_size: int = 10,
    skip_cache: bool = False
):
    """
    Memory-safe graph regeneration.
    
    Steps:
    1. Check for existing cache or chunks
    2. If no cache or skip_cache: Use StreamingDownsampler
    3. Load lightweight DataFrame from cache
    4. Execute graphs in batches with memory monitoring
    5. Report statistics
    """
    import yaml
    from h2_plant.visualization.graph_catalog import GRAPH_REGISTRY
    from h2_plant.visualization.unified_executor import UnifiedGraphExecutor
    from h2_plant.visualization.streaming_downsampler import StreamingDownsampler, MemoryMonitor
    
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
    
    print("\n" + "=" * 60)
    print(f"MEMORY-SAFE GRAPH REGENERATION (Budget: {max_memory_mb} MB)")
    print("=" * 60)
    
    chunks_dir = output_dir / "history_chunks"
    cache_path = output_dir / "simulation_history_hourly.parquet"
    
    # STEP 2: Configure executor (early init to get required columns)
    executor = UnifiedGraphExecutor(GRAPH_REGISTRY, graphs_dir)
    executor.configure_from_yaml(viz_config)
    executor.memory_monitor = MemoryMonitor(max_memory_mb)

    # Calculate required columns for cache creation
    required_cols = list(executor.get_required_columns())
    # Expand patterns if possible (schema check) to avoid missing dynamic columns
    # But since we don't have the DF yet, we rely on patterns.
    # However, streaming downsampler needs ACTUAL column names for read_parquet(columns=...).
    # If we pass patterns like 'soec_module_*', read_parquet will fail.
    # We need to peek at the first chunk to resolve wildcards.
    
    resolved_columns = None
    if chunks_dir.exists():
        try:
            # Peek at first chunk to resolve wildcards
            # We use sorted() to match downsampler order
            first_chunk = sorted(chunks_dir.glob("chunk_*.parquet"))[0]
            import pyarrow.parquet as pq
            schema = pq.read_schema(first_chunk)
            all_cols = schema.names
            
            resolved_columns = list(executor._expand_patterns(set(required_cols), all_cols))
            
            # FORCE MINUTE: Critical fix for missing data issue
            if 'minute' not in resolved_columns:
                resolved_columns.append('minute')
                
            print(f"resolved {len(resolved_columns)} columns from {len(all_cols)} available.")
            if len(resolved_columns) < 20:
                 print(f"Columns: {resolved_columns}")
        except Exception as e:
            logger.warning(f"Failed to resolve column patterns from chunks: {e}. Cache creation might be slow or fail.")
            # If resolution fails, we might have to load all? Or pass None.
            resolved_columns = None
            
    # STEP 1: Ensure downsampled cache exists
    df = None
    
    if skip_cache or not cache_path.exists():
        if chunks_dir.exists():
            print("Creating downsampled cache from chunks...")
            # Reserve 50% of budget for this operation as it builds a new DF
            downsampler = StreamingDownsampler(
                max_memory_mb=max_memory_mb // 2,
                target_resolution_minutes=target_resolution
            )
            
            try:
                # Pass resolved columns to create LEAN cache
                df = downsampler.process_chunks_directory(chunks_dir, cache_path, required_columns=resolved_columns)
                if not df.empty and 'minute' in df.columns:
                     print(f"Cache created: {len(df)} rows. Range: {df.minute.min()}-{df.minute.max()} minutes.")
                else:
                     print(f"Cache created: {len(df)} rows.")
            except Exception as e:
                logger.error(f"Failed to create cache: {e}")
        else:
            print("No chunks directory found. Checking for CSV...")
            csv_path = output_dir / "simulation_history.csv"
            if csv_path.exists():
                 print(f"Found CSV at {csv_path}. Warning: CSV loading is memory intensive.")
                 pass
            else:
                logger.error("No data found (chunks or CSV). Cannot regenerate.")
                return
    else:
        print(f"Using existing cache: {cache_path}")
        try:
            # When loading existing cache, ALSO filter columns to be safe
            # But resolved_columns might be None if we skipped that block.
            # Re-resolve if needed.
            if resolved_columns is None:
                 # Try to peak at cache schema
                 try:
                    import pyarrow.parquet as pq
                    schema = pq.read_schema(cache_path)
                    all_cols = schema.names
                    resolved_columns = list(executor._expand_patterns(set(required_cols), all_cols))
                 except:
                    pass

            df = pd.read_parquet(cache_path, columns=resolved_columns)
            # Normalize immediately
            from h2_plant.visualization.static_graphs import normalize_history
            df = normalize_history(df)
        except Exception as e:
            logger.warning(f"Failed to read cache: {e}. Will attempt other sources.")
            
    # Update executor's memory monitor with CLI arg
    # executor.memory_monitor = MemoryMonitor(max_memory_mb) # Already done above
    
    # If we haven't loaded df yet
    if df is None:
        print("Loading data via executor fallback...")
        df = executor.load_data(
            chunks_dir=chunks_dir if chunks_dir.exists() else None,
            csv_path=output_dir / "simulation_history.csv",
            cache_path=cache_path, # might retry loading checks
            downsample_factor=target_resolution
        )
        
    if df is None or df.empty:
        print("ERROR: No history data found or loaded.")
        return

    # INJECT CONFIG
    if 'config' not in df.attrs:
        df.attrs['config'] = {}
    df.attrs['config'].update(viz_config.get('plant_parameters', {}))
    df.attrs['viz_config'] = viz_config
        
    print(f"Ready to Plot: {df.shape[0]} rows x {df.shape[1]} columns")
    
    # STEP 3: Execute with memory safety
    print(f"Generating graphs with batch size {batch_size}...")
    results = executor.execute_batched(
        df,
        timeout_seconds=timeout_seconds,
        batch_size=batch_size
    )
    
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
    
    executor.memory_monitor.log_usage("Final")
    
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
        description='Regenerate graphs from existing simulation history (Memory-Safe).'
    )
    parser.add_argument(
        'output_dir', type=str,
        help='Path to simulation output directory'
    )
    parser.add_argument(
        '--timeout', type=int, default=60,
        help='Timeout in seconds per graph (default: 60)'
    )
    parser.add_argument(
        '--max-memory-mb', type=int, default=4000,
        help='Maximum RAM usage in MB (default: 4000)'
    )
    parser.add_argument(
        '--target-resolution', type=int, default=60,
        help='Downsampling resolution in minutes (default: 60)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=10,
        help='Graphs per batch (default: 10)'
    )
    parser.add_argument(
        '--skip-cache', action='store_true',
        help='Force rebuild of downsampled cache'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir).resolve()
    if not output_dir.exists():
        print(f"ERROR: Directory not found: {output_dir}")
        sys.exit(1)
    
    regenerate_graphs_safe(
        output_dir,
        timeout_seconds=args.timeout,
        max_memory_mb=args.max_memory_mb,
        target_resolution=args.target_resolution,
        batch_size=args.batch_size,
        skip_cache=args.skip_cache
    )


if __name__ == "__main__":
    main()
