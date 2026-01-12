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
PROJECT_ROOT = Path(__file__).parent
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


def load_history_from_chunks(output_dir: Path) -> dict:
    """Load history from parquet chunks (memory efficient)."""
    chunks_dir = output_dir / "history_chunks"
    
    if not chunks_dir.exists():
        return None
    
    chunk_files = sorted(chunks_dir.glob("chunk_*.parquet"))
    if not chunk_files:
        return None
    
    print(f"Loading from {len(chunk_files)} parquet chunks...")
    dfs = []
    for i, chunk_file in enumerate(chunk_files):
        try:
            df = pd.read_parquet(chunk_file)
            dfs.append(df)
            if (i + 1) % 10 == 0:
                print(f"  Loaded {i + 1}/{len(chunk_files)} chunks...")
        except Exception as e:
            print(f"  Warning: Failed to load {chunk_file.name}: {e}")
    
    if not dfs:
        return None
    
    result_df = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()
    
    print(f"  Total: {len(result_df)} rows, {len(result_df.columns)} columns")
    return {col: result_df[col].values for col in result_df.columns}


def load_history_from_csv(csv_path: Path) -> dict:
    """Load history from CSV file."""
    print(f"Loading from CSV: {csv_path.name}...")
    size_gb = csv_path.stat().st_size / (1024**3)
    print(f"  File size: {size_gb:.2f} GB")
    
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    return {col: df[col].values for col in df.columns}


def regenerate_graphs_with_timeout(output_dir: Path, timeout_seconds: int = 60):
    """
    Regenerate graphs using the same logic as run_integrated_simulation.py,
    with per-graph timeout protection.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import yaml
    
    graphs_dir = output_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    
    # Load history (prefer chunks)
    history = load_history_from_chunks(output_dir)
    
    if history is None:
        csv_path = output_dir / "simulation_history.csv"
        if csv_path.exists():
            try:
                history = load_history_from_csv(csv_path)
            except MemoryError:
                print("ERROR: Out of memory loading CSV.")
                return
        else:
            print("ERROR: No history data found.")
            return
    
    if not history:
        print("ERROR: History is empty.")
        return
    
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
    
    # Filter scalar history (exclude matrices)
    scalar_history = {k: v for k, v in history.items() 
                     if not (isinstance(v, np.ndarray) and v.ndim > 1)}
    
    # Column aliases (same as run_integrated_simulation.py)
    COLUMN_ALIASES = {
        'H2_soec_kg': 'H2_soec', 'H2_pem_kg': 'H2_pem',
        'spot_price': 'Spot', 'P_soec_actual': 'P_soec',
        'steam_soec_kg': 'Steam_soec', 'H2O_soec_out_kg': 'H2O_soec',
        'H2O_pem_kg': 'H2O_pem', 'O2_pem_kg': 'O2_pem',
        'cumulative_h2_kg': 'Cumulative_H2', 'pem_V_cell': 'PEM_V_cell',
        'tank_level_kg': 'Tank_Level', 'tank_pressure_bar': 'Tank_Pressure',
        'compressor_power_kw': 'Compressor_Power',
    }
    
    # Create full DataFrame
    print("\nCreating DataFrame...")
    full_df = pd.DataFrame(scalar_history)
    for old_name, new_name in COLUMN_ALIASES.items():
        if old_name in full_df.columns and new_name not in full_df.columns:
            full_df[new_name] = full_df[old_name]
    if 'minute' in full_df.columns:
        full_df['time'] = full_df['minute'] / 60.0
        full_df['hour'] = full_df['time']
    
    print(f"DataFrame ready: {len(full_df)} rows, {len(full_df.columns)} columns")
    
    success_count = 0
    skip_count = 0
    error_count = 0
    timeout_count = 0
    
    # =========================================================================
    # PART 1: Legacy GRAPH_MAP graphs
    # =========================================================================
    print("\n" + "=" * 60)
    print("PART 1: Legacy Graphs (GRAPH_MAP)")
    print("=" * 60)
    
    # Skip these known-problematic graphs (too slow or cause crashes)
    SKIP_GRAPHS = {
        # 'temporal_averages',  # Optimised! Should work now.
    }
    
    try:
        from h2_plant.visualization.static_graphs import (
            create_dispatch_figure, create_arbitrage_figure,
            create_h2_production_figure, create_oxygen_figure,
            create_water_figure, create_energy_pie_figure,
            create_histogram_figure, create_dispatch_curve_figure,
            create_cumulative_h2_figure, create_cumulative_energy_figure,
            create_efficiency_curve_figure, create_revenue_analysis_figure,
            create_temporal_averages_figure, create_water_removal_total_figure,
            create_drains_discarded_figure, create_chiller_cooling_figure,
            create_coalescer_separation_figure, create_kod_separation_figure,
            create_dry_cooler_figure, create_energy_flow_figure,
            create_q_breakdown_figure, create_thermal_load_breakdown_figure,
            create_plant_balance_figure, create_mixer_comparison_figure,
            create_individual_drains_figure, create_dissolved_gas_concentration_figure,
            create_rfnbo_compliance_figure, create_rfnbo_spot_analysis_figure,
            create_rfnbo_pie_figure, create_cumulative_rfnbo_figure,
            create_soec_module_heatmap_figure, create_soec_module_power_stacked_figure,
            create_soec_module_wear_stats_figure, create_effective_ppa_figure,
            create_water_tank_inventory_figure,
        )
        
        GRAPH_MAP = {
            'dispatch_strategy_stacked': create_dispatch_figure,
            'arbitrage_scatter': create_arbitrage_figure,
            'total_h2_production_stacked': create_h2_production_figure,
            'oxygen_production_stacked': create_oxygen_figure,
            'water_consumption_stacked': create_water_figure,
            'power_consumption_breakdown_pie': create_energy_pie_figure,
            'price_histogram': create_histogram_figure,
            'dispatch_curve_scatter': create_dispatch_curve_figure,
            'cumulative_h2_production': create_cumulative_h2_figure,
            'cumulative_energy': create_cumulative_energy_figure,
            'efficiency_curve': create_efficiency_curve_figure,
            'revenue_analysis': create_revenue_analysis_figure,
            'temporal_averages': create_temporal_averages_figure,
            'water_removal_total': create_water_removal_total_figure,
            'drains_discarded': create_drains_discarded_figure,
            'chiller_cooling': create_chiller_cooling_figure,
            'coalescer_separation': create_coalescer_separation_figure,
            'kod_separation': create_kod_separation_figure,
            'dry_cooler_performance': create_dry_cooler_figure,
            'energy_flow': create_energy_flow_figure,
            'q_breakdown': create_q_breakdown_figure,
            'thermal_load_breakdown': create_thermal_load_breakdown_figure,
            'plant_balance': create_plant_balance_figure,
            'mixer_comparison': create_mixer_comparison_figure,
            'individual_drains': create_individual_drains_figure,
            'dissolved_gas_concentration': create_dissolved_gas_concentration_figure,
            'rfnbo_compliance': create_rfnbo_compliance_figure,
            'rfnbo_spot_analysis': create_rfnbo_spot_analysis_figure,
            'rfnbo_pie': create_rfnbo_pie_figure,
            'cumulative_rfnbo': create_cumulative_rfnbo_figure,
            'soec_module_heatmap': create_soec_module_heatmap_figure,
            'soec_module_power_stacked': create_soec_module_power_stacked_figure,
            'soec_module_wear_stats': create_soec_module_wear_stats_figure,
            'effective_ppa': create_effective_ppa_figure,
            'water_tank_inventory': create_water_tank_inventory_figure,
        }
        
        enabled_graphs = viz_config.get('visualization', {}).get('graphs', {})
        
        for graph_name, graph_func in GRAPH_MAP.items():
            # Skip known-problematic graphs
            if graph_name in SKIP_GRAPHS:
                print(f"  {graph_name}... SKIP (blacklisted)")
                skip_count += 1
                continue
            
            graph_config = enabled_graphs.get(graph_name, {})
            if isinstance(graph_config, dict) and not graph_config.get('enabled', True):
                continue
            
            print(f"  {graph_name}...", end=" ", flush=True)
            
            try:
                with time_limit(timeout_seconds, graph_name):
                    fig = graph_func(full_df)
                    if fig is not None:
                        filename = graph_config.get('filename', graph_name) if isinstance(graph_config, dict) else graph_name
                        output_path = graphs_dir / f"{filename}.png"
                        fig.savefig(output_path, dpi=100, bbox_inches='tight')
                        plt.close(fig)
                        print("OK")
                        success_count += 1
                    else:
                        print("SKIP (None)")
                        skip_count += 1
            except TimeoutException:
                print(f"TIMEOUT ({timeout_seconds}s)")
                timeout_count += 1
                plt.close('all')
            except BaseException as e:
                # Catch ALL exceptions including those wrapped by decorators
                if 'timed out' in str(e).lower() or 'timeout' in type(e).__name__.lower():
                    print(f"TIMEOUT ({timeout_seconds}s)")
                    timeout_count += 1
                else:
                    print(f"ERROR: {str(e)[:50]}")
                    error_count += 1
                plt.close('all')
            
            gc.collect()
            
    except ImportError as e:
        print(f"Warning: Could not import graph functions: {e}")
    
    # =========================================================================
    # PART 2: Orchestrated Graphs (generate_all with timeout wrapper)
    # =========================================================================
    print("\n" + "=" * 60)
    print("PART 2: Orchestrated Graphs (GraphOrchestrator)")
    print("=" * 60)
    
    try:
        from h2_plant.visualization.graph_orchestrator import GraphOrchestrator
        
        orchestrator = GraphOrchestrator(graphs_dir)
        
        # Get orchestrated graph configs
        orchestrated_configs = viz_config.get('visualization', {}).get('orchestrated_graphs', {})
        
        for graph_type, settings in orchestrated_configs.items():
            # Skip disabled or boolean entries
            if isinstance(settings, bool):
                continue
            if not settings.get('enabled', False):
                continue
            
            # Get handler from orchestrator
            handler = orchestrator.handlers.get(graph_type)
            if not handler:
                print(f"  {graph_type}... SKIP (no handler)")
                skip_count += 1
                continue
            
            plots = settings.get('plots', [])
            if not plots:
                print(f"  {graph_type}... SKIP (no plots)")
                skip_count += 1
                continue
            
            for i, plot_config in enumerate(plots):
                title = plot_config.get('title', f"{graph_type}_{i}")
                components = plot_config.get('components', [])
                
                print(f"  {title}...", end=" ", flush=True)
                
                try:
                    with time_limit(timeout_seconds, title):
                        fig = handler(full_df, components, title, plot_config)
                        
                        if fig:
                            safe_title = "".join([c if c.isalnum() else "_" for c in title])
                            filename = graphs_dir / f"{safe_title}.png"
                            fig.savefig(filename, dpi=100, bbox_inches='tight')
                            plt.close(fig)
                            print("OK")
                            success_count += 1
                        else:
                            print("SKIP (None)")
                            skip_count += 1
                except TimeoutException:
                    print(f"TIMEOUT ({timeout_seconds}s)")
                    timeout_count += 1
                    plt.close('all')
                except BaseException as e:
                    if 'timed out' in str(e).lower() or 'timeout' in type(e).__name__.lower():
                        print(f"TIMEOUT ({timeout_seconds}s)")
                        timeout_count += 1
                    else:
                        print(f"ERROR: {str(e)[:50]}")
                        error_count += 1
                    plt.close('all')
                
                gc.collect()
            
    except ImportError as e:
        print(f"Warning: GraphOrchestrator not available: {e}")
    except Exception as e:
        print(f"Warning: Orchestrated graphs failed: {e}")
    
    # =========================================================================
    # PART 3: Daily H2 Production Average
    # =========================================================================
    print("\n" + "=" * 60)
    print("PART 3: Daily H2 Production Graph")
    print("=" * 60)
    
    try:
        daily_config = viz_config.get('visualization', {}).get('orchestrated_graphs', {}).get('daily_h2_production_average', {})
        if daily_config.get('enabled', False):
            print("  daily_h2_production...", end=" ", flush=True)
            try:
                with time_limit(timeout_seconds, 'daily_h2_production'):
                    from scripts.plot_daily_h2_production import generate_daily_h2_production_graph
                    csv_path = output_dir / "simulation_history.csv"
                    output_path = graphs_dir / "daily_h2_production.png"
                    if csv_path.exists():
                        generate_daily_h2_production_graph(str(csv_path), str(output_path))
                        print("OK")
                        success_count += 1
                    else:
                        print("SKIP (no CSV)")
                        skip_count += 1
            except TimeoutException:
                print(f"TIMEOUT ({timeout_seconds}s)")
                timeout_count += 1
            except Exception as e:
                print(f"ERROR: {str(e)[:50]}")
                error_count += 1
        else:
            print("  daily_h2_production... DISABLED")
    except Exception as e:
        print(f"Warning: Daily H2 production failed: {e}")
    
    # Cleanup
    del full_df
    del scalar_history
    del history
    gc.collect()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print(f"  Success:  {success_count}")
    print(f"  Timeout:  {timeout_count}")
    print(f"  Skipped:  {skip_count}")
    print(f"  Errors:   {error_count}")
    print(f"  Output:   {graphs_dir}")
    print("=" * 60)


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
