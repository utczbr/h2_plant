#!/usr/bin/env python3
"""
Regenerate Graphs from Existing Simulation History (Memory-Safe Version).

This script loads an existing simulation history and regenerates graphs using
memory-aware cache creation and execution strategies.
"""

import sys
import argparse
import signal
import gc
import pandas as pd
import psutil
from pathlib import Path
from contextlib import contextmanager
import logging
from typing import Dict, Any, List, Optional, Tuple

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


def compute_effective_memory_budget(user_budget_mb: int) -> Tuple[int, float, float]:
    """
    Cap requested memory budget by current system headroom.

    Returns:
        (effective_budget_mb, total_mb, available_mb)
    """
    vm = psutil.virtual_memory()
    total_mb = vm.total / 1e6
    available_mb = vm.available / 1e6
    effective_budget = min(user_budget_mb, int(min(total_mb * 0.75, available_mb * 0.90)))
    effective_budget = max(256, effective_budget)
    return effective_budget, total_mb, available_mb


def choose_execution_mode(
    execution_mode: str,
    estimated_df_mb: float,
    available_mb: float,
    resolved_columns_count: int,
    enabled_graphs_count: int,
) -> str:
    """Choose final execution mode based on memory and workload size."""
    mode = execution_mode.lower()
    if mode != "auto":
        return mode

    if (
        estimated_df_mb > (0.40 * available_mb)
        or resolved_columns_count >= 900
        or enabled_graphs_count >= 100
    ):
        return "sequential"
    return "batched"


def choose_cache_stride(
    max_extra_downsample: int,
    estimated_df_mb: float,
    available_mb: float,
) -> Tuple[Optional[int], List[int]]:
    """
    Select smallest extra stride that satisfies memory target.

    Target: estimated_df_mb / stride <= 0.35 * available_mb
    """
    max_stride = max(1, int(max_extra_downsample))
    candidates = [1]
    stride = 2
    while stride <= max_stride:
        candidates.append(stride)
        stride *= 2

    target_mb = 0.35 * available_mb
    for candidate in candidates:
        if candidate > 0 and (estimated_df_mb / candidate) <= target_mb:
            return candidate, candidates
    return None, candidates


def _estimate_cache_df_mb(cache_path: Path) -> float:
    """Estimate in-memory DataFrame footprint from parquet cache size."""
    if not cache_path.exists():
        return 0.0
    cache_mb = cache_path.stat().st_size / 1e6
    return cache_mb * 4.0


def _read_cache_row_count(cache_path: Path) -> Optional[int]:
    """Read parquet row count without loading full data if possible."""
    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(cache_path)
        return pf.metadata.num_rows
    except Exception:
        return None


def _inject_df_attrs(df: pd.DataFrame, attrs: Dict[str, Any]) -> None:
    """Merge shared attrs into loaded DataFrame."""
    if not attrs:
        return

    cfg = attrs.get("config")
    if isinstance(cfg, dict):
        if "config" not in df.attrs or not isinstance(df.attrs.get("config"), dict):
            df.attrs["config"] = {}
        df.attrs["config"].update(cfg)

    for key, value in attrs.items():
        if key == "config":
            continue
        df.attrs[key] = value


def _apply_skip_heavy_policy(executor) -> List[str]:
    """
    Disable heavy graphs when oom-policy=skip-heavy.

    Current heuristic:
    - Skip orchestrated graphs.
    - Skip graphs with very broad declared requirements.
    """
    disabled: List[str] = []
    for meta in list(executor.catalog.get_enabled()):
        req_count = len(meta.data_required or [])
        if meta.category == "orchestrated" or req_count > 250:
            executor.catalog.disable(meta.graph_id)
            disabled.append(meta.graph_id)

    if disabled:
        logger.warning("Skip-heavy policy disabled %d graphs.", len(disabled))
    return disabled


def regenerate_graphs_safe(
    output_dir: Path,
    timeout_seconds: int = 60,
    max_memory_mb: int = 4000,
    target_resolution: int = 60,
    batch_size: int = 10,
    skip_cache: bool = False,
    execution_mode: str = "auto",
    oom_policy: str = "auto-degrade",
    max_extra_downsample: int = 8,
):
    """
    Memory-safe graph regeneration.

    Steps:
    1. Build or reuse downsampled cache.
    2. Choose execution strategy (batched/sequential) with RAM-aware auto mode.
    3. Optionally apply extra downsampling stride when memory pressure is high.
    4. Generate graphs with memory monitoring.
    """
    import yaml
    from h2_plant.visualization.graph_catalog import GRAPH_REGISTRY
    from h2_plant.visualization.unified_executor import UnifiedGraphExecutor
    from h2_plant.visualization.streaming_downsampler import StreamingDownsampler, MemoryMonitor

    graphs_dir = output_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)

    effective_budget_mb, total_mb, available_mb = compute_effective_memory_budget(max_memory_mb)

    # Load visualization config
    config_path = PROJECT_ROOT / "scenarios" / "visualization_config.yaml"
    viz_config: Dict[str, Any] = {}
    if config_path.exists():
        try:
            with open(config_path) as f:
                viz_config = yaml.safe_load(f) or {}
            print(f"Loaded visualization config from {config_path.name}")
        except Exception as e:
            print(f"Warning: Could not load config: {e}")

    print("\n" + "=" * 60)
    print("MEMORY-SAFE GRAPH REGENERATION")
    print("=" * 60)
    print(f"Requested budget: {max_memory_mb} MB")
    print(f"Effective budget: {effective_budget_mb} MB (total={total_mb:.0f} MB, available={available_mb:.0f} MB)")

    chunks_dir = output_dir / "history_chunks"
    cache_path = output_dir / "simulation_history_hourly.parquet"

    # Configure executor (early init to compute required columns)
    executor = UnifiedGraphExecutor(GRAPH_REGISTRY, graphs_dir)
    executor.configure_from_yaml(viz_config)
    executor.memory_monitor = MemoryMonitor(effective_budget_mb)

    required_cols = list(executor.get_required_columns())

    # Resolve wildcard patterns to concrete chunk columns where possible
    resolved_columns = None
    if chunks_dir.exists():
        try:
            first_chunk = sorted(chunks_dir.glob("chunk_*.parquet"))[0]
            import pyarrow.parquet as pq
            schema = pq.read_schema(first_chunk)
            all_cols = schema.names

            resolved_columns = list(executor._expand_patterns(set(required_cols), all_cols))
            if 'minute' not in resolved_columns:
                resolved_columns.append('minute')

            print(f"resolved {len(resolved_columns)} columns from {len(all_cols)} available.")
            if len(resolved_columns) < 20:
                print(f"Columns: {resolved_columns}")
        except Exception as e:
            logger.warning(
                f"Failed to resolve column patterns from chunks: {e}. "
                "Cache creation may load more columns than needed."
            )
            resolved_columns = None

    resolved_columns_count = len(resolved_columns) if resolved_columns is not None else len(required_cols)

    # Ensure downsampled cache exists
    if skip_cache or not cache_path.exists():
        if chunks_dir.exists():
            print("Creating downsampled cache from chunks...")
            downsampler = StreamingDownsampler(
                max_memory_mb=max(256, effective_budget_mb // 2),
                target_resolution_minutes=target_resolution,
            )
            try:
                downsampler.process_chunks_directory(
                    chunks_dir,
                    cache_path,
                    required_columns=resolved_columns,
                    write_mode="streaming",
                    return_dataframe=False,
                )
                if cache_path.exists():
                    row_count = _read_cache_row_count(cache_path)
                    if row_count is not None:
                        print(f"Cache created: {row_count} rows at {cache_path.name}")
                    else:
                        print(f"Cache created at {cache_path.name}")
                else:
                    logger.error("Failed to create cache: output cache file was not produced.")
                    return
            except Exception as e:
                logger.error(f"Failed to create cache: {e}")
                return
        else:
            print("No chunks directory found. Checking for CSV...")
            csv_path = output_dir / "simulation_history.csv"
            if csv_path.exists():
                print(f"Found CSV at {csv_path}. Warning: CSV loading is memory intensive.")
            else:
                logger.error("No data found (chunks or CSV). Cannot regenerate.")
                return
    else:
        print(f"Using existing cache: {cache_path}")

    vm_now = psutil.virtual_memory()
    available_mb_now = vm_now.available / 1e6
    estimated_df_mb = _estimate_cache_df_mb(cache_path) if cache_path.exists() else 0.0
    enabled_graphs_count = len(executor.catalog.get_enabled())

    selected_mode = choose_execution_mode(
        execution_mode,
        estimated_df_mb,
        available_mb_now,
        resolved_columns_count,
        enabled_graphs_count,
    )

    cache_stride = 1
    if oom_policy == "auto-degrade":
        if estimated_df_mb > 0 and available_mb_now > 0:
            stride, candidates = choose_cache_stride(
                max_extra_downsample=max_extra_downsample,
                estimated_df_mb=estimated_df_mb,
                available_mb=available_mb_now,
            )
            if stride is None:
                print("ERROR: Memory target not reachable even at max extra downsampling.")
                print(
                    f"  Estimated DF={estimated_df_mb:.0f} MB, Available={available_mb_now:.0f} MB, "
                    f"Candidates={candidates}"
                )
                print("  Suggestion: reduce enabled graphs or increase runtime memory.")
                return
            cache_stride = stride
    elif oom_policy == "fail-fast":
        if estimated_df_mb > (0.35 * available_mb_now):
            print("ERROR: Estimated in-memory footprint exceeds fail-fast threshold.")
            print(f"  Estimated DF={estimated_df_mb:.0f} MB, Threshold={(0.35 * available_mb_now):.0f} MB")
            print("  Re-run with --oom-policy auto-degrade or larger memory runtime.")
            return
    elif oom_policy == "skip-heavy":
        # Keep stride=1 and rely on heavy-graph pruning.
        cache_stride = 1
    else:
        print(f"ERROR: Unknown oom policy '{oom_policy}'")
        return

    # Skip-heavy can force safer mode under high pressure.
    if oom_policy == "skip-heavy" and estimated_df_mb > (0.35 * available_mb_now):
        selected_mode = "sequential"

    print("\nExecution planning:")
    print(f"  Estimated in-memory DF footprint: {estimated_df_mb:.0f} MB")
    print(f"  Enabled graphs: {enabled_graphs_count}")
    print(f"  Resolved columns: {resolved_columns_count}")
    print(f"  Mode: {selected_mode}")
    print(f"  Additional cache stride: {cache_stride}x")
    print(f"  OOM policy: {oom_policy}")

    shared_attrs: Dict[str, Any] = {
        "config": viz_config.get("plant_parameters", {}),
        "viz_config": viz_config,
    }

    if oom_policy == "skip-heavy":
        disabled = _apply_skip_heavy_policy(executor)
        if disabled:
            print(f"Skip-heavy policy disabled {len(disabled)} graphs.")

    if selected_mode == "sequential":
        print("Generating graphs in sequential mode...")
        results = executor.execute_sequentially_by_category(
            chunks_dir=chunks_dir if chunks_dir.exists() else None,
            csv_path=output_dir / "simulation_history.csv",
            cache_path=cache_path if cache_path.exists() else None,
            downsample_factor=target_resolution,
            cache_stride=cache_stride,
            timeout_seconds=timeout_seconds,
            df_attrs=shared_attrs,
        )
    else:
        print("Loading data for batched execution...")
        df = executor.load_data(
            chunks_dir=chunks_dir if chunks_dir.exists() else None,
            csv_path=output_dir / "simulation_history.csv",
            cache_path=cache_path if cache_path.exists() else None,
            downsample_factor=cache_stride if cache_path.exists() else target_resolution,
        )

        if df is None or df.empty:
            print("ERROR: No history data found or loaded.")
            return

        _inject_df_attrs(df, shared_attrs)

        print(f"Ready to Plot: {df.shape[0]} rows x {df.shape[1]} columns")
        print(f"Generating graphs with batch size {batch_size}...")
        results = executor.execute_batched(
            df,
            timeout_seconds=timeout_seconds,
            batch_size=batch_size,
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
        '--downscale', type=str, default='hourly',
        choices=['none', 'hourly', 'daily'],
        help='Downscaling mode: none (1-min), hourly (1-min->1-hour, default), daily (1-min->1-day)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=10,
        help='Graphs per batch (default: 10)'
    )
    parser.add_argument(
        '--skip-cache', action='store_true',
        help='Force rebuild of downsampled cache'
    )
    parser.add_argument(
        '--execution-mode',
        type=str,
        default='auto',
        choices=['auto', 'batched', 'sequential'],
        help='Graph execution mode (default: auto)'
    )
    parser.add_argument(
        '--oom-policy',
        type=str,
        default='auto-degrade',
        choices=['auto-degrade', 'fail-fast', 'skip-heavy'],
        help='Policy when memory risk is high (default: auto-degrade)'
    )
    parser.add_argument(
        '--max-extra-downsample',
        type=int,
        default=8,
        help='Maximum extra stride for auto-degrade policy (default: 8)'
    )

    args = parser.parse_args()

    # Map downscale choice to resolution in minutes
    downscale_resolution = {
        'none': 1,
        'hourly': 60,
        'daily': 1440,
    }
    target_resolution = downscale_resolution[args.downscale]

    output_dir = Path(args.output_dir).resolve()
    if not output_dir.exists():
        print(f"ERROR: Directory not found: {output_dir}")
        sys.exit(1)

    try:
        regenerate_graphs_safe(
            output_dir,
            timeout_seconds=args.timeout,
            max_memory_mb=args.max_memory_mb,
            target_resolution=target_resolution,
            batch_size=args.batch_size,
            skip_cache=args.skip_cache,
            execution_mode=args.execution_mode,
            oom_policy=args.oom_policy,
            max_extra_downsample=args.max_extra_downsample,
        )
    except KeyboardInterrupt:
        try:
            rss_mb = psutil.Process().memory_info().rss / 1e6
            print("\nINTERRUPTED: Received interrupt signal.")
            print(f"Last observed process RSS: {rss_mb:.0f} MB")
            print("If this was not manual, runtime resource pressure may have triggered termination.")
        except Exception:
            print("\nINTERRUPTED: Received interrupt signal.")
        sys.exit(130)


if __name__ == "__main__":
    main()
