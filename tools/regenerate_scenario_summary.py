#!/usr/bin/env python3
"""
Regenerate Scenario Summary from Existing Simulation History (Chunk-Safe).

This script rebuilds scenario_summary.csv using existing history data
from history_chunks (parquet/pequat) or simulation_history.csv.

Optimizations (v2):
- Reuse simulation_history_hourly.parquet cache from graph regeneration (~60x speedup)
- Memory monitoring with adaptive GC (prevents OOM on large simulations)
- Checkpointing for long simulations (resume from interruption)
- Column pre-resolution (eliminates O(N_cols²) per-chunk overhead)

Usage:
    python tools/regenerate_scenario_summary.py scenarios/my_scenario/simulation_output
    python tools/regenerate_scenario_summary.py /path/to/output --scenarios-dir scenarios/my_scenario
    python tools/regenerate_scenario_summary.py /path/to/output --scenario-name Baseline --guaranteed-power-mw 12.5
    python tools/regenerate_scenario_summary.py /path/to/output --append
    python tools/regenerate_scenario_summary.py /path/to/output --no-hourly-cache  # Force raw chunk processing
    python tools/regenerate_scenario_summary.py /path/to/output --resume  # Resume from checkpoint
"""

import argparse
import gc
import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Any

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from h2_plant.reporting.scenario_summary import (  # noqa: E402
    ScenarioSummaryAccumulator,
    SUMMARY_COLUMNS,
    _get_parquet_columns,
    _resolve_summary_columns,
    _expand_patterns,
    generate_scenario_summary_streaming,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Default checkpoint interval (save every N chunks)
DEFAULT_CHECKPOINT_INTERVAL = 100


# =============================================================================
# Memory Monitoring (borrowed from streaming_downsampler.py)
# =============================================================================

class MemoryMonitor:
    """
    Real-time memory monitoring with thresholds.

    Thresholds:
    - Normal: < 60% of max
    - Warning: 60-80% of max
    - Critical: > 80% of max
    """

    def __init__(self, max_memory_mb: int):
        self.max_memory_mb = max_memory_mb
        try:
            import psutil
            self._process = psutil.Process()
            self._enabled = True
        except ImportError:
            self._process = None
            self._enabled = False
            logger.warning("psutil not available; memory monitoring disabled")

    def get_current_mb(self) -> float:
        """Get current process memory in MB."""
        if not self._enabled:
            return 0.0
        return self._process.memory_info().rss / 1e6

    def get_pressure(self) -> float:
        """Get memory pressure as fraction (0.0 to 1.0+)."""
        if self.max_memory_mb <= 0 or not self._enabled:
            return 0.0
        return self.get_current_mb() / self.max_memory_mb

    def is_critical(self) -> bool:
        """Check if memory usage is in critical range."""
        return self.get_pressure() > 0.8

    def log_usage(self, context: str = ""):
        """Log current memory usage."""
        if not self._enabled:
            return
        current = self.get_current_mb()
        pressure = self.get_pressure()
        level = "CRITICAL" if pressure > 0.8 else "WARNING" if pressure > 0.6 else "NORMAL"
        logger.info(f"[{level}] Memory: {current:.0f} MB / {self.max_memory_mb:.0f} MB ({pressure*100:.1f}%) {context}")


# =============================================================================
# Checkpointing
# =============================================================================

@dataclass
class SummaryCheckpoint:
    """Checkpoint state for resumable summary generation."""
    last_processed_chunk: int
    accumulator_state: Dict[str, Any]
    timestamp: str
    scenario_name: str
    guaranteed_power_mw: float

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)

    @classmethod
    def from_json(cls, data: str) -> 'SummaryCheckpoint':
        d = json.loads(data)
        return cls(**d)


def _get_checkpoint_path(output_dir: Path) -> Path:
    return output_dir / ".summary_checkpoint.json"


def _save_checkpoint(
    checkpoint_path: Path,
    chunk_index: int,
    accumulator: ScenarioSummaryAccumulator,
    scenario_name: str,
    guaranteed_power_mw: float
) -> None:
    checkpoint = SummaryCheckpoint(
        last_processed_chunk=chunk_index,
        accumulator_state=accumulator.get_state(),
        timestamp=datetime.now().isoformat(),
        scenario_name=scenario_name,
        guaranteed_power_mw=guaranteed_power_mw,
    )
    checkpoint_path.write_text(checkpoint.to_json())
    logger.info(f"Checkpoint saved at chunk {chunk_index + 1}")


def _load_checkpoint(checkpoint_path: Path) -> Optional[SummaryCheckpoint]:
    if not checkpoint_path.exists():
        return None
    try:
        return SummaryCheckpoint.from_json(checkpoint_path.read_text())
    except Exception as e:
        logger.warning(f"Failed to load checkpoint: {e}")
        return None


# =============================================================================
# Hourly Cache Support
# =============================================================================

def _can_use_hourly_cache(cache_path: Path, required_columns: List[str]) -> bool:
    """Check if hourly cache contains all required summary columns."""
    if not cache_path.exists():
        return False

    try:
        import pyarrow.parquet as pq
        cache_columns = set(pq.read_schema(cache_path).names)
        required_set = set(required_columns)

        missing = required_set - cache_columns
        if missing:
            logger.info(f"Hourly cache missing {len(missing)} summary columns: {list(missing)[:5]}...")
            return False

        return True
    except Exception as e:
        logger.warning(f"Failed to check hourly cache: {e}")
        return False


def _process_hourly_cache(
    cache_path: Path,
    columns: List[str],
    accumulator: ScenarioSummaryAccumulator
) -> None:
    """Process hourly cache as single DataFrame."""
    logger.info(f"Loading hourly cache: {cache_path}")
    try:
        import pyarrow.parquet as pq
        schema_cols = pq.read_schema(cache_path).names
        valid_cols = [c for c in columns if c in schema_cols]

        if not valid_cols:
            logger.warning("No valid columns in hourly cache")
            return

        df = pd.read_parquet(cache_path, columns=valid_cols)
        logger.info(f"Loaded {len(df):,} rows from hourly cache ({len(valid_cols)} columns)")
        accumulator.update(df)
        del df
        gc.collect()
    except Exception as e:
        logger.error(f"Failed to process hourly cache: {e}")
        raise


# =============================================================================
# Helper Functions
# =============================================================================

def _infer_scenarios_dir(output_dir: Path, scenarios_dir: Optional[str]) -> Optional[Path]:
    if scenarios_dir:
        return Path(scenarios_dir).resolve()
    if (output_dir / "plant_topology.yaml").exists():
        return output_dir
    parent = output_dir.parent
    if (parent / "plant_topology.yaml").exists():
        return parent
    return None


def _resolve_scenario_name(scenarios_dir: Optional[Path], fallback: str) -> str:
    if scenarios_dir is None:
        return fallback
    try:
        topo_path = scenarios_dir / "plant_topology.yaml"
        if topo_path.exists():
            import yaml
            with open(topo_path, "r") as f:
                topo_config = yaml.safe_load(f) or {}
            scen_name_candidate = topo_config.get("scenario_name")
            if scen_name_candidate:
                return str(scen_name_candidate)
    except Exception as e:
        logger.debug(f"Scenario name lookup failed: {e}")
    return scenarios_dir.name


def _resolve_guaranteed_power_mw(scenarios_dir: Optional[Path], default: float) -> float:
    if scenarios_dir is None:
        return default
    try:
        eco_path = scenarios_dir / "economics_parameters.yaml"
        if eco_path.exists():
            import yaml
            with open(eco_path, "r") as f:
                eco_config = yaml.safe_load(f) or {}
            return float(eco_config.get("guaranteed_power_mw", default))
    except Exception as e:
        logger.warning(f"Could not load economics config: {e}. Using default guaranteed power.")
    return default


def _sort_chunks(files: Iterable[Path]) -> List[Path]:
    files = list(files)
    try:
        return sorted(files, key=lambda p: int(p.stem.split("_")[-1]))
    except Exception:
        return sorted(files)


def _collect_chunk_files(chunks_dir: Path) -> List[Path]:
    if not chunks_dir.exists():
        return []
    parquet_files = list(chunks_dir.glob("chunk_*.parquet"))
    if parquet_files:
        return _sort_chunks(parquet_files)
    pequat_files = list(chunks_dir.glob("chunk_*.pequat"))
    if pequat_files:
        return _sort_chunks(pequat_files)
    return []


def _iter_parquet_files(files: List[Path], columns: List[str]) -> Iterable[pd.DataFrame]:
    for chunk_file in files:
        if not columns:
            yield pd.read_parquet(chunk_file)
            continue

        try:
            df_chunk = pd.read_parquet(chunk_file, columns=columns)
        except Exception as e:
            logger.debug(f"Chunk {chunk_file.name} missing columns: {e}")
            try:
                import pyarrow.parquet as pq
                schema_cols = pq.read_schema(chunk_file).names
                valid_cols = [c for c in columns if c in schema_cols]
            except Exception:
                valid_cols = [c for c in pd.read_parquet(chunk_file).columns if c in columns]

            if not valid_cols:
                continue
            df_chunk = pd.read_parquet(chunk_file, columns=valid_cols)

        yield df_chunk


# =============================================================================
# Main Processing Function
# =============================================================================

def regenerate_summary(
    output_dir: Path,
    scenario_name: str,
    guaranteed_power_mw: float,
    output_path: Path,
    csv_chunk_rows: int,
    append: bool,
    max_memory_mb: int = 2000,
    use_hourly_cache: bool = True,
    checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL,
    resume: bool = False
) -> int:
    chunks_dir = output_dir / "history_chunks"
    csv_path = output_dir / "simulation_history.csv"
    hourly_cache_path = output_dir / "simulation_history_hourly.parquet"
    checkpoint_path = _get_checkpoint_path(output_dir)

    if output_path.exists() and not append:
        output_path.unlink()

    # Initialize memory monitor
    memory_monitor = MemoryMonitor(max_memory_mb)
    memory_monitor.log_usage("start")

    # Try hourly cache first (biggest speedup)
    if use_hourly_cache and hourly_cache_path.exists():
        # Get columns we need for summary
        cache_columns = _get_parquet_columns(hourly_cache_path)
        columns_to_load = _resolve_summary_columns(cache_columns)

        if _can_use_hourly_cache(hourly_cache_path, columns_to_load):
            logger.info("Using hourly cache for ~60x speedup")
            accumulator = ScenarioSummaryAccumulator(guaranteed_power_mw)
            accumulator.configure_columns(cache_columns)

            _process_hourly_cache(hourly_cache_path, columns_to_load, accumulator)
            accumulator.finalize(scenario_name, output_path)

            memory_monitor.log_usage("complete (hourly cache)")
            return 0
        else:
            logger.info("Hourly cache not compatible; falling back to chunks")

    # Process chunk files
    chunk_files = _collect_chunk_files(chunks_dir)
    if chunk_files:
        logger.info(
            f"Scenario summary streaming from {len(chunk_files)} chunk files "
            f"({chunk_files[0].suffix.lstrip('.')})"
        )
        available_columns = _get_parquet_columns(chunk_files[0])
        columns_to_load = _resolve_summary_columns(available_columns)

        if not columns_to_load:
            logger.warning("No matching columns found in chunked history. Skipping data scan.")
            ScenarioSummaryAccumulator(guaranteed_power_mw).finalize(scenario_name, output_path)
            return 0

        # Initialize accumulator
        accumulator = ScenarioSummaryAccumulator(guaranteed_power_mw)
        accumulator.configure_columns(available_columns)  # Pre-resolve columns

        start_chunk = 0

        # Try to resume from checkpoint
        if resume:
            checkpoint = _load_checkpoint(checkpoint_path)
            if checkpoint is not None:
                # Validate checkpoint matches current run
                if (checkpoint.scenario_name == scenario_name and
                    checkpoint.guaranteed_power_mw == guaranteed_power_mw):
                    start_chunk = checkpoint.last_processed_chunk + 1
                    accumulator.load_state(checkpoint.accumulator_state)
                    logger.info(f"Resuming from checkpoint: chunk {start_chunk + 1}/{len(chunk_files)}")
                else:
                    logger.warning("Checkpoint parameters mismatch; starting fresh")

        # Process chunks
        chunk_iterator = enumerate(_iter_parquet_files(chunk_files[start_chunk:], columns_to_load), start_chunk)

        for i, df_chunk in chunk_iterator:
            # Memory pressure check
            if memory_monitor.is_critical():
                logger.warning(f"Memory pressure at chunk {i + 1}. Forcing GC.")
                gc.collect()

            accumulator.update(df_chunk)
            del df_chunk

            # Adaptive GC frequency based on memory pressure
            pressure = memory_monitor.get_pressure()
            gc_interval = 2 if pressure > 0.6 else (4 if pressure > 0.4 else 8)
            if (i + 1) % gc_interval == 0:
                gc.collect()

            # Checkpoint periodically
            if checkpoint_interval > 0 and (i + 1) % checkpoint_interval == 0:
                _save_checkpoint(checkpoint_path, i, accumulator, scenario_name, guaranteed_power_mw)
                memory_monitor.log_usage(f"after chunk {i + 1}/{len(chunk_files)}")

        accumulator.finalize(scenario_name, output_path)

        # Clean up checkpoint on success
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.info("Checkpoint removed (processing complete)")

        memory_monitor.log_usage("complete (chunks)")
        return 0

    # Fall back to CSV
    if csv_path.exists():
        logger.info("Processing from CSV (no chunks or hourly cache found)")
        generate_scenario_summary_streaming(
            scenario_name=scenario_name,
            output_path=output_path,
            guaranteed_power_mw=guaranteed_power_mw,
            history=None,
            chunks_dir=None,
            csv_path=csv_path,
            csv_chunk_rows=csv_chunk_rows,
        )
        memory_monitor.log_usage("complete (CSV)")
        return 0

    logger.error("No history data found (history_chunks, hourly cache, or simulation_history.csv).")
    return 1


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate scenario summary from existing simulation history."
    )
    parser.add_argument(
        "output_dir", type=str,
        help="Path to simulation output directory"
    )
    parser.add_argument(
        "--scenarios-dir", type=str, default=None,
        help="Path to scenarios directory (for scenario name and economics parameters)"
    )
    parser.add_argument(
        "--scenario-name", type=str, default=None,
        help="Override scenario name (default: inferred from scenarios directory)"
    )
    parser.add_argument(
        "--guaranteed-power-mw", type=float, default=None,
        help="Override guaranteed power in MW (default: economics_parameters.yaml or 10.0)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output CSV path (default: <output_dir>/scenario_summary.csv)"
    )
    parser.add_argument(
        "--csv-chunk-rows", type=int, default=250_000,
        help="CSV chunk size for streaming (default: 250000)"
    )
    parser.add_argument(
        "--append", action="store_true",
        help="Append to existing scenario_summary.csv instead of overwriting"
    )
    # New optimization arguments
    parser.add_argument(
        "--max-memory-mb", type=int, default=2000,
        help="Maximum memory budget in MB (default: 2000)"
    )
    parser.add_argument(
        "--no-hourly-cache", action="store_true",
        help="Force processing from raw chunks (ignore hourly cache)"
    )
    parser.add_argument(
        "--checkpoint-interval", type=int, default=DEFAULT_CHECKPOINT_INTERVAL,
        help=f"Save checkpoint every N chunks (default: {DEFAULT_CHECKPOINT_INTERVAL}, 0=disabled)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing checkpoint if available"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    if not output_dir.exists():
        logger.error(f"Directory not found: {output_dir}")
        sys.exit(1)

    scenarios_dir = _infer_scenarios_dir(output_dir, args.scenarios_dir)
    fallback_name = output_dir.parent.name if output_dir.parent else output_dir.name
    scenario_name = args.scenario_name or _resolve_scenario_name(scenarios_dir, fallback_name)

    guaranteed_power_mw = (
        float(args.guaranteed_power_mw)
        if args.guaranteed_power_mw is not None
        else _resolve_guaranteed_power_mw(scenarios_dir, default=10.0)
    )

    output_path = Path(args.output).resolve() if args.output else (output_dir / "scenario_summary.csv")

    logger.info("\n" + "=" * 60)
    logger.info("REGENERATE SCENARIO SUMMARY (v2 - Optimized)")
    logger.info("=" * 60)
    logger.info(f"Output Dir: {output_dir}")
    if scenarios_dir:
        logger.info(f"Scenarios Dir: {scenarios_dir}")
    logger.info(f"Scenario Name: {scenario_name}")
    logger.info(f"Guaranteed Power: {guaranteed_power_mw} MW")
    logger.info(f"Output CSV: {output_path}")
    logger.info(f"Memory Budget: {args.max_memory_mb} MB")
    logger.info(f"Use Hourly Cache: {not args.no_hourly_cache}")
    logger.info(f"Checkpoint Interval: {args.checkpoint_interval} chunks")
    logger.info(f"Resume: {args.resume}")
    logger.info("=" * 60)

    rc = regenerate_summary(
        output_dir=output_dir,
        scenario_name=scenario_name,
        guaranteed_power_mw=guaranteed_power_mw,
        output_path=output_path,
        csv_chunk_rows=args.csv_chunk_rows,
        append=args.append,
        max_memory_mb=args.max_memory_mb,
        use_hourly_cache=not args.no_hourly_cache,
        checkpoint_interval=args.checkpoint_interval,
        resume=args.resume,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
