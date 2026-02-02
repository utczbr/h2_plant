#!/usr/bin/env python3
"""
Regenerate Scenario Summary from Existing Simulation History (Chunk-Safe).

This script rebuilds scenario_summary.csv using existing history data
from history_chunks (parquet/pequat) or simulation_history.csv.

Usage:
    python tools/regenerate_scenario_summary.py scenarios/my_scenario/simulation_output
    python tools/regenerate_scenario_summary.py /path/to/output --scenarios-dir scenarios/my_scenario
    python tools/regenerate_scenario_summary.py /path/to/output --scenario-name Baseline --guaranteed-power-mw 12.5
    python tools/regenerate_scenario_summary.py /path/to/output --append
"""

import argparse
import gc
import logging
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from h2_plant.reporting.scenario_summary import (  # noqa: E402
    ScenarioSummaryAccumulator,
    _get_parquet_columns,
    _resolve_summary_columns,
    generate_scenario_summary_streaming,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


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


def regenerate_summary(
    output_dir: Path,
    scenario_name: str,
    guaranteed_power_mw: float,
    output_path: Path,
    csv_chunk_rows: int,
    append: bool
) -> int:
    chunks_dir = output_dir / "history_chunks"
    csv_path = output_dir / "simulation_history.csv"

    if output_path.exists() and not append:
        output_path.unlink()

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

        accumulator = ScenarioSummaryAccumulator(guaranteed_power_mw)
        for i, df_chunk in enumerate(_iter_parquet_files(chunk_files, columns_to_load), 1):
            accumulator.update(df_chunk)
            del df_chunk
            if i % 4 == 0:
                gc.collect()

        accumulator.finalize(scenario_name, output_path)
        return 0

    if csv_path.exists():
        generate_scenario_summary_streaming(
            scenario_name=scenario_name,
            output_path=output_path,
            guaranteed_power_mw=guaranteed_power_mw,
            history=None,
            chunks_dir=None,
            csv_path=csv_path,
            csv_chunk_rows=csv_chunk_rows,
        )
        return 0

    logger.error("No history data found (history_chunks or simulation_history.csv).")
    return 1


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
    logger.info("REGENERATE SCENARIO SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Output Dir: {output_dir}")
    if scenarios_dir:
        logger.info(f"Scenarios Dir: {scenarios_dir}")
    logger.info(f"Scenario Name: {scenario_name}")
    logger.info(f"Guaranteed Power: {guaranteed_power_mw} MW")
    logger.info(f"Output CSV: {output_path}")
    logger.info("=" * 60)

    rc = regenerate_summary(
        output_dir=output_dir,
        scenario_name=scenario_name,
        guaranteed_power_mw=guaranteed_power_mw,
        output_path=output_path,
        csv_chunk_rows=args.csv_chunk_rows,
        append=args.append,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
