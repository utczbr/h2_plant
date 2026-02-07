#!/usr/bin/env python3
"""
Regenerate ONLY the net_profit_plotly graph with filtered columns.

This script loads simulation history (preferring history_chunks Parquet)
and generates the "Cumulative Net Profit (Interactive)" Plotly HTML.

Usage:
    python tools/regenerate_net_profit_plotly.py scenarios/simulation_output
    python tools/regenerate_net_profit_plotly.py scenarios/simulation_output --downscale hourly
    python tools/regenerate_net_profit_plotly.py scenarios/simulation_output --downscale none --history-dir /path/to/history_chunks
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple, List

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _resolve_history_source(
    output_dir: Path,
    history_dir: Optional[str],
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Resolve history source for graph generation.

    Returns:
        (chunks_dir, csv_path) preference order:
        1) history_chunks (Parquet)
        2) simulation_history.csv
    """
    def _check_base(base: Path) -> Tuple[Optional[Path], Optional[Path]]:
        chunks_dir = base / "history_chunks"
        if chunks_dir.exists() and list(chunks_dir.glob("chunk_*.parquet")):
            return chunks_dir, None
        csv_path = base / "simulation_history.csv"
        if csv_path.exists():
            return None, csv_path
        return None, None

    if history_dir:
        base = Path(history_dir).resolve()
        if base.is_file() and base.suffix.lower() == ".csv":
            return None, base
        if base.name.lower() == "history_chunks" and base.exists():
            if list(base.glob("chunk_*.parquet")):
                return base, None
        if base.is_dir():
            chunks, csv = _check_base(base)
            if chunks or csv:
                return chunks, csv
            chunks, csv = _check_base(base / "history_chunks")
            if chunks or csv:
                return chunks, csv

    # Default: check output_dir and parent (if Economics)
    chunks, csv = _check_base(output_dir)
    if chunks or csv:
        return chunks, csv
    if output_dir.name.lower() == "economics":
        return _check_base(output_dir.parent)
    if output_dir.parent != output_dir:
        return _check_base(output_dir.parent)
    return None, None


def _find_report(paths: List[Path], filename: str) -> Optional[Path]:
    for base in paths:
        path = base / filename
        if path.exists():
            return path
    return None


def _extract_capex(report_path: Path) -> Optional[float]:
    try:
        data = json.loads(report_path.read_text())
    except Exception as e:
        logger.warning(f"Failed to read CAPEX report {report_path}: {e}")
        return None
    for key in ("total_installed_cost", "total_C_BM", "total_c_bm", "total_capex", "capex"):
        val = data.get(key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    return None


def _extract_opex(report_path: Path) -> Optional[float]:
    try:
        data = json.loads(report_path.read_text())
    except Exception as e:
        logger.warning(f"Failed to read OPEX report {report_path}: {e}")
        return None
    for key in ("total_opex", "opex", "total_annual_opex", "annual_opex"):
        val = data.get(key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    return None


def regenerate_net_profit_plotly(
    output_dir: Path,
    history_dir: Optional[str] = None,
    graphs_dir: Optional[Path] = None,
    downsample_factor: int = 60,
    capex_override: Optional[float] = None,
    opex_override: Optional[float] = None,
    economics_dir: Optional[str] = None,
    h2_price_eur_kg: Optional[float] = None,
) -> int:
    from h2_plant.visualization.graph_catalog import GraphCatalog
    from h2_plant.visualization.unified_executor import UnifiedGraphExecutor
    from h2_plant.visualization.plotly_graphs import plot_cumulative_net_profit

    graphs_dir = graphs_dir or (output_dir / "graphs")
    graphs_dir.mkdir(parents=True, exist_ok=True)

    # Configure catalog for a single graph
    catalog = GraphCatalog()
    catalog.disable_all()
    catalog.enable("net_profit_plotly")
    meta = catalog.get("net_profit_plotly")
    if meta is None:
        logger.error("Graph 'net_profit_plotly' not found in catalog.")
        return 1

    executor = UnifiedGraphExecutor(catalog, graphs_dir)

    chunks_dir, csv_path = _resolve_history_source(output_dir, history_dir)
    if not chunks_dir and not csv_path:
        logger.error("No history data found (history_chunks or simulation_history.csv).")
        return 1

    df = executor.load_data(
        chunks_dir=chunks_dir,
        csv_path=csv_path,
        downsample_factor=downsample_factor,
    )
    if df.empty:
        logger.error("Loaded history is empty. Aborting.")
        return 1

    # Resolve CAPEX/OPEX from reports unless overridden
    capex = capex_override
    opex = opex_override
    if capex is None or opex is None:
        candidates = [output_dir, output_dir / "Economics"]
        if output_dir.name.lower() == "economics":
            candidates.insert(0, output_dir)
            candidates.insert(1, output_dir.parent)
        if economics_dir:
            candidates.insert(0, Path(economics_dir).resolve())

        if capex is None:
            capex_path = _find_report(candidates, "capex_report.json")
            if capex_path:
                capex = _extract_capex(capex_path)
        if opex is None:
            opex_path = _find_report(candidates, "opex_report.json")
            if opex_path:
                opex = _extract_opex(opex_path)

    kwargs = {}
    if capex is not None:
        kwargs["capex"] = capex
        logger.info(f"Using CAPEX override: {capex:,.0f}")
    if opex is not None:
        kwargs["opex"] = opex
        logger.info(f"Using OPEX override: {opex:,.0f}")
    if h2_price_eur_kg is not None:
        kwargs["h2_price_eur_kg"] = h2_price_eur_kg
        logger.info(f"Using H2 price override: {h2_price_eur_kg}")
    if capex is None and opex is None:
        logger.warning("CAPEX/OPEX not found. The graph may be empty without economics data.")

    fig = plot_cumulative_net_profit(df, **kwargs)

    filename = f"{meta.title.replace(' ', '_').replace('/', '_')}.html"
    output_path = graphs_dir / filename
    fig.write_html(
        str(output_path),
        include_plotlyjs="cdn",
        full_html=True,
        config={"displayModeBar": True, "responsive": True, "editable": True},
    )

    logger.info(f"✓ Net profit graph saved: {output_path}")
    return 0


def _parse_downscale(value: str) -> int:
    value = value.lower().strip()
    if value == "none":
        return 1
    if value == "hourly":
        return 60
    if value == "daily":
        return 1440
    raise ValueError(f"Unknown downscale option: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate only the net_profit_plotly graph with filtered columns."
    )
    parser.add_argument(
        "output_dir", type=str,
        help="Path to simulation output directory (contains history_chunks or simulation_history.csv)"
    )
    parser.add_argument(
        "--history-dir", type=str, default=None,
        help="Path to history source (simulation output dir, history_chunks, or CSV file)"
    )
    parser.add_argument(
        "--graphs-dir", type=str, default=None,
        help="Output directory for graphs (default: <output_dir>/graphs)"
    )
    parser.add_argument(
        "--downscale", type=str, default="hourly",
        choices=["none", "hourly", "daily"],
        help="Downscale factor for history loading"
    )
    parser.add_argument(
        "--capex", type=float, default=None,
        help="Override CAPEX value (EUR)"
    )
    parser.add_argument(
        "--opex", type=float, default=None,
        help="Override OPEX value (EUR/year)"
    )
    parser.add_argument(
        "--economics-dir", type=str, default=None,
        help="Directory containing capex_report.json/opex_report.json"
    )
    parser.add_argument(
        "--h2-price", type=float, default=None,
        help="Override H2 price (EUR/kg)"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    if not output_dir.exists():
        logger.error(f"Output directory not found: {output_dir}")
        sys.exit(1)

    graphs_dir = Path(args.graphs_dir).resolve() if args.graphs_dir else None
    downsample_factor = _parse_downscale(args.downscale)

    rc = regenerate_net_profit_plotly(
        output_dir=output_dir,
        history_dir=args.history_dir,
        graphs_dir=graphs_dir,
        downsample_factor=downsample_factor,
        capex_override=args.capex,
        opex_override=args.opex,
        economics_dir=args.economics_dir,
        h2_price_eur_kg=args.h2_price,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
