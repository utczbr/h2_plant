#!/usr/bin/env python3
"""
Regenerate ONLY the net_profit_plotly graph with minimal columns.

This script loads simulation history from history_chunks Parquet and
generates the "Cumulative Net Profit (Interactive)" Plotly HTML.
It loads only: minute + cumulative_h2_kg.

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
from typing import Optional, List

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

NET_PROFIT_TITLE = "Cumulative Net Profit (Interactive)"
# Suppress GraphCatalog import-time logs triggered by plotly_graphs dependencies.
logging.getLogger("h2_plant.visualization.graph_catalog").setLevel(logging.WARNING)

def _resolve_history_chunks(
    output_dir: Path,
    history_dir: Optional[str],
) -> Optional[Path]:
    """
    Resolve history source for graph generation.

    Returns:
        history_chunks directory (Parquet) or None.
    """
    def _check_base(base: Path) -> Optional[Path]:
        chunks_dir = base / "history_chunks"
        if chunks_dir.exists() and list(chunks_dir.glob("chunk_*.parquet")):
            return chunks_dir
        return None

    if history_dir:
        base = Path(history_dir).resolve()
        if base.name.lower() == "history_chunks" and base.exists():
            if list(base.glob("chunk_*.parquet")):
                return base
        if base.is_dir():
            chunks = _check_base(base)
            if chunks:
                return chunks
            chunks = _check_base(base / "history_chunks")
            if chunks:
                return chunks

    # Default: check output_dir and parent (if Economics)
    chunks = _check_base(output_dir)
    if chunks:
        return chunks
    if output_dir.name.lower() == "economics":
        return _check_base(output_dir.parent)
    if output_dir.parent != output_dir:
        return _check_base(output_dir.parent)
    return None


def _load_minimal_history(
    chunks_dir: Path,
    downsample_factor: int,
) -> pd.DataFrame:
    try:
        chunk_files = sorted(
            chunks_dir.glob("chunk_*.parquet"),
            key=lambda p: int(p.stem.split('_')[-1])
        )
    except Exception:
        chunk_files = sorted(chunks_dir.glob("chunk_*.parquet"))

    if not chunk_files:
        raise ValueError(f"No chunk files found in {chunks_dir}")

    try:
        import pyarrow.parquet as pq
        all_cols = pq.read_schema(chunk_files[0]).names
    except OSError as e:
        if getattr(e, "errno", None) == 107:
            raise ValueError(
                "History chunks are on a disconnected mount. "
                "Ensure the drive is mounted or copy history locally."
            ) from e
        raise
    except Exception:
        try:
            df_preview = pd.read_parquet(chunk_files[0], nrows=1)
            all_cols = list(df_preview.columns)
            del df_preview
        except OSError as e:
            if getattr(e, "errno", None) == 107:
                raise ValueError(
                    "History chunks are on a disconnected mount. "
                    "Ensure the drive is mounted or copy history locally."
                ) from e
            raise

    required_cols = ["minute", "cumulative_h2_kg"]
    missing = [c for c in required_cols if c not in all_cols]
    if missing:
        h2_cols = [c for c in all_cols if "h2" in c.lower() and "kg" in c.lower()]
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Available H2 columns: {h2_cols[:20]}"
        )

    dfs = []
    global_idx = 0
    factor = max(1, int(downsample_factor))

    for chunk_file in chunk_files:
        try:
            df_chunk = pd.read_parquet(chunk_file, columns=required_cols)
        except OSError as e:
            if getattr(e, "errno", None) == 107:
                raise ValueError(
                    "History chunks are on a disconnected mount. "
                    "Ensure the drive is mounted or copy history locally."
                ) from e
            raise
        n_rows = len(df_chunk)
        if n_rows == 0:
            continue

        if factor > 1:
            idx = np.arange(global_idx, global_idx + n_rows)
            mask = (idx % factor) == 0
            df_chunk = df_chunk.loc[mask]

        global_idx += n_rows
        if not df_chunk.empty:
            dfs.append(df_chunk)

    if not dfs:
        return pd.DataFrame(columns=required_cols)

    return pd.concat(dfs, ignore_index=True)


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
    from h2_plant.visualization.plotly_graphs import plot_cumulative_net_profit
    graphs_dir = graphs_dir or (output_dir / "graphs")
    graphs_dir.mkdir(parents=True, exist_ok=True)

    chunks_dir = _resolve_history_chunks(output_dir, history_dir)
    if not chunks_dir:
        logger.error("No history_chunks found for Parquet history.")
        return 1

    try:
        df = _load_minimal_history(chunks_dir, downsample_factor)
    except Exception as e:
        logger.error(f"Failed to load minimal history: {e}")
        return 1
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

    if capex is None:
        logger.error("CAPEX not found. Provide capex_report.json or --capex.")
        return 1
    if opex is None:
        logger.error("OPEX not found. Provide opex_report.json or --opex.")
        return 1

    kwargs = {}
    kwargs["capex"] = capex
    kwargs["opex"] = opex
    logger.info(f"Using CAPEX: {capex:,.0f}")
    logger.info(f"Using OPEX: {opex:,.0f}")
    if h2_price_eur_kg is not None:
        kwargs["h2_price_eur_kg"] = h2_price_eur_kg
        logger.info(f"Using H2 price override: {h2_price_eur_kg}")

    kwargs["purification_yield"] = 0.9
    logger.info("Applying purification yield: 0.90")

    fig = plot_cumulative_net_profit(df, **kwargs)

    filename = f"{NET_PROFIT_TITLE.replace(' ', '_').replace('/', '_')}.html"
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
        help="Path to simulation output directory (contains history_chunks)"
    )
    parser.add_argument(
        "--history-dir", type=str, default=None,
        help="Path to history source (simulation output dir or history_chunks)"
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
