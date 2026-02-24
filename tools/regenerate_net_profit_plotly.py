#!/usr/bin/env python3
"""
Regenerate net_profit_plotly graphs for CAPEX low/base/high scenarios.

This script loads simulation history from history_chunks Parquet,
integrates purified H2 from mixer/PSA flow tags,
integrates grid electricity-sale revenue, and
generates three "Cumulative Net Profit (Interactive)" Plotly HTML files
using CAPEX low/base/high from capex_report.json.
It loads only: minute, purified cumulative_h2_kg (computed),
cumulative_grid_revenue_eur (computed), P_pem, P_soec_actual.

Usage:
    python tools/regenerate_net_profit_plotly.py scenarios/simulation_output
    python tools/regenerate_net_profit_plotly.py scenarios/simulation_output --downscale hourly
    python tools/regenerate_net_profit_plotly.py scenarios/simulation_output --downscale none --history-dir /path/to/history_chunks
    python tools/regenerate_net_profit_plotly.py scenarios/simulation_output --opex-variant low
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

NET_PROFIT_TITLE = "Cumulative Net Profit (Interactive)"
# Suppress GraphCatalog import-time logs triggered by plotly_graphs dependencies.
logging.getLogger("h2_plant.visualization.graph_catalog").setLevel(logging.WARNING)

NET_H2_MIXER_FLOW_COL = "H2_Production_Mixer_outlet_mass_flow_kg_h"
NET_H2_PSA_FLOW_COLS = [
    "PEM_H2_PSA_1_outlet_mass_flow_kg_h",
    "SOEC_H2_PSA_1_outlet_mass_flow_kg_h",
    "ATR_PSA_1_outlet_mass_flow_kg_h",
]
SELL_POWER_COL_CANDIDATES = [
    "P_sold",
    "sell_power_mw",
    "coordinator_sell_power_mw",
]
SELL_PRICE_COL_CANDIDATES = [
    "spot_price",
    "Spot",
]
CAPEX_VARIANT_KEY_MAP = {
    "low": "total_installed_cost_low",
    "base": "total_installed_cost",
    "high": "total_installed_cost_high",
}
CAPEX_VARIANT_SUFFIX = {
    "low": "_Capex_Low",
    "base": "_Capex_Base",
    "high": "_Capex_High",
}
OPEX_VARIANT_KEY_MAP = {
    "base": ("total_opex", "opex", "total_annual_opex", "annual_opex"),
    "low": ("total_opex_low", "opex_low", "total_annual_opex_low", "annual_opex_low"),
    "high": ("total_opex_high", "opex_high", "total_annual_opex_high", "annual_opex_high"),
}
OPEX_VARIANT_SUFFIX = {
    "base": "",
    "low": "_Opex_Low",
    "high": "_Opex_High",
}
DEFAULT_DISCOUNT_RATE = 0.08
DEFAULT_PROJECT_LIFETIME_YEARS = 20


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

    required_cols = ["minute"]
    optional_cols = ["P_pem", "P_soec_actual"]
    missing = [c for c in required_cols if c not in all_cols]
    if missing:
        h2_cols = [c for c in all_cols if "h2" in c.lower()]
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Available H2 columns: {h2_cols[:20]}"
        )

    flow_cols: List[str] = []
    flow_source_desc = ""
    if NET_H2_MIXER_FLOW_COL in all_cols:
        flow_cols = [NET_H2_MIXER_FLOW_COL]
        flow_source_desc = NET_H2_MIXER_FLOW_COL
    else:
        flow_cols = [c for c in NET_H2_PSA_FLOW_COLS if c in all_cols]
        if flow_cols:
            flow_source_desc = " + ".join(flow_cols)

    if not flow_cols:
        raise ValueError(
            "No purified H2 flow tags found for net-profit integration. "
            "Expected H2_Production_Mixer_outlet_mass_flow_kg_h or PSA outlet flow tags."
        )

    logger.info(f"Using purified H2 source columns: {flow_source_desc}")

    sell_power_col = next((c for c in SELL_POWER_COL_CANDIDATES if c in all_cols), None)
    sell_price_col = next((c for c in SELL_PRICE_COL_CANDIDATES if c in all_cols), None)
    if sell_power_col and sell_price_col:
        logger.info(
            "Using electricity-sale columns: "
            f"power={sell_power_col}, price={sell_price_col}"
        )
    else:
        if not sell_power_col:
            logger.warning(
                "Electricity-sale power column not found "
                f"(candidates: {SELL_POWER_COL_CANDIDATES})."
            )
        if not sell_price_col:
            logger.warning(
                "Electricity-sale price column not found "
                f"(candidates: {SELL_PRICE_COL_CANDIDATES})."
            )
        logger.warning("Electricity-sale revenue will be set to zero.")

    present_optional = [c for c in optional_cols if c in all_cols]
    for col in optional_cols:
        if col not in present_optional:
            logger.warning(f"Optional column missing: {col}. Lifecycle spikes will be skipped.")

    cols_to_read = required_cols + flow_cols + present_optional
    if sell_power_col:
        cols_to_read.append(sell_power_col)
    if sell_price_col:
        cols_to_read.append(sell_price_col)

    dfs: List[pd.DataFrame] = []
    global_idx = 0
    factor = max(1, int(downsample_factor))
    running_cumulative_h2 = 0.0
    running_cumulative_grid_revenue = 0.0
    prev_minute: Optional[float] = None
    last_dt_h = 1.0 / 60.0
    last_full_row: Optional[pd.DataFrame] = None

    for chunk_idx, chunk_file in enumerate(chunk_files):
        try:
            df_chunk = pd.read_parquet(chunk_file, columns=cols_to_read)
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

        minute_vals = pd.to_numeric(df_chunk["minute"], errors="coerce").to_numpy(dtype=float)
        flow_vals = np.zeros(n_rows, dtype=float)
        for col in flow_cols:
            flow_vals += pd.to_numeric(df_chunk[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)

        # Integrate step mass with chunk-boundary-aware dt.
        dt_h = np.full(n_rows, np.nan, dtype=float)
        if n_rows > 1:
            local_diff_h = np.diff(minute_vals) / 60.0
            dt_h[1:] = local_diff_h
            valid_local = local_diff_h[np.isfinite(local_diff_h) & (local_diff_h > 0)]
            if valid_local.size:
                last_dt_h = float(np.median(valid_local))

        if np.isfinite(minute_vals[0]) and prev_minute is not None:
            boundary_dt_h = (minute_vals[0] - prev_minute) / 60.0
            if np.isfinite(boundary_dt_h) and boundary_dt_h > 0:
                dt_h[0] = boundary_dt_h

        invalid_dt = ~np.isfinite(dt_h) | (dt_h <= 0)
        if invalid_dt.any():
            dt_h[invalid_dt] = last_dt_h

        step_h2_kg = flow_vals * dt_h
        cumulative_h2 = running_cumulative_h2 + np.cumsum(step_h2_kg)
        running_cumulative_h2 = float(cumulative_h2[-1])

        if sell_power_col and sell_price_col:
            sold_mw = pd.to_numeric(df_chunk[sell_power_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            sold_mw = np.clip(sold_mw, a_min=0.0, a_max=None)
            spot_price = pd.to_numeric(df_chunk[sell_price_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            step_grid_revenue_eur = sold_mw * spot_price * dt_h
        else:
            step_grid_revenue_eur = np.zeros(n_rows, dtype=float)
        cumulative_grid_revenue_eur = running_cumulative_grid_revenue + np.cumsum(step_grid_revenue_eur)
        running_cumulative_grid_revenue = float(cumulative_grid_revenue_eur[-1])

        if np.isfinite(minute_vals[-1]):
            prev_minute = float(minute_vals[-1])

        out_chunk = pd.DataFrame({
            "minute": minute_vals,
            "cumulative_h2_kg": cumulative_h2,
            "cumulative_grid_revenue_eur": cumulative_grid_revenue_eur,
        })
        for col in present_optional:
            out_chunk[col] = pd.to_numeric(df_chunk[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)

        global_idx += n_rows
        last_full_row = out_chunk.iloc[[-1]].copy()

        if factor > 1:
            idx = np.arange(global_idx - n_rows, global_idx)
            mask = (idx % factor) == 0
            # Always keep the last point from the final chunk for endpoint stability.
            if chunk_idx == (len(chunk_files) - 1) and n_rows > 0:
                mask[-1] = True
            out_chunk = out_chunk.loc[mask]

        if not out_chunk.empty:
            dfs.append(out_chunk)

    if not dfs:
        return pd.DataFrame(columns=["minute", "cumulative_h2_kg", "cumulative_grid_revenue_eur"])

    df = pd.concat(dfs, ignore_index=True)
    if factor > 1 and last_full_row is not None:
        if df.empty or float(df["minute"].iloc[-1]) != float(last_full_row["minute"].iloc[0]):
            df = pd.concat([df, last_full_row], ignore_index=True)

    for col in optional_cols:
        if col not in df.columns:
            df[col] = 0.0
    return df


def _find_report(paths: List[Path], filename: str) -> Optional[Path]:
    for base in paths:
        path = base / filename
        if path.exists():
            return path
    return None


def _find_config(paths: List[Path], filename: str) -> Optional[Path]:
    for base in paths:
        path = base / filename
        if path.exists():
            return path
    return None


def _find_topology_path(output_dir: Path) -> Optional[Path]:
    direct = output_dir / "plant_topology.yaml"
    if direct.exists():
        return direct
    parent = output_dir.parent / "plant_topology.yaml"
    if parent.exists():
        return parent
    return None


def _find_economics_parameters_path(output_dir: Path) -> Optional[Path]:
    candidates = [
        output_dir / "economics_parameters.yaml",
        output_dir.parent / "economics_parameters.yaml",
        output_dir.parent / "scenarios" / "economics_parameters.yaml",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _load_financial_horizon(output_dir: Path) -> Tuple[float, int]:
    discount_rate = DEFAULT_DISCOUNT_RATE
    project_lifetime_years = DEFAULT_PROJECT_LIFETIME_YEARS
    config_path = _find_economics_parameters_path(output_dir)

    if config_path is None:
        logger.warning(
            "economics_parameters.yaml not found; using defaults "
            f"(discount_rate={discount_rate}, project_lifetime_years={project_lifetime_years})."
        )
        return discount_rate, project_lifetime_years

    try:
        data = yaml.safe_load(config_path.read_text()) or {}
    except Exception as e:
        logger.warning(
            f"Failed to read economics_parameters.yaml ({config_path}): {e}. "
            f"Using defaults (discount_rate={discount_rate}, project_lifetime_years={project_lifetime_years})."
        )
        return discount_rate, project_lifetime_years

    raw_discount = data.get("discount_rate", discount_rate)
    raw_years = data.get("project_lifetime_years", project_lifetime_years)

    try:
        discount_rate = float(raw_discount)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid discount_rate in economics_parameters.yaml (%r). Using default %s.",
            raw_discount,
            DEFAULT_DISCOUNT_RATE,
        )
        discount_rate = DEFAULT_DISCOUNT_RATE

    if discount_rate <= -1.0:
        logger.warning(
            "discount_rate <= -1.0 in economics_parameters.yaml (%s). Using default %s.",
            discount_rate,
            DEFAULT_DISCOUNT_RATE,
        )
        discount_rate = DEFAULT_DISCOUNT_RATE

    try:
        project_lifetime_years = int(raw_years)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid project_lifetime_years in economics_parameters.yaml (%r). Using default %s.",
            raw_years,
            DEFAULT_PROJECT_LIFETIME_YEARS,
        )
        project_lifetime_years = DEFAULT_PROJECT_LIFETIME_YEARS

    if project_lifetime_years <= 0:
        logger.warning(
            "project_lifetime_years <= 0 in economics_parameters.yaml (%s). Using default %s.",
            project_lifetime_years,
            DEFAULT_PROJECT_LIFETIME_YEARS,
        )
        project_lifetime_years = DEFAULT_PROJECT_LIFETIME_YEARS

    logger.info(
        "Using financial horizon: discount_rate=%s, project_lifetime_years=%s (source: %s)",
        discount_rate,
        project_lifetime_years,
        config_path,
    )
    return discount_rate, project_lifetime_years


def _load_lifecycle_hours(topology_path: Optional[Path]) -> Tuple[Optional[float], Optional[float]]:
    if topology_path is None or not topology_path.exists():
        logger.warning("plant_topology.yaml not found; skipping lifecycle-based OPEX spikes.")
        return None, None
    try:
        data = yaml.safe_load(topology_path.read_text()) or {}
    except Exception as e:
        logger.warning(f"Failed to read topology for lifecycles: {e}")
        return None, None

    nodes = data.get("nodes", [])
    pem_lifecycle = None
    soec_lifecycle = None
    for node in nodes:
        ntype = str(node.get("type", "")).strip()
        params = node.get("params", {}) or {}
        lifecycle = params.get("lifecycle")
        if lifecycle is None:
            continue
        if ntype.upper() == "PEM" and pem_lifecycle is None:
            pem_lifecycle = float(lifecycle)
        if ntype.upper() == "SOEC" and soec_lifecycle is None:
            soec_lifecycle = float(lifecycle)

    if pem_lifecycle is None:
        logger.warning("PEM lifecycle not found in topology.")
    if soec_lifecycle is None:
        logger.warning("SOEC lifecycle not found in topology.")
    return pem_lifecycle, soec_lifecycle


def _load_opex_reserves(opex_path: Optional[Path]) -> Tuple[Optional[float], Optional[float]]:
    if opex_path is None or not opex_path.exists():
        logger.warning("opex_config.yaml not found; skipping reserve-based spikes.")
        return None, None
    try:
        data = yaml.safe_load(opex_path.read_text()) or {}
    except Exception as e:
        logger.warning(f"Failed to read opex_config.yaml: {e}")
        return None, None

    items = data.get("opex_items", []) or []
    pem_reserve = None
    soec_reserve = None
    for item in items:
        name = str(item.get("name", "")).lower()
        price = item.get("price")
        if price is None:
            continue
        if "stack replacement reserve" in name and "pem" in name:
            pem_reserve = float(price)
        if "stack replacement reserve" in name and "soec" in name:
            soec_reserve = float(price)

    if pem_reserve is None:
        logger.warning("PEM stack replacement reserve not found in opex_config.yaml.")
    if soec_reserve is None:
        logger.warning("SOEC stack replacement reserve not found in opex_config.yaml.")
    return pem_reserve, soec_reserve


def _extract_capex_variants(report_path: Path) -> Optional[Dict[str, float]]:
    try:
        data = json.loads(report_path.read_text())
    except Exception as e:
        logger.warning(f"Failed to read CAPEX report {report_path}: {e}")
        return None

    variants: Dict[str, float] = {}
    missing_keys: List[str] = []
    invalid_keys: List[str] = []

    for variant, key in CAPEX_VARIANT_KEY_MAP.items():
        raw_val = data.get(key)
        if raw_val is None:
            missing_keys.append(key)
            continue
        try:
            val = float(raw_val)
        except (TypeError, ValueError):
            invalid_keys.append(key)
            continue
        if val <= 0:
            invalid_keys.append(key)
            continue
        variants[variant] = val

    if missing_keys or invalid_keys:
        problems = []
        if missing_keys:
            problems.append(f"missing keys: {', '.join(missing_keys)}")
        if invalid_keys:
            problems.append(f"invalid/non-positive keys: {', '.join(invalid_keys)}")
        logger.warning(f"CAPEX variants are incomplete in {report_path}: {'; '.join(problems)}")
        return None

    return variants


def _extract_opex(report_path: Path, variant: str = "base") -> Optional[float]:
    try:
        data = json.loads(report_path.read_text())
    except Exception as e:
        logger.warning(f"Failed to read OPEX report {report_path}: {e}")
        return None
    keys = OPEX_VARIANT_KEY_MAP.get(variant, OPEX_VARIANT_KEY_MAP["base"])
    for key in keys:
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
    opex_variant: str = "base",
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
    opex = opex_override
    candidates = [output_dir, output_dir / "Economics"]
    if output_dir.name.lower() == "economics":
        candidates.insert(0, output_dir)
        candidates.insert(1, output_dir.parent)
    if economics_dir:
        candidates.insert(0, Path(economics_dir).resolve())

    capex_path = _find_report(candidates, "capex_report.json")
    if not capex_path:
        logger.error("CAPEX report not found. Expected capex_report.json with low/base/high installed costs.")
        return 1
    capex_variants = _extract_capex_variants(capex_path)
    if not capex_variants:
        logger.error(
            "CAPEX variants missing or invalid in capex_report.json. "
            "Required: total_installed_cost_low, total_installed_cost, total_installed_cost_high."
        )
        return 1
    if capex_override is not None:
        logger.warning("--capex is ignored in multi-scenario mode. Using CAPEX low/base/high from capex_report.json.")

    if opex is None:
        opex_path = _find_report(candidates, "opex_report.json")
        if opex_path:
            opex = _extract_opex(opex_path, variant=opex_variant)

    if opex is None:
        if opex_variant == "base":
            logger.error("OPEX not found. Provide opex_report.json or --opex.")
        else:
            expected_keys = ", ".join(OPEX_VARIANT_KEY_MAP[opex_variant])
            logger.error(
                "Requested OPEX variant '%s' not found in opex_report.json. "
                "Expected one of: %s",
                opex_variant,
                expected_keys,
            )
        return 1

    # Load lifecycle + reserve data for spikes
    topology_path = _find_topology_path(output_dir)
    pem_lifecycle_h, soec_lifecycle_h = _load_lifecycle_hours(topology_path)

    config_candidates = []
    if economics_dir:
        config_candidates.append(Path(economics_dir).resolve())
    config_candidates.extend([output_dir / "Economics", output_dir.parent / "Economics"])
    opex_config_path = _find_config(config_candidates, "opex_config.yaml")
    pem_reserve_pct, soec_reserve_pct = _load_opex_reserves(opex_config_path)
    discount_rate, project_lifetime_years = _load_financial_horizon(output_dir)

    kwargs = {}
    kwargs["opex"] = opex
    kwargs["discount_rate"] = discount_rate
    kwargs["project_lifetime_years"] = project_lifetime_years
    if opex_override is not None:
        logger.info(
            "Using OPEX override (--opex); --opex-variant=%s is ignored for value selection.",
            opex_variant,
        )
    else:
        logger.info(f"Using OPEX variant: {opex_variant.upper()}")
    logger.info(f"Using OPEX: {opex:,.0f}")
    if h2_price_eur_kg is not None:
        kwargs["h2_price_eur_kg"] = h2_price_eur_kg
        logger.info(f"Using H2 price override: {h2_price_eur_kg}")

    if pem_lifecycle_h:
        kwargs["pem_lifecycle_h"] = pem_lifecycle_h
    if soec_lifecycle_h:
        kwargs["soec_lifecycle_h"] = soec_lifecycle_h
    if pem_reserve_pct is not None:
        kwargs["pem_reserve_pct"] = pem_reserve_pct
    if soec_reserve_pct is not None:
        kwargs["soec_reserve_pct"] = soec_reserve_pct

    saved = 0
    base_filename = NET_PROFIT_TITLE.replace(" ", "_").replace("/", "_")
    opex_suffix = OPEX_VARIANT_SUFFIX.get(opex_variant, "")
    for variant in ("low", "base", "high"):
        capex_val = capex_variants[variant]
        kwargs_variant = dict(kwargs)
        kwargs_variant["capex"] = capex_val

        logger.info(f"Generating net profit graph for CAPEX {variant.upper()}: {capex_val:,.0f}")
        fig = plot_cumulative_net_profit(df, **kwargs_variant)

        filename = f"{base_filename}{CAPEX_VARIANT_SUFFIX[variant]}{opex_suffix}.html"
        output_path = graphs_dir / filename
        fig.write_html(
            str(output_path),
            include_plotlyjs="cdn",
            full_html=True,
            config={"displayModeBar": True, "responsive": True, "editable": True},
        )
        saved += 1
        logger.info(f"✓ Net profit graph ({variant}) saved: {output_path}")

    logger.info(f"✓ Net profit graph generation completed: {saved} files")
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
        description=(
            "Regenerate 3 net_profit_plotly graphs (CAPEX low/base/high) by integrating "
            "purified H2 from mixer/PSA flow tags and electricity-sale revenue."
        )
    )
    parser.add_argument(
        "output_dir", type=str,
        help=(
            "Path to simulation output directory (contains history_chunks with mixer/PSA "
            "purified H2 flow tags)"
        )
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
        help="Ignored in multi-scenario mode (kept for compatibility)"
    )
    parser.add_argument(
        "--opex", type=float, default=None,
        help="Override OPEX value (EUR/year)"
    )
    parser.add_argument(
        "--opex-variant", type=str, default="base",
        choices=["base", "low", "high"],
        help="OPEX variant from opex_report.json (default: base)"
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
        opex_variant=args.opex_variant,
        economics_dir=args.economics_dir,
        h2_price_eur_kg=args.h2_price,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
