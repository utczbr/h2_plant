#!/usr/bin/env python3
"""
Regenerate discounted LCOH report from CAPEX/OPEX reports and history chunks.

Usage:
    python tools/regenerate_lcoh.py scenarios/simulation_output
    python tools/regenerate_lcoh.py scenarios/simulation_output --economics-dir scenarios/Economics
    python tools/regenerate_lcoh.py scenarios/simulation_output --discount-rate 0.08 --project-years 20
    python tools/regenerate_lcoh.py scenarios/simulation_output --history-dir /tmp/history_chunks
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Lazy-loaded economics classes (kept as module globals for test monkeypatching).
LcohCalculator = None
LcohInputs = None
CapexReport = None
OpexReport = None

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _ensure_economics_imported() -> None:
    global LcohCalculator, LcohInputs, CapexReport, OpexReport
    if (
        LcohCalculator is not None
        and LcohInputs is not None
        and CapexReport is not None
        and OpexReport is not None
    ):
        return

    from h2_plant.economics.lcoh_calculator import (  # noqa: WPS433
        LcohCalculator as _LcohCalculator,
        LcohInputs as _LcohInputs,
    )
    from h2_plant.economics.models import CapexReport as _CapexReport  # noqa: WPS433
    from h2_plant.economics.opex_models import OpexReport as _OpexReport  # noqa: WPS433

    if LcohCalculator is None:
        LcohCalculator = _LcohCalculator
    if LcohInputs is None:
        LcohInputs = _LcohInputs
    if CapexReport is None:
        CapexReport = _CapexReport
    if OpexReport is None:
        OpexReport = _OpexReport


def _load_json_report(path: Path, model_cls):
    data = json.loads(path.read_text())
    return model_cls.model_validate(data)


def _resolve_config_defaults(config_dir: Path) -> dict:
    cfg_path = config_dir / "economics_parameters.yaml"
    if not cfg_path.exists():
        return {}
    try:
        return yaml.safe_load(cfg_path.read_text()) or {}
    except Exception as e:
        logger.warning(f"Failed to load economics_parameters.yaml: {e}")
        return {}


def _resolve_history_chunks(sim_dir: Path, history_dir: Optional[str]) -> Optional[Path]:
    def _check_base(base: Path) -> Optional[Path]:
        if not base.exists() or not base.is_dir():
            return None

        if base.name.lower() == "history_chunks":
            if list(base.glob("chunk_*.parquet")):
                return base
            return None

        chunks = base / "history_chunks"
        if chunks.exists() and list(chunks.glob("chunk_*.parquet")):
            return chunks
        return None

    if history_dir:
        return _check_base(Path(history_dir).resolve())
    return _check_base(sim_dir)


def _find_report(paths: list[Path], filename: str) -> Optional[Path]:
    for path in paths:
        candidate = path / filename
        if candidate.exists():
            return candidate
    return None


def _effective_worker_count(raw_workers: int) -> int:
    if raw_workers <= 0:
        return 0
    return raw_workers


def main() -> None:
    total_start = time.perf_counter()
    parser = argparse.ArgumentParser(
        description="Regenerate discounted LCOH report from CAPEX/OPEX and history."
    )
    parser.add_argument(
        "simulation_output_dir", type=str,
        help="Path to simulation output dir containing history_chunks/"
    )
    parser.add_argument(
        "--economics-dir", type=str, default=None,
        help="Directory containing capex_report.json/opex_report.json"
    )
    parser.add_argument(
        "--config-dir", type=str, default=None,
        help="Scenario config directory (default: parent of simulation_output_dir)"
    )
    parser.add_argument(
        "--discount-rate", type=float, default=None,
        help="Override discount rate (e.g., 0.08)"
    )
    parser.add_argument(
        "--project-years", type=int, default=None,
        help="Override project lifetime in years (e.g., 20)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for lcoh_report.json/csv (default: economics_dir)"
    )
    parser.add_argument(
        "--history-dir", type=str, default=None,
        help="Path to history source (simulation output dir or history_chunks)"
    )
    parser.add_argument(
        "--workers", type=int, default=0,
        help="History scan workers (0=auto, 1=serial)."
    )
    parser.add_argument(
        "--max-memory-mb", type=int, default=0,
        help="Optional memory cap for history scanning (0=auto/no explicit cap)."
    )
    parser.add_argument(
        "--history-scan-mode", type=str, default="auto", choices=["auto", "serial", "parallel"],
        help="History scan mode selection."
    )

    args = parser.parse_args()

    if args.workers < 0:
        logger.error("--workers must be >= 0.")
        sys.exit(1)
    if args.max_memory_mb < 0:
        logger.error("--max-memory-mb must be >= 0.")
        sys.exit(1)

    stage_input_start = time.perf_counter()
    sim_dir = Path(args.simulation_output_dir).resolve()
    if not sim_dir.exists():
        logger.error(f"Simulation output directory not found: {sim_dir}")
        sys.exit(1)

    history_chunks = _resolve_history_chunks(sim_dir, args.history_dir)
    if not history_chunks:
        logger.error(
            "history_chunks not found or empty. "
            "Provide --history-dir /path/to/history_chunks."
        )
        sys.exit(1)

    economics_candidates = []
    if args.economics_dir:
        economics_candidates.append(Path(args.economics_dir).resolve())
    economics_candidates.extend([
        sim_dir,
        sim_dir / "Economics",
        sim_dir.parent / "Economics",
    ])
    economics_candidates_unique = []
    for candidate in economics_candidates:
        if candidate not in economics_candidates_unique:
            economics_candidates_unique.append(candidate)
    economics_candidates = economics_candidates_unique

    if not any(path.exists() for path in economics_candidates):
        logger.error("Economics directory not found. Provide --economics-dir.")
        sys.exit(1)

    capex_path = _find_report(economics_candidates, "capex_report.json")
    opex_path = _find_report(economics_candidates, "opex_report.json")
    if capex_path is None:
        logger.error(
            "CAPEX report not found. Checked: %s",
            ", ".join(str(path / "capex_report.json") for path in economics_candidates),
        )
        sys.exit(1)
    if opex_path is None:
        logger.error(
            "OPEX report not found. Checked: %s",
            ", ".join(str(path / "opex_report.json") for path in economics_candidates),
        )
        sys.exit(1)

    economics_dir = capex_path.parent

    config_dir = Path(args.config_dir).resolve() if args.config_dir else sim_dir.parent
    defaults = _resolve_config_defaults(config_dir)
    stage_input_seconds = time.perf_counter() - stage_input_start

    discount_rate = args.discount_rate if args.discount_rate is not None else defaults.get("discount_rate", 0.08)
    project_years = args.project_years if args.project_years is not None else defaults.get("project_lifetime_years", 20)

    stage_models_start = time.perf_counter()
    _ensure_economics_imported()
    capex_report = _load_json_report(capex_path, CapexReport)
    opex_report = _load_json_report(opex_path, OpexReport)
    stage_models_seconds = time.perf_counter() - stage_models_start

    stage_compute_start = time.perf_counter()
    calc = LcohCalculator()
    try:
        report = calc.generate_variants(LcohInputs(
            capex_report=capex_report,
            opex_report=opex_report,
            history_chunks_dir=history_chunks,
            discount_rate=float(discount_rate),
            project_years=int(project_years),
            history_scan_workers=_effective_worker_count(int(args.workers)),
            history_scan_max_memory_mb=(
                int(args.max_memory_mb) if int(args.max_memory_mb) > 0 else None
            ),
            history_scan_mode=str(args.history_scan_mode),
        ))
    except ValueError as e:
        logger.error(f"Failed to generate LCOH report: {e}")
        sys.exit(1)
    stage_compute_seconds = time.perf_counter() - stage_compute_start

    output_dir = Path(args.output_dir).resolve() if args.output_dir else economics_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "lcoh_report.json"
    csv_path = output_dir / "lcoh_report.csv"

    stage_write_start = time.perf_counter()
    json_path.write_text(report.model_dump_json(indent=2))

    # Combined CSV export (low/base/high variants)
    import csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Plant Summary by Variant"])
        writer.writerow([
            "Variant", "discount_rate", "project_lifetime_years",
            "discount_factor_sum", "capex_total", "opex_annual_total",
            "annual_h2_total_kg", "lcoh_total", "lcoh_weighted_plant",
        ])
        for variant in report.variant_order:
            variant_report = report.variants.get(variant)
            if not variant_report:
                continue
            writer.writerow([
                variant,
                variant_report.discount_rate,
                variant_report.project_lifetime_years,
                variant_report.discount_factor_sum,
                variant_report.capex_total,
                variant_report.opex_annual_total,
                variant_report.annual_h2_total_kg,
                variant_report.lcoh_total,
                variant_report.lcoh_weighted_plant,
            ])

        writer.writerow([])
        writer.writerow(["Pathway Metrics by Variant"])
        writer.writerow(["Variant", "Pathway", "Annual_H2_kg", "CAPEX", "OPEX", "LCOH"])
        for variant in report.variant_order:
            variant_report = report.variants.get(variant)
            if not variant_report:
                continue
            for key in ["pem", "soec", "atr"]:
                writer.writerow([
                    variant,
                    key,
                    variant_report.annual_h2_by_pathway.get(key, 0.0),
                    variant_report.capex_by_pathway.get(key, 0.0),
                    variant_report.opex_by_pathway.get(key, 0.0),
                    variant_report.lcoh_by_pathway.get(key, 0.0),
                ])

        writer.writerow([])
        writer.writerow(["LCOH Breakdown by Variant"])
        writer.writerow(["Variant", "Component", "Value"])
        for variant in report.variant_order:
            variant_report = report.variants.get(variant)
            if not variant_report:
                continue
            for k, v in variant_report.lcoh_breakdown.items():
                writer.writerow([variant, k, v])

        if report.warnings:
            writer.writerow([])
            writer.writerow(["Warnings"])
            for w in report.warnings:
                writer.writerow([w])
    stage_write_seconds = time.perf_counter() - stage_write_start

    lcoh_low = report.variants.get("low")
    lcoh_base = report.variants.get("base")
    lcoh_high = report.variants.get("high")
    if lcoh_low and lcoh_base and lcoh_high:
        logger.info(
            "LCOH variants generated (paired): LOW=%s EUR/kg, BASE=%s EUR/kg, HIGH=%s EUR/kg",
            f"{lcoh_low.lcoh_total:,.4f}",
            f"{lcoh_base.lcoh_total:,.4f}",
            f"{lcoh_high.lcoh_total:,.4f}",
        )

    logger.info(f"LCOH report generated: {json_path}")
    logger.info(f"LCOH CSV generated: {csv_path}")
    logger.info(
        "Stage timing (s): input_resolve=%.3f, model_load=%.3f, lcoh_compute=%.3f, report_write=%.3f, total=%.3f",
        stage_input_seconds,
        stage_models_seconds,
        stage_compute_seconds,
        stage_write_seconds,
        time.perf_counter() - total_start,
    )


if __name__ == "__main__":
    main()
