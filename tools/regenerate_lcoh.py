#!/usr/bin/env python3
"""
Regenerate discounted LCOH report from CAPEX/OPEX reports and history chunks.

Usage:
    python tools/regenerate_lcoh.py scenarios/simulation_output
    python tools/regenerate_lcoh.py scenarios/simulation_output --economics-dir scenarios/Economics
    python tools/regenerate_lcoh.py scenarios/simulation_output --discount-rate 0.08 --project-years 20
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from h2_plant.economics.lcoh_calculator import LcohCalculator, LcohInputs  # noqa: E402
from h2_plant.economics.models import CapexReport  # noqa: E402
from h2_plant.economics.opex_models import OpexReport  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


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


def main() -> None:
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

    args = parser.parse_args()

    sim_dir = Path(args.simulation_output_dir).resolve()
    if not sim_dir.exists():
        logger.error(f"Simulation output directory not found: {sim_dir}")
        sys.exit(1)

    history_chunks = sim_dir / "history_chunks"
    if not history_chunks.exists():
        logger.error(f"history_chunks not found in: {sim_dir}")
        sys.exit(1)

    economics_dir = Path(args.economics_dir).resolve() if args.economics_dir else (sim_dir / "Economics")
    if not economics_dir.exists():
        economics_dir = sim_dir.parent / "Economics"
    if not economics_dir.exists():
        logger.error("Economics directory not found. Provide --economics-dir.")
        sys.exit(1)

    capex_path = economics_dir / "capex_report.json"
    opex_path = economics_dir / "opex_report.json"
    if not capex_path.exists():
        logger.error(f"CAPEX report not found: {capex_path}")
        sys.exit(1)
    if not opex_path.exists():
        logger.error(f"OPEX report not found: {opex_path}")
        sys.exit(1)

    config_dir = Path(args.config_dir).resolve() if args.config_dir else sim_dir.parent
    defaults = _resolve_config_defaults(config_dir)

    discount_rate = args.discount_rate if args.discount_rate is not None else defaults.get("discount_rate", 0.08)
    project_years = args.project_years if args.project_years is not None else defaults.get("project_lifetime_years", 20)

    capex_report = _load_json_report(capex_path, CapexReport)
    opex_report = _load_json_report(opex_path, OpexReport)

    calc = LcohCalculator()
    report = calc.generate(LcohInputs(
        capex_report=capex_report,
        opex_report=opex_report,
        history_chunks_dir=history_chunks,
        discount_rate=float(discount_rate),
        project_years=int(project_years),
    ))

    output_dir = Path(args.output_dir).resolve() if args.output_dir else economics_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "lcoh_report.json"
    csv_path = output_dir / "lcoh_report.csv"

    json_path.write_text(report.model_dump_json(indent=2))

    # Simple CSV export
    import csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["discount_rate", report.discount_rate])
        writer.writerow(["project_lifetime_years", report.project_lifetime_years])
        writer.writerow(["discount_factor_sum", report.discount_factor_sum])
        writer.writerow(["capex_total", report.capex_total])
        writer.writerow(["opex_annual_total", report.opex_annual_total])
        writer.writerow(["annual_h2_total_kg", report.annual_h2_total_kg])
        writer.writerow(["lcoh_total", report.lcoh_total])
        writer.writerow(["lcoh_weighted_plant", report.lcoh_weighted_plant])
        writer.writerow([])
        writer.writerow(["Pathway", "Annual_H2_kg", "CAPEX", "OPEX", "LCOH"])
        for key in ["pem", "soec", "atr"]:
            writer.writerow([
                key,
                report.annual_h2_by_pathway.get(key, 0.0),
                report.capex_by_pathway.get(key, 0.0),
                report.opex_by_pathway.get(key, 0.0),
                report.lcoh_by_pathway.get(key, 0.0),
            ])

        writer.writerow([])
        writer.writerow(["LCOH Breakdown"])
        for k, v in report.lcoh_breakdown.items():
            writer.writerow([k, v])

        if report.warnings:
            writer.writerow([])
            writer.writerow(["Warnings"])
            for w in report.warnings:
                writer.writerow([w])

    logger.info(f"LCOH report generated: {json_path}")
    logger.info(f"LCOH CSV generated: {csv_path}")


if __name__ == "__main__":
    main()
