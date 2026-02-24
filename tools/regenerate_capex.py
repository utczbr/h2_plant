#!/usr/bin/env python3
"""
Regenerate CAPEX Report from Scenario Configuration (No Simulation Required).

Reconstructs the component registry from plant_topology.yaml, then runs
the CAPEX generator using design parameters (or history maxima if available
in simulation_output/).

Usage:
    python tools/regenerate_capex.py scenarios/
    python tools/regenerate_capex.py scenarios/ --output-dir scenarios/simulation_output
    python tools/regenerate_capex.py scenarios/ --capacity-mode design
    python tools/regenerate_capex.py scenarios/ --simulation-hours 8760
    python tools/regenerate_capex.py scenarios/ --history-dir scenarios/simulation_output
    python tools/regenerate_capex.py scenarios/ --no-opex
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple, Sequence

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _infer_output_dir(scenarios_dir: Path, cli_output_dir: Optional[str]) -> Path:
    """Resolve output directory from CLI arg or default."""
    if cli_output_dir:
        return Path(cli_output_dir).resolve()
    candidate = scenarios_dir / "simulation_output"
    if candidate.exists():
        return candidate
    return scenarios_dir / "Economics"


def _resolve_scenario_name(scenarios_dir: Path, override: Optional[str]) -> str:
    """Extract scenario name from topology or fall back to directory name."""
    if override:
        return override
    try:
        import yaml
        topo_path = scenarios_dir / "plant_topology.yaml"
        if topo_path.exists():
            with open(topo_path, "r") as f:
                topo = yaml.safe_load(f) or {}
            name = topo.get("scenario_name")
            if name:
                return str(name)
    except Exception:
        pass
    return scenarios_dir.name


def _resolve_config_dir(scenarios_dir: Path) -> Path:
    """Resolve config directory when user points at simulation_output by mistake."""
    required_files: Sequence[str] = (
        "physics_parameters.yaml",
        "simulation_config.yaml",
        "economics_parameters.yaml",
        "plant_topology.yaml",
    )
    if all((scenarios_dir / f).exists() for f in required_files):
        return scenarios_dir
    parent = scenarios_dir.parent
    if all((parent / f).exists() for f in required_files):
        logger.warning(
            f"Config files not found in {scenarios_dir}; using {parent} instead."
        )
        return parent
    return scenarios_dir


def _resolve_history_source(
    output_dir: Path,
    history_dir: Optional[str],
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Resolve history source for OPEX.
    
    Returns:
        (chunks_dir, csv_path) with preference for history_chunks parquet.
    """
    def _check_base(base: Path) -> Tuple[Optional[Path], Optional[Path]]:
        chunks_dir = base / "history_chunks"
        if chunks_dir.exists():
            if list(chunks_dir.glob("chunk_*.parquet")):
                return chunks_dir, None
        csv_path = base / "simulation_history.csv"
        if csv_path.exists():
            return None, csv_path
        return None, None

    if history_dir:
        base = Path(history_dir).resolve()
        if base.is_file():
            if base.suffix.lower() == ".csv":
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


def _build_registry(scenarios_dir: Path) -> Tuple:
    """
    Reconstruct ComponentRegistry from scenario YAML files.

    Returns:
        (registry, context) tuple
    """
    from h2_plant.config.loader import ConfigLoader
    from h2_plant.core.graph_builder import PlantGraphBuilder
    from h2_plant.core.component_registry import ComponentRegistry

    logger.info("Loading configuration...")
    loader = ConfigLoader(str(scenarios_dir))
    context = loader.load_context()

    logger.info("Building component graph...")
    builder = PlantGraphBuilder(context)
    components = builder.build()

    registry = ComponentRegistry()
    for cid, comp in components.items():
        registry.register(cid, comp)

    logger.info(f"Registry populated: {len(components)} components")
    return registry, context


def regenerate_capex(
    scenarios_dir: Path,
    output_dir: Path,
    capex_config: Optional[Path] = None,
    capacity_mode: Optional[str] = None,
    simulation_name: Optional[str] = None,
    simulation_hours: Optional[int] = None,
    config_dir: Optional[Path] = None,
    generate_opex: bool = True,
    opex_config: Optional[Path] = None,
    history_dir: Optional[Path] = None,
) -> int:
    """
    Generate CAPEX report from scenario configuration files.

    Returns:
        0 on success, 1 on failure.
    """
    from h2_plant.economics import CapexGenerator

    # Build registry from YAML
    try:
        registry, context = _build_registry(config_dir or scenarios_dir)
    except Exception as e:
        logger.error(f"Failed to build registry: {e}")
        return 1

    # Resolve CAPEX config path
    base_config_dir = config_dir or scenarios_dir
    config_path = capex_config or (base_config_dir / "Economics" / "equipment_mappings.yaml")
    if not config_path.exists():
        logger.error(f"Equipment mappings not found: {config_path}")
        return 1

    # Load generator
    logger.info(f"Loading CAPEX config from {config_path.name}...")
    try:
        generator = CapexGenerator.from_yaml(config_path)
    except Exception as e:
        logger.error(f"Failed to load CAPEX config: {e}")
        return 1

    # Apply overrides
    if capacity_mode:
        generator.capacity_mode = capacity_mode
        logger.info(f"Capacity mode override: {capacity_mode}")

    scenario_name = simulation_name or _resolve_scenario_name(
        config_dir or scenarios_dir, simulation_name
    )

    # Generate
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Generating CAPEX report...")

    try:
        report = generator.generate(
            registry=registry,
            monitoring=None,
            output_dir=output_dir,
            simulation_name=scenario_name,
            simulation_hours=simulation_hours,
        )
    except Exception as e:
        logger.error(f"CAPEX generation failed: {e}", exc_info=True)
        return 1

    # Print summary
    print("\n" + "=" * 60)
    print("CAPEX REPORT SUMMARY")
    print("=" * 60)
    print(f"  Scenario:          {scenario_name}")
    print(f"  Capacity Mode:     {generator.capacity_mode}")
    print(f"  Cost Class:        {report.overall_cost_class.value}")
    print(f"  Valid Entries:      {report.entries_with_cost}/{len(report.entries)}")
    if report.entries_out_of_bounds:
        print(f"  Out of Bounds:     {report.entries_out_of_bounds}")
    print("-" * 60)
    print(f"  Equipment Total:   €{report.total_C_BM:>14,.0f}")
    print(f"  Range:             €{report.total_C_BM_low:>14,.0f} - €{report.total_C_BM_high:,.0f}")

    if report.block_summaries:
        print("-" * 60)
        print("  Block Breakdown:")
        for block in report.block_summaries:
            if block.equipment_total > 0:
                print(f"    {block.block_name:<20s} €{block.equipment_total:>12,.0f}  "
                      f"(installed: €{block.total_installed_cost:,.0f}; "
                      f"range: €{block.total_installed_cost_low:,.0f} - €{block.total_installed_cost_high:,.0f})")
        print("-" * 60)
        print(f"  Installation:      €{report.total_installation:>14,.0f}")
        print(f"  Installation Rng:  €{report.total_installation_low:>14,.0f} - €{report.total_installation_high:,.0f}")
        print(f"  Total Installed:   €{report.total_installed_cost:>14,.0f}")
        print(f"  Installed Range:   €{report.total_installed_cost_low:>14,.0f} - €{report.total_installed_cost_high:,.0f}")

    print("=" * 60)
    print(f"  Output:  {output_dir / 'capex_report.json'}")
    print(f"           {output_dir / 'capex_report.csv'}")
    print("=" * 60)

    # Log warnings/errors from entries
    for entry in report.entries:
        if entry.errors:
            for err in entry.errors:
                logger.warning(f"  [{entry.tag}] ERROR: {err}")
        if entry.warnings:
            for warn in entry.warnings:
                logger.info(f"  [{entry.tag}] WARNING: {warn}")

    # Generate OPEX report (optional)
    if generate_opex:
        opex_path = opex_config or (base_config_dir / "Economics" / "opex_config.yaml")
        if not opex_path.exists():
            logger.warning(f"OPEX config not found: {opex_path}. Skipping OPEX.")
        else:
            from h2_plant.economics.opex_generator import OpexGenerator

            opex_generator = OpexGenerator()
            opex_hours = float(simulation_hours) if simulation_hours else 8760.0
            chunks_dir, csv_path = _resolve_history_source(output_dir, str(history_dir) if history_dir else None)

            try:
                if chunks_dir:
                    logger.info(f"Generating OPEX report from Parquet history: {chunks_dir}")
                    opex_report = opex_generator.generate_streaming_parquet(
                        config_path=str(opex_path),
                        chunks_dir=chunks_dir,
                        capex_report=report,
                        output_dir=str(output_dir),
                        simulation_hours=opex_hours,
                    )
                elif csv_path:
                    logger.info(f"Generating OPEX report from CSV history: {csv_path}")
                    opex_report = opex_generator.generate_streaming(
                        config_path=str(opex_path),
                        csv_path=csv_path,
                        capex_report=report,
                        output_dir=str(output_dir),
                        simulation_hours=opex_hours,
                    )
                else:
                    logger.warning("No history data found for OPEX; generating fixed-only report.")
                    opex_report = opex_generator.generate(
                        config_path=str(opex_path),
                        capex_report=report,
                        history_df=None,
                        output_dir=str(output_dir),
                        simulation_hours=opex_hours,
                    )
            except Exception as e:
                logger.error(f"OPEX generation failed: {e}", exc_info=True)
                return 1

            print("\n" + "=" * 60)
            print("OPEX REPORT SUMMARY")
            print("=" * 60)
            print(f"  Scenario:          {opex_report.scenario_name}")
            print(f"  Simulation Hours:  {opex_report.simulation_hours}")
            print(f"  Total OPEX:        €{opex_report.total_opex:>14,.0f}/year")
            if getattr(opex_report, "total_opex_low", None) is not None:
                print(f"  OPEX Low:          €{opex_report.total_opex_low:>14,.0f}/year")
            if getattr(opex_report, "total_opex_high", None) is not None:
                print(f"  OPEX High:         €{opex_report.total_opex_high:>14,.0f}/year")
            print(f"  Variable:          €{opex_report.total_variable_cost:>14,.0f}")
            print(f"  Fixed:             €{opex_report.total_fixed_cost:>14,.0f}")
            print(f"  Maintenance:       €{opex_report.total_maintenance_cost:>14,.0f}")
            if opex_report.annual_h2_production_kg > 0:
                print(f"  H2 Production:     {opex_report.annual_h2_production_kg:>14,.0f} kg/year")
                print(f"  OPEX per kg H2:    €{opex_report.opex_per_kg_h2:>14,.4f}")
            print("-" * 60)
            print(f"  Output:  {output_dir / 'opex_report.json'}")
            print(f"           {output_dir / 'opex_report.csv'}")
            print("=" * 60)

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate CAPEX report from scenario configuration."
    )
    parser.add_argument(
        "scenarios_dir", type=str,
        help="Path to scenarios directory (contains plant_topology.yaml)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for capex_report.json/csv "
             "(default: scenarios_dir/simulation_output or Economics/)"
    )
    parser.add_argument(
        "--capex-config", type=str, default=None,
        help="Path to equipment_mappings.yaml "
             "(default: scenarios_dir/Economics/equipment_mappings.yaml)"
    )
    parser.add_argument(
        "--capacity-mode", type=str, choices=["design", "history"], default=None,
        help="Override capacity extraction mode (default: from YAML config)"
    )
    parser.add_argument(
        "--simulation-name", type=str, default=None,
        help="Override scenario name (default: from plant_topology.yaml)"
    )
    parser.add_argument(
        "--simulation-hours", type=int, default=None,
        help="Simulation duration metadata for the report"
    )
    parser.add_argument(
        "--no-opex", action="store_true",
        help="Skip OPEX report generation"
    )
    parser.add_argument(
        "--opex-config", type=str, default=None,
        help="Path to opex_config.yaml "
             "(default: scenarios_dir/Economics/opex_config.yaml)"
    )
    parser.add_argument(
        "--history-dir", type=str, default=None,
        help="Path to simulation output or history_chunks directory for OPEX history"
    )

    args = parser.parse_args()

    scenarios_dir = Path(args.scenarios_dir).resolve()
    if not scenarios_dir.exists():
        logger.error(f"Directory not found: {scenarios_dir}")
        sys.exit(1)

    config_dir = _resolve_config_dir(scenarios_dir)
    output_dir = _infer_output_dir(scenarios_dir, args.output_dir)
    capex_config = Path(args.capex_config).resolve() if args.capex_config else None
    opex_config = Path(args.opex_config).resolve() if args.opex_config else None

    logger.info("\n" + "=" * 60)
    logger.info("REGENERATE CAPEX REPORT")
    logger.info("=" * 60)
    logger.info(f"Scenarios Dir: {scenarios_dir}")
    if config_dir != scenarios_dir:
        logger.info(f"Config Dir:    {config_dir}")
    logger.info(f"Output Dir:    {output_dir}")
    if capex_config:
        logger.info(f"CAPEX Config:  {capex_config}")
    if opex_config:
        logger.info(f"OPEX Config:   {opex_config}")
    if args.history_dir:
        logger.info(f"History Dir:   {args.history_dir}")
    if args.capacity_mode:
        logger.info(f"Capacity Mode: {args.capacity_mode}")
    logger.info("=" * 60)

    rc = regenerate_capex(
        scenarios_dir=scenarios_dir,
        output_dir=output_dir,
        capex_config=capex_config,
        capacity_mode=args.capacity_mode,
        simulation_name=args.simulation_name,
        simulation_hours=args.simulation_hours,
        config_dir=config_dir,
        generate_opex=not args.no_opex,
        opex_config=opex_config,
        history_dir=Path(args.history_dir).resolve() if args.history_dir else None,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
