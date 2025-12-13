"""
High-Level Simulation Runner Utilities.

This module provides convenient wrapper functions for common simulation
scenarios, abstracting the setup and execution details.

Entry Points:
    - `run_simulation_from_config()`: Single config file execution.
    - `run_scenario_comparison()`: Multi-scenario comparison runs.
    - `main()`: CLI entry point for command-line execution.

Workflow:
    1. Load plant configuration from YAML/JSON.
    2. Build component graph via PlantBuilder.
    3. Create and initialize SimulationEngine.
    4. Execute simulation loop.
    5. Export metrics and results.
"""

from pathlib import Path
from typing import Optional, Dict, Any
import logging
import argparse
import json

from h2_plant.config.plant_builder import PlantBuilder
from h2_plant.simulation.engine import SimulationEngine
from h2_plant.config.plant_config import SimulationConfig

logger = logging.getLogger(__name__)


def run_simulation_from_config(
    config_path: Path | str,
    output_dir: Optional[Path] = None,
    resume_from: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run complete simulation from configuration file.

    One-line simulation execution that handles all setup steps:
    configuration loading, plant building, engine creation,
    execution, and metric export.

    Args:
        config_path (Path | str): Path to plant configuration YAML/JSON.
        output_dir (Path, optional): Directory for outputs.
            Default: './simulation_output'.
        resume_from (str, optional): Path to checkpoint file for resuming
            interrupted simulation.

    Returns:
        Dict[str, Any]: Simulation results including final states and metrics.

    Example:
        >>> results = run_simulation_from_config("configs/plant_baseline.yaml")
        >>> print(f"Total H2: {results['metrics']['total_production_kg']:.1f} kg")
    """
    logging.info(f"Running simulation from config: {config_path}")

    plant = PlantBuilder.from_file(config_path)
    registry = plant.registry

    engine = SimulationEngine(
        registry=registry,
        config=plant.config.simulation,
        output_dir=output_dir or Path("simulation_output"),
        topology=plant.config.topology,
        indexed_topology=plant.config.indexed_topology
    )

    results = engine.run(resume_from_checkpoint=resume_from)

    engine.monitoring.export_timeseries()
    engine.monitoring.export_summary()

    return results


def run_scenario_comparison(
    config_paths: list[Path | str],
    output_dir: Optional[Path] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Run multiple scenarios and generate comparison report.

    Executes each configuration in sequence, collecting results
    for side-by-side comparison of key metrics.

    Args:
        config_paths (list): List of configuration file paths.
        output_dir (Path, optional): Base directory for outputs.
            Each scenario gets a subdirectory.

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary mapping config name to results.

    Example:
        >>> scenarios = [
        ...     "configs/plant_baseline.yaml",
        ...     "configs/plant_grid_only.yaml",
        ...     "configs/plant_pilot.yaml"
        ... ]
        >>> results = run_scenario_comparison(scenarios)
    """
    output_dir = output_dir or Path("scenario_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for config_path in config_paths:
        config_name = Path(config_path).stem
        scenario_dir = output_dir / config_name

        logging.info(f"\n{'='*60}")
        logging.info(f"Running scenario: {config_name}")
        logging.info(f"{'='*60}\n")

        results = run_simulation_from_config(
            config_path=config_path,
            output_dir=scenario_dir
        )

        all_results[config_name] = results

    _generate_comparison_report(all_results, output_dir)

    return all_results


def _generate_comparison_report(
    results: Dict[str, Dict[str, Any]],
    output_dir: Path
) -> None:
    """
    Generate JSON comparison report across scenarios.

    Extracts key metrics from each scenario for side-by-side comparison.

    Args:
        results (Dict): Scenario results dictionary.
        output_dir (Path): Output directory for report file.
    """
    comparison = {}

    for scenario_name, scenario_results in results.items():
        metrics = scenario_results.get('metrics', {})
        comparison[scenario_name] = {
            'total_production_kg': metrics.get('total_production_kg', 0.0),
            'total_cost': metrics.get('total_cost', 0.0),
            'cost_per_kg': metrics.get('average_cost_per_kg', 0.0),
            'demand_fulfillment': metrics.get('demand_fulfillment_rate', 0.0)
        }

    report_path = output_dir / "scenario_comparison.json"
    with open(report_path, 'w') as f:
        json.dump(comparison, f, indent=2)

    logging.info(f"\nScenario comparison saved to: {report_path}")


def main():
    """
    CLI entry point for command-line simulation execution.

    Usage:
        python -m h2_plant.simulation.runner config.yaml --output ./results
    """
    parser = argparse.ArgumentParser(description="Run a hydrogen plant simulation.")
    parser.add_argument("config_file", type=str, help="Path to the plant configuration YAML file.")
    parser.add_argument("--output", type=str, default="simulation_output", help="Directory for simulation outputs.")
    parser.add_argument("--resume", type=str, default=None, help="Path to a checkpoint file to resume from.")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    run_simulation_from_config(
        config_path=args.config_file,
        output_dir=Path(args.output),
        resume_from=args.resume
    )


if __name__ == "__main__":
    main()
