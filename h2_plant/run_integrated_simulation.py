"""
Integrated Dispatch Simulation Runner.

This module provides a unified entry point for running simulations using
the SimulationEngine + DispatchStrategy architecture.

Architecture:
    - **SimulationEngine**: Handles component lifecycle (initialize, step, checkpoint).
    - **HybridArbitrageEngineStrategy**: Handles dispatch decisions and history recording.
    - **Components**: Stepped exactly once per timestep via registry.

Key Features:
    - Pre-allocated NumPy arrays for history (A3 optimization).
    - Automatic topology detection (SOEC-only vs Hybrid).
    - Separation of dispatch setpoints and physics execution.

Usage:
    python -m h2_plant.run_integrated_simulation configs/scenario_dir

    Or programmatically:
        from h2_plant.run_integrated_simulation import run_with_dispatch_strategy
        history = run_with_dispatch_strategy(scenarios_dir, hours=24)
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


def run_with_dispatch_strategy(
    scenarios_dir: str,
    hours: Optional[int] = None,
    output_dir: Optional[Path] = None
) -> Dict[str, np.ndarray]:
    """
    Run simulation using SimulationEngine with integrated DispatchStrategy.

    This is the recommended entry point for production simulations,
    replacing the standalone Orchestrator.run_simulation() method.

    Execution Flow:
        1. Load configuration from scenarios directory.
        2. Build component graph via PlantGraphBuilder.
        3. Create and populate ComponentRegistry.
        4. Load dispatch data (prices, wind power).
        5. Create and configure HybridArbitrageEngineStrategy.
        6. Create SimulationEngine with strategy injection.
        7. Initialize engine and dispatch strategy.
        8. Execute simulation loop.
        9. Return history arrays from dispatch strategy.

    Args:
        scenarios_dir (str): Path to scenarios directory containing:
            - simulation_config.yaml
            - physics_config.yaml
            - Energy price and wind data files.
        hours (int, optional): Simulation duration in hours.
            Default: from configuration.
        output_dir (Path, optional): Output directory for results.
            Default: <scenarios_dir>/simulation_output.

    Returns:
        Dict[str, np.ndarray]: Dictionary of NumPy arrays containing
            simulation history (minute, P_offer, P_soec_actual, h2_kg, etc.).

    Example:
        >>> history = run_with_dispatch_strategy("scenarios/baseline", hours=168)
        >>> print(f"Total H2: {np.sum(history['h2_kg']):.1f} kg")
    """
    from h2_plant.config.loader import ConfigLoader
    from h2_plant.core.graph_builder import PlantGraphBuilder
    from h2_plant.core.component_registry import ComponentRegistry
    from h2_plant.simulation.engine import SimulationEngine
    from h2_plant.control.engine_dispatch import HybridArbitrageEngineStrategy
    from h2_plant.data.price_loader import EnergyPriceLoader

    # Load configuration
    logger.info(f"Loading configuration from {scenarios_dir}")
    loader = ConfigLoader(scenarios_dir)
    context = loader.load_context()

    if hours is None:
        hours = context.simulation.duration_hours

    # Build component graph
    builder = PlantGraphBuilder(context)
    components = builder.build()

    # Create and populate registry
    registry = ComponentRegistry()
    for cid, comp in components.items():
        registry.register(cid, comp)

    # Load dispatch input data
    data_loader = EnergyPriceLoader(scenarios_dir)
    prices, wind = data_loader.load_data(
        context.simulation.energy_price_file,
        context.simulation.wind_data_file,
        hours,
        context.simulation.timestep_hours
    )
    total_steps = len(prices)

    # Create dispatch strategy
    dispatch_strategy = HybridArbitrageEngineStrategy()

    # Create simulation engine with strategy
    if output_dir is None:
        output_dir = Path(scenarios_dir) / "simulation_output"

    engine = SimulationEngine(
        registry=registry,
        config=context.simulation,
        output_dir=output_dir,
        dispatch_strategy=dispatch_strategy
    )

    # Initialize engine and strategy
    engine.initialize()
    engine.set_dispatch_data(prices, wind)
    engine.initialize_dispatch_strategy(context, total_steps)

    # Run simulation
    logger.info(f"Running simulation for {hours} hours ({total_steps} steps)")
    results = engine.run(end_hour=hours)

    # Get history from dispatch strategy
    history = engine.get_dispatch_history()

    # Print summary
    if dispatch_strategy:
        dispatch_strategy.print_summary()

    logger.info("Simulation completed successfully")
    return history


def main():
    """
    CLI entry point for integrated dispatch simulation.

    Parses command-line arguments and executes simulation with
    optional verbose logging.
    """
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="Run H2 Plant simulation with integrated dispatch strategy"
    )
    parser.add_argument(
        "scenarios_dir",
        help="Path to scenarios directory"
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=None,
        help="Simulation duration in hours (default: from config)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run simulation
    output_dir = Path(args.output_dir) if args.output_dir else None
    history = run_with_dispatch_strategy(
        scenarios_dir=args.scenarios_dir,
        hours=args.hours,
        output_dir=output_dir
    )

    # Summary stats
    if history:
        print(f"\nRecorded {len(list(history.values())[0])} timesteps")
        print(f"History keys: {list(history.keys())}")


if __name__ == "__main__":
    main()
