"""
Run Simulation with Integrated Dispatch Strategy.

This module provides a unified entry point for running simulations using the
new SimulationEngine + DispatchStrategy architecture (Phase B refactoring).

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
    
    This is the recommended entry point for new simulations, replacing the
    standalone Orchestrator.run_simulation() method.
    
    Architecture:
    - SimulationEngine handles component lifecycle (initialize, step, checkpoint)
    - HybridArbitrageEngineStrategy handles dispatch decisions and history recording
    - Components are stepped exactly once per timestep (A1 fix)
    - History uses pre-allocated NumPy arrays (A3 fix)
    
    Args:
        scenarios_dir: Path to scenarios directory containing config files
        hours: Simulation duration in hours (None = use config default)
        output_dir: Output directory for results (None = auto-create)
    
    Returns:
        Dictionary of NumPy arrays containing simulation history
    """
    from h2_plant.config.loader import ConfigLoader
    from h2_plant.core.graph_builder import PlantGraphBuilder
    from h2_plant.core.component_registry import ComponentRegistry
    from h2_plant.simulation.engine import SimulationEngine
    from h2_plant.control.engine_dispatch import HybridArbitrageEngineStrategy
    from h2_plant.data.price_loader import EnergyPriceLoader
    
    # 1. Load configuration
    logger.info(f"Loading configuration from {scenarios_dir}")
    loader = ConfigLoader(scenarios_dir)
    context = loader.load_context()
    
    if hours is None:
        hours = context.simulation.duration_hours
    
    # 2. Build component graph
    builder = PlantGraphBuilder(context)
    components = builder.build()
    
    # 3. Create and populate registry
    registry = ComponentRegistry()
    for cid, comp in components.items():
        registry.register(cid, comp)
    
    # 4. Load dispatch data (prices, wind)
    data_loader = EnergyPriceLoader(scenarios_dir)
    prices, wind = data_loader.load_data(
        context.simulation.energy_price_file,
        context.simulation.wind_data_file,
        hours,
        context.simulation.timestep_hours
    )
    total_steps = len(prices)
    
    # 5. Create dispatch strategy
    dispatch_strategy = HybridArbitrageEngineStrategy()
    
    # 6. Create simulation engine with strategy
    if output_dir is None:
        output_dir = Path(scenarios_dir) / "simulation_output"
    
    engine = SimulationEngine(
        registry=registry,
        config=context.simulation,
        output_dir=output_dir,
        dispatch_strategy=dispatch_strategy
    )
    
    # 7. Initialize engine and strategy
    engine.initialize()
    engine.set_dispatch_data(prices, wind)
    engine.initialize_dispatch_strategy(context, total_steps)
    
    # 8. Run simulation
    logger.info(f"Running simulation for {hours} hours ({total_steps} steps)")
    results = engine.run(end_hour=hours)
    
    # 9. Get history from dispatch strategy
    history = engine.get_dispatch_history()
    
    # 10. Print summary
    if dispatch_strategy:
        dispatch_strategy.print_summary()
    
    logger.info("Simulation completed successfully")
    return history


def main():
    """CLI entry point."""
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
