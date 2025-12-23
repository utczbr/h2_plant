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
    from h2_plant.config.plant_config import ConnectionConfig

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

    # Extract connections from topology nodes
    # NodeConnection format: {source_port, target_name, target_port, resource_type}
    # ConnectionConfig format: {source_id, source_port, target_id, target_port, resource_type}
    topology_connections = []
    for node in context.topology.nodes:
        source_id = node.id
        for conn in node.connections:
            topology_connections.append(ConnectionConfig(
                source_id=source_id,
                source_port=conn.source_port,
                target_id=conn.target_name,
                target_port=conn.target_port,
                resource_type=conn.resource_type
            ))
    
    logger.info(f"Extracted {len(topology_connections)} flow connections from topology")

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

    # Create simulation engine with strategy and topology connections
    if output_dir is None:
        output_dir = Path(scenarios_dir) / "simulation_output"

    engine = SimulationEngine(
        registry=registry,
        config=context.simulation,
        output_dir=output_dir,
        dispatch_strategy=dispatch_strategy,
        topology=topology_connections  # Pass extracted connections
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

    # Print dispatch summary
    if dispatch_strategy:
        dispatch_strategy.print_summary()
    
    # Print stream summary table
    from h2_plant.reporting.stream_table import print_stream_summary_table
    
    # Build connection map from topology
    connection_map = {}
    for node in context.topology.nodes:
        targets = [conn.target_name for conn in node.connections]
        connection_map[node.id] = targets
    
    topo_order = [node.id for node in context.topology.nodes]
    components_dict = {cid: comp for cid, comp in registry.list_components()}
    print_stream_summary_table(components_dict, topo_order, connection_map)

    # Generate markdown report
    from h2_plant.reporting.markdown_report import generate_simulation_report
    report_path = output_dir / "simulation_report.md"
    topology_name = getattr(context.topology, 'name', 'SOEC Hydrogen Production')
    generate_simulation_report(
        components=components_dict,
        topo_order=topo_order,
        connection_map=connection_map,
        history=history,
        duration_hours=hours,
        output_path=report_path,
        topology_name=topology_name
    )
    logger.info(f"Markdown report saved to: {report_path}")

    logger.info("Simulation completed successfully")
    return history


def generate_graphs(
    history: Dict[str, np.ndarray],
    scenarios_dir: str,
    output_dir: Path
) -> None:
    """
    Generate graphs based on visualization_config.yaml settings.
    
    Args:
        history: Simulation history dictionary.
        scenarios_dir: Path to scenarios directory.
        output_dir: Output directory for graphs.
    """
    import yaml
    import pandas as pd
    
    # Import graph creation functions
    try:
        from h2_plant.visualization.static_graphs import (
            # Core dispatch/economics graphs
            create_dispatch_figure,
            create_arbitrage_figure,
            create_h2_production_figure,
            create_oxygen_figure,
            create_water_figure,
            create_energy_pie_figure,
            create_histogram_figure,
            create_dispatch_curve_figure,
            create_cumulative_h2_figure,
            create_cumulative_energy_figure,
            create_efficiency_curve_figure,
            create_revenue_analysis_figure,
            create_temporal_averages_figure,
            create_water_removal_total_figure,
            # Thermal & Separation graphs
            create_drains_discarded_figure,
            create_chiller_cooling_figure,
            create_coalescer_separation_figure,
            create_kod_separation_figure,
            create_dry_cooler_figure,
            # Energy & Analysis graphs
            create_energy_flows_figure,
            create_plant_balance_schematic,
            create_mixer_comparison_figure,
            create_individual_drains_figure,
            create_dissolved_gas_figure,
            create_dissolved_gas_efficiency_figure,
            create_crossover_impurities_figure,
            create_thermal_load_breakdown_figure,
            create_drain_concentration_figure,
            create_recirculation_comparison_figure,
            create_entrained_liquid_figure,
            create_drain_line_properties_figure,
            # Profile & Flow tracking graphs
            create_water_vapor_tracking_figure,
            create_total_mass_flow_figure,
        )
        GRAPHS_AVAILABLE = True
    except ImportError as e:
        logger.warning(f"Graph functions not available: {e}")
        GRAPHS_AVAILABLE = False
        return
    
    # Load visualization config
    viz_config_path = Path(scenarios_dir) / "visualization_config.yaml"
    if not viz_config_path.exists():
        logger.info("No visualization_config.yaml found, skipping graph generation")
        return
    
    with open(viz_config_path, 'r') as f:
        viz_config = yaml.safe_load(f)
    
    enabled_graphs = viz_config.get('visualization', {}).get('graphs', {})
    export_config = viz_config.get('visualization', {}).get('export', {})
    
    if not export_config.get('enabled', True):
        logger.info("Graph export disabled in visualization_config.yaml")
        return
    
    # Create graphs directory
    graphs_dir = output_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert history to DataFrame
    df = pd.DataFrame(history)
    
    # Normalize column names to match what graph functions expect
    # The dispatch history uses names like 'H2_soec_kg', 'spot_price', 'P_soec_actual'
    # but graph functions expect 'H2_soec', 'Spot', 'P_soec'
    COLUMN_ALIASES = {
        'H2_soec_kg': 'H2_soec',
        'H2_pem_kg': 'H2_pem',
        'spot_price': 'Spot',
        'P_soec_actual': 'P_soec',
        'steam_soec_kg': 'Steam_soec',
        'H2O_soec_out_kg': 'H2O_soec',
        'H2O_pem_kg': 'H2O_pem',
        'O2_pem_kg': 'O2_pem',
        'cumulative_h2_kg': 'Cumulative_H2',
        'pem_V_cell': 'PEM_V_cell',
        'tank_level_kg': 'Tank_Level',
        'tank_pressure_bar': 'Tank_Pressure',
        'compressor_power_kw': 'Compressor_Power',
    }
    
    for old_name, new_name in COLUMN_ALIASES.items():
        if old_name in df.columns and new_name not in df.columns:
            df[new_name] = df[old_name]
    
    # Also ensure 'time' or 'hour' column exists for X-axis
    if 'minute' in df.columns and 'time' not in df.columns:
        df['time'] = df['minute'] / 60.0  # Convert minutes to hours
        df['hour'] = df['time']
    
    # Map of graph names to creation functions
    GRAPH_MAP = {
        # Core dispatch/economics (primary names)
        'dispatch_strategy_stacked': create_dispatch_figure,
        'arbitrage_scatter': create_arbitrage_figure,
        'total_h2_production_stacked': create_h2_production_figure,
        'oxygen_production_stacked': create_oxygen_figure,
        'water_consumption_stacked': create_water_figure,
        'power_consumption_breakdown_pie': create_energy_pie_figure,
        'price_histogram': create_histogram_figure,
        'dispatch_curve_scatter': create_dispatch_curve_figure,
        'cumulative_h2_production': create_cumulative_h2_figure,
        'cumulative_energy': create_cumulative_energy_figure,
        'efficiency_curve': create_efficiency_curve_figure,
        'revenue_analysis': create_revenue_analysis_figure,
        'temporal_averages': create_temporal_averages_figure,
        
        # Thermal & Separation graphs
        'water_removal_total': create_water_removal_total_figure,
        'drains_discarded': create_drains_discarded_figure,
        'chiller_cooling': create_chiller_cooling_figure,
        'coalescer_separation': create_coalescer_separation_figure,
        'kod_separation': create_kod_separation_figure,
        'dry_cooler_performance': create_dry_cooler_figure,
        
        # Energy & Thermal Analysis
        'energy_flows': create_energy_flows_figure,
        'q_breakdown': create_thermal_load_breakdown_figure,
        'plant_balance': create_plant_balance_schematic,
        'mixer_comparison': create_mixer_comparison_figure,
        'individual_drains': create_individual_drains_figure,
        'dissolved_gas_concentration': create_dissolved_gas_figure,
        'dissolved_gas_efficiency': create_dissolved_gas_efficiency_figure,
        'crossover_impurities': create_crossover_impurities_figure,
        'drain_line_properties': create_drain_line_properties_figure,
        'drain_line_concentration': create_drain_concentration_figure,
        'recirculation_comparison': create_recirculation_comparison_figure,
        'entrained_liquid_flow': create_entrained_liquid_figure,
        
        # Flow tracking
        'water_vapor_tracking': create_water_vapor_tracking_figure,
        'total_mass_flow': create_total_mass_flow_figure,
        
        # Legacy aliases (disabled by default in config, kept for backward compat)
        'dispatch': create_dispatch_figure,
        'arbitrage': create_arbitrage_figure,
        'h2_production': create_h2_production_figure,
        'oxygen_production': create_oxygen_figure,
        'water_consumption': create_water_figure,
        'energy_pie': create_energy_pie_figure,
        'dispatch_curve': create_dispatch_curve_figure,
        'cumulative_h2': create_cumulative_h2_figure,
    }
    
    generated_count = 0
    
    for graph_name, is_enabled in enabled_graphs.items():
        if not is_enabled:
            continue
        
        create_func = GRAPH_MAP.get(graph_name)
        if create_func is None:
            continue  # Graph type not implemented in basic set
        
        try:
            fig = create_func(df)
            if fig is not None:
                # Save as PNG
                output_path = graphs_dir / f"{graph_name}.png"
                fig.savefig(output_path, dpi=100, bbox_inches='tight')
                logger.info(f"Generated: {output_path.name}")
                generated_count += 1
                
                # Close figure to free memory
                import matplotlib.pyplot as plt
                plt.close(fig)
        except Exception as e:
            logger.warning(f"Failed to generate {graph_name}: {e}")
    
    print(f"\n### Graphs Generated: {generated_count} files saved to {graphs_dir}")


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
    parser.add_argument(
        "--no-graphs",
        action="store_true",
        help="Skip graph generation"
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run simulation
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.scenarios_dir) / "simulation_output"
    history = run_with_dispatch_strategy(
        scenarios_dir=args.scenarios_dir,
        hours=args.hours,
        output_dir=output_dir
    )

    # Summary stats
    if history:
        print(f"\nRecorded {len(list(history.values())[0])} timesteps")
        print(f"History keys: {list(history.keys())}")
        
        # Generate graphs unless disabled
        if not args.no_graphs:
            print("\n### Generating Graphs...")
            generate_graphs(history, args.scenarios_dir, output_dir)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    # Add project root to sys.path to allow absolute imports when running directly
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    main()

