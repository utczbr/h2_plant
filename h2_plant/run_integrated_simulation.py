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

# Module-level storage for component metadata (populated by run_with_dispatch_strategy)
_component_metadata: Dict[str, Dict[str, Any]] = {}


def run_with_dispatch_strategy(
    scenarios_dir: str,
    hours: Optional[int] = None,
    output_dir: Optional[Path] = None,
    strategy: Optional[str] = None,
    resume_from_hour: Optional[int] = None,
    resume_checkpoint_path: Optional[Path] = None
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
        strategy (str, optional): Dispatch strategy override.
            Options: SOEC_ONLY, REFERENCE_HYBRID, ECONOMIC_SPOT.
            Default: from simulation_config.yaml (REFERENCE_HYBRID).
        resume_from_hour (int, optional): Hour to resume simulation from.
            If specified, the simulation will start from this hour.

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
    from h2_plant.core.enums import DispatchStrategyEnum
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

    # Determine dispatch strategy (CLI > config > default)
    strategy_name = (strategy or 
                     getattr(context.simulation, 'dispatch_strategy', None) or 
                     'REFERENCE_HYBRID').upper()
    
    logger.info(f"Using dispatch strategy: {strategy_name}")
    
    # Create dispatch strategy with inner strategy based on selection
    dispatch_strategy = HybridArbitrageEngineStrategy()
    
    # Configure inner strategy based on selection
    # The inner strategy selection will be passed to initialize()
    dispatch_strategy._strategy_override = strategy_name

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
    
    # Use chunked history for simulations > 7 days (memory optimization)
    # Threshold: 7 days × 24 hours × 60 minutes = 10,080 steps
    # Always use chunked history for consistent parquet output (enables graph compatibility)
    use_chunked_history = True  # Previously: total_steps > 10_080
    engine.initialize_dispatch_strategy(context, total_steps, use_chunked_history=use_chunked_history)

    # Run simulation
    # If we have a checkpoint path, the engine will load it and update current_hour.
    # We still pass start_hour as a hint or fallback.
    start_hour = resume_from_hour if resume_from_hour else 0
    if resume_from_hour:
        logger.info(f"Resuming simulation from hour {resume_from_hour}, running to hour {hours}")
    else:
        logger.info(f"Running simulation for {hours} hours ({total_steps} steps)")
    
    results = engine.run(
        start_hour=start_hour,
        end_hour=hours,
        resume_from_checkpoint=str(resume_checkpoint_path) if resume_checkpoint_path else None
    )

    # Get history from dispatch strategy (MOVED to later to allow for streaming export)
    # history = engine.get_dispatch_history() 
    history = None # Placeholder

    # Print dispatch summary
    if dispatch_strategy:
        dispatch_strategy.print_summary()
    
    # Build connection map from topology (required for report)
    connection_map = {}
    for node in context.topology.nodes:
        targets = [conn.target_name for conn in node.connections]
        connection_map[node.id] = targets
    
    from h2_plant.reporting.stream_table import print_stream_summary_table
    topo_order = [node.id for node in context.topology.nodes]
    components_dict = {cid: comp for cid, comp in registry.list_components()}
    try:
        print_stream_summary_table(components_dict, topo_order)
    except Exception as e:
        logger.error(f"Failed to print stream summary: {e}", exc_info=True)

    # Extract component metadata for visualization grouping
    component_metadata = {}
    for cid, comp in components_dict.items():
        component_metadata[cid] = {
            'system_group': getattr(comp, 'system_group', None),
            'process_step': getattr(comp, 'process_step', 0)
        }

    # Generate markdown report
    # SUPPRESSED by user request
    # from h2_plant.reporting.markdown_report import generate_simulation_report
    # report_path = output_dir / "simulation_report.md"
    # topology_name = getattr(context.topology, 'name', 'SOEC Hydrogen Production')
    # generate_simulation_report(
    #     components=components_dict,
    #     topo_order=topo_order,
    #     connection_map=connection_map,
    #     history=history,
    #     duration_hours=hours,
    #     output_path=report_path,
    #     topology_name=topology_name
    # )
    # logger.info(f"Markdown report saved to: {report_path}")

    # Generate CAPEX Report
    # =========================================================================
    try:
        from h2_plant.economics import CapexGenerator
        
        capex_config_path = Path(scenarios_dir) / "Economics" / "equipment_mappings.yaml"
        
        if capex_config_path.exists():
            logger.info("Generating CAPEX report...")
            capex_generator = CapexGenerator.from_yaml(capex_config_path)
            
            capex_report = capex_generator.generate(
                registry=registry,
                monitoring=engine.monitoring if hasattr(engine, 'monitoring') else None,
                output_dir=output_dir,
                simulation_name=getattr(context, 'simulation_name', 'H2 Plant Simulation'),
                simulation_hours=hours
            )
            
            logger.info(f"CAPEX Report Generated:")
            logger.info(f"  Total C_BM: ${capex_report.total_C_BM:,.0f}")
            logger.info(f"  Range (±{capex_report.overall_cost_class.value}): "
                       f"${capex_report.total_C_BM_low:,.0f} - ${capex_report.total_C_BM_high:,.0f}")
            logger.info(f"  Valid entries: {capex_report.entries_with_cost}/{len(capex_report.entries)}")
        else:
            logger.debug(f"CAPEX config not found at {capex_config_path}, skipping CAPEX generation")
    except ImportError as e:
        logger.warning(f"CAPEX generator not available: {e}")
    except Exception as e:
        logger.error(f"CAPEX generation failed: {e}", exc_info=True)
        capex_report = None

    # Generate OPEX Report
    # =========================================================================
    try:
        from h2_plant.economics.opex_generator import OpexGenerator
        
        opex_config_path = Path(scenarios_dir) / "Economics" / "opex_config.yaml"
        
        if opex_config_path.exists():
            logger.info("Generating OPEX report...")
            
            csv_path = output_dir / "simulation_history.csv"
            opex_generator = OpexGenerator()
            
            # Use streaming for large files to avoid memory issues
            if csv_path.exists():
                file_size_mb = csv_path.stat().st_size / (1024 * 1024)
                
                if file_size_mb > 500:  # Use streaming for files > 500 MB
                    logger.info(f"Using streaming OPEX (file: {file_size_mb:.0f} MB)")
                    opex_report = opex_generator.generate_streaming(
                        config_path=str(opex_config_path),
                        csv_path=csv_path,
                        capex_report=capex_report,
                        output_dir=str(output_dir),
                        simulation_hours=hours
                    )
                else:
                    # Small file - use full load
                    import pandas as pd
                    history_df = pd.read_csv(csv_path)
                    opex_report = opex_generator.generate(
                        config_path=str(opex_config_path),
                        capex_report=capex_report,
                        history_df=history_df,
                        output_dir=str(output_dir),
                        simulation_hours=hours
                    )
            elif history is not None:
                import pandas as pd
                scalar_data = {k: v for k, v in history.items() if isinstance(v, (list, tuple)) and len(v) > 0}
                if scalar_data:
                    history_df = pd.DataFrame(scalar_data)
                else:
                    history_df = None
                opex_report = opex_generator.generate(
                    config_path=str(opex_config_path),
                    capex_report=capex_report,
                    history_df=history_df,
                    output_dir=str(output_dir),
                    simulation_hours=hours
                )
            else:
                # No data available - generate without simulation data
                opex_report = opex_generator.generate(
                    config_path=str(opex_config_path),
                    capex_report=capex_report,
                    history_df=None,
                    output_dir=str(output_dir),
                    simulation_hours=hours
                )
            
            logger.info(f"OPEX Report Generated:")
            logger.info(f"  Total OPEX: ${opex_report.total_opex:,.0f}/year")
            logger.info(f"  Variable: ${opex_report.total_variable_cost:,.0f}")
            logger.info(f"  Fixed: ${opex_report.total_fixed_cost:,.0f}")
            logger.info(f"  Maintenance: ${opex_report.total_maintenance_cost:,.0f}")
        else:
            logger.debug(f"OPEX config not found at {opex_config_path}, skipping OPEX generation")
    except ImportError as e:
        logger.warning(f"OPEX generator not available: {e}")
    except Exception as e:
        logger.error(f"OPEX generation failed: {e}", exc_info=True)

    # COMMENT: This section handles the generation of the history CSV file, as requested by the user.
    try:
        import pandas as pd
        csv_path = output_dir / "simulation_history.csv"
        npz_path = output_dir / "simulation_matrices.npz"
        
        # Save Scalars to CSV (Additive/Streaming Priority)
        # ---------------------------------------------------------------------
        # If dispatch strategy supports streaming export (ChunkedHistoryManager),
        # use it to save memory.
        exported_via_stream = False
        if hasattr(dispatch_strategy, 'export_history_to_csv'):
            try:
                if dispatch_strategy.export_history_to_csv(csv_path):
                    logger.info(f"Streamed history to {csv_path} (Additive Mode)")
                    exported_via_stream = True
            except Exception as e:
                logger.error(f"Streaming export failed: {e}. Falling back to standard write.")
        
        # Fallback / Standard Write
        if not exported_via_stream and history is not None:
            # Split history into 1D (scalar) and >1D (matrix)
            scalar_history = {}
            matrix_history = {}
            
            for k, v in history.items():
                if isinstance(v, np.ndarray) and v.ndim > 1:
                    matrix_history[k] = v
                else:
                    scalar_history[k] = v

            df_history = pd.DataFrame(scalar_history)
            logger.info(f"Writing CSV: {len(df_history)} rows, {len(df_history.columns)} columns...")
            df_history.to_csv(csv_path, index=False)
            
            file_size_mb = csv_path.stat().st_size / (1024 * 1024)
            logger.info(f"Simulation history exported to: {csv_path} ({file_size_mb:.1f} MB)")
        else:
             # If streamed, we still need to separate matrices if any exist in the return dict
             # The ChunkedHistoryManager might export everything to CSV, but matrices 
             # are usually stored in 'matrix_history' separate from the chunked manager?
             # EngineDispatchStrategy deals with this.
             
             # Fetch ONLY matrices if possible to save RAM
             if hasattr(dispatch_strategy, 'get_matrix_history'):
                 matrix_history = dispatch_strategy.get_matrix_history()
             else:
                 # Fallback: We might have to load full history to extract matrices?
                 # Or just skip matrix export if we are strictly in low-memory mode.
                 pass

        # Force sync to disk
        import os
        if csv_path.exists():
             with open(csv_path, 'r') as f:
                os.fsync(f.fileno())
        
        # Save Matrices to NPZ
        if matrix_history:
             np.savez_compressed(npz_path, **matrix_history)
             logger.info(f"Simulation matrices exported to: {npz_path}")

        # Visualization Generation (Requires Data)
        # ---------------------------------------------------------------------
        # If we streamed, 'history' is currently None. We try to load it now for graphs.
        if history is None:
             try:
                 logger.info("loading history for visualization...")
                 history = engine.get_dispatch_history()
             except Exception as e:
                 logger.warning(f"Could not load history for visualization (likely OOM): {e}")
                 history = {} # Empty dict to skip graphs
        
        # Proceed to graphing with whatever 'history' we obtained
        # (Remaining code expects 'history' dict)
             
    except Exception as e:
        logger.warning(f"Failed to export history CSV/NPZ: {e}")

    logger.info("Simulation completed successfully")
    
    # Return history and metadata for graph generation (if called externally)
    # Store metadata on module level for generate_graphs
    global _component_metadata
    _component_metadata = component_metadata
    
    return history


def generate_graphs(
    history: Dict[str, np.ndarray],
    scenarios_dir: str,
    output_dir: Path
) -> None:
    """
    Generate graphs using UnifiedGraphExecutor.
    
    This function uses the new unified graph generation architecture that:
    - Loads column requirements from GraphCatalog
    - Applies memory-efficient column filtering
    - Executes graphs in priority order with timeout protection
    - Supports both Matplotlib (PNG) and Plotly (HTML) outputs
    
    The legacy GRAPH_MAP loop has been removed. All graphs are now registered
    in `graph_catalog.py` and executed via `UnifiedGraphExecutor`.
    
    To revert to legacy behavior, set `visualization.use_legacy_generator: true`
    in visualization_config.yaml.
    
    Args:
        history: Simulation history dictionary (column_name -> numpy array).
        scenarios_dir: Path to scenarios directory containing YAML configs.
        output_dir: Output directory for simulation results.
    """
    import yaml
    import gc
    
    logger.info("### Starting Graph Generation ###")
    
    # Load visualization config
    viz_config_path = Path(scenarios_dir) / "visualization_config.yaml"
    if viz_config_path.exists():
        with open(viz_config_path, 'r') as f:
            viz_config = yaml.safe_load(f) or {}
    else:
        logger.warning(f"visualization_config.yaml not found at {viz_config_path}")
        viz_config = {}
    
    # Check for legacy fallback flag
    if viz_config.get('visualization', {}).get('use_legacy_generator', False):
        logger.warning("Legacy generator requested but has been removed. Using UnifiedGraphExecutor.")
    
    # Create graphs directory
    graphs_dir = output_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # UNIFIED GRAPH EXECUTION
    # =========================================================================
    try:
        from h2_plant.visualization.graph_catalog import GRAPH_REGISTRY
        from h2_plant.visualization.unified_executor import UnifiedGraphExecutor
        
        # Initialize executor with catalog
        executor = UnifiedGraphExecutor(GRAPH_REGISTRY, graphs_dir)
        
        # Configure enabled graphs from YAML
        executor.configure_from_yaml(viz_config)
        
        # If no explicit config, enable all Matplotlib graphs (default behavior)
        if not viz_config.get('visualization', {}).get('categories') and \
           not viz_config.get('visualization', {}).get('graphs'):
            # No explicit config - use defaults (all Matplotlib graphs enabled)
            logger.info("No explicit graph config, using catalog defaults")
        
        # Load optimized DataFrame with only required columns
        df = executor.load_data(history=history)
        logger.info(f"Loaded DataFrame: {df.shape[0]} rows x {df.shape[1]} columns")
        
        # Execute all enabled graphs with timeout protection
        timeout = viz_config.get('visualization', {}).get('timeout_seconds', 60)
        results = executor.execute(df, timeout_seconds=timeout)
        
        # Log results
        success_count = sum(1 for r in results.values() if r.status == 'success')
        total_count = len(results)
        logger.info(f"### Graphs Generated: {success_count}/{total_count} ###")
        
        # Log any failures
        for graph_id, result in results.items():
            if result.status == 'failed':
                logger.warning(f"  Failed: {graph_id} - {result.error}")
            elif result.status == 'timeout':
                logger.warning(f"  Timeout: {graph_id} - exceeded {timeout}s")
        
        # Cleanup
        del df
        gc.collect()
        
    except ImportError as e:
        logger.error(f"UnifiedGraphExecutor not available: {e}")
        logger.error("Graph generation skipped. Install missing dependencies or check imports.")
    except Exception as e:
        logger.error(f"Graph generation failed: {e}", exc_info=True)
    
    # =========================================================================
    # DAILY H2 PRODUCTION GRAPH (Special case - uses CSV input)
    # =========================================================================
    try:
        daily_config = viz_config.get('visualization', {}).get('orchestrated_graphs', {}).get('daily_h2_production_average', {})
        if daily_config.get('enabled', False):
            from scripts.plot_daily_h2_production import generate_daily_h2_production_graph
            csv_path = output_dir / "simulation_history.csv"
            daily_output = graphs_dir / "daily_h2_production.png"
            if csv_path.exists():
                generate_daily_h2_production_graph(str(csv_path), str(daily_output))
                logger.info("Generated: daily_h2_production.png")
    except ImportError as e:
        logger.debug(f"Daily H2 production graph not available: {e}")
    except Exception as e:
        logger.warning(f"Daily H2 production graph failed: {e}")
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
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["SOEC_ONLY", "REFERENCE_HYBRID", "ECONOMIC_SPOT"],
        default=None,
        help="Dispatch strategy (overrides config). Options: SOEC_ONLY, REFERENCE_HYBRID, ECONOMIC_SPOT"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable cProfile benchmarking"
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume simulation from checkpoint file (e.g., checkpoints/checkpoint_hour_5280.json)"
    )
    parser.add_argument(
        "--memory-profile",
        action="store_true",
        help="Enable memory profiling with tracemalloc (reports peak usage)"
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
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Memory profiling setup
    if args.memory_profile:
        import tracemalloc
        tracemalloc.start()
        logger.info("Memory profiling enabled with tracemalloc")

    # Resume from checkpoint if specified
    resume_hour = None
    if args.resume_from:
        import json
        checkpoint_path = Path(args.resume_from)
        if not checkpoint_path.is_absolute():
            checkpoint_path = output_dir / args.resume_from
        
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint file not found: {checkpoint_path}")
            print(f"ERROR: Checkpoint file not found: {checkpoint_path}")
            return
        
        # Load checkpoint to get resume hour
        with open(checkpoint_path, 'r') as f:
            checkpoint_data = json.load(f)
        
        # Handle both engine checkpoint and ChunkedHistoryManager checkpoint formats
        if 'hour' in checkpoint_data:
            resume_hour = checkpoint_data['hour']
        elif 'total_steps_completed' in checkpoint_data:
            # ChunkedHistoryManager format: steps are 1-minute intervals
            resume_hour = checkpoint_data['total_steps_completed'] // 60
        
        logger.info(f"Will resume from checkpoint at hour {resume_hour}")
        print(f"Resuming from checkpoint: {checkpoint_path} (hour {resume_hour})")

    if args.profile:
        import cProfile
        import pstats
        
        logger.info("Running with cProfile enabled...")
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            history = run_with_dispatch_strategy(
                scenarios_dir=args.scenarios_dir,
                hours=args.hours,
                output_dir=output_dir,
                strategy=args.strategy,
                resume_from_hour=resume_hour,
                resume_checkpoint_path=checkpoint_path if resume_hour is not None else None
            )
        finally:
            profiler.disable()
            stats_path = output_dir / "profile.stats"
            profiler.dump_stats(stats_path)
            
            # Print summary
            print("\n" + "="*80)
            print("BENCHMARKING SUMMARY (Top 20 by Cumulative Time)")
            print("="*80)
            stats = pstats.Stats(profiler).sort_stats('cumtime')
            stats.print_stats(20)
            print("="*80)
            logger.info(f"Profile stats saved to {stats_path}")

    else:
        history = run_with_dispatch_strategy(
            scenarios_dir=args.scenarios_dir,
            hours=args.hours,
            output_dir=output_dir,
            strategy=args.strategy,
            resume_from_hour=resume_hour,
            resume_checkpoint_path=checkpoint_path if resume_hour is not None else None
        )
    
    # Memory profiling report
    if args.memory_profile:
        import tracemalloc
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        print("\n" + "="*80)
        print("MEMORY PROFILING SUMMARY")
        print("="*80)
        print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")
        print(f"Final memory usage: {current / 1024 / 1024:.1f} MB")
        print("="*80)
        logger.info(f"Peak memory: {peak / 1024 / 1024:.1f} MB, Final: {current / 1024 / 1024:.1f} MB")

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
    project_root = Path(sys.argv[0]).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    main()
