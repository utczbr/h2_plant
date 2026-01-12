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
    strategy: Optional[str] = None
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
    use_chunked_history = total_steps > 10_080
    engine.initialize_dispatch_strategy(context, total_steps, use_chunked_history=use_chunked_history)

    # Run simulation
    logger.info(f"Running simulation for {hours} hours ({total_steps} steps)")
    results = engine.run(end_hour=hours)

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

    # Export history to CSV and NPZ
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
        if not exported_via_stream:
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
    Generate graphs based on visualization_config.yaml settings.
    
    .. deprecated:: 2025-12
        The legacy GRAPH_MAP loop in this function is deprecated.
        Use `GraphOrchestrator` with `orchestrated_graphs` section in 
        visualization_config.yaml instead. The GRAPH_MAP loop will be 
        removed in a future version.
    
    Args:
        history: Simulation history dictionary.
        scenarios_dir: Path to scenarios directory.
        output_dir: Output directory for graphs.
    """
    import yaml
    import pandas as pd
    import fnmatch
    import gc
    
    # =========================================================================
    # COLUMN PATTERNS REGISTRY FOR MEMORY OPTIMIZATION
    # Maps graph categories to column patterns (exact matches or glob patterns)
    # This prevents loading all 1000+ columns for each graph.
    # =========================================================================
    CORE_COLUMNS = [
        'minute', 'P_offer', 'P_soec_actual', 'P_pem', 'P_sold', 
        'spot_price', 'h2_kg', 'time', 'hour'
    ]
    
    GRAPH_COLUMN_PATTERNS = {
        # Core Dispatch
        'dispatch_strategy_stacked': CORE_COLUMNS + ['P_bop_mw'],
        'dispatch': CORE_COLUMNS + ['P_bop_mw'],
        'arbitrage_scatter': CORE_COLUMNS,
        'arbitrage': CORE_COLUMNS,
        
        # H2 Production
        'total_h2_production_stacked': CORE_COLUMNS + ['H2_soec_kg', 'H2_pem_kg', 'H2_atr_kg'],
        'h2_production': CORE_COLUMNS + ['H2_soec_kg', 'H2_pem_kg', 'H2_atr_kg'],
        'cumulative_h2_production': CORE_COLUMNS + [
            'H2_soec_kg', 'H2_pem_kg', 'H2_atr_kg', 'cumulative_h2_kg',
            'h2_rfnbo_kg', 'h2_non_rfnbo_kg', 'cumulative_h2_rfnbo_kg', 'cumulative_h2_non_rfnbo_kg',
            '*_PSA_*_outlet_mass_flow_kg_h'
        ],
        'cumulative_h2': CORE_COLUMNS + ['H2_soec_kg', 'H2_pem_kg', 'H2_atr_kg', 'cumulative_h2_kg'],
        
        # O2 Production
        'oxygen_production_stacked': CORE_COLUMNS + ['O2_pem_kg', '*_O2_*'],
        'oxygen_production': CORE_COLUMNS + ['O2_pem_kg'],
        
        # Water
        'water_consumption_stacked': CORE_COLUMNS + ['steam_soec_kg', 'H2O_soec_out_kg', 'H2O_pem_kg'],
        'water_consumption': CORE_COLUMNS + ['steam_soec_kg', 'H2O_soec_out_kg', 'H2O_pem_kg'],
        
        # Economics
        'power_consumption_breakdown_pie': CORE_COLUMNS + ['compressor_power_kw', '*_cooling_load_kw'],
        'energy_pie': CORE_COLUMNS + ['compressor_power_kw'],
        'price_histogram': ['spot_price', 'minute'],
        'dispatch_curve_scatter': CORE_COLUMNS,
        'dispatch_curve': CORE_COLUMNS,
        'efficiency_curve': CORE_COLUMNS + ['H2_soec_kg', 'H2_pem_kg'],
        'revenue_analysis': CORE_COLUMNS + ['cumulative_h2_kg'],
        'temporal_averages': CORE_COLUMNS + ['H2_soec_kg', 'H2_pem_kg'],
        'cumulative_energy': CORE_COLUMNS,
        
        # RFNBO
        'rfnbo_compliance_stacked': CORE_COLUMNS + ['h2_rfnbo_kg', 'h2_non_rfnbo_kg', 'purchase_threshold_eur_mwh'],
        'rfnbo_spot_analysis': CORE_COLUMNS + ['purchase_threshold_eur_mwh'],
        'rfnbo_pie': ['h2_rfnbo_kg', 'h2_non_rfnbo_kg', 'minute'],
        'cumulative_rfnbo': CORE_COLUMNS + ['cumulative_h2_rfnbo_kg', 'cumulative_h2_non_rfnbo_kg'],
        
        # SOEC Modules
        'soec_module_heatmap': ['minute', 'soec_module_*', 'P_soec_actual', 'soec_active_modules'],
        'soec_module_power_stacked': ['minute', 'soec_module_*', 'P_soec_actual'],
        'soec_module_wear_stats': ['minute', 'soec_module_*'],
        'monthly_performance': CORE_COLUMNS + ['H2_soec_kg', 'H2_pem_kg'],
        'monthly_efficiency': CORE_COLUMNS + ['H2_soec_kg', 'H2_pem_kg'],
        'monthly_capacity_factor': CORE_COLUMNS + ['H2_soec_kg', 'H2_pem_kg'],
        
        # Thermal/Separation - use pattern matching
        'water_removal_total': ['minute', '*_liquid_removed_kg*', '*_water_removed*'],
        'drains_discarded': ['minute', '*_liquid_removed_kg*', '*_drain*'],
        'chiller_cooling': ['minute', '*Chiller*', '*_cooling*', '*_duty*'],
        'coalescer_separation': ['minute', '*Coalescer*'],
        'kod_separation': ['minute', '*KOD*', '*_liquid_removed*'],
        'dry_cooler_performance': ['minute', '*DryCooler*', '*Drycooler*', '*_cooling*'],
        
        # Energy/Schematic
        'energy_flows': CORE_COLUMNS + ['compressor_power_kw', '*_power_kw'],
        'q_breakdown': ['minute', '*_cooling*', '*_duty*', '*_heat*'],
        'thermal_load_breakdown_time_series': ['minute', '*_cooling*', '*_duty*', '*Chiller*', '*Intercooler*'],
        'plant_balance': CORE_COLUMNS + ['H2_soec_kg', 'H2_pem_kg'],
        'mixer_comparison': ['minute', '*Mixer*'],
        
        # Separation Analysis (need component-specific columns)
        'individual_drains': ['minute', '*_drain*', '*_liquid*'],
        'dissolved_gas_concentration': ['minute', '*_dissolved*', '*KOD*', '*Mixer*'],
        'dissolved_gas_efficiency': ['minute', '*_dissolved*', '*Mixer*'],
        'crossover_impurities': ['minute', '*_o2_*', '*_h2_*', '*_impurity*', '*_outlet_*'],
        'drain_line_properties': ['minute', '*Drain*', '*_outlet_temp*', '*_outlet_pressure*'],
        'deoxo_profile': ['minute', '*Deoxo*'],
        'drain_mixer_balance': ['minute', '*Drain_Mixer*', '*_mass_flow*'],
        'drain_scheme': ['minute', '*Drain*', '*_mass_flow*'],
        'energy_flow': CORE_COLUMNS + ['*_power_kw', 'compressor_power_kw'],
        'process_scheme': CORE_COLUMNS + ['H2_soec_kg', 'H2_pem_kg'],
        'drain_line_concentration': ['minute', '*Drain_Mixer*', '*_dissolved*'],
        'recirculation_comparison': ['minute', '*recirc*', '*recirculation*'],
        'entrained_liquid_flow': ['minute', '*_entrained*', '*_liquid*'],
        
        # Stacked Properties (temperature, pressure profiles)
        'h2_stacked_properties': ['minute', '*SOEC_H2_*_outlet_*', '*PEM_H2_*_outlet_*', 'SOEC_Cluster_*'],
        'o2_stacked_properties': ['minute', '*O2_*_outlet_*'],
        
        # Flow tracking
        'water_vapor_tracking': ['minute', '*_h2o_*', '*_vapor*', '*_moisture*'],
        'total_mass_flow': ['minute', '*_mass_flow*', '*_flow_kg*'],
        
        # Storage
        'storage_levels': ['minute', 'tank_*', '*Tank*_level*', '*Tank*_pressure*'],
        'compressor_power': ['minute', '*Compressor*_power*', 'compressor_power_kw'],
        
        # Effective PPA (new)
        'effective_ppa': CORE_COLUMNS + ['ppa_price_effective_eur_mwh'],
        
        # Water Tank Inventory (new)
        'water_tank_inventory': ['minute', '*UltraPure_Tank*', '*mass_kg*', '*control_zone*'],
    }
    
    def filter_columns_for_graph(all_columns: list, graph_name: str) -> list:
        """
        Filter column list to only those needed for a specific graph.
        Uses exact matches and glob patterns from GRAPH_COLUMN_PATTERNS.
        
        Args:
            all_columns: List of all available column names
            graph_name: Name of the graph being generated
            
        Returns:
            Filtered list of columns to load (always includes 'minute')
        """
        patterns = GRAPH_COLUMN_PATTERNS.get(graph_name)
        if patterns is None:
            # Unknown graph - return core columns + any obvious matches
            return [c for c in all_columns if any(
                p in c for p in ['minute', 'P_', 'H2_', 'spot', 'cumulative']
            )][:100]  # Cap at 100 columns for safety
        
        matched = set()
        for pattern in patterns:
            if '*' in pattern:
                # Glob pattern
                matched.update(fnmatch.filter(all_columns, pattern))
            else:
                # Exact match
                if pattern in all_columns:
                    matched.add(pattern)
        
        # Always include minute if available
        if 'minute' in all_columns:
            matched.add('minute')
            
        return list(matched)
    
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
            create_crossover_impurities_figure,
            create_thermal_load_breakdown_figure,
            # Profile & Flow tracking graphs
            create_water_vapor_tracking_figure,
            create_total_mass_flow_figure,
            create_monthly_performance_figure,
            create_monthly_efficiency_figure,
            create_monthly_capacity_factor_figure,
            create_soec_module_heatmap_figure,
            create_soec_module_power_stacked_figure,
            create_soec_module_wear_figure,
            create_q_breakdown_figure,
            create_drain_line_properties_figure,
            create_deoxo_profile_figure,
            create_drain_mixer_figure,
            create_drain_scheme_schematic,
            create_energy_flow_figure,
            create_process_scheme_schematic,
            create_drain_concentration_figure,
            create_recirculation_comparison_figure,
            create_entrained_liquid_figure,
            create_h2_stacked_properties,
            create_o2_stacked_properties,
            # RFNBO Compliance graphs
            create_rfnbo_compliance_stacked_figure,
            create_rfnbo_spot_analysis_figure,
            create_rfnbo_pie_figure,
            create_cumulative_rfnbo_figure,
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
    
    # =========================================================================
    # MEMORY-OPTIMIZED LOADING STRATEGY
    # Instead of loading all 1000+ columns at once (~4GB for year-long sims),
    # we store the raw history and create filtered DataFrames per-graph.
    # =========================================================================
    
    # Filter 1D history for DataFrame (exclude matrices)
    scalar_history = {k: v for k, v in history.items() if not (isinstance(v, np.ndarray) and v.ndim > 1)}
    all_columns = list(scalar_history.keys())
    
    logger.info(f"Graph data: {len(all_columns)} columns, {len(scalar_history.get('minute', []))} rows")
    
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
    
    def create_filtered_df(graph_name: str) -> pd.DataFrame:
        """Create a memory-efficient DataFrame with only columns needed for graph."""
        needed_cols = filter_columns_for_graph(all_columns, graph_name)
        
        # Build minimal dict with only needed columns
        filtered_data = {}
        for col in needed_cols:
            if col in scalar_history:
                filtered_data[col] = scalar_history[col]
        
        df = pd.DataFrame(filtered_data)
        
        # Apply aliases
        for old_name, new_name in COLUMN_ALIASES.items():
            if old_name in df.columns and new_name not in df.columns:
                df[new_name] = df[old_name]
        
        # Add time/hour if minute exists
        if 'minute' in df.columns and 'time' not in df.columns:
            df['time'] = df['minute'] / 60.0
            df['hour'] = df['time']
        
        return df
    
    # Create a "full" DataFrame only for orchestrated graphs that need all data
    # For legacy graphs, we'll create per-graph filtered DataFrames
    _full_df_cache = None
    
    def get_full_df() -> pd.DataFrame:
        """Lazy-load full DataFrame (for orchestrator compatibility)."""
        nonlocal _full_df_cache
        if _full_df_cache is None:
            logger.info("Loading full DataFrame for orchestrator...")
            _full_df_cache = pd.DataFrame(scalar_history)
            for old_name, new_name in COLUMN_ALIASES.items():
                if old_name in _full_df_cache.columns and new_name not in _full_df_cache.columns:
                    _full_df_cache[new_name] = _full_df_cache[old_name]
            if 'minute' in _full_df_cache.columns and 'time' not in _full_df_cache.columns:
                _full_df_cache['time'] = _full_df_cache['minute'] / 60.0
                _full_df_cache['hour'] = _full_df_cache['time']
        return _full_df_cache
    
    # =========================================================================
    # DEPRECATED: GRAPH_MAP
    # This legacy mapping is deprecated. Use GraphOrchestrator with 
    # visualization_config.yaml 'orchestrated_graphs' section instead.
    # Will be removed in future version.
    # =========================================================================
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
        'monthly_performance': create_monthly_performance_figure,
        'monthly_efficiency': create_monthly_efficiency_figure,
        'monthly_capacity_factor': create_monthly_capacity_factor_figure,
        'soec_module_heatmap': create_soec_module_heatmap_figure,
        'soec_module_power_stacked': create_soec_module_power_stacked_figure,
        'soec_module_wear_stats': create_soec_module_wear_figure,
        
        # Thermal & Separation graphs
        'water_removal_total': create_water_removal_total_figure,
        'drains_discarded': create_drains_discarded_figure,
        'chiller_cooling': create_chiller_cooling_figure,
        'coalescer_separation': create_coalescer_separation_figure,
        'kod_separation': create_kod_separation_figure,
        'dry_cooler_performance': create_dry_cooler_figure,
        
        # Energy & Thermal Analysis
        'energy_flows': create_energy_flows_figure,
        'q_breakdown': create_q_breakdown_figure,
        'thermal_load_breakdown_time_series': create_thermal_load_breakdown_figure, # Rename old one to clearer name
        'plant_balance': create_plant_balance_schematic,
        'mixer_comparison': create_mixer_comparison_figure,
        'individual_drains': create_individual_drains_figure,
        'dissolved_gas_concentration': create_dissolved_gas_figure,
        'dissolved_gas_efficiency': create_drain_concentration_figure,
        'crossover_impurities': create_crossover_impurities_figure,
        'drain_line_properties': create_drain_line_properties_figure,
        'deoxo_profile': create_deoxo_profile_figure,
        'drain_mixer_balance': create_drain_mixer_figure,
        'drain_scheme': create_drain_scheme_schematic,
        'energy_flow': create_energy_flow_figure,
        'process_scheme': create_process_scheme_schematic,
        'drain_line_concentration': create_drain_concentration_figure,
        'recirculation_comparison': create_recirculation_comparison_figure,
        'entrained_liquid_flow': create_entrained_liquid_figure,
        'h2_stacked_properties': create_h2_stacked_properties,
        'o2_stacked_properties': create_o2_stacked_properties,
        
        # Flow tracking
        'water_vapor_tracking': create_water_vapor_tracking_figure,
        'total_mass_flow': create_total_mass_flow_figure,
        
        # RFNBO Compliance (Economic Spot Dispatch)
        'rfnbo_compliance_stacked': create_rfnbo_compliance_stacked_figure,
        'rfnbo_spot_analysis': create_rfnbo_spot_analysis_figure,
        'rfnbo_pie': create_rfnbo_pie_figure,
        'cumulative_rfnbo': create_cumulative_rfnbo_figure,
        
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
    
    # Access global metadata extracted during simulation
    global _component_metadata
    metadata = _component_metadata
    
    # =========================================================================
    # LEGACY GRAPH_MAP LOOP (DEPRECATED)
    # Skip if 'skip_legacy_graphs' is set in config (default: False)
    # This loop will be removed in a future version.
    # =========================================================================
    skip_legacy = viz_config.get('visualization', {}).get('skip_legacy_graphs', False)
    
    if skip_legacy:
        logger.info("Skipping legacy GRAPH_MAP loop (skip_legacy_graphs=true)")
    else:
        for graph_name, is_enabled in enabled_graphs.items():
            if not is_enabled:
                continue
            
            create_func = GRAPH_MAP.get(graph_name)
            if create_func is None:
                continue  # Graph type not implemented in basic set
            
            try:
                # Create memory-efficient filtered DataFrame for this graph
                df = create_filtered_df(graph_name)
                logger.debug(f"Graph '{graph_name}': using {len(df.columns)} columns")
                
                # Pass metadata to crossover_impurities graph for grouping/sorting
                if graph_name == 'crossover_impurities':
                    fig = create_func(df, metadata=metadata)
                else:
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
                
                # Free memory after each graph
                del df
                gc.collect()
                
            except Exception as e:
                logger.warning(f"Failed to generate {graph_name}: {e}")
    
    print(f"\n### Legacy Graphs Generated: {generated_count} files saved to {graphs_dir}")
    
    # ========================================================================
    # ORCHESTRATED GRAPHS (New Architecture)
    # Uses full DataFrame - lazy-loaded only if orchestrated graphs are enabled
    # ========================================================================
    try:
        from h2_plant.visualization.graph_orchestrator import GraphOrchestrator
        
        orchestrator = GraphOrchestrator(graphs_dir)
        full_df = get_full_df()  # Lazy-load full DataFrame
        orchestrated_count = orchestrator.generate_all(full_df, viz_config)
        
        if orchestrated_count > 0:
            print(f"### Orchestrated Graphs Generated: {orchestrated_count} files")
        
        # Clean up full DataFrame
        del full_df
        gc.collect()
    except ImportError as e:
        logger.debug(f"Graph orchestrator not available: {e}")
    except Exception as e:
        logger.warning(f"Orchestrated graph generation failed: {e}")
    
    # ========================================================================
    # DAILY H2 PRODUCTION AVERAGE GRAPH
    # ========================================================================
    try:
        daily_config = viz_config.get('visualization', {}).get('orchestrated_graphs', {}).get('daily_h2_production_average', {})
        if daily_config.get('enabled', False):
            from scripts.plot_daily_h2_production import generate_daily_h2_production_graph
            csv_path = output_dir / "simulation_history.csv"
            output_path = graphs_dir / "daily_h2_production.png"
            if csv_path.exists():
                generate_daily_h2_production_graph(str(csv_path), str(output_path))
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
                strategy=args.strategy
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
            strategy=args.strategy
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
    project_root = Path(sys.argv[0]).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    main()
