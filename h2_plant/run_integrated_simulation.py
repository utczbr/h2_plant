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
from typing import Dict, Any, Optional, Callable
import numpy as np

logger = logging.getLogger(__name__)

# Module-level storage for component metadata (populated by run_with_dispatch_strategy)
_component_metadata: Dict[str, Dict[str, Any]] = {}


def _resolve_scenario_name(scenarios_dir: str) -> str:
    scen_name_candidate = None
    try:
        topo_path = Path(scenarios_dir) / "plant_topology.yaml"
        if topo_path.exists():
            import yaml
            with open(topo_path, 'r') as f:
                topo_config = yaml.safe_load(f) or {}
                scen_name_candidate = topo_config.get('scenario_name')
    except Exception as e:
        logger.debug(f"Scenario name lookup failed: {e}")

    return scen_name_candidate if scen_name_candidate else Path(scenarios_dir).name


def _resolve_guaranteed_power_mw(scenarios_dir: str, default: float = 10.0) -> float:
    guaranteed_power_mw = default
    try:
        eco_path = Path(scenarios_dir) / "economics_parameters.yaml"
        if eco_path.exists():
            import yaml
            with open(eco_path, 'r') as f:
                eco_config = yaml.safe_load(f) or {}
                guaranteed_power_mw = eco_config.get('guaranteed_power_mw', default)
    except Exception as e:
        logger.warning(f"Could not load economics config: {e}. Using default guaranteed power.")
        guaranteed_power_mw = default
    return guaranteed_power_mw


def _resolve_topology_file_for_graph_metrics(scenarios_dir: str) -> Optional[Path]:
    base = Path(scenarios_dir)
    for filename in ("plant_topology.yaml", "plant_topology_PEM+SOEC.yaml"):
        candidate = base / filename
        if candidate.exists():
            return candidate
    return None


def _load_lifecycle_hours_from_topology(topology_path: Optional[Path]) -> tuple[Optional[float], Optional[float]]:
    if topology_path is None or not topology_path.exists():
        return None, None

    import yaml

    try:
        data = yaml.safe_load(topology_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.warning("Failed to read topology for lifecycle extraction (%s): %s", topology_path, exc)
        return None, None

    pem_lifecycle_h = None
    soec_lifecycle_h = None
    for node in data.get("nodes", []) or []:
        ntype = str(node.get("type", "")).strip().upper()
        params = node.get("params", {}) or {}
        lifecycle = params.get("lifecycle")
        if lifecycle is None:
            continue
        try:
            parsed = float(lifecycle)
        except (TypeError, ValueError):
            continue
        if ntype == "PEM" and pem_lifecycle_h is None:
            pem_lifecycle_h = parsed
        if ntype == "SOEC" and soec_lifecycle_h is None:
            soec_lifecycle_h = parsed
    return pem_lifecycle_h, soec_lifecycle_h


def _load_stack_reserve_percentages(opex_config_path: Optional[Path]) -> tuple[Optional[float], Optional[float]]:
    if opex_config_path is None or not opex_config_path.exists():
        return None, None

    import yaml

    try:
        data = yaml.safe_load(opex_config_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.warning("Failed to read OPEX config for reserve extraction (%s): %s", opex_config_path, exc)
        return None, None

    pem_reserve_pct = None
    soec_reserve_pct = None
    for item in data.get("opex_items", []) or []:
        name = str(item.get("name", "")).lower()
        price = item.get("price")
        if price is None:
            continue
        try:
            parsed_price = float(price)
        except (TypeError, ValueError):
            continue
        if "stack replacement reserve" in name and "pem" in name:
            pem_reserve_pct = parsed_price
        if "stack replacement reserve" in name and "soec" in name:
            soec_reserve_pct = parsed_price
    return pem_reserve_pct, soec_reserve_pct


def _inject_replacement_metrics_for_net_profit(
    metrics: Dict[str, Any],
    *,
    scenarios_dir: str,
    output_dir: Path,
) -> Dict[str, Any]:
    """Inject lifecycle/reserve values used by net-profit replacement spikes."""
    merged = dict(metrics or {})
    scenarios_path = Path(scenarios_dir)

    topology_path = _resolve_topology_file_for_graph_metrics(scenarios_dir)
    if topology_path is not None:
        logger.info("Net-profit replacement lifecycle source: %s", topology_path)
    else:
        logger.warning(
            "No topology file found for replacement lifecycle extraction in %s "
            "(expected plant_topology.yaml or plant_topology_PEM+SOEC.yaml).",
            scenarios_path,
        )
    pem_lifecycle_h, soec_lifecycle_h = _load_lifecycle_hours_from_topology(topology_path)

    opex_candidates = [
        output_dir / "Economics" / "opex_config.yaml",
        output_dir.parent / "Economics" / "opex_config.yaml",
        scenarios_path / "Economics" / "opex_config.yaml",
    ]
    opex_config_path = next((path for path in opex_candidates if path.exists()), None)
    if opex_config_path is not None:
        logger.info("Net-profit replacement reserve source: %s", opex_config_path)
    else:
        logger.warning(
            "No opex_config.yaml found for replacement reserve extraction. Checked: %s",
            ", ".join(str(path) for path in opex_candidates),
        )
    pem_reserve_pct, soec_reserve_pct = _load_stack_reserve_percentages(opex_config_path)

    if pem_lifecycle_h is not None:
        merged.setdefault("pem_lifecycle_h", pem_lifecycle_h)
    if soec_lifecycle_h is not None:
        merged.setdefault("soec_lifecycle_h", soec_lifecycle_h)
    if pem_reserve_pct is not None:
        merged.setdefault("pem_reserve_pct", pem_reserve_pct)
    if soec_reserve_pct is not None:
        merged.setdefault("soec_reserve_pct", soec_reserve_pct)

    logger.info(
        "Net-profit replacement inputs resolved: PEM(lifecycle_h=%r, reserve_pct=%r), "
        "SOEC(lifecycle_h=%r, reserve_pct=%r)",
        merged.get("pem_lifecycle_h"),
        merged.get("pem_reserve_pct"),
        merged.get("soec_lifecycle_h"),
        merged.get("soec_reserve_pct"),
    )
    return merged


def run_with_dispatch_context(
    context,
    *,
    hours: Optional[int] = None,
    data_dir: Optional[str] = None,
    output_dir: Optional[Path] = None,
    strategy: Optional[str] = None,
    resume_from_hour: Optional[int] = None,
    resume_checkpoint_path: Optional[Path] = None,
    load_history: bool = True,
    allow_graph_dispatch_fallback: bool = False,
    return_registry: bool = False,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
):
    """
    Shared simulation core — runs the engine from a pre-loaded SimulationContext.

    This is the single source of truth for engine assembly, dispatch strategy
    selection, and history retrieval.  Both the CLI (``run_with_dispatch_strategy``)
    and the GUI ``SimulationWorker`` delegate to this function.

    Args:
        context: Loaded SimulationContext (from ConfigLoader or GraphToConfigAdapter).
        hours: Override simulation duration in hours (default: from context).
        data_dir: Base directory for resolving relative data file paths (energy prices,
            wind data).  When *None*, the parent directory of ``energy_price_file``
            is used.
        output_dir: Directory for engine output files (default: ./simulation_output).
        strategy: Dispatch strategy override (CLI > context > 'ECONOMIC_SPOT').
        resume_from_hour: Hour to resume from (passed through to engine).
        resume_checkpoint_path: Checkpoint file for resume.
        load_history: Whether to call ``engine.get_dispatch_history()`` before
            returning (default: True).
        allow_graph_dispatch_fallback: When True, missing dispatch-critical components
            cause a silent fallback to physics-only mode instead of raising.
            Set True for GUI graph mode; False for CLI and scenario mode.
        return_registry: When True, return ``(history, registry)`` instead of just
            ``history``.
        progress_callback: Optional callback invoked during simulation steps with
            ``(completed_steps, total_steps, current_hour)``.
        cancel_check: Optional callback polled before each step. If it returns True,
            the run is interrupted via ``KeyboardInterrupt``.

    Returns:
        If ``return_registry`` is False: Dict[str, np.ndarray] history.
        If ``return_registry`` is True: (history, ComponentRegistry) tuple.
    """
    from h2_plant.core.component_registry import ComponentRegistry
    from h2_plant.core.graph_builder import PlantGraphBuilder
    from h2_plant.simulation.engine import SimulationEngine
    from h2_plant.control.engine_dispatch import HybridArbitrageEngineStrategy
    from h2_plant.data.price_loader import EnergyPriceLoader
    from h2_plant.config.plant_config import ConnectionConfig, SimulationConfig

    run_start_hour = resume_from_hour or 0
    run_end_hour = hours if hours is not None else context.simulation.duration_hours
    run_duration_hours = run_end_hour - run_start_hour
    if run_duration_hours <= 0:
        raise ValueError(
            f"Invalid run window: start_hour={run_start_hour}, "
            f"end_hour={run_end_hour}, duration_hours={run_duration_hours}. "
            "Expected a positive duration."
        )
    total_steps = int(run_duration_hours * 60)
    if total_steps <= 0:
        raise ValueError(
            f"Invalid dispatch horizon: duration_hours={run_duration_hours} produced "
            f"total_steps={total_steps}. Expected at least 1 step."
        )
    logger.info(
        "Resolved run window: start_hour=%s, end_hour=%s, duration_hours=%s, "
        "total_dispatch_steps=%s",
        run_start_hour,
        run_end_hour,
        run_duration_hours,
        total_steps,
    )

    # 1. Build component graph and populate registry
    builder = PlantGraphBuilder(context)
    components = builder.build()
    registry = ComponentRegistry()
    for cid, comp in components.items():
        registry.register(cid, comp)

    # 2. Extract topology connections (NodeConnection → ConnectionConfig)
    topology_connections = []
    if hasattr(context, 'topology') and context.topology:
        for node in context.topology.nodes:
            for conn in node.connections:
                topology_connections.append(ConnectionConfig(
                    source_id=node.id,
                    source_port=conn.source_port,
                    target_id=conn.target_name,
                    target_port=conn.target_port,
                    resource_type=conn.resource_type,
                ))
    logger.info(f"Extracted {len(topology_connections)} topology connections")

    # 3. Create dispatch strategy (two independent fallback paths):
    #    a) arbitrage_enabled=False → physics-only (no dispatch strategy).
    #    b) allow_graph_dispatch_fallback=True + missing components → physics-only.
    dispatch_strategy = None
    dispatch_requested = (
        hasattr(context.economics, 'arbitrage_enabled')
        and context.economics.arbitrage_enabled
    )
    _REQUIRED_DISPATCH_IDS = ("SOEC_Transformer", "PEM_Transformer", "BOP_Transformer")

    if dispatch_requested:
        missing = [cid for cid in _REQUIRED_DISPATCH_IDS if not registry.has(cid)]
        if missing and allow_graph_dispatch_fallback:
            logger.warning(
                "Graph mode: missing dispatch components (%s). "
                "Falling back to physics-only mode.",
                ", ".join(missing),
            )
            dispatch_requested = False
        elif missing:
            raise ValueError(
                f"Dispatch requires {_REQUIRED_DISPATCH_IDS}; "
                f"missing: {', '.join(missing)}"
            )

    if dispatch_requested:
        dispatch_strategy = HybridArbitrageEngineStrategy()
        strategy_name = (
            strategy
            or getattr(context.simulation, 'dispatch_strategy', None)
            or 'ECONOMIC_SPOT'
        ).upper()
        dispatch_strategy._strategy_override = strategy_name
        logger.info(f"Dispatch strategy: {strategy_name}")
    else:
        logger.info("Running in physics-only mode (no dispatch strategy)")

    # 4. Create SimulationEngine
    sim_config = SimulationConfig(
        timestep_hours=context.simulation.timestep_hours,
        duration_hours=run_duration_hours,
        start_hour=0,
        dispatch_strategy=getattr(context.simulation, 'dispatch_strategy', 'REFERENCE_HYBRID'),
        checkpoint_interval_hours=getattr(context.simulation, 'checkpoint_interval_hours', 168),
    )
    if output_dir is None:
        output_dir = Path("simulation_output")
    engine = SimulationEngine(
        registry=registry,
        config=sim_config,
        output_dir=output_dir,
        topology=topology_connections,
        dispatch_strategy=dispatch_strategy,
    )

    # 5. Load price/wind data
    if data_dir is None:
        data_dir = str(Path(context.simulation.energy_price_file).parent) or "."
    loader = EnergyPriceLoader(str(data_dir))
    prices, wind = loader.load_data(
        context.simulation.energy_price_file,
        context.simulation.wind_data_file,
        duration_hours=run_duration_hours,
        timestep_hours=context.simulation.timestep_hours,
    )

    orig_price_len = len(prices)
    orig_wind_len = len(wind)
    if orig_price_len == 0 or orig_wind_len == 0:
        raise ValueError(
            f"Dispatch data is empty (prices={orig_price_len}, wind={orig_wind_len}). "
            "Cannot run dispatch simulation."
        )

    # Repeat profile extension policy: tile cyclically to full run horizon.
    if orig_price_len < total_steps:
        prices = np.tile(prices, int(np.ceil(total_steps / orig_price_len)))[:total_steps]
        logger.info(
            "Profile extension applied: prices tiled from %s to %s steps",
            orig_price_len,
            len(prices),
        )
    elif orig_price_len > total_steps:
        prices = prices[:total_steps]
        logger.info(
            "Profile truncation applied: prices trimmed from %s to %s steps",
            orig_price_len,
            len(prices),
        )

    if orig_wind_len < total_steps:
        wind = np.tile(wind, int(np.ceil(total_steps / orig_wind_len)))[:total_steps]
        logger.info(
            "Profile extension applied: wind tiled from %s to %s steps",
            orig_wind_len,
            len(wind),
        )
    elif orig_wind_len > total_steps:
        wind = wind[:total_steps]
        logger.info(
            "Profile truncation applied: wind trimmed from %s to %s steps",
            orig_wind_len,
            len(wind),
        )

    # 6. Initialize engine and dispatch strategy
    engine.set_dispatch_data(prices, wind)
    engine.initialize()
    engine.initialize_dispatch_strategy(
        context, total_steps, use_chunked_history=True, resume=bool(resume_from_hour)
    )

    if progress_callback is not None or cancel_check is not None:
        last_pct = -1

        def _pre_step_callback(hour: float) -> None:
            nonlocal last_pct

            if cancel_check is not None and cancel_check():
                raise KeyboardInterrupt("Simulation cancelled by user")

            if progress_callback is None:
                return

            # pre-step callback fires before processing this minute
            completed_steps = int(round((hour - run_start_hour) * 60))
            completed_steps = max(0, min(total_steps, completed_steps))
            pct = int((completed_steps * 100) / total_steps) if total_steps > 0 else 100
            if pct != last_pct:
                last_pct = pct
                progress_callback(completed_steps, total_steps, hour)

        engine.set_callbacks(pre_step=_pre_step_callback)

    # 7. Run simulation
    if resume_from_hour:
        logger.info(
            f"Resuming from hour {run_start_hour}, running to hour {run_end_hour} "
            f"({total_steps} dispatch steps)"
        )
    else:
        logger.info(
            f"Running simulation for {run_duration_hours} hours ({total_steps} steps)"
        )
    engine.run(
        start_hour=run_start_hour,
        end_hour=run_end_hour,
        resume_from_checkpoint=str(resume_checkpoint_path) if resume_checkpoint_path else None,
    )

    if progress_callback is not None:
        progress_callback(total_steps, total_steps, float(run_end_hour))

    # 8. Optional post-run summary
    if dispatch_strategy and hasattr(dispatch_strategy, 'print_summary'):
        dispatch_strategy.print_summary()

    # 9. Retrieve history
    history = None
    if load_history:
        history = engine.get_dispatch_history()

    if return_registry:
        return history, registry
    return history


def run_with_dispatch_strategy(
    scenarios_dir: str,
    hours: Optional[int] = None,
    output_dir: Optional[Path] = None,
    strategy: Optional[str] = None,
    resume_from_hour: Optional[int] = None,
    resume_checkpoint_path: Optional[Path] = None,
    load_history: bool = True
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

    # Load configuration context
    logger.info(f"Loading configuration from {scenarios_dir}")
    loader = ConfigLoader(scenarios_dir)
    context = loader.load_context()

    if hours is None:
        hours = context.simulation.duration_hours
    if output_dir is None:
        output_dir = Path(scenarios_dir) / "simulation_output"

    # Core: delegate to shared simulation runner (GUI + CLI share the same path)
    history, registry = run_with_dispatch_context(
        context,
        hours=hours,
        data_dir=scenarios_dir,
        output_dir=output_dir,
        strategy=strategy,
        resume_from_hour=resume_from_hour,
        resume_checkpoint_path=resume_checkpoint_path,
        load_history=load_history,
        allow_graph_dispatch_fallback=False,  # CLI is always strict
        return_registry=True,
    )
    
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
                monitoring=None,
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

        logger.error(f"OPEX generation failed: {e}", exc_info=True)


    # COMMENT: This section handles the generation of the history CSV file, as requested by the users.
    if False:
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

        except Exception as e:
            logger.warning(f"Failed to export history CSV/NPZ: {e}")


    # Generate Scenario Summary Report
    # =========================================================================
    if False:
        try:
            logger.info("Generating Scenario Summary Report...")
            summary_csv_path = output_dir / "scenario_summary.csv"
            
            # We need a DataFrame. 
            # 1. Try to load from CSV if it exists (most complete)
            # 2. Convert 'history' dict to DataFrame
            summary_df = None
            
            if (output_dir / "simulation_history.csv").exists():
                # Use chunks or small read depending on size? 
                # Summary needs ALL rows for accurate integration.
                import pandas as pd
                summary_df = pd.read_csv(output_dir / "simulation_history.csv")
            elif history is not None:
                import pandas as pd
                # Filter scalars only for DataFrame construction
                scalar_data = {k: v for k, v in history.items() if isinstance(v, (list, tuple, np.ndarray)) and np.ndim(v) <= 1}
                summary_df = pd.DataFrame(scalar_data)
                
            if summary_df is not None and not summary_df.empty:
                # 3. Determine Scenario Name
                # Priority: 1. topology.yaml, 2. context attribute, 3. directory name
                scen_name_candidate = getattr(context, 'scenario_name', None)
                
                if not scen_name_candidate:
                    try:
                        topo_path = Path(scenarios_dir) / "plant_topology.yaml"
                        if topo_path.exists():
                            import yaml
                            with open(topo_path, 'r') as f:
                                topo_config = yaml.safe_load(f)
                                scen_name_candidate = topo_config.get('scenario_name')
                    except Exception:
                        pass # Fallback to directory name
                
                scenario_name = scen_name_candidate if scen_name_candidate else Path(scenarios_dir).name
                
                # Extract guaranteed power from context or config
                guaranteed_power_mw = 0.0
                try:
                    # Attempt to load from economics parameters file directly if available
                    eco_path = Path(scenarios_dir) / "economics_parameters.yaml"
                    if eco_path.exists():
                        import yaml
                        with open(eco_path, 'r') as f:
                            eco_config = yaml.safe_load(f)
                            guaranteed_power_mw = eco_config.get('guaranteed_power_mw', 10.0) # Default 10 per user request
                except Exception as e:
                    logger.warning(f"Could not load economics config: {e}. Using default guaranteed power.")
                    guaranteed_power_mw = 10.0

                generate_scenario_summary(summary_df, scenario_name, summary_csv_path, guaranteed_power_mw=guaranteed_power_mw)
                
            else:
                logger.warning("No data available for Scenario Summary Report.")
            
        except Exception as e:
            logger.error(f"Failed to generate Scenario Summary Report: {e}", exc_info=True)

    logger.info("Simulation completed successfully")
    
    # Return history and metadata for graph generation (if called externally)
    # Store metadata on module level for generate_graphs
    global _component_metadata
    _component_metadata = component_metadata
    
    return history


def generate_graphs(
    history: Optional[Dict[str, np.ndarray]],
    scenarios_dir: str,
    output_dir: Path,
    resample_freq: Optional[str] = None,
    chunks_dir: Optional[Path] = None
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
        resample_freq: Optional resampling frequency (e.g. '1D', '1H') for graph generation.
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
        # Note: We load full resolution data first, then resample during execution
        # Load optimized DataFrame with only required columns
        # Note: We load full resolution data first, then resample during execution
        df = executor.load_data(history=history, resample_freq=resample_freq, chunks_dir=chunks_dir)
        logger.info(f"Loaded DataFrame: {df.shape[0]} rows x {df.shape[1]} columns")

        # ------------------------------------------------------------------
        # Inject CAPEX/OPEX into df.attrs['metrics'] for economics graphs
        # ------------------------------------------------------------------
        try:
            import csv
            import json

            def _to_float(value):
                if value is None:
                    return None
                if isinstance(value, (int, float)):
                    return float(value)
                text = str(value).strip().replace(",", "").replace("€", "").replace("$", "")
                try:
                    return float(text)
                except ValueError:
                    return None

            def _parse_capex_csv(path: Path) -> Dict[str, float]:
                totals: Dict[str, float] = {}
                total_cbm = None
                total_installed = None

                with open(path, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if not row:
                            continue
                        label = str(row[0]).strip().upper()
                        if label == "TOTAL":
                            # Standard totals row in itemized section
                            if len(row) > 9:
                                total_cbm = _to_float(row[9]) or total_cbm
                        elif label == "OVERALL TOTAL":
                            # Summary row in block summary section
                            if len(row) > 2:
                                total_cbm = total_cbm or _to_float(row[2])
                            if len(row) > 5:
                                total_installed = _to_float(row[5]) or total_installed

                if total_cbm is not None:
                    totals["total_c_bm"] = total_cbm
                    totals["capex"] = total_cbm
                    totals["capex_total"] = total_cbm
                if total_installed is not None:
                    totals["total_installed_cost"] = total_installed
                    totals["fci"] = total_installed
                return totals

            metrics = df.attrs.get("metrics", {})
            if not isinstance(metrics, dict):
                metrics = {}

            # CAPEX (CSV)
            capex_path = output_dir / "capex_report.csv"
            if not capex_path.exists():
                alt_capex = output_dir / "Economics" / "capex_report.csv"
                if alt_capex.exists():
                    capex_path = alt_capex
            if capex_path.exists():
                capex_metrics = _parse_capex_csv(capex_path)
                for k, v in capex_metrics.items():
                    metrics.setdefault(k, v)
            else:
                logger.info(f"CAPEX report not found at {capex_path}")

            # OPEX (JSON)
            opex_path = output_dir / "opex_report.json"
            if not opex_path.exists():
                alt_opex = output_dir / "Economics" / "opex_report.json"
                if alt_opex.exists():
                    opex_path = alt_opex
            if opex_path.exists():
                with open(opex_path, "r", encoding="utf-8") as f:
                    opex_data = json.load(f)
                if isinstance(opex_data, dict):
                    for key in [
                        "total_opex",
                        "total_opex_low",
                        "total_opex_high",
                        "total_fixed_cost",
                        "total_variable_cost",
                        "total_maintenance_cost",
                        "annual_opex",
                        "annual_opex_low",
                        "annual_opex_high",
                        "opex_annual",
                        "opex_annual_low",
                        "opex_annual_high",
                        "opex",
                        "opex_low",
                        "opex_high",
                        "fci",
                    ]:
                        if key in opex_data:
                            val = _to_float(opex_data.get(key))
                            if val is not None:
                                metrics.setdefault(key, val)
                    if "total_opex" in opex_data:
                        metrics.setdefault("opex_total", _to_float(opex_data.get("total_opex")))
                        metrics.setdefault("total_annual_opex", _to_float(opex_data.get("total_opex")))
                    if "total_opex_low" in opex_data:
                        metrics.setdefault("opex_total_low", _to_float(opex_data.get("total_opex_low")))
                        metrics.setdefault("total_annual_opex_low", _to_float(opex_data.get("total_opex_low")))
                    if "total_opex_high" in opex_data:
                        metrics.setdefault("opex_total_high", _to_float(opex_data.get("total_opex_high")))
                        metrics.setdefault("total_annual_opex_high", _to_float(opex_data.get("total_opex_high")))
                else:
                    logger.warning(f"OPEX report JSON did not contain a dict: {opex_path}")
            else:
                logger.info(f"OPEX report not found at {opex_path}")

            metrics = _inject_replacement_metrics_for_net_profit(
                metrics,
                scenarios_dir=scenarios_dir,
                output_dir=output_dir,
            )
            df.attrs["metrics"] = metrics
        except Exception as e:
            logger.warning(f"Failed to inject CAPEX/OPEX metrics: {e}", exc_info=True)
        
        # Execute all enabled graphs with timeout protection and resampling
        timeout = viz_config.get('visualization', {}).get('timeout_seconds', 60)
        
        if resample_freq:
            logger.info(f"Generating graphs with resampling: {resample_freq}")
            
        results = executor.execute(df, timeout_seconds=timeout, resample_freq=resample_freq)
        
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
    # DISABLED by user request
    if False:
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
    parser.add_argument(
        "--resample",
        type=str,
        default=None,
        help="Resample graph data to specified frequency (e.g., '1D', '1H', '1W'). Improves graph readability for long simulations."
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
        if not checkpoint_path.exists() and not checkpoint_path.is_absolute():
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
            resume_checkpoint_path=checkpoint_path if resume_hour is not None else None,
            load_history=True  # Always load history; executor handles resampling from dict
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
        
    # Generate graphs unless disabled (Move OUTSIDE history check to allow streaming from disk)
    if not args.no_graphs:
        print("\n### Generating Graphs...")
        chunks_dir = output_dir / "history_chunks"
        generate_graphs(history, args.scenarios_dir, output_dir, resample_freq=args.resample, chunks_dir=chunks_dir)

    # Generate Scenario Summary Report (after graphs to minimize memory overlap)
    # =========================================================================
    try:
        from h2_plant.reporting.scenario_summary import generate_scenario_summary_streaming

        scenario_name = _resolve_scenario_name(args.scenarios_dir)
        guaranteed_power_mw = _resolve_guaranteed_power_mw(args.scenarios_dir)
        summary_csv_path = output_dir / "scenario_summary.csv"
        chunks_dir = output_dir / "history_chunks"
        csv_path = output_dir / "simulation_history.csv"

        generate_scenario_summary_streaming(
            scenario_name=scenario_name,
            output_path=summary_csv_path,
            guaranteed_power_mw=guaranteed_power_mw,
            history=history,
            chunks_dir=chunks_dir,
            csv_path=csv_path
        )
    except Exception as e:
        logger.error(f"Failed to generate Scenario Summary Report: {e}", exc_info=True)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    # Add project root to sys.path to allow absolute imports when running directly
    project_root = Path(sys.argv[0]).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    main()
