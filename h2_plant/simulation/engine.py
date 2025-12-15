"""
Modular Simulation Engine for Hydrogen Production System.

This module implements the main simulation orchestrator that coordinates
timestep execution, event processing, checkpointing, and monitoring for
hydrogen plant simulations.

Execution Architecture:
    The engine manages the complete simulation lifecycle:
    1. **Initialization**: Prepare all components with timestep and registry.
    2. **Timestep Execution**: Execute components in causal order.
    3. **Flow Propagation**: Route streams between connected components.
    4. **Monitoring**: Collect metrics for analysis.
    5. **Checkpointing**: Save state for recovery.

Causal Execution Order:
    Components are executed in dependency order to ensure correct physics:
    - External inputs and utility (price tracker, environment).
    - Production (electrolyzers generate H₂ and heat).
    - Thermal management (distribute/reject heat).
    - Separation and processing (purify streams).
    - Compression and storage (final product handling).

Dispatch Integration:
    Optional EngineDispatchStrategy can be attached to make power
    allocation decisions before component execution.

Architecture:
    The SimulationEngine fulfills Layer 2 orchestration:
    - Calls `initialize()` on all components at startup.
    - Calls `step()` on each component exactly once per timestep.
    - Collects `get_state()` for monitoring and checkpointing.
"""

import logging
from typing import Optional, Dict, Any, Callable, List
from pathlib import Path
import time
import json
import numpy as np

from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.exceptions import SimulationError
from h2_plant.config.plant_config import SimulationConfig, ConnectionConfig, IndexedConnectionConfig
from h2_plant.simulation.state_manager import StateManager
from h2_plant.simulation.event_scheduler import EventScheduler, Event
from h2_plant.simulation.monitoring import MonitoringSystem
from h2_plant.simulation.flow_network import FlowNetwork
from h2_plant.visualization.dashboard_generator import DashboardGenerator

try:
    from h2_plant.control.engine_dispatch import EngineDispatchStrategy
except ImportError:
    EngineDispatchStrategy = None

logger = logging.getLogger(__name__)


class SimulationEngine:
    """
    Main simulation orchestrator for hydrogen plant simulation.

    Coordinates simulation execution with event processing, checkpointing,
    and monitoring. Executes components in causal order and propagates
    flows through the topology.

    Attributes:
        registry (ComponentRegistry): Central registry for all components.
        config (SimulationConfig): Simulation configuration parameters.
        flow_network (FlowNetwork): Network managing stream propagation.
        is_initialized (bool): Whether engine has been initialized.
        is_running (bool): Whether simulation is currently running.

    Example:
        >>> engine = SimulationEngine(registry, config, topology=connections)
        >>> engine.initialize()
        >>> results = engine.run(start_hour=0, end_hour=24)
    """

    def __init__(
        self,
        registry: ComponentRegistry,
        config: SimulationConfig,
        output_dir: Optional[Path] = None,
        topology: List[ConnectionConfig] = None,
        indexed_topology: List[IndexedConnectionConfig] = None,
        dispatch_strategy: Optional['EngineDispatchStrategy'] = None
    ):
        """
        Initialize the simulation engine.

        Args:
            registry (ComponentRegistry): Central registry for components.
            config (SimulationConfig): Simulation configuration.
            output_dir (Path, optional): Directory for output files.
                Default: './simulation_output'.
            topology (List[ConnectionConfig], optional): Component connections.
            indexed_topology (List[IndexedConnectionConfig], optional):
                Indexed connections for array components.
            dispatch_strategy (EngineDispatchStrategy, optional): Power
                dispatch strategy for allocation decisions.
        """
        self.registry = registry
        self.config = config
        self.output_dir = output_dir or Path("simulation_output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Subsystems
        self.state_manager = StateManager(output_dir=self.output_dir)
        self.event_scheduler = EventScheduler()
        self.monitoring = MonitoringSystem(output_dir=self.output_dir)

        self.flow_network = FlowNetwork(
            registry=self.registry,
            topology=topology or [],
            indexed_topology=indexed_topology or []
        )

        # Simulation state
        self.current_hour: float = 0.0
        self.is_initialized: bool = False
        self.is_running: bool = False
        self.simulation_start_time: float = 0.0
        self.steps_run: int = 0

        # Callbacks
        self.pre_step_callback: Optional[Callable[[float], None]] = None
        self.post_step_callback: Optional[Callable[[float], None]] = None

        # Dispatch strategy integration
        self.dispatch_strategy = dispatch_strategy
        self._dispatch_prices: Optional[np.ndarray] = None
        self._dispatch_wind: Optional[np.ndarray] = None

    def initialize(self) -> None:
        """
        Initialize simulation engine and all components.

        Fulfills Layer 2 orchestration by calling `initialize()` on all
        registered components with the configured timestep.

        Raises:
            SimulationError: If initialization fails.
        """
        logger.info("Initializing simulation engine...")
        try:
            # Use 1-minute timestep for dynamics
            dt_hours = 1.0 / 60.0
            self.config.timestep_hours = dt_hours
            self.registry.initialize_all(dt=dt_hours)
            self.flow_network.initialize()
            self.monitoring.initialize(self.registry)
            self.is_initialized = True
            logger.info(f"Simulation engine initialized successfully (dt={dt_hours:.4f} hours)")
        except Exception as e:
            raise SimulationError(f"Simulation initialization failed: {e}") from e

    def run(
        self,
        start_hour: Optional[int] = None,
        end_hour: Optional[int] = None,
        resume_from_checkpoint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run simulation from start to end hour.

        Executes the main simulation loop, processing each timestep in
        sequence with event handling, component execution, and flow propagation.

        Args:
            start_hour (int, optional): Starting hour. Default: config.start_hour.
            end_hour (int, optional): Ending hour. Default: start + duration.
            resume_from_checkpoint (str, optional): Path to checkpoint file
                for resuming interrupted simulation.

        Returns:
            Dict[str, Any]: Simulation results including final states and metrics.

        Raises:
            SimulationError: If simulation execution fails.
        """
        effective_start_hour = start_hour if start_hour is not None else self.config.start_hour
        effective_end_hour = end_hour if end_hour is not None else (effective_start_hour + self.config.duration_hours)

        # Convert hours to steps (1-minute resolution)
        steps_per_hour = 60
        start_step = int(effective_start_hour * steps_per_hour)
        end_step = int(effective_end_hour * steps_per_hour)

        if resume_from_checkpoint:
            self._resume_from_checkpoint(resume_from_checkpoint)
            effective_start_hour = self.current_hour

        if not self.is_initialized:
            self.initialize()

        logger.info(f"Starting simulation: hours {effective_start_hour} to {effective_end_hour}")
        self.simulation_start_time = time.time()
        self.is_running = True
        self.steps_run = 0

        try:
            for step in range(start_step, end_step):
                hour = step / steps_per_hour
                self.current_hour = hour

                self._execute_timestep(hour)
                self.steps_run += 1

                # Hourly checkpointing and logging
                if step % steps_per_hour == 0:
                    int_hour = int(hour)
                    if self._should_checkpoint(int_hour):
                        self._save_checkpoint(int_hour)

                    if int_hour > 0 and int_hour % 168 == 0:
                        self._log_progress(int_hour, effective_end_hour, effective_start_hour)

                    self.monitoring.collect(hour, self.registry)

            self.is_running = False
            elapsed_time = time.time() - self.simulation_start_time
            rate = self.steps_run / elapsed_time if elapsed_time > 0 else float('inf')

            logger.info(
                f"Simulation complete: {self.steps_run} steps in "
                f"{elapsed_time:.2f} seconds ({rate:.1f} steps/sec)"
            )

            results = self._generate_results(start_hour=effective_start_hour, end_hour=effective_end_hour)
            return results

        except KeyboardInterrupt:
            logger.warning("Simulation interrupted by user")
            self._save_checkpoint(int(self.current_hour), emergency=True)
            raise

        except Exception as e:
            logger.error(f"Simulation failed at hour {self.current_hour}: {e}", exc_info=True)
            self._save_checkpoint(int(self.current_hour), emergency=True)
            raise SimulationError(f"Simulation execution failed: {e}") from e

    def _execute_timestep(self, hour: float) -> None:
        """
        Execute a single simulation timestep.

        Performs complete timestep cycle:
        1. Execute pre-step callback.
        2. Process scheduled events.
        3. Apply dispatch setpoints (if strategy configured).
        4. Execute components in causal order.
        5. Propagate flows through network.
        6. Record post-step dispatch results.
        7. Execute post-step callback.

        Args:
            hour (float): Current simulation time in fractional hours.
        """
        if self.pre_step_callback:
            self.pre_step_callback(hour)

        self.event_scheduler.process_events(hour, self.registry)

        # Apply dispatch setpoints before component execution
        if self.dispatch_strategy and self._dispatch_prices is not None:
            self.dispatch_strategy.decide_and_apply(
                t=hour,
                prices=self._dispatch_prices,
                wind=self._dispatch_wind
            )

        # Causal execution order ensures correct physics propagation
        execution_order = [
            'energy_price_tracker', 'demand_scheduler', 'environment_manager',
            'dual_path_coordinator',
            'soec_cluster', 'pem_electrolyzer_detailed',
            'thermal_manager',
            'chiller', 'chiller_hx5', 'chiller_hx6',
            'separator_sp3', 'psa_d3',
            'compressor_c1', 'compressor_c2', 'filling_compressor',
            'lp_tanks', 'hp_tanks', 'h2_storage_enhanced',
            'logistics_manager'
        ]

        # Execute ordered components first (each exactly once)
        executed_instances = set()

        for comp_id in execution_order:
            if self.registry.has(comp_id):
                try:
                    component = self.registry.get(comp_id)
                    if component not in executed_instances:
                        component.step(hour)
                        executed_instances.add(component)
                except Exception as e:
                    logger.error(f"Component {comp_id} step failed at hour {hour}: {e}")
                    raise

        # Execute remaining components
        for comp_id, component in self.registry.list_components():
            if component not in executed_instances:
                try:
                    component.step(hour)
                    executed_instances.add(component)
                except Exception as e:
                    logger.error(f"Component {comp_id} step failed at hour {hour}: {e}")
                    raise

        # Propagate flows after component updates
        try:
            self.flow_network.execute_flows(hour)
        except Exception as e:
            logger.error(f"Flow execution failed at hour {hour}: {e}")
            raise

        # Record dispatch results after physics execution
        if self.dispatch_strategy and hasattr(self.dispatch_strategy, 'record_post_step'):
            self.dispatch_strategy.record_post_step()

        if self.post_step_callback:
            self.post_step_callback(hour)

    def _should_checkpoint(self, hour: int) -> bool:
        """
        Determine if checkpoint should be saved at current hour.

        Args:
            hour (int): Current simulation hour.

        Returns:
            bool: True if checkpoint should be saved.
        """
        interval = self.config.checkpoint_interval_hours
        if interval <= 0:
            return False
        if hour == self.config.start_hour and not self.steps_run > 1:
            return False
        return hour % interval == 0

    def _save_checkpoint(self, hour: int, emergency: bool = False) -> None:
        """
        Save simulation checkpoint for recovery.

        Args:
            hour (int): Current simulation hour.
            emergency (bool): Whether this is an emergency save.
        """
        checkpoint_type = "emergency" if emergency else "regular"
        logger.info(f"Saving {checkpoint_type} checkpoint at hour {hour}")
        try:
            component_states = self.registry.get_all_states()
            
            # Serialize config - handle both dataclass and Pydantic
            config_dict = {}
            if hasattr(self.config, 'model_dump'):
                # Pydantic v2
                config_dict = self.config.model_dump()
            elif hasattr(self.config, 'dict'):
                # Pydantic v1
                config_dict = self.config.dict()
            elif hasattr(self.config, '__dataclass_fields__'):
                # Dataclass
                from dataclasses import asdict
                config_dict = asdict(self.config)
            else:
                # Fallback: try to convert to dict
                config_dict = dict(vars(self.config)) if hasattr(self.config, '__dict__') else {}
            
            checkpoint_path = self.state_manager.save_checkpoint(
                hour=hour,
                component_states=component_states,
                metadata={'simulation_config': config_dict}
            )
            logger.debug(f"Checkpoint saved to: {checkpoint_path}")
        except Exception as e:
            logger.error(f"Checkpoint save failed: {e}", exc_info=True)

    def _resume_from_checkpoint(self, checkpoint_path: str) -> None:
        """
        Resume simulation from saved checkpoint.

        Args:
            checkpoint_path (str): Path to checkpoint file.

        Raises:
            SimulationError: If checkpoint restoration fails.
        """
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        try:
            checkpoint_data = self.state_manager.load_checkpoint(checkpoint_path)

            if not self.is_initialized:
                self.initialize()

            for component_id, state in checkpoint_data.get('component_states', {}).items():
                if self.registry.has(component_id):
                    component = self.registry.get(component_id)
                    if hasattr(component, 'restore_state'):
                        component.restore_state(state)
                    else:
                        for key, value in state.items():
                            if hasattr(component, key):
                                setattr(component, key, value)

            self.current_hour = checkpoint_data.get('hour', 0) + 1
            logger.info(f"Resuming from hour {self.current_hour}")
        except Exception as e:
            raise SimulationError(f"Failed to resume from checkpoint: {e}") from e

    def _log_progress(self, current_hour: int, end_hour: int, start_hour: int) -> None:
        """
        Log simulation progress with time estimate.

        Args:
            current_hour (int): Current simulation hour.
            end_hour (int): Final simulation hour.
            start_hour (int): Starting simulation hour.
        """
        total_hours = end_hour - start_hour
        hours_done = current_hour - start_hour
        progress_pct = (hours_done / total_hours) * 100 if total_hours > 0 else 0
        elapsed = time.time() - self.simulation_start_time

        est_total = (elapsed / progress_pct * 100) if progress_pct > 0 else 0
        est_remaining = est_total - elapsed

        logger.info(
            f"Progress: {hours_done}/{total_hours} hours ({progress_pct:.1f}%) | "
            f"Elapsed: {elapsed:.1f}s | Remaining: ~{est_remaining:.1f}s"
        )

    def _generate_results(self, start_hour: int, end_hour: int) -> Dict[str, Any]:
        """
        Generate final simulation results.

        Args:
            start_hour (int): Starting simulation hour.
            end_hour (int): Final simulation hour.

        Returns:
            Dict[str, Any]: Complete results dictionary.
        """
        logger.info("Generating final results...")
        final_states = self.registry.get_all_states()
        metrics = self.monitoring.get_summary()

        results = {
            'simulation': {
                'start_hour': start_hour,
                'end_hour': end_hour,
                'duration_hours': self.steps_run / 60.0,
                'timestep_hours': self.config.timestep_hours,
                'execution_time_seconds': time.time() - self.simulation_start_time
            },
            'final_states': final_states,
            'metrics': metrics
        }

        results_path = self.output_dir / "simulation_results.json"
        self.state_manager.save_results(results, results_path)
        logger.info(f"Results saved to: {results_path}")

        # Generate HTML Dashboard
        try:
            dashboard_data_file = self.monitoring.export_dashboard_data()
            with open(dashboard_data_file, 'r') as f:
                dashboard_data = json.load(f)

            results['dashboard_data'] = dashboard_data

            generator = DashboardGenerator(self.output_dir)
            generator.generate(results)
        except Exception as e:
            logger.error(f"Failed to generate dashboard: {e}")

        return results

    def schedule_event(self, event: Event) -> None:
        """
        Schedule an event for future processing.

        Args:
            event (Event): Event to schedule.
        """
        self.event_scheduler.schedule(event)

    def set_callbacks(
        self,
        pre_step: Optional[Callable[[float], None]] = None,
        post_step: Optional[Callable[[float], None]] = None
    ) -> None:
        """
        Set pre/post step callbacks.

        Args:
            pre_step (Callable, optional): Function called before each step.
            post_step (Callable, optional): Function called after each step.
        """
        self.pre_step_callback = pre_step
        self.post_step_callback = post_step

    def set_dispatch_data(self, prices: np.ndarray, wind: np.ndarray) -> None:
        """
        Set dispatch input data for the simulation.

        Args:
            prices (np.ndarray): Energy price array (EUR/MWh) for simulation period.
            wind (np.ndarray): Wind power offer array (MW) for simulation period.
        """
        self._dispatch_prices = prices
        self._dispatch_wind = wind
        logger.info(f"Dispatch data loaded: {len(prices)} price points, {len(wind)} wind points")

    def initialize_dispatch_strategy(self, context: 'SimulationContext', total_steps: int) -> None:
        """
        Initialize the dispatch strategy with context and pre-allocate arrays.

        Args:
            context (SimulationContext): Physics configuration context.
            total_steps (int): Total number of timesteps for pre-allocation.
        """
        if self.dispatch_strategy:
            self.dispatch_strategy.initialize(
                registry=self.registry,
                context=context,
                total_steps=total_steps
            )
            logger.info("Dispatch strategy initialized")

    def get_dispatch_history(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Get recorded history from dispatch strategy.

        Returns:
            Dict[str, np.ndarray]: History arrays, or None if not configured.
        """
        if self.dispatch_strategy and hasattr(self.dispatch_strategy, 'get_history'):
            return self.dispatch_strategy.get_history()
        return None
