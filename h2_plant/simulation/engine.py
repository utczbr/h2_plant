"""
Modular simulation engine for hydrogen production system.

Replaces monolithic Finalsimulation.py with flexible, extensible
execution framework supporting:
- Event-driven execution
- State persistence and checkpointing
- Comprehensive monitoring
- Graceful error handling
"""

import logging
from typing import Optional, Dict, Any, Callable, List
from pathlib import Path
import time
import json

from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.exceptions import SimulationError
from h2_plant.config.plant_config import SimulationConfig, ConnectionConfig, IndexedConnectionConfig
from h2_plant.simulation.state_manager import StateManager
from h2_plant.simulation.event_scheduler import EventScheduler, Event
from h2_plant.simulation.monitoring import MonitoringSystem
from h2_plant.simulation.flow_network import FlowNetwork
from h2_plant.visualization.dashboard_generator import DashboardGenerator

logger = logging.getLogger(__name__)


class SimulationEngine:
    """
    Main simulation orchestrator.
    
    Coordinates simulation execution with event processing,
    checkpointing, and monitoring.
    """
    
    def __init__(
        self,
        registry: ComponentRegistry,
        config: SimulationConfig,
        output_dir: Optional[Path] = None,
        topology: List[ConnectionConfig] = None,
        indexed_topology: List[IndexedConnectionConfig] = None
    ):
        """
        Initialize simulation engine.
        """
        self.registry = registry
        self.config = config
        self.output_dir = output_dir or Path("simulation_output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Subsystems
        self.state_manager = StateManager(output_dir=self.output_dir)
        self.event_scheduler = EventScheduler()
        self.monitoring = MonitoringSystem(output_dir=self.output_dir)
        
        # Initialize FlowNetwork
        self.flow_network = FlowNetwork(
            registry=self.registry, 
            topology=topology or [],
            indexed_topology=indexed_topology or []
        )
        
        # State
        self.current_hour: int = 0
        self.is_initialized: bool = False
        self.is_running: bool = False
        self.simulation_start_time: float = 0.0
        self.steps_run: int = 0
        
        # Callbacks
        self.pre_step_callback: Optional[Callable[[int], None]] = None
        self.post_step_callback: Optional[Callable[[int], None]] = None
    
    def initialize(self) -> None:
        """
        Initialize simulation engine and all components.
        """
        logger.info("Initializing simulation engine...")
        try:
            # Pass timestep in hours (as documented in Component.initialize)
            # For 1-minute step, override config if needed or ensure config is set to 1/60
            # We force 1/60 here for the new dynamics
            dt_hours = 1.0 / 60.0
            self.config.timestep_hours = dt_hours # Update config to match
            self.registry.initialize_all(dt=dt_hours)
            self.flow_network.initialize()
            self.monitoring.initialize(self.registry)
            self.is_initialized = True
            logger.info(f"Simulation engine initialized successfully (dt={dt_hours} hours)")
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
        """
        effective_start_hour = start_hour if start_hour is not None else self.config.start_hour
        effective_end_hour = end_hour if end_hour is not None else (effective_start_hour + self.config.duration_hours)
        
        # Convert hours to steps (minutes)
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
                # Calculate current hour (fractional)
                hour = step / steps_per_hour
                self.current_hour = hour
                
                self._execute_timestep(hour)
                self.steps_run += 1
                
                # Checkpoint logic (based on hours)
                if step % steps_per_hour == 0: # Every hour
                    int_hour = int(hour)
                    if self._should_checkpoint(int_hour):
                        self._save_checkpoint(int_hour)
                    
                    if int_hour > 0 and int_hour % 168 == 0:
                        self._log_progress(int_hour, effective_end_hour, effective_start_hour)
                        
                # Sparse logging: We rely on monitoring system to handle high frequency data
                # But if we were appending to a list here, we would limit it.
                # The monitoring system likely collects every step. We should probably tell it to aggregate?
                # For now, we assume monitoring handles it or we accept large data for short runs.
                # If running full year, we MUST ensure monitoring is sparse.
                # This requires updating MonitoringSystem or just calling collect every 60 steps.
                
                # Monitoring: Collect every step for accurate integration
                    self.monitoring.collect(hour, self.registry)
            
            self.is_running = False
            elapsed_time = time.time() - self.simulation_start_time
            
            # Calculate rate safely handling division by zero
            rate = self.steps_run / elapsed_time if elapsed_time > 0 else float('inf')

            logger.info(
                f"Simulation complete: {self.steps_run} hours in "
                f"{elapsed_time:.2f} seconds ({rate:.1f} hours/sec)"
            )
            
            results = self._generate_results(start_hour=effective_start_hour, end_hour=effective_end_hour)
            return results
            
        except KeyboardInterrupt:
            logger.warning("Simulation interrupted by user")
            self._save_checkpoint(self.current_hour, emergency=True)
            raise
        
        except Exception as e:
            logger.error(f"Simulation failed at hour {self.current_hour}: {e}", exc_info=True)
            self._save_checkpoint(self.current_hour, emergency=True)
            raise SimulationError(f"Simulation execution failed: {e}") from e
    
    def _execute_timestep(self, hour: int) -> None:
        if self.pre_step_callback:
            self.pre_step_callback(hour)
        
        self.event_scheduler.process_events(hour, self.registry)
        
        # Causal Execution Order
        # 1. External Inputs & Utility
        # 2. Production (Generates heat/mass)
        # 3. Thermal Management (Distributes heat)
        # 4. Cooling (Responds to heat)
        # 5. Separation & Processing
        # 6. Compression
        # 7. Storage
        # 8. Logistics/Metrics
        
        execution_order = [
            'energy_price_tracker', 'demand_scheduler', 'environment_manager',  # Utility
            'dual_path_coordinator',                                            # Coordination
            'soec_cluster', 'pem_electrolyzer_detailed',                        # Production (SOEC first, then PEM)
            'thermal_manager',                                                  # Thermal Distribution
            'chiller', 'chiller_hx5', 'chiller_hx6',                            # Cooling
            'separator_sp3', 'psa_d3',                                          # Processing
            'compressor_c1', 'compressor_c2', 'filling_compressor',             # Compression
            'lp_tanks', 'hp_tanks', 'h2_storage_enhanced',                      # Storage
            'logistics_manager'                                                 # Logistics
        ]
        
        # Execute ordered components first
        executed_ids = set()
        executed_instances = set()
        
        for comp_id in execution_order:
            if self.registry.has(comp_id):
                try:
                    component = self.registry.get(comp_id)
                    if component not in executed_instances:
                        component.step(hour)
                        executed_instances.add(component)
                        executed_ids.add(comp_id)
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

        # Execute flows after components have updated their states
        # Note: In a strictly causal model, flows might need to be updated interleaved.
        # But FlowNetwork usually propagates steady-state flows based on port values.
        try:
            self.flow_network.execute_flows(hour)
        except Exception as e:
            logger.error(f"Flow execution failed at hour {hour}: {e}")
            raise
        
        if self.post_step_callback:
            self.post_step_callback(hour)
    
    def _should_checkpoint(self, hour: int) -> bool:
        interval = self.config.checkpoint_interval_hours
        if interval <= 0:
            return False
        if hour == self.config.start_hour and not self.steps_run > 1:
             return False
        return hour % interval == 0

    def _save_checkpoint(self, hour: int, emergency: bool = False) -> None:
        checkpoint_type = "emergency" if emergency else "regular"
        logger.info(f"Saving {checkpoint_type} checkpoint at hour {hour}")
        try:
            component_states = self.registry.get_all_states()
            from dataclasses import asdict
            checkpoint_path = self.state_manager.save_checkpoint(
                hour=hour,
                component_states=component_states,
                metadata={'simulation_config': asdict(self.config)}
            )
            logger.debug(f"Checkpoint saved to: {checkpoint_path}")
        except Exception as e:
            logger.error(f"Checkpoint save failed: {e}", exc_info=True)

    def _resume_from_checkpoint(self, checkpoint_path: str) -> None:
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
        logger.info("Generating final results...")
        final_states = self.registry.get_all_states()
        metrics = self.monitoring.get_summary()
        
        results = {
            'simulation': {
                'start_hour': start_hour,
                'end_hour': end_hour,
                'duration_hours': self.steps_run,
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
            # We need to add the dashboard-specific data structure
            dashboard_data_file = self.monitoring.export_dashboard_data()
            with open(dashboard_data_file, 'r') as f:
                dashboard_data = json.load(f)
            
            # Merge into main results for the generator
            results['dashboard_data'] = dashboard_data
            
            generator = DashboardGenerator(self.output_dir)
            generator.generate(results)
        except Exception as e:
            logger.error(f"Failed to generate dashboard: {e}")
            
        return results

    def schedule_event(self, event: Event) -> None:
        self.event_scheduler.schedule(event)

    def set_callbacks(
        self,
        pre_step: Optional[Callable[[int], None]] = None,
        post_step: Optional[Callable[[int], None]] = None
    ) -> None:
        self.pre_step_callback = pre_step
        self.post_step_callback = post_step
