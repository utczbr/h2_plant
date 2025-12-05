# STEP 3: Technical Specification - Simulation Engine

***

# 06_Simulation_Engine_Specification.md

**Document:** Simulation Engine Technical Specification  
**Project:** Dual-Path Hydrogen Production System - Modular Refactoring v2.0  
**Date:** November 18, 2025  
**Layer:** Layer 5 - Simulation Execution  
**Priority:** MEDIUM  
**Dependencies:** Layers 1-4 (All previous layers)

***

## 1. Overview

### 1.1 Purpose

This specification defines the **modular simulation engine** that replaces the monolithic `Finalsimulation.py` with a flexible, extensible execution framework. The simulation engine addresses critical gaps identified in the critique: lack of modularity, missing state persistence, and absence of event scheduling capabilities.

**Key Objectives:**
- Replace monolithic simulation loop with modular `SimulationEngine`
- Implement state checkpointing and resume capabilities
- Add event scheduling for maintenance, price updates, and demand shifts
- Provide comprehensive monitoring and logging infrastructure
- Enable simulation orchestration with minimal boilerplate

**Critique Remediation:**
- **FAIL → PASS:** "Finalsimulation.py lacks modularity" (Section 2)
- **FAIL → PASS:** "No state persistence and checkpointing" (Section 3)
- **FAIL → PASS:** "No event scheduling" (Section 4)

***

### 1.2 Architecture Overview

```
SimulationEngine
├── ComponentRegistry (from Layer 1)
├── StateManager (checkpointing & resume)
├── EventScheduler (scheduled events)
├── MonitoringSystem (metrics & logging)
└── SimulationConfig (from Layer 4)

Execution Flow:
1. Load configuration
2. Build plant (PlantBuilder)
3. Initialize simulation engine
4. Run simulation loop
   - Process scheduled events
   - Step all components
   - Collect metrics
   - Save checkpoints
5. Generate final report
```

***

### 1.3 Scope

**In Scope:**
- `simulation/engine.py`: Main simulation orchestrator
- `simulation/state_manager.py`: Checkpointing and state persistence
- `simulation/event_scheduler.py`: Event scheduling infrastructure
- `simulation/monitoring.py`: Metrics collection and logging
- `simulation/runner.py`: High-level simulation runner utilities

**Out of Scope:**
- Real-time visualization (future enhancement)
- Distributed simulation (future enhancement)
- Optimization loops (separate optimization module)

***

### 1.4 Design Principles

1. **Separation of Concerns:** Engine orchestrates, components execute
2. **Fail-Safe Operation:** Errors logged, simulation continues where possible
3. **Observability:** Comprehensive metrics and state transparency
4. **Extensibility:** Easy to add new event types and monitors
5. **Performance:** Minimal overhead on component execution

***

## 2. Simulation Engine Core

### 2.1 Design Rationale

**Problem (from critique):**
```python
# Monolithic Finalsimulation.py - hardcoded, inflexible
for t in range(8760):
    # Hardcoded logic
    h2_demand = get_demand(t)
    electrolyzer_power = calculate_power(energy_price[t])
    # ... 200+ lines of tangled logic
    # No checkpointing, no event handling, no monitoring
```

**Solution:**
```python
# Modular SimulationEngine
engine = SimulationEngine(registry, config)
engine.run(start_hour=0, end_hour=8760)

# Automatic:
# - Component orchestration via registry
# - Event processing
# - Checkpointing
# - Monitoring
```

***

### 2.2 Implementation

**File:** `h2_plant/simulation/engine.py`

```python
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
from typing import Optional, Dict, Any, Callable
from pathlib import Path
import time

from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.exceptions import SimulationError
from h2_plant.config.plant_config import SimulationConfig
from h2_plant.simulation.state_manager import StateManager
from h2_plant.simulation.event_scheduler import EventScheduler, Event
from h2_plant.simulation.monitoring import MonitoringSystem

logger = logging.getLogger(__name__)


class SimulationEngine:
    """
    Main simulation orchestrator.
    
    Coordinates simulation execution with event processing,
    checkpointing, and monitoring.
    
    Example:
        # Setup
        registry = ComponentRegistry()
        # ... register components ...
        
        config = SimulationConfig(
            timestep_hours=1.0,
            duration_hours=8760,
            checkpoint_interval_hours=168
        )
        
        # Create engine
        engine = SimulationEngine(registry, config)
        
        # Run simulation
        results = engine.run()
    """
    
    def __init__(
        self,
        registry: ComponentRegistry,
        config: SimulationConfig,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize simulation engine.
        
        Args:
            registry: ComponentRegistry with all plant components
            config: Simulation configuration
            output_dir: Directory for outputs (checkpoints, logs, results)
        """
        self.registry = registry
        self.config = config
        self.output_dir = output_dir or Path("simulation_output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Subsystems
        self.state_manager = StateManager(output_dir=self.output_dir)
        self.event_scheduler = EventScheduler()
        self.monitoring = MonitoringSystem(output_dir=self.output_dir)
        
        # State
        self.current_hour: int = 0
        self.is_initialized: bool = False
        self.is_running: bool = False
        self.simulation_start_time: float = 0.0
        
        # Callbacks (optional user-defined hooks)
        self.pre_step_callback: Optional[Callable[[int], None]] = None
        self.post_step_callback: Optional[Callable[[int], None]] = None
    
    def initialize(self) -> None:
        """
        Initialize simulation engine and all components.
        
        Raises:
            SimulationError: If initialization fails
        """
        logger.info("Initializing simulation engine...")
        
        try:
            # Initialize component registry
            self.registry.initialize_all(dt=self.config.timestep_hours)
            
            # Initialize monitoring
            self.monitoring.initialize(self.registry)
            
            self.is_initialized = True
            logger.info("Simulation engine initialized successfully")
            
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
        
        Args:
            start_hour: Starting hour (default: config.start_hour)
            end_hour: Ending hour (default: start + duration)
            resume_from_checkpoint: Path to checkpoint file to resume from
            
        Returns:
            Dictionary with simulation results and metrics
            
        Raises:
            SimulationError: If simulation execution fails
        """
        # Determine time range
        start_hour = start_hour or self.config.start_hour
        end_hour = end_hour or (start_hour + self.config.duration_hours)
        
        # Resume from checkpoint if requested
        if resume_from_checkpoint:
            self._resume_from_checkpoint(resume_from_checkpoint)
            start_hour = self.current_hour
        
        # Initialize if not already done
        if not self.is_initialized:
            self.initialize()
        
        logger.info(f"Starting simulation: hours {start_hour} to {end_hour}")
        self.simulation_start_time = time.time()
        self.is_running = True
        
        try:
            # Main simulation loop
            for hour in range(start_hour, end_hour):
                self.current_hour = hour
                
                # Execute single timestep
                self._execute_timestep(hour)
                
                # Periodic checkpointing
                if self._should_checkpoint(hour):
                    self._save_checkpoint(hour)
                
                # Progress logging
                if hour > 0 and hour % 168 == 0:  # Weekly
                    self._log_progress(hour, end_hour)
            
            self.is_running = False
            elapsed_time = time.time() - self.simulation_start_time
            
            logger.info(
                f"Simulation complete: {end_hour - start_hour} hours in "
                f"{elapsed_time:.2f} seconds ({(end_hour - start_hour)/elapsed_time:.1f} hours/sec)"
            )
            
            # Generate final results
            results = self._generate_results()
            
            return results
            
        except KeyboardInterrupt:
            logger.warning("Simulation interrupted by user")
            self._save_checkpoint(self.current_hour, emergency=True)
            raise
        
        except Exception as e:
            logger.error(f"Simulation failed at hour {self.current_hour}: {e}")
            self._save_checkpoint(self.current_hour, emergency=True)
            raise SimulationError(f"Simulation execution failed: {e}") from e
    
    def _execute_timestep(self, hour: int) -> None:
        """
        Execute single simulation timestep.
        
        Args:
            hour: Current simulation hour
        """
        # Pre-step callback
        if self.pre_step_callback:
            self.pre_step_callback(hour)
        
        # Process scheduled events
        self.event_scheduler.process_events(hour, self.registry)
        
        # Step all components
        try:
            self.registry.step_all(hour)
        except Exception as e:
            logger.error(f"Component step failed at hour {hour}: {e}")
            raise
        
        # Collect monitoring data
        self.monitoring.collect(hour, self.registry)
        
        # Post-step callback
        if self.post_step_callback:
            self.post_step_callback(hour)
    
    def _should_checkpoint(self, hour: int) -> bool:
        """Determine if checkpoint should be saved at this hour."""
        if hour == 0:
            return False  # Don't checkpoint at start
        
        return hour % self.config.checkpoint_interval_hours == 0
    
    def _save_checkpoint(self, hour: int, emergency: bool = False) -> None:
        """
        Save simulation checkpoint.
        
        Args:
            hour: Current simulation hour
            emergency: If True, mark as emergency checkpoint
        """
        checkpoint_type = "emergency" if emergency else "regular"
        logger.info(f"Saving {checkpoint_type} checkpoint at hour {hour}")
        
        try:
            # Get state from all components
            component_states = self.registry.get_all_states()
            
            # Save checkpoint
            checkpoint_path = self.state_manager.save_checkpoint(
                hour=hour,
                component_states=component_states,
                metadata={
                    'checkpoint_type': checkpoint_type,
                    'simulation_config': {
                        'timestep_hours': self.config.timestep_hours,
                        'total_duration_hours': self.config.duration_hours
                    }
                }
            )
            
            logger.debug(f"Checkpoint saved to: {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Checkpoint save failed: {e}")
            # Don't raise - checkpointing failure shouldn't stop simulation
    
    def _resume_from_checkpoint(self, checkpoint_path: str) -> None:
        """
        Resume simulation from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        
        try:
            checkpoint_data = self.state_manager.load_checkpoint(checkpoint_path)
            
            # Restore component states
            component_states = checkpoint_data['component_states']
            
            for component_id, state in component_states.items():
                if self.registry.has(component_id):
                    component = self.registry.get(component_id)
                    # Components need restore_state() method for this
                    if hasattr(component, 'restore_state'):
                        component.restore_state(state)
            
            # Restore simulation state
            self.current_hour = checkpoint_data['hour']
            
            logger.info(f"Resumed from hour {self.current_hour}")
            
        except Exception as e:
            raise SimulationError(f"Failed to resume from checkpoint: {e}") from e
    
    def _log_progress(self, current_hour: int, end_hour: int) -> None:
        """Log simulation progress."""
        progress_pct = (current_hour / end_hour) * 100
        elapsed = time.time() - self.simulation_start_time
        est_total = elapsed / (current_hour / end_hour) if current_hour > 0 else 0
        est_remaining = est_total - elapsed
        
        logger.info(
            f"Progress: {current_hour}/{end_hour} hours ({progress_pct:.1f}%) | "
            f"Elapsed: {elapsed:.1f}s | Remaining: ~{est_remaining:.1f}s"
        )
    
    def _generate_results(self) -> Dict[str, Any]:
        """
        Generate final simulation results.
        
        Returns:
            Dictionary with aggregated metrics and component states
        """
        logger.info("Generating final results...")
        
        # Get final component states
        final_states = self.registry.get_all_states()
        
        # Get monitoring data
        metrics = self.monitoring.get_summary()
        
        # Aggregate results
        results = {
            'simulation': {
                'start_hour': self.config.start_hour,
                'end_hour': self.current_hour,
                'duration_hours': self.current_hour - self.config.start_hour,
                'timestep_hours': self.config.timestep_hours,
                'execution_time_seconds': time.time() - self.simulation_start_time
            },
            'final_states': final_states,
            'metrics': metrics
        }
        
        # Save results to file
        results_path = self.output_dir / "simulation_results.json"
        self.state_manager.save_results(results, results_path)
        
        logger.info(f"Results saved to: {results_path}")
        
        return results
    
    def schedule_event(self, event: Event) -> None:
        """
        Schedule an event for future execution.
        
        Args:
            event: Event to schedule
        """
        self.event_scheduler.schedule(event)
    
    def set_callbacks(
        self,
        pre_step: Optional[Callable[[int], None]] = None,
        post_step: Optional[Callable[[int], None]] = None
    ) -> None:
        """
        Set callback functions for pre/post step hooks.
        
        Args:
            pre_step: Function called before each timestep
            post_step: Function called after each timestep
        """
        self.pre_step_callback = pre_step
        self.post_step_callback = post_step
```

***

## 3. State Management

### 3.1 Checkpointing System

**File:** `h2_plant/simulation/state_manager.py`

```python
"""
State management for simulation checkpointing and persistence.

Handles:
- Checkpoint creation and loading
- State serialization (HDF5, JSON, Parquet)
- Results export
"""

import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from datetime import datetime

try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False
    logging.warning("h5py not available - HDF5 checkpoints disabled")

logger = logging.getLogger(__name__)


class StateManager:
    """
    Manages simulation state persistence and checkpointing.
    
    Example:
        manager = StateManager(output_dir=Path("checkpoints"))
        
        # Save checkpoint
        manager.save_checkpoint(
            hour=168,
            component_states=registry.get_all_states()
        )
        
        # Load checkpoint
        data = manager.load_checkpoint("checkpoint_hour_168.json")
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize state manager.
        
        Args:
            output_dir: Directory for checkpoint storage
        """
        self.output_dir = output_dir
        self.checkpoint_dir = output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(
        self,
        hour: int,
        component_states: Dict[str, Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
        format: str = "json"
    ) -> Path:
        """
        Save simulation checkpoint.
        
        Args:
            hour: Current simulation hour
            component_states: State from all components
            metadata: Additional metadata to store
            format: Storage format ('json', 'pickle', 'hdf5')
            
        Returns:
            Path to saved checkpoint file
        """
        timestamp = datetime.now().isoformat()
        
        checkpoint_data = {
            'hour': hour,
            'timestamp': timestamp,
            'component_states': component_states,
            'metadata': metadata or {}
        }
        
        filename = f"checkpoint_hour_{hour}.{format}"
        checkpoint_path = self.checkpoint_dir / filename
        
        if format == "json":
            self._save_json(checkpoint_data, checkpoint_path)
        elif format == "pickle":
            self._save_pickle(checkpoint_data, checkpoint_path)
        elif format == "hdf5" and HDF5_AVAILABLE:
            self._save_hdf5(checkpoint_data, checkpoint_path)
        else:
            raise ValueError(f"Unsupported checkpoint format: {format}")
        
        logger.debug(f"Checkpoint saved: {checkpoint_path}")
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: Path | str) -> Dict[str, Any]:
        """
        Load simulation checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Checkpoint data dictionary
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        format = checkpoint_path.suffix.lstrip('.')
        
        if format == "json":
            return self._load_json(checkpoint_path)
        elif format == "pickle":
            return self._load_pickle(checkpoint_path)
        elif format == "hdf5" and HDF5_AVAILABLE:
            return self._load_hdf5(checkpoint_path)
        else:
            raise ValueError(f"Unsupported checkpoint format: {format}")
    
    def save_results(self, results: Dict[str, Any], output_path: Path) -> None:
        """
        Save final simulation results.
        
        Args:
            results: Results dictionary
            output_path: Path to save results
        """
        self._save_json(results, output_path)
        logger.info(f"Results saved to: {output_path}")
    
    def _save_json(self, data: Dict[str, Any], path: Path) -> None:
        """Save data as JSON."""
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_json(self, path: Path) -> Dict[str, Any]:
        """Load data from JSON."""
        with open(path, 'r') as f:
            return json.load(f)
    
    def _save_pickle(self, data: Dict[str, Any], path: Path) -> None:
        """Save data as pickle."""
        with open(path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _load_pickle(self, path: Path) -> Dict[str, Any]:
        """Load data from pickle."""
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def _save_hdf5(self, data: Dict[str, Any], path: Path) -> None:
        """Save data as HDF5 (requires h5py)."""
        with h5py.File(path, 'w') as f:
            f.attrs['hour'] = data['hour']
            f.attrs['timestamp'] = data['timestamp']
            
            # Store component states as groups
            for comp_id, comp_state in data['component_states'].items():
                group = f.create_group(comp_id)
                for key, value in comp_state.items():
                    group.attrs[key] = value
    
    def _load_hdf5(self, path: Path) -> Dict[str, Any]:
        """Load data from HDF5."""
        data = {'component_states': {}}
        
        with h5py.File(path, 'r') as f:
            data['hour'] = f.attrs['hour']
            data['timestamp'] = f.attrs['timestamp']
            
            for comp_id in f.keys():
                group = f[comp_id]
                data['component_states'][comp_id] = dict(group.attrs)
        
        return data
    
    def list_checkpoints(self) -> list[Path]:
        """
        List all available checkpoints.
        
        Returns:
            List of checkpoint file paths
        """
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_hour_*.json"))
        checkpoints.extend(sorted(self.checkpoint_dir.glob("checkpoint_hour_*.pickle")))
        checkpoints.extend(sorted(self.checkpoint_dir.glob("checkpoint_hour_*.hdf5")))
        
        return checkpoints
```

***

## 4. Event Scheduling

### 4.1 Event System

**File:** `h2_plant/simulation/event_scheduler.py`

```python
"""
Event scheduling system for simulation.

Supports:
- Time-based events (executed at specific hours)
- Recurring events (periodic execution)
- Component modification events
- Custom event handlers
"""

from dataclasses import dataclass
from typing import Callable, Dict, Any, List, Optional
import logging
from abc import ABC, abstractmethod

from h2_plant.core.component_registry import ComponentRegistry

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """
    Simulation event.
    
    Example:
        # Component state change event
        event = Event(
            hour=100,
            event_type="maintenance",
            handler=lambda reg: reg.get("electrolyzer").set_state(ProductionState.MAINTENANCE),
            metadata={"component": "electrolyzer", "duration": 24}
        )
    """
    hour: int                                    # Hour to execute event
    event_type: str                              # Event type identifier
    handler: Callable[[ComponentRegistry], None] # Event execution function
    metadata: Dict[str, Any] = None             # Additional event data
    recurring: bool = False                      # If True, reschedule after execution
    recurrence_interval: int = 0                 # Hours between recurrences


class EventScheduler:
    """
    Manages scheduled events during simulation.
    
    Example:
        scheduler = EventScheduler()
        
        # Schedule maintenance
        scheduler.schedule(Event(
            hour=1000,
            event_type="maintenance",
            handler=lambda reg: reg.get("electrolyzer").shutdown()
        ))
        
        # Process events each timestep
        for hour in range(8760):
            scheduler.process_events(hour, registry)
    """
    
    def __init__(self):
        """Initialize event scheduler."""
        self._events: List[Event] = []
    
    def schedule(self, event: Event) -> None:
        """
        Schedule an event.
        
        Args:
            event: Event to schedule
        """
        self._events.append(event)
        self._events.sort(key=lambda e: e.hour)  # Keep sorted by hour
        
        logger.debug(f"Scheduled {event.event_type} event at hour {event.hour}")
    
    def process_events(self, current_hour: int, registry: ComponentRegistry) -> None:
        """
        Process all events scheduled for current hour.
        
        Args:
            current_hour: Current simulation hour
            registry: ComponentRegistry for event handlers
        """
        # Find events for this hour
        events_to_execute = [e for e in self._events if e.hour == current_hour]
        
        if not events_to_execute:
            return
        
        logger.info(f"Processing {len(events_to_execute)} events at hour {current_hour}")
        
        for event in events_to_execute:
            try:
                # Execute event handler
                event.handler(registry)
                
                logger.info(f"Executed {event.event_type} event")
                
                # Handle recurring events
                if event.recurring and event.recurrence_interval > 0:
                    next_event = Event(
                        hour=current_hour + event.recurrence_interval,
                        event_type=event.event_type,
                        handler=event.handler,
                        metadata=event.metadata,
                        recurring=True,
                        recurrence_interval=event.recurrence_interval
                    )
                    self.schedule(next_event)
                
            except Exception as e:
                logger.error(f"Event {event.event_type} failed at hour {current_hour}: {e}")
                # Continue processing other events
        
        # Remove executed non-recurring events
        self._events = [e for e in self._events if e.hour != current_hour or e.recurring]
    
    def get_pending_events(self) -> List[Event]:
        """Return list of all pending events."""
        return self._events.copy()
    
    def clear_events(self) -> None:
        """Clear all scheduled events."""
        self._events.clear()


# Common event factories

def create_maintenance_event(
    hour: int,
    component_id: str,
    duration_hours: int
) -> Event:
    """
    Create component maintenance event.
    
    Args:
        hour: Hour to start maintenance
        component_id: Component to put in maintenance
        duration_hours: Maintenance duration
        
    Returns:
        Maintenance start event
    """
    def handler(registry: ComponentRegistry):
        component = registry.get(component_id)
        if hasattr(component, 'state'):
            from h2_plant.core.enums import ProductionState
            component.state = ProductionState.MAINTENANCE
            logger.info(f"Component {component_id} entering maintenance")
    
    return Event(
        hour=hour,
        event_type="maintenance_start",
        handler=handler,
        metadata={"component": component_id, "duration": duration_hours}
    )


def create_price_update_event(
    hour: int,
    new_price: float
) -> Event:
    """
    Create energy price update event.
    
    Args:
        hour: Hour to update price
        new_price: New energy price ($/MWh)
        
    Returns:
        Price update event
    """
    def handler(registry: ComponentRegistry):
        if registry.has('energy_price_tracker'):
            tracker = registry.get('energy_price_tracker')
            tracker.current_price_per_mwh = new_price
            logger.info(f"Energy price updated to ${new_price:.2f}/MWh")
    
    return Event(
        hour=hour,
        event_type="price_update",
        handler=handler,
        metadata={"new_price": new_price}
    )
```

***

## 5. Monitoring System

### 5.1 Metrics Collection

**File:** `h2_plant/simulation/monitoring.py`

```python
"""
Monitoring and metrics collection system.

Tracks:
- Component-level metrics (production rates, storage levels, costs)
- System-level aggregates
- Time-series data
- Performance indicators (KPIs)
"""

import logging
from pathlib import Path
from typing import Dict, Any, List
import csv
import json

from h2_plant.core.component_registry import ComponentRegistry

logger = logging.getLogger(__name__)


class MonitoringSystem:
    """
    Collects and aggregates simulation metrics.
    
    Example:
        monitor = MonitoringSystem(output_dir=Path("outputs"))
        monitor.initialize(registry)
        
        # Each timestep
        for hour in range(8760):
            registry.step_all(hour)
            monitor.collect(hour, registry)
        
        # Get summary
        summary = monitor.get_summary()
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize monitoring system.
        
        Args:
            output_dir: Directory for metrics output
        """
        self.output_dir = output_dir
        self.metrics_dir = output_dir / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Time-series data storage
        self.timeseries: Dict[str, List[Any]] = {
            'hour': [],
            'total_production_kg': [],
            'total_storage_kg': [],
            'total_demand_kg': [],
            'energy_price_mwh': [],
            'total_cost': []
        }
        
        # Component-specific metrics
        self.component_metrics: Dict[str, Dict[str, List]] = {}
        
        # Aggregate statistics
        self.total_production_kg = 0.0
        self.total_demand_kg = 0.0
        self.total_energy_kwh = 0.0
        self.total_cost = 0.0
    
    def initialize(self, registry: ComponentRegistry) -> None:
        """
        Initialize monitoring for registered components.
        
        Args:
            registry: ComponentRegistry to monitor
        """
        # Setup component-specific metric tracking
        for component_id in registry.get_all_ids():
            self.component_metrics[component_id] = {}
        
        logger.info("Monitoring system initialized")
    
    def collect(self, hour: int, registry: ComponentRegistry) -> None:
        """
        Collect metrics for current timestep.
        
        Args:
            hour: Current simulation hour
            registry: ComponentRegistry to collect from
        """
        # Collect component states
        states = registry.get_all_states()
        
        # Time-series tracking
        self.timeseries['hour'].append(hour)
        
        # Aggregate production
        total_production = 0.0
        for comp_id, state in states.items():
            if 'h2_output_kg' in state:
                total_production += state['h2_output_kg']
        
        self.timeseries['total_production_kg'].append(total_production)
        self.total_production_kg += total_production
        
        # Aggregate storage
        total_storage = 0.0
        for comp_id, state in states.items():
            if 'total_mass_kg' in state:
                total_storage += state['total_mass_kg']
            elif 'mass_kg' in state:
                total_storage += state['mass_kg']
        
        self.timeseries['total_storage_kg'].append(total_storage)
        
        # Demand tracking
        if registry.has('demand_scheduler'):
            demand = registry.get('demand_scheduler')
            demand_state = demand.get_state()
            current_demand = demand_state.get('current_demand_kg', 0.0)
            self.timeseries['total_demand_kg'].append(current_demand)
            self.total_demand_kg += current_demand
        
        # Energy price tracking
        if registry.has('energy_price_tracker'):
            price_tracker = registry.get('energy_price_tracker')
            price_state = price_tracker.get_state()
            self.timeseries['energy_price_mwh'].append(
                price_state.get('current_price_per_mwh', 0.0)
            )
        
        # Cost tracking
        total_cost = 0.0
        for comp_id, state in states.items():
            if 'cumulative_cost' in state:
                total_cost = state['cumulative_cost']  # Use latest cumulative
        
        self.timeseries['total_cost'].append(total_cost)
        self.total_cost = total_cost
        
        # Component-specific metrics
        for comp_id, state in states.items():
            if comp_id not in self.component_metrics:
                self.component_metrics[comp_id] = {}
            
            for metric_name, metric_value in state.items():
                if metric_name not in self.component_metrics[comp_id]:
                    self.component_metrics[comp_id][metric_name] = []
                
                self.component_metrics[comp_id][metric_name].append(metric_value)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics.
        
        Returns:
            Dictionary with aggregate metrics
        """
        summary = {
            'total_production_kg': self.total_production_kg,
            'total_demand_kg': self.total_demand_kg,
            'total_cost': self.total_cost,
            'average_cost_per_kg': (
                self.total_cost / self.total_production_kg
                if self.total_production_kg > 0 else 0.0
            ),
            'demand_fulfillment_rate': (
                self.total_production_kg / self.total_demand_kg
                if self.total_demand_kg > 0 else 0.0
            )
        }
        
        return summary
    
    def export_timeseries(self, filename: str = "timeseries.csv") -> Path:
        """
        Export time-series data to CSV.
        
        Args:
            filename: Output filename
            
        Returns:
            Path to exported file
        """
        output_path = self.metrics_dir / filename
        
        # Convert to rows
        num_rows = len(self.timeseries['hour'])
        headers = list(self.timeseries.keys())
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            
            for i in range(num_rows):
                row = [self.timeseries[header][i] for header in headers]
                writer.writerow(row)
        
        logger.info(f"Time-series data exported to: {output_path}")
        return output_path
    
    def export_summary(self, filename: str = "summary.json") -> Path:
        """
        Export summary statistics to JSON.
        
        Args:
            filename: Output filename
            
        Returns:
            Path to exported file
        """
        output_path = self.metrics_dir / filename
        summary = self.get_summary()
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary exported to: {output_path}")
        return output_path
```

***

## 6. Simulation Runner

### 6.1 High-Level Runner Utility

**File:** `h2_plant/simulation/runner.py`

```python
"""
High-level simulation runner utilities.

Provides convenient wrappers for common simulation scenarios.
"""

from pathlib import Path
from typing import Optional, Dict, Any
import logging

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
    
    One-line simulation execution:
    >>> results = run_simulation_from_config("configs/plant_baseline.yaml")
    
    Args:
        config_path: Path to plant configuration YAML/JSON
        output_dir: Directory for outputs (default: ./simulation_output)
        resume_from: Path to checkpoint to resume from
        
    Returns:
        Simulation results dictionary
    """
    logger.info(f"Running simulation from config: {config_path}")
    
    # Build plant from configuration
    plant = PlantBuilder.from_file(config_path)
    registry = plant.registry
    
    # Create simulation engine
    engine = SimulationEngine(
        registry=registry,
        config=plant.config.simulation,
        output_dir=output_dir or Path("simulation_output")
    )
    
    # Run simulation
    results = engine.run(resume_from_checkpoint=resume_from)
    
    # Export metrics
    engine.monitoring.export_timeseries()
    engine.monitoring.export_summary()
    
    return results


def run_scenario_comparison(
    config_paths: list[Path | str],
    output_dir: Optional[Path] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Run multiple scenarios and compare results.
    
    Example:
        scenarios = [
            "configs/plant_baseline.yaml",
            "configs/plant_grid_only.yaml",
            "configs/plant_pilot.yaml"
        ]
        
        results = run_scenario_comparison(scenarios)
    
    Args:
        config_paths: List of configuration file paths
        output_dir: Base directory for outputs
        
    Returns:
        Dictionary mapping config name to results
    """
    output_dir = output_dir or Path("scenario_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    for config_path in config_paths:
        config_name = Path(config_path).stem
        scenario_dir = output_dir / config_name
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Running scenario: {config_name}")
        logger.info(f"{'='*60}\n")
        
        results = run_simulation_from_config(
            config_path=config_path,
            output_dir=scenario_dir
        )
        
        all_results[config_name] = results
    
    # Generate comparison report
    _generate_comparison_report(all_results, output_dir)
    
    return all_results


def _generate_comparison_report(
    results: Dict[str, Dict[str, Any]],
    output_dir: Path
) -> None:
    """Generate comparison report across scenarios."""
    import json
    
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
    
    logger.info(f"\nScenario comparison saved to: {report_path}")
```

***

## 7. Usage Examples

### 7.1 Basic Simulation

```python
from h2_plant.simulation.runner import run_simulation_from_config

# One-line simulation
results = run_simulation_from_config("configs/plant_baseline.yaml")

print(f"Total Production: {results['metrics']['total_production_kg']:.1f} kg")
print(f"Average Cost: ${results['metrics']['average_cost_per_kg']:.2f}/kg")
```

***

### 7.2 Simulation with Custom Events

```python
from h2_plant.config.plant_builder import PlantBuilder
from h2_plant.simulation.engine import SimulationEngine
from h2_plant.simulation.event_scheduler import create_maintenance_event

# Build plant
plant = PlantBuilder.from_file("configs/plant_baseline.yaml")
engine = SimulationEngine(plant.registry, plant.config.simulation)

# Schedule maintenance event
maintenance = create_maintenance_event(
    hour=1000,
    component_id='electrolyzer',
    duration_hours=24
)
engine.schedule_event(maintenance)

# Run simulation
results = engine.run()
```

***

### 7.3 Checkpoint and Resume

```python
from h2_plant.simulation.runner import run_simulation_from_config

# Run partial simulation
engine = SimulationEngine(registry, config)
engine.run(start_hour=0, end_hour=4000)

# Resume from checkpoint
results = run_simulation_from_config(
    "configs/plant_baseline.yaml",
    resume_from="simulation_output/checkpoints/checkpoint_hour_4000.json"
)
```

***

## 8. Validation Criteria

This Simulation Engine is **COMPLETE** when:

**SimulationEngine:**
- Modular execution framework implemented
- Event processing working
- Checkpointing functional
- Monitoring integrated
- Test coverage 95%+

**StateManager:**
- JSON/Pickle/HDF5 formats supported
- Checkpoint save/load working
- Results export functional

**EventScheduler:**
- Event scheduling working
- Recurring events supported
- Common event factories provided

**MonitoringSystem:**
- Metrics collection complete
- Time-series export working
- Summary statistics accurate

**Integration:**
- Full simulation runs successfully
- Checkpoint/resume validated
- Scenario comparison working

***

## 9. Success Metrics

| **Metric** | **Target** | **Validation** |
|-----------|-----------|----------------|
| Modularity | Complete | No monolithic simulation code |
| Checkpointing | Functional | Save/resume works correctly |
| Event Support | 3+ event types | Maintenance, price, demand events |
| Performance Overhead | <5% | Engine adds minimal execution time |
| Test Coverage | 95%+ | `pytest --cov=h2_plant.simulation` |

***