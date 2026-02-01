# STEP 3: Technical Specification - Core Foundation

---

# 01_Core_Foundation_Specification.md

**Document:** Core Foundation Layer Technical Specification  
**Project:** Dual-Path Hydrogen Production System - Modular Refactoring v2.0  
**Date:** November 18, 2025  
**Layer:** Layer 1 - Foundation  
**Priority:** CRITICAL  
**Dependencies:** None (foundational layer)

---

## 1. Overview

### 1.1 Purpose

This specification defines the **foundational abstractions and infrastructure** upon which the entire modular hydrogen production system is built. The Core Foundation Layer addresses the most critical architectural gap identified in the critique: the absence of standardized component interfaces, runtime management, and type safety.

**Key Objectives:**
- Establish `Component` abstract base class (ABC) with uniform lifecycle contract
- Implement `ComponentRegistry` for dependency injection and runtime orchestration
- Convert all enums to `IntEnum` for NumPy/Numba compatibility
- Centralize physical and operational constants
- Define comprehensive type system for static analysis

**Critique Remediation:**
- **FAIL → PASS:** "Missing Component abstract base class" (Section 2)
- **FAIL → PASS:** "No ComponentRegistry" (Section 3)
- **FAIL → PASS:** "String-based enums" (Section 4)
- **FAIL → PASS:** "No centralized constants" (Section 5)

***

### 1.2 Scope

**In Scope:**
- `core/component.py`: Abstract base class and lifecycle protocols
- `core/component_registry.py`: Runtime component management system
- `core/enums.py`: Integer-based enumeration definitions
- `core/constants.py`: Physical constants and operational parameters
- `core/types.py`: Type aliases, protocols, and custom types
- `core/exceptions.py`: Custom exception hierarchy

**Out of Scope:**
- Component implementations (covered in `03_Component_Standardization_Specification.md`)
- Performance optimizations (covered in `02_Performance_Optimization_Specification.md`)
- Configuration loading (covered in `04_Configuration_System_Specification.md`)

***

### 1.3 Design Principles

1. **Interface Segregation:** Components implement only required methods
2. **Dependency Inversion:** Depend on abstractions (Component ABC), not concrete classes
3. **Single Responsibility:** Each module has one clear purpose
4. **Type Safety:** Comprehensive type hints for static analysis (`mypy --strict` compliance)
5. **NumPy/Numba Compatibility:** All enums use integer backing for vectorization

***

## 2. Component Abstract Base Class

### 2.1 Design Rationale

**Problem:** Current system has inconsistent component interfaces:
- `HydrogenProductionSource.calculate_production()`
- `SourceTaggedTank.fill()` / `.discharge()`
- `FillingCompressor.compress()`
- No unified initialization or state management

**Solution:** Define `Component` ABC with standardized lifecycle:
```
initialize() → step() → step() → ... → get_state()
     ↓           ↓                        ↓
   Setup    Execute timestep         Checkpoint
```

### 2.2 Interface Definition

**File:** `h2_plant/core/component.py`

```python
"""
Core component abstractions for the hydrogen production system.

This module defines the foundational Component abstract base class that all
simulation components must inherit from, ensuring uniform lifecycle management
and state persistence capabilities.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from h2_plant.core.component_registry import ComponentRegistry


class Component(ABC):
    """
    Abstract base class for all simulation components.
    
    Components are the fundamental building blocks of the hydrogen production
    system. Each component implements a standardized lifecycle:
    
    1. initialize(): Setup with timestep and registry access
    2. step(): Execute single simulation timestep (called 8760 times/year)
    3. get_state(): Serialize state for checkpointing/monitoring
    
    All concrete components (production sources, tanks, compressors, etc.)
    must inherit from this class and implement these methods.
    
    Attributes:
        component_id: Unique identifier set during registry registration
        dt: Simulation timestep in hours (typically 1.0)
        _registry: Reference to ComponentRegistry for dependency access
        _initialized: Flag tracking initialization status
    """
    
    def __init__(self) -> None:
        """Initialize component with default state."""
        self.component_id: Optional[str] = None
        self.dt: float = 0.0
        self._registry: Optional['ComponentRegistry'] = None
        self._initialized: bool = False
    
    @abstractmethod
    def initialize(self, dt: float, registry: 'ComponentRegistry') -> None:
        """
        Initialize component before simulation starts.
        
        Called once before the first timestep. Components should:
        - Store the timestep (dt)
        - Store registry reference for dependency access
        - Initialize internal state variables
        - Validate configuration parameters
        - Allocate arrays/buffers if needed
        
        Args:
            dt: Simulation timestep in hours (e.g., 1.0 for hourly)
            registry: ComponentRegistry for accessing other components
            
        Raises:
            ComponentInitializationError: If initialization fails
            
        Example:
            def initialize(self, dt: float, registry: ComponentRegistry) -> None:
                self.dt = dt
                self._registry = registry
                self.mass = 0.0  # Initialize state
                self._initialized = True
        """
        self.dt = dt
        self._registry = registry
        self._initialized = True
    
    @abstractmethod
    def step(self, t: float) -> None:
        """
        Execute single simulation timestep.
        
        Called once per simulation hour (8760 times for annual simulation).
        Components should:
        - Read inputs from other components (via registry)
        - Perform internal calculations
        - Update internal state
        - Expose outputs for downstream components
        
        Args:
            t: Current simulation time in hours (0 to 8760)
            
        Raises:
            ComponentNotInitializedError: If called before initialize()
            ComponentStepError: If timestep execution fails
            
        Example:
            def step(self, t: float) -> None:
                if not self._initialized:
                    raise ComponentNotInitializedError(self.component_id)
                
                # Read input from another component
                power = self._registry.get("grid_power").current_output
                
                # Calculate production
                self.h2_output = power * self.efficiency * self.dt
        """
        if not self._initialized:
            raise ComponentNotInitializedError(
                f"Component {self.component_id} not initialized before step()"
            )
    
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """
        Return current component state for persistence/monitoring.
        
        Called periodically for:
        - Checkpointing (save simulation state)
        - Monitoring dashboards
        - Debugging and logging
        
        Returns:
            Dictionary mapping state variable names to current values.
            All values must be JSON-serializable (primitives, lists, dicts).
            
        Example:
            def get_state(self) -> Dict[str, Any]:
                return {
                    "mass_kg": float(self.mass),
                    "pressure_pa": float(self.pressure),
                    "state": int(self.state),  # IntEnum value
                    "temperature_k": float(self.temperature)
                }
        """
        return {
            "component_id": self.component_id,
            "initialized": self._initialized
        }
    
    def set_component_id(self, component_id: str) -> None:
        """
        Set unique component identifier (called by ComponentRegistry).
        
        Args:
            component_id: Unique identifier for registry lookup
        """
        self.component_id = component_id
    
    def validate_initialized(self) -> None:
        """
        Validate component has been initialized.
        
        Raises:
            ComponentNotInitializedError: If initialize() not called
        """
        if not self._initialized:
            raise ComponentNotInitializedError(
                f"Component {self.component_id} must be initialized before use"
            )


class ComponentNotInitializedError(Exception):
    """Raised when component method called before initialize()."""
    pass


class ComponentInitializationError(Exception):
    """Raised when component initialization fails."""
    pass


class ComponentStepError(Exception):
    """Raised when component timestep execution fails."""
    pass
```

***

### 2.3 Usage Examples

#### Example 1: Simple Tank Component

```python
from h2_plant.core.component import Component
from h2_plant.core.enums import TankState
import numpy as np

class SimpleTank(Component):
    """Hydrogen storage tank with mass tracking."""
    
    def __init__(self, capacity_kg: float, pressure_bar: float):
        super().__init__()
        self.capacity_kg = capacity_kg
        self.pressure_bar = pressure_bar
        self.mass_kg = 0.0
        self.state = TankState.EMPTY
    
    def initialize(self, dt: float, registry: 'ComponentRegistry') -> None:
        super().initialize(dt, registry)
        # Validate configuration
        if self.capacity_kg <= 0:
            raise ComponentInitializationError("Tank capacity must be positive")
    
    def step(self, t: float) -> None:
        super().step(t)  # Validates initialization
        
        # Update state based on mass
        if self.mass_kg >= self.capacity_kg * 0.99:
            self.state = TankState.FULL
        elif self.mass_kg <= 0.01:
            self.state = TankState.EMPTY
        else:
            self.state = TankState.IDLE
    
    def get_state(self) -> Dict[str, Any]:
        return {
            **super().get_state(),
            "mass_kg": float(self.mass_kg),
            "capacity_kg": float(self.capacity_kg),
            "pressure_bar": float(self.pressure_bar),
            "state": int(self.state),
            "fill_percentage": float(self.mass_kg / self.capacity_kg * 100)
        }
    
    def fill(self, mass_kg: float) -> float:
        """Add mass to tank, return actual mass added."""
        available_capacity = self.capacity_kg - self.mass_kg
        actual_mass = min(mass_kg, available_capacity)
        self.mass_kg += actual_mass
        return actual_mass
```

#### Example 2: Production Source Component

```python
from h2_plant.core.component import Component

class ElectrolyzerSource(Component):
    """Electrolyzer hydrogen production component."""
    
    def __init__(self, max_power_mw: float, efficiency: float):
        super().__init__()
        self.max_power_mw = max_power_mw
        self.efficiency = efficiency
        self.power_input_mw = 0.0
        self.h2_output_kg = 0.0
    
    def initialize(self, dt: float, registry: 'ComponentRegistry') -> None:
        super().initialize(dt, registry)
        
        # Get reference to energy price component for cost tracking
        self.energy_price = registry.get("energy_price")
    
    def step(self, t: float) -> None:
        super().step(t)
        
        # Read current energy price
        price = self.energy_price.current_price_per_mwh
        
        # Calculate production (33 kWh/kg H2 @ 100% efficiency)
        energy_per_kg = 33.0 / self.efficiency  # kWh/kg
        self.h2_output_kg = (self.power_input_mw * 1000 * self.dt) / energy_per_kg
        
        # Clamp to max capacity
        max_output = (self.max_power_mw * 1000 * self.dt) / energy_per_kg
        self.h2_output_kg = min(self.h2_output_kg, max_output)
    
    def get_state(self) -> Dict[str, Any]:
        return {
            **super().get_state(),
            "power_input_mw": float(self.power_input_mw),
            "h2_output_kg": float(self.h2_output_kg),
            "efficiency": float(self.efficiency)
        }
```

***

## 3. ComponentRegistry

### 3.1 Design Rationale

**Problem:** Current system uses hardcoded component references:
```python
# Tight coupling - hard to test, reconfigure, or extend
hp_tank_result = hp_tank.fill(h2_mass)
lp_tank_mass = sum(tank.mass for tank in lp_tanks)
```

**Solution:** Dependency injection via central registry:
```python
# Loose coupling - components access dependencies by ID
hp_tank = registry.get("electrolyzer_hp_tank")
lp_tanks = registry.get_by_type("lp_storage")
```

**Benefits:**
- Dynamic component assembly (configuration-driven)
- Simplified testing (mock components by ID)
- Runtime reconfiguration (swap components without code changes)
- Clear dependency tracking

***

### 3.2 Implementation

**File:** `h2_plant/core/component_registry.py`

```python
"""
Component registry for dependency injection and lifecycle management.

The ComponentRegistry serves as the central orchestrator for all simulation
components, providing:
- Component registration and lookup by ID or type
- Lifecycle coordination (initialize all, step all)
- Dependency injection for inter-component communication
- State aggregation for checkpointing
"""

from typing import Dict, List, Optional, Type, Any
from collections import defaultdict
import logging

from h2_plant.core.component import Component, ComponentNotInitializedError

logger = logging.getLogger(__name__)


class ComponentRegistry:
    """
    Central registry for component management and orchestration.
    
    The registry maintains a catalog of all simulation components and provides:
    - Registration: Add components with unique IDs and optional type tags
    - Lookup: Retrieve components by ID or filter by type
    - Lifecycle: Initialize and step all registered components
    - State: Aggregate state from all components for checkpointing
    
    Example:
        registry = ComponentRegistry()
        
        # Register components
        tank = SimpleTank(capacity_kg=200.0, pressure_bar=350)
        registry.register("hp_tank_1", tank, component_type="storage")
        
        # Initialize all components
        registry.initialize_all(dt=1.0)
        
        # Execute simulation
        for t in range(8760):
            registry.step_all(t)
        
        # Get checkpoint data
        state = registry.get_all_states()
    """
    
    def __init__(self) -> None:
        """Initialize empty registry."""
        self._components: Dict[str, Component] = {}
        self._components_by_type: Dict[str, List[Component]] = defaultdict(list)
        self._initialized: bool = False
    
    def register(
        self, 
        component_id: str, 
        component: Component,
        component_type: Optional[str] = None
    ) -> None:
        """
        Register a component in the registry.
        
        Args:
            component_id: Unique identifier for component lookup
            component: Component instance to register
            component_type: Optional type tag for filtering (e.g., "storage", 
                          "production", "compression")
        
        Raises:
            ValueError: If component_id already registered
            TypeError: If component doesn't inherit from Component
            
        Example:
            registry.register("elec_source", electrolyzer, component_type="production")
            registry.register("hp_tank_1", tank, component_type="storage")
        """
        if component_id in self._components:
            raise ValueError(f"Component ID '{component_id}' already registered")
        
        if not isinstance(component, Component):
            raise TypeError(
                f"Component must inherit from Component ABC, got {type(component)}"
            )
        
        # Set component ID
        component.set_component_id(component_id)
        
        # Register in main catalog
        self._components[component_id] = component
        
        # Register in type index if type provided
        if component_type:
            self._components_by_type[component_type].append(component)
        
        logger.debug(f"Registered component '{component_id}' (type: {component_type})")
    
    def get(self, component_id: str) -> Component:
        """
        Retrieve component by ID.
        
        Args:
            component_id: Unique identifier of component
            
        Returns:
            Component instance
            
        Raises:
            KeyError: If component_id not found
            
        Example:
            tank = registry.get("hp_tank_1")
            mass = tank.mass_kg
        """
        if component_id not in self._components:
            raise KeyError(
                f"Component '{component_id}' not found in registry. "
                f"Available: {list(self._components.keys())}"
            )
        return self._components[component_id]
    
    def get_by_type(self, component_type: str) -> List[Component]:
        """
        Retrieve all components of a specific type.
        
        Args:
            component_type: Type tag used during registration
            
        Returns:
            List of components with matching type (empty if none found)
            
        Example:
            all_tanks = registry.get_by_type("storage")
            total_mass = sum(tank.mass_kg for tank in all_tanks)
        """
        return self._components_by_type.get(component_type, [])
    
    def has(self, component_id: str) -> bool:
        """
        Check if component ID exists in registry.
        
        Args:
            component_id: Identifier to check
            
        Returns:
            True if component registered, False otherwise
        """
        return component_id in self._components
    
    def initialize_all(self, dt: float) -> None:
        """
        Initialize all registered components.
        
        Should be called once before simulation starts. Calls initialize()
        on each component in registration order.
        
        Args:
            dt: Simulation timestep in hours
            
        Raises:
            ComponentInitializationError: If any component initialization fails
            
        Example:
            registry.register("tank_1", tank1)
            registry.register("tank_2", tank2)
            registry.initialize_all(dt=1.0)  # Initializes tank1, then tank2
        """
        logger.info(f"Initializing {len(self._components)} components with dt={dt}h")
        
        failed_components = []
        
        for component_id, component in self._components.items():
            try:
                component.initialize(dt, self)
                logger.debug(f"Initialized '{component_id}'")
            except Exception as e:
                logger.error(f"Failed to initialize '{component_id}': {e}")
                failed_components.append((component_id, e))
        
        if failed_components:
            error_msg = "\n".join(
                f"  - {comp_id}: {error}" 
                for comp_id, error in failed_components
            )
            raise ComponentInitializationError(
                f"Failed to initialize components:\n{error_msg}"
            )
        
        self._initialized = True
        logger.info("All components initialized successfully")
    
    def step_all(self, t: float) -> None:
        """
        Execute timestep on all registered components.
        
        Calls step(t) on each component in registration order. Should be
        called once per simulation hour.
        
        Args:
            t: Current simulation time in hours
            
        Raises:
            ComponentNotInitializedError: If initialize_all() not called first
            ComponentStepError: If any component step fails
            
        Example:
            for hour in range(8760):
                registry.step_all(hour)
        """
        if not self._initialized:
            raise ComponentNotInitializedError(
                "Registry not initialized. Call initialize_all() first."
            )
        
        for component_id, component in self._components.items():
            try:
                component.step(t)
            except Exception as e:
                logger.error(f"Component '{component_id}' failed at t={t}h: {e}")
                raise ComponentStepError(
                    f"Component '{component_id}' step failed: {e}"
                ) from e
    
    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        """
        Aggregate state from all components.
        
        Returns:
            Dictionary mapping component IDs to their state dictionaries
            
        Example:
            state = registry.get_all_states()
            # {
            #   "hp_tank_1": {"mass_kg": 150.0, "state": 1, ...},
            #   "electrolyzer": {"power_mw": 2.5, "h2_output_kg": 76.0, ...}
            # }
        """
        return {
            component_id: component.get_state()
            for component_id, component in self._components.items()
        }
    
    def get_component_count(self) -> int:
        """Return total number of registered components."""
        return len(self._components)
    
    def get_all_ids(self) -> List[str]:
        """Return list of all registered component IDs."""
        return list(self._components.keys())
    
    def get_types(self) -> List[str]:
        """Return list of all registered component types."""
        return list(self._components_by_type.keys())


class ComponentStepError(Exception):
    """Raised when component step execution fails."""
    pass
```

***

### 3.3 Usage Examples

#### Example 1: Basic Registration and Lookup

```python
from h2_plant.core.component_registry import ComponentRegistry

# Create registry
registry = ComponentRegistry()

# Register components with types
registry.register("elec_source", electrolyzer, component_type="production")
registry.register("atr_source", atr, component_type="production")
registry.register("hp_tank_1", tank1, component_type="storage")
registry.register("hp_tank_2", tank2, component_type="storage")

# Lookup by ID
tank = registry.get("hp_tank_1")
print(tank.mass_kg)

# Lookup by type
all_production = registry.get_by_type("production")
total_h2 = sum(src.h2_output_kg for src in all_production)

# Check existence
if registry.has("backup_compressor"):
    compressor = registry.get("backup_compressor")
```

#### Example 2: Simulation Loop

```python
# Setup
registry = ComponentRegistry()
registry.register("electrolyzer", electrolyzer, component_type="production")
registry.register("hp_storage", hp_tanks, component_type="storage")
registry.register("compressor", compressor, component_type="compression")

# Initialize all components
registry.initialize_all(dt=1.0)

# Simulation loop
for hour in range(8760):
    # All components execute their step() methods
    registry.step_all(hour)
    
    # Periodic checkpointing
    if hour % 168 == 0:  # Weekly
        state = registry.get_all_states()
        save_checkpoint(state, hour)
```

***

## 4. Integer-Based Enumerations

### 4.1 Design Rationale

**Problem:** Current system uses string-based enums:
```python
# String enums are slow and incompatible with NumPy/Numba
tank_state = "FILLING"  # 40+ bytes, string comparison overhead
states = ["IDLE", "FILLING", "FULL"]  # Can't vectorize
```

**Solution:** Convert to `IntEnum` for performance:
```python
# Integer enums are fast and NumPy/Numba compatible
tank_state = TankState.FILLING  # 4 bytes, integer comparison
states = np.array([TankState.IDLE, TankState.FILLING], dtype=np.int32)
available = np.where(states == TankState.IDLE)[0]  # Vectorized
```

**Performance Impact:**
- Memory: 40+ bytes → 4 bytes per enum value
- Comparison: String equality → Integer equality (10-100x faster)
- Vectorization: Enables NumPy boolean indexing and Numba JIT

***

### 4.2 Implementation

**File:** `h2_plant/core/enums.py`

```python
"""
Integer-based enumerations for high-performance simulation.

All enums use IntEnum for:
- NumPy array compatibility (dtype=np.int32)
- Numba JIT compilation support
- Memory efficiency (4 bytes vs 40+ bytes for strings)
- Fast comparisons and vectorization
"""

from enum import IntEnum


class TankState(IntEnum):
    """
    State of a hydrogen storage tank.
    
    Used for vectorized tank operations in TankArray and scheduling logic.
    
    Examples:
        # Single tank
        if tank.state == TankState.IDLE:
            tank.fill(mass)
        
        # Vectorized (NumPy array of states)
        states = np.array([TankState.IDLE, TankState.FILLING, TankState.FULL])
        available_indices = np.where(states == TankState.IDLE)[0]
    """
    IDLE = 0         # Tank ready for filling or discharging
    FILLING = 1      # Currently being filled
    DISCHARGING = 2  # Currently being discharged
    FULL = 3         # At capacity (>99% full)
    EMPTY = 4        # Depleted (<1% full)
    MAINTENANCE = 5  # Offline for maintenance


class ProductionState(IntEnum):
    """
    State of hydrogen production source (Electrolyzer or ATR).
    
    Examples:
        if electrolyzer.state == ProductionState.RUNNING:
            h2_output = electrolyzer.step(t)
    """
    OFFLINE = 0      # Not producing
    STARTING = 1     # Warm-up phase (ATR only, typically 30-60 min)
    RUNNING = 2      # Active production
    SHUTTING_DOWN = 3  # Cool-down phase
    MAINTENANCE = 4  # Scheduled maintenance
    FAULT = 5        # Error state requiring intervention


class CompressorMode(IntEnum):
    """
    Operating mode for compression equipment.
    
    Examples:
        if compressor.mode == CompressorMode.LP_TO_HP:
            compressor.transfer_mass(lp_tank, hp_tank)
    """
    IDLE = 0         # Not operating
    LP_TO_HP = 1     # Transferring from low-pressure to high-pressure storage
    HP_TO_DELIVERY = 2  # Boosting to delivery pressure (900 bar)
    RECIRCULATION = 3   # Internal pressure balancing


class AllocationStrategy(IntEnum):
    """
    Strategy for allocating hydrogen demand across production pathways.
    
    Used by DualPathCoordinator to split demand between electrolyzer and ATR.
    
    Examples:
        coordinator = DualPathCoordinator(strategy=AllocationStrategy.COST_OPTIMAL)
        elec_demand, atr_demand = coordinator.allocate(total_demand, t)
    """
    COST_OPTIMAL = 0     # Minimize total production cost based on energy prices
    PRIORITY_GRID = 1    # Maximize electrolyzer usage (grid-powered)
    PRIORITY_ATR = 2     # Maximize ATR usage (natural gas)
    BALANCED = 3         # 50/50 split between pathways
    EMISSIONS_OPTIMAL = 4  # Minimize CO2 emissions


class FlowDirection(IntEnum):
    """
    Direction of hydrogen flow in the system.
    
    Used for tracking mass flow through pipelines and validation.
    """
    NONE = 0         # No flow
    PRODUCTION_TO_LP = 1   # Source → Low-pressure storage
    LP_TO_HP = 2     # Low-pressure → High-pressure (compression)
    HP_TO_DELIVERY = 3     # High-pressure → Customer delivery
    RECYCLE = 4      # Return flow (pressure balancing)


class SystemMode(IntEnum):
    """
    Overall system operating mode.
    
    Controls high-level system behavior and safety interlocks.
    """
    STARTUP = 0      # Initial startup sequence
    NORMAL = 1       # Standard operation
    PEAK_DEMAND = 2  # High-demand mode (maximize output)
    LOW_DEMAND = 3   # Low-demand mode (optimize efficiency)
    EMERGENCY_STOP = 4  # Emergency shutdown
    MAINTENANCE = 5  # Maintenance mode
```

***

### 4.3 NumPy/Numba Integration Examples

#### Example 1: Vectorized Tank State Queries

```python
import numpy as np
from h2_plant.core.enums import TankState

# Tank array with integer states
n_tanks = 8
states = np.array([TankState.IDLE, TankState.FILLING, TankState.FULL,
                   TankState.IDLE, TankState.EMPTY, TankState.IDLE,
                   TankState.FULL, TankState.FILLING], dtype=np.int32)

# Vectorized queries (fast!)
idle_indices = np.where(states == TankState.IDLE)[0]  # [0, 3, 5]
full_count = np.sum(states == TankState.FULL)  # 2
available = np.logical_or(states == TankState.IDLE, states == TankState.EMPTY)
```

#### Example 2: Numba JIT with IntEnum

```python
from numba import njit
import numpy as np
from h2_plant.core.enums import TankState

@njit
def find_available_tank(states: np.ndarray, capacities: np.ndarray, 
                       required_mass: float) -> int:
    """
    Find first idle tank with sufficient capacity (Numba-compiled).
    
    Args:
        states: Array of TankState values (np.int32)
        capacities: Array of tank capacities in kg
        required_mass: Mass to store in kg
        
    Returns:
        Index of suitable tank, or -1 if none found
    """
    for i in range(len(states)):
        if states[i] == TankState.IDLE and capacities[i] >= required_mass:
            return i
    return -1

# Usage
states = np.array([TankState.FULL, TankState.IDLE, TankState.IDLE], dtype=np.int32)
capacities = np.array([200.0, 200.0, 150.0], dtype=np.float64)
tank_idx = find_available_tank(states, capacities, 180.0)  # Returns 1
```

***

## 5. Constants Module

### 5.1 Design Rationale

**Problem:** Constants scattered throughout codebase:
```python
# In file1.py
R_H2 = 4124  # J/kg·K

# In file2.py
R_H2 = 4.124  # kJ/kg·K  (INCONSISTENT!)

# In file3.py
hydrogen_gas_constant = 4124  # J/kg·K (DUPLICATE)
```

**Solution:** Centralize all physical and operational constants:
```python
from h2_plant.core.constants import GasConstants, StandardConditions

density = mass / (GasConstants.R_H2 * temperature / pressure)
```

***

### 5.2 Implementation

**File:** `h2_plant/core/constants.py`

```python
"""
Physical constants and operational parameters for hydrogen production system.

This module centralizes all constants to ensure:
- Consistency across codebase
- Single source of truth
- Easy updates and unit conversions
- Type safety with Final annotations
"""

from typing import Final


class GasConstants:
    """Thermodynamic constants for gases."""
    
    # Specific gas constants (J/kg·K)
    R_H2: Final[float] = 4124.0      # Hydrogen
    R_O2: Final[float] = 259.8       # Oxygen
    R_N2: Final[float] = 296.8       # Nitrogen
    R_CH4: Final[float] = 518.3      # Methane (ATR feedstock)
    
    # Universal gas constant
    R_UNIVERSAL: Final[float] = 8.314  # J/mol·K
    
    # Molecular weights (kg/kmol)
    MW_H2: Final[float] = 2.016
    MW_O2: Final[float] = 31.999
    MW_H2O: Final[float] = 18.015
    MW_CH4: Final[float] = 16.043
    
    # Specific heat ratios (γ = Cp/Cv)
    GAMMA_H2: Final[float] = 1.41
    GAMMA_O2: Final[float] = 1.40


class StandardConditions:
    """Standard reference conditions."""
    
    TEMPERATURE_K: Final[float] = 298.15  # 25°C
    TEMPERATURE_C: Final[float] = 25.0
    PRESSURE_PA: Final[float] = 101325.0  # 1 atm
    PRESSURE_BAR: Final[float] = 1.01325


class ConversionFactors:
    """Unit conversion factors."""
    
    # Pressure
    PA_TO_BAR: Final[float] = 1e-5
    BAR_TO_PA: Final[float] = 1e5
    PSI_TO_PA: Final[float] = 6894.76
    
    # Energy
    KWH_TO_J: Final[float] = 3.6e6
    J_TO_KWH: Final[float] = 1 / 3.6e6
    MWH_TO_KWH: Final[float] = 1000.0
    
    # Mass
    KG_TO_G: Final[float] = 1000.0
    KG_TO_LB: Final[float] = 2.20462
    
    # Power
    MW_TO_KW: Final[float] = 1000.0
    KW_TO_W: Final[float] = 1000.0


class ProductionConstants:
    """Hydrogen production parameters."""
    
    # Electrolysis
    H2_ENERGY_CONTENT_LHV_KWH_PER_KG: Final[float] = 33.0  # Lower heating value
    H2_ENERGY_CONTENT_HHV_KWH_PER_KG: Final[float] = 39.4  # Higher heating value
    ELECTROLYSIS_THEORETICAL_ENERGY_KWH_PER_KG: Final[float] = 39.4
    ELECTROLYSIS_TYPICAL_EFFICIENCY: Final[float] = 0.65  # 65% efficient
    
    # ATR (Auto-Thermal Reforming)
    ATR_TYPICAL_EFFICIENCY: Final[float] = 0.75
    ATR_STARTUP_TIME_HOURS: Final[float] = 1.0
    ATR_COOLDOWN_TIME_HOURS: Final[float] = 0.5
    
    # Oxygen byproduct (electrolysis)
    O2_TO_H2_MASS_RATIO: Final[float] = 7.94  # kg O2 per kg H2


class StorageConstants:
    """Hydrogen storage parameters."""
    
    # Typical pressure levels (Pa)
    LOW_PRESSURE_PA: Final[float] = 30e5      # 30 bar
    HIGH_PRESSURE_PA: Final[float] = 350e5    # 350 bar
    DELIVERY_PRESSURE_PA: Final[float] = 900e5  # 900 bar
    
    # Tank sizing
    TYPICAL_LP_CAPACITY_KG: Final[float] = 50.0
    TYPICAL_HP_CAPACITY_KG: Final[float] = 200.0
    
    # Safety margins
    TANK_FULL_THRESHOLD: Final[float] = 0.99  # 99% capacity
    TANK_EMPTY_THRESHOLD: Final[float] = 0.01  # 1% capacity


class CompressionConstants:
    """Compression system parameters."""
    
    # Efficiency
    ISENTROPIC_EFFICIENCY: Final[float] = 0.75  # Typical compressor efficiency
    MECHANICAL_EFFICIENCY: Final[float] = 0.95
    
    # Multi-stage compression
    TYPICAL_STAGE_PRESSURE_RATIO: Final[float] = 3.5
    MAX_STAGES: Final[int] = 4


class EconomicConstants:
    """Economic parameters."""
    
    # Energy pricing ($/MWh) - typical ranges
    ENERGY_PRICE_MIN: Final[float] = 20.0
    ENERGY_PRICE_MAX: Final[float] = 200.0
    ENERGY_PRICE_AVERAGE: Final[float] = 60.0
    
    # Hydrogen pricing ($/kg)
    H2_SELLING_PRICE: Final[float] = 5.0  # Target price
    
    # Natural gas pricing ($/MMBtu)
    NG_PRICE_TYPICAL: Final[float] = 3.5


class SimulationDefaults:
    """Default simulation parameters."""
    
    TIMESTEP_HOURS: Final[float] = 1.0
    ANNUAL_HOURS: Final[int] = 8760
    CHECKPOINT_INTERVAL_HOURS: Final[int] = 168  # Weekly
    
    # Numerical tolerances
    MASS_TOLERANCE_KG: Final[float] = 1e-6
    PRESSURE_TOLERANCE_PA: Final[float] = 1e3
    TEMPERATURE_TOLERANCE_K: Final[float] = 0.01
```

***

### 5.3 Usage Examples

```python
from h2_plant.core.constants import (
    GasConstants, StandardConditions, ProductionConstants, ConversionFactors
)

# Thermodynamic calculations
def calculate_density(pressure_pa: float, temperature_k: float) -> float:
    """Calculate H2 density using ideal gas law."""
    return pressure_pa / (GasConstants.R_H2 * temperature_k)

# Production energy calculation
def electrolysis_power_required(h2_mass_kg: float, efficiency: float) -> float:
    """Calculate power required for electrolysis (kW)."""
    theoretical_energy = h2_mass_kg * ProductionConstants.H2_ENERGY_CONTENT_HHV_KWH_PER_KG
    return theoretical_energy / efficiency

# Unit conversions
pressure_bar = 350.0
pressure_pa = pressure_bar * ConversionFactors.BAR_TO_PA  # 350e5 Pa

# Standard conditions
density_std = calculate_density(
    StandardConditions.PRESSURE_PA,
    StandardConditions.TEMPERATURE_K
)
```

***

## 6. Type Definitions

### 6.1 Implementation

**File:** `h2_plant/core/types.py`

```python
"""
Type aliases and protocols for static type checking.

Enables mypy --strict compliance and improved IDE autocomplete.
"""

from typing import Protocol, Dict, Any, TypeAlias
import numpy as np
import numpy.typing as npt

# Scalar types
Mass: TypeAlias = float          # kg
Pressure: TypeAlias = float      # Pa
Temperature: TypeAlias = float   # K
Power: TypeAlias = float         # MW
Energy: TypeAlias = float        # kWh
Time: TypeAlias = float          # hours
FlowRate: TypeAlias = float      # kg/h

# Array types
MassArray: TypeAlias = npt.NDArray[np.float64]
StateArray: TypeAlias = npt.NDArray[np.int32]  # IntEnum states

# State dictionary type
ComponentState: TypeAlias = Dict[str, Any]

# Configuration types
class ConfigDict(Protocol):
    """Protocol for configuration dictionaries."""
    def __getitem__(self, key: str) -> Any: ...
    def get(self, key: str, default: Any = None) -> Any: ...
```

***

## 7. Exception Hierarchy

### 7.1 Implementation

**File:** `h2_plant/core/exceptions.py`

```python
"""Custom exception hierarchy for hydrogen production system."""


class H2PlantError(Exception):
    """Base exception for all h2_plant errors."""
    pass


class ComponentError(H2PlantError):
    """Base exception for component-related errors."""
    pass


class ComponentNotInitializedError(ComponentError):
    """Raised when component method called before initialize()."""
    pass


class ComponentInitializationError(ComponentError):
    """Raised when component initialization fails."""
    pass


class ComponentStepError(ComponentError):
    """Raised when component timestep execution fails."""
    pass


class RegistryError(H2PlantError):
    """Base exception for registry errors."""
    pass


class ComponentNotFoundError(RegistryError):
    """Raised when component ID not found in registry."""
    pass


class DuplicateComponentError(RegistryError):
    """Raised when attempting to register duplicate component ID."""
    pass


class ConfigurationError(H2PlantError):
    """Raised for configuration loading/validation errors."""
    pass


class SimulationError(H2PlantError):
    """Raised for simulation execution errors."""
    pass
```

***

## 8. Testing Strategy

### 8.1 Unit Tests

**File:** `tests/core/test_component.py`

```python
import pytest
from h2_plant.core.component import Component, ComponentNotInitializedError
from h2_plant.core.component_registry import ComponentRegistry

class MockComponent(Component):
    """Mock component for testing."""
    
    def __init__(self):
        super().__init__()
        self.step_count = 0
    
    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        super().initialize(dt, registry)
    
    def step(self, t: float) -> None:
        super().step(t)
        self.step_count += 1
    
    def get_state(self) -> dict:
        return {**super().get_state(), "step_count": self.step_count}


def test_component_initialization():
    """Test component initialization flow."""
    component = MockComponent()
    registry = ComponentRegistry()
    
    # Not initialized yet
    assert not component._initialized
    
    # Initialize
    component.initialize(dt=1.0, registry=registry)
    assert component._initialized
    assert component.dt == 1.0

def test_step_before_initialize_raises_error():
    """Test that step() before initialize() raises error."""
    component = MockComponent()
    
    with pytest.raises(ComponentNotInitializedError):
        component.step(0.0)

def test_component_state_serialization():
    """Test get_state() returns valid dictionary."""
    component = MockComponent()
    registry = ComponentRegistry()
    component.initialize(1.0, registry)
    
    state = component.get_state()
    assert isinstance(state, dict)
    assert "initialized" in state
    assert state["initialized"] is True
```

**File:** `tests/core/test_component_registry.py`

```python
def test_component_registration():
    """Test basic component registration and lookup."""
    registry = ComponentRegistry()
    component = MockComponent()
    
    registry.register("test_comp", component, component_type="mock")
    
    # Lookup by ID
    retrieved = registry.get("test_comp")
    assert retrieved is component
    assert retrieved.component_id == "test_comp"
    
    # Lookup by type
    by_type = registry.get_by_type("mock")
    assert len(by_type) == 1
    assert by_type[0] is component

def test_duplicate_registration_raises_error():
    """Test that duplicate IDs raise ValueError."""
    registry = ComponentRegistry()
    
    registry.register("comp1", MockComponent())
    
    with pytest.raises(ValueError, match="already registered"):
        registry.register("comp1", MockComponent())

def test_initialize_all():
    """Test initialize_all() calls initialize on all components."""
    registry = ComponentRegistry()
    
    comp1 = MockComponent()
    comp2 = MockComponent()
    registry.register("comp1", comp1)
    registry.register("comp2", comp2)
    
    registry.initialize_all(dt=1.0)
    
    assert comp1._initialized
    assert comp2._initialized
    assert comp1.dt == 1.0
    assert comp2.dt == 1.0

def test_step_all():
    """Test step_all() calls step on all components."""
    registry = ComponentRegistry()
    
    comp1 = MockComponent()
    comp2 = MockComponent()
    registry.register("comp1", comp1)
    registry.register("comp2", comp2)
    registry.initialize_all(dt=1.0)
    
    registry.step_all(0.0)
    
    assert comp1.step_count == 1
    assert comp2.step_count == 1
```

**File:** `tests/core/test_enums.py`

```python
import numpy as np
from h2_plant.core.enums import TankState, ProductionState

def test_enum_integer_values():
    """Test enums have integer values."""
    assert isinstance(TankState.IDLE.value, int)
    assert TankState.IDLE == 0
    assert TankState.FILLING == 1

def test_enum_numpy_compatibility():
    """Test enums work with NumPy arrays."""
    states = np.array([TankState.IDLE, TankState.FULL, TankState.IDLE], dtype=np.int32)
    
    # Vectorized comparison
    idle_mask = states == TankState.IDLE
    assert np.array_equal(idle_mask, [True, False, True])
    
    # Indexing
    idle_indices = np.where(states == TankState.IDLE)[0]
    assert np.array_equal(idle_indices, [0, 2])
```

***

### 8.2 Coverage Targets

- **Component ABC:** 100% coverage (critical interface)
- **ComponentRegistry:** 95% coverage (core infrastructure)
- **Enums:** 100% coverage (simple, complete testing)
- **Constants:** 100% coverage (validation only)
- **Overall Core Module:** 95%+

***

## 9. Migration Guide

### 9.1 Legacy Adapter Pattern

To maintain backward compatibility during migration, create adapters:

**File:** `h2_plant/legacy/component_adapter.py`

```python
"""
Legacy adapters for backward compatibility during migration.

These adapters allow old code to work with new Component-based classes.
"""

from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry

class LegacyProductionSourceAdapter:
    """
    Adapter for old calculate_production() interface.
    
    Usage:
        # Old code (still works):
        h2_mass = source.calculate_production(power_mw, dt)
        
        # New code (preferred):
        source.power_input = power_mw
        source.step(t)
        h2_mass = source.output_mass
    """
    
    def calculate_production(self, power_mw: float, dt: float) -> float:
        """Legacy method - DEPRECATED."""
        import warnings
        warnings.warn(
            "calculate_production() is deprecated. "
            "Use source.power_input + source.step() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        self.power_input = power_mw
        self.step(self._current_time)
        return self.output_mass
```

***

### 9.2 Migration Checklist

**For Each Existing Component:**

1. [ ] Add `Component` inheritance
2. [ ] Implement `initialize(dt, registry)` method
3. [ ] Refactor logic into `step(t)` method
4. [ ] Implement `get_state()` for checkpointing
5. [ ] Update tests to use new interface
6. [ ] Add deprecation warnings to old methods
7. [ ] Update documentation

**Example Migration:**

```python
# BEFORE (old code)
class HydrogenProductionSource:
    def __init__(self, max_power_mw):
        self.max_power_mw = max_power_mw
    
    def calculate_production(self, power_mw, dt):
        return power_mw * dt * 33.0  # Simplified

# AFTER (migrated to Component)
from h2_plant.core.component import Component

class HydrogenProductionSource(Component):
    def __init__(self, max_power_mw):
        super().__init__()
        self.max_power_mw = max_power_mw
        self.power_input = 0.0
        self.output_mass = 0.0
    
    def initialize(self, dt, registry):
        super().initialize(dt, registry)
        # Setup logic here
    
    def step(self, t):
        super().step(t)
        self.output_mass = self.power_input * self.dt * 33.0
    
    def get_state(self):
        return {
            **super().get_state(),
            "power_input": self.power_input,
            "output_mass": self.output_mass
        }
    
    # Legacy adapter (deprecated)
    def calculate_production(self, power_mw, dt):
        import warnings
        warnings.warn("Use step() instead", DeprecationWarning)
        self.power_input = power_mw
        self.step(0)
        return self.output_mass
```

***

## 10. Validation Criteria

This Core Foundation layer is **COMPLETE** when:

 **Component ABC:**
- All 3 abstract methods defined (`initialize`, `step`, `get_state`)
- Comprehensive docstrings with examples
- Unit tests achieve 100% coverage

 **ComponentRegistry:**
- Registration, lookup (by ID and type), initialization, stepping implemented
- State aggregation working
- Exception handling complete
- Unit tests achieve 95%+ coverage

 **Integer Enums:**
- All 6 enum classes defined (TankState, ProductionState, CompressorMode, AllocationStrategy, FlowDirection, SystemMode)
- NumPy compatibility validated
- Numba JIT compatibility validated

 **Constants Module:**
- All physical constants centralized
- Economic and simulation defaults defined
- No hardcoded constants remain in other modules

 **Type System:**
- Type aliases defined for common types
- Protocols for configuration
- `mypy --strict` passes

 **Documentation:**
- All public APIs documented
- Usage examples provided
- Migration guide complete

***

## 11. Success Metrics

| **Metric** | **Target** | **Validation** |
|-----------|-----------|----------------|
| Test Coverage | 95%+ | `pytest --cov=h2_plant.core --cov-report=html` |
| Type Check Pass Rate | 100% | `mypy --strict h2_plant/core` |
| Documentation Coverage | 100% API docs | Manual review of docstrings |
| Performance (Registry Overhead) | <1ms per 100 components | Benchmark `initialize_all()` and `step_all()` |

***
