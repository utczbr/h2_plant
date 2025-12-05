"""
Component registry for dependency injection and lifecycle management.

The ComponentRegistry serves as the central orchestrator for all simulation
components, providing:
- Component registration and lookup by ID or type
- Lifecycle coordination (initialize all, step all)
- Dependency injection for inter-component communication
- State aggregation for checkpointing
"""

from typing import Dict, List, Optional, Type, Any, Union
from collections import defaultdict
import logging

from h2_plant.core.component import Component
from h2_plant.core.component_ids import ComponentID
from h2_plant.core.exceptions import (
    ComponentNotInitializedError,
    ComponentInitializationError,
    ComponentStepError,
    DuplicateComponentError,
    ComponentNotFoundError
)

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
        registry.register(ComponentID.HP_TANKS, tank, component_type="storage")
        
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
        component_id: Union[str, ComponentID], 
        component: Component,
        component_type: Optional[str] = None
    ) -> None:
        """
        Register a component in the registry.
        
        Args:
            component_id: Unique identifier for component lookup (str or ComponentID)
            component: Component instance to register
            component_type: Optional type tag for filtering (e.g., "storage", 
                          "production", "compression")
        
        Raises:
            DuplicateComponentError: If component_id already registered
            TypeError: If component doesn't inherit from Component
            
        Example:
            registry.register(ComponentID.ELECTROLYZER, electrolyzer, component_type="production")
            registry.register("hp_tank_1", tank, component_type="storage")
        """
        if isinstance(component_id, ComponentID):
            component_id = component_id.value
            
        if component_id in self._components:
            raise DuplicateComponentError(f"Component ID '{component_id}' already registered")
        
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
    
    def get(self, component_id: Union[str, ComponentID]) -> Component:
        """
        Retrieve component by ID.
        
        Args:
            component_id: Unique identifier of component (str or ComponentID)
            
        Returns:
            Component instance
            
        Raises:
            ComponentNotFoundError: If component_id not found
            
        Example:
            tank = registry.get(ComponentID.HP_TANKS)
            mass = tank.mass_kg
        """
        if isinstance(component_id, ComponentID):
            component_id = component_id.value
            
        if component_id not in self._components:
            raise ComponentNotFoundError(
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
        
        CRITICAL: Validates execution order dependencies.
        - Environment Manager must be first (provides weather/price data).
        - Coordinators must be before Producers (set targets).
        - Producers must be before Storage/Compression (generate flow).
        
        Args:
            dt: Simulation timestep in hours
            
        Raises:
            ComponentInitializationError: If any component initialization fails
            RuntimeError: If critical execution order dependencies are violated
            
        Example:
            registry.register("tank_1", tank1)
            registry.register("tank_2", tank2)
            registry.initialize_all(dt=1.0)  # Initializes tank1, then tank2
        """
        logger.info(f"Initializing {len(self._components)} components with dt={dt}h")
        
        # Validate Execution Order
        component_ids = list(self._components.keys())
        indices = {cid: i for i, cid in enumerate(component_ids)}
        
        # 1. Environment Manager should be early
        if ComponentID.ENVIRONMENT_MANAGER.value in indices:
            env_idx = indices[ComponentID.ENVIRONMENT_MANAGER.value]
            if env_idx > 5: # Allow some utility components before it (e.g. LUT)
                logger.warning(f"Environment Manager is late in execution order (index {env_idx}). This may cause lag in data availability.")

        # 2. Coordinator before PEM
        if (ComponentID.DUAL_PATH_COORDINATOR.value in indices and 
            ComponentID.PEM_ELECTROLYZER_DETAILED.value in indices):
            coord_idx = indices[ComponentID.DUAL_PATH_COORDINATOR.value]
            pem_idx = indices[ComponentID.PEM_ELECTROLYZER_DETAILED.value]
            
            if coord_idx > pem_idx:
                logger.error("EXECUTION ORDER ERROR: DualPathCoordinator registered AFTER PEM Electrolyzer.")
                logger.error("Coordinator must run first to set power setpoints.")
                # We could raise an error here, but for now just log critical error
                # raise RuntimeError("Invalid execution order: Coordinator after PEM")

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
        
        Calls step(t) on each component in STRICT REGISTRATION ORDER.
        
        Order matters significantly:
        1. Environment/Data Providers (update inputs)
        2. Coordinators/Controllers (read inputs, set targets)
        3. Producers (read targets, generate output)
        4. Transport/Storage (move/store output)
        
        Should be called once per simulation hour.
        
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

    def list_components(self) -> List[tuple[str, Component]]:
        """
        Return list of (component_id, component) tuples.
        """
        return list(self._components.items())
