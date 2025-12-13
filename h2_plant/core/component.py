"""
Core component abstractions for the hydrogen production system.

This module defines the foundational Component abstract base class that all
simulation components must inherit from, ensuring uniform lifecycle management
and state persistence capabilities.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, TYPE_CHECKING, Union

from h2_plant.core.exceptions import (
    ComponentNotInitializedError,
    ComponentInitializationError,
    ComponentStepError,
)

if TYPE_CHECKING:
    from h2_plant.core.component_registry import ComponentRegistry
    from h2_plant.core.component_ids import ComponentID


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
    
    def __init__(self, config: Any = None, **kwargs) -> None:
        """
        Initialize component with default state.
        
        Args:
            config: Optional component configuration object (Pydantic model)
            **kwargs: Additional keyword arguments for forward compatibility:
                - component_id: Optional explicit ID (for tests/manual wiring)
        """
        # Extract optional component_id if provided (enables direct instantiation in tests)
        component_id = kwargs.pop("component_id", None)
        
        # Preserve existing semantics
        self.component_id: Optional[str] = None
        self.dt: float = 0.0
        self._registry: Optional['ComponentRegistry'] = None
        self._initialized: bool = False
        self.config = config  # Store config object (Pydantic model)
        
        # Allow tests or manual code to set the ID directly
        if component_id is not None:
            self.set_component_id(component_id)
    
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
                super().initialize(dt, registry)
                self.mass = 0.0  # Initialize state
        """
        if self._initialized:
             # Already initialized, warn or ignore?
             # For safety, we can just return or raise if re-initialization is bad.
             # Base implementation just sets values.
             pass
             
        self.dt = dt
        self._registry = registry
        self._initialized = True
    
    def get_registry_safe(self, component_id: Union[str, 'ComponentID']) -> Optional['Component']:
        """
        Get component from registry safely, returning None if not found.
        
        Args:
            component_id: ID of component to retrieve (str or ComponentID)
            
        Returns:
            Component instance or None if not found/not initialized
        """
        if not self._initialized:
            return None
        
        if self._registry is None:
            return None
        
        # Convert ComponentID to string if needed
        if hasattr(component_id, 'value'):
            component_id = component_id.value
        
        # Check if component exists before trying to get it
        if self._registry.has(component_id):
            return self._registry.get(component_id)
        
        return None

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
                super().step(t)
                # ... implementation ...
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

    def get_output(self, port_name: str) -> Any:
        """
        Get output from a specific port.
        
        Args:
            port_name: Name of the output port
            
        Returns:
            Stream object (for material) or float (for energy)
            
        Raises:
            ValueError: If port does not exist or is not an output
        """
        raise NotImplementedError(f"Component {self.component_id} does not implement get_output")

    def receive_input(self, port_name: str, value: Any, resource_type: str = None) -> float:
        """
        Receive input into a specific port.
        
        Args:
            port_name: Name of the input port
            value: Stream object (for material) or float (for energy)
            resource_type: Type of resource (e.g., 'hydrogen', 'water')
            
        Returns:
            Amount accepted (e.g., mass in kg or energy in kWh)
            Default implementation returns 0 (accepts nothing).
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Log unhandled inputs for debuggability
        logger.debug(
            f"{self.component_id}: Ignored input port='{port_name}', "
            f"value={value}, resource_type='{resource_type}'"
        )
        
        # Optional strict mode: raise on unhandled inputs
        if self.config and hasattr(self.config, 'get'):
            if self.config.get('strict_inputs', False):
                raise ValueError(
                    f"{self.component_id}: No handler for input port='{port_name}' "
                    f"with resource_type='{resource_type}'"
                )
        
        return 0.0

    def extract_output(self, port_name: str, amount: float, resource_type: str) -> None:
        """
        Deduct extracted amount from component state.
        
        Called by FlowNetwork after successful transfer to target.
        
        Args:
            port_name: Name of the output port
            amount: Amount extracted (e.g., kg or kWh)
            resource_type: Type of resource
            
        Raises:
            ValueError: If port does not exist or amount is invalid
        """
        # Default implementation does nothing (for infinite sources or non-conserved flows)
        pass

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """
        Get information about available ports.
        
        Returns:
            Dictionary mapping port names to metadata:
            {
                'port_name': {
                    'type': 'input' | 'output',
                    'resource_type': 'hydrogen' | 'water' | 'electricity' | ...,
                    'units': 'kg' | 'kWh' | ...
                }
            }
        """
        return {}
