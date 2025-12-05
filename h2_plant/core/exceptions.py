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


class FlashConvergenceError(ComponentStepError):
    """Raised when UV-flash calculation fails to converge."""
    pass


class ThermodynamicDataError(Exception):
    """Raised when thermodynamic data is missing or invalid."""
    pass

