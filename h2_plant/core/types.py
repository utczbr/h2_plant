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
