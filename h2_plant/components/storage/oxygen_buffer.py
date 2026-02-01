"""
Oxygen Buffer Storage Component.

This module implements a simple buffer tank for oxygen byproduct from water
electrolysis. The buffer provides temporary storage with overflow venting
when capacity is exceeded.

Operating Model:
    - **Input**: Oxygen from electrolyzer (approximately 8 kg O₂ per kg H₂).
    - **Output**: Oxygen for industrial sale or oxy-fuel applications.
    - **Overflow**: When buffer is full, excess oxygen is vented to atmosphere.

Mass Balance:
    m_stored = m_added - m_removed - m_vented

Architecture:
    Implements the Component Lifecycle Contract (Layer 1):
    - `initialize()`: Standard initialization.
    - `step()`: No per-timestep logic (instantaneous operations).
    - `get_state()`: Returns mass, capacity, and cumulative tracking.

Economic Tracking:
    Cumulative tracking enables oxygen monetization analysis when oxygen
    can be sold for industrial applications.
"""

from typing import Dict, Any

from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry


class OxygenBuffer(Component):
    """
    Buffer storage for oxygen byproduct from electrolysis.

    Simple mass-balance storage with overflow venting. Tracks cumulative
    additions, removals, and venting for economic analysis.

    This component fulfills the Component Lifecycle Contract (Layer 1):
        - `initialize()`: Standard component initialization.
        - `step()`: Placeholder (operations are instantaneous).
        - `get_state()`: Returns storage level and tracking metrics.

    Attributes:
        capacity_kg (float): Maximum storage capacity (kg).
        mass_kg (float): Current oxygen inventory (kg).
        cumulative_added_kg (float): Total oxygen received (kg).
        cumulative_removed_kg (float): Total oxygen withdrawn (kg).
        cumulative_vented_kg (float): Total oxygen vented (kg).

    Example:
        >>> o2_buffer = OxygenBuffer(capacity_kg=500.0)
        >>> o2_buffer.initialize(dt=1/60, registry=registry)
        >>> vented = o2_buffer.add_oxygen(63.5)
        >>> removed = o2_buffer.remove_oxygen(50.0)
    """

    def __init__(self, capacity_kg: float):
        """
        Initialize the oxygen buffer.

        Args:
            capacity_kg (float): Maximum oxygen storage capacity in kg.
        """
        super().__init__()

        self.capacity_kg = capacity_kg
        self.mass_kg = 0.0

        # Cumulative tracking for economic analysis
        self.cumulative_added_kg = 0.0
        self.cumulative_removed_kg = 0.0
        self.cumulative_vented_kg = 0.0

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """
        Prepare the component for simulation execution.

        Fulfills the Component Lifecycle Contract initialization phase.

        Args:
            dt (float): Simulation timestep in hours.
            registry (ComponentRegistry): Central registry for component access.
        """
        super().initialize(dt, registry)

    def step(self, t: float) -> None:
        """
        Execute one simulation timestep.

        Buffer operations (add/remove) are instantaneous, so no per-step
        logic is required.

        Args:
            t (float): Current simulation time in hours.
        """
        super().step(t)

    def get_state(self) -> Dict[str, Any]:
        """
        Retrieve the component's current operational state.

        Fulfills the Component Lifecycle Contract state access.

        Returns:
            Dict[str, Any]: State dictionary containing:
                - mass_kg (float): Current inventory (kg).
                - capacity_kg (float): Maximum capacity (kg).
                - fill_percentage (float): Utilization percentage.
                - cumulative_added_kg (float): Total received (kg).
                - cumulative_removed_kg (float): Total withdrawn (kg).
                - cumulative_vented_kg (float): Total vented (kg).
        """
        fill_percentage = 0.0
        if self.capacity_kg > 0:
            fill_percentage = self.mass_kg / self.capacity_kg * 100

        return {
            **super().get_state(),
            'mass_kg': float(self.mass_kg),
            'capacity_kg': float(self.capacity_kg),
            'fill_percentage': float(fill_percentage),
            'cumulative_added_kg': float(self.cumulative_added_kg),
            'cumulative_removed_kg': float(self.cumulative_removed_kg),
            'cumulative_vented_kg': float(self.cumulative_vented_kg)
        }

    def add_oxygen(self, mass_kg: float) -> float:
        """
        Add oxygen to the buffer.

        If incoming mass exceeds available capacity, the excess is vented
        to atmosphere and tracked in cumulative_vented_kg.

        Args:
            mass_kg (float): Mass to add in kg.

        Returns:
            float: Mass vented due to overflow (kg). Returns 0.0 if all stored.
        """
        available_capacity = self.capacity_kg - self.mass_kg

        if mass_kg <= available_capacity:
            self.mass_kg += mass_kg
            self.cumulative_added_kg += mass_kg
            return 0.0
        else:
            stored = available_capacity
            vented = mass_kg - stored

            self.mass_kg = self.capacity_kg
            self.cumulative_added_kg += stored
            self.cumulative_vented_kg += vented

            return vented

    def remove_oxygen(self, mass_kg: float) -> float:
        """
        Remove oxygen from the buffer for industrial use or sale.

        Args:
            mass_kg (float): Requested mass to remove in kg.

        Returns:
            float: Actual mass removed (may be less if insufficient stored).
        """
        removed = min(mass_kg, self.mass_kg)
        self.mass_kg -= removed
        self.cumulative_removed_kg += removed

        return removed
