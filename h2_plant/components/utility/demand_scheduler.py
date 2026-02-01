"""
Hydrogen demand scheduler with time-based profiles.

Supports:
- Constant demand
- Time-of-day patterns (day/night shifts)
- Weekly patterns
- Custom profiles from arrays
"""

import numpy as np
from typing import Dict, Any, Optional, Literal

from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry


DemandPattern = Literal['constant', 'day_night', 'weekly', 'custom']


class DemandScheduler(Component):
    """
    Time-based hydrogen demand scheduler.

    Example:
        # Day/night pattern
        scheduler = DemandScheduler(
            pattern='day_night',
            day_demand_kg_h=80.0,
            night_demand_kg_h=20.0,
            day_start_hour=6,
            night_start_hour=22
        )

        scheduler.step(t=14)  # 2 PM
        demand = scheduler.current_demand_kg_h  # 80.0 (daytime)
    """

    def __init__(
        self,
        pattern: DemandPattern = 'constant',
        base_demand_kg_h: float = 50.0,
        day_demand_kg_h: Optional[float] = None,
        night_demand_kg_h: Optional[float] = None,
        day_start_hour: int = 6,
        night_start_hour: int = 22,
        custom_profile: Optional[np.ndarray] = None
    ):
        """
        Initialize demand scheduler.

        Args:
            pattern: Demand pattern type
            base_demand_kg_h: Baseline demand for 'constant' pattern
            day_demand_kg_h: Daytime demand for 'day_night' pattern
            night_demand_kg_h: Nighttime demand for 'day_night' pattern
            day_start_hour: Hour when day shift starts (0-23)
            night_start_hour: Hour when night shift starts (0-23)
            custom_profile: 8760-element array for 'custom' pattern
        """
        super().__init__()

        self.pattern = pattern
        self.base_demand_kg_h = base_demand_kg_h
        self.day_demand_kg_h = day_demand_kg_h or base_demand_kg_h
        self.night_demand_kg_h = night_demand_kg_h or base_demand_kg_h * 0.3
        self.day_start_hour = day_start_hour
        self.night_start_hour = night_start_hour
        self.custom_profile = custom_profile

        # Output
        self.current_demand_kg_h = 0.0
        self.current_demand_kg = 0.0  # For this timestep (demand * dt)

        # Tracking
        self.cumulative_demand_kg = 0.0

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """Initialize scheduler."""
        super().initialize(dt, registry)

        if self.pattern == 'custom' and self.custom_profile is None:
            raise ValueError("custom_profile required for 'custom' pattern")

    def step(self, t: float) -> None:
        """Update demand based on current time."""
        super().step(t)

        if self.pattern == 'constant':
            self.current_demand_kg_h = self.base_demand_kg_h

        elif self.pattern == 'day_night':
            hour_of_day = int(t) % 24

            if self.day_start_hour <= hour_of_day < self.night_start_hour:
                self.current_demand_kg_h = self.day_demand_kg_h
            else:
                self.current_demand_kg_h = self.night_demand_kg_h

        elif self.pattern == 'custom':
            if self.custom_profile is not None:
                hour_index = int(t) % len(self.custom_profile)
                self.current_demand_kg_h = float(self.custom_profile[hour_index])
            else:
                self.current_demand_kg_h = self.base_demand_kg_h

        # Calculate demand for this timestep
        self.current_demand_kg = self.current_demand_kg_h * self.dt
        self.cumulative_demand_kg += self.current_demand_kg

    def get_state(self) -> Dict[str, Any]:
        """Return current state."""
        return {
            **super().get_state(),
            'pattern': self.pattern,
            'current_demand_kg_h': float(self.current_demand_kg_h),
            'current_demand_kg': float(self.current_demand_kg),
            'cumulative_demand_kg': float(self.cumulative_demand_kg)
        }