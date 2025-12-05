"""
Energy price tracker for economic optimization.

Tracks time-varying energy prices (e.g., day-ahead market, real-time pricing)
and provides current price for production cost calculations.
"""

import numpy as np
from typing import Dict, Any, Optional

from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry


class EnergyPriceTracker(Component):
    """
    Time-based energy price tracker.

    Example:
        # Load hourly prices
        prices_mwh = np.array([...])  # 8760 hourly prices

        tracker = EnergyPriceTracker(prices_per_mwh=prices_mwh)
        tracker.step(t=15)

        price = tracker.current_price_per_kwh  # $/kWh at hour 15
    """

    def __init__(
        self,
        prices_per_mwh: np.ndarray,
        default_price_per_mwh: float = 60.0,
        data_resolution_minutes: int = 60
    ):
        """
        Initialize energy price tracker.

        Args:
            prices_per_mwh: Array of prices ($/MWh)
            default_price_per_mwh: Default price if array exhausted
            data_resolution_minutes: Time resolution of the price data in minutes (default 60 for hourly)
        """
        super().__init__()

        self.prices_per_mwh = prices_per_mwh
        self.default_price_per_mwh = default_price_per_mwh
        self.data_resolution_minutes = data_resolution_minutes

        # Outputs
        self.current_price_per_mwh = default_price_per_mwh
        self.current_price_per_kwh = default_price_per_mwh / 1000.0

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """Initialize tracker."""
        super().initialize(dt, registry)

    def step(self, t: float) -> None:
        """Update current price based on time."""
        super().step(t)

        # Calculate index based on simulation time (hours) and data resolution
        # t is in hours. 
        # If resolution is 60 min (1 hr), index = int(t)
        # If resolution is 15 min (0.25 hr), index = int(t * 4)
        points_per_hour = 60 / self.data_resolution_minutes
        data_index = int(t * points_per_hour)

        if data_index < len(self.prices_per_mwh):
            self.current_price_per_mwh = float(self.prices_per_mwh[data_index])
        else:
            self.current_price_per_mwh = self.default_price_per_mwh

        self.current_price_per_kwh = self.current_price_per_mwh / 1000.0

    def get_state(self) -> Dict[str, Any]:
        """Return current state."""
        return {
            **super().get_state(),
            'current_price_per_mwh': float(self.current_price_per_mwh),
            'current_price_per_kwh': float(self.current_price_per_kwh)
        }