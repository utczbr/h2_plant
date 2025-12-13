"""
Energy Price Tracker for Economic Optimization.

This module implements a time-varying energy price tracker for production
cost calculations. Supports day-ahead market pricing, real-time pricing,
and configurable time resolutions.

Economic Model:
    Electricity prices drive optimal dispatch decisions:
    - Low prices → maximize electrolyzer operation
    - High prices → reduce consumption, use stored hydrogen
    - Very high prices → sell stored energy back to grid

    The tracker provides instantaneous price data to:
    - Dispatch strategy optimization
    - Production cost accounting
    - Revenue calculation for grid sales

Architecture:
    Implements the Component Lifecycle Contract (Layer 1):
    - `initialize()`: Standard initialization.
    - `step()`: Updates current price based on simulation time.
    - `get_state()`: Returns current price in $/MWh and $/kWh.
"""

import numpy as np
from typing import Dict, Any, Optional

from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry


class EnergyPriceTracker(Component):
    """
    Time-based energy price tracker for economic optimization.

    Provides current electricity price based on simulation time and
    configurable price time series data.

    This component fulfills the Component Lifecycle Contract (Layer 1):
        - `initialize()`: Standard component initialization.
        - `step()`: Updates current price based on simulation time index.
        - `get_state()`: Returns current prices in $/MWh and $/kWh.

    Index Calculation:
        data_index = int(t × points_per_hour)

        Where points_per_hour = 60 / data_resolution_minutes.
        For hourly data (60 min): index = int(t).
        For 15-minute data (15 min): index = int(t × 4).

    Attributes:
        prices_per_mwh (np.ndarray): Time series of prices ($/MWh).
        default_price_per_mwh (float): Fallback price when data exhausted.
        current_price_per_mwh (float): Current price ($/MWh).
        current_price_per_kwh (float): Current price ($/kWh).

    Example:
        >>> prices = np.array([50, 45, 60, 80, ...])  # 8760 hourly prices
        >>> tracker = EnergyPriceTracker(prices_per_mwh=prices)
        >>> tracker.initialize(dt=1/60, registry=registry)
        >>> tracker.step(t=15.0)
        >>> cost = production_kwh * tracker.current_price_per_kwh
    """

    def __init__(
        self,
        prices_per_mwh: np.ndarray,
        default_price_per_mwh: float = 60.0,
        data_resolution_minutes: int = 60
    ):
        """
        Initialize the energy price tracker.

        Args:
            prices_per_mwh (np.ndarray): Array of electricity prices in $/MWh.
                Length determines coverage (e.g., 8760 for one year hourly).
            default_price_per_mwh (float): Fallback price when time index
                exceeds array length. Default: 60.0.
            data_resolution_minutes (int): Time resolution of price data
                in minutes. Use 60 for hourly, 15 for 15-minute markets.
                Default: 60.
        """
        super().__init__()

        self.prices_per_mwh = prices_per_mwh
        self.default_price_per_mwh = default_price_per_mwh
        self.data_resolution_minutes = data_resolution_minutes

        # Current price state
        self.current_price_per_mwh = default_price_per_mwh
        self.current_price_per_kwh = default_price_per_mwh / 1000.0

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
        Update current price based on simulation time.

        Calculates price array index from simulation time and data resolution,
        then updates current price attributes.

        Args:
            t (float): Current simulation time in hours.
        """
        super().step(t)

        # Calculate index based on time and data resolution
        points_per_hour = 60 / self.data_resolution_minutes
        data_index = int(t * points_per_hour)

        if data_index < len(self.prices_per_mwh):
            self.current_price_per_mwh = float(self.prices_per_mwh[data_index])
        else:
            self.current_price_per_mwh = self.default_price_per_mwh

        self.current_price_per_kwh = self.current_price_per_mwh / 1000.0

    def get_state(self) -> Dict[str, Any]:
        """
        Retrieve the component's current operational state.

        Fulfills the Component Lifecycle Contract state access.

        Returns:
            Dict[str, Any]: State dictionary containing:
                - current_price_per_mwh (float): Current price ($/MWh).
                - current_price_per_kwh (float): Current price ($/kWh).
        """
        return {
            **super().get_state(),
            'current_price_per_mwh': float(self.current_price_per_mwh),
            'current_price_per_kwh': float(self.current_price_per_kwh)
        }

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """
        Define the physical connection ports for this component.

        Returns:
            Dict[str, Dict[str, str]]: Empty dict (no physical connections).
        """
        return {}