"""
Source-Isolated Hydrogen Tank Storage System.

This module implements a storage system that maintains physical separation
between hydrogen produced from different sources. This is essential for
emissions accounting and regulatory compliance.

Design Rationale:
    - **Green hydrogen** (electrolysis): Zero emissions at production.
    - **Blue hydrogen** (ATR with CCS): ~10 kg CO₂/kg H₂ (typical).
    - **Gray hydrogen** (SMR without CCS): ~10-12 kg CO₂/kg H₂.

    Mixing hydrogen from different sources would prevent accurate emissions
    attribution, so this component maintains dedicated tank arrays per source.

Discharge Strategy:
    By default, hydrogen is discharged from the lowest-emissions source first.
    This minimizes the emissions intensity of delivered hydrogen.

Architecture:
    Implements the Component Lifecycle Contract (Layer 1):
    - `initialize()`: Initializes all source-specific tank arrays.
    - `step()`: Advances all tank arrays by one timestep.
    - `get_state()`: Returns per-source and aggregate metrics.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging

from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.enums import TankState


class TankArray:
    """
    Simple tank array for source-isolated storage.

    Provides basic fill/discharge operations with mass tracking.
    This is a lightweight implementation for source isolation.

    Attributes:
        n_tanks (int): Number of tanks in this array.
        capacity_kg (float): Per-tank capacity (kg).
        current_mass (float): Total stored mass (kg).
        max_mass (float): Total array capacity (kg).
    """

    def __init__(self, n_tanks: int, capacity_kg: float, pressure_bar: float):
        """
        Initialize the tank array.

        Args:
            n_tanks (int): Number of tanks.
            capacity_kg (float): Capacity per tank in kg.
            pressure_bar (float): Operating pressure in bar.
        """
        self.n_tanks = n_tanks
        self.capacity_kg = capacity_kg
        self.pressure_bar = pressure_bar
        self.current_mass = 0.0
        self.max_mass = n_tanks * capacity_kg

    def initialize(self, dt: float, registry: Any) -> None:
        """Initialize tank array (no-op)."""
        pass

    def step(self, t: float) -> None:
        """Execute timestep (no-op for simple array)."""
        pass

    def fill(self, mass_kg: float) -> tuple[float, float]:
        """
        Fill tanks with hydrogen.

        Args:
            mass_kg (float): Mass to store in kg.

        Returns:
            tuple[float, float]: (mass_stored, mass_overflow).
        """
        space = self.max_mass - self.current_mass
        stored = min(mass_kg, space)
        self.current_mass += stored
        overflow = mass_kg - stored
        return stored, overflow

    def discharge(self, mass_kg: float) -> float:
        """
        Discharge hydrogen from tanks.

        Args:
            mass_kg (float): Requested mass in kg.

        Returns:
            float: Actual mass discharged.
        """
        discharged = min(mass_kg, self.current_mass)
        self.current_mass -= discharged
        return discharged

    def get_total_mass(self) -> float:
        """Return total stored mass (kg)."""
        return self.current_mass

    def get_available_capacity(self) -> float:
        """Return available capacity (kg)."""
        return self.max_mass - self.current_mass


logger = logging.getLogger(__name__)


@dataclass
class SourceTag:
    """
    Metadata identifying a hydrogen production source.

    Attributes:
        source_id (str): Unique identifier (e.g., 'electrolyzer_1').
        source_type (str): Category (e.g., 'electrolyzer', 'atr').
        emissions_factor (float): Emissions intensity (kg CO₂/kg H₂).
    """
    source_id: str
    source_type: str
    emissions_factor: float


class SourceIsolatedTanks(Component):
    """
    Storage system maintaining physical separation by production source.

    Uses dedicated TankArray instances for each source to ensure no mixing.
    Enables accurate emissions tracking and lowest-emissions-first dispatch.

    This component fulfills the Component Lifecycle Contract (Layer 1):
        - `initialize()`: Initializes all source-specific tank arrays.
        - `step()`: Advances all tank arrays by one timestep.
        - `get_state()`: Returns per-source masses and aggregate metrics.

    Discharge Priority:
        By default, hydrogen is discharged from the source with the lowest
        emissions factor first, minimizing delivered hydrogen intensity.

    Attributes:
        sources (Dict[str, SourceTag]): Source metadata by name.
        tanks_per_source (int): Number of tanks per source.
        fills_by_source (Dict[str, float]): Cumulative fills per source.
        discharges_by_source (Dict[str, float]): Cumulative discharges.

    Example:
        >>> storage = SourceIsolatedTanks(
        ...     sources={
        ...         'electrolyzer': SourceTag('elec_1', 'electrolyzer', 0.0),
        ...         'atr': SourceTag('atr_1', 'atr', 10.5)
        ...     },
        ...     tanks_per_source=4,
        ...     capacity_kg=200.0,
        ...     pressure_bar=350
        ... )
        >>> storage.fill('electrolyzer', 150.0)
        >>> mass, source = storage.discharge(100.0)
    """

    def __init__(
        self,
        sources: Dict[str, SourceTag],
        tanks_per_source: int,
        capacity_kg: float,
        pressure_bar: float
    ):
        """
        Initialize the source-isolated tank system.

        Args:
            sources (Dict[str, SourceTag]): Dictionary mapping source names
                to SourceTag metadata containing emissions factors.
            tanks_per_source (int): Number of tanks allocated per source.
            capacity_kg (float): Capacity of each tank in kg.
            pressure_bar (float): Operating pressure in bar.
        """
        super().__init__()

        self.sources = sources
        self.tanks_per_source = tanks_per_source
        self.capacity_kg = capacity_kg
        self.pressure_bar = pressure_bar

        # Create dedicated TankArray for each source
        self._tank_arrays: Dict[str, TankArray] = {}
        for source_name in sources.keys():
            self._tank_arrays[source_name] = TankArray(
                n_tanks=tanks_per_source,
                capacity_kg=capacity_kg,
                pressure_bar=pressure_bar
            )

        # Cumulative tracking
        self.fills_by_source: Dict[str, float] = {s: 0.0 for s in sources.keys()}
        self.discharges_by_source: Dict[str, float] = {s: 0.0 for s in sources.keys()}

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """
        Prepare all tank arrays for simulation execution.

        Fulfills the Component Lifecycle Contract initialization phase.

        Args:
            dt (float): Simulation timestep in hours.
            registry (ComponentRegistry): Central registry for component access.
        """
        super().initialize(dt, registry)

        for tank_array in self._tank_arrays.values():
            tank_array.initialize(dt, registry)

    def step(self, t: float) -> None:
        """
        Execute one simulation timestep on all tank arrays.

        Args:
            t (float): Current simulation time in hours.
        """
        super().step(t)

        for tank_array in self._tank_arrays.values():
            tank_array.step(t)

    def get_state(self) -> Dict[str, Any]:
        """
        Retrieve the component's current operational state.

        Fulfills the Component Lifecycle Contract state access.

        Returns:
            Dict[str, Any]: State dictionary containing per-source masses,
                capacities, cumulative statistics, and aggregate metrics.
        """
        state = {**super().get_state()}

        for source_name, tank_array in self._tank_arrays.items():
            state[f'{source_name}_mass_kg'] = tank_array.get_total_mass()
            state[f'{source_name}_available_capacity_kg'] = tank_array.get_available_capacity()
            state[f'{source_name}_fills_kg'] = self.fills_by_source[source_name]
            state[f'{source_name}_discharges_kg'] = self.discharges_by_source[source_name]

        state['total_mass_kg'] = self.get_total_mass()
        state['total_available_capacity_kg'] = self.get_total_available_capacity()

        return state

    def fill(self, source_name: str, mass_kg: float) -> tuple[float, float]:
        """
        Fill tanks from a specific production source.

        Args:
            source_name (str): Source identifier (must exist in sources dict).
            mass_kg (float): Mass to store in kg.

        Returns:
            tuple[float, float]: (mass_stored, mass_overflow).

        Raises:
            ValueError: If source_name is not recognized.
        """
        if source_name not in self._tank_arrays:
            raise ValueError(
                f"Unknown source '{source_name}'. Available: {list(self._tank_arrays.keys())}"
            )

        stored, overflow = self._tank_arrays[source_name].fill(mass_kg)
        self.fills_by_source[source_name] += stored

        return stored, overflow

    def discharge(
        self,
        mass_kg: float,
        priority_source: Optional[str] = None
    ) -> tuple[float, str]:
        """
        Discharge hydrogen, prioritizing lowest-emissions source.

        Args:
            mass_kg (float): Mass to discharge in kg.
            priority_source (str, optional): Source to use first. If None,
                uses source with lowest emissions factor.

        Returns:
            tuple[float, str]: (mass_discharged, source_used).
        """
        if priority_source is None:
            priority_source = self._get_lowest_emissions_source()

        discharged = 0.0
        if priority_source in self._tank_arrays:
            discharged = self._tank_arrays[priority_source].discharge(mass_kg)
            self.discharges_by_source[priority_source] += discharged

        # Draw from other sources if needed (lowest emissions first)
        remaining = mass_kg - discharged
        if remaining > 0.01:
            other_sources = sorted(
                [s for s in self.sources.items() if s[0] != priority_source],
                key=lambda item: item[1].emissions_factor
            )
            for source_name, _ in other_sources:
                if source_name in self._tank_arrays:
                    additional = self._tank_arrays[source_name].discharge(remaining)
                    self.discharges_by_source[source_name] += additional
                    discharged += additional
                    remaining -= additional

                    if remaining < 0.01:
                        break

        return discharged, priority_source

    def get_total_mass(self) -> float:
        """
        Return total mass across all sources.

        Returns:
            float: Total stored mass in kg.
        """
        return sum(ta.get_total_mass() for ta in self._tank_arrays.values())

    def get_total_available_capacity(self) -> float:
        """
        Return total available capacity across all sources.

        Returns:
            float: Available capacity in kg.
        """
        return sum(ta.get_available_capacity() for ta in self._tank_arrays.values())

    def get_mass_by_source(self, source_name: str) -> float:
        """
        Return mass stored from a specific source.

        Args:
            source_name (str): Source identifier.

        Returns:
            float: Stored mass in kg.
        """
        return self._tank_arrays[source_name].get_total_mass()

    def _get_lowest_emissions_source(self) -> str:
        """
        Identify the source with lowest emissions that has inventory.

        Returns:
            str: Source name with lowest emissions factor and available mass.
        """
        sources_with_mass = [
            (name, tag) for name, tag in self.sources.items()
            if name in self._tank_arrays and self._tank_arrays[name].get_total_mass() > 0.01
        ]

        if not sources_with_mass:
            return min(self.sources.items(), key=lambda x: x[1].emissions_factor)[0]

        return min(sources_with_mass, key=lambda x: x[1].emissions_factor)[0]

    def get_weighted_emissions_factor(self) -> float:
        """
        Calculate mass-weighted average emissions factor.

        Returns:
            float: Weighted average emissions (kg CO₂/kg H₂).
        """
        total_mass = 0.0
        weighted_emissions = 0.0

        for source_name, source_tag in self.sources.items():
            if source_name in self._tank_arrays:
                mass = self._tank_arrays[source_name].get_total_mass()
                total_mass += mass
                weighted_emissions += mass * source_tag.emissions_factor

        if total_mass > 0:
            return weighted_emissions / total_mass
        return 0.0
