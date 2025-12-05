"""
Source-isolated tank storage system.

Maintains physical separation between hydrogen from different production
sources (e.g., electrolyzer vs ATR) for emissions tracking and compliance.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging

from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.enums import TankState
from h2_plant.components.storage.tank_array import TankArray

logger = logging.getLogger(__name__)


@dataclass
class SourceTag:
    """Tag identifying hydrogen source."""
    source_id: str          # e.g., "electrolyzer_1", "atr_1"
    source_type: str        # e.g., "electrolyzer", "atr"
    emissions_factor: float # kg CO2 per kg H2


class SourceIsolatedTanks(Component):
    """
    Storage system maintaining physical separation by production source.
    
    Uses multiple TankArray instances, each dedicated to a specific source.
    Ensures no mixing between sources for emissions accounting.
    
    Example:
        storage = SourceIsolatedTanks(
            sources={
                'electrolyzer': SourceTag('elec_1', 'electrolyzer', 0.0),
                'atr': SourceTag('atr_1', 'atr', 10.5)
            },
            tanks_per_source=4,
            capacity_kg=200.0,
            pressure_bar=350
        )
        
        # Fill from electrolyzer
        storage.fill('electrolyzer', 150.0)
        
        # Discharge (prioritizes lowest emissions)
        mass, source = storage.discharge(100.0)
    """
    
    def __init__(
        self,
        sources: Dict[str, SourceTag],
        tanks_per_source: int,
        capacity_kg: float,
        pressure_bar: float
    ):
        """
        Initialize source-isolated tank system.
        
        Args:
            sources: Dictionary mapping source names to SourceTag metadata
            tanks_per_source: Number of tanks allocated to each source
            capacity_kg: Capacity of each tank (kg)
            pressure_bar: Operating pressure (bar)
        """
        super().__init__()
        
        self.sources = sources
        self.tanks_per_source = tanks_per_source
        self.capacity_kg = capacity_kg
        self.pressure_bar = pressure_bar
        
        # Create TankArray for each source
        self._tank_arrays: Dict[str, TankArray] = {}
        for source_name in sources.keys():
            self._tank_arrays[source_name] = TankArray(
                n_tanks=tanks_per_source,
                capacity_kg=capacity_kg,
                pressure_bar=pressure_bar
            )
        
        # Tracking
        self.fills_by_source: Dict[str, float] = {s: 0.0 for s in sources.keys()}
        self.discharges_by_source: Dict[str, float] = {s: 0.0 for s in sources.keys()}
    
    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """Initialize all tank arrays."""
        super().initialize(dt, registry)
        
        for tank_array in self._tank_arrays.values():
            tank_array.initialize(dt, registry)
    
    def step(self, t: float) -> None:
        """Execute timestep on all tank arrays."""
        super().step(t)
        
        for tank_array in self._tank_arrays.values():
            tank_array.step(t)
    
    def get_state(self) -> Dict[str, Any]:
        """Return current state."""
        state = {**super().get_state()}
        
        # Add per-source tank states
        for source_name, tank_array in self._tank_arrays.items():
            state[f'{source_name}_mass_kg'] = tank_array.get_total_mass()
            state[f'{source_name}_available_capacity_kg'] = tank_array.get_available_capacity()
            state[f'{source_name}_fills_kg'] = self.fills_by_source[source_name]
            state[f'{source_name}_discharges_kg'] = self.discharges_by_source[source_name]
        
        # Add aggregate metrics
        state['total_mass_kg'] = self.get_total_mass()
        state['total_available_capacity_kg'] = self.get_total_available_capacity()
        
        return state
    
    def fill(self, source_name: str, mass_kg: float) -> tuple[float, float]:
        """
        Fill tanks from specific source.
        
        Args:
            source_name: Source identifier (must exist in sources dict)
            mass_kg: Mass to store (kg)
            
        Returns:
            Tuple of (mass_stored, mass_overflow)
            
        Raises:
            ValueError: If source_name not recognized
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
        Discharge hydrogen, optionally prioritizing a specific source.
        
        Args:
            mass_kg: Mass to discharge (kg)
            priority_source: Source to discharge from first (None = lowest emissions)
            
        Returns:
            Tuple of (mass_discharged, source_name)
        """
        if priority_source is None:
            # Default: discharge from lowest emissions source first
            priority_source = self._get_lowest_emissions_source()

        discharged = 0.0
        if priority_source in self._tank_arrays:
            # Try priority source first
            discharged = self._tank_arrays[priority_source].discharge(mass_kg)
            self.discharges_by_source[priority_source] += discharged
        
        # If insufficient, try other sources
        remaining = mass_kg - discharged
        if remaining > 0.01:
            # Create a sorted list of other sources by emission factor
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
        """Return total mass across all sources (kg)."""
        return sum(ta.get_total_mass() for ta in self._tank_arrays.values())
    
    def get_total_available_capacity(self) -> float:
        """Return total available capacity across all sources (kg)."""
        return sum(ta.get_available_capacity() for ta in self._tank_arrays.values())
    
    def get_mass_by_source(self, source_name: str) -> float:
        """Return mass stored from specific source (kg)."""
        return self._tank_arrays[source_name].get_total_mass()
    
    def _get_lowest_emissions_source(self) -> str:
        """Return source with lowest emissions factor that has mass."""
        
        # Filter sources that have mass
        sources_with_mass = [
            (name, tag) for name, tag in self.sources.items() 
            if name in self._tank_arrays and self._tank_arrays[name].get_total_mass() > 0.01
        ]

        if not sources_with_mass:
            # If no source has mass, return the one with the absolute lowest emissions factor
            return min(self.sources.items(), key=lambda x: x[1].emissions_factor)[0]

        # Return the one with the lowest emissions factor among those with mass
        return min(sources_with_mass, key=lambda x: x[1].emissions_factor)[0]

    
    def get_weighted_emissions_factor(self) -> float:
        """
        Calculate mass-weighted average emissions factor for stored hydrogen.
        
        Returns:
            Weighted average kg CO2 per kg H2
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
