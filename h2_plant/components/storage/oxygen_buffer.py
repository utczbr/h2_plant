"""
Oxygen buffer storage for electrolyzer byproduct.

Simple buffer with overflow venting when capacity exceeded.
"""

from typing import Dict, Any

from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry


class OxygenBuffer(Component):
    """
    Buffer storage for oxygen byproduct from electrolysis.
    
    Features:
    - Simple mass balance tracking
    - Overflow venting (no storage limit violation)
    - Usage tracking (if oxygen monetized)
    
    Example:
        o2_buffer = OxygenBuffer(capacity_kg=500.0)
        
        # Add oxygen from electrolyzer
        o2_buffer.add_oxygen(63.5)  # 8 kg O2 per kg H2
        
        # Remove for industrial use
        o2_buffer.remove_oxygen(50.0)
    """
    
    def __init__(self, capacity_kg: float):
        """
        Initialize oxygen buffer.
        
        Args:
            capacity_kg: Maximum oxygen storage capacity (kg)
        """
        super().__init__()
        
        self.capacity_kg = capacity_kg
        self.mass_kg = 0.0
        
        # Tracking
        self.cumulative_added_kg = 0.0
        self.cumulative_removed_kg = 0.0
        self.cumulative_vented_kg = 0.0
    
    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """Initialize buffer."""
        super().initialize(dt, registry)
    
    def step(self, t: float) -> None:
        """Execute timestep."""
        super().step(t)
        # No per-timestep logic needed
    
    def get_state(self) -> Dict[str, Any]:
        """Return current state."""
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
        Add oxygen to buffer.
        
        Args:
            mass_kg: Mass to add (kg)
            
        Returns:
            Mass vented due to overflow (kg)
        """
        available_capacity = self.capacity_kg - self.mass_kg
        
        if mass_kg <= available_capacity:
            self.mass_kg += mass_kg
            self.cumulative_added_kg += mass_kg
            return 0.0
        else:
            # Partial storage + venting
            stored = available_capacity
            vented = mass_kg - stored
            
            self.mass_kg = self.capacity_kg
            self.cumulative_added_kg += stored
            self.cumulative_vented_kg += vented
            
            return vented
    
    def remove_oxygen(self, mass_kg: float) -> float:
        """
        Remove oxygen from buffer (for sale or industrial use).
        
        Args:
            mass_kg: Mass to remove (kg)
            
        Returns:
            Actual mass removed (may be less if insufficient stored)
        """
        removed = min(mass_kg, self.mass_kg)
        self.mass_kg -= removed
        self.cumulative_removed_kg += removed
        
        return removed
