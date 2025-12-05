"""
CO2 Storage component for carbon capture system.

Stores compressed CO2 from capture process.
"""

from typing import Dict, Any
from h2_plant.core.component import Component


class CO2Storage(Component):
    """
    CO2 storage tank for captured carbon dioxide.
    
    Receives compressed CO2 from capture system for sequestration or utilization.
    """
    
    def __init__(
        self,
        component_id: str = "co2_storage",
        capacity_kg: float = 10000.0,
        pressure_bar: float = 100.0,
        temperature_k: float = 298.15
    ):
        """
        Initialize CO2Storage.
        
        Args:
            component_id: Unique identifier
            capacity_kg: Storage capacity in kg
            pressure_bar: Storage pressure in bar
            temperature_k: Storage temperature in Kelvin
        """
        super().__init__()
        self.component_id = component_id
        self.capacity_kg = capacity_kg
        self.pressure_bar = pressure_bar
        self.temperature_k = temperature_k
        
        # State variables
        self.stored_co2_kg = 0.0
        self.inlet_flow_kg_h = 0.0
        self.cumulative_stored_kg = 0.0
        
    def initialize(self, dt: float, registry: Any) -> None:
        """Initialize component."""
        super().initialize(dt, registry)
        self._initialized = True
        
    def step(self, t: float) -> None:
        """
        Execute one timestep of CO2 storage.
        
        Args:
            t: Current simulation time in hours
        """
        super().step(t)
        
        # Simple accumulation
        inlet_kg = self.inlet_flow_kg_h * self.dt
        self.stored_co2_kg += inlet_kg
        self.cumulative_stored_kg += inlet_kg
        
        # Cap at capacity
        if self.stored_co2_kg > self.capacity_kg:
            self.stored_co2_kg = self.capacity_kg
        
    def get_state(self) -> Dict[str, Any]:
        """Return current component state."""
        return {
            **super().get_state(),
            "stored_co2_kg": self.stored_co2_kg,
            "inlet_flow_kg_h": self.inlet_flow_kg_h,
            "cumulative_stored_kg": self.cumulative_stored_kg,
            "fill_level": self.stored_co2_kg / self.capacity_kg if self.capacity_kg > 0 else 0.0
        }
