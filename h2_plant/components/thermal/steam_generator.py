"""
Steam Generator component for converting water to steam.

Used in SOEC and ATR paths (HX-4, HX-7 in Process Flow).
"""

from typing import Dict, Any
from h2_plant.core.component import Component
from h2_plant.core.stream import Stream


class SteamGenerator(Component):
    """
    Steam generator for producing high-temperature steam.
    
    Converts pressurized water to steam using heat input.
    Used in Process Flow as HX-4 (SOEC) and HX-7 (ATR).
    """
    
    def __init__(
        self,
        component_id: str = "steam_generator",
        max_capacity_kg_h: float = 100.0,
        efficiency: float = 0.90,
        steam_temp_k: float = 423.15  # 150°C default
    ):
        """
        Initialize SteamGenerator.
        
        Args:
            component_id: Unique identifier
            max_capacity_kg_h: Maximum steam generation capacity in kg/h
            efficiency: Heat transfer efficiency (0-1)
            steam_temp_k: Steam outlet temperature in Kelvin
        """
        super().__init__()
        self.component_id = component_id
        self.max_capacity_kg_h = max_capacity_kg_h
        self.efficiency = efficiency
        self.steam_temp_k = steam_temp_k
        
        # State variables
        self.water_inlet_kg_h = 0.0
        self.steam_outlet_kg_h = 0.0
        self.heat_input_kw = 0.0
        
    def initialize(self, dt: float, registry: Any) -> None:
        """Initialize component."""
        super().initialize(dt, registry)
        self._initialized = True
        
    def step(self, t: float) -> None:
        """
        Execute one timestep of steam generation.
        
        Args:
            t: Current simulation time in hours
        """
        super().step(t)
        
        # Simple pass-through: output = input (stub implementation)
        self.steam_outlet_kg_h = self.water_inlet_kg_h
        
        # Estimate heat required (water -> steam)
        # Latent heat of vaporization ~2260 kJ/kg
        self.heat_input_kw = self.water_inlet_kg_h * 2260.0 / 3600.0  # kW
        
    def get_state(self) -> Dict[str, Any]:
        """Return current component state."""
        return {
            **super().get_state(),
            "water_inlet_kg_h": self.water_inlet_kg_h,
            "steam_outlet_kg_h": self.steam_outlet_kg_h,
            "heat_input_kw": self.heat_input_kw,
            "steam_temp_k": self.steam_temp_k
        }

    def get_output(self, port_name: str) -> Any:
        """Get output from specific port."""
        if port_name in ['steam_out', 'out']:
            return Stream(
                mass_flow_kg_h=self.steam_outlet_kg_h,
                temperature_k=self.steam_temp_k,
                pressure_pa=30e5, # Assumed pressure
                composition={'H2O': 1.0},
                phase='gas'
            )
        else:
            raise ValueError(f"Unknown output port '{port_name}' on {self.component_id}")

    def receive_input(self, port_name: str, value: Any, resource_type: str) -> float:
        """Receive input."""
        if port_name in ['water_in', 'in']:
            if isinstance(value, Stream):
                self.water_inlet_kg_h = value.mass_flow_kg_h
            else:
                self.water_inlet_kg_h = float(value)
            return self.water_inlet_kg_h
        return 0.0
