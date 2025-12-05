from typing import Any, Dict
from h2_plant.core.component import Component

class Tank(Component):
    """
    Hydrogen Storage Tank Component.
    Implements simple mass balance storage.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.config = config
        
        self.capacity_kg = config.get("capacity_kg", 1000.0)
        self.initial_level_kg = config.get("initial_level_kg", 0.0)
        self.min_level_ratio = config.get("min_level_ratio", 0.05)
        self.max_pressure_bar = config.get("max_pressure_bar", 200.0)
        
        # State
        self.current_level_kg = self.initial_level_kg
        self.pressure_bar = 1.0 # Simplified pressure model

    def initialize(self, dt: float, registry: Any) -> None:
        """Initialize component."""
        super().initialize(dt, registry)

    def step(self, t: float, flow_in_kg_h: float = 0.0, flow_out_kg_h: float = 0.0) -> None:
        """
        Update tank level.
        Args:
            t: Simulation time (hours since start? No, t is usually absolute time, we need dt)
            flow_in_kg_h: Inflow
            flow_out_kg_h: Outflow
        """
        # Note: Component.step signature is (self, t). We need dt.
        # Assuming dt is passed in initialize or we calculate it.
        # For now, we assume 1 hour steps or we need to track time.
        # Actually, Component.initialize(dt) sets self.dt
        
        super().step(t)
        dt_hours = getattr(self, 'dt', 1.0) # Default to 1h if not set
        
        delta_mass = (flow_in_kg_h - flow_out_kg_h) * dt_hours
        
        self.current_level_kg += delta_mass
        
        # Clamp
        self.current_level_kg = max(0.0, min(self.current_level_kg, self.capacity_kg))
        
        # Simple linear pressure model (Ideal Gas approx at constant T/V)
        # P = (m / m_max) * P_max
        self.pressure_bar = (self.current_level_kg / self.capacity_kg) * self.max_pressure_bar
        
    def get_state(self) -> Dict[str, Any]:
        return {
            "level_kg": self.current_level_kg,
            "fill_percentage": (self.current_level_kg / self.capacity_kg) * 100.0,
            "pressure_bar": self.pressure_bar
        }
