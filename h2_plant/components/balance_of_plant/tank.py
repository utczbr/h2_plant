from typing import Any, Dict, Optional, List
from h2_plant.core.component import Component
from h2_plant.core.stream import Stream

class Tank(Component):
    """
    Hydrogen Storage Tank Component.
    Implements mass balance storage with flow network interface.
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
        self.pressure_bar = 1.0  # Simplified pressure model
        
        # Flow network buffer (Push Architecture)
        self._input_buffer: List[Stream] = []
        self._last_inflow_kg_h = 0.0
        self._last_outflow_kg_h = 0.0

    def initialize(self, dt: float, registry: Any) -> None:
        """Initialize component."""
        super().initialize(dt, registry)

    def receive_input(self, port_name: str, value: Any, resource_type: str = 'hydrogen') -> float:
        """
        Receive input stream from upstream component.
        Buffers the input for processing in step().
        """
        # print(f"DEBUG: Tank {self.component_id} receive_input {port_name}")
        if port_name == 'inlet' or port_name == 'h2_in':
            if isinstance(value, Stream):
                if value.mass_flow_kg_h > 0:
                    self._input_buffer.append(value)
                return value.mass_flow_kg_h
        return 0.0

    def step(self, t: float, flow_in_kg_h: float = None, flow_out_kg_h: float = 0.0) -> None:
        """
        Update tank level.
        
        Mass input comes from either:
        1. Internal buffer (from receive_input, flow network mode)
        2. Explicit flow_in_kg_h argument (legacy mode)
        """
        super().step(t)
        dt_hours = getattr(self, 'dt', 1.0)  # Default to 1h if not set
        
        # Calculate inflow from buffer (flow network mode takes priority)
        if self._input_buffer:
            total_inflow_kg_h = sum(s.mass_flow_kg_h for s in self._input_buffer)
            self._input_buffer = []  # Clear buffer
        elif flow_in_kg_h is not None:
            total_inflow_kg_h = flow_in_kg_h
        else:
            total_inflow_kg_h = 0.0
        
        self._last_inflow_kg_h = total_inflow_kg_h
        self._last_outflow_kg_h = flow_out_kg_h
        
        delta_mass = (total_inflow_kg_h - flow_out_kg_h) * dt_hours
        
        self.current_level_kg += delta_mass
        
        # Clamp
        self.current_level_kg = max(0.0, min(self.current_level_kg, self.capacity_kg))
        
        # Simple linear pressure model (Ideal Gas approx at constant T/V)
        # P = (m / m_max) * P_max
        self.pressure_bar = (self.current_level_kg / self.capacity_kg) * self.max_pressure_bar

    def get_output(self, port_name: str) -> Any:
        """Get output from specific port."""
        if port_name == 'outlet' or port_name == 'h2_out':
            # Return current tank state as a stream
            # In reality, outflow should be demand-driven, but for monitoring we can
            # report what's available
            return Stream(
                mass_flow_kg_h=self._last_outflow_kg_h,
                temperature_k=298.15,  # Assumed storage temp
                pressure_pa=self.pressure_bar * 1e5,
                composition={'H2': 1.0},
                phase='gas'
            )
        elif port_name == 'level':
            return self.current_level_kg
        elif port_name == 'pressure':
            return self.pressure_bar
        else:
            raise ValueError(f"Unknown output port '{port_name}' on {self.component_id}")

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """Return port metadata."""
        return {
            'inlet': {'type': 'input', 'resource_type': 'hydrogen', 'units': 'kg/h'},
            'h2_in': {'type': 'input', 'resource_type': 'hydrogen', 'units': 'kg/h'},
            'outlet': {'type': 'output', 'resource_type': 'hydrogen', 'units': 'kg/h'},
            'h2_out': {'type': 'output', 'resource_type': 'hydrogen', 'units': 'kg/h'},
            'level': {'type': 'output', 'resource_type': 'mass', 'units': 'kg'},
            'pressure': {'type': 'output', 'resource_type': 'pressure', 'units': 'bar'}
        }

    def get_state(self) -> Dict[str, Any]:
        return {
            **super().get_state(),
            "level_kg": self.current_level_kg,
            "fill_percentage": (self.current_level_kg / self.capacity_kg) * 100.0,
            "pressure_bar": self.pressure_bar,
            "last_inflow_kg_h": self._last_inflow_kg_h,
            "last_outflow_kg_h": self._last_outflow_kg_h
        }

