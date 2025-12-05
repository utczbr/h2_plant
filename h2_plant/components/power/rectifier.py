"""
Rectifier/Transformer component for AC/DC power conversion.

Converts grid AC power to DC for electrolyzer stacks.
"""

from typing import Dict, Any
from h2_plant.core.component import Component


class Rectifier(Component):
    """
    AC/DC rectifier and transformer for electrolyzer power conditioning.
    
    Converts AC grid power to DC suitable for electrolyzer stacks.
    Includes voltage transformation and power factor correction.
    
    Used in Process Flow as RT-1 (PEM), RT-2 (SOEC).
    """
    
    def __init__(
        self,
        component_id: str = "rectifier",
        rated_power_mw: float = 10.0,
        efficiency: float = 0.97,
        power_factor: float = 0.95,
        output_voltage_v: float = 1000.0
    ):
        """
        Initialize Rectifier.
        
        Args:
            component_id: Unique identifier
            rated_power_mw: Maximum power rating in MW
            efficiency: Conversion efficiency (0-1)
            power_factor: Input power factor (0-1)
            output_voltage_v: DC output voltage
        """
        super().__init__()
        self.component_id = component_id
        self.rated_power_mw = rated_power_mw
        self.efficiency = efficiency
        self.power_factor = power_factor
        self.output_voltage_v = output_voltage_v
        
        # State variables
        self.ac_input_power_mw: float = 0.0
        self.dc_output_power_mw: float = 0.0
        self.power_loss_mw: float = 0.0
        self.load_factor: float = 0.0
        
    def initialize(self, dt: float, registry: Any) -> None:
        """Initialize component."""
        super().initialize(dt, registry)
        self.initialized = True

    def step(self, t: float) -> None:
        """Execute one timestep of rectifier operation."""
        super().step(t)
        
        # Power conversion with efficiency and losses
        if self.ac_input_power_mw > 0:
            self.dc_output_power_mw = min(
                self.ac_input_power_mw * self.efficiency,
                self.rated_power_mw
            )
            self.power_loss_mw = self.ac_input_power_mw - self.dc_output_power_mw
            self.load_factor = self.dc_output_power_mw / self.rated_power_mw
        else:
            self.dc_output_power_mw = 0.0
            self.power_loss_mw = 0.0
            self.load_factor = 0.0
    
    def get_output(self, port_name: str) -> Any:
        """Get output from specified port."""
        if port_name == "dc_out" or port_name == "electricity_out":
            return self.dc_output_power_mw
        elif port_name == "heat_out":
            return self.power_loss_mw * 1000.0  # Convert to kW
        return 0.0
    
    def receive_input(self, port_name: str, value: Any, resource_type: str = None) -> float:
        """Receive input at specified port."""
        if port_name == "ac_in" or port_name == "electricity_in":
            if isinstance(value, (int, float)):
                self.ac_input_power_mw = min(value, self.rated_power_mw / self.efficiency)
                return self.ac_input_power_mw
        return 0.0
    
    def extract_output(self, port_name: str, amount: float, resource_type: str = None) -> None:
        """Acknowledge extraction of output."""
        pass
    
    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """Return port definitions."""
        return {
            'ac_in': {'type': 'input', 'resource_type': 'electricity'},
            'electricity_in': {'type': 'input', 'resource_type': 'electricity'},
            'dc_out': {'type': 'output', 'resource_type': 'electricity'},
            'electricity_out': {'type': 'output', 'resource_type': 'electricity'},
            'heat_out': {'type': 'output', 'resource_type': 'heat'}
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Return current component state."""
        return {
            **super().get_state(),
            'component_id': self.component_id,
            'ac_input_power_mw': self.ac_input_power_mw,
            'dc_output_power_mw': self.dc_output_power_mw,
            'power_loss_mw': self.power_loss_mw,
            'load_factor': self.load_factor,
            'efficiency': self.efficiency
        }
