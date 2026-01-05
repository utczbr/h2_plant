"""
External Water Source Component.

This module provides a water supply from external sources (municipal line, 
reservoir, etc.) to feed the electrolysis and water treatment systems.

Supply Modes:
    - **fixed_flow**: Constant delivery rate.
    - **on_demand**: Delivers whatever quantity is requested (infinite source).
    - **scaled**: Delivers flow scaling with a factor.

Architecture:
    Implements the Component Lifecycle Contract (Layer 1):
    - `initialize()`: Connects to downstream targets.
    - `step()`: Delivers water according to configured mode.
    - `get_state()`: Returns flow and connection metrics.
"""

import logging
from typing import Dict, Any, Optional

from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.stream import Stream

logger = logging.getLogger(__name__)


class ExternalWaterSource(Component):
    """
    External water supply source providing water at configurable flow rate.

    Attributes:
        mode (str): Operating mode ('fixed_flow', 'on_demand', 'scaled').
        flow_rate_kg_h (float): Configured flow rate (kg/h).
        pressure_bar (float): Supply pressure (bar).
        temperature_c (float): Water temperature (°C).
    """

    def __init__(
        self,
        mode: str = "fixed_flow",
        flow_rate_kg_h: float = 10000.0,
        pressure_bar: float = 5.0,
        cost_per_m3: float = 2.0,
        temperature_c: float = 20.0,
        config: Dict[str, Any] = None
    ):
        """
        Initialize the external water source.

        Args:
            mode (str): 'fixed_flow', 'on_demand', 'scaled'.
            flow_rate_kg_h (float): Flow rate in kg/h.
            pressure_bar (float): Supply pressure in bar.
            cost_per_m3 (float): Cost in EUR per cubic meter.
            temperature_c (float): Water temperature in Celsius.
            config (dict, optional): Configuration dictionary.
        """
        super().__init__()
        
        # Handle dict config if passed (Pattern used by PlantBuilder)
        if isinstance(mode, dict):
            config = mode
            mode = config.get('supply_mode', 'fixed_flow')
            if 'flow_rate_kg_h' in config:
                flow_rate_kg_h = float(config['flow_rate_kg_h'])
            else:
                # mode_value in m³/h from GUI, convert to kg/h (approx 1000 kg/m³)
                mode_value = float(config.get('mode_value', 100.0))
                flow_rate_kg_h = mode_value * 1000.0  # m³/h -> kg/h
            pressure_bar = float(config.get('pressure_bar', 5.0))
            cost_per_m3 = float(config.get('cost_per_m3', 2.0))
            temperature_c = float(config.get('temperature_c', 20.0))
            if 'component_id' in config:
                self.component_id = config['component_id']

        self.mode = mode
        self.flow_rate_kg_h = float(flow_rate_kg_h)
        self.pressure_bar = float(pressure_bar)
        self.cost_per_m3 = float(cost_per_m3)
        self.temperature_c = float(temperature_c)

        # State variables
        self.water_output_kg = 0.0
        self.cumulative_water_kg = 0.0
        self.cumulative_cost = 0.0

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """Prepare the component for simulation."""
        super().initialize(dt, registry)
        
        # Pre-allocate output stream
        self._output_stream = Stream(
            mass_flow_kg_h=self.flow_rate_kg_h,
            temperature_k=273.15 + self.temperature_c,
            pressure_pa=self.pressure_bar * 1e5,
            composition={'H2O': 1.0},
            phase='liquid'
        )

    def step(self, t: float) -> None:
        """Execute one simulation timestep."""
        super().step(t)

        # Determine flow rate based on mode
        current_flow_kg_h = self.flow_rate_kg_h
        
        # Calculate periodic output
        self.water_output_kg = current_flow_kg_h * self.dt
        
        # Update counters
        self.cumulative_water_kg += self.water_output_kg
        # Approx 1000 kg = 1 m³
        self.cumulative_cost += (self.water_output_kg / 1000.0) * self.cost_per_m3
        
        # Update cached stream
        self._output_stream.mass_flow_kg_h = current_flow_kg_h

    def get_output(self, port_name: str) -> Any:
        """Retrieve the output stream from a specified port."""
        if port_name == 'water_out':
            return self._output_stream
        else:
            raise ValueError(f"Unknown output port '{port_name}' on {self.component_id}")

    def extract_output(self, port_name: str, amount: float, resource_type: str) -> None:
        """Called by FlowNetwork after downstream accepts the flow."""
        # For an infinite source, no state update needed
        pass

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """Define the physical connection ports."""
        return {
            'water_out': {'type': 'output', 'resource_type': 'water', 'units': 'kg/h'}
        }

    def get_state(self) -> Dict[str, Any]:
        """Retrieve the component's current operational state."""
        return {
            **super().get_state(),
            'mode': self.mode,
            'flow_rate_kg_h': self.flow_rate_kg_h,
            'water_output_kg': self.water_output_kg,
            'cumulative_water_kg': self.cumulative_water_kg,
            'cumulative_cost': self.cumulative_cost
        }
