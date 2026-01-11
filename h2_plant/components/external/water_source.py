"""
External Water Source Component with Signal Control.

This module provides a water supply from external sources (municipal line,
reservoir, etc.) to feed the electrolysis and water treatment systems.

Supply Modes:
    - **fixed_flow**: Constant delivery rate (default).
    - **external_control**: Flow rate dictated by downstream signal (e.g., tank level).
    - **on_demand**: Delivers whatever quantity is requested (infinite source).

Signal Control Architecture:
    When in 'external_control' mode, the source reads a control signal from
    a downstream tank. The signal arrives as a "Signal Stream" where
    mass_flow_kg_h encodes the requested production rate.
    
    This implements the "Demand Signal Propagation Pattern" where:
    - Tank calculates desired production based on level zones.
    - Source adjusts its output to match the request.
    - Purifier processes whatever arrives (push architecture).

Architecture:
    Implements the Component Lifecycle Contract (Layer 1):
    - `initialize()`: Pre-allocates output stream.
    - `step()`: Delivers water according to configured mode.
    - `get_state()`: Returns flow, cost, and mode metrics.
"""

import logging
from typing import Dict, Any, Optional

from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.stream import Stream

logger = logging.getLogger(__name__)


class ExternalWaterSource(Component):
    """
    External water supply source with signal-controlled flow modulation.

    Supports fixed flow or dynamic control based on downstream tank signals.

    Attributes:
        mode (str): Operating mode ('fixed_flow', 'external_control', 'on_demand').
        flow_rate_kg_h (float): Base/configured flow rate (kg/h).
        current_flow_kg_h (float): Active flow rate after signal modulation.
        pressure_bar (float): Supply pressure (bar).
        temperature_c (float): Water temperature (°C).

    Example (Fixed Mode):
        >>> source = ExternalWaterSource(mode='fixed_flow', flow_rate_kg_h=12000.0)
        
    Example (Signal Control):
        >>> source = ExternalWaterSource(mode='external_control', flow_rate_kg_h=12000.0)
        >>> # Tank sends signal: source.receive_input('control_signal', signal_stream)
        >>> source.step(t=0.0)
        >>> # Output flow now matches signal
    """

    def __init__(
        self,
        mode: str = "fixed_flow",
        flow_rate_kg_h: float = 10000.0,
        pressure_bar: float = 5.0,
        cost_per_m3: float = 2.0,
        temperature_c: float = 15.0,
        config: Dict[str, Any] = None
    ):
        """
        Initialize the external water source.

        Args:
            mode (str): 'fixed_flow', 'external_control', or 'on_demand'.
            flow_rate_kg_h (float): Base flow rate in kg/h.
            pressure_bar (float): Supply pressure in bar.
            cost_per_m3 (float): Cost in EUR per cubic meter.
            temperature_c (float): Water temperature in Celsius.
            config (dict, optional): Configuration dictionary override.
        """
        super().__init__()

        # Handle dict config if passed (Pattern used by PlantBuilder)
        if isinstance(mode, dict):
            config = mode
            mode = config.get('supply_mode', config.get('mode', 'fixed_flow'))
            if 'flow_rate_kg_h' in config:
                flow_rate_kg_h = float(config['flow_rate_kg_h'])
            else:
                # mode_value in m³/h from GUI, convert to kg/h
                mode_value = float(config.get('mode_value', 10.0))
                flow_rate_kg_h = mode_value * 1000.0
            pressure_bar = float(config.get('pressure_bar', 5.0))
            cost_per_m3 = float(config.get('cost_per_m3', 2.0))
            temperature_c = float(config.get('temperature_c', 15.0))
            if 'component_id' in config:
                self.component_id = config['component_id']

        self.mode = mode
        self.flow_rate_kg_h = float(flow_rate_kg_h)  # Base/max rate
        self.current_flow_kg_h = float(flow_rate_kg_h)  # Active rate
        self.pressure_bar = float(pressure_bar)
        self.cost_per_m3 = float(cost_per_m3)
        self.temperature_c = float(temperature_c)

        # Signal buffer (for external_control mode)
        self._signal_request_kg_h: Optional[float] = None

        # State variables
        self.water_output_kg = 0.0
        self.cumulative_water_kg = 0.0
        self.cumulative_cost = 0.0

        # Pre-allocated output stream (updated in step)
        self._output_stream: Optional[Stream] = None

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """
        Prepare the component for simulation.

        Args:
            dt (float): Simulation timestep in hours.
            registry (ComponentRegistry): Central registry for component access.
        """
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
        """
        Execute one simulation timestep.

        In 'external_control' mode, adjusts flow based on received signal.
        The signal is consumed (reset) after each step to enforce
        1-timestep causality.

        Args:
            t (float): Current simulation time in hours.
        """
        super().step(t)

        # Determine flow rate based on mode
        if self.mode == 'external_control':
            if self._signal_request_kg_h is not None:
                # Clamp to physical limits (0 to max capacity)
                self.current_flow_kg_h = max(0.0, min(
                    self._signal_request_kg_h,
                    self.flow_rate_kg_h  # flow_rate_kg_h acts as max
                ))
            else:
                # No control signal received yet - Default to 0 (Fail Closed)
                # This prevents overfilling if the controller hasn't initialized
                self.current_flow_kg_h = 0.0
            
            # DEBUG: Log signal and flow at WARNING level for visibility
            logger.warning(f"WATER_SOURCE STEP: signal_buffer={self._signal_request_kg_h}, flow={self.current_flow_kg_h:.0f} kg/h")
                
            # NOTE: We do NOT reset _signal_request_kg_h to None here.
            # We implement a Zero-Order Hold, maintaining the last valid setpoint
            # until a new signal arrives. This is more robust for simulation steps.
            
        elif self.mode == 'on_demand':
            # In on_demand mode, we could implement pull-based logic
            # For now, treat as fixed_flow
            self.current_flow_kg_h = self.flow_rate_kg_h
            
        else:  # 'fixed_flow' (default)
            self.current_flow_kg_h = self.flow_rate_kg_h

        # Calculate periodic output
        self.water_output_kg = self.current_flow_kg_h * self.dt

        # Update counters
        self.cumulative_water_kg += self.water_output_kg
        # Approx 1000 kg = 1 m³
        self.cumulative_cost += (self.water_output_kg / 1000.0) * self.cost_per_m3

        # Update cached stream
        self._output_stream.mass_flow_kg_h = self.current_flow_kg_h
        self._output_stream.temperature_k = 273.15 + self.temperature_c
        self._output_stream.pressure_pa = self.pressure_bar * 1e5

    def receive_input(self, port_name: str, value: Any, resource_type: str = None) -> float:
        """
        Accept control signal to set flow rate for NEXT step.

        In 'external_control' mode, reads the signal from a downstream
        tank and uses it to modulate output flow.

        Args:
            port_name (str): Port identifier ('control_signal').
            value (Any): Signal Stream where mass_flow_kg_h encodes request.
            resource_type (str, optional): Resource classification hint.

        Returns:
            float: Always 0.0 (signals are not consumed resources).
        """
        if port_name == 'control_signal':
            if isinstance(value, Stream):
                # Signal Stream: mass_flow_kg_h encodes the request
                self._signal_request_kg_h = value.mass_flow_kg_h
                # DEBUG: Log signal reception at WARNING level for visibility
                logger.warning(
                    f"WATER_SOURCE RECEIVE: signal={self._signal_request_kg_h:.0f} kg/h"
                )
            elif isinstance(value, (int, float)):
                # Direct numeric value (legacy support)
                self._signal_request_kg_h = float(value)
            return 0.0
        return 0.0

    def get_output(self, port_name: str) -> Any:
        """
        Retrieve the output stream from a specified port.

        Args:
            port_name (str): Port identifier ('water_out').

        Returns:
            Stream: Water output stream.

        Raises:
            ValueError: If unknown port requested.
        """
        if port_name == 'water_out':
            return self._output_stream
        else:
            logger.warning(f"ExternalWaterSource: Unknown output port '{port_name}'")
            return None

    def extract_output(self, port_name: str, amount: float, resource_type: str) -> None:
        """
        Called by FlowNetwork after downstream accepts the flow.
        
        For an infinite source, no state update needed.
        """
        pass

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """
        Define the physical connection ports.

        Returns:
            Dict[str, Dict[str, str]]: Port definitions.
        """
        return {
            'control_signal': {'type': 'input', 'resource_type': 'signal', 'units': 'kg/h'},
            'water_out': {'type': 'output', 'resource_type': 'water', 'units': 'kg/h'}
        }

    def get_state(self) -> Dict[str, Any]:
        """
        Retrieve the component's current operational state.

        Returns:
            Dict[str, Any]: State dictionary containing flow and cost metrics.
        """
        return {
            **super().get_state(),
            'mode': self.mode,
            'flow_rate_kg_h': self.flow_rate_kg_h,
            'current_flow_kg_h': self.current_flow_kg_h,
            'water_output_kg': self.water_output_kg,
            'cumulative_water_kg': self.cumulative_water_kg,
            'cumulative_cost': self.cumulative_cost
        }
