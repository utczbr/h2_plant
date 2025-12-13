"""
Simplified PSA Unit Component.

This module provides a simplified Pressure Swing Adsorption (PSA) model
for hydrogen purification. It implements a basic recovery efficiency model
without detailed cycle dynamics.

Separation Model:
    - **Fixed Recovery**: 90% of feed hydrogen is recovered as high-purity
      product. The remaining 10% exits with impurities as tail gas.
    - **Cycle Abstraction**: No explicit bed switching or cycle modeling;
      steady-state behavior assumed at all times.

This simplified model is appropriate for system-level mass balance studies
where detailed PSA dynamics are not required.

Architecture:
    Implements the Component Lifecycle Contract (Layer 1):
    - `initialize()`: Inherited from base Component.
    - `step()`: Consumes input buffer and applies recovery factor.
    - `get_state()`: Returns feed and product flow rates.
"""

from typing import Dict, Any

from h2_plant.core.component import Component
from h2_plant.core.stream import Stream


class PSAUnit(Component):
    """
    Simplified Pressure Swing Adsorption unit for hydrogen purification.

    Applies a fixed recovery efficiency (90%) to produce high-purity hydrogen
    product and tail gas containing impurities. No explicit adsorption
    dynamics or cycle switching is modeled.

    This component fulfills the Component Lifecycle Contract (Layer 1):
        - `initialize()`: Inherited, standard initialization.
        - `step()`: Processes input buffer, calculates product/tail flows.
        - `get_state()`: Returns current feed and product flow rates.

    Attributes:
        gas_type (str): Target gas species ('H2' or 'O2').
        feed_gas_kg_h (float): Current feed gas rate (kg/h).
        product_gas_kg_h (float): Current product gas rate (kg/h).

    Example:
        >>> psa = PSAUnit(component_id='PSA-1', gas_type='H2')
        >>> psa.initialize(dt=1/60, registry=registry)
        >>> psa.receive_input('h2_in', h2_stream, 'hydrogen')
        >>> psa.step(t=0.0)
        >>> product = psa.get_output('h2_out')
    """

    def __init__(self, component_id: str, gas_type: str):
        """
        Initialize the simplified PSA unit.

        Args:
            component_id (str): Unique identifier for this component.
            gas_type (str): Target gas species to purify ('H2' or 'O2').
        """
        super().__init__()
        self.component_id = component_id
        self.gas_type = gas_type
        self.feed_gas_kg_h = 0.0
        self.product_gas_kg_h = 0.0
        self.input_stream = None

        # Input buffer for push architecture
        self._input_mass_buffer = 0.0
        self._last_step_time = -1.0

    def step(self, t: float) -> None:
        """
        Execute one simulation timestep.

        Processes accumulated input buffer and applies 90% recovery
        efficiency to calculate product gas output.

        Args:
            t (float): Current simulation time in hours.
        """
        super().step(t)

        # Reset feed rate on new timestep
        if t != self._last_step_time:
            self.feed_gas_kg_h = 0.0
            self._last_step_time = t

        # Consume buffered input
        if self._input_mass_buffer > 0:
            self.feed_gas_kg_h += self._input_mass_buffer
            self._input_mass_buffer = 0.0

        # Apply fixed 90% recovery efficiency
        self.product_gas_kg_h = self.feed_gas_kg_h * 0.9

    def get_state(self) -> Dict[str, Any]:
        """
        Retrieve the component's current operational state.

        Fulfills the Component Lifecycle Contract state access.

        Returns:
            Dict[str, Any]: State dictionary containing:
                - feed_gas_kg_h (float): Current feed rate (kg/h).
                - product_gas_kg_h (float): Current product rate (kg/h).
        """
        return {
            **super().get_state(),
            "feed_gas_kg_h": self.feed_gas_kg_h,
            "product_gas_kg_h": self.product_gas_kg_h
        }

    def get_output(self, port_name: str) -> Any:
        """
        Retrieve output stream from specified port.

        Args:
            port_name (str): Port identifier ('h2_out' or 'tail_gas_out').

        Returns:
            Stream: Output stream with appropriate flow and composition.

        Raises:
            ValueError: If port_name is not a valid output port.
        """
        if port_name == 'h2_out':
            if hasattr(self, 'input_stream') and self.input_stream:
                return Stream(
                    mass_flow_kg_h=self.product_gas_kg_h,
                    temperature_k=self.input_stream.temperature_k,
                    pressure_pa=self.input_stream.pressure_pa * 0.98,
                    composition={self.gas_type: 1.0},
                    phase='gas'
                )
            else:
                return Stream(0.0)
        elif port_name == 'tail_gas_out':
            if hasattr(self, 'input_stream') and self.input_stream:
                tail_mass = self.feed_gas_kg_h - self.product_gas_kg_h
                return Stream(
                    mass_flow_kg_h=tail_mass,
                    temperature_k=self.input_stream.temperature_k,
                    pressure_pa=101325.0,
                    composition={'H2': 0.5, 'Impurities': 0.5},
                    phase='gas'
                )
            else:
                return Stream(0.0)
        else:
            raise ValueError(f"Unknown output port '{port_name}' on {self.component_id}")

    def receive_input(self, port_name: str, value: Any, resource_type: str) -> float:
        """
        Accept input stream at specified port.

        Buffers incoming mass flow for processing during step().

        Args:
            port_name (str): Target port ('h2_in').
            value (Any): Stream object containing feed gas.
            resource_type (str): Resource classification hint.

        Returns:
            float: Mass flow accepted (kg/h).
        """
        if port_name == 'h2_in':
            if isinstance(value, Stream):
                self.input_stream = value
                self._input_mass_buffer += value.mass_flow_kg_h
                return value.mass_flow_kg_h
        return 0.0

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """
        Define the physical connection ports for this component.

        Returns:
            Dict[str, Dict[str, str]]: Port definitions.
        """
        return {
            'h2_in': {'type': 'input', 'resource_type': 'hydrogen', 'units': 'kg/h'},
            'h2_out': {'type': 'output', 'resource_type': 'hydrogen', 'units': 'kg/h'},
            'tail_gas_out': {'type': 'output', 'resource_type': 'tail_gas', 'units': 'kg/h'}
        }
