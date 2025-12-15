"""
Heat Exchanger Component.

This module implements a generic heat exchanger for capacity-limited cooling
of fluid streams. Uses enthalpy-based heat transfer calculation with
bisection search for outlet temperature when capacity is exceeded.

Heat Transfer Model:
    The required cooling duty is calculated from enthalpy difference:
    **Q_required = ṁ × (h_in - h_target)**

    If Q_required exceeds max_heat_removal_kw, the actual outlet temperature
    is found via bisection search on the enthalpy-temperature curve.

Operating Modes:
    - **Unconstrained**: Q_required ≤ max_capacity → outlet at target_temp
    - **Capacity-Limited**: Q_required > max_capacity → outlet warmer than target

Architecture:
    Implements the Component Lifecycle Contract (Layer 1):
    - `initialize()`: Standard initialization.
    - `step()`: Calculates heat removal and outlet conditions.
    - `get_state()`: Returns heat removed and outlet stream properties.
"""

from typing import Dict, Any, Optional

from h2_plant.core.component import Component
from h2_plant.core.stream import Stream
from h2_plant.core.constants import ConversionFactors


class HeatExchanger(Component):
    """
    Generic heat exchanger for capacity-limited stream cooling.

    Calculates heat removal using enthalpy difference and applies capacity
    limits. When capacity is exceeded, uses bisection search to find the
    actual achievable outlet temperature.

    This component fulfills the Component Lifecycle Contract (Layer 1):
        - `initialize()`: Standard component initialization.
        - `step()`: Calculates heat removal, determines outlet conditions.
        - `get_state()`: Returns heat removed and outlet stream.

    Attributes:
        max_heat_removal_kw (float): Maximum heat removal capacity (kW).
        target_outlet_temp_c (float): Desired outlet temperature (°C).
        heat_removed_kw (float): Actual heat removed this timestep (kW).

    Example:
        >>> hx = HeatExchanger(
        ...     component_id='HX-1',
        ...     max_heat_removal_kw=50.0,
        ...     target_outlet_temp_c=25.0
        ... )
        >>> hx.initialize(dt=1/60, registry=registry)
        >>> hx.receive_input('water_in', hot_stream, 'water')
        >>> hx.step(t=0.0)
    """

    def __init__(
        self,
        component_id: str,
        max_heat_removal_kw: float,
        target_outlet_temp_c: float = 25.0
    ):
        """
        Initialize the heat exchanger.

        Args:
            component_id (str): Unique identifier for this component.
            max_heat_removal_kw (float): Maximum heat removal capacity in kW.
            target_outlet_temp_c (float): Target outlet temperature in °C.
                Default: 25.0.
        """
        super().__init__()
        self.component_id = component_id
        self.max_heat_removal_kw = max_heat_removal_kw
        self.target_outlet_temp_c = target_outlet_temp_c

        # Stream state
        self.inlet_flow_kg_h = 0.0
        self.input_stream: Optional[Stream] = None
        self.output_stream: Optional[Stream] = None

        # Thermal tracking
        self.heat_removed_kw = 0.0

    def initialize(self, dt: float, registry: 'ComponentRegistry') -> None:
        """
        Prepare the component for simulation execution.

        Fulfills the Component Lifecycle Contract initialization phase.

        Args:
            dt (float): Simulation timestep in hours.
            registry (ComponentRegistry): Central registry for component access.
        """
        super().initialize(dt, registry)

    def step(self, t: float) -> None:
        """
        Execute one simulation timestep.

        Calculates required cooling and applies capacity limits:
        1. Create target stream at desired outlet conditions.
        2. Calculate required heat removal from enthalpy difference.
        3. If capacity-limited, use bisection to find actual outlet temperature.

        Args:
            t (float): Current simulation time in hours.
        """
        super().step(t)

        # Handle legacy input without stream object
        if self.input_stream is None:
            if self.inlet_flow_kg_h > 0:
                self.input_stream = Stream(
                    mass_flow_kg_h=self.inlet_flow_kg_h,
                    temperature_k=353.15,
                    pressure_pa=101325.0,
                    composition={'H2': 1.0}
                )
            else:
                self.output_stream = None
                self.heat_removed_kw = 0.0
                return

        target_temp_k = self.target_outlet_temp_c + 273.15

        # Create target stream for enthalpy calculation
        target_stream = Stream(
            mass_flow_kg_h=self.input_stream.mass_flow_kg_h,
            temperature_k=target_temp_k,
            pressure_pa=self.input_stream.pressure_pa,
            composition=self.input_stream.composition,
            phase=self.input_stream.phase
        )

        h_in = self.input_stream.specific_enthalpy_j_kg
        h_target = target_stream.specific_enthalpy_j_kg

        # Q = ṁ × (h_in - h_target) [J/h → kW]
        # Positive = Cooling (Heat Removal)
        # Negative = Heating (Heat Addition)
        q_required_j_h = self.input_stream.mass_flow_kg_h * (h_in - h_target)
        q_required_kw = q_required_j_h * ConversionFactors.J_TO_KWH

        # Check against capacity (absolute limit)
        if abs(q_required_kw) <= self.max_heat_removal_kw:
            # Unconstrained: achieve target temperature
            self.heat_removed_kw = q_required_kw
            self.output_stream = target_stream
        else:
            # Capacity-limited: apply max duty with correct sign
            sign = 1.0 if q_required_kw > 0 else -1.0
            self.heat_removed_kw = sign * self.max_heat_removal_kw

            # h_out = h_in - Q_actual / ṁ
            q_actual_j_kg = (self.heat_removed_kw / ConversionFactors.J_TO_KWH) / \
                             self.input_stream.mass_flow_kg_h
            h_out = h_in - q_actual_j_kg

            # Bisection search for T_out
            # Range depends on heating or cooling
            t_low = min(target_temp_k, self.input_stream.temperature_k)
            t_high = max(target_temp_k, self.input_stream.temperature_k)
            
            # Widen search space slightly to ensure bracketing if properties non-linear
            if sign < 0: # Heating: T_out might be slightly less than T_target if limited
                 t_high = target_temp_k
                 t_low = self.input_stream.temperature_k
            else: # Cooling
                 t_high = self.input_stream.temperature_k
                 t_low = target_temp_k

            found_t = t_low # Fallback
            
            for _ in range(20):
                t_mid = (t_low + t_high) / 2
                s_mid = Stream(1.0, t_mid, self.input_stream.pressure_pa,
                              self.input_stream.composition)
                h_mid = s_mid.specific_enthalpy_j_kg

                # We want h_mid == h_out
                # h(T) is monotonically increasing
                if h_mid > h_out:
                    t_high = t_mid
                else:
                    t_low = t_mid
            
            found_t = (t_low + t_high) / 2

            self.output_stream = Stream(
                mass_flow_kg_h=self.input_stream.mass_flow_kg_h,
                temperature_k=found_t,
                pressure_pa=self.input_stream.pressure_pa,
                composition=self.input_stream.composition,
                phase=self.input_stream.phase
            )

    def get_state(self) -> Dict[str, Any]:
        """
        Retrieve the component's current operational state.

        Fulfills the Component Lifecycle Contract state access.

        Returns:
            Dict[str, Any]: State dictionary containing heat removed and
                outlet stream properties.
        """
        state = {
            **super().get_state(),
            'component_id': self.component_id,
            'heat_removed_kw': self.heat_removed_kw
        }

        if self.output_stream:
            state['streams'] = {
                'out': {
                    'mass_flow': self.output_stream.mass_flow_kg_h,
                    'temperature': self.output_stream.temperature_k,
                    'enthalpy': self.output_stream.specific_enthalpy_j_kg
                }
            }
        return state

    def get_output(self, port_name: str) -> Any:
        """
        Retrieve output from specified port.

        Supports generic port naming for multi-purpose heat exchanger use.

        Args:
            port_name (str): Port identifier ('water_out', 'h2_out', 'out',
                'cooled_gas_out', 'syngas_out', or 'heat_out').

        Returns:
            Stream or float: Output stream or heat removed value.

        Raises:
            ValueError: If port_name is not recognized.
        """
        if port_name in ['water_out', 'h2_out', 'out', 'cooled_gas_out', 'syngas_out']:
            if self.output_stream:
                return self.output_stream
            else:
                return Stream(0.0)
        elif port_name == 'heat_out':
            return self.heat_removed_kw
        else:
            raise ValueError(f"Unknown output port '{port_name}' on {self.component_id}")

    def receive_input(self, port_name: str, value: Any, resource_type: str) -> float:
        """
        Accept input at specified port.

        Supports generic port naming for multi-purpose heat exchanger use.

        Args:
            port_name (str): Target port ('water_in', 'h2_in', or 'in').
            value (Any): Input stream.
            resource_type (str): Resource classification hint.

        Returns:
            float: Mass flow accepted (kg/h).
        """
        if port_name in ['water_in', 'h2_in', 'in']:
            if isinstance(value, Stream):
                self.input_stream = value
                return value.mass_flow_kg_h
        return 0.0

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """
        Define the physical connection ports for this component.

        Returns:
            Dict[str, Dict[str, str]]: Port definitions.
        """
        return {
            'water_in': {'type': 'input', 'resource_type': 'water', 'units': 'kg/h'},
            'h2_in': {'type': 'input', 'resource_type': 'hydrogen', 'units': 'kg/h'},
            'water_out': {'type': 'output', 'resource_type': 'water', 'units': 'kg/h'},
            'h2_out': {'type': 'output', 'resource_type': 'hydrogen', 'units': 'kg/h'},
            'heat_out': {'type': 'output', 'resource_type': 'heat', 'units': 'kW'}
        }
