"""
Rectifier/Transformer Component.

This module implements an AC/DC power conversion unit for electrolyzer systems.
The rectifier transforms grid AC power to the regulated DC voltage required by
electrolyzer stacks, accounting for conversion efficiency and power factor.

Power Electronics Model:
    - **Efficiency Loss**: P_DC = P_AC × η, where η typically ranges 0.96-0.98
      for modern thyristor or IGBT-based rectifiers.
    - **Power Factor**: Represents the phase displacement between voltage and
      current on the AC side. Values below 1.0 indicate reactive power draw.
    - **Heat Generation**: Dissipated power (P_loss = P_AC - P_DC) manifests
      as waste heat requiring cooling.

Architecture:
    Implements the Component Lifecycle Contract (Layer 1):
    - `initialize()`: Prepares component for simulation.
    - `step()`: Applies efficiency to convert AC input to DC output.
    - `get_state()`: Returns power flows, losses, and load factor.

Process Flow Integration:
    - RT-1: PEM electrolyzer rectifier
    - RT-2: SOEC electrolyzer rectifier
"""

from typing import Dict, Any

from h2_plant.core.component import Component


class Rectifier(Component):
    """
    AC/DC rectifier for electrolyzer power conditioning.

    Converts AC grid power to DC at controlled voltage for electrolyzer
    stacks. Models efficiency losses and tracks load factor for operational
    optimization.

    This component fulfills the Component Lifecycle Contract (Layer 1):
        - `initialize()`: Standard initialization, marks component ready.
        - `step()`: Applies conversion efficiency, calculates DC output.
        - `get_state()`: Returns AC input, DC output, losses, and load factor.

    Attributes:
        rated_power_mw (float): Maximum power rating (MW).
        efficiency (float): Conversion efficiency (0-1).
        power_factor (float): AC input power factor (0-1).
        output_voltage_v (float): DC output voltage (V).

    Example:
        >>> rect = Rectifier(rated_power_mw=10.0, efficiency=0.97)
        >>> rect.initialize(dt=1/60, registry=registry)
        >>> rect.receive_input('ac_in', 5.0, 'electricity')
        >>> rect.step(t=0.0)
        >>> print(f"DC out: {rect.dc_output_power_mw:.2f} MW")
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
        Initialize the rectifier.

        Args:
            component_id (str): Unique identifier for this component.
                Default: 'rectifier'.
            rated_power_mw (float): Maximum DC output power rating in MW.
                Default: 10.0.
            efficiency (float): AC-to-DC conversion efficiency (0-1).
                Typical values: 0.96-0.98 for modern rectifiers. Default: 0.97.
            power_factor (float): Input power factor (0-1). Affects apparent
                power draw from grid. Default: 0.95.
            output_voltage_v (float): Regulated DC output voltage in V.
                Stack-dependent; higher voltages reduce cable losses. Default: 1000.0.
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
        """
        Prepare the component for simulation execution.

        Fulfills the Component Lifecycle Contract initialization phase.

        Args:
            dt (float): Simulation timestep in hours.
            registry (ComponentRegistry): Central registry for component access.
        """
        super().initialize(dt, registry)
        self.initialized = True

    def step(self, t: float) -> None:
        """
        Execute one simulation timestep.

        Converts AC input power to DC output, applying efficiency losses.
        Output is capped at rated power regardless of input magnitude.

        Args:
            t (float): Current simulation time in hours.
        """
        super().step(t)

        if self.ac_input_power_mw > 0:
            # DC output is efficiency-limited and capacity-capped
            self.dc_output_power_mw = min(
                self.ac_input_power_mw * self.efficiency,
                self.rated_power_mw
            )
            # Power dissipated as heat
            self.power_loss_mw = self.ac_input_power_mw - self.dc_output_power_mw
            # Load factor for equipment utilization tracking
            self.load_factor = self.dc_output_power_mw / self.rated_power_mw
        else:
            self.dc_output_power_mw = 0.0
            self.power_loss_mw = 0.0
            self.load_factor = 0.0

    def get_output(self, port_name: str) -> Any:
        """
        Retrieve output from a specified port.

        Args:
            port_name (str): Port identifier ('dc_out', 'electricity_out', or 'heat_out').

        Returns:
            float: DC power (MW) for power ports, or heat dissipation (kW) for heat port.
        """
        if port_name == "dc_out" or port_name == "electricity_out":
            return self.dc_output_power_mw
        elif port_name == "heat_out":
            return self.power_loss_mw * 1000.0
        return 0.0

    def receive_input(self, port_name: str, value: Any, resource_type: str = None) -> float:
        """
        Accept AC power input.

        Input is capped to prevent exceeding rated power after efficiency losses.

        Args:
            port_name (str): Target port ('ac_in' or 'electricity_in').
            value (Any): Power value in MW.
            resource_type (str, optional): Resource classification hint.

        Returns:
            float: Power accepted (MW).
        """
        if port_name == "ac_in" or port_name == "electricity_in":
            if isinstance(value, (int, float)):
                # Cap input to not exceed rated output after efficiency
                self.ac_input_power_mw = min(value, self.rated_power_mw / self.efficiency)
                return self.ac_input_power_mw
        return 0.0

    def extract_output(self, port_name: str, amount: float, resource_type: str = None) -> None:
        """
        Acknowledge extraction of output (no-op for passive component).

        Args:
            port_name (str): Port from which output was extracted.
            amount (float): Amount extracted.
            resource_type (str, optional): Resource classification hint.
        """
        pass

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """
        Define the physical connection ports for this component.

        Returns:
            Dict[str, Dict[str, str]]: Port definitions including AC input,
                DC output, and waste heat output.
        """
        return {
            'ac_in': {'type': 'input', 'resource_type': 'electricity'},
            'electricity_in': {'type': 'input', 'resource_type': 'electricity'},
            'dc_out': {'type': 'output', 'resource_type': 'electricity'},
            'electricity_out': {'type': 'output', 'resource_type': 'electricity'},
            'heat_out': {'type': 'output', 'resource_type': 'heat'}
        }

    def get_state(self) -> Dict[str, Any]:
        """
        Retrieve the component's current operational state.

        Fulfills the Component Lifecycle Contract state access.

        Returns:
            Dict[str, Any]: State dictionary containing:
                - ac_input_power_mw (float): AC power input (MW).
                - dc_output_power_mw (float): DC power output (MW).
                - power_loss_mw (float): Heat dissipation (MW).
                - load_factor (float): Fraction of rated capacity (0-1).
                - efficiency (float): Conversion efficiency.
        """
        return {
            **super().get_state(),
            'component_id': self.component_id,
            'ac_input_power_mw': self.ac_input_power_mw,
            'dc_output_power_mw': self.dc_output_power_mw,
            'power_loss_mw': self.power_loss_mw,
            'load_factor': self.load_factor,
            'efficiency': self.efficiency
        }
