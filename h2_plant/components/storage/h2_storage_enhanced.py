"""
Enhanced Hydrogen Storage Tank with Pressure Dynamics.

This module implements a hydrogen storage tank with rigorous pressure-volume-
temperature (PVT) dynamics. Unlike simple mass-balance tanks, this component
calculates pressure evolution from the ideal gas equation of state.

Pressure Dynamics:
    The gas accumulator model tracks mass inventory and computes pressure:

    **dP/dt = (RT/V) × (ṁ_in - ṁ_out)**

    This approach captures transient pressure behavior during filling and
    emptying, enabling realistic compressor and valve interactions.

Architecture:
    Implements the Component Lifecycle Contract (Layer 1):
    - `initialize()`: Standard initialization.
    - `step()`: Advances accumulator dynamics by one timestep.
    - `get_state()`: Returns pressure, mass, and fill fraction.

Unified Storage Interface:
    Provides `get_inventory_kg()` and `withdraw_kg()` methods for
    compatibility with the Orchestrator's storage management.
"""

from typing import Dict, Any, Optional

from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.stream import Stream
from h2_plant.models.flow_dynamics import GasAccumulatorDynamics


class H2StorageTankEnhanced(Component):
    """
    Hydrogen storage tank with gas accumulator pressure dynamics.

    Models pressure evolution from mass balance using the ideal gas equation
    of state. Suitable for high-fidelity simulations where pressure transients
    affect compressor and valve operation.

    This component fulfills the Component Lifecycle Contract (Layer 1):
        - `initialize()`: Standard component initialization.
        - `step()`: Advances GasAccumulatorDynamics, updates pressure/mass.
        - `get_state()`: Returns pressure, mass, fill fraction, and flows.

    The pressure model:
    - P = mRT/V (ideal gas at constant temperature)
    - dP/dt = (RT/V) × (dm/dt)
    - Fill fraction = m_current / m_max, where m_max is mass at max pressure

    Attributes:
        volume_m3 (float): Tank internal volume (m³).
        max_pressure_bar (float): Maximum allowable pressure (bar).
        pressure_bar (float): Current pressure (bar).
        mass_kg (float): Current hydrogen inventory (kg).
        fill_fraction (float): Fraction of maximum capacity (0-1).

    Example:
        >>> tank = H2StorageTankEnhanced(
        ...     tank_id='T-101',
        ...     volume_m3=10.0,
        ...     max_pressure_bar=350.0
        ... )
        >>> tank.initialize(dt=1/60, registry=registry)
        >>> tank.receive_input('h2_in', h2_stream, 'hydrogen')
        >>> tank.step(t=0.0)
    """

    def __init__(
        self,
        tank_id: str,
        volume_m3: float = 10.0,
        initial_pressure_bar: float = 40.0,
        max_pressure_bar: float = 350.0
    ):
        """
        Initialize the enhanced hydrogen storage tank.

        Args:
            tank_id (str): Unique identifier for this tank.
            volume_m3 (float): Internal volume in m³. Default: 10.0.
            initial_pressure_bar (float): Initial pressure in bar. Default: 40.0.
            max_pressure_bar (float): Maximum allowable pressure in bar.
                Used for fill fraction calculation. Default: 350.0.
        """
        super().__init__()
        self.tank_id = tank_id
        self.volume_m3 = volume_m3
        self.max_pressure_bar = max_pressure_bar

        # Gas accumulator dynamics model
        self.accumulator = GasAccumulatorDynamics(
            V_tank_m3=volume_m3,
            initial_pressure_pa=initial_pressure_bar * 1e5,
            T_tank_k=298.15
        )

        # Flow rate tracking (reset each timestep)
        self.m_dot_in_kg_s = 0.0
        self.m_dot_out_kg_s = 0.0

        # State variables
        self.pressure_bar = initial_pressure_bar
        self.mass_kg = self.accumulator.M_kg
        self.fill_fraction = 0.0

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
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

        Advances the gas accumulator dynamics model to compute new pressure
        and mass values based on inlet/outlet flow rates.

        Args:
            t (float): Current simulation time in hours.
        """
        super().step(t)
        dt_seconds = self.dt * 3600.0

        # Advance accumulator with current mass flow rates
        P_new_pa = self.accumulator.step(
            dt_s=dt_seconds,
            m_dot_in_kg_s=self.m_dot_in_kg_s,
            m_dot_out_kg_s=self.m_dot_out_kg_s
        )

        # Update state from accumulator
        self.pressure_bar = P_new_pa / 1e5
        self.mass_kg = self.accumulator.M_kg

        # Calculate fill fraction (mass at max pressure = capacity)
        max_mass = (self.max_pressure_bar * 1e5 * self.volume_m3) / (
            self.accumulator.R * self.accumulator.T
        )
        self.fill_fraction = self.mass_kg / max_mass if max_mass > 0 else 0.0

        # Reset flow rates for next timestep
        self.m_dot_in_kg_s = 0.0
        self.m_dot_out_kg_s = 0.0

    def receive_input(self, port_name: str, value: Any, resource_type: str) -> float:
        """
        Accept hydrogen input from upstream component.

        Args:
            port_name (str): Target port ('h2_in').
            value (Any): Stream object containing hydrogen.
            resource_type (str): Resource classification hint.

        Returns:
            float: Mass flow accepted (kg/h).
        """
        if port_name == 'h2_in' and isinstance(value, Stream):
            flow_kg_s = value.mass_flow_kg_h / 3600.0
            self.m_dot_in_kg_s += flow_kg_s
            return value.mass_flow_kg_h
        return 0.0

    def extract_output(self, port_name: str, amount: float, resource_type: str) -> None:
        """
        Register extraction request for outflow tracking.

        Args:
            port_name (str): Source port ('h2_out').
            amount (float): Requested extraction rate (kg/h).
            resource_type (str): Resource classification hint.
        """
        if port_name == 'h2_out':
            flow_kg_s = amount / 3600.0
            self.m_dot_out_kg_s += flow_kg_s

    def get_output(self, port_name: str) -> Any:
        """
        Retrieve output stream from specified port.

        Returns a stream with current tank pressure. Flow rate is set by
        downstream extract_output() calls.

        Args:
            port_name (str): Port identifier ('h2_out').

        Returns:
            Stream: Output stream at tank pressure, or None.
        """
        if port_name == 'h2_out':
            return Stream(
                mass_flow_kg_h=0.0,
                temperature_k=self.accumulator.T,
                pressure_pa=self.accumulator.P,
                composition={'H2': 1.0},
                phase='gas'
            )
        return None

    def get_state(self) -> Dict[str, Any]:
        """
        Retrieve the component's current operational state.

        Fulfills the Component Lifecycle Contract state access.

        Returns:
            Dict[str, Any]: State dictionary containing:
                - pressure_bar (float): Current pressure (bar).
                - mass_kg (float): Current inventory (kg).
                - fill_fraction (float): Fraction of capacity (0-1).
                - flow_in_kg_s (float): Current inflow rate (kg/s).
                - flow_out_kg_s (float): Current outflow rate (kg/s).
        """
        return {
            **super().get_state(),
            'pressure_bar': self.pressure_bar,
            'mass_kg': self.mass_kg,
            'fill_fraction': self.fill_fraction,
            'flow_in_kg_s': self.m_dot_in_kg_s,
            'flow_out_kg_s': self.m_dot_out_kg_s
        }

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """
        Define the physical connection ports for this component.

        Returns:
            Dict[str, Dict[str, str]]: Port definitions.
        """
        return {
            'h2_in': {'type': 'input', 'resource_type': 'hydrogen', 'units': 'kg/h'},
            'h2_out': {'type': 'output', 'resource_type': 'hydrogen', 'units': 'kg/h'}
        }

    # --- Unified Storage Interface ---

    def get_inventory_kg(self) -> float:
        """
        Return total stored hydrogen mass (Unified Storage Interface).

        Returns:
            float: Current inventory in kg.
        """
        return self.mass_kg

    def withdraw_kg(self, amount: float) -> float:
        """
        Withdraw hydrogen directly from storage (Unified Storage Interface).

        Bypasses rate-limited dynamics for Orchestrator push-sweep compatibility.
        Updates accumulator state to maintain pressure consistency.

        Args:
            amount (float): Requested withdrawal mass in kg.

        Returns:
            float: Actual mass withdrawn in kg.
        """
        available = self.mass_kg
        actual = min(amount, available)
        self.mass_kg -= actual

        # Synchronize accumulator state (isochoric process: P = mRT/V)
        self.accumulator.M_kg = self.mass_kg
        self.accumulator.P = (self.accumulator.M_kg * self.accumulator.R *
                              self.accumulator.T) / self.accumulator.V
        self.pressure_bar = self.accumulator.P / 1e5

        return actual
