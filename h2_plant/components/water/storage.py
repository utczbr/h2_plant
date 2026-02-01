"""
Ultrapure Water Storage Tank Component.

This module implements a buffer storage tank for ultrapure water used in
electrolysis systems. Provides capacity management with fill/withdraw
operations and flow tracking for process integration.

Water Quality:
    Electrolysis requires high-purity deionized water to prevent:
    - Electrode contamination
    - Membrane degradation
    - Gas impurity buildup

    Typical specifications: <0.1 μS/cm conductivity, <1 ppb TOC.

Architecture:
    Implements the Component Lifecycle Contract (Layer 1):
    - `initialize()`: Validates capacity, prepares component.
    - `step()`: Placeholder (flow managed by coordinator).
    - `get_state()`: Returns mass, fill ratio, and flow metadata.

    Dual-input design supports:
    - Primary: From water treatment system
    - Secondary: Condensate return from SOEC
"""

from typing import Dict, Any

from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.stream import Stream


class UltrapureWaterStorageTank(Component):
    """
    Buffer storage tank for ultrapure water with dual inputs.

    Provides capacity-limited storage with fill/withdraw operations
    and detailed flow tracking for process integration.

    This component fulfills the Component Lifecycle Contract (Layer 1):
        - `initialize()`: Validates capacity configuration.
        - `step()`: Placeholder (flow managed externally).
        - `get_state()`: Returns mass, fill ratio, and flow metadata.

    Attributes:
        capacity_l (float): Tank capacity in liters.
        capacity_kg (float): Tank capacity in kg (water ρ ≈ 1 kg/L).
        current_mass_kg (float): Current water inventory (kg).
        temperature_c (float): Water temperature (°C).

    Example:
        >>> tank = UltrapureWaterStorageTank(capacity_l=5000.0, initial_fill_ratio=0.5)
        >>> tank.initialize(dt=1/60, registry=registry)
        >>> added = tank.fill(100.0)  # Add 100 kg
        >>> withdrawn = tank.withdraw(50.0)  # Remove 50 kg
    """

    def __init__(self, capacity_l: float = 5000.0, initial_fill_ratio: float = 0.5):
        """
        Initialize the ultrapure water storage tank.

        Args:
            capacity_l (float): Tank capacity in liters. Default: 5000.0.
            initial_fill_ratio (float): Initial fill level as fraction (0-1).
                Default: 0.5 (50% full).
        """
        super().__init__()
        self.capacity_l = capacity_l
        self.capacity_kg = capacity_l  # Water density ≈ 1 kg/L
        self.current_mass_kg = self.capacity_kg * initial_fill_ratio
        self.temperature_c = 20.0
        self.pressure_bar = 1.0

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """
        Prepare the component for simulation execution.

        Fulfills the Component Lifecycle Contract initialization phase.
        Validates that tank capacity is positive.

        Args:
            dt (float): Simulation timestep in hours.
            registry (ComponentRegistry): Central registry for component access.

        Raises:
            ValueError: If capacity_l is not positive.
        """
        super().initialize(dt, registry)
        if self.capacity_l <= 0:
            raise ValueError("Tank capacity must be positive")

    def fill(self, mass_kg: float) -> float:
        """
        Add water to the tank.

        Args:
            mass_kg (float): Mass to add in kg.

        Returns:
            float: Actual mass added (may be less if capacity exceeded).
        """
        available = max(self.capacity_kg - self.current_mass_kg, 0.0)
        actual = min(mass_kg, available)
        self.current_mass_kg += actual
        return actual

    def withdraw(self, mass_kg: float) -> float:
        """
        Remove water from the tank.

        Args:
            mass_kg (float): Mass to remove in kg.

        Returns:
            float: Actual mass removed (may be less if insufficient stored).
        """
        actual = min(mass_kg, self.current_mass_kg)
        self.current_mass_kg -= actual
        return actual

    def step(self, t: float) -> None:
        """
        Execute one simulation timestep.

        Flow management is handled by external coordinator or
        connected components via fill/withdraw methods.

        Args:
            t (float): Current simulation time in hours.
        """
        super().step(t)

    def get_state(self) -> Dict[str, Any]:
        """
        Retrieve the component's current operational state.

        Fulfills the Component Lifecycle Contract state access.

        Returns:
            Dict[str, Any]: State dictionary containing:
                - current_mass_kg (float): Current inventory (kg).
                - capacity_kg (float): Total capacity (kg).
                - fill_ratio (float): Current fill level (0-1).
                - temperature_c (float): Water temperature (°C).
                - pressure_bar (float): Tank pressure (bar).
                - flows (dict): Input/output flow metadata.
        """
        fill_ratio = self.current_mass_kg / self.capacity_kg if self.capacity_kg > 0 else 0.0
        return {
            **super().get_state(),
            "current_mass_kg": float(self.current_mass_kg),
            "capacity_kg": float(self.capacity_kg),
            "fill_ratio": float(fill_ratio),
            "temperature_c": float(self.temperature_c),
            "pressure_bar": float(self.pressure_bar),
            "flows": {
                "inputs": {
                    "from_treatment": {
                        "value": 0.0,
                        "unit": "kg/h",
                        "source": "water_treatment",
                        "flowtype": "WATER_MASS"
                    },
                    "from_soec": {
                        "value": 0.0,
                        "unit": "kg/h",
                        "source": "soec_electrolyzer",
                        "flowtype": "WATER_MASS"
                    }
                },
                "outputs": {
                    "to_pump_a": {
                        "value": 0.0,
                        "unit": "kg/h",
                        "destination": "pump_a",
                        "flowtype": "WATER_MASS"
                    },
                    "to_pump_b": {
                        "value": 0.0,
                        "unit": "kg/h",
                        "destination": "pump_b",
                        "flowtype": "WATER_MASS"
                    }
                }
            }
        }

    def get_output(self, port_name: str) -> Any:
        """
        Retrieve output stream from specified port.

        Args:
            port_name (str): Port identifier ('water_out').

        Returns:
            Stream: Water stream at tank conditions with available mass.

        Raises:
            ValueError: If port_name is not recognized.
        """
        if port_name == 'water_out':
            return Stream(
                mass_flow_kg_h=self.current_mass_kg,
                temperature_k=self.temperature_c + 273.15,
                pressure_pa=self.pressure_bar * 1e5,
                composition={'H2O': 1.0},
                phase='liquid'
            )
        else:
            raise ValueError(f"Unknown output port '{port_name}' on {self.component_id}")

    def receive_input(self, port_name: str, value: Any, resource_type: str) -> float:
        """
        Accept input at specified port.

        Args:
            port_name (str): Target port ('water_in').
            value (Any): Input stream.
            resource_type (str): Resource classification hint.

        Returns:
            float: Mass added to tank (kg).
        """
        if port_name == 'water_in':
            if isinstance(value, Stream):
                added = self.fill(value.mass_flow_kg_h)
                return added
        return 0.0

    def extract_output(self, port_name: str, amount: float, resource_type: str) -> None:
        """
        Acknowledge extraction and deduct from inventory.

        Args:
            port_name (str): Output port.
            amount (float): Mass extracted (kg).
            resource_type (str): Resource classification hint.
        """
        if port_name == 'water_out':
            self.withdraw(amount)

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """
        Define the physical connection ports for this component.

        Returns:
            Dict[str, Dict[str, str]]: Port definitions.
        """
        return {
            'water_in': {'type': 'input', 'resource_type': 'water', 'units': 'kg/h'},
            'water_out': {'type': 'output', 'resource_type': 'water', 'units': 'kg/h'}
        }
