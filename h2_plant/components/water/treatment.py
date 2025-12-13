"""
Water Treatment Block Component.

This module implements a water treatment system for producing ultrapure
water suitable for electrolysis. The treatment process includes
filtration, reverse osmosis, and deionization stages.

Water Quality Requirements:
    PEM and SOEC electrolysis require ultrapure water to prevent:
    - Ion contamination of electrodes and membranes
    - Scaling and fouling of heat exchangers
    - Trace impurities in product hydrogen

    Treatment produces water with:
    - Conductivity: <0.1 μS/cm
    - TOC (Total Organic Carbon): <1 ppb
    - Silica: <1 ppb

Architecture:
    Implements the Component Lifecycle Contract (Layer 1):
    - `initialize()`: Acquires LUTManager reference if available.
    - `step()`: Calculates output flow and test sample fraction.
    - `get_state()`: Returns flows, temperature, and power consumption.

Process Integration:
    - Input: Pretreated municipal/industrial water
    - Output: Ultrapure water to storage tank
    - Test sample: 1% diverted for quality monitoring
"""

from typing import Dict, Any, Optional

from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.stream import Stream
from h2_plant.optimization.lut_manager import LUTManager


class WaterTreatmentBlock(Component):
    """
    Water treatment system producing ultrapure water for electrolysis.

    Produces ultrapure water from pretreated feed water, consuming
    electrical power for pumps, RO membranes, and UV sterilization.

    This component fulfills the Component Lifecycle Contract (Layer 1):
        - `initialize()`: Acquires LUTManager reference for property lookup.
        - `step()`: Calculates ultrapure water output and test sample.
        - `get_state()`: Returns flows, power, and flow metadata.

    Attributes:
        max_flow_m3h (float): Maximum treatment capacity (m³/h).
        power_consumption_kw (float): Electrical power demand (kW).
        output_flow_kgh (float): Current output rate (kg/h).
        test_flow_kgh (float): Test sample rate (kg/h).

    Example:
        >>> treatment = WaterTreatmentBlock(max_flow_m3h=5.0, power_consumption_kw=10.0)
        >>> treatment.initialize(dt=1/60, registry=registry)
        >>> treatment.step(t=0.0)
        >>> ultrapure = treatment.get_output('ultrapure_water_out')
    """

    def __init__(self, max_flow_m3h: float, power_consumption_kw: float):
        """
        Initialize the water treatment block.

        Args:
            max_flow_m3h (float): Maximum treatment capacity in m³/h.
            power_consumption_kw (float): Electrical power consumption in kW.
        """
        super().__init__()
        self.max_flow_m3h = max_flow_m3h
        self.power_consumption_kw = power_consumption_kw

        # Output state
        self.output_flow_kgh = 0.0
        self.output_temp_c = 20.0
        self.output_pressure_bar = 1.0
        self.test_flow_kgh = 0.0

        # LUT Manager for property lookup
        self.lut: Optional[LUTManager] = None

        # Input tracking
        self._input_mass_kg = 0.0

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """
        Prepare the component for simulation execution.

        Fulfills the Component Lifecycle Contract initialization phase.
        Acquires LUTManager reference for thermodynamic property lookup.

        Args:
            dt (float): Simulation timestep in hours.
            registry (ComponentRegistry): Central registry for component access.
        """
        super().initialize(dt, registry)
        if registry.has("lut_manager"):
            self.lut = registry.get("lut_manager")

    def step(self, t: float) -> None:
        """
        Execute one simulation timestep.

        Calculates ultrapure water output based on maximum capacity.
        Diverts 1% of output as test sample for quality monitoring.

        Args:
            t (float): Current simulation time in hours.
        """
        super().step(t)

        # Output flow at maximum capacity (water density ≈ 1000 kg/m³)
        self.output_flow_kgh = self.max_flow_m3h * 1000

        # Test sample: 1% for quality monitoring
        self.test_flow_kgh = self.output_flow_kgh * 0.01

    def get_state(self) -> Dict[str, Any]:
        """
        Retrieve the component's current operational state.

        Fulfills the Component Lifecycle Contract state access.

        Returns:
            Dict[str, Any]: State dictionary containing:
                - output_flow_kgh (float): Ultrapure water rate (kg/h).
                - output_temp_c (float): Output temperature (°C).
                - output_pressure_bar (float): Output pressure (bar).
                - power_consumption_kw (float): Power demand (kW).
                - test_flow_kgh (float): Test sample rate (kg/h).
                - flows (dict): Input/output flow metadata.
        """
        return {
            **super().get_state(),
            "output_flow_kgh": float(self.output_flow_kgh),
            "output_temp_c": float(self.output_temp_c),
            "output_pressure_bar": float(self.output_pressure_bar),
            "power_consumption_kw": float(self.power_consumption_kw),
            "test_flow_kgh": float(self.test_flow_kgh),
            "flows": {
                "inputs": {
                    "tested_water": {
                        "value": self.output_flow_kgh / 1000.0,
                        "unit": "m3/h",
                        "source": "water_quality_test",
                        "flowtype": "WATER_MASS"
                    },
                    "electricity": {
                        "value": self.power_consumption_kw,
                        "unit": "kW",
                        "source": "grid_or_battery",
                        "flowtype": "ELECTRICAL_ENERGY"
                    }
                },
                "outputs": {
                    "ultrapure_water": {
                        "value": self.output_flow_kgh - self.test_flow_kgh,
                        "unit": "kg/h",
                        "destination": "ultrapure_storage_tank",
                        "flowtype": "WATER_MASS"
                    },
                    "test_sample": {
                        "value": self.test_flow_kgh,
                        "unit": "kg/h",
                        "destination": "test_lab",
                        "flowtype": "WATER_MASS"
                    }
                }
            }
        }

    def get_output(self, port_name: str) -> Any:
        """
        Retrieve output stream from specified port.

        Args:
            port_name (str): Port identifier ('ultrapure_water_out' or 'test_sample_out').

        Returns:
            Stream: Output water stream.

        Raises:
            ValueError: If port_name is not recognized.
        """
        if port_name == 'ultrapure_water_out':
            return Stream(
                mass_flow_kg_h=self.output_flow_kgh - self.test_flow_kgh,
                temperature_k=273.15 + self.output_temp_c,
                pressure_pa=self.output_pressure_bar * 1e5,
                composition={'H2O': 1.0},
                phase='liquid'
            )
        elif port_name == 'test_sample_out':
            return Stream(
                mass_flow_kg_h=self.test_flow_kgh,
                temperature_k=273.15 + self.output_temp_c,
                pressure_pa=self.output_pressure_bar * 1e5,
                composition={'H2O': 1.0},
                phase='liquid'
            )
        else:
            raise ValueError(f"Unknown output port '{port_name}' on {self.component_id}")

    def receive_input(self, port_name: str, value: Any, resource_type: str) -> float:
        """
        Accept input at specified port.

        Args:
            port_name (str): Target port ('water_in' or 'electricity_in').
            value (Any): Input stream or power value.
            resource_type (str): Resource classification hint.

        Returns:
            float: Amount accepted.
        """
        if port_name == 'water_in':
            if isinstance(value, Stream):
                available_mass = value.mass_flow_kg_h * self.dt
                max_capacity = self.max_flow_m3h * 1000.0 * self.dt
                accepted_mass = min(available_mass, max_capacity)
                self._input_mass_kg = accepted_mass
                return accepted_mass

        elif port_name == 'electricity_in':
            if isinstance(value, (int, float)):
                return value

        return 0.0

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """
        Define the physical connection ports for this component.

        Returns:
            Dict[str, Dict[str, str]]: Port definitions.
        """
        return {
            'water_in': {'type': 'input', 'resource_type': 'water', 'units': 'kg/h'},
            'electricity_in': {'type': 'input', 'resource_type': 'electricity', 'units': 'kW'},
            'ultrapure_water_out': {'type': 'output', 'resource_type': 'water', 'units': 'kg/h'},
            'test_sample_out': {'type': 'output', 'resource_type': 'water', 'units': 'kg/h'}
        }
