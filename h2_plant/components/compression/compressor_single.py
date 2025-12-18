"""
Single-Stage Hydrogen Compressor Component.

Models a single-stage adiabatic compressor for hydrogen applications. This
simplified model computes compression work using isentropic efficiency without
intercooling, suitable for low pressure ratios or when downstream cooling is
handled separately.

Thermodynamic Model
-------------------
The compression follows an adiabatic path with isentropic efficiency:

    W_actual = (H_2s - H_1) / η_is

where H_2s is the isentropic outlet enthalpy at constant entropy. The actual
outlet temperature rises according to:

    T_2 = T_1 + (T_2s - T_1) / η_is

No intercooling is modeled; the outlet stream retains the elevated discharge
temperature.

Drive Train Model
-----------------
Electrical power consumption includes drive train losses:

    W_electrical = W_shaft / (η_mechanical × η_electrical)

Component Lifecycle Contract (Layer 1)
--------------------------------------
- ``initialize()``: Validates configuration parameters.
- ``step()``: Computes compression work for mass transferred during timestep.
- ``get_state()``: Exposes energy consumption and outlet conditions.

References
----------
- GPSA Engineering Data Book, 14th Ed., Section 13 (Compressors).
"""

import numpy as np
from typing import Dict, Any, Optional
import logging

from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.component_ids import ComponentID
from h2_plant.core.enums import CompressorMode
from h2_plant.core.stream import Stream

try:
    import CoolProp.CoolProp as CP
    COOLPROP_AVAILABLE = True
except ImportError:
    CP = None
    COOLPROP_AVAILABLE = False

logger = logging.getLogger(__name__)


class CompressorSingle(Component):
    """
    Single-stage adiabatic compressor for hydrogen.

    Implements single-stage compression without intercooling. Suitable for
    low-to-moderate pressure ratios where discharge temperature remains
    acceptable, or when downstream equipment provides cooling.

    The compression model computes:

    1. Isentropic outlet enthalpy: H_2s = H(P_out, S_in)
    2. Actual shaft work: W = (H_2s - H_1) / η_is
    3. Electrical work: W_el = W_shaft / (η_m × η_el)
    4. Outlet temperature from actual enthalpy: T_out = f(H_actual, P_out)

    Component Lifecycle Contract (Layer 1):
        - ``initialize()``: Validates configuration.
        - ``step()``: Computes compression work for current mass transfer.
        - ``get_state()``: Returns energy consumption and outlet conditions.

    Attributes:
        max_flow_kg_h: Maximum mass flow capacity in kg/h.
        inlet_pressure_bar: Suction pressure in bar.
        outlet_pressure_bar: Discharge pressure in bar.
        isentropic_efficiency: Thermodynamic efficiency (0-1).
        mechanical_efficiency: Drive train mechanical efficiency (0-1).
        electrical_efficiency: Motor electrical efficiency (0-1).

    Example:
        >>> comp = CompressorSingle(
        ...     max_flow_kg_h=100.0,
        ...     inlet_pressure_bar=1.0,
        ...     outlet_pressure_bar=2.0
        ... )
        >>> comp.initialize(dt=1/60, registry=registry)
        >>> comp.transfer_mass_kg = 10.0
        >>> comp.step(t=0.0)
        >>> print(f"Outlet T: {comp.outlet_temperature_c:.1f}°C")
    """

    def __init__(
        self,
        max_flow_kg_h: float,
        inlet_pressure_bar: float,
        outlet_pressure_bar: float,
        inlet_temperature_c: float = 25.0,
        isentropic_efficiency: float = 0.65,
        mechanical_efficiency: float = 0.96,
        electrical_efficiency: float = 0.93
    ):
        """
        Configure the single-stage compressor.

        Args:
            max_flow_kg_h: Maximum mass flow rate in kg/h.
            inlet_pressure_bar: Suction pressure in bar.
            outlet_pressure_bar: Target discharge pressure in bar.
            inlet_temperature_c: Suction temperature in °C. Defaults to 25°C.
            isentropic_efficiency: Stage isentropic efficiency (0-1).
                Defaults to 0.65.
            mechanical_efficiency: Drive train mechanical efficiency (0-1).
                Defaults to 0.96.
            electrical_efficiency: Motor electrical efficiency (0-1).
                Defaults to 0.93.
        """
        super().__init__()

        self.max_flow_kg_h = max_flow_kg_h
        self.inlet_pressure_bar = inlet_pressure_bar
        self.outlet_pressure_bar = outlet_pressure_bar

        self.inlet_temperature_c = inlet_temperature_c
        self.inlet_temperature_k = inlet_temperature_c + 273.15
        self.isentropic_efficiency = isentropic_efficiency
        self.mechanical_efficiency = mechanical_efficiency
        self.electrical_efficiency = electrical_efficiency

        self.BAR_TO_PA = 1e5
        self.J_TO_KWH = 2.7778e-7

        self.transfer_mass_kg = 0.0

        self.actual_mass_transferred_kg = 0.0
        self.compression_work_kwh = 0.0
        self.energy_consumed_kwh = 0.0
        self.specific_energy_kwh_kg = 0.0

        self.outlet_temperature_c = inlet_temperature_c
        self.outlet_temperature_k = self.inlet_temperature_k
        self.outlet_temperature_isentropic_c = inlet_temperature_c

        self.power_kw = 0.0

        self.mode = CompressorMode.IDLE
        self.cumulative_energy_kwh = 0.0
        self.cumulative_mass_kg = 0.0

        self._last_step_time = -1.0
        self.timestep_power_kw = 0.0
        self.timestep_energy_kwh = 0.0
        
        # Store inlet stream for composition propagation
        self._inlet_stream: Optional[Stream] = None

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """
        Prepare the compressor for simulation execution.

        Fulfills the Component Lifecycle Contract (Layer 1) initialization.

        Args:
            dt: Simulation timestep in hours.
            registry: Central component registry.
        """
        super().initialize(dt, registry)

        pressure_ratio = self.outlet_pressure_bar / self.inlet_pressure_bar
        logger.info(
            f"CompressorSingle '{self.component_id}': "
            f"{self.inlet_pressure_bar:.1f} → {self.outlet_pressure_bar:.1f} bar "
            f"(ratio={pressure_ratio:.2f}), η_is={self.isentropic_efficiency:.2f}"
        )

    def step(self, t: float) -> None:
        """
        Execute one simulation timestep.

        Fulfills the Component Lifecycle Contract (Layer 1) step phase by
        computing single-stage adiabatic compression work.

        Args:
            t: Current simulation time in hours.
        """
        super().step(t)

        if t != self._last_step_time:
            self.timestep_power_kw = 0.0
            self.timestep_energy_kwh = 0.0
            self._last_step_time = t

        self.actual_mass_transferred_kg = 0.0
        self.energy_consumed_kwh = 0.0
        self.compression_work_kwh = 0.0

        if self.transfer_mass_kg > 0:
            self.mode = CompressorMode.LP_TO_HP

            max_transfer = self.max_flow_kg_h * self.dt
            self.actual_mass_transferred_kg = min(self.transfer_mass_kg, max_transfer)

            if self.outlet_pressure_bar <= self.inlet_pressure_bar:
                self._calculate_trivial_pass_through()
            else:
                self._calculate_compression()

            self.energy_consumed_kwh = self.compression_work_kwh

            self.cumulative_energy_kwh += self.energy_consumed_kwh
            self.cumulative_mass_kg += self.actual_mass_transferred_kg

            batch_power_kw = 0.0
            if self.dt > 0:
                batch_power_kw = self.energy_consumed_kwh / self.dt

            self.timestep_power_kw += batch_power_kw
            self.timestep_energy_kwh += self.energy_consumed_kwh

            self.power_kw = self.timestep_power_kw

            self.transfer_mass_kg = 0.0
        else:
            self.mode = CompressorMode.IDLE
            self.power_kw = 0.0

    def _calculate_trivial_pass_through(self) -> None:
        """Handle case where no compression is required."""
        self.compression_work_kwh = 0.0
        self.specific_energy_kwh_kg = 0.0
        self.outlet_temperature_k = self.inlet_temperature_k
        self.outlet_temperature_c = self.inlet_temperature_c
        self.outlet_temperature_isentropic_c = self.inlet_temperature_c

    def _calculate_compression(self) -> None:
        """
        Calculate single-stage adiabatic compression.

        Uses real-gas properties from CoolProp when available, otherwise
        falls back to ideal gas relations.
        """
        p_in_pa = self.inlet_pressure_bar * self.BAR_TO_PA
        p_out_pa = self.outlet_pressure_bar * self.BAR_TO_PA

        if COOLPROP_AVAILABLE:
            self._calculate_compression_realgas(p_in_pa, p_out_pa)
        else:
            self._calculate_compression_idealgas(p_in_pa, p_out_pa)

    def _calculate_compression_realgas(self, p_in_pa: float, p_out_pa: float) -> None:
        """
        Calculate compression using CoolProp real-gas properties.

        Implements isentropic efficiency model:
        1. Get inlet entropy S_1
        2. Find isentropic outlet: H_2s = H(P_out, S_1), T_2s = T(P_out, S_1)
        3. Actual work: W = (H_2s - H_1) / η_is
        4. Actual outlet: H_2 = H_1 + W, T_2 = T(P_out, H_2)
        """
        fluid = 'H2'

        h1 = CP.PropsSI('H', 'T', self.inlet_temperature_k, 'P', p_in_pa, fluid)
        s1 = CP.PropsSI('S', 'T', self.inlet_temperature_k, 'P', p_in_pa, fluid)

        t2s_k = CP.PropsSI('T', 'S', s1, 'P', p_out_pa, fluid)
        h2s = CP.PropsSI('H', 'S', s1, 'P', p_out_pa, fluid)

        self.outlet_temperature_isentropic_c = t2s_k - 273.15

        w_isentropic = h2s - h1
        w_actual = w_isentropic / self.isentropic_efficiency

        h2_actual = h1 + w_actual

        t2_k = CP.PropsSI('T', 'H', h2_actual, 'P', p_out_pa, fluid)

        self.outlet_temperature_k = t2_k
        self.outlet_temperature_c = t2_k - 273.15

        drive_efficiency = self.mechanical_efficiency * self.electrical_efficiency
        w_electrical = w_actual / drive_efficiency

        self.specific_energy_kwh_kg = w_electrical * self.J_TO_KWH
        self.compression_work_kwh = self.specific_energy_kwh_kg * self.actual_mass_transferred_kg

    def _calculate_compression_idealgas(self, p_in_pa: float, p_out_pa: float) -> None:
        """
        Calculate compression using ideal gas relations.

        Uses isentropic temperature-pressure relationship:
            T_2s = T_1 × (P_2/P_1)^((γ-1)/γ)
        """
        gamma = 1.41
        cp = 14300.0

        pressure_ratio = p_out_pa / p_in_pa
        exponent = (gamma - 1) / gamma

        t2s_k = self.inlet_temperature_k * (pressure_ratio ** exponent)
        self.outlet_temperature_isentropic_c = t2s_k - 273.15

        delta_t_ideal = t2s_k - self.inlet_temperature_k
        delta_t_actual = delta_t_ideal / self.isentropic_efficiency

        t2_k = self.inlet_temperature_k + delta_t_actual
        self.outlet_temperature_k = t2_k
        self.outlet_temperature_c = t2_k - 273.15

        w_actual = cp * delta_t_actual

        drive_efficiency = self.mechanical_efficiency * self.electrical_efficiency
        w_electrical = w_actual / drive_efficiency

        self.specific_energy_kwh_kg = w_electrical * self.J_TO_KWH
        self.compression_work_kwh = self.specific_energy_kwh_kg * self.actual_mass_transferred_kg

    def get_state(self) -> Dict[str, Any]:
        """
        Retrieve the component's current operational state.

        Fulfills the Component Lifecycle Contract (Layer 1) state access.

        Returns:
            State dictionary containing:

            - **mode** (int): Operating mode (IDLE=0, LP_TO_HP=1).
            - **compression_work_kwh** (float): Electrical work this timestep.
            - **specific_energy_kwh_kg** (float): Specific energy in kWh/kg.
            - **outlet_temperature_c** (float): Actual outlet temperature in °C.
            - **outlet_temperature_isentropic_c** (float): Isentropic outlet
              temperature in °C (for diagnostics).
            - **cumulative_energy_kwh** (float): Total energy since init.
            - **cumulative_mass_kg** (float): Total mass since init.
        """
        cumulative_specific = 0.0
        if self.cumulative_mass_kg > 0:
            cumulative_specific = self.cumulative_energy_kwh / self.cumulative_mass_kg

        return {
            **super().get_state(),
            'mode': int(self.mode),
            'transfer_mass_kg': float(self.transfer_mass_kg),
            'actual_mass_transferred_kg': float(self.actual_mass_transferred_kg),
            'outlet_o2_ppm_mol': (self._inlet_stream.get_total_mole_frac('O2') * 1e6) if self._inlet_stream else 0.0,
            'compression_work_kwh': float(self.compression_work_kwh),
            'energy_consumed_kwh': float(self.energy_consumed_kwh),
            'specific_energy_kwh_kg': float(self.specific_energy_kwh_kg),
            'outlet_temperature_c': float(self.outlet_temperature_c),
            'outlet_temperature_isentropic_c': float(self.outlet_temperature_isentropic_c),
            'cumulative_energy_kwh': float(self.cumulative_energy_kwh),
            'cumulative_mass_kg': float(self.cumulative_mass_kg),
            'timestep_energy_kwh': float(self.timestep_energy_kwh),
            'cumulative_specific_kwh_kg': float(cumulative_specific),
            'inlet_pressure_bar': float(self.inlet_pressure_bar),
            'outlet_pressure_bar': float(self.outlet_pressure_bar),
            'inlet_temperature_c': float(self.inlet_temperature_c),
            'isentropic_efficiency': float(self.isentropic_efficiency),
            'mechanical_efficiency': float(self.mechanical_efficiency),
            'electrical_efficiency': float(self.electrical_efficiency)
        }

    # =========================================================================
    # Port Interface Methods
    # =========================================================================

    def get_output(self, port_name: str) -> Any:
        """
        Retrieve the output stream from a specified port.

        Returns compressed hydrogen at outlet pressure and actual discharge
        temperature (no aftercooling).

        Args:
            port_name: Port identifier ('h2_out' or 'outlet').

        Returns:
            Stream object with outlet conditions.

        Raises:
            ValueError: If port_name is not recognized.
        """
        if port_name == 'h2_out' or port_name == 'outlet':
            # Propagate inlet composition (compression doesn't change composition)
            if self._inlet_stream and self._inlet_stream.composition:
                out_comp = self._inlet_stream.composition.copy()
            else:
                out_comp = {'H2': 1.0}  # Default if no inlet stream
            
            return Stream(
                mass_flow_kg_h=(self.actual_mass_transferred_kg / self.dt
                               if self.dt > 0 else 0.0),
                temperature_k=self.outlet_temperature_k,
                pressure_pa=self.outlet_pressure_bar * self.BAR_TO_PA,
                composition=out_comp,
                phase='gas'
            )
        else:
            raise ValueError(
                f"Unknown output port '{port_name}' on {self.component_id}"
            )

    def receive_input(self, port_name: str, value: Any, resource_type: str) -> float:
        """
        Accept an input stream at the specified port.

        Args:
            port_name: Target port ('h2_in', 'inlet', or 'electricity_in').
            value: Stream object or power value.
            resource_type: Resource classification hint.

        Returns:
            Amount accepted (kg for hydrogen, value for power).
        """
        if port_name == 'h2_in' or port_name == 'inlet' or port_name == 'gas_in':
            if isinstance(value, Stream):
                available_mass = value.mass_flow_kg_h * self.dt
                max_capacity = self.max_flow_kg_h * self.dt

                space_left = max(0.0, max_capacity - self.transfer_mass_kg)
                accepted_mass = min(available_mass, space_left)

                self.transfer_mass_kg += accepted_mass
                
                # Store inlet stream for composition and temperature propagation
                self._inlet_stream = value
                
                # Use inlet stream temperature for compression calculation
                if value.temperature_k > 0:
                    self.inlet_temperature_k = value.temperature_k
                    self.inlet_temperature_c = value.temperature_k - 273.15
                
                return accepted_mass

        elif port_name == 'electricity_in':
            return value if isinstance(value, (int, float)) else 0.0

        return 0.0

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """
        Define the physical connection ports for this component.

        Returns:
            Port definitions dictionary.
        """
        return {
            'h2_in': {
                'type': 'input',
                'resource_type': 'hydrogen',
                'units': 'kg/h'
            },
            'inlet': {
                'type': 'input',
                'resource_type': 'hydrogen',
                'units': 'kg/h'
            },
            'electricity_in': {
                'type': 'input',
                'resource_type': 'electricity',
                'units': 'kW'
            },
            'h2_out': {
                'type': 'output',
                'resource_type': 'hydrogen',
                'units': 'kg/h'
            },
            'outlet': {
                'type': 'output',
                'resource_type': 'hydrogen',
                'units': 'kg/h'
            }
        }
