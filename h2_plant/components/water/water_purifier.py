"""
Water Purifier Component with Thermal Derating.

This module implements a water purifier using reverse osmosis (RO) to
produce ultrapure water for electrolysis applications. Includes Temperature
Correction Factor (TCF) physics for realistic membrane performance modeling.

Reverse Osmosis Model:
    RO membranes are sensitive to water temperature due to viscosity changes.
    Cold water has higher viscosity, reducing permeation flux.

    **Q_actual = Q_rated × TCF**
    **TCF = exp(0.025 × (T_feed - 25))**
    
    Performance Examples:
    - @ 25°C: TCF = 1.0 (Rated Capacity)
    - @ 15°C: TCF ≈ 0.78 (22% capacity loss)
    - @ 10°C: TCF ≈ 0.69 (31% capacity loss)
    - @  5°C: Membrane shutdown (freeze protection)

Architecture:
    Implements the Component Lifecycle Contract (Layer 1):
    - `initialize()`: Acquires LUTManager reference.
    - `step()`: Calculates purification with TCF derating.
    - `get_state()`: Returns power, energy, capacity, and flow metrics.
"""

import math
import logging
from typing import Dict, Any, Optional

from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.stream import Stream
from h2_plant.config.constants_physics import WaterConstants

logger = logging.getLogger(__name__)


class WaterPurifier(Component):
    """
    High-fidelity water purifier with Temperature Correction Factor physics.

    Processes raw water into electrolysis-grade ultrapure water with
    realistic thermal derating based on feed water temperature.

    This component fulfills the Component Lifecycle Contract (Layer 1):
        - `initialize()`: Acquires LUTManager for property lookup.
        - `step()`: Calculates purification with TCF derating.
        - `get_state()`: Returns power, capacity, and buffer status.

    Physics Model:
        RO membrane flux is proportional to temperature due to viscosity.
        TCF = exp(0.025 × (T_feed - 25)) where T_feed is in °C.
        
    Attributes:
        max_flow_kg_h_rated (float): Rated capacity at 25°C (kg/h).
        current_capacity_kg_h (float): Real-time capacity after TCF derating.
        tcf (float): Current Temperature Correction Factor.

    Example:
        >>> purifier = WaterPurifier(component_id='WP-1', max_flow_kg_h=12000.0)
        >>> purifier.initialize(dt=1/60, registry=registry)
        >>> purifier.receive_input('raw_water_in', feed_stream, 'water')
        >>> purifier.step(t=0.0)
        >>> product = purifier.get_output('ultrapure_out')
    """

    # Physics Constants
    TCF_COEFFICIENT = 0.025  # Typical for polyamide RO membranes
    FREEZE_PROTECTION_TEMP_C = 5.0  # Minimum operating temperature
    REFERENCE_TEMP_C = 25.0  # Standard rating temperature

    def __init__(
        self,
        component_id: str,
        max_flow_kg_h: float = None,
        recovery_ratio: float = None
    ):
        """
        Initialize the water purifier.

        Args:
            component_id (str): Unique identifier for this component.
            max_flow_kg_h (float, optional): Rated capacity at 25°C in kg/h.
                Default: WaterConstants.WATER_PURIFIER_MAX_FLOW_KGH.
            recovery_ratio (float, optional): Fraction of feed becoming product.
                Default: WaterConstants.WATER_RO_RECOVERY_RATIO.
        """
        super().__init__()
        self.component_id = component_id
        
        # Rated capacity at Standard Conditions (25°C)
        self.max_flow_kg_h_rated = max_flow_kg_h if max_flow_kg_h is not None else WaterConstants.WATER_PURIFIER_MAX_FLOW_KGH
        self.recovery_ratio = recovery_ratio if recovery_ratio is not None else WaterConstants.WATER_RO_RECOVERY_RATIO

        # Input buffer with overflow protection
        self.input_mass_kg = 0.0
        self._max_buffer_kg = 0.0  # Set in initialize()

        # Performance metrics
        self.current_capacity_kg_h = 0.0
        self.tcf = 1.0
        self.feed_temperature_c = 25.0

        # Timestep accumulators
        self._last_step_time = -1.0
        self.timestep_power_kw = 0.0
        self.timestep_energy_kwh = 0.0
        self.power_consumed_kw = 0.0

        # Output streams
        self.ultrapure_out_stream: Optional[Stream] = None
        self.waste_out_stream: Optional[Stream] = None
        self.raw_water_in_stream: Optional[Stream] = None

        # LUT Manager
        self.lut = None

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """
        Prepare the component for simulation execution.

        Sets up buffer limits and acquires LUTManager reference.

        Args:
            dt (float): Simulation timestep in hours.
            registry (ComponentRegistry): Central registry for component access.
        """
        super().initialize(dt, registry)
        lut_list = registry.get_by_type("lut_manager")
        self.lut = lut_list[0] if lut_list else None
        
        # Buffer limit: 2 timesteps worth at rated capacity
        self._max_buffer_kg = self.max_flow_kg_h_rated * self.dt * 2.0

    def step(self, t: float) -> None:
        """
        Execute one simulation timestep with TCF physics.

        Performs RO purification with thermal derating:
        1. Determine feed temperature from inlet stream.
        2. Calculate Temperature Correction Factor (TCF).
        3. Apply freeze protection if temperature too low.
        4. Process input up to derated capacity.
        5. Calculate product/waste split via recovery ratio.
        6. Compute energy consumption.
        7. Create output streams.

        Args:
            t (float): Current simulation time in hours.
        """
        super().step(t)

        # Reset timestep accumulators on new timestep
        if t != self._last_step_time:
            self.timestep_power_kw = 0.0
            self.timestep_energy_kwh = 0.0
            self._last_step_time = t

        # 1. Determine Feed Temperature
        # Default to 15°C (groundwater) if no stream connected
        self.feed_temperature_c = 15.0
        if self.raw_water_in_stream:
            self.feed_temperature_c = self.raw_water_in_stream.temperature_k - 273.15

        # 2. Calculate Temperature Correction Factor (TCF)
        # Standard approximation for Polyamide RO membranes
        # TCF = 1 @ 25°C. Lower T -> Higher Viscosity -> Lower Flow.
        
        # 3. Freeze Protection Check
        if self.feed_temperature_c < self.FREEZE_PROTECTION_TEMP_C:
            logger.warning(
                f"WaterPurifier {self.component_id}: Feed temp too low "
                f"({self.feed_temperature_c:.1f}°C < {self.FREEZE_PROTECTION_TEMP_C}°C) - "
                "membrane shutdown for freeze protection"
            )
            self.tcf = 0.0
            self.current_capacity_kg_h = 0.0
        else:
            self.tcf = math.exp(
                self.TCF_COEFFICIENT * (self.feed_temperature_c - self.REFERENCE_TEMP_C)
            )
            self.current_capacity_kg_h = self.max_flow_kg_h_rated * self.tcf

        # 4. Process Input (push architecture - process what's in buffer)
        if self.input_mass_kg > 0 and self.current_capacity_kg_h > 0:
            max_process_step = self.current_capacity_kg_h * self.dt
            processed_mass = min(self.input_mass_kg, max_process_step)

            # 5. Calculate product/waste split
            pure_mass = processed_mass * self.recovery_ratio
            waste_mass = processed_mass - pure_mass

            pure_flow = pure_mass / self.dt if self.dt > 0 else 0.0
            waste_flow = waste_mass / self.dt if self.dt > 0 else 0.0

            # 6. Energy consumption: kWh = kg × specific_energy
            energy_kwh = pure_mass * WaterConstants.WATER_RO_SPEC_ENERGY_KWH_KG
            batch_power_kw = energy_kwh / self.dt if self.dt > 0 else 0.0

            self.timestep_power_kw += batch_power_kw
            self.timestep_energy_kwh += energy_kwh
            self.power_consumed_kw = self.timestep_power_kw

            # Get inlet conditions for output streams
            if self.raw_water_in_stream:
                T_in = self.raw_water_in_stream.temperature_k
                P_in = self.raw_water_in_stream.pressure_pa
            else:
                T_in = WaterConstants.WATER_AMBIENT_T_K
                P_in = WaterConstants.WATER_ATM_P_PA

            P_out = max(
                WaterConstants.WATER_ATM_P_PA,
                P_in - WaterConstants.WATER_PURIFIER_PRESSURE_DROP_PA
            )

            # 7. Create product stream (isothermal RO assumption)
            self.ultrapure_out_stream = Stream(
                mass_flow_kg_h=pure_flow,
                temperature_k=T_in,
                pressure_pa=P_out,
                composition={'H2O': 1.0},
                phase='liquid'
            )

            # Create waste stream (concentrated retentate)
            self.waste_out_stream = Stream(
                mass_flow_kg_h=waste_flow,
                temperature_k=T_in,
                pressure_pa=WaterConstants.WATER_ATM_P_PA,
                composition={'H2O': 0.95, 'Salts': 0.05},
                phase='liquid'
            )

            # Consume from input buffer
            self.input_mass_kg -= processed_mass

        else:
            # No processing this step
            self.power_consumed_kw = self.timestep_power_kw
            self.ultrapure_out_stream = None
            self.waste_out_stream = None

    def receive_input(self, port_name: str, value: Any, resource_type: str = None) -> float:
        """
        Accept input at specified port with buffer overflow protection.

        Args:
            port_name (str): Target port ('raw_water_in').
            value (Any): Input stream.
            resource_type (str, optional): Resource classification hint.

        Returns:
            float: Mass accepted (kg).
        """
        if port_name == 'raw_water_in' and isinstance(value, Stream):
            self.raw_water_in_stream = value
            mass_in = value.mass_flow_kg_h * self.dt
            
            # Buffer overflow protection
            available_space = self._max_buffer_kg - self.input_mass_kg
            if mass_in > available_space:
                rejected = mass_in - available_space
                accepted = available_space
                if rejected > 0.1:  # Only log significant rejections
                    logger.warning(
                        f"WaterPurifier {self.component_id}: Input buffer full, "
                        f"rejected {rejected:.1f} kg"
                    )
            else:
                accepted = mass_in
                
            self.input_mass_kg += accepted
            return accepted
        return 0.0

    def get_output(self, port_name: str) -> Any:
        """
        Retrieve output stream from specified port.

        Args:
            port_name (str): Port identifier ('ultrapure_out' or 'waste_out').

        Returns:
            Stream: Output stream or None.
        """
        if port_name == 'ultrapure_out':
            return self.ultrapure_out_stream
        elif port_name == 'waste_out':
            return self.waste_out_stream
        return None

    def get_state(self) -> Dict[str, Any]:
        """
        Retrieve the component's current operational state.

        Returns:
            Dict[str, Any]: State dictionary containing performance metrics.
        """
        return {
            **super().get_state(),
            'power_kw': self.power_consumed_kw,
            'timestep_energy_kwh': self.timestep_energy_kwh,
            'input_buffer_kg': self.input_mass_kg,
            'current_capacity_kg_h': self.current_capacity_kg_h,
            'rated_capacity_kg_h': self.max_flow_kg_h_rated,
            'tcf': self.tcf,
            'feed_temperature_c': self.feed_temperature_c,
            'ultrapure_flow_kgh': self.ultrapure_out_stream.mass_flow_kg_h if self.ultrapure_out_stream else 0.0
        }

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """
        Define the physical connection ports for this component.

        Returns:
            Dict[str, Dict[str, str]]: Port definitions.
        """
        return {
            'raw_water_in': {'type': 'input', 'resource_type': 'water', 'units': 'kg/h'},
            'ultrapure_out': {'type': 'output', 'resource_type': 'water', 'units': 'kg/h'},
            'waste_out': {'type': 'output', 'resource_type': 'water', 'units': 'kg/h'}
        }
