"""
Water Purifier Component.

This module implements a water purifier using reverse osmosis (RO) to
produce ultrapure water for electrolysis applications. Includes production
control based on downstream tank level.

Reverse Osmosis Model:
    RO separates water from dissolved solids using semi-permeable membranes
    under high pressure. Key parameters:

    - **Recovery Ratio**: Fraction of feed that becomes permeate (typically 0.75-0.85).
    - **Specific Energy**: Energy per unit product (typically 3-4 kWh/m³).
    - **Rejection**: Dissolved solids concentrated in retentate (waste).

    Product and waste flows:
    **ṁ_product = ṁ_feed × recovery**
    **ṁ_waste = ṁ_feed × (1 - recovery)**

Architecture:
    Implements the Component Lifecycle Contract (Layer 1):
    - `initialize()`: Acquires LUTManager reference.
    - `step()`: Calculates purification with tank level control.
    - `get_state()`: Returns power, energy, and flow metrics.
"""

from typing import Dict, Any, Optional

from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.stream import Stream
from h2_plant.core.component_ids import ComponentID
from h2_plant.config.constants_physics import WaterConstants


class WaterPurifier(Component):
    """
    Water purifier producing ultrapure water via reverse osmosis.

    Processes raw water into electrolysis-grade ultrapure water,
    with production controlled by downstream tank level.

    This component fulfills the Component Lifecycle Contract (Layer 1):
        - `initialize()`: Acquires LUTManager for property lookup.
        - `step()`: Calculates purification, energy, and output streams.
        - `get_state()`: Returns power, energy, and buffer status.

    Production Control:
        - Tank fill > 95%: Stop production (prevent overflow).
        - Tank fill < low threshold: Resume production.

    Attributes:
        max_flow_kg_h (float): Maximum processing capacity (kg/h).
        ultrapure_out_stream (Stream): Product water stream.
        waste_out_stream (Stream): Concentrated waste stream.

    Example:
        >>> purifier = WaterPurifier(component_id='WP-1', max_flow_kg_h=1000.0)
        >>> purifier.initialize(dt=1/60, registry=registry)
        >>> purifier.receive_input('raw_water_in', feed_stream, 'water')
        >>> purifier.step(t=0.0)
        >>> product = purifier.get_output('ultrapure_out')
    """

    def __init__(self, component_id: str, max_flow_kg_h: float = None):
        """
        Initialize the water purifier.

        Args:
            component_id (str): Unique identifier for this component.
            max_flow_kg_h (float, optional): Maximum processing capacity in kg/h.
                Default: WaterConstants.WATER_PURIFIER_MAX_FLOW_KGH.
        """
        super().__init__()
        self.component_id = component_id
        self.max_flow_kg_h = max_flow_kg_h if max_flow_kg_h is not None else WaterConstants.WATER_PURIFIER_MAX_FLOW_KGH

        # Input buffer
        self.input_mass_kg = 0.0

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

        Fulfills the Component Lifecycle Contract initialization phase.
        Acquires LUTManager reference for thermodynamic property lookup.

        Args:
            dt (float): Simulation timestep in hours.
            registry (ComponentRegistry): Central registry for component access.
        """
        super().initialize(dt, registry)
        lut_list = registry.get_by_type("lut_manager")
        self.lut = lut_list[0] if lut_list else None

    def step(self, t: float) -> None:
        """
        Execute one simulation timestep.

        Performs RO purification with downstream tank level control:
        1. Check tank level for production permission.
        2. Process input up to max capacity.
        3. Calculate product/waste split via recovery ratio.
        4. Compute energy consumption.
        5. Create output streams.

        Args:
            t (float): Current simulation time in hours.
        """
        super().step(t)

        # Reset timestep accumulators on new timestep
        if t != self._last_step_time:
            self.timestep_power_kw = 0.0
            self.timestep_energy_kwh = 0.0
            self._last_step_time = t

        # Check downstream tank level for production control
        tank = self.get_registry_safe(ComponentID.ULTRAPURE_WATER_STORAGE)
        production_allowed = True
        if tank:
            if hasattr(tank, 'fill_level'):
                if tank.fill_level > 0.95:
                    production_allowed = False
                elif tank.fill_level < WaterConstants.ULTRAPURE_TANK_LOW_FILL_RATIO:
                    production_allowed = True

        # Process input if allowed
        if production_allowed and self.input_mass_kg > 0:
            max_process = self.max_flow_kg_h * self.dt
            processed_mass = min(self.input_mass_kg, max_process)

            # Calculate product/waste split
            recovery = WaterConstants.WATER_RO_RECOVERY_RATIO
            pure_mass = processed_mass * recovery
            waste_mass = processed_mass - pure_mass

            pure_flow = pure_mass / self.dt
            waste_flow = waste_mass / self.dt

            # Energy consumption: kWh = kg × specific_energy
            energy_kwh = pure_mass * WaterConstants.WATER_RO_SPEC_ENERGY_KWH_KG
            batch_power_kw = energy_kwh / self.dt

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

            P_out = max(WaterConstants.WATER_ATM_P_PA, P_in - WaterConstants.WATER_PURIFIER_PRESSURE_DROP_PA)

            # Create product stream (isothermal RO assumption)
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
                composition={'H2O': 0.95, 'Salts': 0.05} if self.raw_water_in_stream else {'H2O': 0.99},
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
        Accept input at specified port.

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
            self.input_mass_kg += mass_in
            return mass_in
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

        Fulfills the Component Lifecycle Contract state access.

        Returns:
            Dict[str, Any]: State dictionary containing:
                - power_kw (float): Current power consumption (kW).
                - timestep_energy_kwh (float): Energy this timestep (kWh).
                - input_buffer_kg (float): Buffered input mass (kg).
                - ultrapure_flow_kgh (float): Product flow rate (kg/h).
        """
        return {
            **super().get_state(),
            'power_kw': self.power_consumed_kw,
            'timestep_energy_kwh': self.timestep_energy_kwh,
            'input_buffer_kg': self.input_mass_kg,
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
