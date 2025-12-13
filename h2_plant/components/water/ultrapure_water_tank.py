"""
Ultra-Pure Water Tank Component.

This module implements a thermodynamically accurate ultrapure water buffer
tank with enthalpy-based mixing and demand-driven outflow management.

Thermodynamic Model:
    When inlet water mixes with stored water, the final temperature is
    calculated via enthalpy balance:

    **(m_stored × H_stored) + (m_inlet × H_inlet) = (m_total × H_final)**

    For liquid water: H ≈ Cp × T, where Cp = 4184 J/(kg·K).

    This ensures energy conservation during mixing of streams at
    different temperatures.

Outflow Management:
    Multiple consumers (e.g., PEM, SOEC) register demand via `request_outflow()`.
    If total demand exceeds available mass, flows are scaled proportionally
    to prevent over-extraction.

Architecture:
    Implements the Component Lifecycle Contract (Layer 1):
    - `initialize()`: Acquires LUTManager reference for property lookup.
    - `step()`: Processes inflow mixing, allocates outflows to consumers.
    - `get_state()`: Returns mass, fill level, and temperature.
"""

from typing import Dict, Any, List, Optional

from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.stream import Stream
from h2_plant.core.component_ids import ComponentID
from h2_plant.config.constants_physics import WaterConstants


class UltraPureWaterTank(Component):
    """
    Ultra-pure water buffer tank with enthalpy-based mixing.

    Provides demand-driven water supply to multiple consumers with
    thermodynamically correct temperature tracking during mixing.

    This component fulfills the Component Lifecycle Contract (Layer 1):
        - `initialize()`: Acquires LUTManager for property lookup.
        - `step()`: Processes inflow mixing, allocates consumer outflows.
        - `get_state()`: Returns mass, fill level, and thermal state.

    Mixing Model:
        T_final = (m_old × T_old + m_in × T_in) / (m_old + m_in)

        This energy-conserving approach prevents temperature discontinuities
        when mixing streams at different temperatures.

    Attributes:
        capacity_kg (float): Tank capacity (kg).
        mass_kg (float): Current water inventory (kg).
        temperature_k (float): Current water temperature (K).

    Example:
        >>> tank = UltraPureWaterTank(component_id='WT-1', capacity_kg=5000.0)
        >>> tank.initialize(dt=1/60, registry=registry)
        >>> tank.request_outflow('pem_electrolyzer', 100.0)
        >>> tank.step(t=0.0)
        >>> pem_stream = tank.outlet_streams.get('pem_electrolyzer')
    """

    def __init__(self, component_id: str, capacity_kg: float = None):
        """
        Initialize the ultra-pure water tank.

        Args:
            component_id (str): Unique identifier for this component.
            capacity_kg (float, optional): Tank capacity in kg.
                Default: WaterConstants.ULTRAPURE_TANK_CAPACITY_KG.
        """
        super().__init__()
        self.component_id = component_id
        self.capacity_kg = capacity_kg if capacity_kg is not None else WaterConstants.ULTRAPURE_TANK_CAPACITY_KG

        # Thermal state (start at 50% fill)
        self.mass_kg = self.capacity_kg * 0.5
        self.temperature_k = WaterConstants.WATER_AMBIENT_T_K
        self.pressure_pa = WaterConstants.WATER_ATM_P_PA

        # LUT Manager for property lookup
        self.lut = None

        # Input stream
        self.inlet_stream: Optional[Stream] = None

        # Output management
        self.outlet_streams: Dict[str, Stream] = {}
        self.outlet_requests: Dict[str, float] = {}

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

    def request_outflow(self, consumer_id: str, amount_kg_h: float) -> None:
        """
        Register a demand from a downstream consumer.

        Consumers call this method to request water. All requests are
        processed during step() with proportional scaling if demand
        exceeds available inventory.

        Args:
            consumer_id (str): Unique identifier of requesting consumer.
            amount_kg_h (float): Requested flow rate in kg/h.
        """
        self.outlet_requests[consumer_id] = amount_kg_h

    def step(self, t: float) -> None:
        """
        Execute one simulation timestep.

        Performs complete tank cycle:
        1. Process inlet flow with enthalpy-based mixing.
        2. Allocate outflows to consumers (proportional scaling if needed).
        3. Update mass and thermal state.
        4. Clear request buffer for next timestep.

        Args:
            t (float): Current simulation time in hours.
        """
        super().step(t)

        # Process inflow with enthalpy mixing
        if self.inlet_stream and self.inlet_stream.mass_flow_kg_h > 0:
            m_in = self.inlet_stream.mass_flow_kg_h * self.dt
            T_in = self.inlet_stream.temperature_k

            # Clamp to available capacity
            m_accepted = min(m_in, self.capacity_kg - self.mass_kg)

            if m_accepted > 0:
                # Enthalpy balance: (m_old × Cp × T_old) + (m_in × Cp × T_in) = m_new × Cp × T_new
                Cp = 4184.0

                H_current = self.mass_kg * Cp * self.temperature_k
                H_in = m_accepted * Cp * T_in

                m_new = self.mass_kg + m_accepted
                H_new = H_current + H_in

                self.temperature_k = H_new / (m_new * Cp)
                self.mass_kg = m_new

        # Process outflows with proportional scaling
        total_req_kg = sum(self.outlet_requests.values()) * self.dt

        scaling = 1.0
        if total_req_kg > self.mass_kg:
            scaling = self.mass_kg / total_req_kg if total_req_kg > 0 else 0.0

        self.outlet_streams.clear()
        total_out_kg = 0.0

        for cid, rate in self.outlet_requests.items():
            actual_rate = rate * scaling
            if actual_rate > 0:
                self.outlet_streams[cid] = Stream(
                    mass_flow_kg_h=actual_rate,
                    temperature_k=self.temperature_k,
                    pressure_pa=self.pressure_pa,
                    composition={'H2O': 1.0},
                    phase='liquid'
                )
                total_out_kg += actual_rate * self.dt

        self.mass_kg -= total_out_kg
        if self.mass_kg < 0:
            self.mass_kg = 0.0

        # Clear requests for next timestep
        self.outlet_requests.clear()

    def receive_input(self, port_name: str, value: Any, resource_type: str = None) -> float:
        """
        Accept input stream at specified port.

        Calculates backpressure-limited acceptance based on
        available tank capacity.

        Args:
            port_name (str): Target port ('ultrapure_in').
            value (Any): Input stream.
            resource_type (str, optional): Resource classification hint.

        Returns:
            float: Mass accepted (kg).
        """
        if port_name == 'ultrapure_in' and isinstance(value, Stream):
            self.inlet_stream = value
            space = self.capacity_kg - self.mass_kg
            accepted_flow = min(value.mass_flow_kg_h, space / self.dt)
            return accepted_flow * self.dt
        return 0.0

    def get_output(self, port_name: str) -> Any:
        """
        Retrieve output stream for specified consumer.

        Args:
            port_name (str): Consumer ID or 'consumer_out' for first available.

        Returns:
            Stream: Output stream for consumer, or None if not found.
        """
        if port_name == 'consumer_out':
            vals = list(self.outlet_streams.values())
            if vals:
                return vals[0]
        return self.outlet_streams.get(port_name)

    def get_state(self) -> Dict[str, Any]:
        """
        Retrieve the component's current operational state.

        Fulfills the Component Lifecycle Contract state access.

        Returns:
            Dict[str, Any]: State dictionary containing:
                - mass_kg (float): Current inventory (kg).
                - fill_level (float): Fill fraction (0-1).
                - temperature_c (float): Water temperature (°C).
                - pressure_bar (float): Tank pressure (bar).
        """
        return {
            **super().get_state(),
            'component_id': self.component_id,
            'mass_kg': self.mass_kg,
            'fill_level': self.mass_kg / self.capacity_kg,
            'temperature_c': self.temperature_k - 273.15,
            'pressure_bar': self.pressure_pa / 1e5
        }

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """
        Define the physical connection ports for this component.

        Returns:
            Dict[str, Dict[str, str]]: Port definitions.
        """
        return {
            'ultrapure_in': {'type': 'input', 'resource_type': 'water', 'units': 'kg/h'},
            'consumer_out': {'type': 'output', 'resource_type': 'water', 'units': 'kg/h'}
        }
