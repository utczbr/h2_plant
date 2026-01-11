"""
Ultra-Pure Water Tank Component with Multi-Channel Distribution.

This module implements a thermodynamically accurate ultrapure water buffer
tank with:
- Enthalpy-based mixing for inlet streams
- 3-zone level control for upstream production feedback
- Multi-channel demand aggregation for downstream distribution

Multi-Channel Architecture:
    The tank accepts demand signals from multiple downstream components
    (e.g., makeup mixers) using a naming convention:
    
    - Input Port: `demand_{channel_id}` (e.g., `demand_PEM`, `demand_SOEC`)
    - Output Port: `water_out_{channel_id}` (e.g., `water_out_PEM`)
    
    This allows scaling to 3, 5, or 10 consumers automatically without
    code changes - just add new port connections in topology.

Inventory Management:
    If total demand exceeds available inventory, flows are scaled
    proportionally (rationing mode) to prevent negative mass.

Control Logic (Feedback Signal):
    The tank also generates a production request signal for upstream
    water sources based on fill level zones with hysteresis.
"""

import logging
from typing import Dict, Any, List, Optional

from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.stream import Stream
from h2_plant.config.constants_physics import WaterConstants

logger = logging.getLogger(__name__)


class UltraPureWaterTank(Component):
    """
    Ultra-pure water buffer tank with multi-channel distribution.

    Aggregates demand signals from multiple downstream components and
    allocates water based on available inventory, while maintaining
    feedback control to upstream production.

    Control Zones (with Hysteresis):
        Zone A (Critical): fill < 60%, request 120% nominal
        Zone B (Nominal): 60% < fill < 90%, request 100% nominal
        Zone C (Throttle): fill > 90%, request 0% (stop)

    Multi-Channel Ports:
        Dynamic input ports: demand_{channel_id}
        Dynamic output ports: water_out_{channel_id}

    Attributes:
        capacity_kg (float): Tank capacity (kg).
        mass_kg (float): Current water inventory (kg).
        temperature_k (float): Current water temperature (K).
        demands (Dict[str, float]): Aggregated demand signals.

    Example YAML:
        - id: "UltraPure_Tank"
          type: "UltraPureWaterTank"
          params:
            capacity_kg: 20000.0
            nominal_production_kg_h: 10000.0
          connections:
            # Receive from purifier
            - source_port: "ultrapure_out"
              target_port: "ultrapure_in"
            # Send control signal to source
            - source_port: "control_signal"
              target_name: "Water_Source"
              target_port: "control_signal"
            # Receive demand from PEM mixer
            - source_name: "PEM_Makeup_Mixer"
              source_port: "demand_signal"
              target_port: "demand_PEM"
            # Send water to PEM pump
            - source_port: "water_out_PEM"
              target_name: "PEM_Water_Pump"
              target_port: "water_in"
    """

    # Zone Thresholds (with Hysteresis)
    ZONE_A_ENTER = 0.60
    ZONE_A_EXIT = 0.65
    ZONE_C_ENTER = 0.90
    ZONE_C_EXIT = 0.85

    # Flow Multipliers
    ZONE_A_MULTIPLIER = 1.2  # 120% of nominal
    ZONE_B_MULTIPLIER = 1.0  # 100% of nominal
    ZONE_C_MULTIPLIER = 0.0  # 0% (stop production)

    def __init__(
        self,
        component_id: str,
        capacity_kg: float = None,
        nominal_production_kg_h: float = None,
        initial_fill_fraction: float = 0.5
    ):
        """
        Initialize the ultra-pure water tank.

        Args:
            component_id (str): Unique identifier for this component.
            capacity_kg (float, optional): Tank capacity in kg.
                Default: WaterConstants.ULTRAPURE_TANK_CAPACITY_KG.
            nominal_production_kg_h (float, optional): Nominal production target in kg/h.
                Default: 10000.0 kg/h (10 m³/h).
            initial_fill_fraction (float): Initial fill level (0-1). Default: 0.5.
        """
        super().__init__()
        self.component_id = component_id
        self.capacity_kg = capacity_kg if capacity_kg is not None else WaterConstants.ULTRAPURE_TANK_CAPACITY_KG
        self.nominal_production_kg_h = nominal_production_kg_h if nominal_production_kg_h is not None else 10000.0

        # Thermal state
        self.mass_kg = self.capacity_kg * initial_fill_fraction
        self.temperature_k = WaterConstants.WATER_AMBIENT_T_K
        self.pressure_pa = WaterConstants.WATER_ATM_P_PA

        # Control state
        self.control_zone = 'B'  # Start in nominal zone
        self.requested_production_kg_h = self.nominal_production_kg_h * self.ZONE_B_MULTIPLIER

        # Fill level (updated in step)
        self.fill_level = initial_fill_fraction

        # LUT Manager for property lookup
        self.lut = None

        # Input stream (from purifier)
        self.inlet_stream: Optional[Stream] = None

        # Multi-channel demand aggregation
        # Key: channel_id (e.g., 'PEM', 'SOEC', 'ATR')
        # Value: demand rate (kg/h)
        self.demands: Dict[str, float] = {}

        # Multi-channel output allocation
        # Key: channel_id
        # Value: allocated Stream
        self.channel_outputs: Dict[str, Stream] = {}

        # Legacy single-consumer support
        self.outlet_streams: Dict[str, Stream] = {}
        self.outlet_requests: Dict[str, float] = {}

        # Metrics
        self.overflow_kg = 0.0
        self.total_demand_kg_h = 0.0
        self.total_allocated_kg_h = 0.0
        self.rationing_factor = 1.0

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """
        Prepare the component for simulation execution.

        Args:
            dt (float): Simulation timestep in hours.
            registry (ComponentRegistry): Central registry for component access.
        """
        super().initialize(dt, registry)
        lut_list = registry.get_by_type("lut_manager")
        self.lut = lut_list[0] if lut_list else None

    def request_outflow(self, consumer_id: str, amount_kg_h: float) -> None:
        """
        Register a demand from a downstream consumer (legacy API).

        For new implementations, use the demand_{channel_id} port instead.

        Args:
            consumer_id (str): Unique identifier of requesting consumer.
            amount_kg_h (float): Requested flow rate in kg/h.
        """
        self.outlet_requests[consumer_id] = amount_kg_h

    def step(self, t: float) -> None:
        """
        Execute one simulation timestep.

        Process:
        1. Update fill level.
        2. Calculate control zone with hysteresis.
        3. Generate production request signal.
        4. Process inlet flow with enthalpy-based mixing.
        5. Aggregate all demands (multi-channel + legacy).
        6. Perform inventory check and rationing if needed.
        7. Allocate water to each channel.
        8. Update mass and clear demands.

        Args:
            t (float): Current simulation time in hours.
        """
        super().step(t)

        # 1. Update Fill Level
        self.fill_level = self.mass_kg / self.capacity_kg if self.capacity_kg > 0 else 0.0

        # 2. Control Logic with Hysteresis
        self._update_control_zone()

        # 3. Generate Production Request Signal
        if self.control_zone == 'A':
            self.requested_production_kg_h = self.nominal_production_kg_h * self.ZONE_A_MULTIPLIER
        elif self.control_zone == 'B':
            self.requested_production_kg_h = self.nominal_production_kg_h * self.ZONE_B_MULTIPLIER
        else:  # Zone C
            self.requested_production_kg_h = self.nominal_production_kg_h * self.ZONE_C_MULTIPLIER

        # 4. Process Inflow with Enthalpy Mixing
        self.overflow_kg = 0.0
        if self.inlet_stream and self.inlet_stream.mass_flow_kg_h > 0:
            m_in = self.inlet_stream.mass_flow_kg_h * self.dt
            T_in = self.inlet_stream.temperature_k

            space = self.capacity_kg - self.mass_kg

            if m_in > space:
                self.overflow_kg = m_in - space
                m_accepted = space
                if self.overflow_kg > 0.1:
                    logger.warning(
                        f"UltraPureWaterTank {self.component_id}: "
                        f"{self.overflow_kg:.1f} kg overflow (tank full)"
                    )
            else:
                m_accepted = m_in

            if m_accepted > 0:
                Cp = 4184.0  # J/(kg·K)
                H_current = self.mass_kg * Cp * self.temperature_k
                H_in = m_accepted * Cp * T_in
                m_new = self.mass_kg + m_accepted
                H_new = H_current + H_in
                self.temperature_k = H_new / (m_new * Cp) if m_new > 0 else self.temperature_k
                self.mass_kg = m_new

        # 5. Aggregate All Demands
        # Combine multi-channel demands with legacy outlet_requests
        all_demands: Dict[str, float] = {}
        all_demands.update(self.demands)
        all_demands.update(self.outlet_requests)

        self.total_demand_kg_h = sum(all_demands.values())
        total_demand_mass = self.total_demand_kg_h * self.dt

        # 6. Inventory Check & Rationing
        self.rationing_factor = 1.0
        if total_demand_mass > self.mass_kg:
            if total_demand_mass > 0:
                self.rationing_factor = self.mass_kg / total_demand_mass
            logger.warning(
                f"UltraPureWaterTank {self.component_id}: Low level! "
                f"Rationing output by {self.rationing_factor:.2%}"
            )

        # 7. Allocate Water to Each Channel
        self.channel_outputs.clear()
        self.outlet_streams.clear()
        self.total_allocated_kg_h = 0.0
        total_out_kg = 0.0

        for channel_id, demand_rate in all_demands.items():
            allocated_rate = demand_rate * self.rationing_factor

            if allocated_rate > 0:
                output_stream = Stream(
                    mass_flow_kg_h=allocated_rate,
                    temperature_k=self.temperature_k,
                    pressure_pa=self.pressure_pa,
                    composition={'H2O': 1.0},
                    phase='liquid'
                )

                # Store in both lookups for compatibility
                self.channel_outputs[channel_id] = output_stream
                self.outlet_streams[channel_id] = output_stream

                total_out_kg += allocated_rate * self.dt
                self.total_allocated_kg_h += allocated_rate

        # 8. Update Mass
        self.mass_kg -= total_out_kg
        if self.mass_kg < 0:
            self.mass_kg = 0.0

        # Update fill level after all operations
        self.fill_level = self.mass_kg / self.capacity_kg if self.capacity_kg > 0 else 0.0

        # 9. Clear Demands for Next Timestep
        self.demands.clear()
        self.outlet_requests.clear()

    def _update_control_zone(self) -> None:
        """
        Update control zone with hysteresis to prevent oscillation.
        """
        if self.control_zone == 'A':
            if self.fill_level > self.ZONE_A_EXIT:
                self.control_zone = 'B'
                logger.debug(f"UltraPureWaterTank {self.component_id}: Zone A -> B")
        elif self.control_zone == 'B':
            if self.fill_level < self.ZONE_A_ENTER:
                self.control_zone = 'A'
                logger.debug(f"UltraPureWaterTank {self.component_id}: Zone B -> A")
            elif self.fill_level > self.ZONE_C_ENTER:
                self.control_zone = 'C'
                logger.debug(f"UltraPureWaterTank {self.component_id}: Zone B -> C")
        elif self.control_zone == 'C':
            if self.fill_level < self.ZONE_C_EXIT:
                self.control_zone = 'B'
                logger.debug(f"UltraPureWaterTank {self.component_id}: Zone C -> B")

    def receive_input(self, port_name: str, value: Any, resource_type: str = None) -> float:
        """
        Accept input at specified ports.

        Static Ports:
            ultrapure_in: Physical water from purifier
            
        Dynamic Ports (pattern-matched):
            demand_{channel_id}: Demand signal from downstream mixer
                (e.g., demand_PEM, demand_SOEC, demand_ATR)

        Args:
            port_name (str): Target port.
            value (Any): Input stream or signal.
            resource_type (str, optional): Resource classification hint.

        Returns:
            float: Mass accepted (kg) or 0.0 for signals.
        """
        # Physical water input from purifier
        if port_name == 'ultrapure_in' and isinstance(value, Stream):
            self.inlet_stream = value
            return value.mass_flow_kg_h * self.dt

        # Multi-channel demand signals (pattern: demand_{channel_id})
        if port_name.startswith('demand_'):
            channel_id = port_name.split('_', 1)[1]  # Extract 'PEM' from 'demand_PEM'

            signal_val = 0.0
            if isinstance(value, Stream):
                signal_val = value.mass_flow_kg_h
            elif isinstance(value, (int, float)):
                signal_val = float(value)

            self.demands[channel_id] = signal_val
            logger.debug(
                f"UltraPureWaterTank {self.component_id}: "
                f"Received demand_{channel_id} = {signal_val:.0f} kg/h"
            )
            return 0.0  # Signals are not consumed mass

        return 0.0

    def get_output(self, port_name: str) -> Any:
        """
        Retrieve output from specified ports.

        Static Ports:
            control_signal: Signal Stream for upstream production control
            consumer_out: Legacy single-consumer output
            
        Dynamic Ports (pattern-matched):
            water_out_{channel_id}: Allocated water for specific channel
                (e.g., water_out_PEM, water_out_SOEC)

        Args:
            port_name (str): Port identifier.

        Returns:
            Stream: Output stream or signal, or None if not found.
        """
        # Control signal for upstream source
        if port_name == 'control_signal':
            return Stream(
                mass_flow_kg_h=self.requested_production_kg_h,
                temperature_k=293.15,
                pressure_pa=101325.0,
                composition={'Signal': 1.0},
                phase='signal'
            )

        # Multi-channel water output (pattern: water_out_{channel_id})
        if port_name.startswith('water_out_'):
            channel_id = port_name.split('_', 2)[2]  # Extract 'PEM' from 'water_out_PEM'
            return self.channel_outputs.get(channel_id, None)

        # Legacy single consumer output
        if port_name == 'consumer_out':
            vals = list(self.outlet_streams.values())
            if vals:
                return vals[0]

        # Direct lookup by consumer ID
        return self.outlet_streams.get(port_name)

    def get_state(self) -> Dict[str, Any]:
        """
        Retrieve the component's current operational state.

        Returns:
            Dict[str, Any]: State dictionary with inventory and control metrics.
        """
        return {
            **super().get_state(),
            'component_id': self.component_id,
            'mass_kg': self.mass_kg,
            'fill_level': self.fill_level,
            'temperature_c': self.temperature_k - 273.15,
            'pressure_bar': self.pressure_pa / 1e5,
            'control_zone': self.control_zone,
            'requested_production_kg_h': self.requested_production_kg_h,
            'total_demand_kg_h': self.total_demand_kg_h,
            'total_allocated_kg_h': self.total_allocated_kg_h,
            'rationing_factor': self.rationing_factor,
            'overflow_kg': self.overflow_kg,
            'active_channels': list(self.channel_outputs.keys())
        }

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """
        Define the connection ports for this component.

        Note: Dynamic ports (demand_*, water_out_*) are pattern-matched
        at runtime and don't need to be declared here.

        Returns:
            Dict[str, Dict[str, str]]: Port definitions.
        """
        return {
            'ultrapure_in': {'type': 'input', 'resource_type': 'water', 'units': 'kg/h'},
            'control_signal': {'type': 'output', 'resource_type': 'signal', 'units': 'kg/h'},
            'consumer_out': {'type': 'output', 'resource_type': 'water', 'units': 'kg/h'},
            # Dynamic ports: demand_{id} (input), water_out_{id} (output)
        }
