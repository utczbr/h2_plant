"""
Signal-Based Makeup Mixer Component.

This component implements a demand-driven mixing node for water recirculation loops
using the "Demand Signal Propagation Pattern". Unlike the standard MakeupMixer,
it does NOT generate makeup water internally. Instead:

1. **Calculate Deficit**: Demand = Target_Flow - Recirculated_Flow
2. **Broadcast Signal**: Send 'demand_signal' stream upstream to tank.
3. **Accept Supply**: Receive 'makeup_water_in' from upstream (Tank/Pump).
4. **Mix**: Outlet = Recirculated + Accepted_Makeup.

Architecture:
    This introduces a **1-timestep delay** between demand calculation and
    water receipt, which is physically realistic for valve actuation and
    piping signal propagation in dynamic simulations.

    Timestep N: Calculate deficit, send signal
    Timestep N+1: Receive water based on previous signal, mix, output

Physics Model (Adiabatic Mixing):
    H_out * m_out = (H_drain * m_drain) + (H_makeup * m_makeup)
    Using rigorous enthalpy balance via LUT Manager.
"""

import logging
from typing import Dict, Any, Optional

from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.stream import Stream
from h2_plant.optimization.numba_ops import solve_water_T_from_H_jit

logger = logging.getLogger(__name__)


class SignalMakeupMixer(Component):
    """
    Signal-based makeup mixer for push-architecture water loops.
    
    Instead of generating water internally, this mixer:
    - Calculates the demand deficit
    - Broadcasts a demand_signal to upstream tank/source
    - Accepts physical water from makeup_water_in port
    - Mixes recirculated + makeup thermodynamically
    
    Attributes:
        target_flow_kg_h (float): Flow control setpoint (kg/h).
        current_demand_kg_h (float): Calculated deficit for signal.
        actual_makeup_kg_h (float): Physical water received this step.
        
    Example YAML:
        - id: "PEM_Makeup_Mixer"
          type: "SignalMakeupMixer"
          params:
            target_flow_kg_h: 1000.0
          connections:
            - source_port: "demand_signal"
              target_name: "UltraPure_Tank"
              target_port: "demand_PEM"
    """

    def __init__(
        self,
        component_id: str,
        target_flow_kg_h: float,
        makeup_temp_c: float = 20.0,
        makeup_pressure_bar: float = 1.0
    ):
        """
        Initialize the signal-based makeup mixer.

        Args:
            component_id (str): Unique identifier for this component.
            target_flow_kg_h (float): Target outlet flow rate (kg/h).
            makeup_temp_c (float): Default makeup water temperature (°C).
                Used as fallback if no makeup stream received.
            makeup_pressure_bar (float): Default outlet pressure (bar).
        """
        super().__init__()
        self.component_id = component_id
        self.target_flow_kg_h = target_flow_kg_h
        self.current_target_kg_h = target_flow_kg_h  # Dynamic target (updated by flow_setpoint)
        self.default_makeup_temp_k = makeup_temp_c + 273.15
        self.outlet_pressure_pa = makeup_pressure_bar * 1e5

        # Input Streams
        self.drain_stream: Optional[Stream] = None
        self.makeup_stream: Optional[Stream] = None
        
        # LUT Manager (set in initialize)
        self._lut_manager = None

        # State
        self.current_demand_kg_h = 0.0  # Deficit signal to send
        self.actual_makeup_kg_h = 0.0   # Physical water received
        self.actual_drain_kg_h = 0.0    # Recirculated water received
        
        # Output
        self.outlet_stream: Optional[Stream] = None

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """
        Prepare the component for simulation.

        Args:
            dt (float): Simulation timestep in hours.
            registry (ComponentRegistry): Central registry for component access.
        """
        super().initialize(dt, registry)
        if registry.has('lut_manager'):
            self._lut_manager = registry.get('lut_manager')

    def step(self, t: float) -> None:
        """
        Execute one simulation timestep.

        Process:
        1. Measure incoming recirculated flow (drain_in).
        2. Calculate demand deficit: demand = target - drain.
        3. Store demand for signal output (read by get_output).
        4. Mix recirculated + actual makeup received this step.
        5. Clear input buffers.

        Args:
            t (float): Current simulation time in hours.
        """
        super().step(t)

        # 1. Measure Drain (Recirculated) Input
        drain_flow = 0.0
        drain_temp_k = self.default_makeup_temp_k
        drain_pressure_pa = self.outlet_pressure_pa
        
        if self.drain_stream:
            drain_flow = self.drain_stream.mass_flow_kg_h
            drain_temp_k = self.drain_stream.temperature_k
            drain_pressure_pa = self.drain_stream.pressure_pa
            
        self.actual_drain_kg_h = drain_flow

        # 2. Calculate Demand (Deficit) for NEXT timestep signal
        # Use dynamic target (from electrolyzer setpoint if connected)
        deficit = self.current_target_kg_h - drain_flow
        self.current_demand_kg_h = max(0.0, deficit)

        # 3. Measure Makeup Water Actually Received THIS step
        # (This is the response to the signal we sent LAST step)
        makeup_flow = 0.0
        makeup_temp_k = self.default_makeup_temp_k
        makeup_pressure_pa = self.outlet_pressure_pa
        
        if self.makeup_stream:
            makeup_flow = self.makeup_stream.mass_flow_kg_h
            makeup_temp_k = self.makeup_stream.temperature_k
            makeup_pressure_pa = self.makeup_stream.pressure_pa
            
        self.actual_makeup_kg_h = makeup_flow

        # 4. Mixing Logic
        total_flow = drain_flow + makeup_flow
        
        if total_flow <= 0:
            self.outlet_stream = Stream(0.0)
            self._clear_inputs()
            return

        # Cap output at target (discard excess if drain > target)
        if total_flow > self.target_flow_kg_h:
            # Scale down flows proportionally to meet target
            scale = self.target_flow_kg_h / total_flow
            drain_flow *= scale
            makeup_flow *= scale
            total_flow = self.target_flow_kg_h
            logger.debug(
                f"SignalMakeupMixer {self.component_id}: "
                f"Excess flow, scaling down to target {self.target_flow_kg_h:.0f} kg/h"
            )

        # 5. Rigorous Enthalpy Mixing
        H_drain = 0.0
        H_makeup = 0.0
        
        if self._lut_manager:
            try:
                H_drain = self._lut_manager.lookup(
                    'H2O', 'H', drain_pressure_pa, drain_temp_k
                )
                H_makeup = self._lut_manager.lookup(
                    'H2O', 'H', makeup_pressure_pa, makeup_temp_k
                )
            except Exception:
                # Fallback: Cp model
                H_drain = 4184.0 * (drain_temp_k - 273.15)
                H_makeup = 4184.0 * (makeup_temp_k - 273.15)
        else:
            # Fallback: Cp model (Cp = 4184 J/kg·K for liquid water)
            H_drain = 4184.0 * (drain_temp_k - 273.15)
            H_makeup = 4184.0 * (makeup_temp_k - 273.15)

        # Mixed enthalpy
        H_mix = ((drain_flow * H_drain) + (makeup_flow * H_makeup)) / total_flow

        # Resolve temperature from enthalpy
        T_guess = (drain_flow * drain_temp_k + makeup_flow * makeup_temp_k) / total_flow
        T_mix = solve_water_T_from_H_jit(H_mix, self.outlet_pressure_pa, T_guess)

        # 6. Create Outlet Stream
        self.outlet_stream = Stream(
            mass_flow_kg_h=total_flow,
            temperature_k=T_mix,
            pressure_pa=self.outlet_pressure_pa,
            composition={'H2O': 1.0},
            phase='liquid'
        )

        # 7. Clear Input Buffers
        self._clear_inputs()

    def _clear_inputs(self) -> None:
        """Clear input stream buffers for next timestep."""
        self.drain_stream = None
        self.makeup_stream = None

    def receive_input(self, port_name: str, value: Any, resource_type: str = None) -> float:
        """
        Accept input streams at specified ports.

        Ports:
            drain_in: Recirculated water from loop
            makeup_water_in: Fresh water from tank/pump

        Args:
            port_name (str): Target port.
            value (Any): Input stream.
            resource_type (str, optional): Resource classification hint.

        Returns:
            float: Mass flow accepted (kg/h).
        """
        if port_name == 'drain_in' and isinstance(value, Stream):
            self.drain_stream = value
            return value.mass_flow_kg_h
            
        elif port_name == 'makeup_water_in' and isinstance(value, Stream):
            self.makeup_stream = value
            return value.mass_flow_kg_h
            
        elif port_name == 'flow_setpoint':
            # Dynamic setpoint from upstream electrolyzer
            if isinstance(value, Stream):
                self.current_target_kg_h = value.mass_flow_kg_h
            elif isinstance(value, (int, float)):
                self.current_target_kg_h = float(value)
            return 0.0
            
        return 0.0

    def get_output(self, port_name: str) -> Any:
        """
        Retrieve output streams from specified ports.

        Ports:
            mixture_out: Mixed water stream
            demand_signal: Signal Stream encoding deficit (kg/h)

        Args:
            port_name (str): Port identifier.

        Returns:
            Stream: Output stream or signal.
        """
        if port_name == 'mixture_out' or port_name == 'water_out':
            return self.outlet_stream
            
        elif port_name == 'demand_signal':
            # Wrap signal in a Stream for topology compatibility
            # Phase is 'signal' to indicate non-physical flow
            return Stream(
                mass_flow_kg_h=self.current_demand_kg_h,
                temperature_k=293.15,  # Placeholder
                pressure_pa=101325.0,  # Placeholder
                composition={'Signal': 1.0},
                phase='signal'
            )
            
        return None

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """
        Define the physical connection ports for this component.

        Returns:
            Dict[str, Dict[str, str]]: Port definitions.
        """
        return {
            'drain_in': {'type': 'input', 'resource_type': 'water', 'units': 'kg/h'},
            'makeup_water_in': {'type': 'input', 'resource_type': 'water', 'units': 'kg/h'},
            'flow_setpoint': {'type': 'input', 'resource_type': 'signal', 'units': 'kg/h'},
            'mixture_out': {'type': 'output', 'resource_type': 'water', 'units': 'kg/h'},
            'demand_signal': {'type': 'output', 'resource_type': 'signal', 'units': 'kg/h'}
        }

    def get_state(self) -> Dict[str, Any]:
        """
        Retrieve the component's current operational state.

        Returns:
            Dict[str, Any]: State dictionary with demand and flow metrics.
        """
        return {
            **super().get_state(),
            'target_flow_kg_h': self.target_flow_kg_h,
            'current_target_kg_h': self.current_target_kg_h,
            'current_demand_kg_h': self.current_demand_kg_h,
            'actual_makeup_kg_h': self.actual_makeup_kg_h,
            'actual_drain_kg_h': self.actual_drain_kg_h,
            'outlet_flow_kg_h': self.outlet_stream.mass_flow_kg_h if self.outlet_stream else 0.0,
            'outlet_temp_c': (self.outlet_stream.temperature_k - 273.15) if self.outlet_stream else 0.0
        }
