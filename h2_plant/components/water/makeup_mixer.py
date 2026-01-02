"""
Makeup Water Mixer Component.

This component implements a demand-driven mixing node for water recirculation loops.
It acts as an "Infinite Source" for fresh water, supplying exactly the difference
between the specific target flow and the available recycled drain flow.

Control Logic:
    1. **Feedback**: Measures incoming mass flow from `drain_in` (m_recycled).
    2. **Setpoint**: Compares against `target_flow_kg_h` (m_target).
    3. **Action**: Injects `m_makeup = max(0, m_target - m_recycled)`.
    4. **Mixing**: Computes mass-weighted average temperature for the outlet.
"""

from typing import Dict, Any, Optional
import numpy as np

from h2_plant.core.component import Component
from h2_plant.core.stream import Stream

class MakeupMixer(Component):
    """
    Active mixing node for water loop inventory management.

    Unlike a passive mixer, this component dynamically varies its input
    (generating mass from the 'makeup' virtual source) to guarantee a
    fixed outlet flow rate.

    Physics Model (Adiabatic Mixing):
        H_out * m_out = (H_drain * m_drain) + (H_makeup * m_makeup)
        (Simplified to Temperature mixing assuming constant Cp for liquid water)

    Attributes:
        target_flow_kg_h (float): Flow control setpoint (kg/h).
        makeup_temp_k (float): Temperature of the fresh water supply (K).
        makeup_pressure_pa (float): Supply pressure (Pa).
    """

    def __init__(
        self,
        component_id: str,
        target_flow_kg_h: float,
        makeup_temp_c: float = 20.0,
        makeup_pressure_bar: float = 1.0
    ):
        super().__init__()
        self.component_id = component_id
        self.target_flow_kg_h = target_flow_kg_h
        self.makeup_temp_k = makeup_temp_c + 273.15
        self.makeup_pressure_pa = makeup_pressure_bar * 1e5

        # Inputs
        self.drain_stream: Optional[Stream] = None
        self._water_consumed_kg_h: float = 0.0  # From electrolyzers
        
        # Outputs
        self.outlet_stream: Optional[Stream] = None
        self.makeup_flow_kg_h: float = 0.0

    def initialize(self, dt: float, registry: 'ComponentRegistry') -> None:
        """
        Executes initialization phase of Component Lifecycle.
        
        Args:
            dt (float): Simulation timestep (hours).
            registry (ComponentRegistry): Central service registry.
        """
        super().initialize(dt, registry)

    def step(self, t: float) -> None:
        """
        Executes mass and energy balance calculations.

        Process:
        1. measure `drain_in` mass flow.
        2. Calculate deficiency: `makeup = target - drain`.
        3. Enforce non-return flow: `makeup = max(0, makeup)`.
        4. Mix streams assuming isobaric, adiabatic conditions.

        Args:
            t (float): Simulation time (hours).
        """
        super().step(t)

        # 1. Determine Drain Input
        drain_flow = 0.0
        drain_temp = self.makeup_temp_k
        drain_h = 0.0 # ref 0 at 0K implies CP model, but let's use weighted T

        if self.drain_stream:
            drain_flow = self.drain_stream.mass_flow_kg_h
            drain_temp = self.drain_stream.temperature_k
        
        # 2. Calculate Makeup Requirement
        # Logic: Supply deficiency to meet target.
        # If Drain > Target, we have excess inventory (makeup=0).
        self.makeup_flow_kg_h = max(0.0, self.target_flow_kg_h - drain_flow)
        
        # Clamp total output to target flow (Overflow logic)
        # If drain > target, we discharge excess to avoid loop accumulation.
        # This ensures Feed Pump never receives > Target.
        total_flow = min(self.target_flow_kg_h, drain_flow + self.makeup_flow_kg_h)
        
        if total_flow <= 0:
             self.outlet_stream = Stream(0.0)
             return

        # 3. Mix Streams (Mass-Weighted Temperature/Enthalpy)
        # Fix: Only account for mass that actually enters the outlet (discard overflow energy)
        used_drain_flow = total_flow - self.makeup_flow_kg_h
        
        weighted_T_sum = (used_drain_flow * drain_temp) + (self.makeup_flow_kg_h * self.makeup_temp_k)
        T_mix = weighted_T_sum / total_flow
        
        # Pressure equalization (lowest of the inputs, usually atmospheric for drain)
        P_mix = self.makeup_pressure_pa
        if self.drain_stream:
             P_mix = min(P_mix, self.drain_stream.pressure_pa)

        self.outlet_stream = Stream(
            mass_flow_kg_h=total_flow,
            temperature_k=T_mix,
            pressure_pa=P_mix,
            composition={'H2O': 1.0},
            phase='liquid'
        )
        
        # Reset input buffer
        self.drain_stream = None

    def receive_input(self, port_name: str, value: Any, resource_type: str = None) -> float:
        if port_name == 'drain_in' and isinstance(value, Stream):
            self.drain_stream = value
            return value.mass_flow_kg_h
        elif port_name == 'consumption_in':
            # Total water consumed by electrolyzers (kg/h)
            if isinstance(value, (int, float)):
                self._water_consumed_kg_h = float(value)
                return float(value)
        return 0.0

    def get_output(self, port_name: str) -> Any:
        if port_name == 'water_out':
            return self.outlet_stream
        return None

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        return {
            'drain_in': {'type': 'input', 'resource_type': 'water', 'units': 'kg/h'},
            'consumption_in': {'type': 'input', 'resource_type': 'signal', 'units': 'kg/h'},
            'water_out': {'type': 'output', 'resource_type': 'water', 'units': 'kg/h'}
        }

    def get_state(self) -> Dict[str, Any]:
        return {
            **super().get_state(),
            'target_flow_kg_h': self.target_flow_kg_h,
            'makeup_flow_kg_h': self.makeup_flow_kg_h,
            'water_consumed_kg_h': self._water_consumed_kg_h
        }
