"""
Regenerative Heat Exchanger (Interchanger).

This component models a counter-flow heat exchanger designed for waste heat recovery.
It transfers thermal energy from a "Hot" stream to a "Cold" stream, subject to:
1. **Conservation of Energy**: Q_hot = Q_cold = Q_transferred
2. **Second Law of Thermodynamics**: Heat flows only from hot to cold (limited by approach temp).

Applications:
    - Pre-heating electrolysis feedwater using stack exhaust.
    - recuperating heat from compressor inter-stage cooling.
"""

from typing import Dict, Any, Optional
import numpy as np

from h2_plant.core.component import Component
from h2_plant.core.stream import Stream
from h2_plant.core.constants import ConversionFactors

class Interchanger(Component):
    """
    Simulates a counter-flow heat exchanger with specified minimum approach temperature.
    
    This model determines the maximum realizable heat transfer rate given inlet 
    conditions and the "Pinch Point" constraint (min_approach_temp).

    Physics Model:
        Q = min(Q_capacity, Q_availability)
    
    Attributes:
        min_approach_temp_k (float): Minimum allowed temperature difference (T_hot_out - T_cold_in).
                                     Represents the practical limit of heat exchanger surface area.
        efficiency (float): Adiabatic efficiency factor (heat loss to environment).
        target_cold_out_temp_k (float): Temperature setpoint for the cold stream.
    """

    def __init__(
        self,
        component_id: str,
        min_approach_temp_k: float = 10.0,
        target_cold_out_temp_c: float = 95.0,
        efficiency: float = 0.95
    ):
        super().__init__()
        self.component_id = component_id
        self.min_approach_temp_k = min_approach_temp_k
        self.target_cold_temp_k = target_cold_out_temp_c + 273.15
        self.efficiency = efficiency

        # Inputs
        self.hot_stream: Optional[Stream] = None
        self.cold_stream: Optional[Stream] = None

        # Outputs
        self.hot_out: Optional[Stream] = Stream(0.0, temperature_k=298.15, pressure_pa=101325.0, phase='gas')
        self.cold_out: Optional[Stream] = Stream(0.0, temperature_k=target_cold_out_temp_c+273.15, pressure_pa=101325.0, phase='liquid')
        
        self.q_transferred_kw = 0.0

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
        Calculates heat transfer and updates stream states.

        Calculation Logic:
        1. **Demand (Cold Side)**: Energy required to heat cold stream to target T.
           `Q_demand = ṁ_c * Cp_c * (T_target - T_c_in)`
           
        2. **Availability (Hot Side)**: Maximum energy extractable without violating 
           the Second Law (Approach Temperature constraint).
           `T_h_out_min = T_c_in + ΔT_approach`
           `Q_avail = ṁ_h * Cp_h * (T_h_in - T_h_out_min)`
           
        3. **Equilibrium**: `Q_transferred = min(Q_demand, Q_avail)`
        
        Args:
            t (float): Simulation time (hours).
        """
        super().step(t)

        if not self.hot_stream or not self.cold_stream:
             # Pass-through or Zero output if missing input
             self.hot_out = self.hot_stream
             self.cold_out = self.cold_stream
             self.q_transferred_kw = 0.0
             return

        # Properties
        m_h = self.hot_stream.mass_flow_kg_h / 3600.0
        T_h_in = self.hot_stream.temperature_k
        h_h_in = self.hot_stream.specific_enthalpy_j_kg
        
        m_c = self.cold_stream.mass_flow_kg_h / 3600.0
        T_c_in = self.cold_stream.temperature_k
        h_c_in = self.cold_stream.specific_enthalpy_j_kg
        
        if m_h <= 0 or m_c <= 0:
             self.hot_out = self.hot_stream
             self.cold_out = self.cold_stream
             return

        # 1. Cold Side Demand (Target T)
        # Simplify using average Cp for robustness, or Enthalpy if available
        # Ideally we iterate or use stream property calls, but for speed simplified:
        Cp_c = 4186.0 # Water J/kgK
        Q_cold_demand_w = m_c * Cp_c * (self.target_cold_temp_k - T_c_in)
        
        # 2. Hot Side Availability (Second Law Limit)
        # Hot stream cannot cool below Cold In + Approach
        T_h_out_min = T_c_in + self.min_approach_temp_k
        
        # If hot stream is already colder than limit, no transfer
        if T_h_in <= T_h_out_min:
             Q_hot_avail_w = 0.0
        else:
             # Estimated Cp for gas mixture (Steam/H2/O2 mix ~ 152C)
             # Cp mix approx 2000 J/kgK (Steam dominated)
             Cp_h = 2200.0 # Approx
             Q_hot_avail_w = m_h * Cp_h * (T_h_in - T_h_out_min)

        # 3. Actual Transfer
        Q_transfer_w = min(max(0, Q_cold_demand_w), max(0, Q_hot_avail_w))
        self.q_transferred_kw = Q_transfer_w / 1000.0

        # 4. Final States
        # Cold Out: T = T_in + Q / (m*Cp)
        T_c_out = T_c_in + (Q_transfer_w / (m_c * Cp_c))
        
        # Hot Out: T = T_in - Q / (m*Cp)
        T_h_out = T_h_in - (Q_transfer_w / (m_h * Cp_h))

        # Update Streams
        self.cold_out = Stream(
            mass_flow_kg_h=self.cold_stream.mass_flow_kg_h,
            temperature_k=T_c_out,
            pressure_pa=self.cold_stream.pressure_pa, # Ignore dP for now
            composition=self.cold_stream.composition,
            phase='liquid'
        )

        self.hot_out = Stream(
            mass_flow_kg_h=self.hot_stream.mass_flow_kg_h,
            temperature_k=T_h_out,
            pressure_pa=self.hot_stream.pressure_pa,
            composition=self.hot_stream.composition,
            phase='gas' # Assume phase change handled downstream in Cooler
        )
        
        # Clear inputs
        self.hot_stream = None
        self.cold_stream = None

    def receive_input(self, port_name: str, value: Any, resource_type: str = None) -> float:
        if port_name == 'hot_in' and isinstance(value, Stream):
            self.hot_stream = value
            return value.mass_flow_kg_h
        elif port_name == 'cold_in' and isinstance(value, Stream):
            self.cold_stream = value
            return value.mass_flow_kg_h
        return 0.0

    def get_output(self, port_name: str) -> Any:
        if port_name == 'hot_out':
            return self.hot_out if self.hot_out else Stream(0.0)
        elif port_name == 'cold_out':
            return self.cold_out if self.cold_out else Stream(0.0)
        return None

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        return {
            'hot_in': {'type': 'input', 'resource_type': 'gas'},
            'cold_in': {'type': 'input', 'resource_type': 'water'},
            'hot_out': {'type': 'output', 'resource_type': 'gas'},
            'cold_out': {'type': 'output', 'resource_type': 'water'}
        }

    def get_state(self) -> Dict[str, Any]:
        """
        Returns component operational telemetry.
        
        Returns:
            Dict[str, Any]: Q_transferred (kW) and outlet temperatures.
        """
        return {
            **super().get_state(),
            'q_transferred_kw': self.q_transferred_kw,
            'hot_out_temp_k': self.hot_out.temperature_k if self.hot_out else 0,
            'cold_out_temp_k': self.cold_out.temperature_k if self.cold_out else 0
        }
