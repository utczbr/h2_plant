"""
Proportional Makeup Mixer for ATR.

Extends the standard MakeupMixer with proportional control logic,
similar to BiogasSource. The target flow is computed dynamically
based on a reference component's output flow (typically the O2 makeup node).

Usage:
    Configured via topology with:
    - reference_component_id: The component to track (e.g., 'O2_Backup_Supply').
    - reference_ratio: Scale factor (water_max / o2_max).
    - OR reference_max_flow_kg_h: Uses max_flow_rate_kg_h / reference_max_flow_kg_h.
"""

from typing import Dict, Any, Optional
import numpy as np

from h2_plant.core.component import Component
from h2_plant.core.stream import Stream
from h2_plant.optimization.numba_ops import solve_water_T_from_H_jit


class ProportionalMakeupMixer(Component):
    """
    Active mixing node with proportional control for ATR water feed.
    
    The target flow rate is scaled dynamically based on a reference
    component's output flow (e.g., the Oxygen supply node).
    """
    
    def __init__(
        self,
        component_id: str,
        max_flow_rate_kg_h: float = 2331.95,  # H2O design max
        makeup_temp_c: float = 20.0,
        makeup_pressure_bar: float = 1.0,
        # Proportional Control (Required for sync)
        reference_component_id: str = None,
        reference_ratio: float = None,
        reference_max_flow_kg_h: float = None  # Max O2 flow for ratio calculation
    ):
        super().__init__()
        self.component_id = component_id
        self.max_flow_rate_kg_h = max_flow_rate_kg_h
        self.makeup_temp_k = makeup_temp_c + 273.15
        self.makeup_pressure_pa = makeup_pressure_bar * 1e5
        
        # Proportional control settings
        self.reference_component_id = reference_component_id
        self._reference_component = None  # Resolved at initialize
        
        if reference_ratio is not None:
            self.reference_ratio = reference_ratio
        elif reference_max_flow_kg_h is not None and reference_max_flow_kg_h > 0:
            self.reference_ratio = max_flow_rate_kg_h / reference_max_flow_kg_h
        else:
            self.reference_ratio = 1.0  # Fallback: 1:1 ratio

        # Inputs
        self.drain_stream: Optional[Stream] = None
        self._lut_manager = None
        
        # Outputs
        self.outlet_stream: Optional[Stream] = None
        self.makeup_flow_kg_h: float = 0.0
        self._current_target_kg_h: float = 0.0  # Dynamically computed

    def initialize(self, dt: float, registry: 'ComponentRegistry') -> None:
        super().initialize(dt, registry)
        
        # Resolve reference component
        if self.reference_component_id and registry:
            self._reference_component = registry.get(self.reference_component_id)
        
        if registry.has('lut_manager'):
            self._lut_manager = registry.get('lut_manager')

    def step(self, t: float) -> None:
        super().step(t)

        # 1. Determine Target Flow from Reference Component
        if self._reference_component and self.reference_ratio:
            ref_flow = getattr(self._reference_component, 'get_output_mass_flow', lambda: 0.0)()
            self._current_target_kg_h = min(ref_flow * self.reference_ratio, self.max_flow_rate_kg_h)
        else:
            self._current_target_kg_h = self.max_flow_rate_kg_h

        # 2. Determine Drain Input
        drain_flow = 0.0
        drain_temp = self.makeup_temp_k

        if self.drain_stream:
            drain_flow = self.drain_stream.mass_flow_kg_h
            drain_temp = self.drain_stream.temperature_k
        
        # 3. Calculate Makeup Requirement
        self.makeup_flow_kg_h = max(0.0, self._current_target_kg_h - drain_flow)
        
        # Clamp total output to current target
        total_flow = min(self._current_target_kg_h, drain_flow + self.makeup_flow_kg_h)
        
        if total_flow <= 0:
            self.outlet_stream = Stream(0.0)
            return

        # 4. Mix Streams (Rigorous Enthalpy Balance)
        used_drain_flow = total_flow - self.makeup_flow_kg_h
        
        H_drain = 0.0
        H_makeup = 0.0
        
        if self._lut_manager:
            try:
                H_drain = self._lut_manager.lookup('H2O', 'H', self.drain_stream.pressure_pa if self.drain_stream else self.makeup_pressure_pa, drain_temp)
                H_makeup = self._lut_manager.lookup('H2O', 'H', self.makeup_pressure_pa, self.makeup_temp_k)
            except:
                H_drain = 4184.0 * (drain_temp - 273.15)
                H_makeup = 4184.0 * (self.makeup_temp_k - 273.15)
        else:
            H_drain = 4184.0 * (drain_temp - 273.15)
            H_makeup = 4184.0 * (self.makeup_temp_k - 273.15)
             
        H_mix = ((used_drain_flow * H_drain) + (self.makeup_flow_kg_h * H_makeup)) / total_flow
        P_mix = self.makeup_pressure_pa
        T_mix = solve_water_T_from_H_jit(H_mix, P_mix, drain_temp)

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
        return 0.0

    def get_output(self, port_name: str) -> Any:
        if port_name == 'water_out':
            return self.outlet_stream
        return None

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        return {
            'drain_in': {'type': 'input', 'resource_type': 'water', 'units': 'kg/h'},
            'water_out': {'type': 'output', 'resource_type': 'water', 'units': 'kg/h'}
        }

    def get_state(self) -> Dict[str, Any]:
        return {
            **super().get_state(),
            'max_flow_rate_kg_h': self.max_flow_rate_kg_h,
            'current_target_kg_h': self._current_target_kg_h,
            'makeup_flow_kg_h': self.makeup_flow_kg_h,
            'reference_ratio': self.reference_ratio
        }
