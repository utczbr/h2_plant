"""
Oxygen Makeup "Source" Component.

This component implements a demand-driven oxygen supply logic.
It acts as a pass-through node that monitors the incoming oxygen flow
and injects supplemental oxygen ("makeup") to ensure a minimum target flow is met.

Topology Placement:
Typically placed *after* a mixer that combines various production sources (e.g. SOEC + PEM).
This ensures the final output stream meets the required demand (e.g. for ATR).

Logic (Updated v2: Min/Max Limits):
1. Receive `inlet_stream` (Arriving O2).
2. If arriving_flow < min_target: Inject makeup. Output = min_target.
3. If min_target <= arriving_flow <= max_limit: Pass through. Output = arriving_flow.
4. If arriving_flow > max_limit: Vent excess. Output = max_limit.
"""

from typing import Dict, Any, Optional
from h2_plant.core.component import Component
from h2_plant.core.stream import Stream
from h2_plant.core.constants import GasConstants

class OxygenMakeupNode(Component):
    """
    Inline compensatory oxygen supply with min/max limiting.
    
    Guarantees a minimum output flow rate by topping up the inlet flow with
    supplemental oxygen from an infinite virtual source. Also caps output
    at a maximum limit, venting surplus production.
    """
    
    def __init__(
        self,
        component_id: str,
        target_flow_kg_h: float = None,  # Legacy: alias for min_target
        min_target_flow_kg_h: float = None,
        max_limit_flow_kg_h: float = None,
        supply_pressure_bar: float = 15.0,
        supply_temperature_c: float = 25.0,
        supply_purity: float = 0.995
    ):
        super().__init__()
        self.component_id = component_id
        
        # Resolve min/max from legacy or new params
        if min_target_flow_kg_h is not None:
            self.min_target_flow_kg_h = min_target_flow_kg_h
        elif target_flow_kg_h is not None:
            self.min_target_flow_kg_h = target_flow_kg_h
        else:
            self.min_target_flow_kg_h = 0.0  # No minimum guarantee
        
        if max_limit_flow_kg_h is not None:
            self.max_limit_flow_kg_h = max_limit_flow_kg_h
        else:
            # Default: same as min (original behavior: clamp to target)
            self.max_limit_flow_kg_h = self.min_target_flow_kg_h
        
        # Supply Conditions
        self.supply_pressure_pa = supply_pressure_bar * 1e5
        self.supply_temp_k = supply_temperature_c + 273.15
        self.supply_purity = supply_purity
        
        # State
        self.inlet_stream: Optional[Stream] = None
        self.outlet_stream: Optional[Stream] = None
        self.makeup_flow_kg_h: float = 0.0
        self.vented_flow_kg_h: float = 0.0  # Track discarded O2
        self._last_output_flow_kg_h: float = 0.0  # Exposed for downstream sync
        
        # Constants
        self.CP_O2 = GasConstants.CP_O2_AVG 

    def initialize(self, dt: float, registry: Any) -> None:
        super().initialize(dt, registry)

    def receive_input(self, port_name: str, value: Any, resource_type: str = None) -> float:
        if port_name in ['inlet', 'inlet_stream', 'o2_in']:
            if isinstance(value, Stream):
                self.inlet_stream = value
                return value.mass_flow_kg_h
        return 0.0

    def step(self, t: float) -> None:
        super().step(t)
        
        # 1. Measure Arriving Flow
        arriving_flow = 0.0
        arriving_temp = self.supply_temp_k
        
        if self.inlet_stream:
            arriving_flow = self.inlet_stream.mass_flow_kg_h
            arriving_temp = self.inlet_stream.temperature_k
            
        # 2. Apply Min/Max Logic
        self.makeup_flow_kg_h = 0.0
        self.vented_flow_kg_h = 0.0
        
        if arriving_flow < self.min_target_flow_kg_h:
            # Below minimum: inject makeup
            self.makeup_flow_kg_h = self.min_target_flow_kg_h - arriving_flow
            total_out = self.min_target_flow_kg_h
        elif arriving_flow > self.max_limit_flow_kg_h:
            # Above maximum: vent excess
            self.vented_flow_kg_h = arriving_flow - self.max_limit_flow_kg_h
            total_out = self.max_limit_flow_kg_h
        else:
            # Within band: pass through
            total_out = arriving_flow
            
        self._last_output_flow_kg_h = total_out
            
        if total_out <= 1e-9:
            self.outlet_stream = Stream(0.0)
            return

        # 3. Mixing Calculation (Enthalpy Balance)
        if self.makeup_flow_kg_h > 0:
            mass_from_inlet = arriving_flow
        else:
            mass_from_inlet = total_out
            
        T_mix = ((mass_from_inlet * arriving_temp) + (self.makeup_flow_kg_h * self.supply_temp_k)) / total_out
        P_out = self.supply_pressure_pa
        
        # Composition
        supply_comp = {'O2': self.supply_purity, 'H2O': 1.0 - self.supply_purity}
        inlet_comp = self.inlet_stream.composition if self.inlet_stream else {'O2': 1.0}
        
        final_comp = {}
        all_species = set(inlet_comp.keys()) | set(supply_comp.keys())
        
        for s in all_species:
            m_s_in = min(mass_from_inlet, arriving_flow) * inlet_comp.get(s, 0.0)
            m_s_mk = self.makeup_flow_kg_h * supply_comp.get(s, 0.0)
            final_comp[s] = (m_s_in + m_s_mk) / total_out
            
        self.outlet_stream = Stream(
            mass_flow_kg_h=total_out,
            temperature_k=T_mix,
            pressure_pa=P_out,
            composition=final_comp,
            phase='gas'
        )
        
        # Reset input
        self.inlet_stream = None

    def get_output(self, port_name: str) -> Any:
        if port_name in ['outlet', 'o2_out']:
            return self.outlet_stream
        return None
    
    def get_output_mass_flow(self) -> float:
        """Return the last computed output mass flow (kg/h) for downstream sync."""
        return self._last_output_flow_kg_h

    def get_state(self) -> Dict[str, Any]:
        return {
            **super().get_state(),
            'min_target_flow_kg_h': self.min_target_flow_kg_h,
            'max_limit_flow_kg_h': self.max_limit_flow_kg_h,
            'makeup_flow_kg_h': self.makeup_flow_kg_h,
            'vented_flow_kg_h': self.vented_flow_kg_h,
            'output_flow_kg_h': self._last_output_flow_kg_h,
            'arriving_flow_kg_h': self.outlet_stream.mass_flow_kg_h - self.makeup_flow_kg_h if self.outlet_stream else 0.0
        }

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        return {
            'inlet': {'type': 'input', 'resource_type': 'oxygen', 'units': 'kg/h'},
            'outlet': {'type': 'output', 'resource_type': 'oxygen', 'units': 'kg/h'}
        }

