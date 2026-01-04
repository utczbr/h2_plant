"""
Oxygen Makeup "Source" Component.

This component implements a demand-driven oxygen supply logic.
It acts as a pass-through node that monitors the incoming oxygen flow
and injects supplemental oxygen ("makeup") to ensure a minimum target flow is met.

Topology Placement:
Typically placed *after* a mixer that combines various production sources (e.g. SOEC + PEM).
This ensures the final output stream meets the required demand (e.g. for ATR).

Logic:
1. Receive `inlet_stream` (Arriving O2).
2. Calculate `makeup_mass = max(0, target_flow - arriving_flow)`.
3. Generate makeup stream at configured P/T.
4. Mix streams (Enthalpy Balance) -> `outlet_stream`.
"""

from typing import Dict, Any, Optional
from h2_plant.core.component import Component
from h2_plant.core.stream import Stream
from h2_plant.core.constants import GasConstants

class OxygenMakeupNode(Component):
    """
    Inline compensatory oxygen supply.
    
    Guarantees a minimum output flow rate by topping up the inlet flow with
    supplemental oxygen from an infinite virtual source.
    """
    
    def __init__(
        self,
        component_id: str,
        target_flow_kg_h: float,
        supply_pressure_bar: float = 15.0,
        supply_temperature_c: float = 25.0,
        supply_purity: float = 0.995
    ):
        super().__init__()
        self.component_id = component_id
        self.target_flow_kg_h = target_flow_kg_h
        
        # Supply Conditions
        self.supply_pressure_pa = supply_pressure_bar * 1e5
        self.supply_temp_k = supply_temperature_c + 273.15
        self.supply_purity = supply_purity
        
        # State
        self.inlet_stream: Optional[Stream] = None
        self.outlet_stream: Optional[Stream] = None
        self.makeup_flow_kg_h: float = 0.0
        
        # Constants
        # Use average Cp for mixing approximation (O2 dominated)
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
        arriving_h = 0.0
        
        if self.inlet_stream:
            arriving_flow = self.inlet_stream.mass_flow_kg_h
            arriving_temp = self.inlet_stream.temperature_k
            
        # 2. Calculate Makeup (Deficiency)
        if arriving_flow >= self.target_flow_kg_h:
            self.makeup_flow_kg_h = 0.0
            total_out = arriving_flow # Pass through excess
        else:
            self.makeup_flow_kg_h = self.target_flow_kg_h - arriving_flow
            total_out = self.target_flow_kg_h
            
        if total_out <= 1e-9:
            self.outlet_stream = Stream(0.0)
            return

        # 3. Mixing Calculation (Enthalpy Balance)
        # H_mix = (m_arr * H_arr + m_mk * H_mk) / m_total
        # H approx = Cp * (T - T_ref)
        # Therefore: T_mix = (m_arr * T_arr + m_mk * T_mk) / m_total (assuming constant Cp)
        
        # This constant Cp assumption is valid for O2 mixing with O2 at similar conditions
        T_mix = ((arriving_flow * arriving_temp) + (self.makeup_flow_kg_h * self.supply_temp_k)) / total_out
        
        # Pressure:
        # The node acts as a manifold. If supply is higher pressure, it dominates?
        # Simplified: Output pressure = Supply Pressure (assuming regulated supply)
        # OR Output pressure = Inlet Pressure (if pass through?)
        # Let's assume the node regulates to `supply_pressure_bar` if makeup is active,
        # or matches inlet if inlet is sufficient?
        # For robustness: use supply_pressure_pa as the regulated downstream setpoint
        P_out = self.supply_pressure_pa
        
        # Composition:
        # Mix composition weighted by mass
        # Supply Comp
        supply_comp = {'O2': self.supply_purity, 'H2O': 1.0 - self.supply_purity} # Simplified impurity
        
        # Inlet Comp
        inlet_comp = self.inlet_stream.composition if self.inlet_stream else {'O2': 1.0}
        
        final_comp = {}
        all_species = set(inlet_comp.keys()) | set(supply_comp.keys())
        
        for s in all_species:
            m_s_in = arriving_flow * inlet_comp.get(s, 0.0)
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

    def get_state(self) -> Dict[str, Any]:
        return {
            **super().get_state(),
            'target_flow_kg_h': self.target_flow_kg_h,
            'makeup_flow_kg_h': self.makeup_flow_kg_h,
            'arriving_flow_kg_h': self.outlet_stream.mass_flow_kg_h - self.makeup_flow_kg_h if self.outlet_stream else 0.0
        }

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        return {
            'inlet': {'type': 'input', 'resource_type': 'oxygen', 'units': 'kg/h'},
            'outlet': {'type': 'output', 'resource_type': 'oxygen', 'units': 'kg/h'}
        }
