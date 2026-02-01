"""
ATR Recovery Components.
"""
from typing import Dict, Any
from h2_plant.core.stream import Stream
from h2_plant.components.reforming.atr_data_manager import ATRBaseComponent, C_TO_K

class ATRSyngasCooler(ATRBaseComponent):
    """
    Syngas Heat Recovery Exchanger (H05).
    Cools the stream from the Integrated Plant to the Dry Cooler inlet temp.
    """
    
    def __init__(self, component_id: str = None, lookup_id: str = "Tin_H05", efficiency: float = 0.95):
        super().__init__()
        self.component_id = component_id
        self.lookup_id = lookup_id
        self.syngas_in = None
        self.syngas_out = None
        self.duty_kw = 0.0

    def receive_input(self, port_name: str, value: Any, resource_type: str = None) -> float:
        if port_name == 'syngas_in' and isinstance(value, Stream):
            self.syngas_in = value
            return value.mass_flow_kg_h
        return 0.0

    def step(self, t: float) -> None:
        if not self.syngas_in or self.syngas_in.mass_flow_kg_h <= 0:
            self.syngas_out = Stream(0.0)
            return

        # 1. Infer Load (Inverse lookup not strictly needed if we trust upstream mass)
        # We just need to cool it to the target temperature defined in the regression model
        # Or simply act as a heat exchanger.
        
        # For simplicity in this cleaned architecture:
        # We pass the stream through, but update its temperature to the 'Tout_H05' value
        # from the regression, which is the inlet to the DryCooler.
        
        # Infer O2 from mass flow (approximate inverse of Fm_offgas_func)
        # mass = 948.5 * (o2 / 7.19) approx linear
        mass_flow = self.syngas_in.mass_flow_kg_h
        inferred_o2 = (mass_flow / 3130.0) * 23.75 # Linear scaling based on max points
        inferred_o2 = max(7.125, min(23.75, inferred_o2))
        
        # Lookup Target Outlet Temp
        target_temp_c = self.data_manager.lookup('Tout_H05_func', inferred_o2)
        
        # Create Output Stream
        self.syngas_out = self.syngas_in.copy()
        self.syngas_out.temperature_k = target_temp_c + C_TO_K
        
        # Calculate Duty (Simplified Q = m * Cp * dT for reporting)
        # Real sim uses enthalpy, but this is a cleanup.
        cp_approx = 2.0 # kJ/kg.K for syngas
        dt = self.syngas_in.temperature_k - self.syngas_out.temperature_k
        self.duty_kw = mass_flow * (1/3600) * cp_approx * dt

    def get_output(self, port_name: str) -> Any:
        if port_name == 'syngas_out':
            return self.syngas_out
        return None

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        return {
            'syngas_in': {'type': 'input', 'resource_type': 'stream'},
            'syngas_out': {'type': 'output', 'resource_type': 'stream'}
        }
        
    def get_state(self) -> Dict[str, Any]:
        return {
            "component_id": self.component_id,
            "duty_kw": self.duty_kw,
            "outlet_temp_c": (self.syngas_out.temperature_k - C_TO_K) if self.syngas_out else 0.0
        }
