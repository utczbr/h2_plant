
from typing import Dict, Any, Optional
from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.stream import Stream

class WaterPurifier(Component):
    """
    Water Purifier (WP) Component.
    Receives raw water and outputs ultra-pure water.
    Now uses Stream for thermodynamic tracking.
    """
    def __init__(self, component_id: str, max_flow_kg_h: float, power_per_kg: float = 0.005):
        super().__init__()
        self.component_id = component_id
        self.max_flow_kg_h = max_flow_kg_h
        self.power_per_kg = power_per_kg # kWh/kg specific energy

        # Inputs
        self.inlet_stream: Optional[Stream] = None
        
        # Outputs
        self.outlet_stream: Optional[Stream] = None
        self.power_consumption_kw = 0.0
        self.efficiency = 0.98 # 2% loss as wastewater

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        super().initialize(dt, registry)

    def step(self, t: float) -> None:
        super().step(t)
        
        # Handle legacy or Stream input
        if self.inlet_stream is None:
            self.outlet_stream = None
            self.power_consumption_kw = 0.0
            return
        
        # Limit flow to capacity
        processed_flow = min(self.inlet_stream.mass_flow_kg_h, self.max_flow_kg_h)
        
        # Calculate output stream
        # Purification: slight pressure drop, temperature maintained
        self.outlet_stream = Stream(
            mass_flow_kg_h=processed_flow * self.efficiency,
            temperature_k=self.inlet_stream.temperature_k,
            pressure_pa=self.inlet_stream.pressure_pa * 0.95, # 5% pressure drop
            composition={'H2O': 1.0}, # Ultra-pure water
            phase='liquid'
        )
        
        self.power_consumption_kw = processed_flow * self.power_per_kg

    def get_state(self) -> Dict[str, Any]:
        state = {
            **super().get_state(),
            'component_id': self.component_id,
            'power_kw': self.power_consumption_kw,
        }
        
        if self.outlet_stream:
            state['streams'] = {
                'out': {
                    'mass_flow': self.outlet_stream.mass_flow_kg_h,
                    'temperature': self.outlet_stream.temperature_k,
                    'pressure': self.outlet_stream.pressure_pa
                }
            }
            state['outlet_flow_kg_h'] = self.outlet_stream.mass_flow_kg_h
            state['temperature_c'] = self.outlet_stream.temperature_k - 273.15
            state['pressure_bar'] = self.outlet_stream.pressure_pa / 1e5
        
        return state
