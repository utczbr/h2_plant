
from typing import Dict, Any
from h2_plant.core.component import Component

class WGSReactor(Component):
    def __init__(self, component_id: str, conversion_rate: float):
        super().__init__()
        self.component_id = component_id
        self.conversion_rate = conversion_rate
        self.syngas_input_kg_h = 0.0
        self.syngas_output_kg_h = 0.0

    def initialize(self, dt: float, registry: Any) -> None:
        super().initialize(dt, registry)

    def step(self, t: float) -> None:
        super().step(t)
        self.syngas_output_kg_h = self.syngas_input_kg_h

    def get_state(self) -> Dict[str, Any]:
        return {**super().get_state(), 'component_id': self.component_id, 'syngas_output_kg_h': self.syngas_output_kg_h}

    def get_output(self, port_name: str) -> Any:
        if port_name in ['syngas_out', 'shifted_gas_out', 'out']:
            from h2_plant.core.stream import Stream
            return Stream(mass_flow_kg_h=self.syngas_output_kg_h)
        elif port_name == 'heat_out':
            return 0.0
        else:
            raise ValueError(f"Unknown output port '{port_name}' on {self.component_id}")

    def receive_input(self, port_name: str, value: Any, resource_type: str) -> float:
        if port_name in ['syngas_in', 'in', 'cooled_gas_in']:
            if hasattr(value, 'mass_flow_kg_h'):
                self.syngas_input_kg_h = value.mass_flow_kg_h
            else:
                self.syngas_input_kg_h = float(value)
            return self.syngas_input_kg_h
        return 0.0
