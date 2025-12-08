from typing import Dict, Any
from h2_plant.core.component import Component
from h2_plant.core.stream import Stream

class PSAUnit(Component):
    def __init__(self, component_id: str, gas_type: str):
        super().__init__()
        self.component_id = component_id
        self.gas_type = gas_type
        self.feed_gas_kg_h = 0.0
        self.product_gas_kg_h = 0.0
        self.input_stream = None

    def step(self, t: float) -> None:
        super().step(t)
        self.product_gas_kg_h = self.feed_gas_kg_h * 0.9 # 90% recovery

    def get_state(self) -> Dict[str, Any]:
        return {
            **super().get_state(),
            "feed_gas_kg_h": self.feed_gas_kg_h,
            "product_gas_kg_h": self.product_gas_kg_h
        }

    def get_output(self, port_name: str) -> Any:
        """Get output from specific port."""
        if port_name == 'h2_out':
            # Return purified gas stream
            if hasattr(self, 'input_stream') and self.input_stream:
                 return Stream(
                    mass_flow_kg_h=self.product_gas_kg_h,
                    temperature_k=self.input_stream.temperature_k,
                    pressure_pa=self.input_stream.pressure_pa * 0.98, # Slight pressure drop
                    composition={self.gas_type: 1.0}, # Pure gas
                    phase='gas'
                )
            else:
                return Stream(0.0)
        elif port_name == 'tail_gas_out':
            # Tail gas (impurities + some H2)
            if hasattr(self, 'input_stream') and self.input_stream:
                tail_mass = self.feed_gas_kg_h - self.product_gas_kg_h
                return Stream(
                    mass_flow_kg_h=tail_mass,
                    temperature_k=self.input_stream.temperature_k,
                    pressure_pa=101325.0, # Vented or low pressure
                    composition={'H2': 0.5, 'Impurities': 0.5}, # Simplified
                    phase='gas'
                )
            else:
                return Stream(0.0)
        else:
            raise ValueError(f"Unknown output port '{port_name}' on {self.component_id}")

    def receive_input(self, port_name: str, value: Any, resource_type: str) -> float:
        """Receive input."""
        if port_name == 'h2_in':
            if isinstance(value, Stream):
                self.input_stream = value
                self.feed_gas_kg_h = value.mass_flow_kg_h
                self.product_gas_kg_h = self.feed_gas_kg_h * 0.9 # 90% recovery
                return self.feed_gas_kg_h
        return 0.0

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """Return port metadata."""
        return {
            'h2_in': {'type': 'input', 'resource_type': 'hydrogen', 'units': 'kg/h'},
            'h2_out': {'type': 'output', 'resource_type': 'hydrogen', 'units': 'kg/h'},
            'tail_gas_out': {'type': 'output', 'resource_type': 'tail_gas', 'units': 'kg/h'}
        }
