from typing import Dict, Any
from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.stream import Stream

class UltrapureWaterStorageTank(Component):
    """5000L ultrapure water storage with dual inputs."""
    
    def __init__(self, capacity_l: float = 5000.0, initial_fill_ratio: float = 0.5):
        super().__init__()
        self.capacity_l = capacity_l
        self.capacity_kg = capacity_l  # Assuming water density ~1 kg/L
        self.current_mass_kg = self.capacity_kg * initial_fill_ratio
        self.temperature_c = 20.0
        self.pressure_bar = 1.0
        
    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        super().initialize(dt, registry)
        if self.capacity_l <= 0:
            raise ValueError("Tank capacity must be positive")
    
    def fill(self, mass_kg: float) -> float:
        """Add water to tank, return actual amount added."""
        available = max(self.capacity_kg - self.current_mass_kg, 0.0)
        actual = min(mass_kg, available)
        self.current_mass_kg += actual
        return actual
    
    def withdraw(self, mass_kg: float) -> float:
        """Remove water from tank, return actual amount removed."""
        actual = min(mass_kg, self.current_mass_kg)
        self.current_mass_kg -= actual
        return actual
    
    def step(self, t: float) -> None:
        super().step(t)
        # In a real simulation, flow would be managed by a coordinator
        # or by pulling from connected components' states.
        pass
        
    def get_state(self) -> Dict[str, Any]:
        fill_ratio = self.current_mass_kg / self.capacity_kg if self.capacity_kg > 0 else 0.0
        return {
            **super().get_state(),
            "current_mass_kg": float(self.current_mass_kg),
            "capacity_kg": float(self.capacity_kg),
            "fill_ratio": float(fill_ratio),
            "temperature_c": float(self.temperature_c),
            "pressure_bar": float(self.pressure_bar),
            "flows": {
                "inputs": {
                    "from_treatment": {
                        "value": 0.0,
                        "unit": "kg/h",
                        "source": "water_treatment",
                        "flowtype": "WATER_MASS"
                    },
                    "from_soec": {
                        "value": 0.0,
                        "unit": "kg/h",
                        "source": "soec_electrolyzer",
                        "flowtype": "WATER_MASS"
                    }
                },
                "outputs": {
                    "to_pump_a": {
                        "value": 0.0,
                        "unit": "kg/h",
                        "destination": "pump_a",
                        "flowtype": "WATER_MASS"
                    },
                    "to_pump_b": {
                        "value": 0.0,
                        "unit": "kg/h",
                        "destination": "pump_b",
                        "flowtype": "WATER_MASS"
                    }
                }
            }
        }

    def get_output(self, port_name: str) -> Any:
        """Get output from specific port."""
        if port_name == 'water_out':
            return Stream(
                mass_flow_kg_h=self.current_mass_kg, # Available mass
                temperature_k=self.temperature_c + 273.15,
                pressure_pa=self.pressure_bar * 1e5,
                composition={'H2O': 1.0},
                phase='liquid'
            )
        else:
            raise ValueError(f"Unknown output port '{port_name}' on {self.component_id}")

    def receive_input(self, port_name: str, value: Any, resource_type: str) -> float:
        """Receive input."""
        if port_name == 'water_in':
            if isinstance(value, Stream):
                # Add to tank
                added = self.fill(value.mass_flow_kg_h)
                return added
        return 0.0

    def extract_output(self, port_name: str, amount: float, resource_type: str) -> None:
        """Acknowledge extraction."""
        if port_name == 'water_out':
            self.withdraw(amount)

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """Return port metadata."""
        return {
            'water_in': {'type': 'input', 'resource_type': 'water', 'units': 'kg/h'},
            'water_out': {'type': 'output', 'resource_type': 'water', 'units': 'kg/h'}
        }
