from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class Stream:
    """
    Mock implementation of the Stream class from h2_plant.core.stream
    Standard units: Pa, K, kg/h
    """
    mass_flow_kg_h: float
    temperature_k: float
    pressure_pa: float
    composition: Dict[str, float]
    phase: str

    @property
    def mass_flow_kg_s(self) -> float:
        return self.mass_flow_kg_h / 3600.0

class ComponentRegistry:
    """Mock registry"""
    pass

class Component:
    """
    Mock implementation of the base Component class.
    """
    def __init__(self):
        self.dt = 1.0
        self.component_id = "generic_component"

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        self.dt = dt

    def step(self, t: float) -> None:
        pass

    def get_state(self) -> Dict[str, Any]:
        return {}

    def get_output(self, port_name: str) -> Any:
        raise NotImplementedError

    def receive_input(self, port_name: str, value: Any, resource_type: str) -> float:
        raise NotImplementedError

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        raise NotImplementedError