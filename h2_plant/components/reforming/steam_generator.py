
from typing import Dict, Any
from h2_plant.core.component import Component

class SteamGenerator(Component):
    def __init__(self, component_id: str, max_flow_kg_h: float):
        super().__init__()
        self.component_id = component_id
        self.max_flow_kg_h = max_flow_kg_h
        self.water_input_kg_h = 0.0
        self.steam_output_kg_h = 0.0
        self.heat_input_kw = 0.0

    def step(self, t: float) -> None:
        super().step(t)
        self.steam_output_kg_h = self.water_input_kg_h
        self.heat_input_kw = self.steam_output_kg_h * 0.6 # Dummy factor

    def get_state(self) -> Dict[str, Any]:
        return {**super().get_state(), 'component_id': self.component_id, 'steam_output_kg_h': self.steam_output_kg_h}
