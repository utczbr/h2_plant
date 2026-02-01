"""
Consumer/Refueling Station.
"""

from h2_plant.core.component import Component
from typing import Dict, Any

class Consumer(Component):
    """
    Hydrogen Consumer (e.g., Refueling Station).
    """
    def __init__(self, num_bays: int, filling_rate_kg_h: float):
        super().__init__()
        self.num_bays = num_bays
        self.filling_rate_kg_h = filling_rate_kg_h
        
        # State
        self.h2_consumed_kg_h = 0.0
        self.active_bays = 0

    def step(self, t: float) -> None:
        super().step(t)
        # Demand logic would be controlled externally
        self.h2_consumed_kg_h = self.active_bays * self.filling_rate_kg_h

    def get_state(self) -> Dict[str, Any]:
        return {
            **super().get_state(),
            'h2_consumed_kg_h': self.h2_consumed_kg_h,
            'active_bays': self.active_bays
        }
