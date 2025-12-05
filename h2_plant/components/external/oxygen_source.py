"""
External oxygen source component.

Provides oxygen input from external suppliers, configurable by flow rate
or pressure-driven delivery.
"""

import logging
from typing import Dict, Any, Optional
from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.constants import GasConstants

logger = logging.getLogger(__name__)


class ExternalOxygenSource(Component):
    """
    External oxygen supply source.
    """
    
    def __init__(
        self,
        mode: str = "fixed_flow",
        flow_rate_kg_h: float = 0.0,
        pressure_bar: float = 5.0,
        cost_per_kg: float = 0.15,
        max_capacity_kg_h: float = 100.0
    ):
        super().__init__()
        
        self.mode = mode
        self.flow_rate_kg_h = flow_rate_kg_h
        self.pressure_bar = pressure_bar
        self.cost_per_kg = cost_per_kg
        self.max_capacity_kg_h = max_capacity_kg_h
        
        self.o2_output_kg = 0.0
        self.cumulative_o2_kg = 0.0
        self.cumulative_cost = 0.0
        self._target_component: Optional[Component] = None
    
    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """Initialize external oxygen source."""
        super().initialize(dt, registry)
        
        if registry.has('oxygen_mixer'):
            self._target_component = registry.get('oxygen_mixer')
        elif registry.has('oxygen_buffer'):
            self._target_component = registry.get('oxygen_buffer')
    
    def step(self, t: float) -> None:
        """Execute timestep - deliver oxygen."""
        super().step(t)
        
        if self.mode == "fixed_flow":
            self.o2_output_kg = self.flow_rate_kg_h * self.dt
        
        elif self.mode == "pressure_driven":
            if self._target_component:
                required_flow = self._estimate_required_flow()
                self.o2_output_kg = min(required_flow, self.max_capacity_kg_h * self.dt)
            else:
                self.o2_output_kg = 0.0
        
        self.cumulative_o2_kg += self.o2_output_kg
        self.cumulative_cost += self.o2_output_kg * self.cost_per_kg
    
    def get_state(self) -> Dict[str, Any]:
        """Return current state."""
        return {
            **super().get_state(),
            'mode': self.mode,
            'flow_rate_kg_h': float(self.flow_rate_kg_h),
            'o2_output_kg': float(self.o2_output_kg),
            'pressure_bar': float(self.pressure_bar),
            'cumulative_o2_kg': float(self.cumulative_o2_kg),
            'cumulative_cost': float(self.cumulative_cost),
            'flows': {
                'outputs': {
                    'oxygen': {
                        'value': float(self.o2_output_kg),
                        'unit': 'kg',
                        'destination': self._target_component.component_id if self._target_component else 'unconnected'
                    }
                }
            }
        }
    
    def _estimate_required_flow(self) -> float:
        """
        Estimate required O₂ flow to maintain target pressure.
        This is a placeholder for a real thermodynamic model.
        """
        # In a real implementation, this would involve complex thermo calculations
        # For now, return a placeholder value
        return 10.0
