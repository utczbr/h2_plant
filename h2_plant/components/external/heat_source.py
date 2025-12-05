"""
External waste heat source component.

Provides thermal energy from external sources for process heating,
water preheating, or power generation (future).
"""

from typing import Dict, Any, Optional
import random
from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry


class ExternalHeatSource(Component):
    """
    External waste heat supply source.
    """
    
    def __init__(
        self,
        thermal_power_kw: float = 500.0,
        temperature_c: float = 150.0,
        availability_factor: float = 1.0,
        cost_per_kwh: float = 0.0,
        min_output_fraction: float = 0.2
    ):
        super().__init__()
        
        self.thermal_power_kw = thermal_power_kw
        self.temperature_k = temperature_c + 273.15
        self.temperature_c = temperature_c
        self.availability_factor = availability_factor
        self.cost_per_kwh = cost_per_kwh
        self.min_output_fraction = min_output_fraction
        
        self.available = True
        self.heat_output_kwh = 0.0
        self.current_power_kw = 0.0
        self.cumulative_heat_kwh = 0.0
        self.cumulative_cost = 0.0
        self.heat_demand_kw = 0.0
    
    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """Initialize heat source."""
        super().initialize(dt, registry)
    
    def step(self, t: float) -> None:
        """Execute timestep - provide heat."""
        super().step(t)
        
        self.available = random.random() < self.availability_factor
        
        if not self.available:
            self.current_power_kw = 0.0
        elif self.heat_demand_kw > 0 and self.heat_demand_kw >= self.thermal_power_kw * self.min_output_fraction:
            self.current_power_kw = min(self.heat_demand_kw, self.thermal_power_kw)
        else:
            self.current_power_kw = 0.0
        
        self.heat_output_kwh = self.current_power_kw * self.dt
        
        self.cumulative_heat_kwh += self.heat_output_kwh
        self.cumulative_cost += self.heat_output_kwh * self.cost_per_kwh
    
    def get_state(self) -> Dict[str, Any]:
        """Return current state."""
        return {
            **super().get_state(),
            'available': self.available,
            'thermal_power_kw': float(self.thermal_power_kw),
            'temperature_c': float(self.temperature_c),
            'temperature_k': float(self.temperature_k),
            'current_power_kw': float(self.current_power_kw),
            'heat_output_kwh': float(self.heat_output_kwh),
            'heat_demand_kw': float(self.heat_demand_kw),
            'cumulative_heat_kwh': float(self.cumulative_heat_kwh),
            'cumulative_cost': float(self.cumulative_cost),
            'utilization': float(self.current_power_kw / self.thermal_power_kw) if self.thermal_power_kw > 0 else 0.0,
            'flows': {
                'outputs': {
                    'thermal_energy': {
                        'value': float(self.heat_output_kwh),
                        'unit': 'kWh',
                        'temperature_c': float(self.temperature_c),
                        'destination': 'heat_consumers'
                    }
                }
            }
        }
    
    def set_demand(self, demand_kw: float) -> None:
        """
        Set heat demand from consuming components.
        """
        self.heat_demand_kw = demand_kw
