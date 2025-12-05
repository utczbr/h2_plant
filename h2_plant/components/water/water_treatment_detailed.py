"""
Detailed Water Treatment system.

Models the water purification and storage system:
- Water Purifier (WP)
- Ultra-pure Water Tank (WT)
"""

from typing import Dict, Any
from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.composite_component import CompositeComponent

class WaterPurifier(Component):
    """
    Purifies external water to ultra-pure standards.
    """
    def __init__(self, purifier_id: str, max_flow_kg_h: float, efficiency: float = 0.95):
        super().__init__()
        self.purifier_id = purifier_id
        self.max_flow_kg_h = max_flow_kg_h
        self.efficiency = efficiency
        self.input_flow_kg_h = 0.0
        self.output_flow_kg_h = 0.0
        self.power_kw = 0.0
        
    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        super().initialize(dt, registry)
        
    def step(self, t: float) -> None:
        super().step(t)
        # Limit input to max capacity
        processed_flow = min(self.input_flow_kg_h, self.max_flow_kg_h)
        self.output_flow_kg_h = processed_flow * self.efficiency
        
        # Simplified power: 0.5 kWh/m3 -> 0.0005 kWh/kg
        self.power_kw = (processed_flow * 0.0005) / self.dt if self.dt > 0 else 0.0
        
    def get_state(self) -> Dict[str, Any]:
        return {
            **super().get_state(),
            'output_flow_kg_h': float(self.output_flow_kg_h),
            'power_kw': float(self.power_kw)
        }

class WaterTank(Component):
    """
    Atmospheric water storage tank.
    """
    def __init__(self, tank_id: str, capacity_kg: float):
        super().__init__()
        self.tank_id = tank_id
        self.capacity_kg = capacity_kg
        self.current_mass_kg = 0.0
        self.inlet_flow_kg_h = 0.0
        self.outlet_flow_kg_h = 0.0
        
    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        super().initialize(dt, registry)
        self.current_mass_kg = self.capacity_kg * 0.5 # Start half full
        
    def step(self, t: float) -> None:
        super().step(t)
        
        # Mass balance
        delta_mass = (self.inlet_flow_kg_h - self.outlet_flow_kg_h) * self.dt
        self.current_mass_kg += delta_mass
        
        # Clamp
        if self.current_mass_kg > self.capacity_kg:
            self.current_mass_kg = self.capacity_kg
            # Overflow logic could be added
        elif self.current_mass_kg < 0:
            self.current_mass_kg = 0.0
            
    def get_state(self) -> Dict[str, Any]:
        return {
            **super().get_state(),
            'current_mass_kg': float(self.current_mass_kg),
            'fill_level': float(self.current_mass_kg / self.capacity_kg)
        }

class DetailedWaterTreatment(CompositeComponent):
    """
    Complete Water Treatment system.
    """
    def __init__(self, max_flow_kg_h: float = 2000.0, tank_capacity_kg: float = 10000.0):
        super().__init__()
        
        self.add_subsystem('purifier_wp', WaterPurifier('WP', max_flow_kg_h))
        self.add_subsystem('tank_wt', WaterTank('WT', tank_capacity_kg))
        
        self.external_water_input_kg_h = 0.0
        self.demand_pem_kg_h = 0.0
        self.demand_soec_kg_h = 0.0
        self.demand_atr_kg_h = 0.0
        
    def step(self, t: float) -> None:
        Component.step(self, t)
        
        # 1. Purification
        self.purifier_wp.input_flow_kg_h = self.external_water_input_kg_h
        self.purifier_wp.step(t)
        
        # 2. Storage
        total_demand = self.demand_pem_kg_h + self.demand_soec_kg_h + self.demand_atr_kg_h
        
        self.tank_wt.inlet_flow_kg_h = self.purifier_wp.output_flow_kg_h
        self.tank_wt.outlet_flow_kg_h = total_demand
        self.tank_wt.step(t)
        
    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state['summary'] = {
            'stored_water_kg': self.tank_wt.current_mass_kg,
            'total_demand_kg_h': self.tank_wt.outlet_flow_kg_h
        }
        return state
