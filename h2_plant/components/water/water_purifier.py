
from typing import Dict, Any, Optional
from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.stream import Stream
from h2_plant.core.component_ids import ComponentID
from h2_plant.config.constants_physics import WaterConstants

class WaterPurifier(Component):
    """
    Water Purifier (WP) Component.
    Processes raw water into ultrapure water for electrolyzers.
    """
    def __init__(self, component_id: str, max_flow_kg_h: float = None):
        super().__init__()
        self.component_id = component_id
        # Use constant default if not provided
        self.max_flow_kg_h = max_flow_kg_h if max_flow_kg_h is not None else WaterConstants.WATER_PURIFIER_MAX_FLOW_KGH
        
        # State
        self.input_mass_kg = 0.0
        self.power_consumed_kw = 0.0
        
        # Ports
        self.ultrapure_out_stream: Optional[Stream] = None
        self.waste_out_stream: Optional[Stream] = None
        self.raw_water_in_stream: Optional[Stream] = None

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        super().initialize(dt, registry)
        self.lut = registry.get_by_type("lut_manager")[0] if registry.get_by_type("lut_manager") else None
        
    def step(self, t: float) -> None:
        super().step(t)
        
        # 1. Determine Production Demand
        # Check storage tank level to avoid overfilling
        tank = self.get_registry_safe(ComponentID.ULTRAPURE_WATER_STORAGE)
        production_allowed = True
        if tank:
             # Basic on/off logic based on fill ratio? 
             # Or just push and let tank reject/waste?
             # Spec says: "max_flow if tank fill < low_ratio else 0"
             # But simplistic check first:
             if hasattr(tank, 'fill_level'):
                 if tank.fill_level > 0.95: # Near full
                     production_allowed = False
                 elif tank.fill_level < WaterConstants.ULTRAPURE_TANK_LOW_FILL_RATIO:
                     production_allowed = True
                 # Hysterisis could be better, but sticking to simple for now
        
        # 2. Process Input
        if production_allowed and self.input_mass_kg > 0:
            # How much can we process?
            max_process = self.max_flow_kg_h * self.dt
            processed_mass = min(self.input_mass_kg, max_process)
            
            # 3. Calculate Flows
            recovery = WaterConstants.WATER_RO_RECOVERY_RATIO
            pure_mass = processed_mass * recovery
            waste_mass = processed_mass - pure_mass
            
            pure_flow = pure_mass / self.dt
            waste_flow = waste_mass / self.dt
            
            # 4. Energy Consumption
            # kWh = kg * kWh/kg
            energy_kwh = pure_mass * WaterConstants.WATER_RO_SPEC_ENERGY_KWH_KG
            self.power_consumed_kw = energy_kwh / self.dt # kW
            
            # 5. Thermodynamics (using LUT or defaults)
            if self.raw_water_in_stream:
                T_in = self.raw_water_in_stream.temperature_k
                P_in = self.raw_water_in_stream.pressure_pa
            else:
                T_in = WaterConstants.WATER_AMBIENT_T_K
                P_in = WaterConstants.WATER_ATM_P_PA
                
            P_out = max(WaterConstants.WATER_ATM_P_PA, P_in - WaterConstants.WATER_PURIFIER_PRESSURE_DROP_PA)
            
            # Create Streams
            self.ultrapure_out_stream = Stream(
                mass_flow_kg_h=pure_flow,
                temperature_k=T_in, # Isothermal assumption for RO
                pressure_pa=P_out,
                composition={'H2O': 1.0},
                phase='liquid'
            )
            
            self.waste_out_stream = Stream(
                mass_flow_kg_h=waste_flow,
                temperature_k=T_in, 
                pressure_pa=WaterConstants.WATER_ATM_P_PA, # Released to drain/atm
                composition={'H2O': 0.95, 'Salts': 0.05} if self.raw_water_in_stream else {'H2O': 0.99},
                phase='liquid'
            )
            
            # Consume input buffer
            self.input_mass_kg -= processed_mass
            
        else:
            self.power_consumed_kw = 0.0
            self.ultrapure_out_stream = None
            self.waste_out_stream = None
            # Input mass remains in buffer
            
    def receive_input(self, port_name: str, value: Any, resource_type: str = None) -> float:
        if port_name == 'raw_water_in' and isinstance(value, Stream):
            self.raw_water_in_stream = value
            mass_in = value.mass_flow_kg_h * self.dt
            # Infinite input buffer for now, or processed immediately in step?
            # Typically receive_input happens BEFORE step.
            self.input_mass_kg += mass_in
            return mass_in
        return 0.0

    def get_output(self, port_name: str) -> Any:
        if port_name == 'ultrapure_out':
            return self.ultrapure_out_stream
        elif port_name == 'waste_out':
            return self.waste_out_stream
        return None

    def get_state(self) -> Dict[str, Any]:
        return {
            **super().get_state(),
            'power_kw': self.power_consumed_kw,
            'input_buffer_kg': self.input_mass_kg,
            'ultrapure_flow_kgh': self.ultrapure_out_stream.mass_flow_kg_h if self.ultrapure_out_stream else 0.0
        }
