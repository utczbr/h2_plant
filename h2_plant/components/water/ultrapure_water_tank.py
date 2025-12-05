
from typing import Dict, Any, List, Optional
from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.stream import Stream

class UltraPureWaterTank(Component):
    """
    Ultra-pure Water Tank (WT) Component.
    Buffer storage for purified water with multiple outlets.
    Now uses Stream for thermodynamic tracking.
    """
    def __init__(self, component_id: str, capacity_kg: float, initial_level: float = 0.5):
        super().__init__()
        self.component_id = component_id
        self.capacity_kg = capacity_kg

        # Intrinsic Properties
        self.mass_kg = capacity_kg * initial_level
        self.temperature_k = 293.15 # 20°C bulk temperature
        self.pressure_pa = 101325.0 # Atmospheric
        self.density_kg_m3 = 998.0

        # Inputs
        self.inlet_stream: Optional[Stream] = None

        # Multiple Outlets: Dictionary mapping Consumer ID -> Stream
        self.outlet_streams: Dict[str, Stream] = {}
        self.outlet_requests: Dict[str, float] = {} # Consumer ID -> requested kg/h

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        super().initialize(dt, registry)

    def request_outflow(self, consumer_id: str, amount_kg_h: float):
        """Register a demand from a consumer."""
        self.outlet_requests[consumer_id] = amount_kg_h

    def step(self, t: float) -> None:
        super().step(t)

        # 1. Process Inflow
        if self.inlet_stream and self.inlet_stream.mass_flow_kg_h > 0:
            in_mass = self.inlet_stream.mass_flow_kg_h * self.dt
            
            # Mix incoming stream with stored water
            if self.mass_kg > 0:
                stored_stream = Stream(
                    mass_flow_kg_h=self.mass_kg / self.dt,
                    temperature_k=self.temperature_k,
                    pressure_pa=self.pressure_pa,
                    composition={'H2O': 1.0},
                    phase='liquid'
                )
                
                mixed_stream = stored_stream.mix_with(self.inlet_stream)
                self.temperature_k = mixed_stream.temperature_k
            else:
                # Tank empty, takes inlet state
                self.temperature_k = self.inlet_stream.temperature_k
            
            self.mass_kg += in_mass

        # 2. Process Outflows
        total_requested_kg = sum(self.outlet_requests.values()) * self.dt
        
        if total_requested_kg > self.mass_kg:
            # Shortage! Scale down all outflows proportionally
            scaling_factor = self.mass_kg / total_requested_kg if total_requested_kg > 0 else 0
        else:
            scaling_factor = 1.0
        
        # Create outlet streams
        self.outlet_streams.clear()
        total_out = 0.0
        
        for consumer_id, requested_flow in self.outlet_requests.items():
            actual_flow = requested_flow * scaling_factor
            
            if actual_flow > 0:
                self.outlet_streams[consumer_id] = Stream(
                    mass_flow_kg_h=actual_flow,
                    temperature_k=self.temperature_k,
                    pressure_pa=self.pressure_pa,
                    composition={'H2O': 1.0},
                    phase='liquid'
                )
                total_out += actual_flow * self.dt
        
        self.mass_kg -= total_out
        if self.mass_kg < 0:
            self.mass_kg = 0 # Safety

        # Cap at capacity (overflow)
        if self.mass_kg > self.capacity_kg:
            self.mass_kg = self.capacity_kg

    def get_state(self) -> Dict[str, Any]:
        state = {
            **super().get_state(),
            'component_id': self.component_id,
            'mass_kg': self.mass_kg,
            'fill_level': self.mass_kg / self.capacity_kg,
            'temperature_c': self.temperature_k - 273.15,
            'pressure_bar': self.pressure_pa / 1e5,
            'total_outflow_kg_h': sum(s.mass_flow_kg_h for s in self.outlet_streams.values()),
            'num_outlets': len(self.outlet_streams)
        }
        
        if self.outlet_streams:
            state['streams'] = {
                f'out_{cid}': {
                    'mass_flow': stream.mass_flow_kg_h,
                    'temperature': stream.temperature_k,
                    'pressure': stream.pressure_pa
                }
                for cid, stream in self.outlet_streams.items()
            }
        
        return state
