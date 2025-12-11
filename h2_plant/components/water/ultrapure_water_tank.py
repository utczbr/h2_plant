
from typing import Dict, Any, List, Optional
from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.stream import Stream
from h2_plant.core.component_ids import ComponentID
from h2_plant.config.constants_physics import WaterConstants

class UltraPureWaterTank(Component):
    """
    Ultra-pure Water Tank (WT) Component.
    Buffer storage for purified water with enthalpy mixing and correct thermodynamics.
    """
    def __init__(self, component_id: str, capacity_kg: float = None):
        super().__init__()
        self.component_id = component_id
        self.capacity_kg = capacity_kg if capacity_kg is not None else WaterConstants.ULTRAPURE_TANK_CAPACITY_KG
        
        # Intrinsic Properties (State)
        self.mass_kg = self.capacity_kg * 0.5 # Start 50% full
        self.temperature_k = WaterConstants.WATER_AMBIENT_T_K
        self.pressure_pa = WaterConstants.WATER_ATM_P_PA
        
        # LUT Manager reference
        self.lut = None

        # Inputs
        self.inlet_stream: Optional[Stream] = None

        # Outlet logic
        self.outlet_streams: Dict[str, Stream] = {}
        self.outlet_requests: Dict[str, float] = {}

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        super().initialize(dt, registry)
        self.lut = registry.get_by_type("lut_manager")[0] if registry.get_by_type("lut_manager") else None

    def request_outflow(self, consumer_id: str, amount_kg_h: float):
        """Register a demand from a consumer."""
        self.outlet_requests[consumer_id] = amount_kg_h

    def step(self, t: float) -> None:
        super().step(t)
        
        # 0. Detect Consumers (if using push/pull)
        # Assuming consumers call request_outflow OR we pull from registry
        # The spec says "get_demand: sum consumer requests (e.g. PEM waterinput_kgh via registry)"
        # But 'request_outflow' is a cleaner push model from consumers.
        # Let's support polling PEM if needed? 
        # For now, rely on registered requests.
        
        # 1. Process Inflow (Mixing)
        if self.inlet_stream and self.inlet_stream.mass_flow_kg_h > 0:
            m_in = self.inlet_stream.mass_flow_kg_h * self.dt
            T_in = self.inlet_stream.temperature_k
            
            # Accepted mass (clamp to capacity)
            m_accepted = min(m_in, self.capacity_kg - self.mass_kg)
            
            if m_accepted > 0:
                # Enthalpy Balance: (m_old * H_old + m_in * H_in) = m_new * H_new
                # H approx = Cp * T for liquid water
                # Cp_water approx 4184 J/kgK
                Cp = 4184.0 
                # Or use LUT if available
                # if self.lut: Cp = self.lut.lookup('Water', 'C', self.pressure_pa, self.temperature_k)
                
                H_current = self.mass_kg * Cp * self.temperature_k
                H_in = m_accepted * Cp * T_in
                
                m_new = self.mass_kg + m_accepted
                H_new = H_current + H_in
                
                self.temperature_k = H_new / (m_new * Cp)
                self.mass_kg = m_new
                
        # 2. Process Outflows
        total_req_kg = sum(self.outlet_requests.values()) * self.dt
        
        scaling = 1.0
        if total_req_kg > self.mass_kg:
            scaling = self.mass_kg / total_req_kg if total_req_kg > 0 else 0.0
            
        self.outlet_streams.clear()
        total_out_kg = 0.0
        
        for cid, rate in self.outlet_requests.items():
            actual_rate = rate * scaling
            if actual_rate > 0:
                self.outlet_streams[cid] = Stream(
                    mass_flow_kg_h=actual_rate,
                    temperature_k=self.temperature_k,
                    pressure_pa=self.pressure_pa,
                    composition={'H2O': 1.0},
                    phase='liquid'
                )
                total_out_kg += actual_rate * self.dt
                
        self.mass_kg -= total_out_kg
        if self.mass_kg < 0: self.mass_kg = 0.0
        
        # Clear requests for next step
        self.outlet_requests.clear()
        
    def receive_input(self, port_name: str, value: Any, resource_type: str = None) -> float:
        if port_name == 'ultrapure_in' and isinstance(value, Stream):
            self.inlet_stream = value
            # Calculate how much we CAN accept for upstream backpressure?
            # Returns accepted amount
            space = self.capacity_kg - self.mass_kg
            accepted_flow = min(value.mass_flow_kg_h, space / self.dt)
            return accepted_flow * self.dt
        return 0.0

    def get_output(self, port_name: str) -> Any:
        # If port_name matches a consumer request, return that stream
        # Or generic 'consumer_out'
        if port_name == 'consumer_out':
             # Return valid stream if strictly 1 consumer? 
             # Or return sum?
             # For now return random valid one or None?
             vals = list(self.outlet_streams.values())
             if vals: return vals[0]
        return self.outlet_streams.get(port_name)

    def get_state(self) -> Dict[str, Any]:
        return {
            **super().get_state(),
            'component_id': self.component_id,
            'mass_kg': self.mass_kg,
            'fill_level': self.mass_kg / self.capacity_kg,
            'temperature_c': self.temperature_k - 273.15,
            'pressure_bar': self.pressure_pa / 1e5
        }
