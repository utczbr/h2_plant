"""
Pressure Swing Adsorption (PSA) unit for gas purification.

Purifies H2, O2, or syngas using cyclic adsorption/desorption.
"""

from typing import Dict, Any
from h2_plant.core.component import Component
from h2_plant.core.stream import Stream


class PSA(Component):
    """
    Pressure Swing Adsorption unit for gas purification.
    
    Multi-bed cyclic operation alternating between adsorption (high pressure)
    and regeneration (low pressure). Produces high-purity product gas and
    tail gas containing impurities.
    
    Used in Process Flow as D-1, D-2, D-3, D-4.
    """
    
    def __init__(
        self,
        component_id: str = "psa",
        num_beds: int = 2,
        cycle_time_min: float = 5.0,
        purity_target: float = 0.9999,
        recovery_rate: float = 0.90,
        power_consumption_kw: float = 10.0
    ):
        """
        Initialize PSA unit.
        
        Args:
            component_id: Unique identifier
            num_beds: Number of adsorption beds
            cycle_time_min: Total cycle time (adsorption + regeneration)
            purity_target: Target purity of product gas (mole fraction)
            recovery_rate: Fraction of feed gas recovered as product
            power_consumption_kw: Electrical power for valves/controls
        """
        super().__init__()
        self.component_id = component_id
        self.num_beds = num_beds
        self.cycle_time_min = cycle_time_min
        self.purity_target = purity_target
        self.recovery_rate = recovery_rate
        self.power_consumption_kw = power_consumption_kw
        
        # State variables
        self.inlet_stream: Stream = Stream(0.0)
        self.product_outlet: Stream = Stream(0.0)
        self.tail_gas_outlet: Stream = Stream(0.0)
        self.cycle_position: float = 0.0  # Current position in cycle (0-1)
        self.active_beds: int = num_beds // 2  # Half beds adsorbing, half regenerating
        
    def initialize(self, dt: float, registry: Any) -> None:
        """Initialize component."""
        super().initialize(dt, registry)
        self.initialized = True

    def step(self, t: float) -> None:
        """Execute one timestep of PSA operation."""
        super().step(t)
        
        if self.inlet_stream.mass_flow_kg_h <= 0:
            self.product_outlet = Stream(0.0)
            self.tail_gas_outlet = Stream(0.0)
            return
        
        # Advance cycle position
        dt_hours = 1.0  # Assuming 1-hour timestep
        cycle_fraction = (dt_hours * 60) / self.cycle_time_min
        self.cycle_position = (self.cycle_position + cycle_fraction) % 1.0
        
        # Determine target species (H2, O2, or CO based on inlet composition)
        composition = self.inlet_stream.composition
        target_species = max(composition, key=composition.get) if composition else 'H2'
        
        # Calculate product and tail gas flows
        inlet_flow = self.inlet_stream.mass_flow_kg_h
        product_flow = inlet_flow * self.recovery_rate
        tail_gas_flow = inlet_flow - product_flow
        
        # Product composition (purified)
        product_composition = {target_species: self.purity_target}
        for species in composition:
            if species != target_species:
                product_composition[species] = (1 - self.purity_target) / max(1, len(composition) - 1)
        
        # Tail gas composition (enriched in impurities)
        tail_gas_composition = {}
        for species in composition:
            if species == target_species:
                tail_gas_composition[species] = max(0, (composition[species] * inlet_flow - product_composition[species] * product_flow) / tail_gas_flow)
            else:
                tail_gas_composition[species] = composition[species] * inlet_flow / tail_gas_flow
        
        # Create outlet streams
        self.product_outlet = Stream(
            mass_flow_kg_h=product_flow,
            temperature_k=self.inlet_stream.temperature_k,
            pressure_pa=self.inlet_stream.pressure_pa,
            composition=product_composition
        )
        
        self.tail_gas_outlet = Stream(
            mass_flow_kg_h=tail_gas_flow,
            temperature_k=self.inlet_stream.temperature_k,
            pressure_pa=101325.0,  # Low pressure for tail gas
            composition=tail_gas_composition
        )
    
    def get_output(self, port_name: str) -> Any:
        """Get output from specified port."""
        if port_name == "purified_gas_out":
            return self.product_outlet
        elif port_name == "tail_gas_out":
            return self.tail_gas_outlet
        return Stream(0.0)
    
    def receive_input(self, port_name: str, value: Any, resource_type: str = None) -> float:
        """Receive input at specified port."""
        if port_name == "gas_in" and isinstance(value, Stream):
            self.inlet_stream = value
            return value.mass_flow_kg_h
        elif port_name == "electricity_in" and isinstance(value, (int, float)):
            return min(value, self.power_consumption_kw)
        return 0.0
    
    def extract_output(self, port_name: str, amount: float, resource_type: str = None) -> None:
        """Acknowledge extraction of output."""
        pass
    
    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """Return port definitions."""
        return {
            'gas_in': {'type': 'input', 'resource_type': 'gas'},
            'electricity_in': {'type': 'input', 'resource_type': 'electricity'},
            'purified_gas_out': {'type': 'output', 'resource_type': 'gas'},
            'tail_gas_out': {'type': 'output', 'resource_type': 'gas'}
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Return current component state."""
        return {
            **super().get_state(),
            'component_id': self.component_id,
            'product_flow_kg_h': self.product_outlet.mass_flow_kg_h,
            'tail_gas_flow_kg_h': self.tail_gas_outlet.mass_flow_kg_h,
            'cycle_position': self.cycle_position,
            'power_consumption_kw': self.power_consumption_kw
        }
