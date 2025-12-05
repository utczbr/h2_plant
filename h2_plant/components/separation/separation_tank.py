"""
Separation Tank component for gas/liquid separation.

Separates liquid water from gas streams (H2 or O2).
"""

from typing import Dict, Any
from h2_plant.core.component import Component
from h2_plant.core.stream import Stream


class SeparationTank(Component):
    """
    Gas/Liquid separation tank.
    
    Separates liquid phase (water) from gas phase (H2 or O2) based on density/gravity.
    Used in Process Flow as ST-1, ST-2, ST-3, ST-4.
    """
    
    def __init__(
        self,
        component_id: str = "separation_tank",
        volume_m3: float = 5.0,
        efficiency: float = 0.99
    ):
        """
        Initialize SeparationTank.
        
        Args:
            component_id: Unique identifier
            volume_m3: Tank volume in cubic meters
            efficiency: Separation efficiency (fraction of liquid removed)
        """
        super().__init__()
        self.component_id = component_id
        self.volume_m3 = volume_m3
        self.efficiency = efficiency
        
        # State variables
        self.inlet_stream: Stream = Stream(0.0)
        self.gas_outlet: Stream = Stream(0.0)
        self.liquid_outlet: Stream = Stream(0.0)
        self.liquid_level: float = 0.0  # Fraction 0-1
        
    def initialize(self, dt: float, registry: Any) -> None:
        """Initialize component."""
        super().initialize(dt, registry)
        self.initialized = True
        
    def step(self, t: float) -> None:
        """Execute one timestep of separation."""
        super().step(t)
        
        if self.inlet_stream.mass_flow_kg_h <= 0:
            self.gas_outlet = Stream(0.0)
            self.liquid_outlet = Stream(0.0)
            return
            
        # Simple separation logic: assume all gas goes to gas_out, 
        # most liquid goes to liquid_out
        
        # Determine composition
        total_flow = self.inlet_stream.mass_flow_kg_h
        composition = self.inlet_stream.composition
        
        # Identify gas and liquid species (simplified)
        # Assuming H2, O2, N2, CO2, CH4 are gases, H2O is liquid
        gas_fraction = 0.0
        liquid_fraction = 0.0
        
        gas_comp = {}
        liquid_comp = {}
        
        for species, fraction in composition.items():
            if species == 'H2O':
                liquid_fraction += fraction
                liquid_comp[species] = 1.0 # Pure water in liquid phase
            else:
                gas_fraction += fraction
                gas_comp[species] = fraction # Normalize later
                
        # Normalize gas composition
        total_gas_frac = sum(gas_comp.values())
        if total_gas_frac > 0:
            for s in gas_comp:
                gas_comp[s] /= total_gas_frac
        
        # Calculate flows
        gas_flow = total_flow * gas_fraction
        liquid_flow = total_flow * liquid_fraction
        
        # Apply efficiency (some liquid might carry over)
        carry_over_liquid = liquid_flow * (1 - self.efficiency)
        captured_liquid = liquid_flow * self.efficiency
        
        final_gas_flow = gas_flow + carry_over_liquid
        
        # Create outlet streams
        self.gas_outlet = Stream(
            mass_flow_kg_h=final_gas_flow,
            temperature_k=self.inlet_stream.temperature_k,
            pressure_pa=self.inlet_stream.pressure_pa,
            composition=gas_comp # Simplified, ignoring carry over composition effect
        )
        
        self.liquid_outlet = Stream(
            mass_flow_kg_h=captured_liquid,
            temperature_k=self.inlet_stream.temperature_k,
            pressure_pa=self.inlet_stream.pressure_pa,
            composition={'H2O': 1.0},
            phase='liquid'
        )
        
    def get_output(self, port_name: str) -> Any:
        """Get output from specified port."""
        if port_name == "gas_out":
            return self.gas_outlet
        elif port_name == "liquid_out":
            return self.liquid_outlet
        return Stream(0.0)
    
    def receive_input(self, port_name: str, value: Any, resource_type: str = None) -> float:
        """Receive input at specified port."""
        if port_name == "mixture_in" and isinstance(value, Stream):
            self.inlet_stream = value
            return value.mass_flow_kg_h
        return 0.0
    
    def extract_output(self, port_name: str, amount: float, resource_type: str = None) -> None:
        """Acknowledge extraction of output."""
        pass
    
    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """Return port definitions."""
        return {
            'mixture_in': {'type': 'input', 'resource_type': 'stream'},
            'gas_out': {'type': 'output', 'resource_type': 'gas'},
            'liquid_out': {'type': 'output', 'resource_type': 'water'}
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Return current component state."""
        return {
            **super().get_state(),
            'component_id': self.component_id,
            'gas_flow_kg_h': self.gas_outlet.mass_flow_kg_h,
            'liquid_flow_kg_h': self.liquid_outlet.mass_flow_kg_h
        }
