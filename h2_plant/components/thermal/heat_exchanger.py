
from typing import Dict, Any, Optional
from h2_plant.core.component import Component
from h2_plant.core.stream import Stream
from h2_plant.core.constants import ConversionFactors

class HeatExchanger(Component):
    def __init__(self, component_id: str, max_heat_removal_kw: float, target_outlet_temp_c: float = 25.0):
        super().__init__()
        self.component_id = component_id
        self.max_heat_removal_kw = max_heat_removal_kw
        self.target_outlet_temp_c = target_outlet_temp_c
        
        # Inputs
        self.inlet_flow_kg_h = 0.0
        self.input_stream: Optional[Stream] = None
        
        # Outputs
        self.output_stream: Optional[Stream] = None
        self.heat_removed_kw = 0.0

    def initialize(self, dt: float, registry: 'ComponentRegistry') -> None:
        super().initialize(dt, registry)

    def step(self, t: float) -> None:
        super().step(t)
        
        # Handle legacy input
        if self.input_stream is None:
            if self.inlet_flow_kg_h > 0:
                # Assume some default hot temp if not specified (e.g. 80C from electrolyzer)
                self.input_stream = Stream(
                    mass_flow_kg_h=self.inlet_flow_kg_h,
                    temperature_k=353.15, # 80C default
                    pressure_pa=101325.0,
                    composition={'H2': 1.0}
                )
            else:
                self.output_stream = None
                self.heat_removed_kw = 0.0
                return

        # Target state
        target_temp_k = self.target_outlet_temp_c + 273.15
        
        # If input is already cooler than target, do nothing
        if self.input_stream.temperature_k <= target_temp_k:
            self.output_stream = self.input_stream
            self.heat_removed_kw = 0.0
            return

        # Calculate required cooling
        # Create a hypothetical stream at target temp to get enthalpy
        target_stream = Stream(
            mass_flow_kg_h=self.input_stream.mass_flow_kg_h,
            temperature_k=target_temp_k,
            pressure_pa=self.input_stream.pressure_pa, # Assume isobaric cooling
            composition=self.input_stream.composition,
            phase=self.input_stream.phase
        )
        
        h_in = self.input_stream.specific_enthalpy_j_kg
        h_target = target_stream.specific_enthalpy_j_kg
        
        # Q_required = m * (h_in - h_target)
        # Units: kg/h * J/kg = J/h
        q_required_j_h = self.input_stream.mass_flow_kg_h * (h_in - h_target)
        q_required_kw = q_required_j_h * ConversionFactors.J_TO_KWH
        
        # Limit by capacity
        if q_required_kw <= self.max_heat_removal_kw:
            self.heat_removed_kw = q_required_kw
            self.output_stream = target_stream
        else:
            # Capacity limited
            self.heat_removed_kw = self.max_heat_removal_kw
            
            # Calculate actual outlet enthalpy
            # h_out = h_in - Q_max / m
            q_removed_j_kg = (self.max_heat_removal_kw / ConversionFactors.J_TO_KWH) / self.input_stream.mass_flow_kg_h
            h_out = h_in - q_removed_j_kg
            
            # Find T_out for h_out
            # Simple bisection search between target_temp and input_temp
            t_low = target_temp_k
            t_high = self.input_stream.temperature_k
            
            for _ in range(10): # 10 iterations is enough for <0.1K precision
                t_mid = (t_low + t_high) / 2
                s_mid = Stream(1.0, t_mid, self.input_stream.pressure_pa, self.input_stream.composition)
                h_mid = s_mid.specific_enthalpy_j_kg
                
                if h_mid > h_out:
                    t_high = t_mid
                else:
                    t_low = t_mid
            
            self.output_stream = Stream(
                mass_flow_kg_h=self.input_stream.mass_flow_kg_h,
                temperature_k=t_high,
                pressure_pa=self.input_stream.pressure_pa,
                composition=self.input_stream.composition,
                phase=self.input_stream.phase
            )

    def get_state(self) -> Dict[str, Any]:
        state = {
            **super().get_state(), 
            'component_id': self.component_id, 
            'heat_removed_kw': self.heat_removed_kw
        }
        
        if self.output_stream:
            state['streams'] = {
                'out': {
                    'mass_flow': self.output_stream.mass_flow_kg_h,
                    'temperature': self.output_stream.temperature_k,
                    'enthalpy': self.output_stream.specific_enthalpy_j_kg
                }
            }
        return state
            
    def get_output(self, port_name: str) -> Any:
        """Get output from specific port."""
        # Generic output port naming? 
        # Or specific based on what is flowing?
        # The topology uses 'water_out' or 'h2_out'.
        # But HeatExchanger is generic.
        # Let's support both or check what we have.
        if port_name in ['water_out', 'h2_out', 'out', 'cooled_gas_out', 'syngas_out']:
            if self.output_stream:
                return self.output_stream
            else:
                # Return empty stream if no input
                return Stream(0.0)
        elif port_name == 'heat_out':
            # Heat removed
            return self.heat_removed_kw
        else:
            raise ValueError(f"Unknown output port '{port_name}' on {self.component_id}")

    def receive_input(self, port_name: str, value: Any, resource_type: str) -> float:
        """Receive input."""
        if port_name in ['water_in', 'h2_in', 'in']:
            if isinstance(value, Stream):
                self.input_stream = value
                # We accept all flow, as it's a pass-through component
                return value.mass_flow_kg_h
        return 0.0

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """Return port metadata."""
        # Dynamic ports based on usage?
        # Or just advertise generic ports?
        # Let's advertise common ones.
        return {
            'water_in': {'type': 'input', 'resource_type': 'water', 'units': 'kg/h'},
            'h2_in': {'type': 'input', 'resource_type': 'hydrogen', 'units': 'kg/h'},
            'water_out': {'type': 'output', 'resource_type': 'water', 'units': 'kg/h'},
            'h2_out': {'type': 'output', 'resource_type': 'hydrogen', 'units': 'kg/h'},
            'heat_out': {'type': 'output', 'resource_type': 'heat', 'units': 'kW'}
        }
