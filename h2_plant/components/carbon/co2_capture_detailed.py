"""
Detailed CO2 Capture and Storage system.

Models the carbon management system:
- CO2 Capture Unit (CO2C)
- Compression (C-5)
- Storage (CO2S)
"""

from typing import Dict, Any
from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.composite_component import CompositeComponent
# from h2_plant.components.production.soec_electrolyzer_detailed import ProcessCompressor

class ProcessCompressor(Component):
    """Simple compressor for CO2 process."""
    def __init__(self, comp_id: str, max_flow_kg_h: float):
        super().__init__()
        self.comp_id = comp_id
        self.max_flow_kg_h = max_flow_kg_h
        self.input_flow_kg_h = 0.0
        self.output_flow_kg_h = 0.0
        self.power_kw = 0.0
        
    def step(self, t: float) -> None:
        super().step(t)
        # Simple flow pass-through limited by max capacity
        self.output_flow_kg_h = min(self.input_flow_kg_h, self.max_flow_kg_h)
        
        # Approximate power: 0.1 kWh/kg for moderate compression
        self.power_kw = self.output_flow_kg_h * 0.1
        
    def get_state(self) -> Dict[str, Any]:
        return {
            **super().get_state(),
            'input_flow_kg_h': self.input_flow_kg_h,
            'output_flow_kg_h': self.output_flow_kg_h,
            'power_kw': self.power_kw
        }
from h2_plant.components.water.water_treatment_detailed import WaterTank # Reuse simple tank logic

class CO2CaptureUnit(Component):
    """
    Separates CO2 from tail gas.
    """
    def __init__(self, capture_id: str, max_flow_kg_h: float, capture_rate: float = 0.9):
        super().__init__()
        self.capture_id = capture_id
        self.max_flow_kg_h = max_flow_kg_h
        self.capture_rate = capture_rate
        self.tail_gas_input_kg_h = 0.0
        self.tail_gas_stream = None # Store reference to stream for composition
        self.co2_captured_kg_h = 0.0
        self.recycled_gas_kg_h = 0.0
        self.power_kw = 0.0
        
    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        super().initialize(dt, registry)
        
    def step(self, t: float) -> None:
        super().step(t)
        input_flow = min(self.tail_gas_input_kg_h, self.max_flow_kg_h)
        
        # Determine CO2 content from stream composition
        co2_fraction = 0.5 # Fallback
        if self.tail_gas_stream and self.tail_gas_stream.composition:
            co2_fraction = self.tail_gas_stream.composition.get('CO2', 0.0)
            
        co2_content = input_flow * co2_fraction
        self.co2_captured_kg_h = co2_content * self.capture_rate
        self.recycled_gas_kg_h = input_flow - self.co2_captured_kg_h
        
        # Energy penalty: ~3 MJ/kg CO2 captured -> 0.83 kWh/kg
        # Power (kW) = Flow (kg/h) * Energy (kWh/kg)
        self.power_kw = self.co2_captured_kg_h * 0.83
        
    def get_state(self) -> Dict[str, Any]:
        return {
            **super().get_state(),
            'co2_captured_kg_h': float(self.co2_captured_kg_h),
            'recycled_gas_kg_h': float(self.recycled_gas_kg_h),
            'power_kw': float(self.power_kw)
        }

    def get_output(self, port_name: str) -> Any:
        """Get output from specific port."""
        if port_name in ['co2_out', 'out']:
            from h2_plant.core.stream import Stream
            return Stream(
                mass_flow_kg_h=self.co2_captured_kg_h,
                temperature_k=300.0,
                pressure_pa=1e5,
                composition={'CO2': 1.0},
                phase='gas'
            )
        else:
            raise ValueError(f"Unknown output port '{port_name}' on {self.capture_id}")

    def receive_input(self, port_name: str, value: Any, resource_type: str) -> float:
        """Receive input."""
        if port_name in ['tail_gas_in', 'in']:
            if hasattr(value, 'mass_flow_kg_h'):
                self.tail_gas_input_kg_h = value.mass_flow_kg_h
                self.tail_gas_stream = value
            else:
                self.tail_gas_input_kg_h = float(value)
                self.tail_gas_stream = None
            return self.tail_gas_input_kg_h
        return 0.0

class DetailedCO2Capture(CompositeComponent):
    """
    Complete CO2 Capture system.
    """
    def __init__(self, max_flow_kg_h: float = 500.0, storage_capacity_kg: float = 50000.0):
        super().__init__()
        
        self.add_subsystem('capture_co2c', CO2CaptureUnit('CO2C', max_flow_kg_h))
        self.add_subsystem('compressor_c5', ProcessCompressor('C-5', max_flow_kg_h))
        self.add_subsystem('storage_co2s', WaterTank('CO2S', storage_capacity_kg)) # Reuse tank
        
        self.tail_gas_input_kg_h = 0.0
        
    def step(self, t: float) -> None:
        Component.step(self, t)
        
        # 1. Capture
        self.capture_co2c.tail_gas_input_kg_h = self.tail_gas_input_kg_h
        self.capture_co2c.step(t)
        
        # 2. Compression
        self.compressor_c5.input_flow_kg_h = self.capture_co2c.co2_captured_kg_h
        self.compressor_c5.step(t)
        
        # 3. Storage
        self.storage_co2s.inlet_flow_kg_h = self.compressor_c5.output_flow_kg_h
        self.storage_co2s.outlet_flow_kg_h = 0.0 # Permanent storage
        self.storage_co2s.step(t)
        
    @property
    def recycled_gas_kg_h(self):
        return self.capture_co2c.recycled_gas_kg_h
        
    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state['summary'] = {
            'captured_co2_total_kg': self.storage_co2s.current_mass_kg,
            'current_capture_rate_kg_h': self.compressor_c5.output_flow_kg_h
        }
        return state
