"""
ATR Thermal Components.
Contains discrete heat exchangers/boilers for heat recovery reporting.
"""

import logging
from typing import Dict, Any
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.stream import Stream
from h2_plant.components.reforming.atr_data_manager import ATRBaseComponent

logger = logging.getLogger(__name__)

class Boiler(ATRBaseComponent):
    """
    Represents Heaters H01, H02, H04.
    Acts as a 'Sensor' that reports the heat duty available at specific 
    points in the plant based on the regression model.
    """
    def __init__(self, component_id: str = None, lookup_id: str = None):
        super().__init__()
        self.component_id = component_id
        self.lookup_id = lookup_id or component_id
        self.duty_kw = 0.0
        self.current_stream: Stream = None
        self.current_o2_flow_kmol_h = 15.0 # Default fallback

    def step(self, t: float) -> None:
        # Use current O2 flow to calculate duty
        self.duty_kw = self.data_manager.lookup(f'{self.lookup_id}_Q_func', self.current_o2_flow_kmol_h)

    def receive_input(self, port_name: str, value: Any, resource_type: str = None) -> float:
        if port_name == 'inlet' and isinstance(value, Stream):
            self.current_stream = value
            return value.mass_flow_kg_h
            
        elif port_name == 'o2_signal_in':
            # Signal input from ATR_Plant
            if isinstance(value, (int, float)):
                self.current_o2_flow_kmol_h = float(value)
            return 0.0
            
        return 0.0

    def get_output(self, port_name: str) -> Any:
        # Pass-through for fluid stream
        if port_name == 'outlet':
            return self.current_stream
            
        # Heat reporting
        if port_name == 'heat_out':
            return self.duty_kw
            
        return None
            
        return None

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        return {
            'inlet': {'type': 'input', 'resource_type': 'stream'},
            'o2_signal_in': {'type': 'input', 'resource_type': 'signal', 'units': 'kmol/h'},
            'outlet': {'type': 'output', 'resource_type': 'stream'},
            'heat_out': {'type': 'output', 'resource_type': 'heat', 'units': 'kW'}
        }
        
    def get_state(self) -> Dict[str, Any]:
        return {
            "component_id": self.component_id,
            "duty_kw": self.duty_kw
        }
