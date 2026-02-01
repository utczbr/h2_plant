from typing import Dict, Any
from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry

class WaterQualityTestBlock(Component):
    """Records water quality parameters from incoming feed."""
    
    def __init__(self, sample_interval_hours: float = 1.0):
        super().__init__()
        self.sample_interval_hours = sample_interval_hours
        self.inlet_flow_m3h = 0.0
        self.inlet_temp_c = 20.0
        self.inlet_pressure_bar = 0.5
        self.purity_ppm = 0.0
        
    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        super().initialize(dt, registry)
        if self.sample_interval_hours <= 0:
            raise ValueError("Sample interval must be positive")
    
    def step(self, t: float) -> None:
        super().step(t)
        # Placeholder for reading external data
        pass
        
    def get_state(self) -> Dict[str, Any]:
        return {
            **super().get_state(),
            "inlet_flow_m3h": float(self.inlet_flow_m3h),
            "inlet_temp_c": float(self.inlet_temp_c),
            "inlet_pressure_bar": float(self.inlet_pressure_bar),
            "purity_ppm": float(self.purity_ppm),
            "flows": {
                "inputs": {
                    "raw_water": {
                        "value": self.inlet_flow_m3h,
                        "unit": "m3/h",
                        "source": "external_network",
                        "flowtype": "WATER_MASS"
                    }
                },
                "outputs": {
                    "tested_water": {
                        "value": self.inlet_flow_m3h,
                        "unit": "m3/h",
                        "destination": "water_treatment",
                        "flowtype": "WATER_MASS"
                    }
                }
            }
        }
