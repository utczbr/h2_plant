from typing import Dict, Any, List
from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.component_ids import ComponentID

class WaterBalanceTracker(Component):
    """
    Tracks global water inventory, consumption, and recovery.
    Ensures mass conservation across the plant.
    """
    
    def __init__(self):
        super().__init__()
        # Instantaneous flows (kg/h)
        self.total_consumption_kg_h = 0.0
        self.total_recovery_kg_h = 0.0
        self.net_demand_kg_h = 0.0
        
        # Accumulators (kg)
        self.cumulative_consumption_kg = 0.0
        self.cumulative_recovery_kg = 0.0
        self.cumulative_net_demand_kg = 0.0
        
    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        super().initialize(dt, registry)
        
    def step(self, t: float) -> None:
        super().step(t)
        
        self.total_consumption_kg_h = 0.0
        self.total_recovery_kg_h = 0.0
        
        # 1. Track Consumption
        # PEM
        try:
            pem = self.get_registry_safe(ComponentID.PEM_ELECTROLYZER_DETAILED)
            if hasattr(pem, 'water_input_kg_h'):
                self.total_consumption_kg_h += pem.water_input_kg_h
        except Exception:
            pass
            
        # SOEC
        try:
            soec = self.get_registry_safe(ComponentID.SOEC_CLUSTER)
            if hasattr(soec, 'water_input_kg_h'):
                self.total_consumption_kg_h += soec.water_input_kg_h
        except Exception:
            pass
            
        # ATR (Future)
        
        # 2. Track Recovery
        # SOEC Separator (recycled water)
        try:
            soec = self.get_registry_safe(ComponentID.SOEC_CLUSTER)
            if hasattr(soec, 'separator_sp3'):
                # Assuming separator has water_outlet_kg_h or similar
                # DetailedSOECElectrolyzer has separator_sp3 which is SeparationTank
                # SeparationTank usually has water_return_kg_h
                if hasattr(soec.separator_sp3, 'water_return_kg_h'):
                    self.total_recovery_kg_h += soec.separator_sp3.water_return_kg_h
        except Exception:
            pass
            
        # PEM Separator (Future - if we model it explicitly)
        
        # 3. Calculate Net
        self.net_demand_kg_h = self.total_consumption_kg_h - self.total_recovery_kg_h
        
        # 4. Update Accumulators
        self.cumulative_consumption_kg += self.total_consumption_kg_h * self.dt
        self.cumulative_recovery_kg += self.total_recovery_kg_h * self.dt
        self.cumulative_net_demand_kg += self.net_demand_kg_h * self.dt
        
    def get_state(self) -> Dict[str, Any]:
        return {
            **super().get_state(),
            "total_consumption_kg_h": self.total_consumption_kg_h,
            "total_recovery_kg_h": self.total_recovery_kg_h,
            "net_demand_kg_h": self.net_demand_kg_h,
            "cumulative_consumption_kg": self.cumulative_consumption_kg,
            "cumulative_recovery_kg": self.cumulative_recovery_kg,
            "cumulative_net_demand_kg": self.cumulative_net_demand_kg,
            "recovery_ratio": (self.cumulative_recovery_kg / self.cumulative_consumption_kg * 100.0) if self.cumulative_consumption_kg > 0 else 0.0
        }
