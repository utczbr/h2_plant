from typing import Dict, Any, Optional
from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.component_ids import ComponentID

class ThermalManager(Component):
    """
    Centralized Thermal Management System.
    
    Responsibilities:
    1. Collect waste heat from high-temperature sources (PEM, Compressors).
    2. Distribute heat to heat sinks (SOEC Steam Gen, ATR Pre-heat).
    3. Track thermal efficiency and waste heat.
    """
    
    def __init__(self):
        super().__init__()
        # State
        self.total_heat_available_kw = 0.0
        self.total_heat_demand_kw = 0.0
        self.heat_utilized_kw = 0.0
        self.heat_wasted_kw = 0.0
        
        # Accumulators
        self.cumulative_heat_recovered_kwh = 0.0
        self.cumulative_heat_utilized_kwh = 0.0
        self.cumulative_heat_wasted_kwh = 0.0
        
    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        super().initialize(dt, registry)
        
    def step(self, t: float) -> None:
        super().step(t)
        
        self.total_heat_available_kw = 0.0
        self.total_heat_demand_kw = 0.0
        self.heat_utilized_kw = 0.0
        
        # 1. Collect Heat (Sources)
        # PEM Electrolyzer
        try:
            pem = self.get_registry_safe(ComponentID.PEM_ELECTROLYZER_DETAILED)
            if hasattr(pem, 'heat_output_kw'):
                self.total_heat_available_kw += pem.heat_output_kw
        except Exception:
            pass
            
        # Compressors
        # Iterate over all components to find compressors with heat output
        # Or use specific IDs if known. PlantBuilder registers 'compressor_i'.
        # Let's search for components with 'heat_output_kw' that are not PEM
        for comp_id, comp in self._registry._components.items():
             if "compressor" in comp_id and hasattr(comp, 'heat_output_kw'):
                 self.total_heat_available_kw += comp.heat_output_kw
        
        # 2. Identify Demand (Sinks)
        # SOEC Steam Generator
        steam_gen = None
        try:
            soec = self.get_registry_safe(ComponentID.SOEC_CLUSTER)
            # Access internal steam generator state if possible, or exposed property
            # DetailedSOECElectrolyzer has 'steam_gen_hx4' subsystem
            if hasattr(soec, 'steam_gen_hx4'):
                steam_gen = soec.steam_gen_hx4
                # Prefer total_heat_demand_kw (new attribute), fallback to heat_input_kw (old)
                if hasattr(steam_gen, 'total_heat_demand_kw'):
                    self.total_heat_demand_kw += steam_gen.total_heat_demand_kw
                elif hasattr(steam_gen, 'heat_input_kw'):
                    self.total_heat_demand_kw += steam_gen.heat_input_kw
        except Exception:
            pass
            
        # ATR (Future)
        
        # 3. Distribute Heat
        # Simple logic: Supply as much demand as possible from available
        self.heat_utilized_kw = min(self.total_heat_available_kw, self.total_heat_demand_kw)
        self.heat_wasted_kw = max(0.0, self.total_heat_available_kw - self.heat_utilized_kw)
        
        # Actively set external heat input on sinks
        if steam_gen:
            # For now, assume SOEC is the only sink, so it gets all utilized heat
            # If multiple sinks, we'd need allocation logic (proportional or priority)
            if hasattr(steam_gen, 'external_heat_input_kw'):
                steam_gen.external_heat_input_kw = self.heat_utilized_kw
        
        # 4. Update Accumulators
        self.cumulative_heat_recovered_kwh += self.total_heat_available_kw * self.dt
        self.cumulative_heat_utilized_kwh += self.heat_utilized_kw * self.dt
        self.cumulative_heat_wasted_kwh += self.heat_wasted_kw * self.dt
        
    def get_state(self) -> Dict[str, Any]:
        return {
            **super().get_state(),
            "total_heat_available_kw": self.total_heat_available_kw,
            "total_heat_demand_kw": self.total_heat_demand_kw,
            "heat_utilized_kw": self.heat_utilized_kw,
            "heat_wasted_kw": self.heat_wasted_kw,
            "cumulative_heat_recovered_kwh": self.cumulative_heat_recovered_kwh,
            "cumulative_heat_utilized_kwh": self.cumulative_heat_utilized_kwh,
            "thermal_utilization_efficiency": (self.cumulative_heat_utilized_kwh / self.cumulative_heat_recovered_kwh * 100.0) if self.cumulative_heat_recovered_kwh > 0 else 0.0
        }
