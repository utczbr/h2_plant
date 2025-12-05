import numpy as np
from typing import Any, Dict, Tuple, Optional
from h2_plant.core.component import Component

from h2_plant.optimization.coolprop_lut import CoolPropLUT

# Try importing CoolProp
try:
    import CoolProp.CoolProp as CP
    COOLPROP_OK = True
except ImportError:
    COOLPROP_OK = False

class Pump(Component):

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.config = config
        
        self.eta_is = config.get("eta_is", 0.82)
        self.eta_m = config.get("eta_m", 0.96)
        self.fluid = "Water"
        
        # State
        self.power_kw = 0.0
        self.outlet_temp_c = 0.0
        self.flow_rate_kg_h = 0.0

    def initialize(self, dt: float, registry: Any) -> None:
        """Initialize component."""
        super().initialize(dt, registry)

    def step(self, t: float, mass_flow_kg_h: float = 0.0, p_in_bar: float = 1.0, p_out_bar: float = 30.0, t_in_c: float = 20.0) -> None:
        """
        Update pump state.
        """
        super().step(t)
        self.flow_rate_kg_h = mass_flow_kg_h
        
        if mass_flow_kg_h <= 0 or p_out_bar <= p_in_bar:
            self.power_kw = 0.0
            self.outlet_temp_c = t_in_c
            return

        if not COOLPROP_OK:
            # Simple hydraulic power calculation
            # Power (kW) = Q (m3/h) * dP (bar) * 100 / (3600 * eta) 
            # Approx density 1000 kg/m3
            vol_flow_m3_h = mass_flow_kg_h / 1000.0
            dp_bar = p_out_bar - p_in_bar
            # 1 bar = 100 kPa. 1 m3/h * 1 bar = 100000 Pa * 1/3600 m3/s = 27.7 W
            hydraulic_power_kw = (vol_flow_m3_h * dp_bar * 100) / 3600.0
            self.power_kw = hydraulic_power_kw / (self.eta_is * self.eta_m)
            self.outlet_temp_c = t_in_c # Ignore temp rise in fallback
            return

        try:
            P_in_Pa = p_in_bar * 1e5
            P_out_Pa = p_out_bar * 1e5
            T_in_K = t_in_c + 273.15
            
            # 1. Inlet State
            h1 = CoolPropLUT.PropsSI('H', 'P', P_in_Pa, 'T', T_in_K, self.fluid)
            s1 = CoolPropLUT.PropsSI('S', 'P', P_in_Pa, 'T', T_in_K, self.fluid)
            
            # 2. Isentropic Outlet
            h2s = CoolPropLUT.PropsSI('H', 'P', P_out_Pa, 'S', s1, self.fluid)
            
            # 3. Real Work
            w_s = h2s - h1 # J/kg
            w_real = w_s / self.eta_is
            
            h2 = h1 + w_real
            
            # 4. Outlet Temp
            T2_K = CoolPropLUT.PropsSI('T', 'P', P_out_Pa, 'H', h2, self.fluid)
            self.outlet_temp_c = T2_K - 273.15
            
            # 5. Power
            mass_flow_kg_s = mass_flow_kg_h / 3600.0
            power_fluid_w = mass_flow_kg_s * w_real
            self.power_kw = (power_fluid_w / 1000.0) / self.eta_m
            
        except Exception as e:
            # Fallback
            self.power_kw = 0.0
            self.outlet_temp_c = t_in_c

    def get_state(self) -> Dict[str, Any]:
        return {
            "power_kw": self.power_kw,
            "outlet_temp_c": self.outlet_temp_c,
            "flow_rate_kg_h": self.flow_rate_kg_h
        }
