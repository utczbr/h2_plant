import numpy as np
from typing import Any, Dict, Tuple, Optional
from h2_plant.core.component import Component

from h2_plant.optimization.coolprop_lut import CoolPropLUT

# Try importing CoolProp (still needed for check, but we use LUT wrapper)
try:
    import CoolProp.CoolProp as CP
    COOLPROP_OK = True
except ImportError:
    COOLPROP_OK = False

class Compressor(Component):
    """
    Hydrogen Compressor Component.
    Implements multi-stage compression logic with intercooling.
    Based on legacy 'Compressor Armazenamento.py'.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.config = config
        
        # Parameters
        self.efficiency = config.get("efficiency", 0.65)
        self.cop = config.get("cop", 3.0) # Coefficient of Performance for cooling
        self.t_in_c = config.get("t_in_c", 10.0)
        self.t_max_c = config.get("t_max_c", 85.0)
        self.fluid = "H2"
        
        # State
        self.power_kw = 0.0
        self.heat_kw = 0.0
        self.outlet_temp_c = self.t_in_c
        self.stages = 1
        
        # Constants
        self.P_TO_PA = 1e5
        self.J_KG_TO_KWH_KG = 2.7778e-7

    def initialize(self, dt: float, registry: Any) -> None:
        """Initialize component."""
        super().initialize(dt, registry)
        self.P_TO_PA = 1e5
        self.J_KG_TO_KWH_KG = 2.7778e-7

    def _calculate_compression_energy(self, P_in_bar: float, P_out_bar: float) -> Tuple[float, int, float]:
        # ... (checks remain same) ...
        
        try:
            T_in_K = self.t_in_c + 273.15
            T_max_K = self.t_max_c + 273.15
            P_in_Pa = P_in_bar * self.P_TO_PA
            P_out_Pa = P_out_bar * self.P_TO_PA
            
            # 1. Determine Stages
            s1 = CoolPropLUT.PropsSI('S', 'P', P_in_Pa, 'T', T_in_K, self.fluid)
            h1 = CoolPropLUT.PropsSI('H', 'P', P_in_Pa, 'T', T_in_K, self.fluid)
            
            try:
                P_out_1s_max_T = CoolPropLUT.PropsSI('P', 'S', s1, 'T', T_max_K, self.fluid)
                r_stage_max = P_out_1s_max_T / P_in_Pa
            except:
                r_stage_max = 2.0 
                
            r_stage_max = max(1.5, r_stage_max)
            r_total = P_out_Pa / P_in_Pa
            
            N_stages = int(np.ceil(np.log(r_total) / np.log(r_stage_max)))
            N_stages = max(1, N_stages)
            
            # 2. Calculate Cycle
            W_comp_total = 0.0
            Q_removed_total = 0.0
            P_current = P_in_Pa
            r_stage = r_total**(1/N_stages)
            
            final_T_K = T_in_K
            
            for i in range(N_stages):
                P_next = P_current * r_stage
                if i == N_stages - 1: P_next = P_out_Pa
                
                # Isentropic compression
                s_in_i = CoolPropLUT.PropsSI('S', 'P', P_current, 'T', T_in_K, self.fluid)
                h_in_i = CoolPropLUT.PropsSI('H', 'P', P_current, 'T', T_in_K, self.fluid)
                
                h_out_s = CoolPropLUT.PropsSI('H', 'P', P_next, 'S', s_in_i, self.fluid)
                w_s = h_out_s - h_in_i
                w_a = w_s / self.efficiency
                
                h_out_a = h_in_i + w_a
                W_comp_total += w_a
                
                # Temp check
                T_out_a = CoolPropLUT.PropsSI('T', 'P', P_next, 'H', h_out_a, self.fluid)
                final_T_K = T_out_a
                
                # Intercooling
                if i < N_stages - 1:
                    h_cooled = CoolPropLUT.PropsSI('H', 'P', P_next, 'T', T_in_K, self.fluid)
                    q_removed = h_out_a - h_cooled
                    Q_removed_total += q_removed
                    P_current = P_next
            
            W_cooling = Q_removed_total / self.cop
            W_total_J_kg = W_comp_total + W_cooling
            
            return W_total_J_kg * self.J_KG_TO_KWH_KG, N_stages, final_T_K - 273.15
            
        except Exception as e:
            return 0.0, 1, self.t_in_c

    def step(self, t: float, mass_flow_kg_h: float = 0.0, p_in_bar: float = 30.0, p_out_bar: float = 200.0) -> None:
        """
        Update compressor state.
        Args:
            t: Simulation time
            mass_flow_kg_h: Mass flow rate
            p_in_bar: Inlet pressure
            p_out_bar: Outlet pressure (target)
        """
        super().step(t)
        
        if mass_flow_kg_h <= 0:
            self.power_kw = 0.0
            self.heat_kw = 0.0
            self.outlet_temp_c = self.t_in_c
            return

        # Check for invalid or no-compression conditions
        if p_in_bar <= 0.1 or p_out_bar <= p_in_bar:
            self.power_kw = 0.0
            self.stages = 1
            self.outlet_temp_c = self.t_in_c
            return

        spec_energy, stages, t_out = self._calculate_compression_energy(p_in_bar, p_out_bar)
        
        self.power_kw = max(0.0, spec_energy * mass_flow_kg_h) # Ensure non-negative
        self.stages = stages
        self.outlet_temp_c = t_out
        
    def get_state(self) -> Dict[str, Any]:
        return {
            "power_kw": self.power_kw,
            "stages": self.stages,
            "outlet_temp_c": self.outlet_temp_c
        }
