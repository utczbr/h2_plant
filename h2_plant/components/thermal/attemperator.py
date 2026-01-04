"""
Attemperator (Desuperheater) Component.

An in-line direct contact heat exchanger that reduces superheated steam temperature
by injecting atomized liquid water (spray water).

Physics:
    - Mass Balance: m_out = m_steam + m_water_injected
    - Excess Water: m_drain = m_water_in - m_water_injected
    - Energy Balance: m_out * h_out = m_steam * h_steam + m_water_injected * h_water
    - Control Logic: Calculates required m_water to achieve a target T_out setpoint.

Applications:
    - Main steam temperature control before turbines.
    - Syngas cooling in reforming loops.
    - Process steam conditioning.
"""

import logging
from typing import Dict, Any, Optional

from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.stream import Stream

logger = logging.getLogger(__name__)

class Attemperator(Component):
    """
    Spray Attemperator for steam temperature control.
    
    Attributes:
        target_temp_k (float): Temperature setpoint for outlet steam.
        max_water_flow_kg_h (float): Maximum capacity of the spray valve.
        min_superheat_delta_k (float): Safety margin above saturation (default 5K).
        pressure_drop_bar (float): Steam side pressure drop.
    """

    def __init__(
        self,
        component_id: str,
        target_temp_k: float,
        max_water_flow_kg_h: float = 1000.0,
        pressure_drop_bar: float = 0.5,
        min_superheat_delta_k: float = 5.0
    ):
        super().__init__()
        self.component_id = component_id
        self.target_temp_k = target_temp_k
        self.max_water_flow_kg_h = max_water_flow_kg_h
        self.pressure_drop_bar = pressure_drop_bar
        self.min_superheat_delta_k = min_superheat_delta_k
        
        # Dependencies
        self.lut_manager = None
        
        # Internal State
        self.steam_in_buffer: Optional[Stream] = None
        self.water_in_buffer: Optional[Stream] = None
        self.output_stream: Optional[Stream] = None
        self.drain_stream: Optional[Stream] = None
        
        # Metrics
        self.water_injected_kg_h = 0.0
        self.outlet_superheat_k = 0.0
        self.is_saturated = False

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """
        Initialize component and link to thermodynamics engine.
        """
        super().initialize(dt, registry)
        
        if registry.has('lut_manager'):
            self.lut_manager = registry.get('lut_manager')
        else:
            logger.warning(f"Attemperator {self.component_id}: LUTManager not found. "
                          "Thermodynamics will be approximate (Ideal Gas).")

    def step(self, t: float) -> None:
        """
        Execute control logic and mixing physics.
        """
        super().step(t)
        
        # 1. Check Inlets
        if not self.steam_in_buffer or self.steam_in_buffer.mass_flow_kg_h <= 1e-6:
            self._set_idle_state()
            return

        s_in = self.steam_in_buffer
        w_in = self.water_in_buffer
        
        # Assume water is available if connected; if not, flow is 0
        w_avail_flow = w_in.mass_flow_kg_h if w_in else 0.0
        w_temp = w_in.temperature_k if w_in else 298.15
        
        # Pressure Calculation
        P_out = max(1e5, s_in.pressure_pa - (self.pressure_drop_bar * 1e5))
        
        # Hydraulic Check: Can we inject?
        # Assuming regulated nozzle, but supply pressure must be > P_out
        can_inject = True
        if w_in and w_in.pressure_pa < P_out:
            can_inject = False
            # logger.warning(f"Attemperator {self.component_id}: Injection pressure too low ({w_in.pressure_pa:.0f} < {P_out:.0f} Pa)")

        # 2. Thermodynamic Bounds Check (Saturation)
        T_sat = 373.15 # Fallback
        h_sat_vap = 2676000.0  # h_g
        h_sat_liq = 419000.0   # h_f
        
        if self.lut_manager:
            sat_props = self.lut_manager.get_saturation_properties(P_out)
            T_sat = sat_props['T_sat_K']
            h_sat_vap = sat_props['h_g_Jkg']
            h_sat_liq = sat_props['h_l_Jkg']
        
        # Enforce Minimum Superheat (Target cannot be below T_sat + margin)
        safe_target_T = max(self.target_temp_k, T_sat + self.min_superheat_delta_k)
        
        # 3. Calculate Required Water (Adiabatic Energy Balance)
        required_water_kg_h = 0.0
        
        # Only inject if steam is hotter than target AND we have hydraulic head
        if s_in.temperature_k > safe_target_T and can_inject:
            # Get Enthalpies
            h_s = s_in.specific_enthalpy_j_kg
            
            # Estimate h_target at P_out, safe_target_T
            h_target = 0.0
            if self.lut_manager:
                h_target = self.lut_manager.lookup('H2O', 'H', P_out, safe_target_T)
            else:
                # Ideal gas fallback (Cp ~ 2.0 kJ/kgK for steam)
                h_target = h_s - 2000.0 * (s_in.temperature_k - safe_target_T)
            
            # Calculate Water Enthalpy if not provided (assume liquid at 300K if buffer empty)
            w_h = 0.0
            if w_in:
                 w_h = w_in.specific_enthalpy_j_kg
                 if w_h == 0.0 and self.lut_manager:
                      w_h = self.lut_manager.lookup('Water', 'H', w_in.pressure_pa, w_temp)
            
            if w_h == 0.0:
                w_h = 4184.0 * (w_temp - 273.15) # Fallback Cp_liq * dT
            
            # Denominator check (Target enthalpy must be > Water enthalpy)
            denom = h_target - w_h
            if denom > 1000.0: # Avoid div/0
                required_water_kg_h = s_in.mass_flow_kg_h * (h_s - h_target) / denom
        
        # 4. Apply Constraints
        required_water_kg_h = max(0.0, required_water_kg_h)
        required_water_kg_h = min(required_water_kg_h, self.max_water_flow_kg_h)
        if w_in:
            required_water_kg_h = min(required_water_kg_h, w_in.mass_flow_kg_h)
            
        self.water_injected_kg_h = required_water_kg_h
        
        # Handle Excess Water (Drain)
        drain_flow = w_avail_flow - self.water_injected_kg_h
        if drain_flow > 1e-6:
             self.drain_stream = Stream(
                 mass_flow_kg_h=drain_flow,
                 temperature_k=w_temp,
                 pressure_pa=w_in.pressure_pa if w_in else 101325.0,
                 composition={'H2O': 1.0},
                 phase='liquid'
             )
        else:
             self.drain_stream = None

        # 5. Calculate Final State
        total_mass = s_in.mass_flow_kg_h + required_water_kg_h
        
        # Determine water enthalpy for mixing
        w_h_mix = 0.0
        if w_in:
             w_h_mix = w_in.specific_enthalpy_j_kg
             if w_h_mix == 0.0: # If stream didn't have it set
                  w_h_mix = 4184.0 * (w_temp - 273.15) # Approximate
        else:
            w_h_mix = 4184.0 * (w_temp - 273.15)

        if total_mass > 0:
            h_mix = (s_in.mass_flow_kg_h * s_in.specific_enthalpy_j_kg + 
                     required_water_kg_h * w_h_mix) / total_mass
            
            # Phase Stability Check
            if h_mix <= h_sat_vap:
                # Saturation / Wet Steam
                T_final = T_sat
                self.is_saturated = True
                self.outlet_superheat_k = 0.0
                phase = 'mixed'
                # Optionally calculate vapor quality x = (h - hf) / hfg
            else:
                # Superheated Region - Solve for Temperature
                self.is_saturated = False
                T_final = safe_target_T # Initial guess
                
                if self.lut_manager:
                    # Simple Newton Method
                    T_curr = safe_target_T
                    for _ in range(5):
                        h_curr = self.lut_manager.lookup('H2O', 'H', P_out, T_curr)
                        cp_curr = self.lut_manager.lookup('H2O', 'C', P_out, T_curr)
                        if cp_curr < 100: cp_curr = 2000.0 # Sanity check
                        err = h_mix - h_curr
                        if abs(err) < 100: break
                        T_curr += err / cp_curr
                    T_final = T_curr
                
                self.outlet_superheat_k = T_final - T_sat
                phase = 'gas'
            
            # Update output stream
            self.output_stream = Stream(
                mass_flow_kg_h=total_mass,
                temperature_k=T_final,
                pressure_pa=P_out,
                composition={'H2O': 1.0},
                phase=phase,
                specific_enthalpy_j_kg=h_mix
            )
            
            # Consume inputs
            self.steam_in_buffer = None
            self.water_in_buffer = None # Fully processed (split into injected + drain)
            
        else:
            self._set_idle_state()

    def receive_input(self, port_name: str, value: Any, resource_type: str = None) -> float:
        if port_name == 'steam_in' and isinstance(value, Stream):
            self.steam_in_buffer = value
            return value.mass_flow_kg_h
        elif port_name == 'water_in' and isinstance(value, Stream):
            self.water_in_buffer = value
            return value.mass_flow_kg_h
        return 0.0

    def get_output(self, port_name: str) -> Optional[Stream]:
        if port_name == 'steam_out':
            return self.output_stream
        elif port_name == 'water_drain':
            return self.drain_stream
        return None

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        return {
            'steam_in': {'type': 'input', 'resource_type': 'steam', 'units': 'kg/h'},
            'water_in': {'type': 'input', 'resource_type': 'water', 'units': 'kg/h'},
            'steam_out': {'type': 'output', 'resource_type': 'steam', 'units': 'kg/h'},
            'water_drain': {'type': 'output', 'resource_type': 'water', 'units': 'kg/h', 'description': 'Excess water return'}
        }

    def get_state(self) -> Dict[str, Any]:
        return {
            **super().get_state(),
            'water_injected_kg_h': self.water_injected_kg_h,
            'outlet_temp_k': self.output_stream.temperature_k if self.output_stream else 0.0,
            'is_saturated': self.is_saturated
        }

    def _set_idle_state(self):
        self.output_stream = Stream(0.0)
        self.drain_stream = None
        self.water_injected_kg_h = 0.0
        self.outlet_superheat_k = 0.0
