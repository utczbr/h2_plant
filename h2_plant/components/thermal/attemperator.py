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
from h2_plant.optimization.numba_ops import solve_temperature_from_enthalpy_jit

logger = logging.getLogger(__name__)

class Attemperator(Component):
    """
    Spray Attemperator for steam temperature control.
    
    Attributes:
        target_temp_k (float): Temperature setpoint for outlet steam.
        max_water_flow_kg_h (float): Maximum capacity of the spray valve.
        min_superheat_delta_k (float): Safety margin above saturation (default 5K).
        pressure_drop_bar (float): Steam side pressure drop.
        pipe_diameter_m (Optional[float]): Internal pipe diameter for velocity checks.
        heat_loss_kw (float): Fixed heat loss to ambient (default 0.0).
    """

    def __init__(
        self,
        component_id: str,
        target_temp_k: float,
        max_water_flow_kg_h: float = 1000.0,
        pressure_drop_bar: float = 0.0,
        min_superheat_delta_k: float = 5.0,
        pipe_diameter_m: Optional[float] = None,
        volume_m3: float = 0.05,
        heat_loss_kw: float = 0.0
    ):
        super().__init__()
        self.component_id = component_id
        self.target_temp_k = target_temp_k
        self.max_water_flow_kg_h = max_water_flow_kg_h
        self.pressure_drop_bar = pressure_drop_bar
        self.min_superheat_delta_k = min_superheat_delta_k
        self.pipe_diameter_m = pipe_diameter_m
        self.volume_m3 = volume_m3
        self.heat_loss_kw = heat_loss_kw
        
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
        self.steam_velocity_m_s = 0.0
        self.residence_time_s = 0.0
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
        
        # === DYNAMIC PRESSURE DROP (Venturi / Square Law) ===
        # Standard: dP varies with flow squared.
        # Design point calibration: 3600 kg/h @ pressure_drop_bar.
        DESIGN_FLOW = 3600.0   # kg/h (matches pump capacity)
        DESIGN_DP = self.pressure_drop_bar  # bar (instance param, e.g. 0.5)
        MIN_DP = 0.05          # bar (minimum resistance at low flow)
        
        # Ratio of current flow to design flow
        flow_ratio = s_in.mass_flow_kg_h / DESIGN_FLOW
        
        # Calculate dynamic drop: dP = dP_design * (Q / Q_design)^2
        # Clamp flow_ratio to 2.0 to prevent explosion during startup transients
        dynamic_dp_bar = DESIGN_DP * (min(flow_ratio, 2.0) ** 2)
        
        # Enforce minimum resistance
        dynamic_dp_bar = max(MIN_DP, dynamic_dp_bar)
        
        # Set Outlet Pressure (Main Phase Dominant)
        P_out = max(1e5, s_in.pressure_pa - (dynamic_dp_bar * 1e5))
        
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
            h_sat_liq = sat_props['h_f_Jkg']
        
        # Enforce Minimum Superheat (Target cannot be below T_sat + margin)
        safe_target_T = max(self.target_temp_k, T_sat + self.min_superheat_delta_k)
        
        # 3. Calculate Required Water (Adiabatic Energy Balance)
        required_water_kg_h = 0.0
        
        # Only inject if steam is hotter than target AND we have hydraulic head
        if s_in.temperature_k > safe_target_T and can_inject:
            # Get Enthalpies
            h_s = 0.0
            if self.lut_manager:
                 h_s = self.lut_manager.lookup('H2O', 'H', P_out, s_in.temperature_k)
            else:
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
                      w_h = self.lut_manager.lookup('H2O', 'H', w_in.pressure_pa, w_temp)
            
            if w_h == 0.0:
                w_h = 4184.0 * (w_temp - 273.15) # Fallback Cp_liq * dT
            
            # Denominator check (Target enthalpy must be > Water enthalpy)
            denom = h_target - w_h
            if denom > 1000.0: # Avoid div/0
                # Q_loss (J/h) = kW * 3.6e6
                q_loss_jh = self.heat_loss_kw * 3.6e6
                
                # Energy Balance: m_s*(h_s - h_tgt) - Q_loss = m_w*(h_tgt - h_w)
                numerator = s_in.mass_flow_kg_h * (h_s - h_target) - q_loss_jh
                required_water_kg_h = numerator / denom
        
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
        
        # Determine Enthalpies for Mixing (LUT Preferred)
        h_s_mix = s_in.specific_enthalpy_j_kg
        if self.lut_manager:
              h_s_mix = self.lut_manager.lookup('H2O', 'H', P_out, s_in.temperature_k)
        
        w_h_mix = 0.0
        if w_in:
             w_h_mix = w_in.specific_enthalpy_j_kg
             if w_h_mix == 0.0 and self.lut_manager:
                  w_h_mix = self.lut_manager.lookup('H2O', 'H', w_in.pressure_pa, w_temp)
        if w_h_mix == 0.0:
            w_h_mix = 4184.0 * (w_temp - 273.15)

        if total_mass > 0:
            # Enthalpy Mixing with Heat Loss
            q_loss_jh = self.heat_loss_kw * 3.6e6
            total_enthalpy_flow = (s_in.mass_flow_kg_h * h_s_mix + 
                                  required_water_kg_h * w_h_mix - 
                                  q_loss_jh)
            
            h_mix = total_enthalpy_flow / total_mass
            
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
                    # JIT Compiled Solver (Rigorous and Fast)
                    # Solves T = f(h, P) using bilinear interpolation and Newton-Raphson
                    T_final = solve_temperature_from_enthalpy_jit(
                         h_target=h_mix,
                         pressure_pa=P_out,
                         T_guess=safe_target_T, 
                         P_grid=self.lut_manager._pressure_grid,
                         T_grid=self.lut_manager._temperature_grid,
                         H_lut=self.lut_manager._luts['H2O']['H'],
                         C_lut=self.lut_manager._luts['H2O']['C']
                    )
                
                self.outlet_superheat_k = T_final - T_sat
                phase = 'gas'
            
            # Update output stream
            self.output_stream = Stream(
                mass_flow_kg_h=total_mass,
                temperature_k=T_final,
                pressure_pa=P_out,
                composition={'H2O': 1.0},
                phase=phase
            )
            
            # === Velocity Calculation (Hydraulic Check) ===
            # Only perform if diameter is provided AND LUT is active
            if self.pipe_diameter_m is not None and self.lut_manager:
                try:
                    # 1. Calculate Cross-sectional Area
                    area_m2 = 3.14159265 * (self.pipe_diameter_m / 2.0) ** 2
                    
                    # 2. Lookup Steam Density at outlet conditions
                    rho_steam = self.lut_manager.lookup(
                        'H2O', 'D', P_out, T_final
                    )
                    
                    # 3. Calculate Velocity (m/s)
                    # mass_flow is kg/h, convert to kg/s -> / 3600
                    if rho_steam > 1e-3 and area_m2 > 1e-9:
                        m_dot_s = total_mass / 3600.0
                        self.steam_velocity_m_s = m_dot_s / (rho_steam * area_m2)
                    else:
                        self.steam_velocity_m_s = 0.0
                    
                    # 4. Engineering Checks (Industrial Design Limits)
                    if self.steam_velocity_m_s > 60.0:
                        logger.warning(
                            f"Attemperator {self.component_id}: High steam velocity "
                            f"({self.steam_velocity_m_s:.1f} m/s) - erosion/vibration risk"
                        )
                    elif self.steam_velocity_m_s < 5.0 and self.steam_velocity_m_s > 0.1:
                        logger.warning(
                            f"Attemperator {self.component_id}: Low velocity "
                            f"({self.steam_velocity_m_s:.1f} m/s) - poor atomization risk"
                        )
                        
                except Exception as e:
                    logger.debug(f"Attemperator {self.component_id}: Could not calc velocity: {e}")
                    self.steam_velocity_m_s = 0.0
            
            # Consume inputs
            self.steam_in_buffer = None
            self.water_in_buffer = None  # Fully processed (split into injected + drain)
            
        # === NEW PHYSICS: Residence Time Calculation ===
        # Performed after mixing to use final density and flow
        if self.output_stream and self.lut_manager:
            try:
                # 1. Get Outlet Density (Real Gas)
                # T_final and P_out are calculated in the existing mixing logic
                rho_mix = self.lut_manager.lookup(
                    'H2O', 'D', self.output_stream.pressure_pa, self.output_stream.temperature_k
                )
                
                # 2. Calculate Volumetric Flow (m3/s)
                # mass_flow is in kg/h -> convert to kg/s
                m_dot_s = self.output_stream.mass_flow_kg_h / 3600.0
                if rho_mix > 1e-3:
                    vol_flow_m3_s = m_dot_s / rho_mix
                    
                    # 3. Calculate Residence Time (tau = V / Q)
                    if vol_flow_m3_s > 1e-6:
                        self.residence_time_s = self.volume_m3 / vol_flow_m3_s
                    else:
                        self.residence_time_s = 0.0
            except Exception:
                self.residence_time_s = 0.0
            
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
            'is_saturated': self.is_saturated,
            'steam_velocity_m_s': self.steam_velocity_m_s,
            'residence_time_s': self.residence_time_s,
            'volume_m3': self.volume_m3
        }

    def _set_idle_state(self):
        self.output_stream = Stream(0.0)
        self.drain_stream = None
        self.water_injected_kg_h = 0.0
        self.outlet_superheat_k = 0.0
        self.steam_velocity_m_s = 0.0
        self.residence_time_s = 0.0
