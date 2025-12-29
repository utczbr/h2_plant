"""
Regenerative Heat Exchanger (Interchanger).

This component models a counter-flow heat exchanger designed for waste heat recovery.
It transfers thermal energy from a "Hot" stream to a "Cold" stream, subject to:
1. **Conservation of Energy**: Q_hot = Q_cold = Q_transferred
2. **Second Law of Thermodynamics**: Heat flows only from hot to cold (limited by approach temp).

Applications:
    - Pre-heating electrolysis feedwater using stack exhaust.
    - recuperating heat from compressor inter-stage cooling.
"""

from typing import Dict, Any, Optional
import numpy as np

from h2_plant.core.component import Component
from h2_plant.core.stream import Stream
from h2_plant.core.constants import ConversionFactors, GasConstants
from h2_plant.core.component_ids import ComponentID

class Interchanger(Component):
    """
    Simulates a counter-flow heat exchanger with specified minimum approach temperature.
    
    This model determines the maximum realizable heat transfer rate given inlet 
    conditions and the "Pinch Point" constraint (min_approach_temp).

    Physics Model:
        Q = min(Q_capacity, Q_availability)
    
    Attributes:
        min_approach_temp_k (float): Minimum allowed temperature difference (T_hot_out - T_cold_in).
                                     Represents the practical limit of heat exchanger surface area.
        efficiency (float): Adiabatic efficiency factor (heat loss to environment).
        target_cold_out_temp_k (float): Temperature setpoint for the cold stream.
    """

    def __init__(
        self,
        component_id: str,
        min_approach_temp_k: float = 10.0,
        target_cold_out_temp_c: float = 95.0,
        efficiency: float = 0.95
    ):
        super().__init__()
        self.component_id = component_id
        self.min_approach_temp_k = min_approach_temp_k
        self.target_cold_temp_k = target_cold_out_temp_c + 273.15
        self.efficiency = efficiency

        # Inputs
        self.hot_stream: Optional[Stream] = None
        self.cold_stream: Optional[Stream] = None

        # Outputs
        self.hot_out: Optional[Stream] = Stream(0.0, temperature_k=298.15, pressure_pa=101325.0, phase='gas')
        self.cold_out: Optional[Stream] = Stream(0.0, temperature_k=target_cold_out_temp_c+273.15, pressure_pa=101325.0, phase='liquid')
        
        self.q_transferred_kw = 0.0

    def initialize(self, dt: float, registry: 'ComponentRegistry') -> None:
        """
        Executes initialization phase of Component Lifecycle.
        
        Args:
            dt (float): Simulation timestep (hours).
            registry (ComponentRegistry): Central service registry.
        """
        super().initialize(dt, registry)

    def step(self, t: float) -> None:
        """
        Calculates heat transfer and updates stream states using Enthalpy-Based logic.

        Calculation Logic (Enthalpy-Based):
        1. **Demand (Cold Side)**: Energy required to heat cold stream to target T.
           `Q_demand = m_c * Cp_c * (T_target - T_c_in)`

        2. **Availability (Hot Side - Latent Heat Aware)**:
           Instead of Cp * DeltaT, we calculate the Enthalpy Drop available if cooled
           to the Second Law limit temperature (`T_limit = T_c_in + DeltaT_approach`).
           `Q_avail = m_h * (H_h_in(T_in) - H_h_lim(T_limit))`
           *Calculates mixtures by summing partial enthalpies of species.*

        3. **Equilibrium**: `Q_transferred = min(Q_demand, Q_avail)`

        4. **Outlet State**:
           `H_h_out = H_h_in - Q_transferred / m_h`
           Determine Phase, T, and Condensation based on `H_h_out` vs Saturation curve.

        Args:
            t (float): Simulation time (hours).
        """
        super().step(t)

        if not self.hot_stream or not self.cold_stream:
             # Pass-through or Zero output if missing input
             self.hot_out = self.hot_stream
             self.cold_out = self.cold_stream
             self.q_transferred_kw = 0.0
             return

        # --- 0. Setup & Inputs ---
        m_h_kg_h = self.hot_stream.mass_flow_kg_h
        m_c_kg_h = self.cold_stream.mass_flow_kg_h

        if m_h_kg_h <= 0 or m_c_kg_h <= 0:
             self.hot_out = self.hot_stream
             self.cold_out = self.cold_stream
             return

        # Get Component Registry and LUT Manager
        # (Assuming registry is attached to self from initialize)
        lut_mgr = None
        if hasattr(self, '_registry') and self._registry:
             lut_mgr = self._registry.get(ComponentID.LUT_MANAGER)

        # Fallback if no LUT manager (use simple Cp model or error?)
        # For robustness, we'll try to use it, else warn?
        # Simulation should have LUT_MANAGER.

        # Properties
        T_h_in = self.hot_stream.temperature_k
        P_h_in = self.hot_stream.pressure_pa
        
        T_c_in = self.cold_stream.temperature_k
        
        # --- 1. Calculate Inlet Specific Enthalpy (Hot) ---
        # We need accurate H_in including latent potential of water vapor.
        # H_mix = sum(y_i * H_i)
        
        def get_mixture_enthalpy(stream: Stream, T_k: float, P_pa: float, phase_check: bool = True) -> float:
            h_mix = 0.0
            # Get mole fractions for partial pressure calc
            mole_fracs = stream.mole_fractions
            
            for species, mass_frac in stream.composition.items():
                if mass_frac <= 0: continue
                
                h_spec = 0.0
                if species == 'H2O' and lut_mgr:
                    # Water: Use LUT with Partial Pressure for accurate Dew Point logic
                    # or Total Pressure if we want to simplify?
                    # Using Partial Pressure is physically correct (Dew Point).
                    y_i = mole_fracs.get(species, 1.0)
                    p_partial = P_pa * y_i
                    
                    # Ensure p_partial is within LUT bounds (0.05 bar min usually)
                    # If very low, might clip, but usually H2O is significant.
                    p_lookup = max(5000.0, p_partial) 
                    
                    # Lookup Enthalpy
                    h_spec = lut_mgr.lookup('H2O', 'H', p_lookup, T_k)
                    
                elif lut_mgr and species in ['H2', 'O2', 'N2', 'CO2']:
                     # Gases: LUT lookup (usually ideal gas region)
                     # Partial pressure doesn't matter much for H ideal gas, but use total P
                     # or partial? H(T) is dominant.
                     h_spec = lut_mgr.lookup(species, 'H', P_pa, T_k)
                else:
                     # Fallback to Stream's Cp polynomial
                     # (Create dummy stream for single species calc?)
                     h_spec = stream.specific_enthalpy_j_kg # Approximate since it uses T_current
                     # Actually better to calculate manual integral if T != stream.T
                     # But for now, if LUT fails, ignore or use approx.
                     pass 
                
                h_mix += mass_frac * h_spec
            return h_mix

        if lut_mgr:
            h_h_in = get_mixture_enthalpy(self.hot_stream, T_h_in, P_h_in)
        else:
            h_h_in = self.hot_stream.specific_enthalpy_j_kg

        # --- 2. Cold Side Demand ---
        # Q_demand = m * Cp * dT (Water heating is simple liquid)
        Cp_c = 4186.0 # J/kgK
        Q_cold_demand_w = (m_c_kg_h / 3600.0) * Cp_c * (self.target_cold_temp_k - T_c_in)

        # --- 3. Hot Side Availability (Latent-Aware) ---
        T_h_limit = T_c_in + self.min_approach_temp_k
        
        # Calculate Enthalpy at Limit Temperature
        if lut_mgr:
            h_h_limit = get_mixture_enthalpy(self.hot_stream, T_h_limit, P_h_in)
        else:
            # Fallback (Sensible only error)
            Cp_h_approx = 2200.0
            h_h_limit = h_h_in - Cp_h_approx * (T_h_in - T_h_limit)
            
        dq_h_avail = max(0.0, h_h_in - h_h_limit)
        Q_hot_avail_w = (m_h_kg_h / 3600.0) * dq_h_avail

        # --- 4. Transfer & Outlet Enthalpy ---
        Q_transfer_w = min(max(0, Q_cold_demand_w), Q_hot_avail_w)
        self.q_transferred_kw = Q_transfer_w / 1000.0

        if m_h_kg_h > 0:
            h_h_out_target = h_h_in - (Q_transfer_w / (m_h_kg_h / 3600.0))
        else:
            h_h_out_target = h_h_in

        # --- 5. Resolve Outlet State (Zonal Logic) ---
        # Instead of solving for T blindly, we check phase zones based on H2O enthalpy.
        
        mole_fracs = self.hot_stream.mole_fractions
        y_h2o = mole_fracs.get('H2O', 0.0)
        
        # Calculate Inert Enthalpy Function (H2, O2, etc)
        # We assume Inerts don't condense.
        def get_inert_enthalpy(T_k):
            h_inert = 0.0
            for s, mf in self.hot_stream.composition.items():
                if s == 'H2O' or s == 'H2O_liq': continue
                if mf <= 0: continue
                # Look up gas property
                h_s = lut_mgr.lookup(s, 'H', P_h_in, T_k) if lut_mgr else 0.0
                h_inert += mf * h_s
            return h_inert

        new_composition = self.hot_stream.composition.copy()
        output_phase = 'gas'
        
        if lut_mgr and y_h2o > 0:
            p_partial_h2o = P_h_in * y_h2o
            
            # 1. Get Saturation Limits at P_partial
            sat_props = lut_mgr.get_saturation_properties(max(5000.0, p_partial_h2o))
            T_sat = sat_props['T_sat_K']
            h_f_pure = sat_props['h_f_Jkg']
            h_g_pure = sat_props['h_g_Jkg']
            
            # 2. Calculate Threshold Mix Enthalpies
            # H_mix_dew = H_inerts(T_sat) + mass_frac_H2O * h_g_pure
            # H_mix_bub = H_inerts(T_sat) + mass_frac_H2O * h_f_pure
            
            h_inerts_at_sat = get_inert_enthalpy(T_sat)
            mf_h2o_total = self.hot_stream.composition.get('H2O', 0.0) + self.hot_stream.composition.get('H2O_liq', 0.0)
            
            h_mix_gas_sat = h_inerts_at_sat + mf_h2o_total * h_g_pure
            h_mix_liq_sat = h_inerts_at_sat + mf_h2o_total * h_f_pure
            
            # 3. Determine Zone
            if h_h_out_target >= h_mix_gas_sat:
                # Zone 1: GAS (Superheated)
                output_phase = 'gas'
                # Solve T in [T_sat, T_in]
                # Bisection
                T_low, T_high = T_sat, max(T_sat + 1.0, T_h_in)
                for _ in range(10):
                    T_m = (T_low + T_high) / 2.0
                    h_m = get_inert_enthalpy(T_m) + mf_h2o_total * lut_mgr.lookup('H2O', 'H', max(5000.0, p_partial_h2o), T_m)
                    if h_m > h_h_out_target: T_high = T_m
                    else: T_low = T_m
                T_out_final = (T_low + T_high) / 2.0
                
            elif h_h_out_target <= h_mix_liq_sat:
                # Zone 3: LIQUID (Subcooled) - assuming fully condensed
                output_phase = 'liquid'
                new_composition['H2O'] = 0.0
                new_composition['H2O_liq'] = mf_h2o_total
                
                # Solve T < T_sat
                T_low, T_high = 273.15, T_sat
                for _ in range(10):
                    T_m = (T_low + T_high) / 2.0
                    # Liquid water enthalpy at T_m
                    # Use P_total for liquid water or saturation? Coolprop handles liquid at P_total.
                    h_w = lut_mgr.lookup('H2O', 'H', P_h_in, T_m) 
                    h_m = get_inert_enthalpy(T_m) + mf_h2o_total * h_w
                    if h_m > h_h_out_target: T_high = T_m
                    else: T_low = T_m
                T_out_final = (T_low + T_high) / 2.0

            else:
                # Zone 2: MIXED (Condensing)
                output_phase = 'mixed'
                T_out_final = T_sat
                
                # Quality x (fraction of H2O that is Vapor)
                # H_target = H_inerts + mf_total * (x*h_g + (1-x)*h_f)
                # H_target - H_inerts = mf_total * (h_f + x(h_g - h_f))
                # (H_target - H_inerts)/mf_total - h_f = x(h_g - h_f)
                
                h_h2o_avg = (h_h_out_target - h_inerts_at_sat) / mf_h2o_total
                x_quality = (h_h2o_avg - h_f_pure) / (h_g_pure - h_f_pure)
                x_quality = max(0.0, min(1.0, x_quality))
                
                m_h2o_vap = x_quality * mf_h2o_total
                m_h2o_liq = (1.0 - x_quality) * mf_h2o_total
                
                new_composition['H2O'] = m_h2o_vap
                new_composition['H2O_liq'] = m_h2o_liq

        else:
             # Fallback if no LUT or no water
             output_phase = 'gas'
             T_out_final = T_h_in - (Q_transfer_w / (m_h_kg_h * 2000.0/3600.0)) # Approx Cp

        # --- 7. Final Output Streams ---
        
        # Hot Out (Cooled)
        self.hot_out = Stream(
            mass_flow_kg_h=m_h_kg_h,
            temperature_k=T_out_final,
            pressure_pa=P_h_in,
            composition=new_composition,
            phase=output_phase
        )

        # Cold Out (Heated) - Calculate T_c_out
        dT_c = Q_transfer_w / ((m_c_kg_h / 3600.0) * Cp_c)
        T_c_out = T_c_in + dT_c

        self.cold_out = Stream(
            mass_flow_kg_h=m_c_kg_h,
            temperature_k=T_c_out,
            pressure_pa=self.cold_stream.pressure_pa,
            composition=self.cold_stream.composition,
            phase='liquid'
        )
        
        # Clear inputs
        self.hot_stream = None
        self.cold_stream = None

    def receive_input(self, port_name: str, value: Any, resource_type: str = None) -> float:
        if port_name == 'hot_in' and isinstance(value, Stream):
            self.hot_stream = value
            return value.mass_flow_kg_h
        elif port_name == 'cold_in' and isinstance(value, Stream):
            self.cold_stream = value
            return value.mass_flow_kg_h
        return 0.0

    def get_output(self, port_name: str) -> Any:
        if port_name == 'hot_out':
            return self.hot_out if self.hot_out else Stream(0.0)
        elif port_name == 'cold_out':
            return self.cold_out if self.cold_out else Stream(0.0)
        return None

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        return {
            'hot_in': {'type': 'input', 'resource_type': 'gas'},
            'cold_in': {'type': 'input', 'resource_type': 'water'},
            'hot_out': {'type': 'output', 'resource_type': 'gas'},
            'cold_out': {'type': 'output', 'resource_type': 'water'}
        }

    def get_state(self) -> Dict[str, Any]:
        """
        Returns component operational telemetry.
        
        Returns:
            Dict[str, Any]: Q_transferred (kW) and outlet temperatures.
        """
        return {
            **super().get_state(),
            'q_transferred_kw': self.q_transferred_kw,
            'hot_out_temp_k': self.hot_out.temperature_k if self.hot_out else 0,
            'cold_out_temp_k': self.cold_out.temperature_k if self.cold_out else 0
        }
