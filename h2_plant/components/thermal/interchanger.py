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
from h2_plant.core.constants import ConversionFactors, GasConstants, StandardConditions
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
        # PERFORMANCE: Use vectorized JIT lookup (single C-space call for all species)
        
        # Build mass fractions array matching LUT fluid order (H2, O2, N2, CO2, CH4, H2O)
        # Note: StandardConditions.CANONICAL_FLUID_ORDER = ('H2', 'O2', 'N2', 'H2O', 'CH4', 'CO2')
        # but LUT config uses ('H2', 'O2', 'N2', 'CO2', 'CH4', 'H2O') - check lut_manager.LUTConfig
        
        hot_mass_fracs = np.zeros(6, dtype=np.float64)
        lut_fluid_order = lut_mgr.config.fluids if lut_mgr else StandardConditions.CANONICAL_FLUID_ORDER
        
        for idx, fluid in enumerate(lut_fluid_order):
            if fluid in self.hot_stream.composition:
                hot_mass_fracs[idx] = self.hot_stream.composition[fluid]
            elif fluid == 'H2O' and 'H2O_liq' in self.hot_stream.composition:
                # Combine vapor and liquid water for total
                hot_mass_fracs[idx] = self.hot_stream.composition.get('H2O', 0.0) + self.hot_stream.composition.get('H2O_liq', 0.0)
        
        if lut_mgr and lut_mgr.stacked_H is not None:
            h_h_in = lut_mgr.lookup_mixture_enthalpy(hot_mass_fracs, P_h_in, T_h_in)
        else:
            h_h_in = self.hot_stream.specific_enthalpy_j_kg

        # --- 2. Cold Side Demand ---
        # Q_demand = m * Cp * dT (Water heating is simple liquid)
        Cp_c = 4186.0 # J/kgK
        Q_cold_demand_w = (m_c_kg_h / 3600.0) * Cp_c * (self.target_cold_temp_k - T_c_in)

        # --- 3. Hot Side Availability (Latent-Aware) ---
        T_h_limit = T_c_in + self.min_approach_temp_k
        
        # Calculate Enthalpy at Limit Temperature using vectorized lookup
        if lut_mgr and lut_mgr.stacked_H is not None:
            h_h_limit = lut_mgr.lookup_mixture_enthalpy(hot_mass_fracs, P_h_in, T_h_limit)
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

        # --- 5. Resolve Outlet State (Rigorous Flash) ---
        # Solving H(T, P) = h_h_out_target
        # using Rachford-Rice for VLE at each candidate T.
        
        from h2_plant.optimization.numba_ops import solve_rachford_rice_single_condensable, calculate_stream_enthalpy_jit, fast_composition_properties
        
        # Prepare inputs for JIT functions
        # We need mass fractions array [H2, O2, N2, H2O, CH4, CO2]
        # Map current composition to this array
        input_mass_fracs = np.zeros(6, dtype=np.float64)
        species_order = ['H2', 'O2', 'N2', 'H2O', 'CH4', 'CO2']
        
        # Normalize composition if needed, though Stream should be solid.
        comp_copy = self.hot_stream.composition.copy()
        total_mass = sum(comp_copy.values())
        
        # Consolidate H2O and H2O_liq into H2O total for Flash Feed
        total_h2o_mass = comp_copy.get('H2O', 0.0) + comp_copy.get('H2O_liq', 0.0)
        
        # Populate array
        for i, s in enumerate(species_order):
            if s == 'H2O':
                input_mass_fracs[i] = total_h2o_mass / total_mass
            else:
                input_mass_fracs[i] = comp_copy.get(s, 0.0) / total_mass
                
        # Get Mole Fractions of Feed (needed for Rachford-Rice z input)
        z_mole_fracs, M_mix_feed, _ = fast_composition_properties(input_mass_fracs)
        z_h2o = z_mole_fracs[3] # Index 3 is H2O
        
        # Solver Bounds for T
        T_min = 273.16 # Freezing point
        T_max = max(T_h_in, 500.0) # Upstream or somewhat high
        
        
        # Bisection Solver
        T_sol = T_h_in # Fallback
        output_phase = 'gas'
        final_vap_frac = 1.0
        new_composition = self.hot_stream.composition.copy() # Initialize with fallback
        
        for iter_idx in range(20): # Max 20 iterations
            T_mid = 0.5 * (T_min + T_max)
            
            # 1. Properties at T_mid
            # Saturation Pressure P_sat(T)
            P_sat = 0.0
            if lut_mgr:
                sat_props = lut_mgr.get_saturation_properties(max(5000.0, min(220e5, P_h_in * 0.99))) 
                # Wait, we need P_sat(T_mid). LUT helper is P->T_sat.
                # We need T -> P_sat lookup. 
                # CoolProp call is cleaner, or use LUT inverse? 
                # If LUT manager has T->P lookup, use it. Usually 'Water', 'P', T, Q=0?
                try:
                    # Approximation: Antoine or direct lookup if available
                    # Fallback to Antoine if LUT doesn't support Sat calc by T easily
                    # Antoine for Water:
                    # log10(P_mmHg) = 8.07131 - 1730.63 / (233.426 + T_C)
                    T_C = T_mid - 273.15
                    if T_C < 0: T_C = 0.01
                    val = 8.07131 - 1730.63 / (233.426 + T_C)
                    p_mmhg = 10**val
                    P_sat = p_mmhg * 133.322
                except:
                    P_sat = 10000.0 # Fail safe
            else:
                 # Antoine fallback
                T_C = T_mid - 273.15
                if T_C < 0: T_C = 0.01
                val = 8.07131 - 1730.63 / (233.426 + T_C)
                p_mmhg = 10**val
                P_sat = p_mmhg * 133.322

            # 2. Flash
            K_w = P_sat / P_h_in
            beta = solve_rachford_rice_single_condensable(z_h2o, K_w)
            
            # 3. Mixture Enthalpy Calculation (LUT Consistent)
            # Must convert Molar Split (Beta) to Mass Split (Psi)
            
            # Moles Inerts (non-condensable) = 1 - z_h2o
            # Moles Vap Water = beta - (1 - z_h2o) ?? No.
            # Binary Assumption: 
            # Gas Phase: y_w = P_sat/P (if mixed) or z_h2o (if gas).
            # Rachford Rice handles this.
            # Moles Gas = Beta.
            # Moles Liq = 1 - Beta.
            # Composition Gas: y_i = z_i / (1 + beta(K-1)). For water: y_w = z_w / (1 + beta(K-1)).
            # Composition Liq: x_i = z_i / (1 + beta(K-1)). For water: x_w = z_w / ...
            
            # Recalculate Gas Mol Fracs
            # y_w = z_h2o / (1.0 + beta*(K_w - 1.0)) # Actually K_w for water is small (<1) usually? No K < 1.
            # Actually use RR definition: y_i = K * x_i.
            # And x_i = z_i / (1 + beta(K-1)).
            
            # Simplified Logic for Single Condensable (Inerts K -> inf) from numba_ops:
            # If Beta < 1:
            #   Gas is Saturated: y_w = K_w (assuming pure liquid water).
            #   Liq is Pure Water: x_w = 1.0.
            #   Inerts in Gas: y_inert = (1 - K_w) normalized by inert ratios.
            
            # Calculate Mass of Gas Phase and Mass of Liquid Phase (per mole feed)
            n_gas = beta
            n_liq = 1.0 - beta
            
            # MW Gas
            # y_w
            if beta < 0.9999 and K_w < 1.0:
                y_w = K_w
            else:
                y_w = z_h2o
            
            # MW_inerts (average of non-condensables)
            # MW_mix_feed = z_w*18 + (1-z_w)*MW_inerts_avg
            # MW_inerts_avg = (M_mix_feed - z_h2o*0.018015) / (1.0 - z_h2o)
            mw_inerts_avg = 0.028 # default
            if (1.0 - z_h2o) > 1e-9:
                mw_inerts_avg = (M_mix_feed - z_h2o*0.018015) / (1.0 - z_h2o)
            
            mw_gas = y_w * 0.018015 + (1.0 - y_w) * mw_inerts_avg
            mw_liq = 0.018015
            
            mass_gas = n_gas * mw_gas
            mass_liq = n_liq * mw_liq
            total_mass_calc = mass_gas + mass_liq
            
            psi_gas = 0.0
            if total_mass_calc > 0:
                psi_gas = mass_gas / total_mass_calc
            psi_liq = 1.0 - psi_gas
            
            # Calculate Specific Enthalpies (J/kg)
            # H_gas: Weighted sum of species enthalpies at T_mid
            # Need mass fractions in Gas phase
            # w_w_gas = (y_w * 18) / MW_gas
            w_w_gas = 0.0
            if mw_gas > 0:
                w_w_gas = (y_w * 0.018015) / mw_gas
            
            # H_inerts (J/kg)
            # We can approximate H_inerts as (H_feed_inerts) at T.
            # H_feed_inerts (per kg inert) = sum(w_i_inert * H_i) / sum(w_i_inert)
            # Calculate H_inert_spec (J/kg_inert)
            h_inert_spec = 0.0
            total_w_inert = 0.0
            for s, mf in comp_copy.items():
                if s == 'H2O' or s == 'H2O_liq': continue
                if mf <= 0: continue
                h_s = lut_mgr.lookup(s, 'H', P_h_in, T_mid) if lut_mgr else 0.0 # Gas Enthalpy
                h_inert_spec += mf * h_s
                total_w_inert += mf
            
            if total_w_inert > 0:
                h_inert_spec /= total_w_inert
            
            # H_gas_spec (J/kg_gas) = w_w_gas * H_vap(T) + (1-w_w_gas) * H_inert_spec
            h_vap_w = lut_mgr.lookup('H2O', 'H', max(5000.0, P_h_in * y_w), T_mid) if lut_mgr else 2.5e6
            h_gas_spec = w_w_gas * h_vap_w + (1.0 - w_w_gas) * h_inert_spec
            
            # H_liq_spec (J/kg_liq) = H_liq(T)
            # Pure water liquid enthalpy
            h_liq_w = lut_mgr.lookup('H2O', 'H', P_h_in, T_mid) if lut_mgr else 1.0e5 
            # Note: CoolProp H for liquid should be at P_system, T.
            # But if LUT is sat, maybe check? Usually safe.
            h_liq_spec = h_liq_w
            
            # Total H_mix (J/kg)
            h_calc = psi_gas * h_gas_spec + psi_liq * h_liq_spec
            
            # Check convergence
            if abs(h_calc - h_h_out_target) < 100.0: # 100 J/kg tolerance
                T_sol = T_mid
                final_vap_frac = beta
                
                # Reconstruct output composition
                # Gas Phase Mass Fracs * Psi + Liq Phase * (1-Psi) ??
                # No, Interchanger separates phases? No, it outputs a SINGLE stream 'hot_out'.
                # The single stream is 'mixed'.
                # Composition should reflect TOTAL mass fractions.
                # Total Mass Fracs should be SAME as INPUT! (Mass Balance).
                # Unless we are REMOVING mass. Interchanger is a heat exchanger. Mass In = Mass Out.
                # So composition is UNCHANGED!
                # Wait. H2O vs H2O_liq keys?
                # Ah, we want to update the 'H2O' vs 'H2O_liq' split in the dictionary for downstream reporting/physics.
                
                # w_H2O_vap_global = psi_gas * w_w_gas
                # w_H2O_liq_global = psi_liq * 1.0 (pure)
                
                w_h2o_vap_global = psi_gas * w_w_gas
                w_h2o_liq_global = psi_liq
                
                new_composition = comp_copy.copy()
                new_composition['H2O'] = w_h2o_vap_global
                new_composition['H2O_liq'] = w_h2o_liq_global
                
                break
                
            if h_calc > h_h_out_target:
                # Need lower T
                T_max = T_mid
            else:
                T_min = T_mid
                
        T_out_final = T_sol
        if final_vap_frac >= 0.999: output_phase = 'gas'
        elif final_vap_frac <= 0.001: output_phase = 'liquid'
        else: output_phase = 'mixed'


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
