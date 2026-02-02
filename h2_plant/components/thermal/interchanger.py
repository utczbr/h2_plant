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
import logging
import numpy as np

from h2_plant.core.component import Component
from h2_plant.core.stream import Stream
from h2_plant.core.constants import ConversionFactors, GasConstants, StandardConditions
from h2_plant.core.component_ids import ComponentID
from h2_plant.optimization.numba_ops import (
    solve_interchanger_flash_jit,
    remap_canonical_to_lut,
    fast_composition_properties,
    solve_rachford_rice_single_condensable,
)

logger = logging.getLogger(__name__)

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
        efficiency: float = 0.95,
        area_m2: float = 100.0
    ):
        super().__init__()
        self.component_id = component_id
        self.min_approach_temp_k = min_approach_temp_k
        self.target_cold_temp_k = target_cold_out_temp_c + 273.15
        self.efficiency = efficiency
        self._area_m2 = area_m2

        # Inputs
        self.hot_stream: Optional[Stream] = None
        self.cold_stream: Optional[Stream] = None

        # Outputs
        self.hot_out: Optional[Stream] = Stream(0.0, temperature_k=298.15, pressure_pa=101325.0, phase='gas')
        self.cold_out: Optional[Stream] = Stream(0.0, temperature_k=target_cold_out_temp_c+273.15, pressure_pa=101325.0, phase='liquid')
        
        self.q_transferred_kw = 0.0

    @property
    def area_m2(self) -> float:
        """Heat transfer area for CAPEX sizing (m²)."""
        return self._area_m2

    def initialize(self, dt: float, registry: 'ComponentRegistry') -> None:
        """
        Executes initialization phase of Component Lifecycle.

        Args:
            dt (float): Simulation timestep (hours).
            registry (ComponentRegistry): Central service registry.
        """
        super().initialize(dt, registry)

        # P1: Pre-resolve LUT manager and species mapping for JIT flash
        self._lut_mgr = None
        self._species_map = None
        self._lut_mass_fracs = np.zeros(7, dtype=np.float64)
        self._jit_flash_available = False

        if registry:
            lut_mgr = registry.get(ComponentID.LUT_MANAGER)
            if lut_mgr and lut_mgr.stacked_H is not None:
                self._lut_mgr = lut_mgr
                self._species_map = lut_mgr.get_species_map()
                self._jit_flash_available = True

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

        # --- 0.5 Merge Entrained Liquid (Standardize Total Mass) ---
        # Checks for entrained liquid in 'extra' and merges it into the bulk stream
        # so it participates in physics and mass balance is clear.
        if self.hot_stream.extra and 'm_dot_H2O_liq_accomp_kg_s' in self.hot_stream.extra:
            entrained_kg_s = self.hot_stream.extra.get('m_dot_H2O_liq_accomp_kg_s', 0.0)
            entrained_kg_h = entrained_kg_s * 3600.0
            
            if entrained_kg_h > 0:
                # Calculate new total mass
                m_h_total_old = m_h_kg_h
                m_h_kg_h = m_h_total_old + entrained_kg_h
                
                # Update composition
                comp = self.hot_stream.composition.copy()
                
                # Existing masses
                m_h2o_liq_old = comp.get('H2O_liq', 0.0) * m_h_total_old
                
                # New total liquid mass
                m_h2o_liq_new = m_h2o_liq_old + entrained_kg_h
                
                # Recalculate fractions
                comp['H2O_liq'] = m_h2o_liq_new / m_h_kg_h
                for s in comp:
                    if s != 'H2O_liq':
                        comp[s] = (comp[s] * m_h_total_old) / m_h_kg_h
                
                # Create a temporary merged stream for property calculations
                # (We don't modify self.hot_stream in place to avoid side effects on input)
                # But for this step scope, we pretend self.hot_stream IS the merged one?
                # Actually, better to just update the local variables used for physics.
                
                # Update local composition map for later use
                self.hot_stream_comp = comp
            else:
                 self.hot_stream_comp = self.hot_stream.composition
        else:
             self.hot_stream_comp = self.hot_stream.composition

        # Use pre-resolved LUT manager from initialize()
        lut_mgr = self._lut_mgr if self._jit_flash_available else None
        if lut_mgr is None and hasattr(self, '_registry') and self._registry:
            lut_mgr = self._registry.get(ComponentID.LUT_MANAGER)

        # Properties
        T_h_in = self.hot_stream.temperature_k
        P_h_in = self.hot_stream.pressure_pa
        
        T_c_in = self.cold_stream.temperature_k
        
        # --- 1. Calculate Inlet Specific Enthalpy (Hot) ---
        # PERFORMANCE: Use vectorized JIT lookup (single C-space call for all species)
        
        # Build mass fractions array matching LUT fluid order (H2, O2, N2, CO2, CH4, H2O)
        # Note: StandardConditions.CANONICAL_FLUID_ORDER = ('H2', 'O2', 'N2', 'H2O', 'CH4', 'CO2')
        # but LUT config uses ('H2', 'O2', 'N2', 'CO2', 'CH4', 'H2O') - check lut_manager.LUTConfig
        
        lut_fluid_order = lut_mgr.config.fluids if lut_mgr else StandardConditions.CANONICAL_FLUID_ORDER
        hot_mass_fracs = np.zeros(len(lut_fluid_order), dtype=np.float64)
        
        for idx, fluid in enumerate(lut_fluid_order):
            if fluid in self.hot_stream_comp:
                hot_mass_fracs[idx] = self.hot_stream_comp[fluid]
            elif fluid == 'H2O' and 'H2O_liq' in self.hot_stream_comp:
                # Combine vapor and liquid water for total
                hot_mass_fracs[idx] = self.hot_stream_comp.get('H2O', 0.0) + self.hot_stream_comp.get('H2O_liq', 0.0)
        
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
        # Solving H(T, P) = h_h_out_target using JIT-compiled solver (P1)

        comp_copy = self.hot_stream_comp.copy()
        total_mass = sum(comp_copy.values())
        total_h2o_mass = comp_copy.get('H2O', 0.0) + comp_copy.get('H2O_liq', 0.0)

        T_sol = T_h_in
        output_phase = 'gas'
        final_vap_frac = 1.0
        new_composition = self.hot_stream_comp.copy()

        if self._jit_flash_available:
            lut_mgr = self._lut_mgr

            # Build canonical 6-element mass fracs from stream
            canonical_fracs, _, _, _ = self.hot_stream.get_composition_arrays()

            # H2O_liq fraction for folding
            h2o_liq_frac = self.hot_stream_comp.get('H2O_liq', 0.0)

            # Remap to 7-element LUT order
            lut_fracs = remap_canonical_to_lut(canonical_fracs, h2o_liq_frac, self._species_map)

            # Normalize
            lut_total = lut_fracs.sum()
            if lut_total > 1e-12:
                lut_fracs /= lut_total

            # Get mole fractions using LUT-ordered array and fast_composition_properties
            z_mole_fracs, M_mix_feed, _ = fast_composition_properties(lut_fracs)
            z_h2o = z_mole_fracs[5]  # H2O is at LUT index 5

            T_sol, final_vap_frac, was_clamped = solve_interchanger_flash_jit(
                z_h2o, M_mix_feed, P_h_in, h_h_out_target, T_h_in,
                lut_fracs,
                lut_mgr.stacked_H, lut_mgr._pressure_grid, lut_mgr._temperature_grid,
                40, 100.0
            )

            if was_clamped:
                logger.warning(f"Interchanger {self.component_id}: flash clamped to LUT bounds")

            # Reconstruct H2O/H2O_liq split from flash result
            from h2_plant.optimization.numba_ops import _antoine_psat_water
            P_sat_sol = _antoine_psat_water(T_sol)
            K_w_sol = P_sat_sol / P_h_in
            beta = final_vap_frac

            MW_H2O = 0.018015
            mw_inerts_avg = 0.028
            if (1.0 - z_h2o) > 1e-9:
                mw_inerts_avg = (M_mix_feed - z_h2o * MW_H2O) / (1.0 - z_h2o)

            if beta < 0.9999 and K_w_sol < 1.0:
                y_w = K_w_sol
            else:
                y_w = z_h2o

            mw_gas = y_w * MW_H2O + (1.0 - y_w) * mw_inerts_avg
            mass_gas = beta * mw_gas
            mass_liq = (1.0 - beta) * MW_H2O
            total_mass_calc = mass_gas + mass_liq

            psi_gas = mass_gas / total_mass_calc if total_mass_calc > 0.0 else 1.0
            w_w_gas = (y_w * MW_H2O) / mw_gas if mw_gas > 0.0 else 0.0

            w_h2o_vap_global = psi_gas * w_w_gas
            w_h2o_liq_global = 1.0 - psi_gas

            # Preserve inert mass fractions exactly
            new_composition = {}
            for s, mf in comp_copy.items():
                if s not in ('H2O', 'H2O_liq'):
                    new_composition[s] = mf

            w_h2o_total_input = comp_copy.get('H2O', 0.0) + comp_copy.get('H2O_liq', 0.0)
            calc_total_water_split = w_h2o_vap_global + w_h2o_liq_global

            vap_ratio = 1.0
            if calc_total_water_split > 1e-12:
                vap_ratio = w_h2o_vap_global / calc_total_water_split
            elif final_vap_frac < 0.01:
                vap_ratio = 0.0

            new_composition['H2O'] = w_h2o_total_input * vap_ratio
            new_composition['H2O_liq'] = w_h2o_total_input * (1.0 - vap_ratio)

        else:
            # Fallback: Python bisection (legacy path for when LUT is unavailable)
            lut_mgr = None
            if hasattr(self, '_registry') and self._registry:
                lut_mgr = self._registry.get(ComponentID.LUT_MANAGER)

            comp_copy_norm = {}
            for s, mf in comp_copy.items():
                if s == 'H2O':
                    comp_copy_norm[s] = total_h2o_mass / total_mass
                elif s == 'H2O_liq':
                    continue
                else:
                    comp_copy_norm[s] = mf / total_mass

            # Build 7-element array in LUT order for fast_composition_properties
            lut_fluid_order = lut_mgr.config.fluids if lut_mgr else ('H2', 'O2', 'N2', 'CO2', 'CH4', 'H2O', 'CO')
            input_mass_fracs = np.zeros(len(lut_fluid_order), dtype=np.float64)
            for i, fluid in enumerate(lut_fluid_order):
                input_mass_fracs[i] = comp_copy_norm.get(fluid, 0.0)

            z_mole_fracs, M_mix_feed, _ = fast_composition_properties(input_mass_fracs)
            # H2O in LUT order is at index 5
            z_h2o = z_mole_fracs[5]

            T_lo = 273.16
            T_hi = max(T_h_in, 500.0)

            for iter_idx in range(50):
                T_mid = 0.5 * (T_lo + T_hi)
                T_sol = T_mid

                T_C = T_mid - 273.15
                if T_C < 0.01:
                    T_C = 0.01
                val = 8.07131 - 1730.63 / (233.426 + T_C)
                P_sat = (10 ** val) * 133.322

                K_w = P_sat / P_h_in
                beta = solve_rachford_rice_single_condensable(z_h2o, K_w)

                n_gas = beta
                n_liq = 1.0 - beta

                if beta < 0.9999 and K_w < 1.0:
                    y_w = K_w
                else:
                    y_w = z_h2o

                mw_inerts_avg = 0.028
                if (1.0 - z_h2o) > 1e-9:
                    mw_inerts_avg = (M_mix_feed - z_h2o * 0.018015) / (1.0 - z_h2o)

                mw_gas = y_w * 0.018015 + (1.0 - y_w) * mw_inerts_avg
                mass_gas = n_gas * mw_gas
                mass_liq = n_liq * 0.018015
                total_mass_calc = mass_gas + mass_liq
                psi_gas = mass_gas / total_mass_calc if total_mass_calc > 0 else 0.0
                psi_liq = 1.0 - psi_gas

                w_w_gas = (y_w * 0.018015) / mw_gas if mw_gas > 0 else 0.0

                h_inert_spec = 0.0
                total_w_inert = 0.0
                for s, mf in comp_copy.items():
                    if s in ('H2O', 'H2O_liq') or mf <= 0:
                        continue
                    h_s = lut_mgr.lookup(s, 'H', P_h_in, T_mid) if lut_mgr else 0.0
                    h_inert_spec += mf * h_s
                    total_w_inert += mf
                if total_w_inert > 0:
                    h_inert_spec /= total_w_inert

                h_vap_w = lut_mgr.lookup('H2O', 'H', max(5000.0, P_h_in * y_w), T_mid) if lut_mgr else 2.5e6
                h_gas_spec = w_w_gas * h_vap_w + (1.0 - w_w_gas) * h_inert_spec
                h_liq_w = lut_mgr.lookup('H2O', 'H', P_h_in, T_mid) if lut_mgr else 1.0e5
                h_calc = psi_gas * h_gas_spec + psi_liq * h_liq_w

                if abs(h_calc - h_h_out_target) < 100.0:
                    final_vap_frac = beta
                    w_h2o_vap_global = psi_gas * w_w_gas
                    w_h2o_liq_global = psi_liq

                    new_composition = {}
                    for s, mf in comp_copy.items():
                        if s not in ('H2O', 'H2O_liq'):
                            new_composition[s] = mf
                    w_h2o_total_input = comp_copy.get('H2O', 0.0) + comp_copy.get('H2O_liq', 0.0)
                    calc_total_water_split = w_h2o_vap_global + w_h2o_liq_global
                    vap_ratio = 1.0
                    if calc_total_water_split > 1e-12:
                        vap_ratio = w_h2o_vap_global / calc_total_water_split
                    elif final_vap_frac < 0.01:
                        vap_ratio = 0.0
                    new_composition['H2O'] = w_h2o_total_input * vap_ratio
                    new_composition['H2O_liq'] = w_h2o_total_input * (1.0 - vap_ratio)
                    break

                if h_calc > h_h_out_target:
                    T_hi = T_mid
                else:
                    T_lo = T_mid
                
        T_out_final = T_sol
        if final_vap_frac >= 0.999: output_phase = 'gas'
        elif final_vap_frac <= 0.001: output_phase = 'liquid'
        else: output_phase = 'mixed'



        # --- 7. Final Output Streams ---
        
        # Hot Out (Cooled)
        # Check if we dropped 'extra' (entrained liquid) from the input
        out_extra = {}
        if self.hot_stream and self.hot_stream.extra:
             out_extra = self.hot_stream.extra.copy()
             # Remove merged liquid from extra to avoid double counting
             if 'm_dot_H2O_liq_accomp_kg_s' in out_extra:
                 del out_extra['m_dot_H2O_liq_accomp_kg_s']

        self.hot_out = Stream(
            mass_flow_kg_h=m_h_kg_h,
            temperature_k=T_out_final,
            pressure_pa=P_h_in,
            composition=new_composition,
            phase=output_phase,
            extra=out_extra
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
            'hot_in': {'type': 'input', 'resource_type': 'stream'},
            'cold_in': {'type': 'input', 'resource_type': 'stream'},
            'hot_out': {'type': 'output', 'resource_type': 'stream'},
            'cold_out': {'type': 'output', 'resource_type': 'stream'}
        }

    def get_state(self) -> Dict[str, Any]:
        """
        Returns component operational telemetry.
        
        Returns:
            Dict[str, Any]: Q_transferred (kW) and outlet temperatures.
        """
        # Calculate robust reported mass flow (Bulk + Entrained) for Consistency
        # Since we now merge entrained liquid, hot_out.mass_flow_kg_h IS the total mass.
        hot_mass = 0.0
        if self.hot_out:
            hot_mass = self.hot_out.mass_flow_kg_h

        return {
            **super().get_state(),
            'q_transferred_kw': self.q_transferred_kw,
            'hot_out_temp_k': self.hot_out.temperature_k if self.hot_out else 0,
            'cold_out_temp_k': self.cold_out.temperature_k if self.cold_out else 0,
            'outlet_mass_flow_kg_h': hot_mass, # Bulk is now Total
            'outlet_total_mass_flow_kg_h': hot_mass, # Standardized
            'outlet_gas_mass_flow_kg_h': self.hot_out.mass_flow_kg_h if self.hot_out else 0.0, # Bulk (Process Path)
            'outlet_entrained_mass_kg_h': 0.0, # Merged
            'hot_outlet_mass_flow_kg_h': self.hot_out.mass_flow_kg_h if self.hot_out else 0.0,
            'cold_outlet_mass_flow_kg_h': self.cold_out.mass_flow_kg_h if self.cold_out else 0,
            'outlet_H2O_molf': self.hot_out.get_total_mole_frac('H2O') if self.hot_out else 0.0 # Total H2O
        }
        
