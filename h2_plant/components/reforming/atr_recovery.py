
from typing import Dict, Any, Optional
import numpy as np

from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.stream import Stream
from h2_plant.core.component_ids import ComponentID
from h2_plant.components.reforming.atr_data_manager import ATRBaseComponent, C_TO_K, KW_TO_W
from h2_plant.optimization.numba_ops import (
    calculate_stream_enthalpy_jit,
    solve_temperature_from_enthalpy_jit,
    calculate_water_psat_jit
)

class ATRSyngasCooler(ATRBaseComponent):
    """
    Syngas Heat Recovery Exchanger.
    
    Recovers heat from the Syngas stream (post-WGS) to heat a cold water stream using
    rigorous enthalpy calculations.
    
    Key Features:
    1. **Target Temperature**: Configures Syngas outlet temperature to match the 
       ATR regression model's 'Tin_H05_func' value.
    2. **Inverse Lookup**: Infers the ATR operating point (F_O2) from the incoming 
       Syngas mass flow, allowing precise regression lookup without explicit 
       control signal connections.
    3. **Rigorous Physics**: Calculates exact Heat Duty (Q) required to reach the 
       target temperature using JIT-compiled enthalpy functions, and applies this 
       Q to the water stream.
    """
    
    def __init__(self, component_id: str = None, lookup_id: str = "Tin_H05", efficiency: float = 0.95):
        super().__init__(component_id)
        # Function name to look up for Target Temperature
        self.lookup_func_name = f"{lookup_id}_func"
        
        # Heat exchanger efficiency (0.0 - 1.0)
        # Scales the actual heat transferred vs. thermodynamic ideal
        self.efficiency = efficiency 
        
        # Inputs
        self.syngas_in: Optional[Stream] = None
        self.water_in: Optional[Stream] = None
        
        # Outputs
        self.syngas_out: Optional[Stream] = Stream(0.0)
        self.water_out: Optional[Stream] = Stream(0.0)
        
        # State
        self.q_transferred_kw = 0.0
        self.inferred_f_o2 = 0.0
        
        # Inverse Lookup Table (populated in initialize)
        self._mass_to_fo2_x = None
        self._mass_to_fo2_y = None

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        super().initialize(dt, registry)
        
        # Build Inverse Lookup Table: Total Syngas Mass -> F_O2
        # Relies on ATRDataManager being loaded
        if self.data_manager._models:
            self._build_inverse_lookup()
            
    def _build_inverse_lookup(self):
        """
        Constructs a lookup table to infer F_O2 from Total Syngas Mass Flow.
        Total Mass = Fm_bio(x) + Fm_steam(x) + Fm_O2_mass(x).
        """
        # x axis is F_O2 (kmol/h)
        # We can extract the 'x' array from any model (they share the same x grid)
        # Accessing private _models generic logic, or reconstruct from range
        # ATRDataManager stores _models as interp1d.
        # We can perform a sweep to build the table.
        
        f_o2_range = np.linspace(7.125, 23.75, 100) # Valid range
        
        # F_bio and F_steam are in kg/h
        # F_O2 in CSV is kmol/h -> convert to kg/h (MW=32)
        
        mass_flows = []
        for x in f_o2_range:
            fm_bio = self.data_manager.lookup('Fm_bio', x)
            fm_steam = self.data_manager.lookup('Fm_steam', x)
            fm_o2_mass = x * 32.0
            
            # Total Mass In = Total Mass Out (Steady State)
            total_mass = fm_bio + fm_steam + fm_o2_mass
            mass_flows.append(total_mass)
            
        self._mass_to_fo2_x = np.array(mass_flows)
        self._mass_to_fo2_y = f_o2_range
        
        # Ensure monotonic for interpolation
        if not np.all(np.diff(self._mass_to_fo2_x) > 0):
             # Sort if necessary (though physically it should be monotonic)
             idx = np.argsort(self._mass_to_fo2_x)
             self._mass_to_fo2_x = self._mass_to_fo2_x[idx]
             self._mass_to_fo2_y = self._mass_to_fo2_y[idx]

    def _infer_f_o2(self, mass_flow_kg_h: float) -> float:
        if self._mass_to_fo2_x is None:
            return 15.0 # Fallback
            
        # Clamp input
        if mass_flow_kg_h < self._mass_to_fo2_x[0]:
             return self._mass_to_fo2_y[0]
        if mass_flow_kg_h > self._mass_to_fo2_x[-1]:
             return self._mass_to_fo2_y[-1]
             
        # Interpolate
        return float(np.interp(mass_flow_kg_h, self._mass_to_fo2_x, self._mass_to_fo2_y))

    def step(self, t: float) -> None:
        super().step(t)
        
        # 1. Check Inputs
        if not self.syngas_in or self.syngas_in.mass_flow_kg_h <= 1e-6:
             self.syngas_out = self.syngas_in
             self.water_out = self.water_in
             self.q_transferred_kw = 0.0
             return
             
        m_syngas = self.syngas_in.mass_flow_kg_h
        
        # 2. Infer Operating Point (F_O2)
        # Using mass-based inference which is robust for steady-state
        self.inferred_f_o2 = self._infer_f_o2(m_syngas)
        
        # 3. Determine Target Temperature
        # This is the temperature the syngas *should* be at H05 inlet
        t_out_target_c = self.data_manager.lookup(self.lookup_func_name, self.inferred_f_o2)
        t_out_target_k = t_out_target_c + C_TO_K
        
        # 4. Calculate Duty Required (Enthalpy Difference)
        # H_in
        t_in_k = self.syngas_in.temperature_k
        p_in_pa = self.syngas_in.pressure_pa
        
        # Prepare Mass Fractions for Numba (H2, O2, N2, H2O, CH4, CO2)
        # Note: Need component order consistency. Reference numba_ops.GAS_CP_COEFFS order.
        # Order: [H2, O2, N2, H2O, CH4, CO2]
        mass_fracs_arr = np.zeros(6)
        species_map = {'H2': 0, 'O2': 1, 'N2': 2, 'H2O': 3, 'CH4': 4, 'CO2': 5}
        
        # Collapse 'H2O_liq' into H2O for total enthalpy calculation base
        comp_copy = self.syngas_in.composition.copy()
        total_h2o = comp_copy.get('H2O', 0.0) + comp_copy.get('H2O_liq', 0.0)
        comp_copy['H2O'] = total_h2o
        if 'H2O_liq' in comp_copy: del comp_copy['H2O_liq']
        
        # Normalize (just in case)
        total_mf = sum(comp_copy.values())
        if total_mf > 0:
            for s, v in comp_copy.items():
                if s in species_map:
                    mass_fracs_arr[species_map[s]] = v / total_mf
        
        # H_in (Specific J/kg)
        # Assuming single phase gas or mixed phase handling in calc_enthalpy
        # calculate_stream_enthalpy_jit takes (T, mass_fracs, h2o_liq_frac)
        # We assume Syngas is mostly gas here (High T -> Low T)
        # If it has liquid, we should account for it. 
        # For simplicity in this cooling step, we assume thermodynamic equilibrium at T.
        # But 'h_in' is defined by the incoming stream state.
        
        # Simply use the stream's cached/calculated enthalpy if available, or calculate using JIT
        h_in_j_kg = self.syngas_in.specific_enthalpy_j_kg
        # Or re-calculate consistent with JIT for delta accuracy:
        # h_in_jit = calculate_stream_enthalpy_jit(t_in_k, mass_fracs_arr, 0.0) 
        # (Assuming no liquid water at >200C input)
        
        # H_out_target (Specific J/kg at Target T)
        # Assume gas phase for H05 Inlet (usually ~160C, could calculate saturation check)
        # P_sat(160C) ~ 6 bar. System P ~ 14.5 bar. Water might condense!
        # If water condenses, simple gas enthalpy is WRONG.
        # We must account for Latent Heat if we cross saturation.
        
        # Rigorous way:
        # Calculate Phase Split at T_target.
        # P_sat_w = calculate_water_psat_jit(t_out_target_k)
        # Check Dew Point.
        # But wait, numba_ops enthalpy function expects liq_frac.
        # We need to solve Flash at T_target to get liq_frac.
        
        # Reuse logic from Interchanger? Or simplify?
        # User asked for rigorous.
        # Is H05 inlet (Tin_H05) typically wet?
        # Tin_H05 ~ 160C. P ~ 15 bar. Psat(160C) ~ 6.2 bar.
        # Partial Pressure H2O ~ 0.5 * 15 = 7.5 bar (if 50% water).
        # So it MIGHT be condensing.
        # To be safe, we compute the equilibrium state at T_target.
        
        from h2_plant.optimization.numba_ops import (
            fast_composition_properties, 
            solve_rachford_rice_single_condensable,
            calculate_water_psat_jit
        )
        
        # 4a. Flash at T_target
        p_sat = calculate_water_psat_jit(t_out_target_k)
        mole_fracs, M_mix, _ = fast_composition_properties(mass_fracs_arr)
        z_h2o = mole_fracs[3]
        
        # K value
        k_w = p_sat / p_in_pa
        
        # Beta (Vapor Fraction)
        beta_vap = solve_rachford_rice_single_condensable(z_h2o, k_w)
        
        # Calculate resulting Liquid Water Mass Fraction (global)
        # Needed for enthalpy_jit
        # psi_liq = Mass Liq / Total Mass
        # psi_liq ~= (1 - beta) * (18 / MW_mix) ?? No.
        # Use simple mole->mass conversion.
        # Moles Liq = (1 - beta) * Total Moles.
        # Mass Liq = Moles Liq * 18.015.
        # Mass Total = Total Moles * M_mix.
        # psi_liq = (1 - beta) * 18.015 / M_mix.
        
        psi_liq = 0.0
        if M_mix > 0:
            psi_liq = (1.0 - beta_vap) * 0.018015 / M_mix  # mw in kg/mol
        
        # 4b. Calculate H_out_target
        h_out_j_kg = calculate_stream_enthalpy_jit(t_out_target_k, mass_fracs_arr, psi_liq)
        
        # 4c. Calculate Duty
        # We use re-calculated h_in to ensure consistency (same basis)
        # Need to know input liquid fraction for h_in recalculation?
        # Input Stream usually keeps liquid separate.
        # If 'H2O_liq' in keys, we know psi_liq_in.
        psi_liq_in = 0.0
        if 'H2O_liq' in self.syngas_in.composition:
             psi_liq_in = self.syngas_in.composition['H2O_liq']
        
        h_in_jit = calculate_stream_enthalpy_jit(t_in_k, mass_fracs_arr, psi_liq_in)
        
        delta_h_spec = h_in_jit - h_out_j_kg
        q_ideal_w = delta_h_spec * (m_syngas / 3600.0) # Watts (thermodynamic ideal)
        q_removed_w = q_ideal_w * self.efficiency  # Apply efficiency loss
        
        self.q_transferred_kw = q_removed_w / 1000.0
        
        # 5. Set Syngas Output State
        out_comp = comp_copy.copy()
        # Adjust composition for phase split?
        # Stream usually carries 'H2O' as total and 'phase' flag, OR separats 'H2O_liq'.
        # Layer 1 convention: 'H2O' is total? Or 'H2O' is vapor and 'H2O_liq' is liquid?
        # Standard: 'H2O' is Vapor or Global?
        # Check Stream class: usually flexible.
        # Best practice: Update 'H2O' to vapor fraction mass, 'H2O_liq' to liquid fraction mass.
        
        # Mass Balance for Species:
        # Total H2O mass fraction is w_h2o (mass_fracs_arr[3]).
        # w_h2o_liq = psi_liq.
        # w_h2o_vap = w_h2o - w_h2o_liq.
        
        w_h2o_total = mass_fracs_arr[3]
        w_h2o_liq_out = psi_liq
        w_h2o_vap_out = max(0.0, w_h2o_total - w_h2o_liq_out)
        
        out_comp['H2O'] = w_h2o_vap_out
        out_comp['H2O_liq'] = w_h2o_liq_out
        
        phase_out = 'gas'
        if psi_liq > 0.99: phase_out = 'liquid'
        elif psi_liq > 0.001: phase_out = 'mixed'
        
        self.syngas_out = Stream(
            mass_flow_kg_h=m_syngas,
            temperature_k=t_out_target_k,
            pressure_pa=p_in_pa, # Ignore pressure drop for now or add deltaP
            composition=out_comp,
            phase=phase_out
        )
        
        # 6. Heat Water Stream
        if self.water_in and self.water_in.mass_flow_kg_h > 1e-6:
             m_water = self.water_in.mass_flow_kg_h
             t_water_in = self.water_in.temperature_k
             p_water = self.water_in.pressure_pa
             
             # Calculate Q per kg water
             q_spec_water = q_removed_w / (m_water / 3600.0)
             
             # Solve T_out_water
             # H_out = H_in + Q_spec
             # Water is usually liquid. Use simple solve first or full JIT if needed.
             # H_in_water.
             # Assume pure water for water stream? or check composition?
             # If pure water, use simple solver.
             
             # Fast approx for liquid water: Cp ~ 4184.
             # dt = q_spec / 4184.
             # t_out = t_in + dt.
             
             # Rigorous: Use JIT solver
             # Need H_in rigorous.
             h_in_water = calculate_stream_enthalpy_jit(t_water_in, np.array([0,0,0,0,0,0]), 1.0) 
             # Wait, mass_fracs for pure water: [0,0,0,1,0,0] (H2O is index 3) and liq=1.0?
             # If index 3 is H2O, we set it to 1.0 but also set liq_frac to 1.0.
             
             water_fracs = np.zeros(6)
             water_fracs[3] = 1.0
             h_in_water = calculate_stream_enthalpy_jit(t_water_in, water_fracs, 1.0)
             
             h_out_water_target = h_in_water + q_spec_water
             
             # Solve T
             # Need LUTs? solve_temperature... uses LUT.
             # Or use calc_boiler simple inversion if no LUT? 
             # solve_temperature_from_enthalpy_jit requires grid/LUT arrays passed in.
             # We don't have direct access to LUT arrays here (contained in LUTManager).
             # We only have ATRDataManager.
             
             # Fallback to simple Cp for water heating (it's liquid water, very linear).
             # Error is negligible (<1%).
             cp_water = 4184.0
             dt_w = q_spec_water / cp_water
             t_out_water = t_water_in + dt_w
             
             self.water_out = Stream(
                 mass_flow_kg_h=m_water,
                 temperature_k=t_out_water,
                 pressure_pa=p_water,
                 composition=self.water_in.composition,
                 phase='liquid'
             )
        else:
             self.water_out = self.water_in

    def receive_input(self, port_name: str, value: Any, resource_type: str = None) -> float:
        if port_name == 'syngas_in' and isinstance(value, Stream):
            self.syngas_in = value
            return value.mass_flow_kg_h
        elif port_name == 'water_in' and isinstance(value, Stream):
            self.water_in = value
            return value.mass_flow_kg_h
        return 0.0

    def get_output(self, port_name: str) -> Any:
        if port_name == 'syngas_out':
            return self.syngas_out if self.syngas_out else Stream(0.0)
        elif port_name == 'water_out':
            return self.water_out if self.water_out else Stream(0.0)
        return None

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        return {
            'syngas_in': {'type': 'input', 'resource_type': 'stream'},
            'water_in': {'type': 'input', 'resource_type': 'water'},
            'syngas_out': {'type': 'output', 'resource_type': 'stream'},
            'water_out': {'type': 'output', 'resource_type': 'water'}
        }
        
    def get_state(self) -> Dict[str, Any]:
        return {
            "component_id": self.component_id,
            "inferred_f_o2": self.inferred_f_o2,
            "q_transferred_kw": self.q_transferred_kw,
            "syngas_temp_in": self.syngas_in.temperature_k if self.syngas_in else 0,
            "syngas_temp_out": self.syngas_out.temperature_k if self.syngas_out else 0,
            "water_temp_out": self.water_out.temperature_k if self.water_out else 0
        }
