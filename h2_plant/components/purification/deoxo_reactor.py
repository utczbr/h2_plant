from typing import Dict, Any, Optional
import numpy as np
from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.stream import Stream
from h2_plant.core.constants import DeoxoConstants, GasConstants
from h2_plant.optimization import numba_ops

class DeoxoReactor(Component):
    """
    Catalytic Deoxidizer (Deoxo) for O2 removal from H2 stream.
    Uses Pd/Al2O3 catalyst in a uniform fixed bed reactor.
    
    Physics:
    - PFR model with coupled Mass/Energy balances
    - Solved via JIT-compiled RK4 (see numba_ops.solve_deoxo_pfr_step)
    - Reaction: 2H2 + O2 -> 2H2O (Exothermic, -242 kJ/mol O2)
    """
    
    def __init__(self, component_id: str):
        super().__init__()
        self.component_id = component_id
        self.input_stream: Optional[Stream] = None
        self.output_stream: Optional[Stream] = None
        
        # State tracking
        self.last_conversion_o2 = 0.0
        self.last_peak_temp_k = 0.0
        self.last_pressure_drop_bar = 0.0
        
    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        super().initialize(dt, registry)

    def step(self, t: float) -> None:
        super().step(t)
        
        if self.input_stream is None or self.input_stream.mass_flow_kg_h <= 0:
            # No flow / shutdown
            self.output_stream = Stream(0.0)
            return

        stream = self.input_stream
        
        # 1. Prepare Physics Inputs
        # Need molar flow of mixture and O2 mole fraction
        # Usually stream.composition is in mole fractions.
        
        # Calculate Mixture Molecular Weight
        # MW_mix = sum(yi * MWi)
        
        comp = stream.composition
        y_o2_in = comp.get('O2', 0.0)
        y_h2_in = comp.get('H2', 0.0)
        y_h2o_in = comp.get('H2O', 0.0)
        # Assume balance is impurities if any, but simplistic MW calc:
        
        # Safe default MWs from constants or hardcoded if simple
        MW_H2 = 2.016e-3
        MW_O2 = 32.00e-3
        MW_H2O = 18.015e-3
        
        # Normalize just in case
        total_y = sum(comp.values())
        if total_y <= 0: total_y = 1.0
        
        mw_mix = 0.0
        for s, y in comp.items():
            if s == 'H2': mw_i = MW_H2
            elif s == 'O2': mw_i = MW_O2
            elif s == 'H2O': mw_i = MW_H2O
            else: mw_i = 28e-3 # Generic constant for N2 etc
            mw_mix += (y/total_y) * mw_i
            
        molar_flow_total = (stream.mass_flow_kg_h / 3600.0) / mw_mix
        
        # 2. Solve PFR Physics
        conversion, t_out, t_peak = numba_ops.solve_deoxo_pfr_step(
            L_total=DeoxoConstants.L_REACTOR_M,
            steps=50,
            T_in=stream.temperature_k,
            P_in_pa=stream.pressure_pa,
            molar_flow_total=molar_flow_total,
            y_o2_in=y_o2_in,
            k0=DeoxoConstants.K0_VOL_S1,
            Ea=DeoxoConstants.EA_J_MOL,
            R=GasConstants.R_UNIVERSAL_J_PER_MOL_K,
            delta_H=DeoxoConstants.DELTA_H_RXN_J_MOL_O2,
            U_a=DeoxoConstants.U_A_W_M3_K,
            T_jacket=DeoxoConstants.T_JACKET_K,
            Area=DeoxoConstants.AREA_REACTOR_M2,
            Cp_mix=DeoxoConstants.CP_MIX_AVG_J_MOL_K
        )
        
        self.last_conversion_o2 = conversion
        self.last_peak_temp_k = t_peak
        
        # 3. Mass Balance Update
        # Reaction: 2 H2 + 1 O2 -> 2 H2O
        # Moles O2 consumed:
        n_o2_in = molar_flow_total * y_o2_in
        n_o2_consumed = n_o2_in * conversion
        n_h2_consumed = 2 * n_o2_consumed
        n_h2o_gen = 2 * n_o2_consumed
        
        # New Molar Flows
        # Need to handle dictionary carefully
        new_moles = {}
        for s, y in comp.items():
            new_moles[s] = molar_flow_total * y
            
        new_moles['O2'] = new_moles.get('O2', 0.0) - n_o2_consumed
        new_moles['H2'] = new_moles.get('H2', 0.0) - n_h2_consumed
        new_moles['H2O'] = new_moles.get('H2O', 0.0) + n_h2o_gen
        
        # Re-normalize and calculate total mass
        total_moles_out = sum(new_moles.values())
        new_comp = {s: n/total_moles_out for s, n in new_moles.items()}
        
        # New Mass Flow
        # Mass in = Mass out (Reaction conserves mass)
        # So typically mass_flow_out = mass_flow_in
        # (2*2 + 32 = 36. 2*18 = 36). Conserved.
        mass_flow_out = stream.mass_flow_kg_h
        
        # 4. Pressure Drop (Ergun)
        # Simplified: Use design DP or basic calculation?
        # User prompt mentioned: "Ergun equation for DP ... pressure drop minimized to 0.0019 bar"
        # Since minimizing is the goal and DP is small, accurate calc might be overkill properly but let's be rigorous if easy.
        # But we don't have explicit Ergun solver in numba_ops.
        # Let's use simplified scaling from design point.
        # DP ~ u^1 to u^2.
        # Design u = 0.06 m/s, DP = 0.0019 bar.
        # Current u = Q_vol / Area.
        # Q_vol approx proportional to MolFlow * T / P.
        
        u_design = 0.06
        dp_design = 0.0019 # bar
        
        rho_gas = (stream.pressure_pa * mw_mix) / (GasConstants.R_UNIVERSAL_J_PER_MOL_K * stream.temperature_k)
        vol_flow = (stream.mass_flow_kg_h / 3600.0) / rho_gas
        u_curr = vol_flow / DeoxoConstants.AREA_REACTOR_M2
        
        # Scaling: DP_new = DP_design * (u_new / u_design)^1.75 (approx Ergun mix)
        # Or Just linear/quadratic.
        if u_design > 0:
            ratio = u_curr / u_design
            dp_curr = dp_design * (ratio ** 1.5) # approximate
        else:
            dp_curr = 0.0
            
        self.last_pressure_drop_bar = dp_curr
        p_out = stream.pressure_pa - (dp_curr * 1e5)
        
        # 5. Output Stream
        self.output_stream = Stream(
            mass_flow_kg_h=mass_flow_out,
            temperature_k=t_out,
            pressure_pa=p_out,
            composition=new_comp,
            phase='gas'
        )
        
    def receive_input(self, port_name: str, value: Any, resource_type: str = 'stream') -> None:
        if port_name == 'inlet' and isinstance(value, Stream):
            self.input_stream = value
            
    def get_output(self, port_name: str) -> Any:
        if port_name == 'outlet':
            return self.output_stream
        return None

    def get_state(self) -> Dict[str, Any]:
        return {
            **super().get_state(),
            'conversion_o2_percent': self.last_conversion_o2 * 100.0,
            'peak_temperature_c': self.last_peak_temp_k - 273.15,
            'pressure_drop_mbar': self.last_pressure_drop_bar * 1000.0
        }
