"""
Catalytic Deoxidizer (DeOxo) Reactor Component.

This module implements a plug flow reactor (PFR) for catalytic removal of
residual oxygen from hydrogen streams. The DeOxo reactor is critical for
achieving fuel-cell grade purity (<1 ppm O₂).

Reaction Chemistry:
    **2H₂ + O₂ → 2H₂O**    ΔH = -242 kJ/mol O₂ (highly exothermic)

    The reaction proceeds over Pd/Al₂O₃ catalyst at temperatures above 50°C.
    At typical H₂ stream O₂ concentrations (0.1-0.5%), the reaction generates
    significant heat requiring thermal management.

Physical Model:
    - **Plug Flow Reactor (PFR)**: Coupled mass and energy balances solved
      via 4th-order Runge-Kutta integration along reactor length.
    - **Kinetics**: First-order in O₂ with Arrhenius temperature dependence.
    - **Heat Transfer**: Jacketed cooling maintains temperature below catalyst
      deactivation threshold (~150°C).
    - **Pressure Drop**: Ergun-type correlation scaled from design point.

Architecture:
    Implements the Component Lifecycle Contract (Layer 1):
    - `initialize()`: Standard initialization.
    - `step()`: Solves PFR physics via JIT-compiled integrator.
    - `get_state()`: Returns conversion, peak temperature, and pressure drop.

Performance Optimization:
    The PFR solver is JIT-compiled via Numba (solve_deoxo_pfr_step) for
    real-time simulation performance.
"""

from typing import Dict, Any, Optional
import numpy as np

from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.stream import Stream
from h2_plant.core.constants import DeoxoConstants, GasConstants
from h2_plant.optimization import numba_ops

import logging
logger = logging.getLogger(__name__)


class DeoxoReactor(Component):
    """
    Catalytic deoxidizer for oxygen removal from hydrogen streams.

    Implements a PFR model with coupled mass/energy balances to track
    O₂ conversion, temperature profile, and pressure drop. Uses Pd/Al₂O₃
    catalyst in a uniform fixed bed configuration.

    This component fulfills the Component Lifecycle Contract (Layer 1):
        - `initialize()`: Standard component initialization.
        - `step()`: Solves PFR physics, updates output stream composition.
        - `get_state()`: Returns conversion, peak temperature, and ΔP.

    The reactor model:
    1. Receives H₂ stream with trace O₂ contamination.
    2. Solves RK4-integrated PFR for temperature and conversion profiles.
    3. Updates stream composition based on stoichiometry (2H₂ + O₂ → 2H₂O).
    4. Calculates pressure drop using Ergun-type scaling.

    Attributes:
        input_stream (Stream): Current inlet stream.
        output_stream (Stream): Processed outlet stream.
        last_conversion_o2 (float): O₂ conversion fraction (0-1).
        last_peak_temp_k (float): Maximum temperature along reactor (K).
        last_pressure_drop_bar (float): Total pressure drop (bar).

    Example:
        >>> deoxo = DeoxoReactor(component_id='DEOXO-1')
        >>> deoxo.initialize(dt=1/60, registry=registry)
        >>> deoxo.receive_input('inlet', h2_stream, 'stream')
        >>> deoxo.step(t=0.0)
        >>> outlet = deoxo.get_output('outlet')
    """

    def __init__(self, component_id: str):
        """
        Initialize the DeOxo reactor.

        Args:
            component_id (str): Unique identifier for this component.
        """
        super().__init__()
        self.component_id = component_id
        self.input_stream: Optional[Stream] = None
        self.output_stream: Optional[Stream] = None

        # Performance tracking
        self.last_conversion_o2 = 0.0
        self.last_peak_temp_k = 0.0
        self.last_pressure_drop_bar = 0.0
        
        # Profile data for plotting (L, T, X)
        self.last_profiles: Dict[str, np.ndarray] = {
            'L': np.array([]),
            'T': np.array([]),
            'X': np.array([])
        }

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """
        Prepare the component for simulation execution.

        Fulfills the Component Lifecycle Contract initialization phase.

        Args:
            dt (float): Simulation timestep in hours.
            registry (ComponentRegistry): Central registry for component access.
        """
        super().initialize(dt, registry)

    def step(self, t: float) -> None:
        """
        Execute one simulation timestep.

        Performs complete DeOxo reactor calculation sequence:
        1. Calculate mixture molecular weight and molar flow from mass flow.
        2. Invoke JIT-compiled PFR solver for conversion and temperature.
        3. Apply reaction stoichiometry to update stream composition.
        4. Calculate pressure drop using Ergun-type velocity scaling.

        Args:
            t (float): Current simulation time in hours.
        """
        super().step(t)

        if self.input_stream is None or self.input_stream.mass_flow_kg_h <= 0:
            self.output_stream = Stream(0.0)
            return

        stream = self.input_stream

        # Extract composition
        comp = stream.composition
        y_o2_in = comp.get('O2', 0.0)
        y_h2_in = comp.get('H2', 0.0)
        y_h2o_in = comp.get('H2O', 0.0)

        # Molecular weights
        MW = {
            'H2': 2.016e-3,
            'O2': 32.00e-3,
            'H2O': 18.015e-3,
            'other': 28.0e-3
        }

        # 1. Convert Mass Fractions to Mole Fractions
        # n_i_rel = x_i / M_i
        n_rel = {}
        total_n_rel = 0.0
        
        for s, x in comp.items():
            mw_i = MW.get(s, MW['other'])
            ni = x / mw_i
            n_rel[s] = ni
            total_n_rel += ni
            
        if total_n_rel <= 0:
            total_n_rel = 1.0
            
        y_fracs = {s: n/total_n_rel for s, n in n_rel.items()}
        y_o2_in = y_fracs.get('O2', 0.0)
        
        # DEBUG: Log inlet O2 concentration
        #logger.warning(f"[DEOXO DEBUG] {self.component_id}: Inlet O2 = {y_o2_in*1e6:.1f} ppm (y_O2 = {y_o2_in:.6f})")
        
        # 2. Calculate Mix MW
        mw_mix = sum(y * MW.get(s, MW['other']) for s, y in y_fracs.items())
        
        # Convert mass flow to molar flow
        molar_flow_total = (stream.mass_flow_kg_h / 3600.0) / mw_mix
        
        # DEBUG: Log PFR inputs
        #logger.warning(f"[DEOXO] In: T={stream.temperature_k:.1f}K, molar_flow={molar_flow_total:.4f} mol/s, y_O2={y_o2_in*1e6:.1f} ppm")

        # Solve PFR physics via Numba JIT integrator
        conversion, t_out, t_peak, L_prof, T_prof, X_prof = numba_ops.solve_deoxo_pfr_step(
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
            Cp_mix=DeoxoConstants.CP_MIX_AVG_J_MOL_K,
            y_o2_target=DeoxoConstants.MAX_ALLOWED_O2_OUT_MOLE_FRAC
        )
        
        # DEBUG LOG
        # print(f"DEBUG DEOXO: X={conversion}, T_out={t_out}, T_peak={t_peak}")

        self.last_conversion_o2 = conversion
        self.last_peak_temp_k = t_peak
        self.last_profiles = {
            'L': L_prof,
            'T': T_prof,
            'X': X_prof
        }

        # Stoichiometric mass balance: 2H₂ + O₂ → 2H₂O
        n_o2_in = molar_flow_total * y_o2_in
        n_o2_consumed = n_o2_in * conversion
        n_h2_consumed = 2 * n_o2_consumed
        n_h2o_gen = 2 * n_o2_consumed

        # Update molar inventories
        new_moles = {}
        for s, y in y_fracs.items():
            new_moles[s] = molar_flow_total * y

        new_moles['O2'] = max(0.0, new_moles.get('O2', 0.0) - n_o2_consumed)
        new_moles['H2'] = max(0.0, new_moles.get('H2', 0.0) - n_h2_consumed)
        new_moles['H2O'] = new_moles.get('H2O', 0.0) + n_h2o_gen

        # Normalize to mole fractions (internal) -> Then convert to Mass for Output Stream
        total_moles_out = sum(new_moles.values())
        if total_moles_out > 0:
            y_out = {s: n / total_moles_out for s, n in new_moles.items()}
        else:
            y_out = y_fracs

        # Convert back to Mass Fractions for Stream
        # x_i = (y_i * M_i) / M_mix_out
        mw_mix_out = sum(y * MW.get(s, MW['other']) for s, y in y_out.items())
        
        new_comp = {}
        if mw_mix_out > 0:
            for s, y in y_out.items():
                mw_i = MW.get(s, MW['other'])
                new_comp[s] = (y * mw_i) / mw_mix_out
        else:
            new_comp = comp # Fallback

        # Mass is conserved (2×2 + 32 = 36 g reactants, 2×18 = 36 g product)
        mass_flow_out = stream.mass_flow_kg_h

        # Pressure drop using Ergun-type velocity scaling
        u_design = DeoxoConstants.DESIGN_SURF_VEL_M_S # Calibrated
        dp_design = DeoxoConstants.DESIGN_DP_BAR      # Calibrated (0.05 bar)

        # Current velocity from volumetric flow
        rho_gas = (stream.pressure_pa * mw_mix) / (
            GasConstants.R_UNIVERSAL_J_PER_MOL_K * stream.temperature_k
        )
        vol_flow = (stream.mass_flow_kg_h / 3600.0) / rho_gas
        u_curr = vol_flow / DeoxoConstants.AREA_REACTOR_M2

        # Ergun scaling: ΔP ∝ u^1.5 (intermediate regime)
        if u_design > 0:
            ratio = u_curr / u_design
            dp_curr = dp_design * (ratio ** 1.5)
        else:
            dp_curr = 0.0

        self.last_pressure_drop_bar = dp_curr
        p_out = stream.pressure_pa - (dp_curr * 1e5)

        # Create output stream
        self.output_stream = Stream(
            mass_flow_kg_h=mass_flow_out,
            temperature_k=t_out,
            pressure_pa=p_out,
            composition=new_comp,
            phase='gas'
        )

    def receive_input(self, port_name: str, value: Any, resource_type: str = 'stream') -> float:
        """
        Accept inlet stream from upstream component.

        Args:
            port_name (str): Target port ('inlet').
            value (Any): Stream object containing H₂ with trace O₂.
            resource_type (str): Resource classification hint. Default: 'stream'.

        Returns:
            float: Mass flow rate accepted (kg/h), or 0.0 if input rejected.
        """
        if port_name == 'inlet' and isinstance(value, Stream):
            self.input_stream = value
            return value.mass_flow_kg_h
        return 0.0

    def get_output(self, port_name: str) -> Any:
        """
        Retrieve outlet stream from specified port.

        Args:
            port_name (str): Port identifier ('outlet').

        Returns:
            Stream: Processed outlet stream with reduced O₂ content, or None.
        """
        if port_name == 'outlet':
            return self.output_stream
        return None

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """
        Define the physical connection ports for this component.

        Returns:
            Dict[str, Dict[str, str]]: Port definitions.
        """
        return {
            'inlet': {'type': 'input', 'resource_type': 'gas', 'units': 'kg/h'},
            'outlet': {'type': 'output', 'resource_type': 'gas', 'units': 'kg/h'}
        }

    def get_state(self) -> Dict[str, Any]:
        """
        Retrieve the component's current operational state.

        Fulfills the Component Lifecycle Contract state access.

        Returns:
            Dict[str, Any]: State dictionary containing:
                - conversion_o2_percent (float): O₂ conversion (%).
                - peak_temperature_c (float): Maximum reactor temperature (°C).
                - pressure_drop_mbar (float): Total pressure drop (mbar).
        """
        return {
            **super().get_state(),
            'conversion_o2_percent': self.last_conversion_o2 * 100.0,
            'peak_temperature_c': self.last_peak_temp_k - 273.15,
            'pressure_drop_mbar': self.last_pressure_drop_bar * 1000.0
        }

    def get_last_profiles(self) -> Dict[str, np.ndarray]:
        """
        Retrieve internal profiles from the last step.
        
        Returns:
            Dict containing arrays for 'L' (m), 'T' (K), and 'X' (conversion).
        """
        return self.last_profiles
