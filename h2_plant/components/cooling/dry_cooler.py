"""
Dry Cooler (Indirect Cooling System) Component.

This module implements a two-stage indirect cooling system for hydrogen and
oxygen process streams. The architecture separates explosive process gases
from the large external air-cooled heat exchanger, enhancing safety.

Physical Model (Coupled Exchangers)
-----------------------------------
The system solves two coupled heat transfer problems using the $\varepsilon$-NTU method:

1.  **Stage 1: Process Gas $\to$ Glycol Loop (TQC)**:
    *   **Type**: Counter-flow Shell & Tube.
    *   **Physics**: Counter-flow follows:
        $$ \varepsilon = \frac{1 - \exp[-NTU(1-C_r)]}{1 - C_r \exp[-NTU(1-C_r)]} $$
        where $C_r = C_{min}/C_{max}$ and $NTU = UA/C_{min}$.

2.  **Stage 2: Glycol Loop $\to$ Atmosphere (Dry Cooler)**:
    *   **Type**: Cross-flow Finned Tube (Unmixed/Unmixed).
    *   **Physics**: Cross-flow follows empirical correlations for unmixed fluids.

3.  **Thermal Inertia**:
    The glycol loop temperature propagates quasi-dynamically between time steps:
    $$ T_{glycol, cold}^{t+1} \leftarrow f(T_{glycol, hot}^t, Q_{DC}^t, C_{glycol}) $$

Architecture
------------
*   **Component Lifecycle Contract (Layer 1)**:
    *   `initialize()`: Defers geometry setup until first fluid receipt (Lazy Config).
    *   **`step()`**: Executes the sequential thermal solution (Gas $\to$ TQC $\to$ DC $\to$ Air).
    *   `get_state()`: Exposes effectiveness ($\varepsilon$) and duties ($Q$) for analysis.

References
----------
*   Incropera, F.P. & DeWitt, D.P. (2007). Fundamentals of Heat and Mass Transfer.
*   Shah, R.K. & Sekulić, D.P. (2003). Fundamentals of Heat Exchanger Design.
"""

import logging
from typing import Dict, Any, Optional
import numpy as np

from h2_plant.core.component import Component
from h2_plant.core.stream import Stream
from h2_plant.core.constants import (
    GasConstants,
    ConversionFactors,
    DryCoolerIndirectConstants as DCC
)
from h2_plant.optimization import numba_ops

# Import mixture thermodynamics for rigorous Cp calculations
try:
    from h2_plant.optimization import mixture_thermodynamics as mix_thermo
    MIX_THERMO_AVAILABLE = True
except ImportError:
    mix_thermo = None
    MIX_THERMO_AVAILABLE = False
# Explicitly import numba_ops (avoid circular import issues if any)
from h2_plant.optimization import numba_ops
from h2_plant.optimization.coolprop_lut import CoolPropLUT
import math

logger = logging.getLogger(__name__)


class DryCooler(Component):
    """
    Two-stage indirect cooling system (Process Gas $\to$ Glycol $\to$ Ambient Air).

    This component models a closed safety loop. Process gas heat is transferred to an intermediate 
    glycol/water mixture, which is subsequently cooled by forced ambient air.

    **Architecture & Lifecycle (Layer 1)**:
    *   **Initialization**: Lazy configuration upon first `receive_input` enables auto-detection 
        of gas species (H2 or O2) to select appropriate heat exchanger geometries.
    *   **Execution**: `step()` solves the thermal circuit sequentially.

    **Thermodynamic State Variables**:
    *   $T_{glycol, hot}$: Temperature leaving TQC, entering DC.
    *   $T_{glycol, cold}$: Temperature leaving DC, entering TQC (re-circulated).

    Attributes:
        fluid_type (str): Detected active species ('H2' or 'O2').
        tqc_duty_kw (float): Thermal load on the process interchanger [kW].
        dc_duty_kw (float): Thermal heat rejection to atmosphere [kW].
        fan_power_kw (float): Parasitic electrical load for air movement [kW].
        tqc_effectiveness (float): Realized effectiveness $\varepsilon_{TQC}$ [0-1].
        dc_effectiveness (float): Realized effectiveness $\varepsilon_{DC}$ [0-1].
    """

    def __init__(self, component_id: str = "dry_cooler", use_central_utility: bool = True, **kwargs) -> None:
        """
        Initialize the DryCooler component.

        Creates an indirect cooling system instance with default state.
        Heat exchanger geometry and flow rates are configured automatically
        when the first input stream is received, based on detected gas species.

        Args:
            component_id (str): Unique identifier for this component instance.
                Used for logging and registry lookup. Default: "dry_cooler".
            use_central_utility (bool): If True, use CoolingManager for glycol.
            **kwargs: Configuration overrides (e.g., target_outlet_temp_c).
        """
        super().__init__()
        self.component_id = component_id
        self.use_central_utility = use_central_utility
        self.cooling_manager = None  # Set during initialize()

        # Process stream state
        self.fluid_type = "Unknown"
        self.inlet_stream: Optional[Stream] = None
        self.outlet_stream: Optional[Stream] = None

        # Glycol loop state variables
        self.glycol_flow_kg_s = 0.0
        self.glycol_cp_j_kg_k = 0.0
        self.t_glycol_cold_c = DCC.T_REF_IN_TQC_DEFAULT
        self.t_glycol_hot_c = DCC.T_REF_IN_TQC_DEFAULT
        
        # Override target temperature if provided
        self.target_outlet_temp_c = kwargs.get('target_outlet_temp_c', kwargs.get('target_temp_c', None))
        
        # Scaling factor based on design capacity (Baseline = 100 kW)
        self.design_capacity_kw = kwargs.get('design_capacity_kw', 100.0)
        self.scaling_factor = self.design_capacity_kw / 100.0

        # TQC heat exchanger parameters
        self.tqc_area_m2 = 0.0
        self.tqc_u_value = DCC.U_VALUE_TQC_W_M2_K

        # DC heat exchanger parameters
        self.dc_area_m2 = 0.0
        self.dc_u_value = DCC.U_VALUE_DC_W_M2_K
        self.dc_air_flow_kg_s = 0.0

        # Thermal performance metrics
        self.tqc_duty_kw = 0.0
        self.dc_duty_kw = 0.0
        self.fan_power_kw = 0.0
        self.outlet_temp_c = 0.0
        self.tqc_effectiveness = 0.0
        self.dc_effectiveness = 0.0

    @property
    def power_kw(self) -> float:
        """Expose power consumption in kW for dispatch tracking."""
        return self.fan_power_kw

    def initialize(self, dt: float, registry: Any) -> None:
        """
        Prepare the component for simulation execution.

        **Lifecycle Contract**:
        Stores simulation context. Note that physical geometry configuration is **deferred** 
        until `_configure_geometry()` is triggered by the first mass flow, allowing 
        runtime adaptation to the connected process fluid (H2 vs O2).

        Args:
            dt (float): Simulation timestep [hours].
            registry (ComponentRegistry): Central services provider.
        """
        super().initialize(dt, registry)
        
        # Look up CoolingManager if using centralized cooling
        if self.use_central_utility:
            self.cooling_manager = registry.get("cooling_manager") if registry else None
            if not self.cooling_manager:
                logger.warning(f"{self.component_id}: CoolingManager not found. Using local loop.")

    def _configure_geometry(self, stream: Stream) -> None:
        """
        Configure heat exchanger geometry based on detected gas species.

        Selects appropriate heat transfer areas, coolant flow rates, and
        air flow rates from design constants. The glycol mixture heat capacity
        is calculated assuming a 40% glycol / 60% water volumetric mixture.

        Physical Basis:
            Glycol Cp is computed as a mass-weighted average:
            Cp_mix = (1 - f_glycol) × Cp_water + f_glycol × Cp_glycol
            where Cp_water ≈ 4180 J/(kg·K) and Cp_glycol ≈ 2430 J/(kg·K).

        Args:
            stream (Stream): Input stream used to detect gas species from
                composition. H₂-dominant streams use hydrogen-rated equipment;
                O₂-dominant streams use oxygen-rated equipment.
        """
        h2_frac = stream.composition.get('H2', 0.0)
        o2_frac = stream.composition.get('O2', 0.0)

        f_g = DCC.GLYCOL_FRACTION
        self.glycol_cp_j_kg_k = (1 - f_g) * 4180.0 + f_g * 2430.0

        if h2_frac > o2_frac:
            self.fluid_type = "H2"
            self.tqc_area_m2 = DCC.AREA_H2_TQC_M2 * self.scaling_factor
            self.dc_area_m2 = DCC.AREA_H2_DC_M2 * self.scaling_factor
            self.glycol_flow_kg_s = DCC.M_DOT_REF_H2 * self.scaling_factor
            self.dc_air_flow_kg_s = DCC.MDOT_AIR_DESIGN_H2_KG_S * self.scaling_factor
        else:
            self.fluid_type = "O2"
            self.tqc_area_m2 = DCC.AREA_O2_TQC_M2 * self.scaling_factor
            self.dc_area_m2 = DCC.AREA_O2_DC_M2 * self.scaling_factor
            self.glycol_flow_kg_s = DCC.M_DOT_REF_O2 * self.scaling_factor
            self.dc_air_flow_kg_s = DCC.MDOT_AIR_DESIGN_O2_KG_S * self.scaling_factor

    def step(self, t: float) -> None:
        """
        Execute one simulation timestep.

        **Coupled Thermal Solvers**:
        The method solves two heat exchangers in series:

        1.  **TQC (Process $\to$ Glycol)**:
            *   Calculates heat capacity rates $C_{gas}, C_{glycol}$.
            *   Solves $\varepsilon$-NTU (Counter-Flow) $\to$ $Q_{TQC}$.
            *   Updates $T_{gas,out}$ and $T_{glycol,hot}$.

        2.  **DC (Glycol $\to$ Air)**:
            *   Calculates $C_{air}$.
            *   Solves $\varepsilon$-NTU (Cross-Flow) $\to$ $Q_{DC}$.
            *   Updates $T_{glycol,cold}$ for next step (thermal memory).

        3.  **Condensation Logic**:
            *   Checks partial pressure $P_{H2O} > P_{sat}(T_{out})$.
            *   Performs mass transfer Vapor $\to$ Liquid if saturated.

        Args:
            t (float): Current simulation time [hours].
        """
        super().step(t)

        if not self.inlet_stream or self.inlet_stream.mass_flow_kg_h <= 0:
            self.outlet_stream = None
            self.tqc_duty_kw = 0.0
            self.fan_power_kw = 0.0
            return

        # Gas stream properties
        m_dot_gas = self.inlet_stream.mass_flow_kg_h / 3600.0

        # PERFORMANCE: Only use rigorous mix_thermo for true multi-species streams
        # For near-pure streams (>98% single species), fallback is faster and equally accurate
        comp = self.inlet_stream.composition
        dominant_frac = max(comp.values()) if comp else 0
        use_mix_thermo = (dominant_frac < 0.98 and MIX_THERMO_AVAILABLE and mix_thermo is not None)
        cp_gas_mix = 0.0
        
        if use_mix_thermo:
            try:
                lut_manager = None
                if hasattr(self, 'registry') and self.registry is not None:
                    lut_manager = self.registry.get('lut_manager')

                if (lut_manager is not None and 
                    lut_manager.stacked_C is not None and 
                    lut_manager._pressure_grid is not None):
                    
                    # optimized JIT path
                    mass_fracs_arr, _, _, _ = self.inlet_stream.get_composition_arrays()
                    cp_gas_mix = numba_ops.get_mix_cp_jit(
                        self.inlet_stream.pressure_pa,
                        self.inlet_stream.temperature_k,
                        mass_fracs_arr,
                        lut_manager.stacked_C,
                        lut_manager._pressure_grid,
                        lut_manager._temperature_grid
                    )
                elif lut_manager is not None:
                     # Python fallback (slow but robust if JIT not ready)
                     P_in = self.inlet_stream.pressure_pa
                     T_in = self.inlet_stream.temperature_k
                     h_t1 = mix_thermo.get_mixture_enthalpy(comp, P_in, T_in, lut_manager)
                     h_t2 = mix_thermo.get_mixture_enthalpy(comp, P_in, T_in - 1.0, lut_manager)
                     cp_gas_mix = h_t1 - h_t2
            except Exception as e:
                # logger.warning(f"DryCooler mixture thermodynamics error: {e}")
                pass

        # Fallback: Mass-weighted average of constant Cp (FAST)
        if cp_gas_mix <= 0:
            for sp, y in comp.items():
                if sp == 'H2O':
                    cp_sp = 1860.0  # Water vapor Cp (J/kg·K)
                elif sp == 'H2':
                    cp_sp = GasConstants.CP_H2_AVG
                elif sp == 'O2':
                    cp_sp = GasConstants.CP_O2_AVG
                else:
                    cp_sp = 1000.0  # Conservative default
                cp_gas_mix += y * cp_sp

        # Enhancement for wet streams with entrained liquid
        h2o_liq_frac = comp.get('H2O_liq', 0.0)
        if h2o_liq_frac > 0:
            # Liquid water thermal mass included implicitly via higher effective Cp
            pass

        # Heat capacity rates
        C_hot_gas = m_dot_gas * cp_gas_mix
        C_coolant = self.glycol_flow_kg_s * self.glycol_cp_j_kg_k

        # ================================================================
        # Stage 1: TQC (Counter-Flow Heat Exchanger)
        # ================================================================
        # Counter-flow achieves higher effectiveness than parallel-flow
        # for the same NTU, making it preferred for process applications.
        T_gas_in_k = self.inlet_stream.temperature_k
        
        # Get glycol inlet temperature from CoolingManager or local state
        if self.cooling_manager:
            T_glycol_in_k = self.cooling_manager.glycol_supply_temp_c + 273.15
        else:
            T_glycol_in_k = self.t_glycol_cold_c + 273.15

        C_min_tqc = min(C_hot_gas, C_coolant)
        C_max_tqc = max(C_hot_gas, C_coolant)

        if C_min_tqc > 1e-9:
            R_tqc = C_min_tqc / C_max_tqc if C_max_tqc > 1e-9 else 0.0
            NTU_tqc = (self.tqc_u_value * self.tqc_area_m2) / C_min_tqc
            eff_tqc = numba_ops.counter_flow_ntu_effectiveness(NTU_tqc, R_tqc)
        else:
            NTU_tqc = 0.0
            eff_tqc = 0.0

        self.tqc_effectiveness = eff_tqc

        Q_max_tqc = C_min_tqc * (T_gas_in_k - T_glycol_in_k)
        Q_tqc = eff_tqc * Q_max_tqc
        self.tqc_duty_kw = Q_tqc / 1000.0

        # Energy balance: outlet temperatures
        if C_hot_gas > 1e-9:
            T_gas_out_k = T_gas_in_k - Q_tqc / C_hot_gas
        else:
            T_gas_out_k = T_gas_in_k

        if C_coolant > 1e-9:
            T_glycol_out_k = T_glycol_in_k + Q_tqc / C_coolant
        else:
            T_glycol_out_k = T_glycol_in_k
            
        self.t_glycol_hot_c = T_glycol_out_k - 273.15

        # Register load with CoolingManager (centralized utility mode)
        if self.cooling_manager:
            self.cooling_manager.register_glycol_load(
                duty_kw=self.tqc_duty_kw,
                flow_kg_s=self.glycol_flow_kg_s,
                return_temp_c=self.t_glycol_hot_c
            )

        # ================================================================
        # Stage 2: DC (Cross-Flow Air Cooler)
        # ================================================================
        # Cross-flow geometry is standard for forced-air coolers due to
        # fan and duct arrangement, though less effective than counter-flow.
        T_air_in_k = DCC.T_AIR_DESIGN_C + 273.15
        C_air = self.dc_air_flow_kg_s * DCC.CP_AIR_J_KG_K

        C_min_dc = min(C_coolant, C_air)
        C_max_dc = max(C_coolant, C_air)
        
        if C_min_dc > 1e-9:
            R_dc = C_min_dc / C_max_dc if C_max_dc > 1e-9 else 0.0
            NTU_dc = (self.dc_u_value * self.dc_area_m2) / C_min_dc
            eff_dc = numba_ops.dry_cooler_ntu_effectiveness(NTU_dc, R_dc)
        else:
            NTU_dc = 0.0
            eff_dc = 0.0

        self.dc_effectiveness = eff_dc

        Q_max_dc = C_min_dc * (T_glycol_out_k - T_air_in_k)
        Q_dc = eff_dc * Q_max_dc
        self.dc_duty_kw = Q_dc / 1000.0

        # Glycol return temperature (quasi-dynamic state update)
        if C_coolant > 1e-9:
            T_glycol_return_k = T_glycol_out_k - Q_dc / C_coolant
        else:
            T_glycol_return_k = T_glycol_out_k
        self.t_glycol_cold_c = T_glycol_return_k - 273.15

        # Gas-side pressure drop through TQC internals
        P_out = self.inlet_stream.pressure_pa - (DCC.DP_LIQ_TQC_BAR * 1e5)

        # ================================================================
        # Min Approach Temperature Constraint (Energy-Conserving)
        # ================================================================
        # Industrial Dry Coolers have min 5-10°C approach. If theoretical 
        # T_out violates this, we must back-calculate the achievable Q.
        # Use user-defined target if available, otherwise physics limit
        MIN_APPROACH_K = 5.0
        phys_limit_k = T_air_in_k + MIN_APPROACH_K
        
        T_out_limit_k = phys_limit_k
        if self.target_outlet_temp_c is not None:
             # If user wants 50C, but physics says min is 30C (25 air + 5), 
             # then limit is max(50, 30) = 50.
             # If user wants 20C, but physics says min is 30C, 
             # then limit is max(20, 30) = 30.
             user_target_k = self.target_outlet_temp_c + 273.15
             T_out_limit_k = max(user_target_k, phys_limit_k)
        
        if T_gas_out_k < T_out_limit_k:
            # Clamp T_out and recalculate Q to conserve energy
            T_gas_out_k = T_out_limit_k
            Q_tqc_real = C_hot_gas * (T_gas_in_k - T_gas_out_k)
            self.tqc_duty_kw = Q_tqc_real / 1000.0

        # Prepare output stream
        self.outlet_temp_c = T_gas_out_k - 273.15

        # --- Flash Calculation (Condensation, Rigorous) ---
        # Checks for phase change. If $y_{H2O} > y_{sat} = P_{sat}/P_{total}$, condenses excess water.
        
        inlet_comp = self.inlet_stream.composition.copy()
        outlet_comp = inlet_comp.copy()
        m_dot_total_kg_h = self.inlet_stream.mass_flow_kg_h
        
        # 1. Quantify Total Inlet Water ( Vapor + Liquid from upstream )
        # Note: 'H2O' in composition is vapor. 'H2O_liq' is liquid.
        x_H2O_vap_in = inlet_comp.get('H2O', 0.0)
        x_H2O_liq_in = inlet_comp.get('H2O_liq', 0.0)
        
        # Add extra liquid if present
        m_H2O_liq_extra_in = 0.0
        if self.inlet_stream.extra:
            m_H2O_liq_extra_in = self.inlet_stream.extra.get('m_dot_H2O_liq_accomp_kg_s', 0.0) * 3600.0
            
        m_H2O_vap_in = x_H2O_vap_in * m_dot_total_kg_h
        m_H2O_liq_in = (x_H2O_liq_in * m_dot_total_kg_h) + m_H2O_liq_extra_in
        m_H2O_total_in = m_H2O_vap_in + m_H2O_liq_in
        
        # 2. Convert to Molar Flows for Flash
        # Need average MW for non-condensables to get total moles
        # Simplified: Assume binary system of (H2O) + (Rest)
        MW_H2O = 18.015e-3
        MW_H2 = 2.016e-3
        
        n_dot_H2O_total = m_H2O_total_in / MW_H2O
        
        n_dot_others = 0.0
        for sp, frac in inlet_comp.items():
            if sp not in ('H2O', 'H2O_liq') and frac > 1e-9:
                # Fallback MW if not H2O
                mw = MW_H2 if sp == 'H2' else 28.0e-3 # Default to N2/CO2 approx
                n_dot_others += (frac * m_dot_total_kg_h) / mw
                
        n_dot_total = n_dot_H2O_total + n_dot_others
        
        if n_dot_total > 1e-9 and P_out > 0:
            z_H2O = n_dot_H2O_total / n_dot_total
            
            # 3. Calculate Saturation & K-value
            # Use JIT-optimized Antoine for speed
            P_sat = numba_ops.calculate_water_psat_jit(T_gas_out_k)
            K_value = P_sat / P_out
            
            # 4. Solve Flash (Rachford-Rice optimized for single condensable)
            # Returns beta (Vapor Fraction V/F)
            beta = numba_ops.solve_rachford_rice_single_condensable(z_H2O, K_value)
            
            # 5. Convert back to Mass
            # 5. Convert back to Mass
            n_dot_vapor = n_dot_total * beta
            n_dot_liquid = n_dot_total * (1.0 - beta)
            
            # In single condensable assumption, liquid is pure water
            # m_H2O_liq_out represents TOTAL liquid water at equilibrium
            m_H2O_liq_out = n_dot_liquid * MW_H2O
            m_H2O_vapor_out = m_H2O_total_in - m_H2O_liq_out
                 
            # Clamp for physics safety
            m_H2O_liq_out = max(0.0, min(m_H2O_total_in, m_H2O_liq_out))
            m_H2O_vapor_out = max(0.0, m_H2O_total_in - m_H2O_liq_out)

        else:
             m_H2O_liq_out = m_H2O_liq_in
             m_H2O_vapor_out = m_H2O_vap_in

        # 3. Update Composition & Mass Balance
        # Incorporate 'extra' liquid water from upstream sources (if any)
        # into the main stream mass definition for the outlet.
        m_total_out = self.inlet_stream.mass_flow_kg_h
        m_H2O_liq_total_out = m_H2O_liq_out
        


        # New total mass - DO NOT add extra liquid (it's already in inlet mass_flow if present)
        # Prevents double-counting when upstream (e.g., Cyclone) includes entrained liquid in mass_flow
        m_total_new = m_total_out
        
        if m_total_new > 0:
             # Recalculate fractions based on new total mass
             outlet_comp['H2O'] = m_H2O_vapor_out / m_total_new
             outlet_comp['H2O_liq'] = m_H2O_liq_total_out / m_total_new
             
             # Re-normalize other species (mass conserved, fraction decreases)
             for s in inlet_comp:
                 if s not in ('H2O', 'H2O_liq'):
                     m_s = inlet_comp[s] * m_total_out
                     outlet_comp[s] = m_s / m_total_new
        
        # Prepare output stream using new total mass
        # Remove the 'extra' liquid key since it's now merged into composition
        out_extra = self.inlet_stream.extra.copy() if self.inlet_stream.extra else {}
        if 'm_dot_H2O_liq_accomp_kg_s' in out_extra:
            del out_extra['m_dot_H2O_liq_accomp_kg_s']

        self.outlet_stream = Stream(
            mass_flow_kg_h=m_total_new,
            temperature_k=T_gas_out_k,
            pressure_pa=P_out,
            composition=outlet_comp,
            phase='mixed',
            extra=out_extra
        )

        # Fan power consumption: P = V̇ × ΔP / η_fan
        vol_air = self.dc_air_flow_kg_s / DCC.RHO_AIR_KG_M3
        power_j_s = (vol_air * DCC.DP_AIR_DESIGN_PA) / DCC.ETA_FAN
        self.fan_power_kw = power_j_s / 1000.0

    def receive_input(self, port_name: str, value: Any, resource_type: str = None) -> float:
        """
        Accept an input stream at the specified port.

        **Lazy Configuration Logic**:
        On first `fluid_in` receipt, this method triggers `_configure_geometry()` logic to 
        specialize the component for Hydrogen or Oxygen service based on stream composition.

        Args:
            port_name (str): Target port ('fluid_in', 'electricity_in').
            value (Any): Stream or float (Power).
            resource_type (str): Optional hint.

        Returns:
            float: Accepted mass flow [kg/h] or Power demand [kW].
        """
        if port_name == "fluid_in" and isinstance(value, Stream):
            self.inlet_stream = value
            self._configure_geometry(value)
            return value.mass_flow_kg_h
        elif port_name == "electricity_in":
            return self.fan_power_kw
        return 0.0

    def get_output(self, port_name: str) -> Any:
        """
        Retrieve the output stream from a specified port.

        **Physics**:
        Returns state computed by the coupled $\varepsilon$-NTU solver. Includes any condensed 
        liquid phase if separation occurred.

        Args:
            port_name (str): 'fluid_out'.

        Returns:
            Stream: Cooled process gas.
        """
        if port_name == "fluid_out":
            return self.outlet_stream if self.outlet_stream else Stream(0.0)
        return None

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """
        Define the physical connection ports for this component.

        Returns:
            Dict[str, Dict[str, str]]:
            -   **fluid_in**: Hot process gas source.
            -   **fluid_out**: Cooled process gas destination.
            -   **electricity_in**: Fan power supply.
        """
        return {
            'fluid_in': {'type': 'input', 'resource_type': 'stream'},
            'electricity_in': {'type': 'input', 'resource_type': 'electricity'},
            'fluid_out': {'type': 'output', 'resource_type': 'stream'}
        }

    def get_state(self) -> Dict[str, Any]:
        """
        Retrieve the component's current operational state.

        **Layer 1 Contract**:
        Returns thermal telemetry for central monitoring, critical for verifying 
        heat rejection limits ($Q_{DC}$) and process outlet temperatures.

        Returns:
            Dict[str, Any]: State metrics:
            -   **tqc_duty_kw** (float): Process heat load.
            -   **dc_duty_kw** (float): Rejected heat load.
            -   **t_glycol_hot/cold** (float): Internal loop temperatures [°C].
            -   **effectiveness** (float): Realized $\varepsilon$ for diagnostics.
        """
        return {
            **super().get_state(),
            'fluid_type': self.fluid_type,
            'tqc_duty_kw': self.tqc_duty_kw,
            'dc_duty_kw': self.dc_duty_kw,
            'heat_rejected_kw': self.dc_duty_kw,  # Alias for engine_dispatch
            'fan_power_kw': self.fan_power_kw,
            'outlet_temp_c': self.outlet_temp_c,
            'glycol_hot_c': self.t_glycol_hot_c,
            'glycol_cold_c': self.t_glycol_cold_c,
            'tqc_effectiveness': self.tqc_effectiveness,
            'dc_effectiveness': self.dc_effectiveness,
            'outlet_o2_ppm_mol': (self.outlet_stream.get_total_mole_frac('O2') * 1e6) if self.outlet_stream else 0.0
        }
