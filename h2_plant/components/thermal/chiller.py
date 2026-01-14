"""
Chiller Component for Thermal Management.

This module implements a chiller/heat exchanger for cooling fluid streams
in PEM and SOEC electrolysis systems. The component uses enthalpy-based
heat transfer calculations with gas-specific Cp fallback.

Heat Transfer Model:
    Primary method uses enthalpy difference:
    **Q = ṁ × (h_in - h_out)**

    Fallback for gas streams uses constant Cp:
    **Q = ṁ × Cp × (T_in - T_out)**

    Where Cp(H₂) = 14,300 J/(kg·K) and Cp(O₂) = 918 J/(kg·K).

Electrical Consumption:
    Chillers consume electrical power based on Coefficient of Performance:
    **W_elec = |Q| / COP**

    Typical COP values: 3-5 for vapor-compression systems.

Architecture:
    Implements the Component Lifecycle Contract (Layer 1):
    - `initialize()`: Standard initialization.
    - `step()`: Calculates cooling load, electrical power, outlet conditions.
    - `get_state()`: Returns thermal loads, outlet conditions, and power.

Process Flow Integration:
    - HX-1, HX-2, HX-3: PEM electrolyzer cooling
    - HX-5, HX-6: Compressor intercooling
    - HX-10, HX-11: Product gas cooling
"""

from typing import Dict, Any, Optional
import numpy as np
import logging

from h2_plant.core.component import Component
from h2_plant.core.stream import Stream
from h2_plant.core.constants import GasConstants
from h2_plant.models.thermal_inertia import ThermalInertiaModel
from h2_plant.models.flow_dynamics import PumpFlowDynamics
from h2_plant.optimization.coolprop_lut import CoolPropLUT
from h2_plant.optimization import numba_ops
import math

# Import mixture thermodynamics for rigorous enthalpy calculations
try:
    from h2_plant.optimization import mixture_thermodynamics as mix_thermo
    MIX_THERMO_AVAILABLE = True
except ImportError:
    mix_thermo = None
    MIX_THERMO_AVAILABLE = False


class Chiller(Component):
    """
    Heat exchanger for cooling fluid streams with COP-based electrical model.

    Calculates cooling load using enthalpy difference (preferred) or gas-specific
    Cp (fallback). Models electrical consumption via COP and tracks heat rejected.

    This component fulfills the Component Lifecycle Contract (Layer 1):
        - `initialize()`: Standard component initialization.
        - `step()`: Calculates cooling, updates outlet stream.
        - `get_state()`: Returns cooling load, power, and outlet conditions.

    Energy Balance:
        - Q_cooling = ṁ × (h_in - h_target) [enthalpy method]
        - W_electrical = Q_cooling / COP
        - Q_rejected = Q_cooling + W_electrical (heat to cooling water)

    Attributes:
        cooling_capacity_kw (float): Maximum cooling capacity (kW).
        cop (float): Coefficient of Performance for electrical calculation.
        target_temp_k (float): Target outlet temperature (K).
        pressure_drop_bar (float): Pressure loss through exchanger (bar).

    Example:
        >>> chiller = Chiller(cooling_capacity_kw=100.0, cop=4.0, target_temp_k=298.15)
        >>> chiller.initialize(dt=1/60, registry=registry)
        >>> chiller.receive_input('fluid_in', hot_stream, 'stream')
        >>> chiller.step(t=0.0)
        >>> cooled = chiller.get_output('fluid_out')
    """

    def __init__(
        self,
        component_id: str = "chiller",
        cooling_capacity_kw: float = 100.0,
        efficiency: float = 0.95,
        target_temp_k: float = 298.15,
        cop: float = 4.0,
        pressure_drop_bar: float = 0.2,
        enable_dynamics: bool = False,
        use_central_utility: bool = True
    ):
        """
        Initialize the chiller.

        Args:
            component_id (str): Unique identifier. Default: 'chiller'.
            cooling_capacity_kw (float): Maximum cooling capacity in kW. Default: 100.0.
            efficiency (float): Heat transfer efficiency (0-1). Default: 0.95.
            target_temp_k (float): Target outlet temperature in K. Default: 298.15 (25°C).
            cop (float): Coefficient of Performance. Higher values indicate more
                efficient cooling. Typical range: 3-5. Default: 4.0.
            pressure_drop_bar (float): Pressure loss through exchanger in bar.
                Default: 0.2.
            enable_dynamics (bool): Enable pump and thermal inertia dynamics.
                Disable for steady-state reference matching. Default: False.
        """
        super().__init__()
        self.component_id = component_id
        self.cooling_capacity_kw = cooling_capacity_kw
        self.efficiency = efficiency
        self.target_temp_k = target_temp_k
        self.cop = cop
        self.pressure_drop_bar = pressure_drop_bar
        self.enable_dynamics = enable_dynamics
        self.use_central_utility = use_central_utility
        self.cooling_manager = None  # Set during initialize()
        self.cop_nominal = cop  # Store nominal COP for correction

        self.logger = logging.getLogger(f"chiller.{component_id}")

        # Stream state
        self.inlet_stream: Stream = Stream(0.0)
        self.cooling_water_inlet: Optional[Stream] = None
        self.outlet_stream: Stream = Stream(0.0)

        # Thermal tracking
        self.cooling_load_kw: float = 0.0
        self.cooling_water_flow_kg_h: float = 0.0
        self.heat_rejected_kw: float = 0.0
        self.electrical_power_kw: float = 0.0

        # Optional dynamics models
        if self.enable_dynamics:
            self.pump = PumpFlowDynamics(
                initial_flow_m3_h=0.0,
                fluid_inertance_kg_m4=1e9
            )
            self.coolant_thermal = ThermalInertiaModel(
                C_thermal_J_K=1.0e6,
                h_A_passive_W_K=50.0,
                T_initial_K=293.15,
                max_cooling_kw=cooling_capacity_kw
            )
        else:
            self.pump = None
            self.coolant_thermal = None

        # Input buffer for push architecture
        self._input_buffer: list[Stream] = []
        self._last_step_time = -1.0
        self.timestep_cooling_load_kw = 0.0
        self.timestep_electrical_power_kw = 0.0
        self.timestep_heat_rejected_kw = 0.0

    def initialize(self, dt: float, registry: Any) -> None:
        """
        Prepare the component for simulation execution.

        Fulfills the Component Lifecycle Contract initialization phase.

        Args:
            dt (float): Simulation timestep in hours.
            registry (ComponentRegistry): Central registry for component access.
        """
        super().initialize(dt, registry)
        
        # Look up CoolingManager if using centralized cooling
        if self.use_central_utility:
            self.cooling_manager = registry.get("cooling_manager") if registry else None
            if not self.cooling_manager:
                self.logger.warning(f"{self.component_id}: CoolingManager not found. Using fixed COP.")

    def _calculate_cooling_fallback(self) -> float:
        """
        Calculate cooling load using mixture enthalpy (preferred) or constant Cp.

        Priority:
        1. Rigorous: mix_thermo.get_mixture_enthalpy for inlet/outlet
        2. Fallback: Gas-specific constant Cp (H2: 14300, O2: 918 J/kg·K)

        Returns:
            float: Cooling load in Watts (positive = heat removed).
        """
        composition = self.inlet_stream.composition
        mass_flow_kg_s = self.inlet_stream.mass_flow_kg_h / 3600.0
        T_in = self.inlet_stream.temperature_k
        P_in = self.inlet_stream.pressure_pa
        T_target = self.target_temp_k
        
        # PERFORMANCE: Only use rigorous mix_thermo for true multi-species streams
        dominant_frac = max(composition.values()) if composition else 0
        use_mix_thermo = (dominant_frac < 0.98 and MIX_THERMO_AVAILABLE and mix_thermo is not None)
        
        if use_mix_thermo:
            try:
                lut_manager = None
                if hasattr(self, 'registry') and self.registry is not None:
                    lut_manager = self.registry.get('lut_manager')
                
                if (lut_manager is not None and 
                    lut_manager.stacked_H is not None and 
                    lut_manager._pressure_grid is not None):
                    
                    # optimized JIT path
                    mass_fracs_arr, _, _, _ = self.inlet_stream.get_composition_arrays()
                    
                    # Import numba_ops locally if not at top level (or assume it's available via module)
                    # Ideally import at top, but for minimal diff we use the module reference
                    from h2_plant.optimization import numba_ops # Ensure imported
                    
                    h_in = numba_ops.get_mix_enthalpy_fast_jit(
                        P_in, T_in, mass_fracs_arr, 
                        lut_manager.stacked_H, lut_manager._pressure_grid, lut_manager._temperature_grid
                    )
                    h_out = numba_ops.get_mix_enthalpy_fast_jit(
                        P_in, T_target, mass_fracs_arr, 
                        lut_manager.stacked_H, lut_manager._pressure_grid, lut_manager._temperature_grid
                    )

                    Q_dot_W = mass_flow_kg_s * (h_in - h_out)
                    return Q_dot_W
                
                elif lut_manager is not None:
                    h_in = mix_thermo.get_mixture_enthalpy(composition, P_in, T_in, lut_manager)
                    h_out = mix_thermo.get_mixture_enthalpy(composition, P_in, T_target, lut_manager)
                    
                    Q_dot_W = mass_flow_kg_s * (h_in - h_out)
                    
                    self.logger.debug(
                        f"Chiller {self.component_id} using mixture enthalpy: "
                        f"h_in={h_in/1000:.1f} kJ/kg, h_out={h_out/1000:.1f} kJ/kg"
                    )
                    return Q_dot_W
            except Exception as e:
                self.logger.debug(f"Mixture enthalpy failed, using Cp fallback: {e}")
        
        # Fallback to constant Cp
        h2_fraction = composition.get('H2', 0.0)
        o2_fraction = composition.get('O2', 0.0)

        if h2_fraction > o2_fraction:
            Cp_avg = GasConstants.CP_H2_AVG
            gas_type = 'H2'
        else:
            Cp_avg = GasConstants.CP_O2_AVG
            gas_type = 'O2'

        delta_T = T_in - T_target

        # Q = ṁ × Cp × ΔT (positive when cooling: T_in > T_target)
        Q_dot_W = mass_flow_kg_s * Cp_avg * delta_T

        self.logger.debug(
            f"Chiller {self.component_id} using Cp fallback for {gas_type}: "
            f"Cp={Cp_avg:.0f} J/kg·K, ΔT={delta_T:.1f} K"
        )

        return Q_dot_W

    def step(self, t: float) -> None:
        """
        Execute one simulation timestep.

        Performs complete chiller calculation:
        1. Aggregate buffered input streams.
        2. Calculate cooling load (enthalpy method with Cp fallback).
        3. Apply capacity limit and calculate actual outlet temperature.
        4. Compute electrical power from COP.
        5. Estimate cooling water flow for heat rejection.

        Args:
            t (float): Current simulation time in hours.
        """
        super().step(t)

        # Reset timestep accumulators on new timestep
        if t != self._last_step_time:
            self.timestep_cooling_load_kw = 0.0
            self.timestep_electrical_power_kw = 0.0
            self.timestep_heat_rejected_kw = 0.0
            self._last_step_time = t

        # Handle empty input buffer
        if not self._input_buffer:
            self.cooling_load_kw = self.timestep_cooling_load_kw
            self.electrical_power_kw = self.timestep_electrical_power_kw
            self.heat_rejected_kw = self.timestep_heat_rejected_kw
            return

        # Mix buffered streams
        combined_stream = self._input_buffer[0]
        for s in self._input_buffer[1:]:
            combined_stream = combined_stream.mix_with(s)
        self._input_buffer = []

        self.inlet_stream = combined_stream
        mass_flow_kg_s = self.inlet_stream.mass_flow_kg_h / 3600.0

        # Calculate outlet pressure with drop
        outlet_pressure_pa = self.inlet_stream.pressure_pa - (self.pressure_drop_bar * 1e5)
        outlet_pressure_pa = max(outlet_pressure_pa, 1e4)

        # Primary: Enthalpy-based cooling calculation
        try:
            h_in = self.inlet_stream.specific_enthalpy_j_kg

            target_stream = Stream(
                mass_flow_kg_h=self.inlet_stream.mass_flow_kg_h,
                temperature_k=self.target_temp_k,
                pressure_pa=outlet_pressure_pa,
                composition=self.inlet_stream.composition.copy()
            )
            h_target = target_stream.specific_enthalpy_j_kg

            # Q = ṁ × (h_in - h_target): positive when cooling
            if h_in < h_target:
                Q_dot_W = 0.0
                self.logger.debug(f"Chiller {self.component_id}: Inlet enthalpy below target. No cooling needed.")
            else:
                Q_dot_W = mass_flow_kg_s * (h_in - h_target)
                
        except Exception as e:
            # Fallback to Cp-based calculation
            self.logger.debug(f"Enthalpy calc failed, using fallback: {e}")
            Q_dot_W = self._calculate_cooling_fallback()
            
        # Apply capacity limit
        max_Q_W = self.cooling_capacity_kw * 1000.0 * self.efficiency
        final_temp_k = self.target_temp_k

        if Q_dot_W == 0.0:
            final_temp_k = self.inlet_stream.temperature_k
        elif abs(Q_dot_W) > max_Q_W:
            Q_dot_W = np.sign(Q_dot_W) * max_Q_W
            self.logger.warning(
                f"Chiller {self.component_id}: Capacity exceeded. "
                f"Capped at {self.cooling_capacity_kw} kW"
            )
            # Recalculate outlet temperature with limited Q
            # Use LUT-derived Cp or fallback to estimated constant
            comp = self.inlet_stream.composition
            T_in = self.inlet_stream.temperature_k
            P_in = self.inlet_stream.pressure_pa
            
            Cp_est = 14300.0 if comp.get('H2', 0) > 0.5 else 918.0  # Default
            
            # PERFORMANCE: Only use rigorous mix_thermo for multi-species streams
            dominant_frac = max(comp.values()) if comp else 0
            if dominant_frac < 0.98 and MIX_THERMO_AVAILABLE and mix_thermo is not None:
                try:
                    lut_manager = None
                    if hasattr(self, 'registry') and self.registry is not None:
                        lut_manager = self.registry.get('lut_manager')
                    if (lut_manager is not None and 
                        lut_manager.stacked_C is not None):
                         
                        # JIT Path
                        from h2_plant.optimization import numba_ops
                        mass_fracs_arr, _, _, _ = self.inlet_stream.get_composition_arrays()
                        Cp_est = numba_ops.get_mix_cp_jit(
                            P_in, T_in, mass_fracs_arr, 
                            lut_manager.stacked_C, lut_manager._pressure_grid, lut_manager._temperature_grid
                        )
                    elif lut_manager is not None:
                        h_t1 = mix_thermo.get_mixture_enthalpy(comp, P_in, T_in, lut_manager)
                        h_t2 = mix_thermo.get_mixture_enthalpy(comp, P_in, T_in - 1.0, lut_manager)
                        Cp_est = h_t1 - h_t2  # dH/dT
                except Exception:
                    pass
            
            delta_T_real = Q_dot_W / (mass_flow_kg_s * Cp_est)
            final_temp_k = self.inlet_stream.temperature_k - delta_T_real

        cooling_load_kw = Q_dot_W / 1000.0
        self.cooling_load_kw = cooling_load_kw # Update instance state with sensible load

        # Dynamic COP based on cooling water temperature (from CoolingManager)
        # Physics: Carnot Efficiency degradation as condenser temp rises
        # COP approx = COP_nominal * (1 - k * (T_condenser - T_design))
        cw_temp_c = 25.0  # Default design condition
        if self.cooling_manager:
            cw_temp_c = self.cooling_manager.cw_supply_temp_c
        
        cop_correction = 1.0 - 0.03 * (cw_temp_c - 25.0)
        actual_cop = max(1.5, self.cop_nominal * cop_correction)
        self.cop = actual_cop  # Update for state reporting

        # COP-based electrical consumption
        if self.cop > 0:
            batch_electrical_kw = abs(self.cooling_load_kw) / self.cop
        else:
            batch_electrical_kw = 0.0
        
        self.electrical_power_kw = batch_electrical_kw
        
        # Heat rejected to cooling water
        batch_heat_rejected_kw = abs(self.cooling_load_kw) + batch_electrical_kw
        self.heat_rejected_kw = batch_heat_rejected_kw

        # Create outlet stream with flash condensation
        # Flash calculation: determine how much water condenses at outlet T/P
        inlet_comp = self.inlet_stream.composition.copy()
        outlet_comp = inlet_comp.copy()
        m_condensed_kg_h = 0.0
        
        # Get water content from inlet
        x_H2O_in = inlet_comp.get('H2O', 0.0)  # Mass fraction
        
        if x_H2O_in > 0 and outlet_pressure_pa > 0:
            # Calculate saturation pressure at outlet temperature
            try:
                P_sat = CoolPropLUT.PropsSI('P', 'T', final_temp_k, 'Q', 0.0, 'Water')
                if P_sat <= 1e-6 or not math.isfinite(P_sat):
                    raise ValueError("Invalid P_sat")
            except Exception:
                # Antoine equation fallback
                T_C = final_temp_k - 273.15
                A, B, C = 8.07131, 1730.63, 233.426
                P_sat_mmHg = 10 ** (A - B / (C + T_C))
                P_sat = P_sat_mmHg * 133.322
            
            # Maximum water vapor mole fraction at equilibrium
            y_H2O_sat = P_sat / outlet_pressure_pa if outlet_pressure_pa > 0 else 1.0
            y_H2O_sat = min(y_H2O_sat, 1.0)
            
            # Convert inlet mass fraction to mole fraction for comparison
            MW_H2O = 18.015e-3  # kg/mol
            MW_H2 = 2.016e-3
            MW_other = 28.0e-3
            
            # Calculate mole fractions from mass fractions
            # n_i = x_i / M_i (relative moles)
            # y_i = n_i / Σ(n_j) (mole fraction)
            n_total = sum(
                frac / (MW_H2 if s == 'H2' else MW_H2O if s in ('H2O', 'H2O_liq') else MW_other)
                for s, frac in inlet_comp.items()
            )
            if n_total > 0:
                y_H2O_in = (x_H2O_in / MW_H2O) / n_total  # Correct mole fraction
            else:
                y_H2O_in = x_H2O_in
            
            # If inlet water exceeds saturation, condense
            if y_H2O_in > y_H2O_sat:
                # Fraction of water that condenses
                condensation_fraction = 1.0 - (y_H2O_sat / y_H2O_in) if y_H2O_in > 0 else 0.0
                condensation_fraction = max(0.0, min(1.0, condensation_fraction))
                
                # Mass of water condensed from VAPOR
                m_H2O_vapor_in = x_H2O_in * self.inlet_stream.mass_flow_kg_h
                m_condensed_kg_h = m_H2O_vapor_in * condensation_fraction
                
                # Check for existing LIQUID in inlet from TWO sources:
                # 1. Composition-tracked liquid (H2O_liq in composition)
                # 2. Extra liquid (m_dot_H2O_liq_accomp_kg_s in extra dict)
                x_H2O_liq_in = inlet_comp.get('H2O_liq', 0.0)
                m_H2O_liq_comp_in = x_H2O_liq_in * self.inlet_stream.mass_flow_kg_h
                
                # Extra liquid from upstream (e.g., DryCooler, KOD carryover)
                m_H2O_liq_extra_in = 0.0
                if hasattr(self.inlet_stream, 'extra') and self.inlet_stream.extra:
                    m_H2O_liq_extra_in = self.inlet_stream.extra.get('m_dot_H2O_liq_accomp_kg_s', 0.0) * 3600.0
                
                # Total inlet liquid = composition + extra
                m_H2O_liq_in = m_H2O_liq_comp_in + m_H2O_liq_extra_in
                
                # Update outlet composition - PASS WATER DOWNSTREAM
                # We DO NOT remove mass from the stream, just change phase
                m_total_out = self.inlet_stream.mass_flow_kg_h
                
                # New Total Mass must include extra liquid for conservation
                m_total_new = m_total_out + m_H2O_liq_extra_in
                
                if m_total_new > 0:
                    # Recalculate mass fractions
                    m_H2O_vapor_out = m_H2O_vapor_in - m_condensed_kg_h
                    
                    # Total liquid out = Newly Condensed + Inlet Liquid (comp + extra)
                    m_H2O_liq_total_out = m_condensed_kg_h + m_H2O_liq_in
                    
                    # Vapor phase water
                    outlet_comp['H2O'] = m_H2O_vapor_out / m_total_new
                    
                    # Liquid phase water (passed to next component)
                    outlet_comp['H2O_liq'] = m_H2O_liq_total_out / m_total_new
                    
                    # Other species: Calculate physical mass, then new fraction
                    for species in inlet_comp:
                        if species not in ('H2O', 'H2O_liq'):
                            m_species = inlet_comp[species] * m_total_out
                            outlet_comp[species] = m_species / m_total_new

        # Track condensation for state reporting (only new condensation counts for latent heat)
        self.water_condensed_kg_h = m_condensed_kg_h
        
        # Determine final mass flow (fallback if loop above didn't run or m_total_new=0)
        # If m_total_new was calculated in the if block, use it.
        # But wait, m_total_new depends on m_extra which is always calc'd.
        # We need to make sureoutlet_mass_flow is correct in all paths.
        outlet_mass_flow = self.inlet_stream.mass_flow_kg_h + (
            self.inlet_stream.extra.get('m_dot_H2O_liq_accomp_kg_s', 0.0) * 3600.0 
            if hasattr(self.inlet_stream, 'extra') and self.inlet_stream.extra else 0.0
        )
        
        # Handle fallback for H2O_liq if not set (no condensation path?)
        # Logic matches DryCooler structure
        if 'H2O_liq' not in outlet_comp and 'H2O_liq' in inlet_comp:
             # Just dilute existing liquid
             m_liq = inlet_comp['H2O_liq'] * outlet_mass_flow # Approx
             outlet_comp['H2O_liq'] = m_liq / outlet_mass_flow

        self.outlet_stream = Stream(
            mass_flow_kg_h=outlet_mass_flow,
            temperature_k=final_temp_k,
            pressure_pa=outlet_pressure_pa,
            composition=outlet_comp
            # NOTE: Do NOT set extra['m_dot_H2O_liq_accomp_kg_s'] here!
            # Liquid is already tracked in composition['H2O_liq'].
            # Setting both would cause double-counting in get_total_mole_frac.
        )

        # Calculate Latent Heat from condensation Rigorously
        # Uses final_temp_k as the saturation temperature
        try:
            # Enthalpy of Saturated Vapor (Q=1)
            h_vap_sat = CoolPropLUT.PropsSI('H', 'T', final_temp_k, 'Q', 1.0, 'Water')
            # Enthalpy of Saturated Liquid (Q=0)
            h_liq_sat = CoolPropLUT.PropsSI('H', 'T', final_temp_k, 'Q', 0.0, 'Water')
            
            if h_vap_sat == 0.0 or h_liq_sat == 0.0:
                 delta_h_vap = 2450.0 * 1000.0 # Fallback
            else:
                 delta_h_vap = h_vap_sat - h_liq_sat
        except Exception:
             delta_h_vap = 2450.0 * 1000.0
             
        self.latent_heat_kw = (self.water_condensed_kg_h / 3600.0) * (delta_h_vap / 1000.0)
        
        # FIX: The Q_dot_W calculated initially was based on Gas-Phase Enthalpy difference 
        # (Sensible Heat only) because target_stream assumed vapor composition.
        # We must ADD the Latent Heat to get the Total Cooling Load.
        self.sensible_heat_kw = self.cooling_load_kw  # Store initial sensible calc
        self.cooling_load_kw += self.latent_heat_kw   # Update Total Load
        
        # Re-calculate Electrical Power based on CORRECTED Total Load
        if self.cop > 0:
            batch_electrical_kw = abs(self.cooling_load_kw) / self.cop
        else:
            batch_electrical_kw = 0.0
        
        self.electrical_power_kw = batch_electrical_kw
        
        # Re-calculate Heat Rejected
        batch_heat_rejected_kw = abs(self.cooling_load_kw) + batch_electrical_kw
        self.heat_rejected_kw = batch_heat_rejected_kw

        self.timestep_cooling_load_kw += self.cooling_load_kw
        self.timestep_heat_rejected_kw += batch_heat_rejected_kw
        self.timestep_electrical_power_kw += batch_electrical_kw

        # Register heat rejection with CoolingManager (centralized utility mode)
        if self.cooling_manager:
            # Estimate CW flow needed for 10K delta T
            cw_flow_kg_s = batch_heat_rejected_kw / (4.18 * 10.0) if batch_heat_rejected_kw > 0 else 0.0
            self.cooling_manager.register_cw_load(
                duty_kw=batch_heat_rejected_kw,
                flow_kg_s=cw_flow_kg_s
            )

        # Update public state
        self.cooling_load_kw = self.timestep_cooling_load_kw
        self.heat_rejected_kw = self.timestep_heat_rejected_kw
        self.electrical_power_kw = self.timestep_electrical_power_kw

        # Cooling water flow estimate (ΔT ≈ 10K)
        Cp_water = 4.18
        if self.heat_rejected_kw > 0:
            self.cooling_water_flow_kg_h = (
                self.heat_rejected_kw / (Cp_water * 10.0)
            ) * 3600.0
        else:
            self.cooling_water_flow_kg_h = 0.0

        # Optional dynamics update
        if self.enable_dynamics and self.pump is not None and self.coolant_thermal is not None:
            dt_seconds = self.dt * 3600.0
            temp_error = self.inlet_stream.temperature_k - self.target_temp_k
            pump_speed = np.clip(temp_error / 30.0, 0, 1) if temp_error > 0 else 0.0

            Q_cool_m3_h = self.pump.step(dt_s=dt_seconds, pump_speed_fraction=pump_speed)
            Q_absorbed_W = abs(cooling_load_kw) * 1000.0
            T_coolant_K = self.coolant_thermal.step(
                dt_s=dt_seconds,
                heat_generated_W=Q_absorbed_W,
                T_control_K=self.target_temp_k
            )

    def get_output(self, port_name: str) -> Any:
        """
        Retrieve output from specified port.

        Args:
            port_name (str): Port identifier ('fluid_out', 'heat_out',
                'cooling_water_out', or 'electricity_in').

        Returns:
            Stream or float: Output stream or thermal/electrical value.
        """
        if port_name == "fluid_out":
            return self.outlet_stream
        elif port_name == "heat_out":
            return self.heat_rejected_kw
        elif port_name == "cooling_water_out":
            if self.cooling_water_inlet:
                return Stream(
                    mass_flow_kg_h=self.cooling_water_flow_kg_h,
                    temperature_k=self.cooling_water_inlet.temperature_k + 10.0,
                    pressure_pa=self.cooling_water_inlet.pressure_pa,
                    composition=self.cooling_water_inlet.composition
                )
            else:
                return Stream(
                    mass_flow_kg_h=self.cooling_water_flow_kg_h,
                    temperature_k=308.15,
                    pressure_pa=101325.0
                )
        elif port_name == "electricity_in":
            return self.electrical_power_kw
        
        raise ValueError(f"Unknown output port '{port_name}'")

    def receive_input(self, port_name: str, value: Any, resource_type: str = None) -> float:
        """
        Accept input at specified port.

        Args:
            port_name (str): Target port ('fluid_in', 'cooling_water_in', or 'electricity_in').
            value (Any): Input stream or power value.
            resource_type (str, optional): Resource classification hint.

        Returns:
            float: Amount accepted.
        """
        if port_name == "fluid_in" and isinstance(value, Stream):
            if value.mass_flow_kg_h > 0:
                self._input_buffer.append(value)
            return value.mass_flow_kg_h
        elif port_name == "cooling_water_in" and isinstance(value, Stream):
            self.cooling_water_inlet = value
            return value.mass_flow_kg_h
        elif port_name == "electricity_in":
            return self.electrical_power_kw
        return 0.0

    def extract_output(self, port_name: str, amount: float, resource_type: str = None) -> None:
        """
        Acknowledge extraction of output (no-op for pass-through component).

        Args:
            port_name (str): Output port.
            amount (float): Amount extracted.
            resource_type (str, optional): Resource classification hint.
        """
        pass

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """
        Define the physical connection ports for this component.

        Returns:
            Dict[str, Dict[str, str]]: Port definitions.
        """
        return {
            'fluid_in': {'type': 'input', 'resource_type': 'stream'},
            'cooling_water_in': {'type': 'input', 'resource_type': 'water'},
            'electricity_in': {'type': 'input', 'resource_type': 'electricity'},
            'fluid_out': {'type': 'output', 'resource_type': 'stream'},
            'cooling_water_out': {'type': 'output', 'resource_type': 'water'},
            'heat_out': {'type': 'output', 'resource_type': 'heat'}
        }

    def get_state(self) -> Dict[str, Any]:
        """
        Retrieve the component's current operational state.

        Fulfills the Component Lifecycle Contract state access.

        Returns:
            Dict[str, Any]: State dictionary containing:
                - cooling_load_kw (float): Current cooling load (kW).
                - outlet_temp_k (float): Outlet temperature (K).
                - outlet_pressure_bar (float): Outlet pressure (bar).
                - heat_rejected_kw (float): Heat to cooling water (kW).
                - electrical_power_kw (float): COP-based power (kW).
                - cooling_water_flow_kg_h (float): Cooling water rate (kg/h).
        """
        return {
            **super().get_state(),
            'component_id': self.component_id,
            'cooling_load_kw': self.cooling_load_kw,
            'sensible_heat_kw': getattr(self, 'sensible_heat_kw', 0.0), # Safer access
            'latent_heat_kw': getattr(self, 'latent_heat_kw', 0.0),
            'outlet_temp_k': self.outlet_stream.temperature_k,
            'outlet_pressure_bar': self.outlet_stream.pressure_pa / 1e5,
            'heat_rejected_kw': self.heat_rejected_kw,
            'electrical_power_kw': self.electrical_power_kw,
            'timestep_energy_kwh': self.electrical_power_kw * self.dt,
            'cooling_water_flow_kg_h': self.cooling_water_flow_kg_h,
            'cop': self.cop,
            'pressure_drop_bar': self.pressure_drop_bar,
            'outlet_o2_ppm_mol': (self.outlet_stream.get_total_mole_frac('O2') * 1e6) if hasattr(self, 'outlet_stream') and self.outlet_stream else 0.0
        }
