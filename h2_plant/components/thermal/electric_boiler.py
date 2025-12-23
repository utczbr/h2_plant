"""
Continuous Flow Electric Boiler Component.

This component simulates an isobaric heater transforming electrical energy into
thermal enthalpy of a fluid stream.

Thermodynamic Principle:
    First Law (Open System): dH/dt = ṁ_in * h_in - ṁ_out * h_out + Q_net
    Steady State approximation: h_out = h_in + (P_elec * η) / ṁ

Operational Modes:
    1. **Two-Phase (Water/Steam)**: Performs flash calculations using the LUTManager
       to determine Vapor Quality (x) and Phase (Subcooled/Mixed/Superheated).
    2. **Single-Phase (Gas)**: Uses specific heat capacity (Cp) integration for
       simple heating of gases (e.g., H2, O2).
"""

import math
import numpy as np
from typing import Dict, Any, Optional, TYPE_CHECKING

from h2_plant.core.component import Component
from h2_plant.core.stream import Stream
from h2_plant.optimization.numba_ops import calc_boiler_outlet_enthalpy

if TYPE_CHECKING:
    from h2_plant.core.component_registry import ComponentRegistry
    from h2_plant.optimization.lut_manager import LUTManager


class ElectricBoiler(Component):
    """
    Simulates a continuous flow electric resistance heater.

    This component serves as a primary thermal source, capable of generating
    steam for electrolysis or process heating.

    Architecture:
        - Acts as a **Transformer** node in the plant graph.
        - Interfaces with **LUTManager** for high-fidelity property lookups.

    Attributes:
        max_power_w (float): Rated electrical capacity (W).
        efficiency (float): Thermal conversion efficiency (0-1).
        design_pressure_pa (float): Structural pressure limit (Pa).
        current_power_w (float): Power consumption in current timestep (W).
        outlet_temp_k (float): Fluid outlet temperature (K).
        vapor_fraction (float): Vapor quality (0.0-1.0) for two-phase fluids.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs) -> None:
        """
        Initialize boiler configuration.
        
        Args:
            config (dict): Configuration containing:
                - max_power_kw (float): Maximum heating power in kW (default: 1000)
                - efficiency (float): Thermal efficiency 0.0-1.0 (default: 0.99)
                - target_temp_c (float): Optional target outlet temperature (thermostat mode)
                - design_pressure_bar (float): Operational pressure limit in bar (default: 10)
        """
        super().__init__(config, **kwargs)
        
        config = config or {}
        
        # Physics Parameters (SI units)
        self.max_power_w = config.get('max_power_kw', 1000.0) * 1000.0
        self.efficiency = config.get('efficiency', 0.99)
        self.design_pressure_pa = config.get('design_pressure_bar', 10.0) * 1e5
        
        # Pressure Drop Model (optional)
        # k_friction: Pressure drop coefficient (Pa/(kg/h)^2)
        # At 100 kg/h flow, k_friction=0.5 gives 5000 Pa (0.05 bar) drop
        self.k_friction = config.get('k_friction', 0.0)  # Default: isobaric (no drop)
        
        # Thermostat mode
        self.target_temp_k = None
        if 'target_temp_c' in config:
            self.target_temp_k = config['target_temp_c'] + 273.15
        
        # State tracking
        self.current_power_w = 0.0
        self.outlet_temp_k = 298.15
        self.vapor_fraction = 0.0
        self.total_energy_kwh = 0.0
        self.dt_hours = 0.0  # Stored in initialize()
        
        # Input buffers
        self._input_stream: Optional[Stream] = None
        self._power_setpoint_w = 0.0
        self._output_stream: Optional[Stream] = None
        
        # Dependencies
        self.lut: Optional['LUTManager'] = None
        
        # History
        self.history_temp = np.zeros(1)
        self.step_idx = 0

    def initialize(self, dt: float, registry: 'ComponentRegistry') -> None:
        """
        Executes the initialization phase of the Component Lifecycle Contract.

        Tasks:
        1. Link to shared services (LUTManager).
        2. Allocate memory for history buffers.
        3. Store simulation timestep for energy integration.

        Args:
            dt (float): Simulation timestep (hours).
            registry (ComponentRegistry): Central service registry.
        """
        super().initialize(dt, registry)
        self.dt_hours = dt
        self.lut = registry.get('lut_manager') if registry.has('lut_manager') else None
        
        steps = int(24 / dt) + 1
        self.history_temp = np.zeros(steps)
        self.step_idx = 0


    def _solve_temperature_from_enthalpy(
        self, 
        h_target: float, 
        pressure_pa: float, 
        fluid: str,
        T_guess: float,
        tol: float = 0.01,
        max_iter: int = 20
    ) -> float:
        """
        Solves the inverse state equation T = f(h, P) using Newton-Raphson.

        Since h(T, P) is non-linear (especially near phase transitions), we iterate:
        
        **T_{i+1} = T_i + (h_{target} - h(T_i)) / Cp(T_i)**

        Why Newton-Raphson?
        Direct inversion of the enthalpy LUT is computationally expensive (requires search).
        Forward iterators using local derivatives (Cp) are 10-50x faster for
        small perturbations.

        Args:
            h_target (float): Target specific enthalpy (J/kg).
            pressure_pa (float): Isobaric system pressure (Pa).
            fluid (str): Fluid identifier for LUT ('Water', 'H2', etc.).
            T_guess (float): Initial temperature estimate (K).
            tol (float): Convergence tolerance (K). Default: 0.01.
            max_iter (int): Iteration limit. Default: 20.

        Returns:
            float: Converged temperature (K).
        """
        if self.lut is None:
            # Fallback: Simple Cp approximation (when no LUT available)
            # delta_h = h_target - h(T_guess) ≈ (T_target - T_guess) * Cp
            # Since we don't know h(T_guess) without LUT, we use:
            # T_out = T_guess + delta_h / Cp where delta_h is provided via h_target relative to h_in
            # For thermostat mode, the enthalpy change was calculated, so we need to solve:
            # h_out - h_in = Cp * (T_out - T_in), hence T_out = T_in + (h_out - h_in) / Cp
            # But here h_target is h_out, and we don't have h_in directly.
            # The callers pass T_guess = T_in, and h_target = h_out, so:
            # We approximate: T_out ≈ T_guess since power was already applied correctly
            # Actually, just return T_guess as best estimate
            cp_default = {'Water': 4180.0, 'H2': 14304.0, 'O2': 918.0}.get(fluid, 4180.0)
            # The enthalpy balance was already computed, estimate T from h
            # h = h_ref + Cp * (T - T_ref), so T = T_ref + (h - h_ref) / Cp
            # Use T_ref = 298.15 K, h_ref = 0 (approximation)
            T_ref = 298.15
            T_approx = T_ref + h_target / cp_default
            return max(T_approx, 200.0)  # Clamp to physical minimum
        
        T = T_guess
        
        for i in range(max_iter):
            try:
                # Get h(T) and Cp(T) from LUT
                h_current = self.lut.lookup(fluid, 'H', pressure_pa, T)
                cp_current = self.lut.lookup(fluid, 'C', pressure_pa, T)
                
                
                # Guard against zero/negative/NaN Cp
                if math.isnan(cp_current) or cp_current < 100.0:
                    cp_current = 4180.0 if fluid == 'Water' else 14304.0
                
                # Newton-Raphson step: T_new = T + (h_target - h) / Cp
                residual = h_target - h_current
                if math.isnan(residual):
                     # If enthalpy lookup failed (NaN), break loop and return guess
                     break

                T_new = T + residual / cp_current
                
                # Clamp to physical bounds (avoid negative temperatures)
                T_new = max(T_new, 273.15)  # 273.15 K minimum (LUT bound)
                T_new = min(T_new, 1500.0)  # 1500 K maximum
                
                # Check convergence
                if abs(T_new - T) < tol:
                    return T_new
                
                T = T_new
                
            except Exception:
                # LUT lookup failed - return current best guess
                break
        
        # Return best estimate after max iterations
        return T

    def step(self, t: float) -> None:
        """
        Executes the physics simulation step.

        Process Logic:
        1. **Demand Calculation**: Determines required power for thermostat control (if active).
        2. **Constraint Application**: Clips demand to `max_power_w`.
        3. **Physics Execution**:
           - Updates Pressure (Draft/Friction model).
           - Calculates Outlet Enthalpy (1st Law).
           - Solves Phase/Temperature (Flash Calcs).
        4. **State Update**: Updates output ports and energy accumulators.

        Args:
            t (float): Current simulation time (hours).
        """
        super().step(t)

        inflow = self._input_stream
        
        # 1. Determine Power Demand
        power_demand_w = self._power_setpoint_w
        
        # Thermostat Logic
        if power_demand_w <= 1e-6 and self.target_temp_k is not None and inflow is not None:
            T_in = inflow.temperature_k
            m_dot_kg_s = inflow.mass_flow_kg_h / 3600.0
            
            if m_dot_kg_s > 0:
                # Robust Enthalpy-Based Calculation (Handles Phase Change)
                if self.lut:
                    try:
                        # 1. Look up target enthalpy at (P, T_target)
                        # Note: If T_target is exactly saturation, CoolProp might return liquid or vapor.
                        # For a boiler, we usually want Vapor if T >= Tsat.
                        # However, LUT lookup('H', P, T) returns stable phase.
                        
                        # Use target pressure (current operating pressure)
                        P_op = inflow.pressure_pa
                        
                        # Check Saturation first to determine desired phase if close to Tsat
                        sat_props = self.lut.get_saturation_properties(P_op)
                        T_sat = sat_props['T_sat_K']
                        
                        if self.target_temp_k > T_sat - 0.1:
                             # Target is Steam (or Superheated)
                             # If T_target ~= T_sat, we aim for Saturated Vapor enthalpy to ensure boiling
                             if abs(self.target_temp_k - T_sat) < 1.0:
                                 h_target = sat_props['h_g_Jkg']
                             else:
                                 # Superheated
                                 h_target = self.lut.lookup('Water', 'H', P_op, self.target_temp_k)
                        else:
                             # Target is Liquid
                             h_target = self.lut.lookup('Water', 'H', P_op, self.target_temp_k)
                        
                        h_in = inflow.specific_enthalpy_j_kg
                        Q_needed_w = m_dot_kg_s * (h_target - h_in)
                        
                        # Handle case where Q < 0 (cooling needed?) - Boiler only heats
                        Q_needed_w = max(0.0, Q_needed_w)
                        power_demand_w = Q_needed_w / self.efficiency
                        
                    except Exception:
                        # Fallback to Cp logic if LUT fails
                        # Estimate Cp based on fluid type
                        cp_avg = 4180.0  # Default: Water
                        if inflow.composition.get('H2', 0) > 0.5:
                            cp_avg = 14304.0  # Hydrogen
                        elif inflow.composition.get('O2', 0) > 0.5:
                            cp_avg = 918.0  # Oxygen
                        
                        dT = self.target_temp_k - T_in
                        if dT > 0:
                            Q_needed_w = m_dot_kg_s * cp_avg * dT
                            power_demand_w = Q_needed_w / self.efficiency
                        else:
                            power_demand_w = 0.0
                            
                else:
                    # Fallback to Cp logic if no LUT
                    cp_avg = 4180.0
                    dT = self.target_temp_k - T_in
                    if dT > 0:
                         Q_needed_w = m_dot_kg_s * cp_avg * dT
                         power_demand_w = Q_needed_w / self.efficiency
            else:
                 power_demand_w = 0.0

        # 2. Apply Power Limits
        applied_power_w = min(power_demand_w, self.max_power_w)
        
        # 3. Process Flow & Physics
        if inflow is None or inflow.mass_flow_kg_h <= 0:
            outflow = Stream(
                mass_flow_kg_h=0.0,
                temperature_k=298.15,
                pressure_pa=101325,
                composition={'H2O': 1.0},
                phase='liquid'
            )
            self.current_power_w = 0.0
            self.vapor_fraction = 0.0
        else:
            h_in = inflow.specific_enthalpy_j_kg
            
            # Pressure Drop Model (optional)
            # ΔP = k_friction * ṁ² (simplified Darcy-Weisbach)
            if self.k_friction > 0 and inflow.mass_flow_kg_h > 0:
                pressure_drop_pa = self.k_friction * (inflow.mass_flow_kg_h ** 2)
                current_pressure = max(inflow.pressure_pa - pressure_drop_pa, 1e5)  # Floor at 1 bar
            else:
                current_pressure = inflow.pressure_pa
            
            # Calculate Outlet Enthalpy (Conservation of Energy)
            h_out = calc_boiler_outlet_enthalpy(
                h_in_j_kg=h_in,
                mass_flow_kg_h=inflow.mass_flow_kg_h,
                power_input_w=applied_power_w,
                efficiency=self.efficiency
            )



            # --- DUAL MODE PHYSICS LOGIC ---
            is_water_system = (inflow.composition.get('H2O', 0) > 0.5 or 
                               inflow.composition.get('Water', 0) > 0.5)

            if is_water_system:
                # === MODE A: WATER/STEAM FLASH ===
                
                # Get Saturation Properties via public LUT API
                if self.lut:
                    sat_props = self.lut.get_saturation_properties(current_pressure)
                    t_sat = sat_props['T_sat_K']
                    h_sat_liq = sat_props['h_f_Jkg']
                    h_sat_vap = sat_props['h_g_Jkg']
                else:
                    # Fallback constants (1 atm)
                    t_sat = 373.15
                    h_sat_liq = 419000.0
                    h_sat_vap = 2676000.0

                # Flash Calculation
                if h_out < h_sat_liq:
                    # Subcooled Liquid - Use Newton-Raphson solver
                    self.vapor_fraction = 0.0
                    phase = 'liquid'
                    # Use t_sat as initial guess, solve for exact T
                    T_out_k = self._solve_temperature_from_enthalpy(
                        h_target=h_out,
                        pressure_pa=current_pressure,
                        fluid='Water',
                        T_guess=t_sat - 10.0  # Start below saturation
                    )
                    
                elif h_out > h_sat_vap:
                    # Superheated Vapor - Use Newton-Raphson solver
                    self.vapor_fraction = 1.0
                    phase = 'gas'
                    # Use t_sat as initial guess, solve for exact T
                    T_out_k = self._solve_temperature_from_enthalpy(
                        h_target=h_out,
                        pressure_pa=current_pressure,
                        fluid='Water',
                        T_guess=t_sat + 10.0  # Start above saturation
                    )
                    
                else:
                    # Saturated Mixture (Boiling)
                    phase = 'mixed'
                    T_out_k = t_sat
                    denom = (h_sat_vap - h_sat_liq)
                    self.vapor_fraction = (h_out - h_sat_liq) / denom if denom > 1e-6 else 0.0

            else:
                # === MODE B: GENERIC GAS/LIQUID HEATING ===
                # Used for H2 gas, O2 gas, thermal oil, etc.
                self.vapor_fraction = 0.0  # Not applicable
                phase = inflow.phase
                
                # Determine fluid name for LUT
                if inflow.composition.get('H2', 0) > 0.5:
                    fluid_name = 'H2'
                elif inflow.composition.get('O2', 0) > 0.5:
                    fluid_name = 'O2'
                else:
                    fluid_name = 'Water'  # Default
                
                # ROBUST HEATING: delta_H = Q / m_dot
                # Use mean Cp to find delta_T.
                # T_out = T_in + (h_out - h_in) / Cp_avg
                
                delta_h = h_out - h_in  # This is exactly Q_net / m_dot
                
                # Estimate Cp at inlet conditions
                t_in = inflow.temperature_k
                if self.lut:
                    try:
                        cp_in = self.lut.lookup(fluid_name, 'C', current_pressure, t_in)
                        if math.isnan(cp_in) or cp_in < 1.0:
                             cp_in = 14304.0 if fluid_name == 'H2' else 1000.0
                    except:
                        cp_in = 14304.0 if fluid_name == 'H2' else 1000.0
                else:
                    cp_in = 14304.0 if fluid_name == 'H2' else 1000.0
                
                # Simple step: T = T_in + dH/Cp
                # For large dH, iterate once to update Cp
                t_est = t_in + delta_h / cp_in
                
                # update Cp at t_average
                t_avg = 0.5 * (t_in + t_est)
                if self.lut:
                    try:
                        cp_avg = self.lut.lookup(fluid_name, 'C', current_pressure, t_avg)
                        if math.isnan(cp_avg) or cp_avg < 1.0:
                             cp_avg = cp_in
                    except:
                         cp_avg = cp_in
                else:
                    cp_avg = cp_in
                    
                if cp_avg > 1.0:
                    T_out_k = t_in + delta_h / cp_avg
                else:
                    T_out_k = t_est # Fallback
                
                # Clamp results
                T_out_k = max(T_out_k, 273.15)


            # Construct Output Stream
            outflow = Stream(
                mass_flow_kg_h=inflow.mass_flow_kg_h,
                pressure_pa=current_pressure,
                temperature_k=T_out_k,
                composition=inflow.composition,
                phase=phase,
                extra=inflow.extra.copy() if inflow.extra else {}
            )
            
            self.current_power_w = applied_power_w
            self.outlet_temp_k = outflow.temperature_k

        self._output_stream = outflow
        
        # 4. Energy Accumulation
        if self.dt_hours > 0:
            self.total_energy_kwh += (self.current_power_w / 1000.0) * self.dt_hours
        
        # History tracking with Auto-Resize for year-long simulations
        if self.step_idx >= len(self.history_temp):
            # Buffer full: Extend by 24 hours (doubling strategy prevents reallocation churn)
            extension_size = max(int(24 / self.dt_hours) if self.dt_hours > 0 else 1440, len(self.history_temp))
            new_history = np.zeros(len(self.history_temp) + extension_size)
            new_history[:len(self.history_temp)] = self.history_temp
            self.history_temp = new_history
        
        self.history_temp[self.step_idx] = self.outlet_temp_k
        self.step_idx += 1
        
        # Clear input buffers
        self._input_stream = None
        self._power_setpoint_w = 0.0

    def receive_input(self, port_name: str, value: Any, resource_type: str = None) -> float:
        if port_name in ('fluid_in', 'water_in') and isinstance(value, Stream):
            self._input_stream = value
            return value.mass_flow_kg_h
        elif port_name == 'power_in':
            self._power_setpoint_w = float(value)
            return float(value)
        return super().receive_input(port_name, value, resource_type)

    def get_output(self, port_name: str = 'fluid_out') -> Optional[Stream]:
        """Get output without clearing buffer (peek)."""
        if port_name in ('fluid_out', 'water_out'):
            return self._output_stream
        return None

    def extract_output(self, port_name: str, amount: float = 0.0, resource_type: str = None) -> None:
        """
        Acknowledge extraction of output (no-op for pass-through component).
        
        The output stream persists until the next step() call, allowing
        get_output() to work for stream summary reporting.
        
        Args:
            port_name: Port to extract from.
            amount: Requested amount (ignored).
            resource_type: Resource type hint (ignored).
        """
        pass  # No-op: buffer cleared on next step()

    def get_state(self) -> Dict[str, Any]:
        """
        Retrieves current component state.

        Fulfills Layer 1 Contract for telemetry and GUI plotting.

        Returns:
            Dict[str, Any]: Operational metrics (Power, Temp, Phase).
        """
        return {
            **super().get_state(),
            "power_input_w": self.current_power_w,
            "power_input_kw": self.current_power_w / 1000.0,
            "outlet_temp_c": self.outlet_temp_k - 273.15,
            "outlet_temp_k": self.outlet_temp_k,
            "vapor_fraction": self.vapor_fraction,
            "phase": getattr(self._output_stream, 'phase', 'unknown') if self._output_stream else 'unknown',
            "total_energy_kwh": self.total_energy_kwh,
            "target_temp_c": (self.target_temp_k - 273.15) if self.target_temp_k else None
        }

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        return {
            'fluid_in': {'type': 'input', 'resource_type': 'stream'},
            'power_in': {'type': 'input', 'resource_type': 'electricity'},
            'fluid_out': {'type': 'output', 'resource_type': 'stream'}
        }
