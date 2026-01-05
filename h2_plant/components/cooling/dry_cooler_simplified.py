"""
Dry Cooler Component (Simplified).

Implements an air-cooled heat exchanger for process gas cooling.
Replaces legacy 'modelo_dry_cooler.py' with rigorous Layer 1 architecture.

Key Physics:
    - Isobaric cooling (with minor pressure drop).
    - Rigorous Condensation check using Numba-accelerated Rachford-Rice flash.
    - Energy Balance: Q = m_dot * (h_in - h_out).
    - Phase Separation: Tracks liquid water fallout for downstream knockout.

Optimization:
    - Uses LUTManager for all property lookups (No CoolProp in loop).
    - Uses JIT-compiled flash solvers for condensation logic.
"""

import logging
from typing import Dict, Any, Optional
import numpy as np

from h2_plant.core.component import Component
from h2_plant.core.stream import Stream

# Import Numba-optimized solvers
try:
    from h2_plant.optimization.numba_ops import (
        solve_rachford_rice_single_condensable,
        calculate_water_psat_jit
    )
    JIT_AVAILABLE = True
except ImportError:
    JIT_AVAILABLE = False
    logging.warning("Numba ops not found. DryCoolerSimplified will run in fallback mode.")

# Import mixture thermodynamics for rigorous enthalpy calculations
try:
    from h2_plant.optimization import mixture_thermodynamics as mix_thermo
    MIX_THERMO_AVAILABLE = True
except ImportError:
    mix_thermo = None
    MIX_THERMO_AVAILABLE = False

logger = logging.getLogger(__name__)


class DryCoolerSimplified(Component):
    """
    Simplified Dry Cooler / Aftercooler.
    
    Cools a process stream to a target temperature, calculating heat duty
    and liquid water condensation. Mass is conserved (liquid remains entrained).
    
    Physics Model:
        - Q = m_dot * (h_in - h_out) [Enthalpy-based heat duty]
        - Flash calculation via Rachford-Rice for water condensation
        - P_out = P_in - ΔP [Isobaric with pressure drop]
    
    Attributes:
        target_temp_k (float): Setpoint temperature (e.g., 313.15 K = 40°C).
        pressure_drop_bar (float): Design pressure drop (default 0.05 bar).
        fan_specific_power_kw_per_mw (float): Fan power per MW rejected (~1.5%).
    """

    def __init__(
        self,
        component_id: str,
        target_temp_k: float = 313.15,
        pressure_drop_bar: float = 0.05,
        fan_specific_power_kw_per_mw: float = 15.0
    ):
        """
        Initialize the DryCoolerSimplified component.
        
        Args:
            component_id: Unique identifier for registry lookup.
            target_temp_k: Target outlet temperature (K). Default: 313.15 K (40°C).
            pressure_drop_bar: Pressure drop across cooler (bar). Default: 0.05.
            fan_specific_power_kw_per_mw: Fan power consumption per MW of heat 
                rejected (kW/MW). Default: 15.0 (~1.5% of duty).
        """
        super().__init__()
        self.component_id = component_id
        self.target_temp_k = target_temp_k
        self.pressure_drop_bar = pressure_drop_bar
        self.fan_spec_power = fan_specific_power_kw_per_mw
        
        # Dependencies
        self.lut_manager = None
        
        # Internal State
        self.inlet_buffer: Optional[Stream] = None
        self.outlet_stream: Optional[Stream] = None
        self.heat_rejected_kw = 0.0
        self.fan_power_kw = 0.0
        self.condensed_water_kg_h = 0.0
        self.vapor_fraction = 1.0

    def initialize(self, dt: float, registry: Any) -> None:
        """
        Initialize and link to High-Performance Thermodynamics backend.
        
        Args:
            dt: Simulation timestep (hours).
            registry: ComponentRegistry for accessing LUTManager.
        """
        super().initialize(dt, registry)
        
        if registry and registry.has('lut_manager'):
            self.lut_manager = registry.get('lut_manager')
        else:
            logger.warning(f"DryCoolerSimplified {self.component_id}: LUTManager not available")

    def step(self, t: float) -> None:
        """
        Execute cooling physics with rigorous condensation check.
        
        Physics:
            1. Calculate outlet P and T
            2. Flash calculation for condensation (Rachford-Rice)
            3. Calculate condensed water mass (mass preserved in stream)
            4. Calculate heat duty via enthalpy balance
            5. Compute fan power
        
        Args:
            t: Current simulation time (hours).
        """
        super().step(t)
        
        # 1. Idle Check
        if not self.inlet_buffer or self.inlet_buffer.mass_flow_kg_h <= 1e-6:
            self._set_idle_state()
            return

        s_in = self.inlet_buffer
        
        # 2. Determine Outlet Conditions
        P_out_pa = max(101325.0, s_in.pressure_pa - (self.pressure_drop_bar * 1e5))
        T_out_k = self.target_temp_k
        
        # Cannot cool below inlet (it's a cooler, not a heater)
        if T_out_k > s_in.temperature_k:
            T_out_k = s_in.temperature_k
        
        # 3. Flash Calculation (Check for Condensation)
        y_h2o = s_in.get_mole_frac('H2O')
        beta = 1.0  # Vapor fraction default
        
        if y_h2o > 0 and JIT_AVAILABLE:
            # Calculate Saturation Pressure of Water at T_out
            P_sat_water = calculate_water_psat_jit(T_out_k)
            
            # K-value = P_sat / P_sys
            K_val = P_sat_water / P_out_pa
            
            # Solve Flash - returns beta (vapor fraction of total feed)
            beta = solve_rachford_rice_single_condensable(y_h2o, K_val)
        
        self.vapor_fraction = beta
        
        # 4. Mass Balance (Condensation tracking)
        m_total = s_in.mass_flow_kg_h
        
        if beta < 0.9999:
            # Get mixture molar mass for conversion
            _, _, M_mix, _ = s_in.get_composition_arrays()
            
            # Calculate condensed water
            # Beta is global molar vapor fraction
            # Liquid moles = Total moles * (1 - beta)
            total_moles_approx = m_total / M_mix
            liq_moles = total_moles_approx * (1.0 - beta)
            condensed_rate = liq_moles * 18.015  # kg/h
            
            # Clamp to available water
            m_h2o_in = s_in.composition.get('H2O', 0.0) * m_total
            condensed_rate = min(condensed_rate, m_h2o_in)
            self.condensed_water_kg_h = condensed_rate
        else:
            self.condensed_water_kg_h = 0.0

        # 5. Calculate Heat Duty (Enthalpy Method)
        h_in = s_in.specific_enthalpy_j_kg
        
        # Get outlet gas enthalpy
        h_out_gas = 0.0
        if MIX_THERMO_AVAILABLE and self.lut_manager:
            h_out_gas = mix_thermo.get_mixture_enthalpy(
                s_in.composition, P_out_pa, T_out_k, self.lut_manager
            )
        else:
            # Fallback: Simple Cp-based estimate
            h_out_gas = h_in - self._estimate_dh_fallback(s_in, T_out_k)
        
        # Correction for latent heat if condensing
        h_out_real = h_out_gas
        if self.condensed_water_kg_h > 0:
            # Latent heat of vaporization approx 2442 kJ/kg at 25°C
            h_fg = 2442300.0  # J/kg
            w_liq = self.condensed_water_kg_h / m_total
            h_out_real -= w_liq * h_fg

        # Q = m * (h_in - h_out)
        q_j_kg = h_in - h_out_real
        self.heat_rejected_kw = (m_total * q_j_kg) / 3600.0 / 1000.0  # J/h -> kW
        
        # Ensure positive heat rejection (cooling removes heat)
        if self.heat_rejected_kw < 0:
            self.heat_rejected_kw = 0.0
        
        # 6. Fan Power
        # P_fan = Specific_Power * Duty (in MW)
        self.fan_power_kw = abs(self.heat_rejected_kw / 1000.0) * self.fan_spec_power

        # 7. Create Output Stream (Mass preserved, liquid entrained)
        self.outlet_stream = Stream(
            mass_flow_kg_h=m_total,
            temperature_k=T_out_k,
            pressure_pa=P_out_pa,
            composition=s_in.composition,  # Composition preserved
            phase='mixed' if self.condensed_water_kg_h > 0 else 'gas',
            extra={
                'm_dot_H2O_liq_accomp_kg_s': self.condensed_water_kg_h / 3600.0,
                'vapor_fraction': self.vapor_fraction
            }
        )
        
        # Clear input buffer
        self.inlet_buffer = None

    def _estimate_dh_fallback(self, stream: Stream, T_out: float) -> float:
        """
        Fallback enthalpy change estimate using gas-specific Cp.
        
        Returns:
            float: Enthalpy change (J/kg) for cooling from stream.T to T_out.
        """
        dT = stream.temperature_k - T_out
        
        # Select Cp based on dominant species
        h2_frac = stream.composition.get('H2', 0.0)
        o2_frac = stream.composition.get('O2', 0.0)
        
        if h2_frac > 0.5:
            cp = 14300.0  # J/kg/K for H2
        elif o2_frac > 0.5:
            cp = 918.0    # J/kg/K for O2
        else:
            cp = 1000.0   # Generic gas
            
        return cp * dT

    def _set_idle_state(self):
        """Reset outputs to idle/zero state."""
        self.heat_rejected_kw = 0.0
        self.fan_power_kw = 0.0
        self.condensed_water_kg_h = 0.0
        self.vapor_fraction = 1.0
        self.outlet_stream = Stream(0.0)

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """
        Define the physical connection ports for this component.
        
        Returns:
            Port definitions for inlet and outlet streams.
        """
        return {
            'inlet': {'type': 'input', 'resource_type': 'stream'},
            'outlet': {'type': 'output', 'resource_type': 'stream'},
            'electricity_in': {'type': 'input', 'resource_type': 'electricity'}
        }

    def get_state(self) -> Dict[str, Any]:
        """
        Retrieve the component's current operational state.
        
        Returns:
            State dictionary with thermal and operational parameters.
        """
        return {
            **super().get_state(),
            'heat_rejected_kw': self.heat_rejected_kw,
            'fan_power_kw': self.fan_power_kw,
            'condensed_water_kg_h': self.condensed_water_kg_h,
            'vapor_fraction': self.vapor_fraction,
            'outlet_temp_k': self.target_temp_k,
            'outlet_temp_c': self.target_temp_k - 273.15,
            'pressure_drop_bar': self.pressure_drop_bar
        }

    def receive_input(self, port_name: str, value: Any, resource_type: str = None) -> float:
        """
        Accept input at specified port.
        
        Args:
            port_name: Target port ('inlet').
            value: Input stream.
            resource_type: Resource classification hint.
        
        Returns:
            Amount accepted (mass flow kg/h).
        """
        if port_name == 'inlet' and isinstance(value, Stream):
            self.inlet_buffer = value
            return value.mass_flow_kg_h
        return 0.0
        
    def get_output(self, port_name: str) -> Optional[Stream]:
        """
        Retrieve output from specified port.
        
        Args:
            port_name: Output port ('outlet').
        
        Returns:
            Cooled process stream.
        """
        if port_name == 'outlet':
            return self.outlet_stream
        elif port_name == 'electricity_in':
            return self.fan_power_kw
        return None
        
    def extract_output(self, port_name: str, amount: float, resource_type: str = None) -> None:
        """
        Acknowledge extraction of output (no-op for pass-through component).
        
        Args:
            port_name: Output port.
            amount: Amount extracted.
            resource_type: Resource classification hint.
        """
        if port_name == 'outlet':
            self.outlet_stream = None

    @property
    def power_kw(self) -> float:
        """Expose power consumption in kW for dispatch tracking."""
        return self.fan_power_kw
