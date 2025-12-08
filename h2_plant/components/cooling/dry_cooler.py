"""
Dry Cooler (Air-Cooled Heat Exchanger) Component.

Implements rigorous NTU-Effectiveness thermal modeling for cooling
hydrogen and oxygen streams from PEM electrolyzers.
Ref: dry_cooler-1.pdf, drydim.py
"""

import logging
from typing import Dict, Any, Optional
import numpy as np

from h2_plant.core.component import Component
from h2_plant.core.stream import Stream
from h2_plant.core.constants import (
    GasConstants,
    ConversionFactors,
    DryCoolerConstants as DCC
)
from h2_plant.optimization import numba_ops

logger = logging.getLogger(__name__)

class DryCooler(Component):
    """
    Air-cooled heat exchanger for process cooling.
    
    Automatically adapts strict geometry/performance parameters based on
    fluid type (H2 or O2) detected at inlet.
    
    Physics:
    - Crossflow/Counterflow Heat Exchanger (Unmixed-Mixed)
    - NTU-Effectiveness Method
    - Variable heat capacity based on phase (Gas + Liquid Water)
    """
    
    def __init__(self, component_id: str = "dry_cooler"):
        super().__init__()
        self.component_id = component_id
        
        # State
        self.fluid_type = "Unknown" 
        self.inlet_stream: Optional[Stream] = None
        self.outlet_stream: Optional[Stream] = None
        
        # Results
        self.heat_duty_kw = 0.0
        self.fan_power_kw = 0.0
        self.outlet_temp_c = 0.0
        self.effectiveness = 0.0
        self.ntu = 0.0
        self.air_mass_flow_kg_s = 0.0
        
        # Constants loaded dynamically based on fluid
        self.area_m2 = 0.0
        self.design_air_flow_kg_s = 0.0
        
    def initialize(self, dt: float, registry: Any) -> None:
        """Initialize component."""
        super().initialize(dt, registry)

    def _detect_fluid_config(self, stream: Stream) -> None:
        """Configure exchanger geometry based on dominant species."""
        h2_frac = stream.composition.get('H2', 0.0)
        o2_frac = stream.composition.get('O2', 0.0)
        
        if h2_frac > o2_frac:
            self.fluid_type = "H2"
            self.area_m2 = DCC.AREA_H2_M2
            self.design_air_flow_kg_s = DCC.MDOT_AIR_DESIGN_H2_KG_S
        else:
            self.fluid_type = "O2"
            self.area_m2 = DCC.AREA_O2_M2
            self.design_air_flow_kg_s = DCC.MDOT_AIR_DESIGN_O2_KG_S

    def receive_input(self, port_name: str, value: Any, resource_type: str = None) -> float:
        if port_name == "fluid_in" and isinstance(value, Stream):
            self.inlet_stream = value
            # Auto-configure on first valid input or if type changes
            self._detect_fluid_config(value)
            return value.mass_flow_kg_h
        elif port_name == "electricity_in":
            return self.fan_power_kw
        return 0.0

    def get_output(self, port_name: str) -> Any:
        if port_name == "fluid_out":
            return self.outlet_stream
        return None

    def step(self, t: float) -> None:
        super().step(t)
        
        if not self.inlet_stream or self.inlet_stream.mass_flow_kg_h <= 0:
            self.outlet_stream = None
            self.heat_duty_kw = 0.0
            self.fan_power_kw = 0.0
            return

        # 1. Process Conditions
        m_dot_total = self.inlet_stream.mass_flow_kg_h / 3600.0 # kg/s
        T_h_in = self.inlet_stream.temperature_k
        P_in = self.inlet_stream.pressure_pa
        
        # Liquid fraction handling (simplified assumption: H2O is liquid if explicitly stated or T < 100C @ 1atm check)
        # Using composition for mass balance. In rigorous mode, stream.phase might be 'mixed'
        # Approximate: separate water flow logic if needed, but for now assuming 'wet gas' properties
        
        # Mass fractions calculation (User Requirement: Convert mole fractions to mass fractions)
        # 1. Calculate average molecular weight
        mw_mix = 0.0
        for sp, mole_frac in self.inlet_stream.composition.items():
            if sp in GasConstants.SPECIES_DATA:
                mw_sp = GasConstants.SPECIES_DATA[sp]['molecular_weight']
                mw_mix += mole_frac * mw_sp
        
        # Avoid division by zero
        if mw_mix < 1e-12: mw_mix = 18.015 # Fallback to water
        
        # 2. Calculate mass fractions for specific heat weighting
        w_h2o = 0.0
        w_gas = 0.0
        
        # We need to sum Cp contribution from all species
        # But for this specific model, we are blending a "Gas" Cp and a "Liquid" Cp
        # "Gas" here is the non-condensable part (H2 or O2), "Liquid" is H2O (simplified)
        
        Cp_weighted = 0.0
        
        for sp, mole_frac in self.inlet_stream.composition.items():
            if sp in GasConstants.SPECIES_DATA:
                mw_sp = GasConstants.SPECIES_DATA[sp]['molecular_weight']
                mass_frac = (mole_frac * mw_sp) / mw_mix
                
                # Determine Cp for this species
                if sp == 'H2O':
                     # Simplification: Treat H2O as liquid for conservative cooling?
                     # Or use vapor? Reference says "Liq + Vap".
                     # For NTU "C_hot" capacity rate, using liquid Cp for water portion
                     # approximates the high thermal inertia of the wet stream.
                     cp_sp = 4186.0
                elif sp == 'H2':
                     cp_sp = GasConstants.CP_H2_AVG
                elif sp == 'O2':
                     cp_sp = GasConstants.CP_O2_AVG
                else:
                     cp_sp = 1000.0 # Generic fallback
                
                Cp_weighted += mass_frac * cp_sp
        
        # C_hot = m_dot * Cp_mix
        C_hot = m_dot_total * Cp_weighted
        
        # 2. Air Conditions (Cold Side)
        # Fixed design air flow (Fan runs at constant speed in this model)
        m_dot_air = self.design_air_flow_kg_s
        C_air = m_dot_air * DCC.CP_AIR_J_KG_K
        T_c_in = DCC.T_A_IN_DESIGN_C + 273.15 # 20C (293.15K)
        
        # 3. NTU Parameters
        C_min = min(C_hot, C_air)
        C_max = max(C_hot, C_air)
        R_capacity = C_min / C_max
        
        area_eff = self.area_m2 # Geometry
        
        NTU = (DCC.U_W_M2_K * area_eff) / C_min
        
        # 4. Effectiveness (JIT)
        eff = numba_ops.dry_cooler_ntu_effectiveness(NTU, R_capacity)
        self.effectiveness = eff
        self.ntu = NTU
        
        # 5. Heat Duty
        Q_max = C_min * (T_h_in - T_c_in)
        Q_actual = eff * Q_max
        
        self.heat_duty_kw = Q_actual / 1000.0
        
        # 6. Outlet Conditions
        # Q = C_hot * (T_h_in - T_h_out) -> T_h_out = T_h_in - Q/C_hot
        if C_hot > 1e-9:
            T_h_out = T_h_in - (Q_actual / C_hot)
        else:
            T_h_out = T_h_in
        
        # Pressure Drop
        P_out = P_in - (DCC.DP_FLUID_BAR * 1e5)
        if P_out < 101325: P_out = 101325 # Clamp to ambient minimum
        
        self.outlet_temp_c = T_h_out - 273.15
        
        self.outlet_stream = Stream(
            mass_flow_kg_h = self.inlet_stream.mass_flow_kg_h,
            temperature_k = T_h_out,
            pressure_pa = P_out,
            composition = self.inlet_stream.composition,
            phase = 'mixed' # Cooling likely results in two-phase
        )
        
        # 7. Fan Power
        # Power = (V_dot * dP) / eta
        # V_dot = m_dot / rho
        vol_air = m_dot_air / DCC.RHO_AIR_KG_M3
        power_j_s = (vol_air * DCC.DP_AIR_PA) / DCC.ETA_FAN
        self.fan_power_kw = power_j_s / 1000.0
        self.air_mass_flow_kg_s = m_dot_air

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        return {
            'fluid_in': {'type': 'input', 'resource_type': 'stream'},
            'electricity_in': {'type': 'input', 'resource_type': 'electricity'},
            'fluid_out': {'type': 'output', 'resource_type': 'stream'}
        }

    def get_state(self) -> Dict[str, Any]:
        return {
            **super().get_state(),
            'fluid_type': self.fluid_type,
            'heat_duty_kw': self.heat_duty_kw,
            'fan_power_kw': self.fan_power_kw,
            'outlet_temp_c': self.outlet_temp_c,
            'effectiveness': self.effectiveness,
            'ntu': self.ntu,
            'air_flow_kg_s': self.air_mass_flow_kg_s
        }
