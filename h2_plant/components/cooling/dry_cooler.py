"""
Dry Cooler (Air-Cooled Heat Exchanger) Component.

This module implements a rigorous NTU-effectiveness model for air-cooled heat
exchangers used to cool hydrogen and oxygen streams from PEM electrolyzers.
The model automatically adapts geometry parameters based on stream composition.

Heat Transfer Principles:
    - **NTU-Effectiveness Method**: Relates actual heat transfer to theoretical
      maximum based on the Number of Transfer Units and capacity ratio. This
      method avoids iterative LMTD calculations for complex geometries.
    - **Crossflow Configuration**: Models unmixed-mixed crossflow, typical of
      finned-tube air coolers where tube-side fluid is mixed and air-side is
      unmixed (each air streamline contacts multiple tube rows).
    - **Capacity Ratio**: R = C_min/C_max determines effectiveness behavior.
      For high R (balanced flows), effectiveness is limited by temperature
      pinch at the exchanger outlet.

Architecture:
    Implements the Component Lifecycle Contract (Layer 1):
    - `initialize()`: Prepares component for simulation (no configuration required).
    - `step()`: Computes heat duty, outlet temperature, and fan power.
    - `get_state()`: Returns thermal performance metrics for monitoring.

Design Philosophy:
    The cooler automatically detects fluid type (H₂ or O₂) from inlet composition
    and selects appropriate geometry (area, design air flow) from pre-defined
    constants. This enables a single component class to serve multiple services.

References:
    - Incropera, DeWitt (2007). Fundamentals of Heat and Mass Transfer, 6th Ed.
    - Kays, London (1984). Compact Heat Exchangers, 3rd Ed.
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
    Air-cooled heat exchanger for process stream cooling.

    Uses the NTU-effectiveness method to calculate heat duty and outlet
    temperature for hydrogen or oxygen streams. Geometry parameters are
    automatically selected based on the dominant species in the inlet stream.

    This component fulfills the Component Lifecycle Contract (Layer 1):
        - `initialize()`: Base initialization (geometry set dynamically on input).
        - `step()`: Computes NTU, effectiveness, heat duty, and outlet conditions.
        - `get_state()`: Returns thermal performance and fan power metrics.

    The NTU-effectiveness method computes:
        ε = f(NTU, C_min/C_max)  for crossflow geometry
        Q = ε × C_min × (T_hot_in - T_cold_in)
        T_hot_out = T_hot_in - Q / C_hot

    Attributes:
        fluid_type (str): Detected fluid species ('H2', 'O2', or 'Unknown').
        heat_duty_kw (float): Current heat transfer rate (kW).
        fan_power_kw (float): Fan electrical consumption (kW).
        effectiveness (float): Thermal effectiveness (0-1).
        ntu (float): Number of Transfer Units.

    Example:
        >>> cooler = DryCooler(component_id='DC_H2')
        >>> cooler.initialize(dt=1/60, registry=registry)
        >>> cooler.receive_input('fluid_in', hot_h2_stream, 'gas')
        >>> cooler.step(t=0.0)
        >>> cold_stream = cooler.get_output('fluid_out')
        >>> print(f"Outlet: {cooler.outlet_temp_c:.1f}°C, Q={cooler.heat_duty_kw:.1f} kW")
    """

    def __init__(self, component_id: str = "dry_cooler"):
        """
        Initialize the dry cooler component.

        Args:
            component_id (str): Unique identifier for this component instance.
                Default: 'dry_cooler'.
        """
        super().__init__()
        self.component_id = component_id

        # Stream state
        self.fluid_type = "Unknown"
        self.inlet_stream: Optional[Stream] = None
        self.outlet_stream: Optional[Stream] = None

        # Performance metrics
        self.heat_duty_kw = 0.0
        self.fan_power_kw = 0.0
        self.outlet_temp_c = 0.0
        self.effectiveness = 0.0
        self.ntu = 0.0
        self.air_mass_flow_kg_s = 0.0

        # Geometry (set dynamically based on fluid type)
        self.area_m2 = 0.0
        self.design_air_flow_kg_s = 0.0

    def initialize(self, dt: float, registry: Any) -> None:
        """
        Prepare the component for simulation execution.

        Fulfills the Component Lifecycle Contract initialization phase.
        Geometry parameters are set dynamically when the first valid stream
        is received, based on detected fluid composition.

        Args:
            dt (float): Simulation timestep in hours.
            registry (ComponentRegistry): Central registry for component access.
        """
        super().initialize(dt, registry)

    def _detect_fluid_config(self, stream: Stream) -> None:
        """
        Configure exchanger geometry based on dominant species.

        Detects whether the stream is primarily hydrogen or oxygen by
        comparing mass fractions, then loads appropriate geometry constants
        (heat transfer area and design air flow rate).

        Args:
            stream (Stream): Inlet stream for composition analysis.
        """
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
        """
        Accept an input stream at the specified port.

        Stores the inlet stream and triggers geometry configuration based
        on detected fluid composition.

        Args:
            port_name (str): Target port ('fluid_in' or 'electricity_in').
            value (Any): Stream object for fluid or float for power.
            resource_type (str, optional): Resource classification hint.

        Returns:
            float: Mass flow rate accepted (kg/h) or fan power (kW).
        """
        if port_name == "fluid_in" and isinstance(value, Stream):
            self.inlet_stream = value
            self._detect_fluid_config(value)
            return value.mass_flow_kg_h
        elif port_name == "electricity_in":
            return self.fan_power_kw
        return 0.0

    def get_output(self, port_name: str) -> Any:
        """
        Retrieve the output stream from a specified port.

        Args:
            port_name (str): Port identifier. Expected: 'fluid_out'.

        Returns:
            Stream: Cooled stream at outlet conditions, or None if no flow.
        """
        if port_name == "fluid_out":
            return self.outlet_stream
        return None

    def step(self, t: float) -> None:
        """
        Execute one simulation timestep.

        Computes heat transfer using the NTU-effectiveness method:
        1. Calculate mixture heat capacity using mass-weighted species Cp.
        2. Compute capacity rates: C_hot = ṁ × Cp, C_air = ṁ_air × Cp_air.
        3. Determine NTU = UA / C_min and capacity ratio R = C_min / C_max.
        4. Calculate effectiveness from crossflow correlation (Numba JIT).
        5. Compute heat duty: Q = ε × C_min × (T_hot_in - T_cold_in).
        6. Calculate outlet temperature and fan power.

        Args:
            t (float): Current simulation time in hours.

        Note:
            For wet streams containing liquid water, the water heat capacity
            (4186 J/kg·K) is used to represent the high thermal inertia.
        """
        super().step(t)

        if not self.inlet_stream or self.inlet_stream.mass_flow_kg_h <= 0:
            self.outlet_stream = None
            self.heat_duty_kw = 0.0
            self.fan_power_kw = 0.0
            return

        # ====================================================================
        # Hot Side (Process Stream) Properties
        # ====================================================================
        m_dot_total = self.inlet_stream.mass_flow_kg_h / 3600.0
        T_h_in = self.inlet_stream.temperature_k
        P_in = self.inlet_stream.pressure_pa

        # Calculate mixture molecular weight for mass fraction conversion
        mw_mix = 0.0
        for sp, mole_frac in self.inlet_stream.composition.items():
            if sp in GasConstants.SPECIES_DATA:
                mw_sp = GasConstants.SPECIES_DATA[sp]['molecular_weight']
                mw_mix += mole_frac * mw_sp

        if mw_mix < 1e-12:
            mw_mix = 18.015  # Default to water if undefined

        # Mass-weighted heat capacity calculation
        Cp_weighted = 0.0
        for sp, mole_frac in self.inlet_stream.composition.items():
            if sp in GasConstants.SPECIES_DATA:
                mw_sp = GasConstants.SPECIES_DATA[sp]['molecular_weight']
                mass_frac = (mole_frac * mw_sp) / mw_mix

                # Species-specific heat capacity
                if sp == 'H2O':
                    # Liquid water Cp for wet stream thermal inertia
                    cp_sp = 4186.0
                elif sp == 'H2':
                    cp_sp = GasConstants.CP_H2_AVG
                elif sp == 'O2':
                    cp_sp = GasConstants.CP_O2_AVG
                else:
                    cp_sp = 1000.0

                Cp_weighted += mass_frac * cp_sp

        C_hot = m_dot_total * Cp_weighted

        # ====================================================================
        # Cold Side (Air) Properties
        # ====================================================================
        m_dot_air = self.design_air_flow_kg_s
        C_air = m_dot_air * DCC.CP_AIR_J_KG_K
        T_c_in = DCC.T_A_IN_DESIGN_C + 273.15

        # ====================================================================
        # NTU-Effectiveness Calculation
        # ====================================================================
        C_min = min(C_hot, C_air)
        C_max = max(C_hot, C_air)
        R_capacity = C_min / C_max

        area_eff = self.area_m2

        # NTU = UA / C_min
        NTU = (DCC.U_W_M2_K * area_eff) / C_min

        # Effectiveness from crossflow correlation (JIT-compiled)
        eff = numba_ops.dry_cooler_ntu_effectiveness(NTU, R_capacity)
        self.effectiveness = eff
        self.ntu = NTU

        # ====================================================================
        # Heat Duty and Outlet Conditions
        # ====================================================================
        Q_max = C_min * (T_h_in - T_c_in)
        Q_actual = eff * Q_max

        self.heat_duty_kw = Q_actual / 1000.0

        # Hot side outlet temperature: Q = C_hot × (T_in - T_out)
        if C_hot > 1e-9:
            T_h_out = T_h_in - (Q_actual / C_hot)
        else:
            T_h_out = T_h_in

        # Fluid side pressure drop
        P_out = P_in - (DCC.DP_FLUID_BAR * 1e5)
        if P_out < 101325:
            P_out = 101325

        self.outlet_temp_c = T_h_out - 273.15

        self.outlet_stream = Stream(
            mass_flow_kg_h=self.inlet_stream.mass_flow_kg_h,
            temperature_k=T_h_out,
            pressure_pa=P_out,
            composition=self.inlet_stream.composition,
            phase='mixed'
        )

        # ====================================================================
        # Fan Power Calculation
        # ====================================================================
        # P_fan = (V̇ × ΔP) / η_fan
        vol_air = m_dot_air / DCC.RHO_AIR_KG_M3
        power_j_s = (vol_air * DCC.DP_AIR_PA) / DCC.ETA_FAN
        self.fan_power_kw = power_j_s / 1000.0
        self.air_mass_flow_kg_s = m_dot_air

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """
        Define the physical connection ports for this component.

        Returns:
            Dict[str, Dict[str, str]]: Port definitions with keys:
                - fluid_in: Hot process stream inlet.
                - electricity_in: Fan power supply.
                - fluid_out: Cooled process stream outlet.
        """
        return {
            'fluid_in': {'type': 'input', 'resource_type': 'stream'},
            'electricity_in': {'type': 'input', 'resource_type': 'electricity'},
            'fluid_out': {'type': 'output', 'resource_type': 'stream'}
        }

    def get_state(self) -> Dict[str, Any]:
        """
        Retrieve the component's current operational state.

        Fulfills the Component Lifecycle Contract state access, providing
        thermal performance metrics for monitoring and logging.

        Returns:
            Dict[str, Any]: State dictionary containing:
                - fluid_type (str): Detected species ('H2' or 'O2').
                - heat_duty_kw (float): Heat transfer rate (kW).
                - fan_power_kw (float): Fan electrical consumption (kW).
                - outlet_temp_c (float): Outlet temperature (°C).
                - effectiveness (float): Thermal effectiveness (0-1).
                - ntu (float): Number of Transfer Units.
                - air_flow_kg_s (float): Air mass flow rate (kg/s).
        """
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
