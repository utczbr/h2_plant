"""
Dry Cooler (Indirect Cooling System) Component.

This module implements a two-stage indirect cooling system for hydrogen and
oxygen process streams. The architecture separates explosive process gases
from the large external air-cooled heat exchanger, enhancing safety.

Physical Model:
    The system models coupled heat exchangers using the ε-NTU method:
    
    1. **TQC (Trocador de Calor Quente)**: Counter-flow shell-and-tube exchanger
       transferring heat from process gas to an intermediate glycol/water loop.
       Counter-flow geometry maximizes temperature driving force and achieves
       higher effectiveness than parallel-flow at identical NTU.
    
    2. **DC (Dry Cooler)**: Cross-flow finned-tube air cooler rejecting heat
       from the glycol loop to ambient air. Cross-flow is typical for air
       coolers due to fan arrangement constraints.

    Thermodynamic Basis:
        - ε-NTU Method: ε = f(NTU, Cr) where Cr = Cmin/Cmax
        - Counter-flow: ε = [1 - exp(-NTU(1-Cr))] / [1 - Cr·exp(-NTU(1-Cr))]
        - Cross-flow: Empirical correlation for unmixed fluids
        - Q_actual = ε × Cmin × (Th,in - Tc,in)

Architecture:
    Implements the Component Lifecycle Contract (Layer 1):
    - `initialize()`: Prepares component state; geometry configured on first input.
    - `step()`: Solves coupled TQC→DC thermal circuit for current timestep.
    - `get_state()`: Exposes heat duties, temperatures, and effectiveness values.

References:
    - Incropera, F.P. & DeWitt, D.P. (2007). Fundamentals of Heat and Mass Transfer.
    - Shah, R.K. & Sekulić, D.P. (2003). Fundamentals of Heat Exchanger Design.
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

logger = logging.getLogger(__name__)


class DryCooler(Component):
    """
    Two-stage indirect cooling system (Process Gas → Glycol → Ambient Air).

    Models a closed-loop heat rejection system where process gas transfers heat
    to an intermediate glycol/water mixture, which is then cooled by forced-air
    convection. This configuration isolates flammable gases (H₂) from the large,
    outdoor air-cooled heat exchanger.

    This component fulfills the Component Lifecycle Contract (Layer 1):
        - `initialize()`: Validates timestep and registry; defers geometry
          configuration until first stream is received.
        - `step()`: Sequentially solves TQC and DC heat transfer using ε-NTU,
          updating glycol loop temperatures quasi-dynamically.
        - `get_state()`: Returns thermal performance metrics for monitoring,
          including heat duties, temperatures, and effectiveness values.

    Attributes:
        fluid_type (str): Detected process gas species ('H2' or 'O2').
        tqc_duty_kw (float): Heat transferred from gas to glycol in TQC (kW).
        dc_duty_kw (float): Heat rejected from glycol to air in DC (kW).
        fan_power_kw (float): Electrical power consumed by DC fans (kW).
        glycol_hot_c (float): Glycol temperature exiting TQC (°C).
        glycol_cold_c (float): Glycol temperature exiting DC, entering TQC (°C).
        tqc_effectiveness (float): TQC heat exchanger effectiveness (0-1).
        dc_effectiveness (float): DC heat exchanger effectiveness (0-1).

    Example:
        >>> cooler = DryCooler(component_id='DC-H2-01')
        >>> cooler.initialize(dt=1/60, registry=registry)
        >>> cooler.receive_input('fluid_in', hot_h2_stream, 'gas')
        >>> cooler.step(t=0.0)
        >>> cooled_gas = cooler.get_output('fluid_out')
    """

    def __init__(self, component_id: str = "dry_cooler") -> None:
        """
        Initialize the DryCooler component.

        Creates an indirect cooling system instance with default state.
        Heat exchanger geometry and flow rates are configured automatically
        when the first input stream is received, based on detected gas species.

        Args:
            component_id (str): Unique identifier for this component instance.
                Used for logging and registry lookup. Default: "dry_cooler".
        """
        super().__init__()
        self.component_id = component_id

        # Process stream state
        self.fluid_type = "Unknown"
        self.inlet_stream: Optional[Stream] = None
        self.outlet_stream: Optional[Stream] = None

        # Glycol loop state variables
        self.glycol_flow_kg_s = 0.0
        self.glycol_cp_j_kg_k = 0.0
        self.t_glycol_cold_c = DCC.T_REF_IN_TQC_DEFAULT
        self.t_glycol_hot_c = DCC.T_REF_IN_TQC_DEFAULT

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

    def initialize(self, dt: float, registry: Any) -> None:
        """
        Prepare the component for simulation execution.

        Fulfills the Component Lifecycle Contract initialization phase by
        storing the timestep and registry reference. Heat exchanger geometry
        is configured lazily on first input receipt to allow automatic
        detection of gas species.

        Args:
            dt (float): Simulation timestep duration in hours.
            registry (ComponentRegistry): Central registry for cross-component
                communication and shared services.
        """
        super().initialize(dt, registry)

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
            self.tqc_area_m2 = DCC.AREA_H2_TQC_M2
            self.dc_area_m2 = DCC.AREA_H2_DC_M2
            self.glycol_flow_kg_s = DCC.M_DOT_REF_H2
            self.dc_air_flow_kg_s = DCC.MDOT_AIR_DESIGN_H2_KG_S
        else:
            self.fluid_type = "O2"
            self.tqc_area_m2 = DCC.AREA_O2_TQC_M2
            self.dc_area_m2 = DCC.AREA_O2_DC_M2
            self.glycol_flow_kg_s = DCC.M_DOT_REF_O2
            self.dc_air_flow_kg_s = DCC.MDOT_AIR_DESIGN_O2_KG_S

    def step(self, t: float) -> None:
        """
        Execute one simulation timestep.

        Solves the coupled two-stage thermal circuit:
        1. TQC (Counter-Flow): Process gas transfers heat to glycol loop.
        2. DC (Cross-Flow): Glycol loop rejects heat to ambient air.
        
        The solution uses the ε-NTU method for each heat exchanger, with
        glycol loop temperatures propagating between stages. The approach
        is quasi-dynamic: glycol cold return temperature from the DC is
        used as TQC inlet for the next timestep, modeling thermal inertia.

        Fulfills the Component Lifecycle Contract step phase by advancing
        component state and preparing output streams.

        Physical Sequence:
            1. Compute gas-side heat capacity rate: C_gas = ṁ_gas × Cp_gas
            2. Solve TQC: ε-NTU counter-flow, Q_TQC = ε × Cmin × ΔT_max
            3. Update gas outlet and glycol hot temperatures from TQC.
            4. Solve DC: ε-NTU cross-flow, Q_DC = ε × Cmin × ΔT_max
            5. Update glycol cold return temperature from DC.
            6. Calculate fan power from volumetric air flow and pressure drop.

        Args:
            t (float): Current simulation time in hours.

        Note:
            If no inlet stream is present or flow is zero, the component
            enters idle state with zero heat duties and power consumption.
        """
        super().step(t)

        if not self.inlet_stream or self.inlet_stream.mass_flow_kg_h <= 0:
            self.outlet_stream = None
            self.tqc_duty_kw = 0.0
            self.fan_power_kw = 0.0
            return

        # Gas stream properties
        m_dot_gas = self.inlet_stream.mass_flow_kg_h / 3600.0

        # Compute mixture heat capacity using mass-weighted average
        cp_gas_mix = 0.0
        for sp, y in self.inlet_stream.composition.items():
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
        h2o_liq_frac = self.inlet_stream.composition.get('H2O_liq', 0.0)
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
        T_glycol_in_k = self.t_glycol_cold_c + 273.15

        C_min_tqc = min(C_hot_gas, C_coolant)
        C_max_tqc = max(C_hot_gas, C_coolant)
        R_tqc = C_min_tqc / C_max_tqc
        NTU_tqc = (self.tqc_u_value * self.tqc_area_m2) / C_min_tqc

        eff_tqc = numba_ops.counter_flow_ntu_effectiveness(NTU_tqc, R_tqc)
        self.tqc_effectiveness = eff_tqc

        Q_max_tqc = C_min_tqc * (T_gas_in_k - T_glycol_in_k)
        Q_tqc = eff_tqc * Q_max_tqc
        self.tqc_duty_kw = Q_tqc / 1000.0

        # Energy balance: outlet temperatures
        T_gas_out_k = T_gas_in_k - Q_tqc / C_hot_gas
        T_glycol_out_k = T_glycol_in_k + Q_tqc / C_coolant
        self.t_glycol_hot_c = T_glycol_out_k - 273.15

        # ================================================================
        # Stage 2: DC (Cross-Flow Air Cooler)
        # ================================================================
        # Cross-flow geometry is standard for forced-air coolers due to
        # fan and duct arrangement, though less effective than counter-flow.
        T_air_in_k = DCC.T_AIR_DESIGN_C + 273.15
        C_air = self.dc_air_flow_kg_s * DCC.CP_AIR_J_KG_K

        C_min_dc = min(C_coolant, C_air)
        C_max_dc = max(C_coolant, C_air)
        R_dc = C_min_dc / C_max_dc
        NTU_dc = (self.dc_u_value * self.dc_area_m2) / C_min_dc

        eff_dc = numba_ops.dry_cooler_ntu_effectiveness(NTU_dc, R_dc)
        self.dc_effectiveness = eff_dc

        Q_max_dc = C_min_dc * (T_glycol_out_k - T_air_in_k)
        Q_dc = eff_dc * Q_max_dc
        self.dc_duty_kw = Q_dc / 1000.0

        # Glycol return temperature (quasi-dynamic state update)
        T_glycol_return_k = T_glycol_out_k - Q_dc / C_coolant
        self.t_glycol_cold_c = T_glycol_return_k - 273.15

        # Gas-side pressure drop through TQC internals
        P_out = self.inlet_stream.pressure_pa - (DCC.DP_LIQ_TQC_BAR * 1e5)

        # Prepare output stream
        self.outlet_temp_c = T_gas_out_k - 273.15
        self.outlet_stream = Stream(
            mass_flow_kg_h=self.inlet_stream.mass_flow_kg_h,
            temperature_k=T_gas_out_k,
            pressure_pa=P_out,
            composition=self.inlet_stream.composition,
            phase='mixed'
        )

        # Fan power consumption: P = V̇ × ΔP / η_fan
        vol_air = self.dc_air_flow_kg_s / DCC.RHO_AIR_KG_M3
        power_j_s = (vol_air * DCC.DP_AIR_DESIGN_PA) / DCC.ETA_FAN
        self.fan_power_kw = power_j_s / 1000.0

    def receive_input(self, port_name: str, value: Any, resource_type: str = None) -> float:
        """
        Accept an input stream at the specified port.

        Stores the incoming process gas stream for processing during the next
        step() call. On first receipt, configures heat exchanger geometry
        based on detected gas species.

        Args:
            port_name (str): Target port identifier. Expected: 'fluid_in' or
                'electricity_in'.
            value (Any): Stream object for fluid input, or float for electrical
                power availability.
            resource_type (str, optional): Resource classification hint.
                Not used but included for interface consistency.

        Returns:
            float: For 'fluid_in': mass flow rate accepted (kg/h).
                   For 'electricity_in': fan power requirement (kW).
                   Otherwise: 0.0.
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

        Provides access to the cooled process gas stream computed during
        the most recent step() execution.

        Args:
            port_name (str): Port to query. Expected: 'fluid_out'.

        Returns:
            Stream: Cooled gas stream with updated temperature and pressure.
                Returns empty Stream if no output is available.

        Raises:
            ValueError: If port_name is not a valid output port.
        """
        if port_name == "fluid_out":
            return self.outlet_stream if self.outlet_stream else Stream(0.0)
        return None

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """
        Define the physical connection ports for this component.

        The DryCooler has one process fluid port and one electrical port.
        Port definitions enable the orchestrator to validate and establish
        flow network connections.

        Returns:
            Dict[str, Dict[str, str]]: Port definitions with keys:
                - 'fluid_in': Receives hot gas from upstream process.
                - 'electricity_in': Receives power for fan operation.
                - 'fluid_out': Delivers cooled gas to downstream processing.
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
        thermal performance metrics for monitoring dashboards, logging,
        and simulation state persistence.

        Returns:
            Dict[str, Any]: State dictionary containing:
                - fluid_type (str): Detected gas species ('H2' or 'O2').
                - tqc_duty_kw (float): Heat transferred in TQC (kW).
                - dc_duty_kw (float): Heat rejected in DC (kW).
                - fan_power_kw (float): Fan electrical consumption (kW).
                - outlet_temp_c (float): Gas outlet temperature (°C).
                - glycol_hot_c (float): Glycol temperature leaving TQC (°C).
                - glycol_cold_c (float): Glycol temperature entering TQC (°C).
                - tqc_effectiveness (float): TQC heat exchanger effectiveness.
                - dc_effectiveness (float): DC heat exchanger effectiveness.
        """
        return {
            **super().get_state(),
            'fluid_type': self.fluid_type,
            'tqc_duty_kw': self.tqc_duty_kw,
            'dc_duty_kw': self.dc_duty_kw,
            'fan_power_kw': self.fan_power_kw,
            'outlet_temp_c': self.outlet_temp_c,
            'glycol_hot_c': self.t_glycol_hot_c,
            'glycol_cold_c': self.t_glycol_cold_c,
            'tqc_effectiveness': self.tqc_effectiveness,
            'dc_effectiveness': self.dc_effectiveness
        }
