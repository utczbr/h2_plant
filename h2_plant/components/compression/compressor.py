"""
Multi-Stage Hydrogen Compressor Component.

Models a multi-stage reciprocating compressor for hydrogen storage applications,
implementing staged compression with intercooling to minimize work input while
respecting discharge temperature limits.

Thermodynamic Model
-------------------
The compression process follows polytropic behavior between isentropic and
isothermal limits. Each stage applies the isentropic efficiency relationship:

    W_actual = (H_2s - H_1) / η_is

where H_2s is the isentropic outlet enthalpy at constant entropy, and η_is
accounts for irreversibilities. Intercooling between stages reduces gas
temperature to the inlet value, shifting the overall process toward the
isothermal limit and reducing total work.

The number of stages is determined by the constraint that isentropic discharge
temperature must not exceed the maximum allowable limit (default 85°C),
preventing lubricant degradation and seal damage.

Drive Train Model
-----------------
Electrical power consumption includes losses through the drive train:

    W_electrical = W_shaft / (η_mechanical × η_electrical)

where mechanical efficiency accounts for bearing and coupling losses, and
electrical efficiency represents motor conversion losses.

Fluid Property Source
---------------------
This component uses real-gas properties from the LUTManager for accurate
enthalpy and entropy calculations. When LUTManager is unavailable, it falls
back to ideal gas relations with hydrogen-specific heat capacity.

Component Lifecycle Contract (Layer 1)
--------------------------------------
- ``initialize()``: Calculates optimal stage configuration using real-gas
  thermodynamic properties from LUTManager.
- ``step()``: Computes compression work and cooling duty for mass transferred
  during the current timestep.
- ``get_state()``: Exposes energy consumption, mass throughput, and stage
  metrics for monitoring and persistence.

References
----------
- Campbell, J.M. (2015). Gas Conditioning and Processing, Vol. 2.
- GPSA Engineering Data Book, 14th Ed., Section 13 (Compressors).
"""

import numpy as np
from typing import Dict, Any, Optional
import logging

from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.component_ids import ComponentID
from h2_plant.core.enums import CompressorMode
from h2_plant.core.constants import ConversionFactors
from h2_plant.core.stream import Stream

try:
    import CoolProp.CoolProp as CP
    COOLPROP_AVAILABLE = True
except ImportError:
    CP = None
    COOLPROP_AVAILABLE = False

try:
    from h2_plant.optimization.coolprop_lut import CoolPropLUT
except ImportError:
    CoolPropLUT = None

logger = logging.getLogger(__name__)


class CompressorStorage(Component):
    """
    Multi-stage reciprocating compressor for hydrogen storage operations.

    Implements staged compression with intercooling to achieve high pressure
    ratios efficiently. Stage count is automatically determined based on the
    maximum allowable discharge temperature per stage.

    The compression model iterates through stages computing:

    1. Isentropic outlet enthalpy via constant-entropy path: H_2s = H(P_out, S_in)
    2. Actual shaft work accounting for efficiency: W = (H_2s - H_1) / η_is
    3. Intercooler duty to return gas to inlet temperature: Q = H_actual - H_cooled
    4. Electrical power including drive train losses: W_el = W_shaft / (η_m × η_el)

    Component Lifecycle Contract (Layer 1):
        - ``initialize()``: Calculates optimal stage count using real-gas
          thermodynamic properties. Validates configuration parameters.
        - ``step()``: Computes compression work and cooling duty for mass
          transferred during the current timestep.
        - ``get_state()``: Returns energy consumption, mass throughput, and
          cumulative statistics for monitoring and persistence.

    Attributes:
        max_flow_kg_h: Maximum mass flow capacity in kg/h.
        inlet_pressure_bar: Suction pressure in bar.
        outlet_pressure_bar: Discharge pressure in bar.
        num_stages: Calculated number of compression stages.
        stage_pressure_ratio: Pressure ratio per stage (equal distribution).
        isentropic_efficiency: Thermodynamic efficiency per stage (0-1).
        mechanical_efficiency: Drive train mechanical efficiency (0-1).
        electrical_efficiency: Motor electrical efficiency (0-1).

    Example:
        >>> compressor = CompressorStorage(
        ...     max_flow_kg_h=100.0,
        ...     inlet_pressure_bar=40.0,
        ...     outlet_pressure_bar=140.0
        ... )
        >>> compressor.initialize(dt=1/60, registry=registry)
        >>> compressor.transfer_mass_kg = 50.0
        >>> compressor.step(t=0.0)
        >>> print(f"Specific energy: {compressor.specific_energy_kwh_kg:.3f} kWh/kg")
    """

    def __init__(
        self,
        max_flow_kg_h: float,
        inlet_pressure_bar: float,
        outlet_pressure_bar: float,
        inlet_temperature_c: float = 10.0,
        max_temperature_c: float = 85.0,
        isentropic_efficiency: float = 0.65,
        mechanical_efficiency: float = 0.96,
        electrical_efficiency: float = 0.93,
        chiller_cop: float = 3.0
    ):
        """
        Configure the multi-stage compressor design parameters.

        Stage configuration is computed during ``initialize()`` using real-gas
        properties from LUTManager when available.

        The efficiency chain models complete drive train losses:

            W_electrical = W_isentropic / (η_is × η_m × η_el)

        Args:
            max_flow_kg_h: Maximum mass flow rate in kg/h. Constrains mass
                transfer per timestep.
            inlet_pressure_bar: Suction pressure in bar. Typically set by
                upstream process conditions.
            outlet_pressure_bar: Target discharge pressure in bar. Determines
                overall compression ratio.
            inlet_temperature_c: Suction temperature in °C. Also serves as
                intercooler target temperature. Defaults to 10°C.
            max_temperature_c: Maximum allowable discharge temperature per
                stage in °C. Constrains stage pressure ratio to prevent
                lubricant degradation. Defaults to 85°C.
            isentropic_efficiency: Stage isentropic efficiency (0-1).
                Accounts for internal thermodynamic irreversibilities.
                Defaults to 0.65.
            mechanical_efficiency: Drive train mechanical efficiency (0-1).
                Accounts for bearing and coupling losses. Defaults to 0.96.
            electrical_efficiency: Motor electrical efficiency (0-1).
                Accounts for electrical-to-mechanical conversion losses.
                Defaults to 0.93.
            chiller_cop: Coefficient of performance for intercooler chillers.
                Relates cooling thermal duty to electrical consumption.
                Defaults to 3.0.
        """
        super().__init__()

        self.max_flow_kg_h = max_flow_kg_h
        self.inlet_pressure_bar = inlet_pressure_bar
        self.outlet_pressure_bar = outlet_pressure_bar

        self.inlet_temperature_c = inlet_temperature_c
        self.inlet_temperature_k = inlet_temperature_c + 273.15
        self.max_temperature_c = max_temperature_c
        self.max_temperature_k = max_temperature_c + 273.15
        self.isentropic_efficiency = isentropic_efficiency
        self.mechanical_efficiency = mechanical_efficiency
        self.electrical_efficiency = electrical_efficiency
        self.chiller_cop = chiller_cop

        self.BAR_TO_PA = 1e5
        self.J_TO_KWH = 2.7778e-7

        self.num_stages = 0
        self.stage_pressure_ratio = 1.0

        self.transfer_mass_kg = 0.0

        self.actual_mass_transferred_kg = 0.0
        self.compression_work_kwh = 0.0
        self.chilling_work_kwh = 0.0
        self.energy_consumed_kwh = 0.0
        self.heat_removed_kwh = 0.0
        self.specific_energy_kwh_kg = 0.0

        self.power_kw = 0.0

        self.mode = CompressorMode.IDLE
        self.cumulative_energy_kwh = 0.0
        self.cumulative_mass_kg = 0.0

        self._last_step_time = -1.0
        self.timestep_power_kw = 0.0
        self.timestep_energy_kwh = 0.0

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """
        Prepare the compressor for simulation execution.

        Fulfills the Component Lifecycle Contract (Layer 1) initialization
        phase by calculating the optimal number of compression stages using
        real-gas thermodynamic properties.

        The stage calculation determines the maximum pressure ratio that keeps
        isentropic discharge temperature below the specified limit, then
        distributes the total compression ratio equally across stages.

        Args:
            dt: Simulation timestep in hours.
            registry: Central component registry providing access to LUTManager
                for real-gas property lookups.

        Note:
            Stage calculation requires CoolProp for inverse property lookups
            (pressure from entropy and temperature). Falls back to a
            conservative fixed ratio of 4.0 when CoolProp is unavailable.
        """
        super().initialize(dt, registry)

        self._calculate_stage_configuration()

        logger.info(
            f"CompressorStorage '{self.component_id}': "
            f"{self.inlet_pressure_bar:.0f} → {self.outlet_pressure_bar:.0f} bar, "
            f"{self.num_stages} stages, ratio={self.stage_pressure_ratio:.2f}, "
            f"T_in={self.inlet_temperature_c:.0f}°C, T_max={self.max_temperature_c:.0f}°C"
        )

    def step(self, t: float) -> None:
        """
        Execute one simulation timestep.

        Fulfills the Component Lifecycle Contract (Layer 1) step phase by
        computing compression and cooling work for the requested mass transfer.

        The step sequence:

        1. Limit mass transfer to maximum flow capacity for the timestep.
        2. Compute multi-stage compression work using real-gas properties.
        3. Calculate intercooler thermal duty and chiller electrical consumption.
        4. Accumulate energy for current timestep (supports multiple calls).
        5. Update cumulative statistics and expose power for orchestrator.

        Args:
            t: Current simulation time in hours.
        """
        super().step(t)

        if t != self._last_step_time:
            self.timestep_power_kw = 0.0
            self.timestep_energy_kwh = 0.0
            self._last_step_time = t

        self.actual_mass_transferred_kg = 0.0
        self.energy_consumed_kwh = 0.0
        self.compression_work_kwh = 0.0
        self.chilling_work_kwh = 0.0
        self.heat_removed_kwh = 0.0

        if self.transfer_mass_kg > 0:
            self.mode = CompressorMode.LP_TO_HP

            max_transfer = self.max_flow_kg_h * self.dt
            self.actual_mass_transferred_kg = min(self.transfer_mass_kg, max_transfer)

            if self.outlet_pressure_bar <= self.inlet_pressure_bar:
                self._calculate_trivial_pass_through()
            else:
                self._calculate_compression_physics()

            self.energy_consumed_kwh = (
                self.compression_work_kwh + self.chilling_work_kwh
            )

            self.cumulative_energy_kwh += self.energy_consumed_kwh
            self.cumulative_mass_kg += self.actual_mass_transferred_kg

            batch_power_kw = 0.0
            if self.dt > 0:
                batch_power_kw = self.energy_consumed_kwh / self.dt

            self.timestep_power_kw += batch_power_kw
            self.timestep_energy_kwh += self.energy_consumed_kwh

            self.power_kw = self.timestep_power_kw

            self.transfer_mass_kg = 0.0
        else:
            self.mode = CompressorMode.IDLE
            self.power_kw = 0.0

    def get_state(self) -> Dict[str, Any]:
        """
        Retrieve the component's current operational state.

        Fulfills the Component Lifecycle Contract (Layer 1) state access,
        providing compression metrics for monitoring, logging, and state
        persistence.

        Returns:
            State dictionary containing:

            - **mode** (int): Operating mode (IDLE=0, LP_TO_HP=1).
            - **num_stages** (int): Number of compression stages.
            - **stage_pressure_ratio** (float): Pressure ratio per stage.
            - **compression_work_kwh** (float): Electrical compression work
              this timestep in kWh.
            - **chilling_work_kwh** (float): Chiller electrical consumption
              this timestep in kWh.
            - **energy_consumed_kwh** (float): Total electrical energy this
              timestep in kWh.
            - **specific_energy_kwh_kg** (float): Specific energy consumption
              in kWh/kg.
            - **cumulative_energy_kwh** (float): Total energy consumed since
              initialization in kWh.
            - **cumulative_mass_kg** (float): Total mass compressed since
              initialization in kg.
            - **isentropic_efficiency** (float): Configured stage efficiency.
            - **mechanical_efficiency** (float): Configured drive efficiency.
            - **electrical_efficiency** (float): Configured motor efficiency.
        """
        cumulative_specific = 0.0
        if self.cumulative_mass_kg > 0:
            cumulative_specific = self.cumulative_energy_kwh / self.cumulative_mass_kg

        return {
            **super().get_state(),
            'mode': int(self.mode),
            'num_stages': int(self.num_stages),
            'stage_pressure_ratio': float(self.stage_pressure_ratio),
            'transfer_mass_kg': float(self.transfer_mass_kg),
            'actual_mass_transferred_kg': float(self.actual_mass_transferred_kg),
            'compression_work_kwh': float(self.compression_work_kwh),
            'chilling_work_kwh': float(self.chilling_work_kwh),
            'energy_consumed_kwh': float(self.energy_consumed_kwh),
            'heat_removed_kwh': float(self.heat_removed_kwh),
            'specific_energy_kwh_kg': float(self.specific_energy_kwh_kg),
            'cumulative_energy_kwh': float(self.cumulative_energy_kwh),
            'cumulative_mass_kg': float(self.cumulative_mass_kg),
            'timestep_energy_kwh': float(self.timestep_energy_kwh),
            'cumulative_specific_kwh_kg': float(cumulative_specific),
            'inlet_pressure_bar': float(self.inlet_pressure_bar),
            'outlet_pressure_bar': float(self.outlet_pressure_bar),
            'inlet_temperature_c': float(self.inlet_temperature_c),
            'max_temperature_c': float(self.max_temperature_c),
            'isentropic_efficiency': float(self.isentropic_efficiency),
            'mechanical_efficiency': float(self.mechanical_efficiency),
            'electrical_efficiency': float(self.electrical_efficiency)
        }

    def _calculate_stage_configuration(self) -> None:
        """
        Determine optimal number of compression stages.

        Uses real-gas thermodynamic properties to find the maximum stage
        pressure ratio that keeps isentropic discharge temperature below the
        specified limit. The algorithm:

        1. Obtain inlet entropy S_1 at (P_in, T_in) from LUTManager.
        2. Find pressure P where isentropic compression reaches T_max using
           CoolProp inverse lookup: P = f(S_1, T_max).
        3. Maximum stage ratio r_max = P / P_in (floor at 2.0 for stability).
        4. Number of stages n = ceil(ln(r_total) / ln(r_max)).
        5. Actual stage ratio = r_total^(1/n) for equal pressure distribution.

        When CoolProp is unavailable, falls back to a conservative fixed
        maximum stage ratio of 4.0.
        """
        if not COOLPROP_AVAILABLE:
            logger.warning(
                "CoolProp not available. Using fallback stage calculation "
                "(may not match legacy behavior exactly)"
            )
            self._calculate_stages_fallback()
            return

        lut = self.get_registry_safe(ComponentID.LUT_MANAGER)

        if lut is None:
            logger.warning(
                "LUT Manager not available. Using fallback stage calculation."
            )
            self._calculate_stages_fallback()
            return

        p_in_pa = self.inlet_pressure_bar * self.BAR_TO_PA
        p_out_pa = self.outlet_pressure_bar * self.BAR_TO_PA

        s1 = lut.lookup('H2', 'S', p_in_pa, self.inlet_temperature_k)

        try:
            if CoolPropLUT:
                p_out_1s_max_t = CoolPropLUT.PropsSI(
                    'P', 'S', s1, 'T', self.max_temperature_k, 'H2'
                )
            elif COOLPROP_AVAILABLE:
                p_out_1s_max_t = CP.PropsSI(
                    'P', 'S', s1, 'T', self.max_temperature_k, 'H2'
                )
            else:
                raise RuntimeError("CoolProp unavailable")
        except Exception as e:
            logger.warning(f"CoolProp inverse lookup failed: {e}. Using fallback.")
            self._calculate_stages_fallback()
            return

        r_stage_max_isentropic = p_out_1s_max_t / p_in_pa
        r_stage_max_isentropic = max(2.0, r_stage_max_isentropic)

        r_total = p_out_pa / p_in_pa

        n_stages = int(np.ceil(np.log(r_total) / np.log(r_stage_max_isentropic)))
        self.num_stages = max(1, n_stages)

        self.stage_pressure_ratio = r_total ** (1.0 / self.num_stages)

    def _calculate_trivial_pass_through(self) -> None:
        """
        Handle case where no compression is required.

        When outlet pressure is at or below inlet pressure, the compressor
        acts as a pass-through with zero energy consumption. This prevents
        division-by-zero and negative work calculations.
        """
        self.compression_work_kwh = 0.0
        self.chilling_work_kwh = 0.0
        self.specific_energy_kwh_kg = 0.0
        self.heat_removed_kwh = 0.0

    def _calculate_stages_fallback(self) -> None:
        """
        Calculate stage configuration using ideal gas assumptions.

        Fallback method when CoolProp or LUTManager are unavailable. Uses a
        conservative maximum stage pressure ratio of 4.0, typical for
        hydrogen compressors with standard temperature limits.
        """
        p_in_pa = self.inlet_pressure_bar * self.BAR_TO_PA
        p_out_pa = self.outlet_pressure_bar * self.BAR_TO_PA
        r_total = p_out_pa / p_in_pa

        r_stage_max = 4.0
        n_stages = int(np.ceil(np.log(r_total) / np.log(r_stage_max)))
        self.num_stages = max(1, n_stages)
        self.stage_pressure_ratio = r_total ** (1.0 / self.num_stages)

    def _calculate_compression_fallback(self) -> None:
        """
        Calculate compression energy using ideal gas relations.

        Fallback method when LUTManager is unavailable. Uses the ideal gas
        isentropic temperature-pressure relationship:

            T_2s = T_1 × (P_2/P_1)^((γ-1)/γ)

        with hydrogen-specific heat capacity (Cp = 14.3 kJ/kg·K) and heat
        ratio (γ = 1.41).

        Stage work is computed as:

            W = Cp × ΔT_actual = Cp × (T_2s - T_1) / η_is

        Intercooler duty returns gas to inlet temperature after each stage.
        """
        gamma = 1.41
        cp = 14300.0

        p_in_pa = self.inlet_pressure_bar * self.BAR_TO_PA
        p_out_pa = self.outlet_pressure_bar * self.BAR_TO_PA
        r_total = p_out_pa / p_in_pa

        exponent = (gamma - 1) / gamma

        w_compression_total = 0.0
        q_removed_total = 0.0
        t_current = self.inlet_temperature_k

        for i in range(self.num_stages):
            t_out_isentropic = t_current * (self.stage_pressure_ratio ** exponent)

            delta_t_ideal = t_out_isentropic - t_current
            delta_t_actual = delta_t_ideal / self.isentropic_efficiency
            t_out_actual = t_current + delta_t_actual

            w_stage = cp * delta_t_actual
            w_compression_total += w_stage

            q_stage = cp * (t_out_actual - self.inlet_temperature_k)
            q_removed_total += q_stage
            t_current = self.inlet_temperature_k

        drive_efficiency = self.mechanical_efficiency * self.electrical_efficiency
        w_compression_electrical = w_compression_total / drive_efficiency

        compress_kwh_kg = w_compression_electrical * self.J_TO_KWH

        q_chiller_j_kg = q_removed_total / self.chiller_cop
        self.chilling_energy_kwh_kg = q_chiller_j_kg * self.J_TO_KWH

        self.specific_energy_kwh_kg = compress_kwh_kg + self.chilling_energy_kwh_kg

        self.compression_work_kwh = compress_kwh_kg * self.actual_mass_transferred_kg
        self.chilling_work_kwh = self.chilling_energy_kwh_kg * self.actual_mass_transferred_kg

    def _calculate_compression_physics(self) -> None:
        """
        Calculate compression energy using real-gas thermodynamics.

        Implements multi-stage compression with intercooling using enthalpy
        and entropy from LUTManager. Each stage follows the isentropic
        efficiency model:

        1. **Isentropic compression**: Outlet enthalpy at constant entropy
           H_2s = H(P_out, S_in) from LUTManager or CoolProp.

        2. **Actual shaft work**: Accounting for thermodynamic irreversibilities
           W_shaft = (H_2s - H_1) / η_is

        3. **Electrical work**: Including drive train losses
           W_el = W_shaft / (η_m × η_el)

        4. **Intercooler duty**: Heat removal to return gas to inlet temperature
           Q = H_actual - H(P_out, T_inlet)

        5. **Chiller consumption**: Electrical power for cooling
           W_chill = Q / COP

        Total electrical consumption is the sum of compression and chilling work.
        """
        lut = self.get_registry_safe(ComponentID.LUT_MANAGER)

        if lut is None:
            self._calculate_compression_fallback()
            return

        p_in_pa = self.inlet_pressure_bar * self.BAR_TO_PA
        p_out_pa = self.outlet_pressure_bar * self.BAR_TO_PA

        h1 = lut.lookup('H2', 'H', p_in_pa, self.inlet_temperature_k)
        s1 = lut.lookup('H2', 'S', p_in_pa, self.inlet_temperature_k)

        r_total = p_out_pa / p_in_pa
        r_stage = r_total ** (1.0 / self.num_stages)

        w_compression_total = 0.0
        q_removed_total = 0.0
        p_current = p_in_pa

        t_stage_in = self.inlet_temperature_k

        for i in range(self.num_stages):
            s_stage_in = lut.lookup('H2', 'S', p_current, t_stage_in)
            h_stage_in = lut.lookup('H2', 'H', p_current, t_stage_in)

            p_out_stage = p_current * r_stage
            if i == self.num_stages - 1:
                p_out_stage = p_out_pa

            try:
                h2s = lut.lookup_isentropic_enthalpy('H2', p_out_stage, s_stage_in)
            except Exception:
                if CoolPropLUT:
                    h2s = CoolPropLUT.PropsSI('H', 'P', p_out_stage, 'S', s_stage_in, 'H2')
                elif COOLPROP_AVAILABLE:
                    h2s = CP.PropsSI('H', 'P', p_out_stage, 'S', s_stage_in, 'H2')
                else:
                    h2s = h_stage_in * (p_out_stage/p_current)**0.28

            ws = h2s - h_stage_in
            wa = ws / self.isentropic_efficiency
            h2a = h_stage_in + wa
            w_compression_total += wa

            h_cooled_next = lut.lookup('H2', 'H', p_out_stage, self.inlet_temperature_k)
            q_removed = h2a - h_cooled_next
            q_removed_total += q_removed

            p_current = p_out_stage
            t_stage_in = self.inlet_temperature_k

        w_chilling_total = q_removed_total / self.chiller_cop

        drive_efficiency = self.mechanical_efficiency * self.electrical_efficiency
        w_compression_electrical = w_compression_total / drive_efficiency

        w_total_j_kg = w_compression_electrical + w_chilling_total

        self.specific_energy_kwh_kg = w_total_j_kg * self.J_TO_KWH

        self.compression_work_kwh = (w_compression_electrical * self.J_TO_KWH *
                                     self.actual_mass_transferred_kg)
        self.chilling_work_kwh = (w_chilling_total * self.J_TO_KWH *
                                  self.actual_mass_transferred_kg)
        self.heat_removed_kwh = (q_removed_total * self.J_TO_KWH *
                                 self.actual_mass_transferred_kg)

    # =========================================================================
    # Port Interface Methods (Component Lifecycle Contract Layer 1)
    # =========================================================================

    def get_output(self, port_name: str) -> Any:
        """
        Retrieve the output stream from a specified port.

        Returns compressed hydrogen at outlet pressure and inlet temperature
        (after final aftercooling stage).

        Args:
            port_name: Port identifier. Valid values are 'h2_out' or 'outlet'.

        Returns:
            Stream object containing compressed hydrogen with:

            - mass_flow_kg_h: Flow rate based on mass transferred this step.
            - temperature_k: Inlet temperature (gas is aftercooled).
            - pressure_pa: Design outlet pressure.
            - composition: Pure hydrogen {'H2': 1.0}.

        Raises:
            ValueError: If port_name is not a recognized output port.
        """
        if port_name == 'h2_out' or port_name == 'outlet':
            return Stream(
                mass_flow_kg_h=(self.actual_mass_transferred_kg / self.dt
                               if self.dt > 0 else 0.0),
                temperature_k=self.inlet_temperature_k,
                pressure_pa=self.outlet_pressure_bar * self.BAR_TO_PA,
                composition={'H2': 1.0},
                phase='gas'
            )
        else:
            raise ValueError(
                f"Unknown output port '{port_name}' on {self.component_id}"
            )

    def receive_input(self, port_name: str, value: Any, resource_type: str) -> float:
        """
        Accept an input stream at the specified port.

        Accumulates incoming hydrogen into the transfer buffer, respecting
        maximum flow capacity. Supports both hydrogen streams and electricity
        acknowledgment.

        Args:
            port_name: Target port identifier. Valid values are 'h2_in',
                'inlet', or 'electricity_in'.
            value: Input value. Stream object for hydrogen ports, float for
                electricity port.
            resource_type: Resource classification hint (unused but required
                by interface).

        Returns:
            Amount accepted in appropriate units:

            - Hydrogen ports: Mass accepted in kg.
            - Electricity port: Power value echoed back.
        """
        if port_name == 'h2_in' or port_name == 'inlet':
            if isinstance(value, Stream):
                available_mass = value.mass_flow_kg_h * self.dt
                max_capacity = self.max_flow_kg_h * self.dt

                space_left = max(0.0, max_capacity - self.transfer_mass_kg)
                accepted_mass = min(available_mass, space_left)

                self.transfer_mass_kg += accepted_mass
                return accepted_mass

        elif port_name == 'electricity_in':
            return value if isinstance(value, (int, float)) else 0.0

        return 0.0

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """
        Define the physical connection ports for this component.

        Returns:
            Port definitions dictionary with keys:

            - **h2_in**: Low-pressure hydrogen feed input.
            - **electricity_in**: Grid power for motor and chiller.
            - **h2_out**: High-pressure hydrogen product output.
            - **outlet**: Alias for h2_out (backward compatibility).
        """
        return {
            'h2_in': {
                'type': 'input',
                'resource_type': 'hydrogen',
                'units': 'kg/h'
            },
            'electricity_in': {
                'type': 'input',
                'resource_type': 'electricity',
                'units': 'MW'
            },
            'h2_out': {
                'type': 'output',
                'resource_type': 'hydrogen',
                'units': 'kg/h'
            },
            'outlet': {
                'type': 'output',
                'resource_type': 'hydrogen',
                'units': 'kg/h'
            }
        }
