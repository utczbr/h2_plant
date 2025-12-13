"""
Multi-Stage Hydrogen Compressor Component.

This module implements a multi-stage reciprocating compressor for hydrogen
storage applications. The compressor achieves high pressure ratios through
staged compression with intercooling, minimizing work input while respecting
discharge temperature limits.

Thermodynamic Principles:
    - **Polytropic Compression**: Each stage compresses hydrogen following a
      polytropic process between isentropic and isothermal limits. The actual
      path depends on heat transfer and internal losses.
    - **Isentropic Efficiency**: Relates ideal (reversible adiabatic) work to
      actual work: η_is = W_isentropic / W_actual.
    - **Intercooling**: Cooling gas between stages reduces inlet temperature
      for subsequent stages, significantly reducing total compression work.
      The work reduction approaches the isothermal limit with many stages.
    - **Stage Optimization**: Number of stages is determined by limiting
      discharge temperature to prevent lubricant degradation and seal damage.

Architecture:
    Implements the Component Lifecycle Contract (Layer 1):
    - `initialize()`: Calculates optimal stage configuration using real-gas
      properties from LUTManager.
    - `step()`: Computes compression and cooling work for the current mass transfer.
    - `get_state()`: Exposes energy consumption and stage metrics for monitoring.

Model Approach:
    Uses real-gas properties from CoolProp/LUTManager for accurate enthalpy
    and entropy calculations. Stage pressure ratio is determined by limiting
    isentropic discharge temperature to the specified maximum.

References:
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
    ratios efficiently. The number of stages is automatically calculated based
    on maximum allowable discharge temperature per stage.

    This component fulfills the Component Lifecycle Contract (Layer 1):
        - `initialize()`: Calculates optimal number of stages using real-gas
          thermodynamic properties. Validates configuration parameters.
        - `step()`: Computes actual compression work and cooling duty for the
          mass transferred during the current timestep.
        - `get_state()`: Returns energy consumption, mass throughput, and
          cumulative statistics for monitoring and persistence.

    The compression model iterates through stages, computing:
    1. Isentropic outlet enthalpy: H2s = H(P_out, S_in)
    2. Actual work: W_actual = (H2s - H1) / η_is
    3. Cooling duty: Q = H_actual - H_cooled

    Attributes:
        max_flow_kg_h (float): Maximum mass flow capacity (kg/h).
        inlet_pressure_bar (float): Suction pressure (bar).
        outlet_pressure_bar (float): Discharge pressure (bar).
        num_stages (int): Calculated number of compression stages.
        stage_pressure_ratio (float): Pressure ratio per stage.

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
        chiller_cop: float = 3.0
    ):
        """
        Initialize the multi-stage compressor.

        Configures the compressor with design operating conditions. Stage
        configuration is computed during initialize() using real-gas properties.

        Args:
            max_flow_kg_h (float): Maximum mass flow rate in kg/h. Determines
                the upper limit for mass transfer in each timestep.
            inlet_pressure_bar (float): Suction pressure in bar. Typically set
                by upstream tank or process pressure.
            outlet_pressure_bar (float): Target discharge pressure in bar.
                Determines overall compression ratio.
            inlet_temperature_c (float): Suction temperature in °C. Also used
                as the target temperature for intercoolers. Default: 10°C.
            max_temperature_c (float): Maximum allowable discharge temperature
                per stage in °C. Limits stage pressure ratio. Default: 85°C.
            isentropic_efficiency (float): Stage isentropic efficiency (0-1).
                Accounts for internal losses. Default: 0.65.
            chiller_cop (float): Coefficient of performance for intercooler
                chillers. Relates cooling duty to electrical consumption.
                Default: 3.0.
        """
        super().__init__()

        # Design configuration
        self.max_flow_kg_h = max_flow_kg_h
        self.inlet_pressure_bar = inlet_pressure_bar
        self.outlet_pressure_bar = outlet_pressure_bar

        # Operating conditions
        self.inlet_temperature_c = inlet_temperature_c
        self.inlet_temperature_k = inlet_temperature_c + 273.15
        self.max_temperature_c = max_temperature_c
        self.max_temperature_k = max_temperature_c + 273.15
        self.isentropic_efficiency = isentropic_efficiency
        self.chiller_cop = chiller_cop

        # Unit conversion constants
        self.BAR_TO_PA = 1e5
        self.J_TO_KWH = 2.7778e-7

        # Stage configuration (computed in initialize)
        self.num_stages = 0
        self.stage_pressure_ratio = 1.0

        # Input interface (set by flow network or control logic)
        self.transfer_mass_kg = 0.0

        # Timestep output variables
        self.actual_mass_transferred_kg = 0.0
        self.compression_work_kwh = 0.0
        self.chilling_work_kwh = 0.0
        self.energy_consumed_kwh = 0.0
        self.heat_removed_kwh = 0.0
        self.specific_energy_kwh_kg = 0.0

        # Power for orchestrator monitoring
        self.power_kw = 0.0

        # Operational state
        self.mode = CompressorMode.IDLE
        self.cumulative_energy_kwh = 0.0
        self.cumulative_mass_kg = 0.0

        # Timestep accumulation for handling multiple calls per step
        self._last_step_time = -1.0
        self.timestep_power_kw = 0.0
        self.timestep_energy_kwh = 0.0

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """
        Prepare the compressor for simulation execution.

        Fulfills the Component Lifecycle Contract initialization phase by
        calculating the optimal number of compression stages. Uses real-gas
        thermodynamic properties to determine the maximum stage pressure ratio
        that keeps discharge temperature below the specified limit.

        Args:
            dt (float): Simulation timestep in hours.
            registry (ComponentRegistry): Central registry for component access.

        Note:
            Stage calculation requires CoolProp for inverse property lookups
            (P from S,T). Falls back to conservative fixed ratio if unavailable.
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

        Computes compression and cooling work for the mass transfer request.
        Updates cumulative statistics and exposes power consumption for
        orchestrator monitoring.

        This method fulfills the Component Lifecycle Contract step phase:
        1. Limits mass transfer to maximum flow capacity.
        2. Computes multi-stage compression using real-gas properties.
        3. Calculates intercooler duty and chiller electrical consumption.
        4. Accumulates energy for current timestep (handles multiple calls).

        Args:
            t (float): Current simulation time in hours.
        """
        super().step(t)

        # Reset accumulators on new timestep
        if t != self._last_step_time:
            self.timestep_power_kw = 0.0
            self.timestep_energy_kwh = 0.0
            self._last_step_time = t

        # Reset step-local variables
        self.actual_mass_transferred_kg = 0.0
        self.energy_consumed_kwh = 0.0
        self.compression_work_kwh = 0.0
        self.chilling_work_kwh = 0.0
        self.heat_removed_kwh = 0.0

        if self.transfer_mass_kg > 0:
            self.mode = CompressorMode.LP_TO_HP

            # Limit transfer to capacity
            max_transfer = self.max_flow_kg_h * self.dt
            self.actual_mass_transferred_kg = min(self.transfer_mass_kg, max_transfer)

            # Handle trivial case (no compression needed)
            if self.outlet_pressure_bar <= self.inlet_pressure_bar:
                self._calculate_trivial_pass_through()
            else:
                self._calculate_compression_physics()

            # Total energy for this step
            self.energy_consumed_kwh = (
                self.compression_work_kwh + self.chilling_work_kwh
            )

            # Update cumulative counters
            self.cumulative_energy_kwh += self.energy_consumed_kwh
            self.cumulative_mass_kg += self.actual_mass_transferred_kg

            # Accumulate power for timestep
            batch_power_kw = 0.0
            if self.dt > 0:
                batch_power_kw = self.energy_consumed_kwh / self.dt

            self.timestep_power_kw += batch_power_kw
            self.timestep_energy_kwh += self.energy_consumed_kwh

            # Expose for orchestrator monitoring
            self.power_kw = self.timestep_power_kw

            # Clear input for next step (consumed)
            self.transfer_mass_kg = 0.0
        else:
            self.mode = CompressorMode.IDLE
            self.power_kw = 0.0

    def get_state(self) -> Dict[str, Any]:
        """
        Retrieve the component's current operational state.

        Fulfills the Component Lifecycle Contract state access, providing
        compression metrics for monitoring, logging, and state persistence.

        Returns:
            Dict[str, Any]: State dictionary containing:
                - mode (int): Current operating mode (IDLE or LP_TO_HP).
                - num_stages (int): Number of compression stages.
                - stage_pressure_ratio (float): Pressure ratio per stage.
                - compression_work_kwh (float): Mechanical work this step (kWh).
                - chilling_work_kwh (float): Cooling energy this step (kWh).
                - energy_consumed_kwh (float): Total energy this step (kWh).
                - specific_energy_kwh_kg (float): Specific energy (kWh/kg).
                - cumulative_energy_kwh (float): Total energy consumed (kWh).
                - cumulative_mass_kg (float): Total mass compressed (kg).
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
            'max_temperature_c': float(self.max_temperature_c)
        }

    def _calculate_stage_configuration(self) -> None:
        """
        Determine optimal number of compression stages.

        Calculates the maximum allowable pressure ratio per stage based on
        the constraint that isentropic discharge temperature must not exceed
        T_max. Uses real-gas properties for accurate entropy-temperature
        relationships.

        The algorithm:
        1. Get inlet entropy at (P_in, T_in).
        2. Find pressure P where isentropic compression reaches T_max.
        3. Maximum stage ratio = P_max / P_in (with safety floor of 2.0).
        4. Number of stages = ceil(log(r_total) / log(r_stage_max)).
        5. Actual stage ratio = r_total^(1/n_stages) for equal distribution.

        Note:
            Requires CoolProp for inverse lookup (P from S,T). Falls back to
            conservative fixed ratio of 4.0 if CoolProp is unavailable.
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

        # Inlet entropy from LUT
        s1 = lut.lookup('H2', 'S', p_in_pa, self.inlet_temperature_k)

        # Inverse lookup: find P where T = T_max at constant S
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

        # Maximum isentropic stage pressure ratio
        r_stage_max_isentropic = p_out_1s_max_t / p_in_pa
        r_stage_max_isentropic = max(2.0, r_stage_max_isentropic)

        # Total ratio and number of stages
        r_total = p_out_pa / p_in_pa

        n_stages = int(np.ceil(np.log(r_total) / np.log(r_stage_max_isentropic)))
        self.num_stages = max(1, n_stages)

        # Equal distribution of pressure ratio across stages
        self.stage_pressure_ratio = r_total ** (1.0 / self.num_stages)

    def _calculate_trivial_pass_through(self) -> None:
        """
        Handle case where no compression is required.

        When outlet pressure equals or is less than inlet pressure, the
        compressor acts as a pass-through with no energy consumption.
        """
        self.compression_work_kwh = 0.0
        self.chilling_work_kwh = 0.0
        self.specific_energy_kwh_kg = 0.0
        self.heat_removed_kwh = 0.0

    def _calculate_stages_fallback(self) -> None:
        """
        Calculate stage configuration using conservative assumptions.

        Fallback method when CoolProp is unavailable. Uses a fixed maximum
        stage pressure ratio of 4.0, which is conservative for hydrogen.
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
        Calculate compression energy using ideal gas approximation.

        Fallback method when LUT is unavailable. Uses ideal gas relations
        with hydrogen-specific heat ratio (γ = 1.41).

        The ideal gas adiabatic work per stage is:
            W = Cp × T1 × [(P2/P1)^((γ-1)/γ) - 1] / η_is
        """
        gamma = 1.41
        cp = 14300.0  # J/(kg·K) for H2

        p_in_pa = self.inlet_pressure_bar * self.BAR_TO_PA
        p_out_pa = self.outlet_pressure_bar * self.BAR_TO_PA
        r_total = p_out_pa / p_in_pa

        exponent = (gamma - 1) / gamma

        w_compression_total = 0.0
        q_removed_total = 0.0
        t_current = self.inlet_temperature_k

        for i in range(self.num_stages):
            # Isentropic temperature rise
            t_out_isentropic = t_current * (self.stage_pressure_ratio ** exponent)

            # Actual temperature rise (accounting for efficiency)
            delta_t_ideal = t_out_isentropic - t_current
            delta_t_actual = delta_t_ideal / self.isentropic_efficiency
            t_out_actual = t_current + delta_t_actual

            # Stage work
            w_stage = cp * delta_t_actual
            w_compression_total += w_stage

            # Intercooling duty (all stages cooled to inlet temperature)
            q_stage = cp * (t_out_actual - self.inlet_temperature_k)
            q_removed_total += q_stage
            t_current = self.inlet_temperature_k

        # Convert to kWh/kg
        compress_kwh_kg = w_compression_total * self.J_TO_KWH

        # Chilling energy
        q_chiller_j_kg = q_removed_total / self.chiller_cop
        self.chilling_energy_kwh_kg = q_chiller_j_kg * self.J_TO_KWH

        self.specific_energy_kwh_kg = compress_kwh_kg + self.chilling_energy_kwh_kg

        # Calculate actual energy for this step
        self.compression_work_kwh = compress_kwh_kg * self.actual_mass_transferred_kg
        self.chilling_work_kwh = self.chilling_energy_kwh_kg * self.actual_mass_transferred_kg

    def _calculate_compression_physics(self) -> None:
        """
        Calculate compression energy using real-gas thermodynamics.

        Implements multi-stage compression with intercooling using enthalpy
        and entropy from the LUT manager. Each stage follows:
        1. Isentropic compression: H2s = H(P_out, S_in)
        2. Actual work: W_actual = (H2s - H1) / η_is
        3. Actual outlet enthalpy: H2_actual = H1 + W_actual
        4. Intercooling: Q = H2_actual - H_cooled

        The total energy includes mechanical compression work plus chiller
        electrical consumption (Q_removed / COP).
        """
        lut = self.get_registry_safe(ComponentID.LUT_MANAGER)

        if lut is None:
            self._calculate_compression_fallback()
            return

        p_in_pa = self.inlet_pressure_bar * self.BAR_TO_PA
        p_out_pa = self.outlet_pressure_bar * self.BAR_TO_PA

        # Inlet properties
        h1 = lut.lookup('H2', 'H', p_in_pa, self.inlet_temperature_k)
        s1 = lut.lookup('H2', 'S', p_in_pa, self.inlet_temperature_k)

        r_total = p_out_pa / p_in_pa
        r_stage = r_total ** (1.0 / self.num_stages)

        # Accumulators
        w_compression_total = 0.0
        q_removed_total = 0.0
        p_current = p_in_pa

        t_stage_in = self.inlet_temperature_k

        for i in range(self.num_stages):
            # Stage inlet properties (recalculate after intercooling)
            s_stage_in = lut.lookup('H2', 'S', p_current, t_stage_in)
            h_stage_in = lut.lookup('H2', 'H', p_current, t_stage_in)

            # Stage outlet pressure
            p_out_stage = p_current * r_stage
            if i == self.num_stages - 1:
                p_out_stage = p_out_pa

            # Isentropic outlet enthalpy: H(P_out, S_in)
            try:
                h2s = lut.lookup_isentropic_enthalpy('H2', p_out_stage, s_stage_in)
            except Exception:
                if CoolPropLUT:
                    h2s = CoolPropLUT.PropsSI('H', 'P', p_out_stage, 'S', s_stage_in, 'H2')
                elif COOLPROP_AVAILABLE:
                    h2s = CP.PropsSI('H', 'P', p_out_stage, 'S', s_stage_in, 'H2')
                else:
                    h2s = h_stage_in * (p_out_stage/p_current)**0.28

            # Actual work accounting for efficiency
            ws = h2s - h_stage_in
            wa = ws / self.isentropic_efficiency
            h2a = h_stage_in + wa
            w_compression_total += wa

            # Intercooling (including aftercooling on final stage)
            h_cooled_next = lut.lookup('H2', 'H', p_out_stage, self.inlet_temperature_k)
            q_removed = h2a - h_cooled_next
            q_removed_total += q_removed

            # Update for next stage
            p_current = p_out_stage
            t_stage_in = self.inlet_temperature_k

        # Chilling work (heat removed / COP)
        w_chilling_total = q_removed_total / self.chiller_cop

        w_total_j_kg = w_compression_total + w_chilling_total

        self.specific_energy_kwh_kg = w_total_j_kg * self.J_TO_KWH

        # Calculate totals for this step
        self.compression_work_kwh = (w_compression_total * self.J_TO_KWH *
                                     self.actual_mass_transferred_kg)
        self.chilling_work_kwh = (w_chilling_total * self.J_TO_KWH *
                                  self.actual_mass_transferred_kg)
        self.heat_removed_kwh = (q_removed_total * self.J_TO_KWH *
                                 self.actual_mass_transferred_kg)

    # ========================================================================
    # Port Interface Methods
    # ========================================================================

    def get_output(self, port_name: str) -> Any:
        """
        Retrieve the output stream from a specified port.

        Returns compressed hydrogen at outlet pressure and inlet temperature
        (after final aftercooling).

        Args:
            port_name (str): Port identifier ('h2_out' or 'outlet').

        Returns:
            Stream: Compressed hydrogen stream with design outlet conditions.

        Raises:
            ValueError: If port_name is not a valid output port.
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

        Accumulates incoming hydrogen into transfer_mass_kg buffer, respecting
        maximum flow capacity.

        Args:
            port_name (str): Target port ('h2_in', 'inlet', or 'electricity_in').
            value (Any): Stream object for hydrogen or float for power.
            resource_type (str): Resource classification hint.

        Returns:
            float: Amount accepted (kg for hydrogen, value for power).
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
            Dict[str, Dict[str, str]]: Port definitions with keys:
                - h2_in: Low-pressure hydrogen feed.
                - electricity_in: Grid power for motor and chiller.
                - h2_out: High-pressure hydrogen product.
                - outlet: Alias for h2_out (legacy compatibility).
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
