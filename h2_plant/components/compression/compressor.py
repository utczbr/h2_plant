"""
Multi-Stage Hydrogen Compressor Component.

Models a multi-stage reciprocating compressor for hydrogen storage applications,
implementing staged compression with intercooling to minimize work input while
respecting discharge temperature limits.

Thermodynamic Model
-------------------
The compression process follows polytropic behavior bounded by isentropic and
isothermal limits. 

1.  **Stage Work**: Each stage allows adiabatic compression corrected by isentropic efficiency:
    $$ W_{actual} = \frac{H_{2s} - H_{in}}{\eta_{isen}} $$
    where $H_{2s} = H(P_{out}, S_{in})$.

2.  **Intercooling**: Between stages, gas is cooled back to $T_{inlet}$, reducing the specific volume 
    and total work for subsequent stages.
    $$ Q_{cool} = H_{actual, discharge} - H(P_{out}, T_{inlet}) $$

3.  **Drive Train**:
    $$ W_{electrical} = \frac{W_{shaft}}{\eta_{mech} \cdot \eta_{elec}} $$

Component Lifecycle Contract (Layer 1)
--------------------------------------
*   **initialize()**: Determines optimal $N_{stages}$ to satisfy $T_{discharge} \le T_{max}$ using 
    real-gas EOS.
*   **step()**: Computes compression work and cooling duty for time-variant mass flow.
*   **get_state()**: Exposes energy consumption, mass throughput, and stage configuration.
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

    Architecture:
        - **Component Lifecycle Contract (Layer 1)**: Self-configuring initialization (`initialize`) 
          and standardized state reporting.
        - **Thermodynamic Layer (Layer 2)**: Utilizes `LUTManager` for high-fidelity property 
          access during stage calculation and runtime stepping.

    Attributes:
        max_flow_kg_h (float): Rated mass flow capacity [kg/h].
        inlet_pressure_bar (float): Suction pressure design point [bar].
        outlet_pressure_bar (float): Discharge pressure design point [bar].
        num_stages (int): Calculated number of compression stages.
        stage_pressure_ratio (float): Pressure ratio per stage assuming equal distribution.
        isentropic_efficiency (float): Thermodynamic efficiency per stage ($\eta_s$) [0-1].
        mechanical_efficiency (float): Drive train mechanical efficiency ($\eta_m$) [0-1].
        electrical_efficiency (float): Motor electrical efficiency ($\eta_e$) [0-1].
        chiller_cop (float): Coefficient of Performance for intercooling.
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

        Stage configuration ($N_{stages}$) is computed during `initialize()` using real-gas
        properties to respect `max_temperature_c`.

        Args:
            max_flow_kg_h (float): Maximum mass flow capacity [kg/h].
            inlet_pressure_bar (float): Suction pressure design point [bar].
            outlet_pressure_bar (float): Target discharge pressure [bar].
            inlet_temperature_c (float): Suction/Interstage temperature [°C]. Defaults to 10.0.
            max_temperature_c (float): Maximum allowable discharge temperature per stage [°C].
                Defaults to 85.0.
            isentropic_efficiency (float): Stage isentropic efficiency [0-1]. Defaults to 0.65.
            mechanical_efficiency (float): Drive train mechanical efficiency [0-1]. Defaults to 0.96.
            electrical_efficiency (float): Motor electrical efficiency [0-1]. Defaults to 0.93.
            chiller_cop (float): Chiller Coefficient of Performance. Defaults to 3.0.
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

        **Strategic Action**:
        Calculates the optimal number of compression stages ($N$) during initialization to 
        safeguard against runtime temperature excursions. This static configuration 
        optimization is key to the "Design Once, Run Many" philosophy of the simulator.

        **Method**:
        Determines max pressure ratio per stage, then distributes total ratio equally:
        $$ N_{stages} = \lceil \frac{\ln(P_{target}/P_{suction})}{\ln(r_{max})} \rceil $$

        Args:
            dt (float): Simulation timestep [hours].
            registry (ComponentRegistry): Central component registry providing access to `LUTManager`.
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

        **Physics Sequence**:
        1.  **Mass Intake**: Limits transfer to available capacity.
        2.  **Compression Cycle**: Invokes `_calculate_compression_physics` (or fallbacks).
            -   Iterates through stages.
            -   Computes work ($W$) and heat rejection ($Q$).
        3.  **State Update**: Accumulates energy counters for integration.

        Args:
            t (float): Current simulation time [hours].
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

        **Layer 1 Contract**:
        Returns standardized telemetry for orchestrator.

        Returns:
            Dict[str, Any]: 
            -   **mode** (int): Operating mode (IDLE=0, LP_TO_HP=1).
            -   **num_stages** (int): Configured stage count ($N$).
            -   **specific_energy_kwh_kg** (float): Total energy intensity.
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
        Determine optimal number of compression stages using Real Gas EOS.

        **Algorithm**:
        1.  **Entropy Lookup**: $S_{in} = S(P_{in}, T_{in})$ via LUTManager.
        2.  **Isentropic Limit**: Find pressure $P$ where isentropic compression reaches $T_{max}$:
            $$ P(S_{in}, T_{max}) = \text{InverseProps}(S_{in}, T_{max}) $$
        3.  **Max Ratio**: $r_{max} = P / P_{in}$ (floored at 2.0).
        4.  **Stage Count**: 
            $$ N = \lceil \frac{\ln(P_{target}/P_{suction})}{\ln(r_{max})} \rceil $$
        5.  **Actual Ratio**: $r_{actual} = (P_{target}/P_{suction})^{1/N}$.
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
        
        **Logic**:
        Check valve open ($P_{out} \le P_{in}$). Zero energy operation.
        """
        self.compression_work_kwh = 0.0
        self.chilling_work_kwh = 0.0
        self.specific_energy_kwh_kg = 0.0
        self.heat_removed_kwh = 0.0

    def _calculate_stages_fallback(self) -> None:
        """
        Calculate stage configuration using ideal gas assumptions (Safe Mode).

        **Logic**:
        Defaults to a conservative fixed maximum stage ratio of 4.0, typical for
        hydrogen compressors to stay within temperature limits without real-gas data.
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

        **Approximation**:
        Uses the isentropic ideal gas temperature-pressure relationship:
        $$ T_{2s} = T_{in} \cdot \left(\frac{P_{out}}{P_{in}}\right)^{\frac{\gamma-1}{\gamma}} $$
        With Hydrogen constants: $C_p \approx 14.3 \text{ kJ/kgK}$, $\gamma = 1.41$.

        **Work**:
        $$ W = C_p \cdot (T_{2s} - T_{in}) / \eta_{isen} $$
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
        Calculate compression energy using real-gas thermodynamics (Step Logic).

        **Loop Invariant**:
        For each stage $i \in [0, N-1]$:
        1.  **Isentropic Compression**:
            $$ H_{2s} = H(P_{out,i}, S_{in,i}) $$
            Where $S_{in,i} = S(P_{in,i}, T_{inlet})$ due to intercooling.
        2.  **Work**:
            $$ W_{shaft,i} = \frac{H_{2s} - H_{in,i}}{\eta_{isen}} $$
        3.  **Real State**:
            $$ H_{real,i} = H_{in,i} + W_{shaft,i} $$
        4.  **Intercooling Duty**:
            $$ Q_{cool,i} = H_{real,i} - H(P_{out,i}, T_{inlet}) $$
            (Assumes perfect intercooling back to $T_{inlet}$).
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

        **Physics**:
        Returns hydrogen at outlet pressure and *inlet temperature* (stage intercooling + aftercooling).
        Assumes perfect heat rejection to $T_{in}$.

        Args:
            port_name (str): 'h2_out' or 'outlet'.

        Returns:
            Stream: Product stream with $\{P_{out}, T_{in}, 100\% H_2\}$.

        Raises:
            ValueError: If `port_name` unknown.
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

        **Interface**:
        -   **Hydrogen**: Buffers mass up to max capacity.
        -   **Electricity**: Virtual power connection.

        Args:
            port_name (str): 'h2_in', 'inlet', 'electricity_in'.
            value (Any): Stream or float.
            resource_type (str): Hint.

        Returns:
            float: Accepted amount.
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
            Dict[str, Dict[str, str]]:
            -   **h2_in**: Low-pressure hydrogen feed.
            -   **h2_out**: High-pressure product.
            -   **electricity_in**: Drive power (MW).
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
