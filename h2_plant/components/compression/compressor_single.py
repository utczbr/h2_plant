"""
Single-Stage Hydrogen Compressor.

This component models an adiabatic compression process using isentropic efficiency relations.
It is designed for hydrogen systems where accurate temperature rise prediction is critical 
for material limits and cooling requirements.

Physics Model (Isentropic Compression)
--------------------------------------
The compression process is modeled as an adiabatic reversible (isentropic) process corrected 
by an isentropic efficiency factor.

1.  **Ideal Path (Isentropic)**:
    Entropy remains constant ($ds = 0$).
    $$ S_{suction} = S(P_{suction}, T_{suction}) $$
    $$ H_{2s} = H(P_{discharge}, S_{suction}) $$

2.  **Real Path (Adiabatic with Irreversibilities)**:
    Irreversibilities increase discharge enthalpy relative to the ideal path.
    $$ W_{ideal} = H_{2s} - H_{suction} $$
    $$ W_{actual} = \frac{W_{ideal}}{\eta_{isentropic}} $$
    $$ H_{discharge} = H_{suction} + W_{actual} $$

3.  **Temperature Rise**:
    The final discharge temperature is determined from the fluid state at the discharge pressure 
    and actual enthalpy.
    $$ T_{discharge} = T(P_{discharge}, H_{discharge}) $$

Computational Strategy
----------------------
*   **Primary**: Numba-compiled Look-Up Tables (LUTs) for ~100x speedup in property queries.
*   **Fallback**: Real-gas EOS (Equation of State) via CoolProp (HEOS).
*   **Last Resort**: Ideal Gas Law ($PV=nRT$) if property databases are unavailable.

Operating Modes
---------------
*   **Fixed Pressure**: Standard mode. Compresses to `outlet_pressure_bar`.
*   **Temperature Limited**: Uses `max_temp_c` to iteratively solve for the maximum discharge 
    pressure that satisfies the constraint $T_{discharge} \le T_{max}$.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging

from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.component_ids import ComponentID
from h2_plant.core.enums import CompressorMode
from h2_plant.core.stream import Stream

try:
    import CoolProp.CoolProp as CP
    COOLPROP_AVAILABLE = True
except ImportError:
    CP = None
    COOLPROP_AVAILABLE = False

# Import JIT function at module level for performance
try:
    from h2_plant.optimization.numba_ops import calculate_compression_realgas_jit
    JIT_AVAILABLE = True
except ImportError:
    calculate_compression_realgas_jit = None
    JIT_AVAILABLE = False

logger = logging.getLogger(__name__)


class CompressorSingle(Component):
    """
    Simulates a single-stage adiabatic compressor.

    This component serves as the fundamental unit for pressure elevation in the
    hydrogen plant. It can be chained in series to form multi-stage trains with
    intercooling.

    Architecture:
        - **Component Lifecycle Contract (Layer 1)**: Fulfills standard initialization, stepping,
          and state reporting requirements.
        - **Optimization Layer (Layer 2)**: Supports optimized JIT execution via `numba_ops` 
          and utilizes `LUTManager` for thermodynamic property lookups.

    Attributes:
        max_flow_kg_h (float): Rated mass flow capacity [kg/h].
        inlet_pressure_bar (float): Suction pressure design point [bar].
        outlet_pressure_bar (float): Discharge pressure design point (or max cap) [bar].
        max_temp_c (float): Maximum allowable discharge temperature [°C]. 
            If set, enables temperature-limited operation.
        temperature_limited (bool): If True, dynamically calculates outlet pressure 
            to maintain `max_temp_c` constraint.
        isentropic_efficiency (float): Ratio of ideal to actual work ($\eta_s$) [0-1].
        mechanical_efficiency (float): Shaft/Bearing efficiency ($\eta_m$) [0-1].
        electrical_efficiency (float): Motor drive efficiency ($\eta_e$) [0-1].
    """


    def __init__(
        self,
        max_flow_kg_h: float,
        inlet_pressure_bar: float,
        outlet_pressure_bar: float,
        inlet_temperature_c: float = 25.0,
        isentropic_efficiency: float = 0.65,
        mechanical_efficiency: float = 0.96,
        electrical_efficiency: float = 0.93,
        max_temp_c: Optional[float] = None,
        temperature_limited: bool = False,
        max_pressure_bar: float = 1000.0
    ):
        """
        Configure the single-stage compressor.

        Args:
            max_flow_kg_h (float): Maximum mass flow rate [kg/h].
            inlet_pressure_bar (float): Suction pressure design point [bar].
            outlet_pressure_bar (float): Target discharge pressure [bar].
                In `temperature_limited` mode, this acts as the maximum cap.
            inlet_temperature_c (float): Suction temperature [°C]. Defaults to 25.0.
            isentropic_efficiency (float): Stage isentropic efficiency [0-1].
                Defaults to 0.65.
            mechanical_efficiency (float): Drive train mechanical efficiency [0-1].
                Defaults to 0.96.
            electrical_efficiency (float): Motor electrical efficiency [0-1].
                Defaults to 0.93.
            max_temp_c (Optional[float]): Maximum allowable discharge temperature [°C].
                If set with `temperature_limited=True`, the component will dynamically
                limit discharge pressure to satisfy $T_{out} \le T_{max}$. Defaults to None.
            temperature_limited (bool): If True, enables temperature-limited operation mode.
                Defaults to False.
            max_pressure_bar (float): Absolute maximum pressure cap [bar].
                Used as the upper bound in the binary search algorithm for temperature limiting.
                Defaults to 1000.0.
        """
        super().__init__()

        self.max_flow_kg_h = max_flow_kg_h
        self.inlet_pressure_bar = inlet_pressure_bar
        self.outlet_pressure_bar = outlet_pressure_bar
        self.target_outlet_pressure_bar = outlet_pressure_bar  # Original target

        self.inlet_temperature_c = inlet_temperature_c
        self.inlet_temperature_k = inlet_temperature_c + 273.15
        self.isentropic_efficiency = isentropic_efficiency
        self.mechanical_efficiency = mechanical_efficiency
        self.electrical_efficiency = electrical_efficiency
        
        # Temperature limiting
        self.max_temp_c = max_temp_c
        self.temperature_limited = temperature_limited
        self.max_pressure_bar = max_pressure_bar
        self.max_temp_k = (max_temp_c + 273.15) if max_temp_c is not None else None
        
        # Actual inlet pressure (updated from stream)
        self.actual_inlet_pressure_bar = inlet_pressure_bar
        self.actual_inlet_pressure_pa = inlet_pressure_bar * 1e5
        
        # Actual outlet pressure (may differ from target in temp-limited mode)
        self.actual_outlet_pressure_bar = outlet_pressure_bar
        self.actual_outlet_pressure_pa = outlet_pressure_bar * 1e5

        self.BAR_TO_PA = 1e5
        self.J_TO_KWH = 2.7778e-7

        self.transfer_mass_kg = 0.0

        self.actual_mass_transferred_kg = 0.0
        self.compression_work_kwh = 0.0
        self.energy_consumed_kwh = 0.0
        self.specific_energy_kwh_kg = 0.0

        self.outlet_temperature_c = inlet_temperature_c
        self.outlet_temperature_k = self.inlet_temperature_k
        self.outlet_temperature_isentropic_c = inlet_temperature_c

        self.power_kw = 0.0

        self.mode = CompressorMode.IDLE
        self.cumulative_energy_kwh = 0.0
        self.cumulative_mass_kg = 0.0

        self._last_step_time = -1.0
        self.timestep_power_kw = 0.0
        self.timestep_energy_kwh = 0.0
        
        # Store inlet stream for composition propagation
        self._inlet_stream: Optional[Stream] = None
        
        # LUT arrays for optimized compression (populated in initialize)
        self._lut_available: bool = False
        self._P_grid: Optional[np.ndarray] = None
        self._T_grid: Optional[np.ndarray] = None
        self._S_grid: Optional[np.ndarray] = None
        self._H_lut: Optional[np.ndarray] = None
        self._S_lut: Optional[np.ndarray] = None
        self._C_lut: Optional[np.ndarray] = None
        self._H_from_PS_lut: Optional[np.ndarray] = None

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """
        Executes initialization phase of Component Lifecycle.

        **Strategic Action**:
        Pre-fetches Numba-optimized Look-Up Tables (LUTs) from the `LUTManager` (Layer 2).
        Mapping these arrays to local variables enables the `_calculate_compression_lut`
        method to bypass Python object overhead during high-frequency stepping, crucial
        for simulation performance.

        Args:
            dt (float): Simulation timestep [hours].
            registry (ComponentRegistry): Central service registry for dependency injection.
        """
        super().initialize(dt, registry)

        pressure_ratio = self.outlet_pressure_bar / self.inlet_pressure_bar
        mode_str = "TEMP-LIMITED" if self.temperature_limited else "FIXED-PRESSURE"
        logger.info(
            f"CompressorSingle '{self.component_id}': "
            f"{self.inlet_pressure_bar:.1f} → {self.outlet_pressure_bar:.1f} bar "
            f"(ratio={pressure_ratio:.2f}), η_is={self.isentropic_efficiency:.2f}, "
            f"mode={mode_str}"
        )
        if self.temperature_limited and self.max_temp_c:
            logger.info(
                f"CompressorSingle '{self.component_id}': "
                f"Temperature limit: {self.max_temp_c:.1f}°C"
            )
        
        # Attempt to retrieve LUT arrays for optimized compression
        try:
            lut_manager = registry.get('lut_manager')
            if lut_manager is not None and hasattr(lut_manager, '_luts'):
                h2_luts = lut_manager._luts.get('H2', {})
                if 'H' in h2_luts and 'S' in h2_luts and 'C' in h2_luts and 'H_from_PS' in h2_luts:
                    self._P_grid = lut_manager._pressure_grid
                    self._T_grid = lut_manager._temperature_grid
                    self._S_grid = lut_manager._entropy_grid
                    self._H_lut = h2_luts['H']
                    self._S_lut = h2_luts['S']
                    self._C_lut = h2_luts['C']
                    self._H_from_PS_lut = h2_luts['H_from_PS']
                    self._lut_available = True
                    logger.info(f"CompressorSingle '{self.component_id}': LUT optimization enabled")
        except Exception as e:
            logger.debug(f"LUT retrieval failed: {e}. Using CoolProp fallback.")

    def step(self, t: float) -> None:
        """
        Executes the physics simulation step.

        **Process Logic**:
        1.  **Mass Transfer**: Accepts mass pushed by upstream or pulled by downstream (Pressure-driven flow abstraction).
        2.  **Mode Selection**: 
            -   **Compression**: $P_{out} > P_{in}$ (Standard operation).
            -   **Pass-Through**: $P_{out} \le P_{in}$ (Free flow/Check valve open).
        3.  **Work Calculation**: Computes energy required for the mass transfer based on thermodynamic path.
        4.  **Accumulation**: Updates cumulative energy/mass counters for solving system-wide mass/energy balance.

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

        if self.transfer_mass_kg > 0:
            self.mode = CompressorMode.LP_TO_HP

            max_transfer = self.max_flow_kg_h * self.dt
            self.actual_mass_transferred_kg = min(self.transfer_mass_kg, max_transfer)

            if self.outlet_pressure_bar <= self.actual_inlet_pressure_bar:
                self._calculate_trivial_pass_through()
            else:
                self._calculate_compression()

            # Check for discharge temperature limit (warning only, not enforced)
            if self.max_temp_c is not None and self.outlet_temperature_c > self.max_temp_c:
                if not self.temperature_limited:
                    # Only warn if not in temperature-limited mode
                    logger.warning(
                        f"Compressor {self.component_id} OVERHEAT: "
                        f"{self.outlet_temperature_c:.1f}°C > {self.max_temp_c:.1f}°C limit!"
                    )

            self.energy_consumed_kwh = self.compression_work_kwh

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

    def _calculate_trivial_pass_through(self) -> None:
        """
        Handle case where no compression is required (Check Valve logic).
        
        If $P_{out} \le P_{in}$, the compressor acts as a passive pipe, consuming zero energy.
        """
        self.compression_work_kwh = 0.0
        self.specific_energy_kwh_kg = 0.0
        self.outlet_temperature_k = self.inlet_temperature_k
        self.outlet_temperature_c = self.inlet_temperature_c
        self.outlet_temperature_isentropic_c = self.inlet_temperature_c
        self.actual_outlet_pressure_bar = self.actual_inlet_pressure_bar
        self.actual_outlet_pressure_pa = self.actual_inlet_pressure_pa

    def _get_fluid_properties(self) -> Tuple[str, float, float]:
        """
        Determine fluid properties based on inlet composition.
        
        Selects appropriate Equation of State (EOS) backend or ideal gas constants.

        Returns:
            Tuple[str, float, float]: 
                - **CoolProp_Fluid_Name** (str): Key for CoolProp EOS (e.g., 'Hydrogen', 'Oxygen').
                - **Cp_J_kg_K** (float): Specific heat capacity at constant pressure [$J/(kg \cdot K)$].
                - **Gamma** (float): Heat capacity ratio ($\gamma = C_p / C_v$).
        """
        # Default to Hydrogen if no stream
        if not self._inlet_stream:
            return 'Hydrogen', 14300.0, 1.41
            
        comp = self._inlet_stream.composition
        h2_frac = comp.get('H2', 0.0)
        o2_frac = comp.get('O2', 0.0)
        
        if o2_frac > h2_frac:
            # Oxygen properties
            # Cp ~ 918 J/kg*K, Gamma ~ 1.40
            return 'Oxygen', 918.0, 1.395 
        else:
            # Hydrogen properties (CoolProp uses 'Hydrogen' not 'H2')
            return 'Hydrogen', 14300.0, 1.41

    def _compute_outlet_temp(self, p_out_pa: float) -> float:
        """
        Compute discharge temperature for a given outlet pressure.
        
        This helper is used by the binary search solver to find the maximum
        pressure that keeps $T_{out} \le T_{max}$.
        
        Args:
            p_out_pa (float): Outlet pressure [Pa].
            
        Returns:
            float: Outlet temperature [K].
        """
        p_in_pa = self.actual_inlet_pressure_pa
        fluid_name, cp, gamma = self._get_fluid_properties()
        
        # Use LUTs for Hydrogen (fastest)
        if self._lut_available and fluid_name == 'Hydrogen' and JIT_AVAILABLE:
            from h2_plant.optimization.numba_ops import calculate_compression_realgas_jit
            try:
                _, T_out_k, _ = calculate_compression_realgas_jit(
                    p_in_pa,
                    p_out_pa,
                    self.inlet_temperature_k,
                    self.isentropic_efficiency,
                    self._P_grid,
                    self._T_grid,
                    self._S_grid,
                    self._H_lut,
                    self._S_lut,
                    self._C_lut,
                    self._H_from_PS_lut
                )
                return T_out_k
            except Exception as e:
                logger.debug(f"LUT compression failed: {e}. Trying CoolProp.")  # Fall through
        
        # CoolProp fallback
        if COOLPROP_AVAILABLE:
            try:
                h1 = CP.PropsSI('H', 'T', self.inlet_temperature_k, 'P', p_in_pa, fluid_name)
                s1 = CP.PropsSI('S', 'T', self.inlet_temperature_k, 'P', p_in_pa, fluid_name)
                h2s = CP.PropsSI('H', 'S', s1, 'P', p_out_pa, fluid_name)
                w_actual = (h2s - h1) / self.isentropic_efficiency
                h2_actual = h1 + w_actual
                T_out_k = CP.PropsSI('T', 'H', h2_actual, 'P', p_out_pa, fluid_name)
                return T_out_k
            except Exception as e:
                logger.debug(f"CoolProp compression failed: {e}. Using ideal gas.")  # Fall through
        
        # Ideal gas fallback
        pressure_ratio = p_out_pa / p_in_pa
        exponent = (gamma - 1.0) / gamma
        T_isen_k = self.inlet_temperature_k * (pressure_ratio ** exponent)
        delta_t_actual = (T_isen_k - self.inlet_temperature_k) / self.isentropic_efficiency
        return self.inlet_temperature_k + delta_t_actual

    def _solve_pressure_for_temp_limit(self) -> float:
        """
        Numerically solve for maximum outlet pressure satisfying temperature constraint.
        
        **Algorithm**:
        Solves the root finding problem $f(P) = T_{discharge}(P) - T_{max} = 0$ using 
        the Bisection Method (Binary Search) over the domain $[P_{in}, P_{target}]$.

        **Convergence Criteria**:
        - Iterates until relative pressure error $\frac{|P_{high} - P_{low}|}{P_{low}} < 0.1\%$.
        - Max iterations: 30 (guarantees convergence for monotonic $T(P)$).
        
        Returns:
            float: Maximum outlet pressure [Pa] that satisfies $T_{out} \le T_{max}$.
        """
        if self.max_temp_k is None:
            return self.outlet_pressure_bar * self.BAR_TO_PA
        
        p_in_pa = self.actual_inlet_pressure_pa
        
        # Check if inlet temperature already exceeds limit
        if self.inlet_temperature_k >= self.max_temp_k:
            logger.warning(
                f"Compressor {self.component_id}: Inlet temperature "
                f"{self.inlet_temperature_c:.1f}°C already at/above limit {self.max_temp_c:.1f}°C"
            )
            return p_in_pa  # No compression possible
        
        # Upper bound: min of target pressure and max_pressure_bar
        p_high = min(self.outlet_pressure_bar, self.max_pressure_bar) * self.BAR_TO_PA
        
        # Check if target pressure is achievable
        T_at_target = self._compute_outlet_temp(p_high)
        if T_at_target <= self.max_temp_k:
            return p_high  # Target pressure is achievable within temp limit
        
        # Binary search for maximum pressure
        p_low = p_in_pa
        
        for _ in range(30):  # Converges in ~20 iterations for 0.001% precision
            p_mid = (p_low + p_high) / 2.0
            T_mid = self._compute_outlet_temp(p_mid)
            
            if T_mid <= self.max_temp_k:
                p_low = p_mid
            else:
                p_high = p_mid
            
            # Converge when pressure difference is < 0.1%
            if abs(p_high - p_low) / p_low < 0.001:
                break
        
        return p_low

    def _calculate_compression(self) -> None:
        """
        Calculate single-stage adiabatic compression.

        **Logic Flow**:
        1.  **Pressure Determination**:
            -   If `temperature_limited`: Call `_solve_pressure_for_temp_limit()` to finding limiting discharge pressure.
            -   Else: Use fixed `outlet_pressure_bar`.
        2.  **Property Evaluation**:
            -   **LUT (Fast Path)**: If fluid is Hydrogen and LUTs available, invoke JIT kernel.
            -   **CoolProp (Precision)**: Fallback for other fluids or out-of-bounds states.
            -   **Ideal Gas (Backup)**: Approximate if EOS fails.
        """
        p_in_pa = self.actual_inlet_pressure_pa
        
        # Determine outlet pressure
        if self.temperature_limited and self.max_temp_c is not None:
            p_out_pa = self._solve_pressure_for_temp_limit()
            self.actual_outlet_pressure_pa = p_out_pa
            self.actual_outlet_pressure_bar = p_out_pa / self.BAR_TO_PA
        else:
            p_out_pa = self.outlet_pressure_bar * self.BAR_TO_PA
            self.actual_outlet_pressure_pa = p_out_pa
            self.actual_outlet_pressure_bar = self.outlet_pressure_bar
        
        fluid_name, _, _ = self._get_fluid_properties()

        # Only use LUTs for Hydrogen
        if self._lut_available and fluid_name == 'Hydrogen' and JIT_AVAILABLE:
            self._calculate_compression_lut(p_in_pa, p_out_pa)
        elif COOLPROP_AVAILABLE:
            self._calculate_compression_realgas(p_in_pa, p_out_pa)
        else:
            self._calculate_compression_idealgas(p_in_pa, p_out_pa)
    
    def _calculate_compression_lut(self, p_in_pa: float, p_out_pa: float) -> None:
        """
        Calculate compression using Numba JIT-compiled LUT lookups.

        **Performance Rationale**:
        Compressor calculation is the "inner loop" of the system optimization. 
        Using Numba-compiled interpolation on pre-loaded arrays (~0.1$\mu$s) vs CoolProp calls (~100$\mu$s) 
        provides a ~1000x speedup, enabling real-time optimization of complex compressor trains.

        **Method**:
        Invokes `calculate_compression_realgas_jit` from `numba_ops`, passing raw numpy arrays 
        to avoid Python-C API overhead.
        """
        try:
            # Call JIT function (imported at module level)
            w_specific, T_out_k, _ = calculate_compression_realgas_jit(
                p_in_pa,
                p_out_pa,
                self.inlet_temperature_k,
                self.isentropic_efficiency,
                self._P_grid,
                self._T_grid,
                self._S_grid,
                self._H_lut,
                self._S_lut,
                self._C_lut,
                self._H_from_PS_lut
            )
            
            # Check for LUT clamping at upper bound (1200 K)
            if T_out_k >= 1199.0:  # Close to 1200 K upper bound
                logger.warning(
                    f"Compressor {self.component_id}: Discharge temperature {T_out_k:.1f} K "
                    f"exceeds LUT upper bound. Result may be clamped!"
                )
            
            self.outlet_temperature_k = T_out_k
            self.outlet_temperature_c = T_out_k - 273.15
            
            # Estimate isentropic outlet temp (for diagnostics)
            gamma = 1.41
            exponent = (gamma - 1.0) / gamma
            T_isen_k = self.inlet_temperature_k * (p_out_pa / p_in_pa)**exponent
            self.outlet_temperature_isentropic_c = T_isen_k - 273.15
            
            # Drive efficiency
            drive_efficiency = self.mechanical_efficiency * self.electrical_efficiency
            w_electrical = w_specific / drive_efficiency
            
            self.specific_energy_kwh_kg = w_electrical * self.J_TO_KWH
            self.compression_work_kwh = self.specific_energy_kwh_kg * self.actual_mass_transferred_kg
            
        except Exception as e:
            logger.warning(f"LUT compression failed for {self.component_id}: {e}. Falling back to CoolProp.")
            if COOLPROP_AVAILABLE:
                self._calculate_compression_realgas(p_in_pa, p_out_pa)
            else:
                self._calculate_compression_idealgas(p_in_pa, p_out_pa)

    def _calculate_compression_realgas(self, p_in_pa: float, p_out_pa: float) -> None:
        """
        Calculates compression using CoolProp High-Precision EOS.

        **Fallback Strategy**:
        Invoked when LUTs are unavailable (e.g., non-Hydrogen fluids or out-of-bounds conditions). 
        Thermodynamically exact but computationally expensive.

        **Algorithm**:
        1.  Determine Inlet Entropy: $$ S_{in} = S(P_{in}, T_{in}) $$
        2.  Isentropic Flash: $$ H_{2s} = H(P_{out}, S_{in}) $$
        3.  Actual Work: $$ W_{actual} = \frac{H_{2s} - H_{in}}{\eta_{isen}} $$
        4.  Discharge Temperature: $$ T_{out} = T(P_{out}, H_{in} + W_{actual}) $$
        """
        fluid, _, _ = self._get_fluid_properties()

        h1 = CP.PropsSI('H', 'T', self.inlet_temperature_k, 'P', p_in_pa, fluid)
        s1 = CP.PropsSI('S', 'T', self.inlet_temperature_k, 'P', p_in_pa, fluid)

        t2s_k = CP.PropsSI('T', 'S', s1, 'P', p_out_pa, fluid)
        h2s = CP.PropsSI('H', 'S', s1, 'P', p_out_pa, fluid)

        self.outlet_temperature_isentropic_c = t2s_k - 273.15

        w_isentropic = h2s - h1
        w_actual = w_isentropic / self.isentropic_efficiency

        h2_actual = h1 + w_actual

        t2_k = CP.PropsSI('T', 'H', h2_actual, 'P', p_out_pa, fluid)

        self.outlet_temperature_k = t2_k
        self.outlet_temperature_c = t2_k - 273.15

        drive_efficiency = self.mechanical_efficiency * self.electrical_efficiency
        w_electrical = w_actual / drive_efficiency

        self.specific_energy_kwh_kg = w_electrical * self.J_TO_KWH
        self.compression_work_kwh = self.specific_energy_kwh_kg * self.actual_mass_transferred_kg

    def _calculate_compression_idealgas(self, p_in_pa: float, p_out_pa: float) -> None:
        """
        Calculates compression using Ideal Gas Law approximation.

        **Applicability**:
        Used as a last resort when property databases are missing. Accuracy degrades 
        significantly above 50 bar for Hydrogen due to compressibility ($Z$) deviating from 1.0.

        **Formula**:
        $$ T_{2s} = T_{in} \cdot \left(\frac{P_{out}}{P_{in}}\right)^{\frac{\gamma-1}{\gamma}} $$
        $$ \Delta T_{actual} = \frac{T_{2s} - T_{in}}{\eta_{isen}} $$
        $$ W_{actual} = C_p \cdot \Delta T_{actual} $$
        """
        _, cp, gamma = self._get_fluid_properties()

        pressure_ratio = p_out_pa / p_in_pa
        exponent = (gamma - 1) / gamma

        t2s_k = self.inlet_temperature_k * (pressure_ratio ** exponent)
        self.outlet_temperature_isentropic_c = t2s_k - 273.15

        delta_t_ideal = t2s_k - self.inlet_temperature_k
        delta_t_actual = delta_t_ideal / self.isentropic_efficiency

        t2_k = self.inlet_temperature_k + delta_t_actual
        self.outlet_temperature_k = t2_k
        self.outlet_temperature_c = t2_k - 273.15

        w_actual = cp * delta_t_actual

        drive_efficiency = self.mechanical_efficiency * self.electrical_efficiency
        w_electrical = w_actual / drive_efficiency

        self.specific_energy_kwh_kg = w_electrical * self.J_TO_KWH
        self.compression_work_kwh = self.specific_energy_kwh_kg * self.actual_mass_transferred_kg

    def get_state(self) -> Dict[str, Any]:
        """
        Retrieves component operational telemetry.

        **Layer 1 Contract Fulfillment**:
        Provides real-time state data for:
        -   **GUI Visualization**: Status, temperature, pressure.
        -   **Data Logging**: Energy consumption, mass throughput.
        -   **Orchestration**: System-wide energy balance verification.

        Returns:
            Dict[str, Any]: Key metrics including specific energy [kWh/kg] and efficiency factors.
        """
        cumulative_specific = 0.0
        if self.cumulative_mass_kg > 0:
            cumulative_specific = self.cumulative_energy_kwh / self.cumulative_mass_kg

        return {
            **super().get_state(),
            'mode': int(self.mode),
            'power_kw': float(self.power_kw),
            'transfer_mass_kg': float(self.transfer_mass_kg),
            'actual_mass_transferred_kg': float(self.actual_mass_transferred_kg),
            'outlet_o2_ppm_mol': (self._inlet_stream.get_total_mole_frac('O2') * 1e6) if self._inlet_stream else 0.0,
            'compression_work_kwh': float(self.compression_work_kwh),
            'energy_consumed_kwh': float(self.energy_consumed_kwh),
            'specific_energy_kwh_kg': float(self.specific_energy_kwh_kg),
            'outlet_temperature_c': float(self.outlet_temperature_c),
            'outlet_temperature_isentropic_c': float(self.outlet_temperature_isentropic_c),
            'cumulative_energy_kwh': float(self.cumulative_energy_kwh),
            'cumulative_mass_kg': float(self.cumulative_mass_kg),
            'timestep_energy_kwh': float(self.timestep_energy_kwh),
            'cumulative_specific_kwh_kg': float(cumulative_specific),
            'inlet_pressure_bar': float(self.actual_inlet_pressure_bar),
            'outlet_pressure_bar': float(self.actual_outlet_pressure_bar),
            'target_outlet_pressure_bar': float(self.target_outlet_pressure_bar),
            'inlet_temperature_c': float(self.inlet_temperature_c),
            'isentropic_efficiency': float(self.isentropic_efficiency),
            'mechanical_efficiency': float(self.mechanical_efficiency),
            'electrical_efficiency': float(self.electrical_efficiency),
            'temperature_limited': self.temperature_limited,
            'max_temp_c': self.max_temp_c
        }

    # =========================================================================
    # Port Interface Methods
    # =========================================================================

    def get_output(self, port_name: str) -> Any:
        """
        Retrieve the output stream from a specified port.

        **Physics**:
        Returns compressed hydrogen at the defined `outlet_pressure_bar` and the calculated 
        discharge temperature. Note that this component does **not** include an aftercooler; 
        discharge gas is hot ($T_{out} > T_{in}$).

        Args:
            port_name (str): Port identifier ('h2_out' or 'outlet').

        Returns:
            Stream: Object encapsulating mass flow, thermodynamic state ($P, T$), and composition.

        Raises:
            ValueError: If `port_name` is not recognized.
        """
        if port_name == 'h2_out' or port_name == 'outlet':
            # Propagate inlet composition (compression doesn't change composition)
            if self._inlet_stream and self._inlet_stream.composition:
                out_comp = self._inlet_stream.composition.copy()
                # Use inlet stream mass flow for output (compressor preserves mass)
                out_mass_flow = self._inlet_stream.mass_flow_kg_h
                # Propagate extra dict (contains entrained liquid info for mass balance)
                out_extra = self._inlet_stream.extra.copy() if self._inlet_stream.extra else {}
            else:
                out_comp = {'H2': 1.0}  # Default if no inlet stream
                out_mass_flow = (self.actual_mass_transferred_kg / self.dt
                               if self.dt > 0 else 0.0)
                out_extra = {}
            
            return Stream(
                mass_flow_kg_h=out_mass_flow,
                temperature_k=self.outlet_temperature_k,
                pressure_pa=self.actual_outlet_pressure_pa,
                composition=out_comp,
                phase='gas',
                extra=out_extra
            )
        else:
            raise ValueError(
                f"Unknown output port '{port_name}' on {self.component_id}"
            )

    def receive_input(self, port_name: str, value: Any, resource_type: str) -> float:
        """
        Accept an input stream at the specified port.

        **Logic**:
        -   **Hydrogen**: Accepts mass up to `max_flow_kg_h` limit. Prioritizes filling available capacity.
        -   **Electricity**: Acknowledges power availability (virtual connection).

        Args:
            port_name (str): Target port ('h2_in', 'inlet', or 'electricity_in').
            value (Any): Stream object (for mass) or power value (for electricity).
            resource_type (str): Resource classification hint from the graph traversal engine.

        Returns:
            float: Amount accepted (kg for hydrogen, arbitrary value for power).
        """
        if port_name == 'h2_in' or port_name == 'inlet' or port_name == 'gas_in':
            if isinstance(value, Stream):
                available_mass = value.mass_flow_kg_h * self.dt
                max_capacity = self.max_flow_kg_h * self.dt

                space_left = max(0.0, max_capacity - self.transfer_mass_kg)
                accepted_mass = min(available_mass, space_left)

                self.transfer_mass_kg += accepted_mass
                
                # Store inlet stream for composition and temperature propagation
                self._inlet_stream = value
                
                # Capture actual inlet pressure from stream
                if value.pressure_pa > 0:
                    self.actual_inlet_pressure_pa = value.pressure_pa
                    self.actual_inlet_pressure_bar = value.pressure_pa / self.BAR_TO_PA
                
                # Use inlet stream temperature for compression calculation
                if value.temperature_k > 0:
                    self.inlet_temperature_k = value.temperature_k
                    self.inlet_temperature_c = value.temperature_k - 273.15
                
                return accepted_mass

        elif port_name == 'electricity_in':
            return value if isinstance(value, (int, float)) else 0.0

        return 0.0

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """
        Define the physical connection ports for this component.

        **Topology Definition**:
        Declares inputs/outputs for the `PlantGraphBuilder` to construct the system node graph.

        Returns:
            Dict[str, Dict[str, str]]: Port definitions showing type, resource, and units.
        """
        return {
            'h2_in': {
                'type': 'input',
                'resource_type': 'hydrogen',
                'units': 'kg/h'
            },
            'inlet': {
                'type': 'input',
                'resource_type': 'hydrogen',
                'units': 'kg/h'
            },
            'electricity_in': {
                'type': 'input',
                'resource_type': 'electricity',
                'units': 'kW'
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
