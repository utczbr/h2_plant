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
    from h2_plant.optimization.numba_ops import (
        calculate_compression_realgas_jit, 
        calculate_mixture_compression_jit,
        solve_temp_limited_pressure_jit
    )
    JIT_AVAILABLE = True
except ImportError:
    calculate_compression_realgas_jit = None
    calculate_mixture_compression_jit = None
    solve_temp_limited_pressure_jit = None
    JIT_AVAILABLE = False

# Import mixture thermodynamics module  
try:
    from h2_plant.optimization import mixture_thermodynamics as mix_thermo
    MIX_THERMO_AVAILABLE = True
except ImportError:
    mix_thermo = None
    MIX_THERMO_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# OFF-DESIGN PERFORMANCE CONFIGURATION
# =============================================================================

from dataclasses import dataclass
from enum import Enum, auto


class CompressorType(Enum):
    """
    Compressor Machine Type - determines which physics model is used.
    
    CENTRIFUGAL: Dynamic compression (kinetic energy conversion).
        - Used for: Steam MRV, Oxygen, Air compressors
        - Off-design: Surge/Choke limits based on corrected flow
        - Reference: ASME GT2023-87110
    
    RECIPROCATING: Positive displacement (volumetric).
        - Used for: Hydrogen, Biogas compressors
        - Off-design: Volumetric efficiency vs pressure ratio
        - Reference: Int. J. Hydrogen Energy 2024
    """
    CENTRIFUGAL = auto()
    RECIPROCATING = auto()


@dataclass
class CompressorMapConfig:
    """
    Configuration for off-design performance physics.
    
    This dataclass unifies parameters for both Centrifugal and Reciprocating
    compressor models. The `type` field determines which physics engine is used.
    
    Attributes:
        type (CompressorType): Machine type - CENTRIFUGAL or RECIPROCATING.
        
        --- VFD Parameters (Both Types) ---
        allow_variable_speed (bool): Enable VFD speed control.
        min_speed_ratio (float): Minimum RPM as fraction of design (e.g., 0.40 = 40%).
        max_speed_ratio (float): Maximum RPM as fraction of design (e.g., 1.05 = 105%).
        
        --- Centrifugal Specifics ---
        design_polytropic_eff (float): Peak polytropic efficiency at design point.
        surge_margin (float): Minimum allowable corrected flow fraction.
        choke_margin (float): Maximum allowable corrected flow fraction.
        curve_shape_k (float): Steepness of efficiency parabola.
        
        --- Reciprocating Specifics ---
        clearance_volume_fraction (float): Clearance volume as fraction of stroke (C).
        valve_loss_factor (float): Pressure drop losses across valves.
        mechanical_loss_factor (float): Friction losses in crank/piston.
    
    References:
        [1] ASME GT2023-87110: Off-Design Performance of Hydrogen Compressors
        [2] Int. J. Hydrogen Energy 2024: Volumetric Efficiency in H2 Reciprocating
        [3] API 617: Centrifugal Compressors
        [4] API 618: Reciprocating Compressors
    """
    # Machine Type
    type: CompressorType = CompressorType.CENTRIFUGAL
    
    # VFD Parameters (shared)
    allow_variable_speed: bool = True
    min_speed_ratio: float = 0.40
    max_speed_ratio: float = 1.05
    
    # Centrifugal Specifics
    design_polytropic_eff: float = 0.78
    surge_margin: float = 0.20
    choke_margin: float = 1.30
    curve_shape_k: float = 2.5
    
    # Reciprocating Specifics
    # eta_vol = 1 - C * (PR^(1/gamma) - 1) - valve_losses
    clearance_volume_fraction: float = 0.10   # 10% clearance (typical)
    valve_loss_factor: float = 0.05           # Valve pressure drop losses
    mechanical_loss_factor: float = 0.04      # Crank/piston friction


# =============================================================================
# FLUID STRATEGY SELECTION
# =============================================================================

from enum import Enum, auto

class FluidStrategy(Enum):
    """
    Thermodynamic calculation strategy selector.
    
    Determines which computational path is used for compression calculations:
    - LUT: Pure fluid with pre-computed Lookup Tables (~1 µs per call)
    - LUT_MIXTURE: Multi-component using Ideal Mixing of Real Gases (~6 µs per call)
    - COOLPROP_MIXTURE: Multi-component HEOS backend (~100 µs per call)  
    - IDEAL_GAS: Fallback with mixture-weighted constants
    """
    LUT = auto()
    LUT_MIXTURE = auto()
    COOLPROP_MIXTURE = auto()
    IDEAL_GAS = auto()


# Reference thermodynamic data for supported species
# Format: (CoolProp_Name, Molar_Mass_kg/mol, Cp_J/kg·K, gamma)
FLUID_REF_DATA: Dict[str, Tuple[str, float, float, float]] = {
    'H2':  ('Hydrogen',       2.016e-3,  14304.0, 1.41),
    'O2':  ('Oxygen',        32.00e-3,    918.0, 1.39),
    'N2':  ('Nitrogen',      28.01e-3,   1040.0, 1.40),
    'CH4': ('Methane',       16.04e-3,   2226.0, 1.32),
    'CO2': ('CarbonDioxide', 44.01e-3,    846.0, 1.30),
    'H2O': ('Water',         18.02e-3,   1850.0, 1.33),
}


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
        max_pressure_bar: float = 1000.0,
        map_config: Optional[CompressorMapConfig] = None
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
        self.design_isentropic_eff = isentropic_efficiency  # Original design efficiency
        self.current_isentropic_eff = isentropic_efficiency  # Dynamic (updated each step)
        self.current_polytropic_eff = 0.0  # Computed from map
        self.mechanical_efficiency = mechanical_efficiency
        self.electrical_efficiency = electrical_efficiency
        
        # Design point for corrected flow calculation
        self.design_flow_kg_h = max_flow_kg_h
        self.design_inlet_p_bar = inlet_pressure_bar
        self.design_inlet_t_k = inlet_temperature_c + 273.15
        
        # Off-design performance map configuration
        self.use_performance_map = map_config is not None
        self.map_config = map_config or CompressorMapConfig()
        
        # Surge/Choke event counters
        self.surge_events = 0
        self.choke_events = 0
        
        # Reciprocating compressor telemetry
        self.current_volumetric_eff = 1.0  # η_vol (only used for RECIPROCATING type)
        
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
        
        # Multi-fluid LUT storage for optimized compression (populated in initialize)
        # Dict structure: {fluid_key: {'H': arr, 'S': arr, 'C': arr, 'H_from_PS': arr}}
        self._luts: Dict[str, Dict[str, np.ndarray]] = {}
        self._lut_manager = None  # Reference to LUTManager for mixture calculations
        self._P_grid: Optional[np.ndarray] = None
        self._T_grid: Optional[np.ndarray] = None
        self._S_grid: Optional[np.ndarray] = None
        
        # Stacked LUTs for JIT mixture calculations
        self._H_luts_stacked: Optional[np.ndarray] = None
        self._S_luts_stacked: Optional[np.ndarray] = None
        self._C_luts_stacked: Optional[np.ndarray] = None
        self._fluid_indices: Dict[str, int] = {}
        self._molar_masses_arr: Optional[np.ndarray] = None
        
        # Cache for JIT temp-limited solver result (P_out, T_out, W_actual)
        self._jit_cached_result: Optional[Tuple[float, float, float]] = None

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
        
        # Retrieve multi-fluid LUT arrays for optimized compression
        try:
            lut_manager = registry.get('lut_manager')
            if lut_manager is not None and hasattr(lut_manager, '_luts'):
                self._lut_manager = lut_manager  # Store reference for mixture calcs
                self._P_grid = lut_manager._pressure_grid
                self._T_grid = lut_manager._temperature_grid
                self._S_grid = lut_manager._entropy_grid
                
                # Load LUTs for all supported fluids
                for fluid_key in FLUID_REF_DATA.keys():
                    fluid_data = lut_manager._luts.get(fluid_key, {})
                    if all(k in fluid_data for k in ('H', 'S', 'C', 'H_from_PS')):
                        self._luts[fluid_key] = fluid_data
                
                if self._luts:
                    logger.info(
                        f"CompressorSingle '{self.component_id}': "
                        f"LUT optimization enabled for {list(self._luts.keys())}"
                    )
                    
                    # Prepare Stacked LUTs for JIT Mixture (Zero-Copy Reference)
                    try:
                        if hasattr(self._lut_manager, 'stacked_H') and self._lut_manager.stacked_H is not None:
                             self._H_luts_stacked = self._lut_manager.stacked_H
                             self._S_luts_stacked = self._lut_manager.stacked_S
                             self._C_luts_stacked = self._lut_manager.stacked_C
                             logger.debug(f"CompressorSingle '{self.component_id}': Linked to global stacked LUTs")
                        else:
                             logger.warning(f"CompressorSingle '{self.component_id}': Global stacked LUTs not available")
                    except Exception as stack_err:
                        logger.warning(f"Failed to link LUTs for JIT mixture: {stack_err}")
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
            
            # Calculate raw mass flow rate for performance map lookup
            raw_mass_flow_rate = (self.transfer_mass_kg / self.dt) if self.dt > 0 else 0.0

            # ============================================================
            # OFF-DESIGN PERFORMANCE (if map_config was provided)
            # ============================================================
            eta_poly, limited_flow_rate, is_surge, is_choke = self._update_off_design_performance(
                raw_mass_flow_rate
            )
            
            # Track surge/choke events
            if is_surge:
                self.surge_events += 1
                if self.surge_events % 10 == 0:
                    logger.warning(
                        f"Compressor {self.component_id} in SURGE! "
                        f"Flow: {raw_mass_flow_rate:.1f} kg/h"
                    )
            
            if is_choke:
                self.choke_events += 1
                if self.choke_events == 1:
                    logger.warning(
                        f"Compressor {self.component_id} at CHOKE limit! "
                        f"Clamping flow to {limited_flow_rate:.1f} kg/h"
                    )
                # Enforce choke limit on mass transfer
                self.actual_mass_transferred_kg = min(limited_flow_rate * self.dt, self.transfer_mass_kg)
            else:
                max_transfer = self.max_flow_kg_h * self.dt
                self.actual_mass_transferred_kg = min(self.transfer_mass_kg, max_transfer)

            # ============================================================
            # DYNAMIC EFFICIENCY UPDATE
            # ============================================================
            if self.use_performance_map:
                # Get pressure ratio for conversion
                pr = max(1.0, self.target_outlet_pressure_bar / max(0.1, self.actual_inlet_pressure_bar))
                gamma = self._get_fluid_gamma()
                
                # Store polytropic and convert to isentropic
                self.current_polytropic_eff = eta_poly
                self.current_isentropic_eff = self._convert_poly_to_isentropic(eta_poly, pr, gamma)
                
                # Temporarily apply dynamic efficiency for thermodynamic calculation
                original_eff = self.isentropic_efficiency
                self.isentropic_efficiency = self.current_isentropic_eff
            else:
                self.current_isentropic_eff = self.design_isentropic_eff
                original_eff = None  # No need to restore

            # ============================================================
            # THERMODYNAMIC CALCULATION
            # ============================================================
            if self.outlet_pressure_bar <= self.actual_inlet_pressure_bar:
                self._calculate_trivial_pass_through()
            else:
                self._calculate_compression()
            
            # Restore design efficiency if we modified it
            if self.use_performance_map and original_eff is not None:
                self.isentropic_efficiency = original_eff

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

    # =========================================================================
    # OFF-DESIGN PERFORMANCE MAP METHODS
    # =========================================================================

    def _update_off_design_performance(self, mass_flow_rate_kg_h: float) -> Tuple[float, float, bool, bool]:
        """
        Calculate dynamic efficiency and operating limits based on compressor type.
        
        Routes to the appropriate physics model:
        - CENTRIFUGAL: Corrected flow, surge/choke limits, polytropic efficiency map
        - RECIPROCATING: Volumetric efficiency, clearance volume, pressure ratio limits
        
        Args:
            mass_flow_rate_kg_h: Actual mass flow rate [kg/h]
        
        Returns:
            Tuple: (isentropic_eff, limited_mass_flow_kg_h, is_surge, is_choke)
        """
        if not self.use_performance_map or mass_flow_rate_kg_h < 1e-6:
            return self.design_isentropic_eff, mass_flow_rate_kg_h, False, False

        # Route to appropriate physics model
        if self.map_config.type == CompressorType.RECIPROCATING:
            return self._calc_reciprocating_performance(mass_flow_rate_kg_h)
        else:
            return self._calc_centrifugal_performance(mass_flow_rate_kg_h)

    def _calc_centrifugal_performance(self, mass_flow_rate_kg_h: float) -> Tuple[float, float, bool, bool]:
        """
        Centrifugal compressor off-design physics (corrected flow, surge/choke).
        
        Physics Model (Ref: ASME GT2023-87110):
            1. Corrected Flow: m_corr = m_act * sqrt(T_in/T_des) / (P_in/P_des)
            2. Surge: m_corr_norm < surge_margin
            3. Choke: m_corr_norm > choke_margin (flow clamped)
            4. Polytropic Efficiency: η_p = η_peak * [1 - k * (m_norm - 1)²]
        
        Returns:
            Tuple: (polytropic_eff, limited_mass_flow_kg_h, is_surge, is_choke)
        """
        # 1. Calculate Corrected Mass Flow
        theta = (self.inlet_temperature_k / self.design_inlet_t_k)
        delta = max(1e-3, self.actual_inlet_pressure_bar / self.design_inlet_p_bar)
        
        m_corr = mass_flow_rate_kg_h * np.sqrt(theta) / delta
        m_corr_norm = m_corr / self.design_flow_kg_h

        # 2. Check Surge / Choke Limits
        is_surge = False
        is_choke = False
        limit_mass_flow = mass_flow_rate_kg_h

        if m_corr_norm < self.map_config.surge_margin:
            is_surge = True
        elif m_corr_norm > self.map_config.choke_margin:
            is_choke = True
            m_corr_limit = self.map_config.choke_margin * self.design_flow_kg_h
            limit_mass_flow = m_corr_limit * delta / np.sqrt(theta)

        # 3. Calculate Polytropic Efficiency (Parabolic fit)
        penalty = self.map_config.curve_shape_k * (m_corr_norm - 1.0)**2
        eta_poly = self.map_config.design_polytropic_eff * (1.0 - penalty)
        eta_poly = max(0.20, eta_poly)

        return eta_poly, limit_mass_flow, is_surge, is_choke

    def _calc_reciprocating_performance(self, req_flow_kg_h: float) -> Tuple[float, float, bool, bool]:
        """
        Reciprocating (positive displacement) compressor off-design physics.
        
        Physics Model (Ref: Int. J. Hydrogen Energy 2024):
            1. Volumetric Efficiency: η_vol = 1 - C * (PR^(1/γ) - 1) - valve_losses
            2. Capacity scales with η_vol, inlet density, and RPM
            3. No aerodynamic surge (PD machine), but capacity-limited at high PR
        
        Args:
            req_flow_kg_h: Requested mass flow rate [kg/h]
        
        Returns:
            Tuple: (isentropic_eff, max_flow_capacity_kg_h, is_surge=False, is_capacity_limited)
        """
        # 1. Pressure Ratio
        p_in = max(0.1, self.actual_inlet_pressure_bar)
        p_out = max(p_in, self.target_outlet_pressure_bar)
        pr = p_out / p_in
        
        # 2. Get Gas Properties (gamma)
        gamma = self._get_fluid_gamma()
        
        # 3. Calculate Volumetric Efficiency
        # η_vol = 1 - C * (PR^(1/γ) - 1) - valve_losses
        C = self.map_config.clearance_volume_fraction
        
        if pr < 1.0:
            pr = 1.0
        
        re_expansion = (pr**(1.0/gamma)) - 1.0
        eta_vol = 1.0 - C * re_expansion - self.map_config.valve_loss_factor
        eta_vol = max(0.0, eta_vol)  # Cannot be negative (deadhead)
        
        # Store for telemetry
        self.current_volumetric_eff = eta_vol
        
        # 4. Calculate Maximum Capacity at this PR
        # Capacity = Design_Flow * (η_vol / η_vol_design) * density_correction * speed_factor
        pr_design = self.outlet_pressure_bar / max(0.1, self.design_inlet_p_bar) if hasattr(self, 'outlet_pressure_bar') else pr
        eta_vol_design = 1.0 - C * ((pr_design**(1.0/gamma)) - 1.0) - self.map_config.valve_loss_factor
        eta_vol_design = max(0.01, eta_vol_design)  # Protect against div/0
        
        # Density correction (recip moves volume, not mass)
        density_correction = (p_in / self.design_inlet_p_bar) * (self.design_inlet_t_k / self.inlet_temperature_k)
        
        # VFD speed range
        max_rpm_factor = self.map_config.max_speed_ratio
        
        max_possible_flow = self.design_flow_kg_h * (eta_vol / eta_vol_design) * density_correction * max_rpm_factor
        
        # 5. Check if capacity-limited
        is_capacity_limited = req_flow_kg_h > max_possible_flow
        limit_flow = max_possible_flow
        
        # 6. Isentropic Efficiency (relatively flat for recip, slight degradation at high PR)
        pr_design_ref = self.outlet_pressure_bar / self.design_inlet_p_bar if hasattr(self, 'outlet_pressure_bar') else 5.0
        pr_deviation = abs(pr - pr_design_ref)
        eta_is = self.design_isentropic_eff * (1.0 - 0.01 * pr_deviation)
        eta_is = max(0.50, min(0.90, eta_is))  # Recip efficiency bounds
        
        # Reciprocating compressors don't surge (no aerodynamic instability)
        return eta_is, limit_flow, False, is_capacity_limited

    def _convert_poly_to_isentropic(self, eta_poly: float, pressure_ratio: float, gamma: float = 1.41) -> float:
        """
        Convert Polytropic Efficiency to Isentropic Efficiency.
        
        This is required because the thermodynamic calculations use isentropic
        efficiency, but performance maps are typically expressed in polytropic
        efficiency (which is more stable across pressure ratios).
        
        Formula (Ref: ASME GT2023-87110):
            η_is = (PR^((k-1)/k) - 1) / (PR^((k-1)/(k·η_p)) - 1)
        
        Args:
            eta_poly: Polytropic efficiency [0-1]
            pressure_ratio: Discharge/Suction pressure ratio
            gamma: Ratio of specific heats (k = Cp/Cv)
        
        Returns:
            float: Isentropic efficiency [0-1]
        """
        if pressure_ratio <= 1.0 + 1e-4:
            return 1.0  # Trivial case: no compression
            
        k_term = (gamma - 1.0) / gamma
        num = pressure_ratio**k_term - 1.0
        den = pressure_ratio**(k_term / eta_poly) - 1.0
        
        if abs(den) < 1e-9:
            return eta_poly  # Avoid division by zero
        return num / den

    def _get_fluid_gamma(self) -> float:
        """Get gamma (Cp/Cv) for the dominant fluid species."""
        strategy, fluid_id, mix_const = self._determine_fluid_strategy()
        if strategy == FluidStrategy.LUT and fluid_id in FLUID_REF_DATA:
            return FLUID_REF_DATA[fluid_id][3]  # gamma from reference data
        elif mix_const and 'gamma' in mix_const:
            return mix_const['gamma']
        return 1.41  # Default H2 gamma

    def _determine_fluid_strategy(self) -> Tuple[FluidStrategy, str, Dict[str, float]]:
        """
        Analyze inlet composition and select thermodynamic calculation strategy.
        
        **Strategy Selection Logic (Priority Order):**
        1. Pure Fluid (≥98.0% single species): Use LUT for maximum performance
        2. LUT_MIXTURE: All species have LUTs → Ideal Mixing of Real Gases
        3. COOLPROP_MIXTURE: CoolProp HEOS for rigorous mixing (slower)
        4. IDEAL_GAS: Fallback with mixture-weighted constants
        
        Returns:
            Tuple[FluidStrategy, str, Dict]:
                - strategy: The selected FluidStrategy enum value
                - fluid_id: Pure fluid key, 'LUT_MIX' marker, HEOS string, or empty
                - mix_constants: Dict with 'cp' and 'gamma' for ideal gas path
        """
        # Default to hydrogen if no stream data
        if not self._inlet_stream or not self._inlet_stream.composition:
            if 'H2' in self._luts:
                return FluidStrategy.LUT, 'H2', {}
            return FluidStrategy.IDEAL_GAS, '', {'cp': 14304.0, 'gamma': 1.41}
        
        comp = self._inlet_stream.composition
        
        # Find dominant species and check LUT coverage
        max_frac = 0.0
        dominant = 'H2'
        all_have_luts = True
        active_species = []
        
        for species, frac in comp.items():
            if species in ('H2O_liq',) or frac < 1e-6:
                continue
            active_species.append(species)
            if species in FLUID_REF_DATA and frac > max_frac:
                max_frac = frac
                dominant = species
            if species not in self._luts:
                all_have_luts = False
        
        # Pure fluid path (≥98.0% single species)
        # PERFORMANCE: Relaxed from 99.9% to 98.0% to allow near-pure streams (e.g. wet H2)
        # to use the fast LUT path instead of the expensive mixture solver.
        if max_frac >= 0.980 and dominant in self._luts:
            return FluidStrategy.LUT, dominant, {}
        
        # LUT Mixture path: all species have LUTs AND mix_thermo available
        if all_have_luts and MIX_THERMO_AVAILABLE and self._lut_manager is not None:
            return FluidStrategy.LUT_MIXTURE, 'LUT_MIX', {}
        
        # CoolProp Mixture path
        if COOLPROP_AVAILABLE:
            parts = []
            for species, frac in comp.items():
                if species in FLUID_REF_DATA and frac > 1e-6:
                    coolprop_name = FLUID_REF_DATA[species][0]
                    parts.append(f"{coolprop_name}[{frac}]")
            if parts:
                mixture_str = "HEOS::" + "&".join(parts)
                return FluidStrategy.COOLPROP_MIXTURE, mixture_str, {}
        
        # Ideal gas fallback with mixture-weighted constants
        mix_const = self._calculate_mixture_constants(comp)
        return FluidStrategy.IDEAL_GAS, '', mix_const
    
    def _calculate_mixture_constants(self, composition: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate mixture-averaged thermodynamic constants using mass-weighted mixing rules.
        
        **Mixing Rules:**
        - Molar mass: $M_{mix} = \\sum_i x_i M_i$
        - Mass fractions: $w_i = \\frac{x_i M_i}{M_{mix}}$
        - Mixture Cp: $C_{p,mix} = \\sum_i w_i C_{p,i}$
        - Mixture gamma: $\\gamma_{mix} = \\frac{C_{p,mix}}{C_{p,mix} - R_{mix}}$
        
        Args:
            composition: Dict of species mole fractions {species: x_i}
            
        Returns:
            Dict with 'cp' (J/kg·K) and 'gamma' (dimensionless)
        """
        R_UNIVERSAL = 8.314  # J/(mol·K)
        
        # Calculate mixture molar mass
        M_mix = 0.0
        for species, x_i in composition.items():
            if species in FLUID_REF_DATA and x_i > 1e-9:
                M_i = FLUID_REF_DATA[species][1]  # kg/mol
                M_mix += x_i * M_i
        
        if M_mix < 1e-9:
            # Fallback to hydrogen if composition is invalid
            return {'cp': 14304.0, 'gamma': 1.41}
        
        # Calculate mass fractions and mixture Cp
        Cp_mix = 0.0
        for species, x_i in composition.items():
            if species in FLUID_REF_DATA and x_i > 1e-9:
                M_i = FLUID_REF_DATA[species][1]
                Cp_i = FLUID_REF_DATA[species][2]  # J/kg·K
                w_i = (x_i * M_i) / M_mix  # Mass fraction
                Cp_mix += w_i * Cp_i
        
        # Mixture gas constant (specific)
        R_mix = R_UNIVERSAL / M_mix  # J/(kg·K)
        
        # Mixture gamma: γ = Cp / (Cp - R)
        Cv_mix = Cp_mix - R_mix
        gamma_mix = Cp_mix / Cv_mix if Cv_mix > 0 else 1.4
        
        return {'cp': Cp_mix, 'gamma': gamma_mix}
    
    def _get_fluid_properties(self) -> Tuple[str, float, float]:
        """
        Legacy method for backward compatibility.
        
        Returns CoolProp name, Cp, and gamma for the dominant fluid.
        Delegates to _determine_fluid_strategy for actual logic.
        """
        strategy, fluid_id, mix_const = self._determine_fluid_strategy()
        
        if strategy == FluidStrategy.LUT:
            # Return CoolProp name and properties for the pure fluid
            coolprop_name, _, cp, gamma = FLUID_REF_DATA[fluid_id]
            return coolprop_name, cp, gamma
        elif strategy == FluidStrategy.COOLPROP_MIXTURE:
            # For mixtures, return dominant fluid properties (approximate)
            if self._inlet_stream and self._inlet_stream.composition:
                dominant = max(self._inlet_stream.composition.items(), 
                              key=lambda x: x[1] if x[0] in FLUID_REF_DATA else 0)
                if dominant[0] in FLUID_REF_DATA:
                    coolprop_name, _, cp, gamma = FLUID_REF_DATA[dominant[0]]
                    return coolprop_name, cp, gamma
            return 'Hydrogen', 14304.0, 1.41
        else:
            # Ideal gas path
            return 'Hydrogen', mix_const.get('cp', 14304.0), mix_const.get('gamma', 1.41)

    def _compute_outlet_temp(self, p_out_pa: float) -> float:
        """
        Compute discharge temperature for a given outlet pressure.
        
        This helper is used by the binary search solver to find the maximum
        pressure that keeps $T_{out} \\le T_{max}$.
        
        Args:
            p_out_pa (float): Outlet pressure [Pa].
            
        Returns:
            float: Outlet temperature [K].
        """
        p_in_pa = self.actual_inlet_pressure_pa
        strategy, fluid_id, mix_const = self._determine_fluid_strategy()
        
        # Use LUTs for pure fluids (fastest)
        if strategy == FluidStrategy.LUT and JIT_AVAILABLE and fluid_id in self._luts:
            try:
                lut_data = self._luts[fluid_id]
                _, T_out_k, _ = calculate_compression_realgas_jit(
                    p_in_pa,
                    p_out_pa,
                    self.inlet_temperature_k,
                    self.isentropic_efficiency,
                    self._P_grid,
                    self._T_grid,
                    self._S_grid,
                    lut_data['H'],
                    lut_data['S'],
                    lut_data['C'],
                    lut_data['H_from_PS']
                )
                return T_out_k
            except Exception as e:
                logger.debug(f"LUT compression failed: {e}. Trying CoolProp.")
        
        # Mixture LUT Path
        if strategy == FluidStrategy.LUT_MIXTURE and MIX_THERMO_AVAILABLE and self._lut_manager is not None:
            try:
                comp_mass = self._inlet_stream.composition
                
                # Calculate mole fractions locally (needed for entropy mixing term)
                mole_comp = {}
                moles_temp = {}
                total_moles = 0.0
                for sp, w_i in comp_mass.items():
                    if w_i > 1e-9 and sp in FLUID_REF_DATA:
                        mw = FLUID_REF_DATA[sp][1]
                        if mw > 0:
                            n = w_i / mw
                            moles_temp[sp] = n
                            total_moles += n
                
                if total_moles > 1e-12:
                    mole_comp = {k: v/total_moles for k,v in moles_temp.items()}
                
                # Use simplified iterative solver for T_out
                # Optimized JIT path for Mixtures
                if JIT_AVAILABLE and calculate_mixture_compression_jit is not None and self._H_luts_stacked is not None:
                     # Get cached composition arrays (canonical order)
                     # Stream guarantees strict canonical order matching the LUT stacks
                     weights, mole_fracs, M_mix, sum_ylny = self.inlet_stream.get_composition_arrays()
                     
                     _, T_out_k, _ = calculate_mixture_compression_jit(
                        p_in_pa,
                        p_out_pa,
                        self.inlet_temperature_k,
                        self.isentropic_efficiency,
                        self._P_grid,
                        self._T_grid,
                        self._H_luts_stacked,
                        self._S_luts_stacked,
                        self._C_luts_stacked,
                        weights,
                        mole_fracs,
                        M_mix,
                        sum_ylny
                     )
                     return T_out_k
                
                # Fallback to Python implementation (slower)
                w_act, T_out_k, _ = mix_thermo.calculate_compression_work(
                     comp_mass, 
                     mole_comp,
                     p_in_pa, 
                     self.inlet_temperature_k, 
                     p_out_pa, 
                     self.isentropic_efficiency, 
                     self._lut_manager
                )
                return T_out_k
            except Exception as e:
                logger.debug(f"Mix LUT temp calc failed: {e}")

        # CoolProp paths (pure fluid or mixture)
        if COOLPROP_AVAILABLE:
            try:
                if strategy == FluidStrategy.COOLPROP_MIXTURE:
                    # Use mixture HEOS backend
                    backend_str = fluid_id
                    h1 = CP.PropsSI('H', 'T', self.inlet_temperature_k, 'P', p_in_pa, backend_str)
                    s1 = CP.PropsSI('S', 'T', self.inlet_temperature_k, 'P', p_in_pa, backend_str)
                    h2s = CP.PropsSI('H', 'S', s1, 'P', p_out_pa, backend_str)
                    w_actual = (h2s - h1) / self.isentropic_efficiency
                    h2_actual = h1 + w_actual
                    T_out_k = CP.PropsSI('T', 'H', h2_actual, 'P', p_out_pa, backend_str)
                    return T_out_k
                elif strategy == FluidStrategy.COOLPROP:
                    # Pure fluid via CoolProp
                    coolprop_name = FLUID_REF_DATA.get(fluid_id, ('Hydrogen',))[0] if fluid_id else 'Hydrogen'
                    h1 = CP.PropsSI('H', 'T', self.inlet_temperature_k, 'P', p_in_pa, coolprop_name)
                    s1 = CP.PropsSI('S', 'T', self.inlet_temperature_k, 'P', p_in_pa, coolprop_name)
                    h2s = CP.PropsSI('H', 'S', s1, 'P', p_out_pa, coolprop_name)
                    w_actual = (h2s - h1) / self.isentropic_efficiency
                    h2_actual = h1 + w_actual
                    T_out_k = CP.PropsSI('T', 'H', h2_actual, 'P', p_out_pa, coolprop_name)
                    return T_out_k
            except Exception as e:
                logger.debug(f"CoolProp compression failed: {e}. Using ideal gas.")
        
        # Ideal gas fallback
        gamma = mix_const.get('gamma', 1.41) if mix_const else 1.41
        pressure_ratio = p_out_pa / p_in_pa
        exponent = (gamma - 1.0) / gamma
        T_isen_k = self.inlet_temperature_k * (pressure_ratio ** exponent)
        delta_t_actual = (T_isen_k - self.inlet_temperature_k) / self.isentropic_efficiency
        return self.inlet_temperature_k + delta_t_actual

    def _solve_pressure_for_temp_limit(self) -> float:
        """
        Numerically solve for maximum outlet pressure satisfying temperature constraint.
        
        **Performance Optimization**:
        Uses JIT-compiled bisection when stacked LUTs are available, eliminating
        ~30 Python→JIT roundtrips per call. Falls back to Python bisection otherwise.
        
        Returns:
            float: Maximum outlet pressure [Pa] that satisfies T_out <= T_max.
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
        p_max_pa = min(self.outlet_pressure_bar, self.max_pressure_bar) * self.BAR_TO_PA
        
        # ========== JIT-OPTIMIZED PATH ==========
        # Use fully JIT-compiled solver when stacked LUTs are available
        if (JIT_AVAILABLE and 
            solve_temp_limited_pressure_jit is not None and
            self._H_luts_stacked is not None and 
            self._inlet_stream is not None):
            try:
                # Get cached composition arrays from stream (canonical order)
                weights, mole_fracs, M_mix, sum_ylny = self._inlet_stream.get_composition_arrays()
                
                # Single JIT call replaces entire bisection loop
                p_out_pa, T_out_k, w_actual = solve_temp_limited_pressure_jit(
                    p_in_pa,
                    p_max_pa,
                    self.inlet_temperature_k,
                    self.max_temp_k,
                    self.isentropic_efficiency,
                    self._P_grid,
                    self._T_grid,
                    self._H_luts_stacked,
                    self._S_luts_stacked,
                    self._C_luts_stacked,
                    weights,
                    mole_fracs,
                    M_mix,
                    sum_ylny
                )
                
                # Cache the compression result for later use in _calculate_compression
                self._jit_cached_result = (p_out_pa, T_out_k, w_actual)
                return p_out_pa
                
            except Exception as e:
                logger.debug(f"JIT temp-limited solver failed: {e}. Falling back to Python.")
        
        # ========== PYTHON FALLBACK ==========
        # Check if target pressure is achievable
        T_at_target = self._compute_outlet_temp(p_max_pa)
        if T_at_target <= self.max_temp_k:
            return p_max_pa  # Target pressure is achievable within temp limit
        
        # Binary search for maximum pressure
        p_low = p_in_pa
        p_high = p_max_pa
        
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
        2.  **JIT Cache Check**: If temp-limited JIT solver already computed, use cached result.
        3.  **Strategy Dispatch**:
            -   **LUT**: Pure fluid with cached lookup tables (~1 µs).
            -   **COOLPROP_MIXTURE**: Multi-component HEOS backend (~100 µs).
            -   **IDEAL_GAS**: Mass-weighted constants fallback.
        """
        p_in_pa = self.actual_inlet_pressure_pa
        
        # Reset JIT cache
        self._jit_cached_result = None
        
        # Determine outlet pressure
        if self.temperature_limited and self.max_temp_c is not None:
            p_out_pa = self._solve_pressure_for_temp_limit()
            self.actual_outlet_pressure_pa = p_out_pa
            self.actual_outlet_pressure_bar = p_out_pa / self.BAR_TO_PA
            
            # ========== USE CACHED JIT RESULT ==========
            # If JIT solver already computed everything, use it directly
            if self._jit_cached_result is not None:
                _, T_out_k, w_actual = self._jit_cached_result
                
                # Set outlet properties
                self.actual_outlet_temperature_k = T_out_k
                self.outlet_temperature_k = T_out_k  # FIX: Also set the primary outlet temp
                self.outlet_temperature_c = T_out_k - 273.15  # FIX: Update Celsius attribute
                self.compression_work_j_kg = w_actual
                
                # Update Celsius attribute
                self.outlet_temperature_c = T_out_k - 273.15
                self.outlet_temperature_k = T_out_k
                self.actual_outlet_temperature_k = T_out_k # Redundant but safe
                
                # Calculate energy compliant with step() method architecture
                drive_efficiency = self.mechanical_efficiency * self.electrical_efficiency
                w_electrical = w_actual / drive_efficiency if drive_efficiency > 0 else w_actual
                
                self.specific_energy_kwh_kg = w_electrical * self.J_TO_KWH
                self.compression_work_kwh = self.specific_energy_kwh_kg * self.actual_mass_transferred_kg
                
                return  # Done - skip strategy dispatch
        else:
            p_out_pa = self.outlet_pressure_bar * self.BAR_TO_PA
            self.actual_outlet_pressure_pa = p_out_pa
            self.actual_outlet_pressure_bar = self.outlet_pressure_bar
        
        # Determine thermodynamic strategy
        strategy, fluid_id, mix_const = self._determine_fluid_strategy()

        # Dispatch by strategy
        if strategy == FluidStrategy.LUT and fluid_id in self._luts and JIT_AVAILABLE:
            self._calculate_compression_lut(p_in_pa, p_out_pa, fluid_id)
        elif strategy == FluidStrategy.LUT_MIXTURE and MIX_THERMO_AVAILABLE:
            self._calculate_compression_lut_mixture(p_in_pa, p_out_pa)
        elif strategy == FluidStrategy.COOLPROP_MIXTURE and COOLPROP_AVAILABLE:
            self._calculate_compression_mixture(p_in_pa, p_out_pa, fluid_id)
        elif COOLPROP_AVAILABLE:
            self._calculate_compression_realgas(p_in_pa, p_out_pa)
        else:
            self._calculate_compression_idealgas(p_in_pa, p_out_pa, mix_const)
    
    def _calculate_compression_lut(self, p_in_pa: float, p_out_pa: float, 
                                    fluid_key: str = 'H2') -> None:
        """
        Calculate compression using Numba JIT-compiled LUT lookups.

        **Performance Rationale**:
        Compressor calculation is the "inner loop" of the system optimization. 
        Using Numba-compiled interpolation on pre-loaded arrays (~0.1 µs) vs CoolProp calls (~100 µs) 
        provides a ~1000x speedup, enabling real-time optimization of complex compressor trains.

        Args:
            p_in_pa: Inlet pressure [Pa].
            p_out_pa: Outlet pressure [Pa].
            fluid_key: Key into self._luts dict (e.g., 'H2', 'O2', 'CH4').
        """
        try:
            lut_data = self._luts[fluid_key]
            _, gamma = FLUID_REF_DATA[fluid_key][2], FLUID_REF_DATA[fluid_key][3]
            
            # Call JIT function with fluid-specific LUT arrays
            w_specific, T_out_k, _ = calculate_compression_realgas_jit(
                p_in_pa,
                p_out_pa,
                self.inlet_temperature_k,
                self.isentropic_efficiency,
                self._P_grid,
                self._T_grid,
                self._S_grid,
                lut_data['H'],
                lut_data['S'],
                lut_data['C'],
                lut_data['H_from_PS']
            )
            
            # Check for LUT clamping at upper bound (1200 K)
            if T_out_k >= 1199.0:
                logger.warning(
                    f"Compressor {self.component_id}: Discharge temperature {T_out_k:.1f} K "
                    f"exceeds LUT upper bound. Result may be clamped!"
                )
            
            self.outlet_temperature_k = T_out_k
            self.outlet_temperature_c = T_out_k - 273.15
            
            # Estimate isentropic outlet temp using fluid-specific gamma
            gamma = FLUID_REF_DATA[fluid_key][3]
            exponent = (gamma - 1.0) / gamma
            T_isen_k = self.inlet_temperature_k * (p_out_pa / p_in_pa)**exponent
            self.outlet_temperature_isentropic_c = T_isen_k - 273.15
            
            # Drive efficiency
            drive_efficiency = self.mechanical_efficiency * self.electrical_efficiency
            w_electrical = w_specific / drive_efficiency
            
            self.specific_energy_kwh_kg = w_electrical * self.J_TO_KWH
            self.compression_work_kwh = self.specific_energy_kwh_kg * self.actual_mass_transferred_kg
            
        except Exception as e:
            logger.warning(f"LUT compression failed for {self.component_id} ({fluid_key}): {e}")
            if COOLPROP_AVAILABLE:
                self._calculate_compression_realgas(p_in_pa, p_out_pa)
            else:
                self._calculate_compression_idealgas(p_in_pa, p_out_pa, {})

    def _calculate_compression_lut_mixture(self, p_in_pa: float, p_out_pa: float) -> None:
        """
        Calculate compression for multi-component mixtures using Ideal Mixing of Real Gases.

        **Physical Basis**:
        Uses LUT-based pure component properties with rigorous mixing rules:
        - Enthalpy: h_mix = Σ w_i · h_i(P, T)
        - Entropy: s_mix = Σ w_i · s_i(P, T) - R_mix · Σ y_i · ln(y_i)
        - Newton-Raphson solver for isentropic outlet temperature

        **Performance**: ~6 µs per call (6 LUT lookups) vs ~100 µs for CoolProp.

        Args:
            p_in_pa: Inlet pressure [Pa].
            p_out_pa: Outlet pressure [Pa].
        """
        try:
            # Get composition from inlet stream
            comp_mass = self._inlet_stream.composition.copy()
            comp_mole = self._inlet_stream.mole_fractions
            
            # Calculate compression using mixture thermodynamics
            # Optimized JIT path for Mixtures
            if JIT_AVAILABLE and calculate_mixture_compression_jit is not None and self._H_luts_stacked is not None:
                 # Get cached composition arrays (canonical order)
                 # Stream guarantees strict canonical order matching the LUT stacks
                 weights, mole_fracs, M_mix, sum_ylny = self._inlet_stream.get_composition_arrays()
                 
                 w_actual, T_out_actual, T_out_isen = calculate_mixture_compression_jit(
                    p_in_pa,
                    p_out_pa,
                    self.inlet_temperature_k,
                    self.isentropic_efficiency,
                    self._P_grid,
                    self._T_grid,
                    self._H_luts_stacked,
                    self._S_luts_stacked,
                    self._C_luts_stacked,
                    weights,     # Mass fractions
                    mole_fracs,  # Mole fractions
                    M_mix,       # Molar mass
                    sum_ylny     # Entropy mixing term
                )
            else:
                # Fallback purely if JIT specific setup failed (should not happen if initialized)
                logger.warning("JIT mixture solver unavailable, using Python fallback.")
                comp_mass = self._inlet_stream.composition.copy()
                comp_mole = self._inlet_stream.mole_fractions
                w_actual, T_out_actual, T_out_isen = mix_thermo.calculate_compression_work(
                    comp_mass,
                    comp_mole,
                    p_in_pa,
                    self.inlet_temperature_k,
                    p_out_pa,
                    self.isentropic_efficiency,
                    self._lut_manager
                )
            
            # Update state
            self.outlet_temperature_k = T_out_actual
            self.outlet_temperature_c = T_out_actual - 273.15
            self.outlet_temperature_isentropic_c = T_out_isen - 273.15
            
            # Apply drive efficiency
            drive_efficiency = self.mechanical_efficiency * self.electrical_efficiency
            w_electrical = w_actual / drive_efficiency
            
            self.specific_energy_kwh_kg = w_electrical * self.J_TO_KWH
            self.compression_work_kwh = self.specific_energy_kwh_kg * self.actual_mass_transferred_kg
            
            logger.debug(
                f"LUT_MIXTURE compression: T_out={T_out_actual:.1f}K, "
                f"w={w_actual/1000:.2f} kJ/kg"
            )
            
        except Exception as e:
            logger.warning(
                f"LUT_MIXTURE compression failed for {self.component_id}: {e}. "
                f"Falling back to CoolProp."
            )
            if COOLPROP_AVAILABLE:
                self._calculate_compression_realgas(p_in_pa, p_out_pa)
            else:
                mix_const = self._calculate_mixture_constants(self._inlet_stream.composition)
                self._calculate_compression_idealgas(p_in_pa, p_out_pa, mix_const)

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

    def _calculate_compression_mixture(self, p_in_pa: float, p_out_pa: float, 
                                        mixture_string: str) -> None:
        """
        Calculates compression for multi-component mixtures using CoolProp HEOS.

        **Physics**:
        Uses CoolProp's HEOS backend for rigorous mixture thermodynamics, accounting for:
        - Non-ideal mixing (activity coefficients)
        - Component-specific heat capacities
        - Mixture compressibility

        Args:
            p_in_pa: Inlet pressure [Pa].
            p_out_pa: Outlet pressure [Pa].
            mixture_string: CoolProp HEOS backend string (e.g., "HEOS::Hydrogen[0.5]&Methane[0.5]").
        """
        try:
            h1 = CP.PropsSI('H', 'T', self.inlet_temperature_k, 'P', p_in_pa, mixture_string)
            s1 = CP.PropsSI('S', 'T', self.inlet_temperature_k, 'P', p_in_pa, mixture_string)

            t2s_k = CP.PropsSI('T', 'S', s1, 'P', p_out_pa, mixture_string)
            h2s = CP.PropsSI('H', 'S', s1, 'P', p_out_pa, mixture_string)

            self.outlet_temperature_isentropic_c = t2s_k - 273.15

            w_isentropic = h2s - h1
            w_actual = w_isentropic / self.isentropic_efficiency

            h2_actual = h1 + w_actual

            t2_k = CP.PropsSI('T', 'H', h2_actual, 'P', p_out_pa, mixture_string)

            self.outlet_temperature_k = t2_k
            self.outlet_temperature_c = t2_k - 273.15

            drive_efficiency = self.mechanical_efficiency * self.electrical_efficiency
            w_electrical = w_actual / drive_efficiency

            self.specific_energy_kwh_kg = w_electrical * self.J_TO_KWH
            self.compression_work_kwh = self.specific_energy_kwh_kg * self.actual_mass_transferred_kg

        except Exception as e:
            logger.warning(f"Mixture compression failed for {self.component_id}: {e}. Falling back to ideal gas.")
            # Fall back to ideal gas with mixture constants
            if self._inlet_stream and self._inlet_stream.composition:
                mix_const = self._calculate_mixture_constants(self._inlet_stream.composition)
            else:
                mix_const = {'cp': 14304.0, 'gamma': 1.41}
            self._calculate_compression_idealgas(p_in_pa, p_out_pa, mix_const)

    def _calculate_compression_idealgas(self, p_in_pa: float, p_out_pa: float,
                                         mix_const: Optional[Dict[str, float]] = None) -> None:
        """
        Calculates compression using Ideal Gas Law approximation.

        **Applicability**:
        Used as a last resort when property databases are missing. Accuracy degrades 
        significantly above 50 bar for Hydrogen due to compressibility ($Z$) deviating from 1.0.

        **Formula**:
        $$ T_{2s} = T_{in} \\cdot \\left(\\frac{P_{out}}{P_{in}}\\right)^{\\frac{\\gamma-1}{\\gamma}} $$
        $$ \\Delta T_{actual} = \\frac{T_{2s} - T_{in}}{\\eta_{isen}} $$
        $$ W_{actual} = C_p \\cdot \\Delta T_{actual} $$

        Args:
            p_in_pa: Inlet pressure [Pa].
            p_out_pa: Outlet pressure [Pa].
            mix_const: Optional dict with 'cp' and 'gamma' for mixture. If None, uses legacy lookup.
        """
        if mix_const:
            cp = mix_const.get('cp', 14304.0)
            gamma = mix_const.get('gamma', 1.41)
        else:
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
