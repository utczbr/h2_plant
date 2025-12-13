"""
PEM Electrolyzer Stack Component.

This module implements a detailed Proton Exchange Membrane (PEM) electrolyzer
model with mechanistic electrochemistry and time-dependent degradation tracking.
The model captures the key physics governing water electrolysis performance.

Electrochemical Principles:
    - **Nernst Equation**: Calculates thermodynamic reversible voltage as a
      function of temperature and pressure. Higher pressures shift equilibrium
      favorably, reducing subsequent compression work.
    - **Overvoltage Components**: The model separates total cell voltage into
      reversible potential plus activation, ohmic, and concentration losses.
    - **Faraday's Law**: Relates current to molar production rates through the
      stoichiometry 2H₂O → 2H₂ + O₂ with z=2 electrons per mole H₂.
    - **Degradation**: Cell voltage increases over operating hours due to
      catalyst dissolution, membrane thinning, and interfacial degradation.

Architecture:
    Implements the Component Lifecycle Contract (Layer 1):
    - `initialize()`: Loads degradation polynomials and connects to LUT manager.
    - `step()`: Executes electrochemical calculations and mass/energy balances.
    - `get_state()`: Exposes production rates, efficiency, and aging metrics.

Balance of Plant (BoP):
    Ancillary power consumption includes water purification, circulation pumps,
    and power electronics. BoP is modeled as fixed plus variable components
    proportional to stack power.

Physical Constants (from PEMConstants):
    - Faraday constant: 96,485.33 C/mol
    - Universal gas constant: 8.314 J/(mol·K)
    - Lower heating value H₂: 33.33 kWh/kg

References:
    - Barbir, F. (2005). PEM Electrolysis for Production of Hydrogen from
      Renewable Energy Sources. Solar Energy, 78(5), 661-669.
    - Carmo, M. et al. (2013). A comprehensive review on PEM water electrolysis.
      International Journal of Hydrogen Energy, 38(12), 4901-4934.
"""

from typing import Dict, Any, Optional
import numpy as np
import pickle
from pathlib import Path
import os
import warnings

from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.component_ids import ComponentID
from h2_plant.config.constants_physics import PEMConstants
from h2_plant.models.pem_physics import calculate_eta_F
from h2_plant.models.thermal_inertia import ThermalInertiaModel
from h2_plant.core.stream import Stream
from numpy.polynomial import Polynomial

CONST = PEMConstants()


class DetailedPEMElectrolyzer(Component):
    """
    Mechanistic PEM electrolyzer model with degradation tracking.

    Implements white-box electrochemistry including Nernst potential,
    overvoltage contributions (activation, ohmic, concentration), and
    Balance of Plant (BoP) power consumption. Degradation is tracked
    cumulatively to model voltage fade over the stack lifetime.

    This component fulfills the Component Lifecycle Contract (Layer 1):
        - `initialize()`: Loads degradation curves from LUTManager or pickle files,
          and registers with the component registry.
        - `step()`: Computes current density, cell voltage, production rates,
          and thermal output for each simulation timestep.
        - `get_state()`: Returns operational metrics including efficiency,
          cumulative production, and aging indicators.

    The electrochemical model solves for current density (j) given a power
    setpoint, then calculates mass flows via Faraday's law:
        ṁ_H₂ = (j × A) / (z × F) × M_H₂

    Attributes:
        max_power_mw (float): Maximum stack power consumption in MW.
        base_efficiency (float): Nominal system efficiency at rated power.
        use_polynomials (bool): If True, use pre-computed j(P) polynomials
            for faster simulation.
        out_pressure_pa (float): Hydrogen output pressure in Pa.
        t_op_h (float): Cumulative operating hours for degradation tracking.

    Example:
        >>> pem = DetailedPEMElectrolyzer({'max_power_mw': 5.0, 'use_polynomials': False})
        >>> pem.initialize(dt=1/60, registry=registry)
        >>> pem.set_power_input_mw(4.0)
        >>> pem.step(t=0.0)
        >>> state = pem.get_state()
        >>> print(f"H2 production: {state['h2_production_kg_h']:.2f} kg/h")
    """

    def __init__(self, config: Any):
        """
        Initialize the PEM electrolyzer component.

        Parses configuration from either a Pydantic model or legacy dictionary
        format. Initializes electrochemical parameters, degradation tracking,
        and thermal management subsystems.

        Args:
            config (Union[dict, PEMPhysicsSpec]): Configuration object containing:
                - max_power_mw (float): Maximum power consumption in MW.
                - base_efficiency (float): Nominal efficiency (0.0-1.0).
                - use_polynomials (bool): Enable polynomial j(P) lookup.
                - water_excess_factor (float): Excess water fraction beyond stoichiometry.
                - out_pressure_pa (float, optional): H₂ output pressure in Pa.

        Note:
            Electrochemical constants are defined at the class level from
            PEMConstants to maintain consistency across model components.
        """
        super().__init__(config)

        # Configuration parsing: support Pydantic models and legacy dicts
        if hasattr(config, 'max_power_mw'):
            self.max_power_mw = config.max_power_mw
            self.base_efficiency = config.base_efficiency
            self.use_polynomials = config.use_polynomials
            self.water_excess_factor = getattr(config, 'water_excess_factor', 0.02)

            from h2_plant.core.constants import StorageConstants
            self.out_pressure_pa = getattr(config, 'out_pressure_pa', StorageConstants.LOW_PRESSURE_PA)
        else:
            self.config = config
            self.max_power_mw = config.get('max_power_mw', 5.0)
            self.base_efficiency = config.get('base_efficiency', 0.65)
            self.use_polynomials = config.get('use_polynomials', False)
            self.water_excess_factor = config.get('water_excess_factor', 0.02)
            if 'component_id' in config:
                self.component_id = config['component_id']

            from h2_plant.core.constants import StorageConstants
            self.out_pressure_pa = config.get('out_pressure_pa', StorageConstants.LOW_PRESSURE_PA)

        # ====================================================================
        # Electrochemical State Variables
        # ====================================================================
        self.t_op_h = 0.0
        self.P_consumed_W = 0.0
        self.m_H2_kg_s = 0.0
        self.m_H2O_kg_s = 0.0
        self.I_total = 0.0

        # Fundamental constants (SI units)
        self.F = 96485.33  # Faraday constant (C/mol)
        self.R = 8.314     # Universal gas constant (J/(mol·K))
        self.P_ref = 1.0e5 # Reference pressure (Pa)
        self.z = 2         # Electrons transferred per mole H₂
        self.MH2 = 2.016e-3   # Molar mass H₂ (kg/mol)
        self.MO2 = 31.998e-3  # Molar mass O₂ (kg/mol)
        self.MH2O = 18.015e-3 # Molar mass H₂O (kg/mol)

        # ====================================================================
        # Stack Configuration
        # ====================================================================
        # Geometry sized for ~5 MW nominal operation
        self.N_stacks = 35
        self.N_cell_per_stack = 85
        self.A_cell = 300  # Active area per cell (cm²)
        self.Area_Total = self.N_stacks * self.N_cell_per_stack * self.A_cell
        self.j_nom = 2.91  # Nominal current density (A/cm²)

        # ====================================================================
        # Operating Conditions
        # ====================================================================
        self.T = 333.15    # Operating temperature (K) - typical 60°C
        self.P_op = 40.0e5 # Operating pressure (Pa) - 40 bar

        # ====================================================================
        # Electrochemical Parameters
        # ====================================================================
        # Ohmic losses: membrane conductivity and thickness
        self.delta_mem = 100 * 1e-4  # Membrane thickness (m), 100 μm typical
        self.sigma_base = 0.1        # Base conductivity (S/cm)

        # Activation losses: exchange current density and symmetry factor
        self.j0 = 1.0e-6   # Exchange current density (A/cm²)
        self.alpha = 0.5   # Charge transfer coefficient (symmetric)

        # Concentration losses: limiting current density
        self.j_lim = 4.0   # Limiting current density (A/cm²)

        # ====================================================================
        # Balance of Plant (BoP)
        # ====================================================================
        # BoP includes pumps, control systems, and power electronics
        self.floss = 0.02  # Fractional parasitic losses
        self.P_nominal_sistema_W = self.max_power_mw * 1e6
        self.P_bop_fixo = 0.025 * self.P_nominal_sistema_W  # Fixed BoP (W)
        self.k_bop_var = 0.04  # Variable BoP as fraction of stack power

        # ====================================================================
        # Degradation Model
        # ====================================================================
        self.H_MES = 730.0  # Hours per month (for polynomial indexing)
        self.polynomial_list = []
        if self.use_polynomials:
            self._load_polynomials()

        self._lut = None

        # ====================================================================
        # Output State Variables
        # ====================================================================
        self.m_O2_kg_s = 0.0
        self.V_cell = 0.0
        self.heat_output_kw = 0.0
        self.state = "OFF"

        self._target_power_mw = 0.0

        # Per-timestep production (for monitoring)
        self.h2_output_kg = 0.0
        self.o2_output_kg = 0.0

        # Cumulative counters
        self.cumulative_h2_kg = 0.0
        self.cumulative_o2_kg = 0.0
        self.cumulative_energy_kwh = 0.0

        # ====================================================================
        # Water Management
        # ====================================================================
        # Buffer accumulates water from receive_input for mass-balance limiting
        self.water_buffer_kg = 0.0
        self.available_water_kg_h = float('inf')

        # ====================================================================
        # Degradation Interpolation Tables
        # ====================================================================
        # Convert yearly degradation data to hourly basis for interpolation
        self.t_op_h_table = np.array(CONST.DEGRADATION_YEARS) * 8760.0
        self.v_cell_table = np.array(CONST.DEGRADATION_V_STACK) / CONST.N_cell_per_stack

        try:
            from h2_plant.models import pem_physics as phys
            # Beginning-of-Life (BOL) reference voltage for degradation delta
            self.V_CELL_BOL_NOM = phys.calculate_Vcell_base(CONST.j_nom, self.T, self.P_op)
        except Exception as e:
            warnings.warn(f"Failed to initialize BOL voltage reference: {e}")
            self.deg_interpolator = None
            self.V_CELL_BOL_NOM = 0.0

        # ====================================================================
        # Thermal Management
        # ====================================================================
        # Lumped thermal model for stack temperature dynamics
        self.thermal_model = ThermalInertiaModel(
            C_thermal_J_K=2.6e6,      # Thermal mass for ~5 MW stack
            h_A_passive_W_K=100.0,    # Natural convection coefficient
            T_initial_K=298.15,       # Ambient start temperature
            max_cooling_kw=100.0      # Active cooling capacity
        )

    def _load_polynomials(self) -> None:
        """
        Load pre-computed polynomial j(P) models from disk.

        Polynomials provide O(1) lookup of current density given power setpoint,
        avoiding iterative solver calls during simulation. Each polynomial
        corresponds to a specific degradation month, capturing aging effects.

        The polynomial file is searched in multiple locations for portability:
        1. Current working directory
        2. Package data directory (h2_plant/data/)

        Note:
            Falls back to iterative solver if polynomials cannot be loaded.
        """
        try:
            local_path = Path("degradation_polynomials.pkl")
            if local_path.exists():
                with open(local_path, 'rb') as f:
                    self.polynomial_list = pickle.load(f)
                    return

            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent
            data_path = project_root / 'data' / 'degradation_polynomials.pkl'

            if data_path.exists():
                with open(data_path, 'rb') as f:
                    self.polynomial_list = pickle.load(f)
            else:
                self.use_polynomials = False

        except Exception as e:
            warnings.warn(f"Failed to load polynomials: {e}. Using iterative solver.")
            self.use_polynomials = False

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """
        Prepare the electrolyzer for simulation execution.

        Fulfills the Component Lifecycle Contract initialization phase. Connects
        to the LUT manager for property lookups and validates physical parameters.

        Args:
            dt (float): Simulation timestep in hours.
            registry (ComponentRegistry): Central registry for component access.
        """
        super().initialize(dt, registry)

        if registry.has(ComponentID.LUT_MANAGER.value):
            self._lut = registry.get(ComponentID.LUT_MANAGER)

    def step(self, t: float) -> None:
        """
        Execute one simulation timestep.

        Computes the electrochemical state for the current power setpoint:
        1. Solve for current density j using polynomial or iterative method.
        2. Check water availability and throttle if mass-balance limited.
        3. Calculate cell voltage including degradation overpotential.
        4. Compute mass flows via Faraday's law.
        5. Update thermal model and calculate waste heat.

        This method fulfills the Component Lifecycle Contract step phase,
        advancing all electrochemical state variables.

        Args:
            t (float): Current simulation time in hours.

        Note:
            Power setpoint is obtained from _target_power_mw, which is set
            either via set_power_input_mw() or by the DualPathCoordinator.
        """
        super().step(t)

        P_setpoint_mw = self._target_power_mw

        # Clamp to rated capacity
        if P_setpoint_mw > self.max_power_mw:
            P_setpoint_mw = self.max_power_mw

        # Query coordinator for setpoint if not set directly
        if P_setpoint_mw <= 0:
            try:
                coordinator = self._registry.get(ComponentID.DUAL_PATH_COORDINATOR)
                if coordinator:
                    P_setpoint_mw = coordinator.pem_setpoint_mw
            except Exception:
                pass

        if P_setpoint_mw <= 0.001:
            # Electrolyzer OFF state
            self.m_H2_kg_s = 0.0
            self.m_O2_kg_s = 0.0
            self.m_H2O_kg_s = 0.0
            self.P_consumed_W = 0.0
            self.V_cell = 0.0
            self.I_total = 0.0
            self.heat_output_kw = 0.0
            self.state = "OFF"
            self.h2_output_kg = 0.0
            self.o2_output_kg = 0.0
            return

        self.state = "ON"

        try:
            target_power_W = P_setpoint_mw * 1e6
            j_op = 0.0

            # ================================================================
            # Current Density Determination
            # ================================================================
            if self.use_polynomials and self.polynomial_list:
                # Polynomial lookup: O(1) complexity
                month_index = int(self.t_op_h / self.H_MES)
                if month_index >= len(self.polynomial_list):
                    month_index = len(self.polynomial_list) - 1
                poly_object = self.polynomial_list[month_index]

                # Handle split polynomial for high/low power regimes
                if isinstance(poly_object, dict):
                    if target_power_W <= poly_object['split_point']:
                        j_op = poly_object['poly_low'](target_power_W)
                    else:
                        j_op = poly_object['poly_high'](target_power_W)
                else:
                    j_op = poly_object(target_power_W)
            else:
                # Iterative solver: Newton-Raphson via Numba JIT
                from h2_plant.optimization.numba_ops import solve_pem_j_jit

                j_guess = self.j_nom * (target_power_W / self.P_nominal_sistema_W)

                try:
                    j_op = solve_pem_j_jit(
                        target_power_W,
                        self.T,
                        self.P_op,
                        self.Area_Total,
                        self.P_bop_fixo,
                        self.k_bop_var,
                        j_guess,
                        self.R, self.F, self.z, self.alpha, self.j0, self.j_lim,
                        self.delta_mem, self.sigma_base, self.P_ref
                    )
                except Exception as e:
                    warnings.warn(f"JIT solver failed: {e}. Using linear approximation.")
                    j_op = j_guess

            # Clamp to physically realizable current density
            j_op = max(0.001, min(j_op, CONST.j_lim))

            # ================================================================
            # Mass Balance Constraint
            # ================================================================
            # Verify sufficient water is available for stoichiometric reaction.
            # Throttle current density if water-limited.
            from h2_plant.models import pem_physics as phys

            m_H2_p, m_O2_p, m_H2O_p = phys.calculate_flows(j_op)

            dt_seconds = self.dt * 3600.0
            proposed_water_cons_kg = m_H2O_p * dt_seconds * (1.0 + self.water_excess_factor)

            reduction_factor = 1.0
            if proposed_water_cons_kg > 1e-9:
                if self.water_buffer_kg < proposed_water_cons_kg:
                    reduction_factor = self.water_buffer_kg / proposed_water_cons_kg

            j_op = j_op * reduction_factor

            # ================================================================
            # Electrochemical Calculations
            # ================================================================
            self.I_total = j_op * self.Area_Total

            # Mass flows from constrained current density
            self.m_H2_kg_s, self.m_O2_kg_s, self.m_H2O_kg_s = phys.calculate_flows(j_op)

            # Cell voltage includes degradation overpotential
            U_deg = self._calculate_U_deg(self.t_op_h)
            self.V_cell = phys.calculate_Vcell_base(j_op, self.T, CONST.P_op_default) + U_deg

            # Stack and system power
            P_stack = self.I_total * self.V_cell
            P_bop = CONST.P_bop_fixo + CONST.k_bop_var * P_stack
            self.P_consumed_W = P_stack + P_bop

            # ================================================================
            # Thermal Calculations
            # ================================================================
            # Heat generation = irreversible losses = (V_cell - V_tn) × I
            # V_tn is thermoneutral voltage where all input energy becomes H₂ enthalpy
            U_rev = self._calculate_U_rev(self.T)
            U_tn = 1.481

            heat_power_W = self.I_total * (self.V_cell - U_tn)
            if heat_power_W < 0:
                heat_power_W = 0

            heat_power_W += self.P_consumed_W * 0.01  # BoP thermal contribution
            self.heat_output_kw = heat_power_W / 1000.0

            # Update thermal model (held at constant T for simplified simulation)
            self.thermal_model.step(dt_seconds, heat_power_W, 333.15)
            self.thermal_model.T_current_K = 333.15

        except Exception as e:
            warnings.warn(f"Electrochemical calculation error: {e}")
            import traceback
            traceback.print_exc()
            self.P_consumed_W = 0.0

        # ====================================================================
        # Output Integration
        # ====================================================================
        dt_seconds = self.dt * 3600.0
        self.h2_output_kg = self.m_H2_kg_s * dt_seconds
        self.o2_output_kg = self.m_O2_kg_s * dt_seconds

        reaction_water_kg = self.m_H2O_kg_s * dt_seconds
        self.water_consumption_kg = reaction_water_kg * (1.0 + self.water_excess_factor)

        # Debit water buffer
        self.water_buffer_kg -= self.water_consumption_kg
        if self.water_buffer_kg < 0:
            self.water_buffer_kg = 0.0

        # Update cumulative counters
        self.cumulative_h2_kg += self.h2_output_kg
        self.cumulative_o2_kg += self.o2_output_kg
        self.cumulative_energy_kwh += (self.P_consumed_W / 1000.0) * self.dt

        if hasattr(self, 'dt'):
            self.t_op_h += self.dt

    def _calculate_U_rev(self, T: float) -> float:
        """
        Calculate the Nernst reversible potential.

        Computes the thermodynamic minimum voltage for water splitting at the
        given temperature and operating pressure. Temperature dependence follows
        the Gibbs-Helmholtz equation; pressure dependence uses the Nernst equation.

        Args:
            T (float): Operating temperature in K.

        Returns:
            float: Reversible cell voltage in V.
        """
        P_op = 40.0e5
        # Temperature correction: dU/dT ≈ -0.9 mV/K near 298 K
        U_rev_T = 1.229 - 0.9e-3 * (T - 298.15)
        # Pressure correction: Nernst for O₂/H₂O equilibrium
        pressure_ratio = P_op / 1.0e5
        Nernst_correction = (CONST.R * T) / (CONST.z * CONST.F) * np.log(pressure_ratio**1.5)
        return U_rev_T + Nernst_correction

    def _calculate_voltage_base(self, j_op: float, T: float) -> float:
        """
        Calculate base cell voltage at Beginning of Life (BOL).

        Combines reversible potential with overvoltage contributions:
        - Activation overpotential (Tafel equation)
        - Ohmic losses (membrane resistance)
        - Concentration polarization (mass transport)

        Args:
            j_op (float): Operating current density in A/cm².
            T (float): Operating temperature in K.

        Returns:
            float: BOL cell voltage in V.
        """
        from h2_plant.models import pem_physics as phys
        return phys.calculate_Vcell_base(j_op, T, CONST.P_op_default)

    def _calculate_degradation_voltage(self, t_op_h: float) -> float:
        """
        Calculate voltage degradation increment.

        Interpolates degradation tables to determine additional overpotential
        due to aging mechanisms including catalyst dissolution and membrane
        degradation.

        Args:
            t_op_h (float): Cumulative operating hours.

        Returns:
            float: Degradation overpotential increment in V.

        Note:
            Returns 0.0 if degradation interpolator is not initialized.
        """
        if self.deg_interpolator is None:
            return 0.0

    def _calculate_U_deg(self, t_op_h: float) -> float:
        """
        Calculate degradation overpotential from empirical tables.

        Uses linear interpolation on measured stack voltage data to determine
        the voltage increase relative to BOL performance. This delta-based
        approach ensures degradation is additive to the mechanistic voltage model.

        Args:
            t_op_h (float): Cumulative operating hours.

        Returns:
            float: Degradation overpotential in V (always ≥ 0).
        """
        V_cell_degraded = np.interp(t_op_h, self.t_op_h_table, self.v_cell_table)
        U_deg = np.maximum(0.0, V_cell_degraded - self.V_CELL_BOL_NOM)
        return float(U_deg)

    def _calculate_voltage(self, j_op: float, T: float) -> float:
        """
        Calculate total cell voltage including degradation.

        Combines BOL electrochemical voltage with degradation overpotential
        to give the actual operating voltage.

        Args:
            j_op (float): Operating current density in A/cm².
            T (float): Operating temperature in K.

        Returns:
            float: Total cell voltage in V.
        """
        base_V = self._calculate_voltage_base(j_op, T)
        U_deg = self._calculate_U_deg(self.t_op_h)
        return base_V + U_deg

    def set_power_input_mw(self, P_mw: float) -> None:
        """
        Set the power setpoint for the next simulation step.

        Called by the dispatch coordinator to specify electrolyzer power
        allocation. Negative values are clamped to zero.

        Args:
            P_mw (float): Target power consumption in MW.
        """
        self._target_power_mw = max(0.0, P_mw)

    def shutdown(self) -> None:
        """
        Transition electrolyzer to shutdown state.

        Called by coordinator when PEM capacity is not required. Sets power
        setpoint to zero and updates state flag.
        """
        self._target_power_mw = 0.0
        self.state = "SHUTDOWN"

    def get_state(self) -> Dict[str, Any]:
        """
        Retrieve the component's current operational state.

        Fulfills the Component Lifecycle Contract state access, returning
        all key performance indicators and internal metrics for monitoring.

        Returns:
            Dict[str, Any]: State dictionary containing:
                - h2_production_kg_h (float): Hydrogen production rate (kg/h).
                - o2_production_kg_h (float): Oxygen production rate (kg/h).
                - water_consumption_kg (float): Water consumed this timestep (kg).
                - power_consumption_mw (float): System power draw (MW).
                - heat_output_kw (float): Waste heat generation (kW).
                - system_efficiency_percent (float): LHV-based efficiency (%).
                - cell_voltage_v (float): Operating cell voltage (V).
                - state (str): Operational state ("ON", "OFF", "SHUTDOWN").
                - cumulative_h2_kg (float): Total H₂ produced (kg).
                - cumulative_o2_kg (float): Total O₂ produced (kg).
                - cumulative_energy_kwh (float): Total energy consumed (kWh).
        """
        return {
            **super().get_state(),
            "h2_production_kg_h": self.m_H2_kg_s * 3600,
            "o2_production_kg_h": self.m_O2_kg_s * 3600,
            "h2_output_kg": self.h2_output_kg,
            "o2_output_kg": self.o2_output_kg,
            "water_consumption_kg": getattr(self, 'water_consumption_kg', 0.0),
            "power_consumption_mw": self.P_consumed_W / 1e6,
            "heat_output_kw": self.heat_output_kw,
            "system_efficiency_percent": (
                (self.m_H2_kg_s * CONST.LHVH2_kWh_kg * 3.6e6) / self.P_consumed_W * 100
                if self.P_consumed_W > 0 else 0.0
            ),
            "cell_voltage_v": self.V_cell,
            "state": self.state,
            "cumulative_h2_kg": self.cumulative_h2_kg,
            "cumulative_o2_kg": self.cumulative_o2_kg,
            "cumulative_energy_kwh": self.cumulative_energy_kwh
        }

    def get_output(self, port_name: str) -> Any:
        """
        Retrieve the output stream from a specified port.

        Constructs Stream objects representing hydrogen, oxygen, and heat
        outputs based on current production rates.

        Args:
            port_name (str): Port identifier ('h2_out', 'oxygen_out', 'heat_out').

        Returns:
            Stream: For gas ports, a Stream object with mass flow, temperature,
                pressure, and composition. For 'heat_out', returns float in kW.

        Raises:
            ValueError: If port_name is not a valid output port.

        Note:
            H₂ output includes unreacted water vapor (mist) based on the
            unreacted_water_fraction constant.
        """
        if port_name == 'h2_out':
            water_fraction = CONST.unreacted_water_fraction
            m_H2O_carryover_kg_s = self.m_H2_kg_s * water_fraction / (1.0 - water_fraction)
            m_total_out_kg_s = self.m_H2_kg_s + m_H2O_carryover_kg_s

            return Stream(
                mass_flow_kg_h=m_total_out_kg_s * 3600.0,
                temperature_k=353.15,
                pressure_pa=self.out_pressure_pa,
                composition={
                    'H2': self.m_H2_kg_s / m_total_out_kg_s if m_total_out_kg_s > 0 else 0.0,
                    'H2O': m_H2O_carryover_kg_s / m_total_out_kg_s if m_total_out_kg_s > 0 else 0.0
                },
                phase='gas'
            )
        elif port_name == 'oxygen_out':
            return Stream(
                mass_flow_kg_h=self.m_O2_kg_s * 3600.0,
                temperature_k=353.15,
                pressure_pa=101325.0,
                composition={'O2': 1.0},
                phase='gas'
            )
        elif port_name == 'heat_out':
            return self.heat_output_kw
        else:
            raise ValueError(f"Unknown output port '{port_name}' on {self.component_id}")

    def receive_input(self, port_name: str, value: Any, resource_type: str) -> float:
        """
        Accept an input stream at the specified port.

        Accumulates water into an internal buffer for stoichiometric limiting.
        Power inputs update the target setpoint for the next step.

        Args:
            port_name (str): Target port ('water_in' or 'power_in').
            value (Any): Stream object for water, or float for power (MW).
            resource_type (str): Resource classification hint.

        Returns:
            float: Amount accepted (kg/h for water, MW for power), or 0.0.
        """
        if port_name == 'water_in':
            if isinstance(value, Stream):
                water_received_kg = value.mass_flow_kg_h * self.dt
                self.water_buffer_kg += water_received_kg
                self.available_water_kg_h = value.mass_flow_kg_h
                return value.mass_flow_kg_h
        elif port_name == 'power_in':
            if isinstance(value, (int, float)):
                self._target_power_mw = float(value)
                return float(value)
        return 0.0

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """
        Define the physical connection ports for this component.

        Returns:
            Dict[str, Dict[str, str]]: Port definitions with keys:
                - water_in: Ultrapure water feed.
                - power_in: Electrical power from grid/rectifier.
                - h2_out: Wet hydrogen product stream.
                - oxygen_out: Oxygen byproduct stream.
                - heat_out: Waste heat for thermal integration.
        """
        return {
            'water_in': {'type': 'input', 'resource_type': 'water', 'units': 'kg/h'},
            'power_in': {'type': 'input', 'resource_type': 'electricity', 'units': 'MW'},
            'h2_out': {'type': 'output', 'resource_type': 'hydrogen', 'units': 'kg/h'},
            'oxygen_out': {'type': 'output', 'resource_type': 'oxygen', 'units': 'kg/h'},
            'heat_out': {'type': 'output', 'resource_type': 'heat', 'units': 'kW'}
        }


# ============================================================================
# Auxiliary Components for PEM Subsystem
# ============================================================================

class RecirculationPump(Component):
    """
    Water recirculation pump for PEM stack cooling loop.

    Provides continuous water flow through the stack for heat removal and
    product gas humidification. This simplified model tracks flow rate
    without detailed head-flow characteristics.

    Implements Component Lifecycle Contract with pass-through step behavior.
    """

    def __init__(self, *args, **kwargs):
        """Initialize pump with default zero flow rate."""
        super().__init__()
        self.flow_rate_kg_h = 0.0

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """Prepare component for simulation."""
        super().initialize(dt, registry)

    def step(self, t: float) -> None:
        """No-op step - flow rate set externally."""
        pass

    def get_state(self) -> Dict[str, Any]:
        """Return component identifier."""
        return {"component_id": self.component_id}


class HeatExchanger(Component):
    """
    Heat exchanger for PEM thermal management.

    Transfers waste heat from stack cooling loop to external sink (ambient
    or process heating). Simplified model without detailed NTU/effectiveness
    calculations.

    Implements Component Lifecycle Contract with pass-through behavior.
    """

    def __init__(self, *args, **kwargs):
        """Initialize heat exchanger with default zero flow."""
        super().__init__()
        self.inlet_flow_kg_h = 0.0

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """Prepare component for simulation."""
        super().initialize(dt, registry)

    def step(self, t: float) -> None:
        """No-op step - heat duty set externally."""
        pass

    def get_state(self) -> Dict[str, Any]:
        """Return component identifier."""
        return {"component_id": self.component_id}


class SeparationTank(Component):
    """
    Gas-liquid separation tank for product streams.

    Removes entrained water from hydrogen or oxygen streams via gravity
    settling. Simplified model with unity separation efficiency (all liquid
    removed).

    Implements Component Lifecycle Contract with pass-through flow behavior.
    """

    def __init__(self, *args, **kwargs):
        """Initialize separation tank with default zero flows."""
        super().__init__()
        self.gas_inlet_kg_h = 0.0
        self.dry_gas_outlet_kg_h = 0.0

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """Prepare component for simulation."""
        super().initialize(dt, registry)

    def step(self, t: float) -> None:
        """Pass inlet flow to outlet (simplified unity separation)."""
        self.dry_gas_outlet_kg_h = self.gas_inlet_kg_h

    def get_state(self) -> Dict[str, Any]:
        """Return component identifier."""
        return {"component_id": self.component_id}


class PSAUnit(Component):
    """
    Pressure Swing Adsorption unit for hydrogen purification.

    Removes residual water and oxygen from hydrogen product stream to
    achieve pipeline or fuel cell grade purity. Simplified model with
    pass-through behavior (no recovery loss modeled).

    Implements Component Lifecycle Contract.
    """

    def __init__(self, *args, **kwargs):
        """Initialize PSA unit with default zero flows."""
        super().__init__()
        self.feed_gas_kg_h = 0.0
        self.product_gas_kg_h = 0.0

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """Prepare component for simulation."""
        super().initialize(dt, registry)

    def step(self, t: float) -> None:
        """Pass feed to product (simplified unity recovery)."""
        self.product_gas_kg_h = self.feed_gas_kg_h

    def get_state(self) -> Dict[str, Any]:
        """Return component identifier."""
        return {"component_id": self.component_id}


class RectifierTransformer(Component):
    """
    Power electronics for AC/DC conversion.

    Converts AC grid power to regulated DC for electrolyzer stacks.
    Includes transformer, rectifier, and DC bus regulation. Simplified
    model assumes unity conversion efficiency.

    Implements Component Lifecycle Contract.

    Attributes:
        max_power_kw (float): Rated power capacity in kW.
        dc_output_kw (float): Current DC output power in kW.
        ac_input_kw (float): Current AC input power in kW.
    """

    def __init__(self, max_power_kw: float, *args, **kwargs):
        """
        Initialize rectifier-transformer unit.

        Args:
            max_power_kw (float): Maximum rated power capacity in kW.
        """
        super().__init__()
        self.max_power_kw = max_power_kw
        self.dc_output_kw = max_power_kw
        self.ac_input_kw = max_power_kw

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """Prepare component for simulation."""
        super().initialize(dt, registry)

    def step(self, t: float) -> None:
        """No-op step - power set externally."""
        pass

    def get_state(self) -> Dict[str, Any]:
        """Return component identifier."""
        return {"component_id": self.component_id}
