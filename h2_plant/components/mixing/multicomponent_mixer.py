"""
Multi-Component Gas Mixer with Rigorous Thermodynamics.

This module implements a thermodynamically rigorous mixer for multi-species
gas streams. The mixer solves UV (internal energy, volume) flash calculations
to determine equilibrium temperature and pressure from the combined stream
properties.

Thermodynamic Model:
    - **UV Flash**: Given total internal energy U and volume V, solves for
      equilibrium temperature T such that U(T, V, n) = U_target. This is the
      thermodynamically correct approach for adiabatic mixing in a fixed volume.
    - **Ideal Gas Mixture**: Pressure computed from P = nRT/V after T is found.
    - **Heat Capacity Integration**: Shomate polynomials integrate Cp from
      reference temperature (298.15 K) to operating temperature for enthalpy.

Architecture:
    Implements the Component Lifecycle Contract (Layer 1):
    - `initialize()`: Connects to LUTManager and initializes internal energy.
    - `step()`: Processes input buffer, updates U, performs UV flash.
    - `get_state()`: Returns temperature, pressure, and mole inventories.

Species Handling:
    Supports O₂, CO₂, CH₄, H₂O, H₂, and N₂. Each species has thermodynamic
    data (formation enthalpy, Cp coefficients) stored in GasConstants.

Performance:
    UV flash solver is JIT-compiled via Numba for speed during simulation.
    Property arrays are pre-computed at initialization for O(1) access.

References:
    - Smith, Van Ness & Abbott (2005). Intro to Chemical Engineering
      Thermodynamics, 7th Ed., Chapter 12 (Phase Equilibrium).
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from scipy.optimize import brentq
import logging

from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.constants import GasConstants
from h2_plant.core.exceptions import ComponentStepError, FlashConvergenceError
from h2_plant.optimization.lut_manager import LUTManager

logger = logging.getLogger(__name__)


class MultiComponentMixer(Component):
    """
    Multi-species gas mixer with UV flash thermodynamics.

    Mixes gas streams from multiple sources while rigorously tracking
    internal energy and solving for equilibrium conditions. Supports
    pressure relief and heat loss modeling.

    This component fulfills the Component Lifecycle Contract (Layer 1):
        - `initialize()`: Pre-computes property arrays and connects to LUTManager.
        - `step()`: Accumulates inputs, performs UV flash, checks pressure relief.
        - `get_state()`: Returns temperature, pressure, and inventory metrics.

    The UV flash algorithm:
    1. Accumulate input enthalpy and moles from buffered streams.
    2. Apply heat loss to total internal energy.
    3. Solve T such that U_mixture(T) = U_total using Newton iteration.
    4. Calculate P from ideal gas law: P = nRT/V.

    Attributes:
        volume_m3 (float): Mixer vessel volume (m³).
        moles_stored (Dict[str, float]): Per-species mole inventory.
        temperature_k (float): Current equilibrium temperature (K).
        pressure_pa (float): Current equilibrium pressure (Pa).

    Example:
        >>> mixer = MultiComponentMixer(volume_m3=10.0, enable_phase_equilibrium=True)
        >>> mixer.initialize(dt=1/60, registry=registry)
        >>> mixer.receive_input('inlet', biogas_stream, 'gas')
        >>> mixer.step(t=0.0)
    """

    def __init__(
        self,
        volume_m3: float,
        enable_phase_equilibrium: bool = True,
        heat_loss_coeff_W_per_K: float = 0.0,
        pressure_relief_threshold_bar: float = 50.0,
        initial_temperature_k: float = 298.15,
    ):
        """
        Initialize the multi-component mixer.

        Args:
            volume_m3 (float): Internal volume of the mixer vessel in m³.
            enable_phase_equilibrium (bool): Enable vapor-liquid equilibrium
                calculations (not fully implemented). Default: True.
            heat_loss_coeff_W_per_K (float): Heat loss coefficient in W/K.
                Q_loss = UA × (T - T_ambient). Default: 0.0 (adiabatic).
            pressure_relief_threshold_bar (float): Pressure at which relief
                valve opens in bar. Default: 50.0.
            initial_temperature_k (float): Initial temperature in K.
                Default: 298.15 (25°C).
        """
        super().__init__()

        self.volume_m3 = volume_m3
        self.heat_loss_coeff = heat_loss_coeff_W_per_K
        self.pressure_relief_pa = pressure_relief_threshold_bar * 1e5
        self.enable_vle = enable_phase_equilibrium

        # Input buffer for push architecture
        self._input_buffer: List[Any] = []

        # Species inventory (moles)
        self.moles_stored = {'O2': 0.0, 'CO2': 0.0, 'CH4': 0.0, 'H2O': 0.0, 'H2': 0.0, 'N2': 0.0}
        self.total_internal_energy_J = 0.0

        # Equilibrium state
        self.temperature_k = initial_temperature_k
        self.pressure_pa = 1e5
        self.vapor_fraction = 1.0
        self.liquid_moles = {k: 0.0 for k in self.moles_stored}

        # Diagnostic counters
        self.cumulative_input_moles = 0.0
        self.cumulative_vented_moles = 0.0
        self.flash_convergence_failures = 0

        # Pre-compute property arrays for JIT solver
        self.species_keys = list(self.moles_stored.keys())
        n_species = len(self.species_keys)

        self._h_formations = np.zeros(n_species)
        self._cp_coeffs = np.zeros((n_species, 5))

        for i, s in enumerate(self.species_keys):
            data = GasConstants.SPECIES_DATA.get(s, GasConstants.SPECIES_DATA['N2'])
            self._h_formations[i] = data['h_formation']
            self._cp_coeffs[i, :] = data['cp_coeffs']

        self._lut_manager: Optional[LUTManager] = None

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """
        Prepare the mixer for simulation execution.

        Fulfills the Component Lifecycle Contract initialization phase.
        Connects to LUTManager if available and initializes internal energy
        from initial temperature.

        Args:
            dt (float): Simulation timestep in hours.
            registry (ComponentRegistry): Central registry for component access.
        """
        super().initialize(dt, registry)

        if registry.has('lut_manager'):
            self._lut_manager = registry.get('lut_manager')

        if sum(self.moles_stored.values()) > 0:
            self._initialize_internal_energy()

    def receive_input(self, port_name: str, value: Any, resource_type: str = None) -> float:
        """
        Accept input stream from upstream component.

        Buffers streams for processing during step(). Multiple inputs can
        be received per timestep.

        Args:
            port_name (str): Target port ('inlet' or 'gas_in').
            value (Any): Stream object with composition and flow.
            resource_type (str, optional): Resource classification hint.

        Returns:
            float: Mass flow accepted (kg/h), or 0.0 if rejected.
        """
        if port_name == 'inlet' or port_name == 'gas_in':
            if hasattr(value, 'mass_flow_kg_h') and value.mass_flow_kg_h > 0:
                self._input_buffer.append(value)
                return value.mass_flow_kg_h
        return 0.0

    def step(self, t: float) -> None:
        """
        Execute one simulation timestep.

        Processes buffered input streams:
        1. Converts mass flows to molar flows using molecular weights.
        2. Accumulates enthalpy contribution from each stream.
        3. Applies heat loss based on temperature difference from ambient.
        4. Solves UV flash to find new equilibrium temperature.
        5. Calculates pressure from ideal gas law.
        6. Activates pressure relief if threshold exceeded.

        Args:
            t (float): Current simulation time in hours.
        """
        super().step(t)
        dt_sec = self.dt * 3600.0

        total_enthalpy_in_J = 0.0
        total_moles_in = 0.0

        for stream in self._input_buffer:
            mass_flow_kg_s = stream.mass_flow_kg_h / 3600.0

            # Stream enthalpy contribution (J/s × dt_s = J)
            H_in_J_s = mass_flow_kg_s * stream.specific_enthalpy_j_kg
            total_enthalpy_in_J += H_in_J_s * dt_sec

            # Species mole accumulation
            for species, mass_frac in stream.composition.items():
                if species in self.moles_stored:
                    mass_flow_species_kg_s = mass_flow_kg_s * mass_frac
                    mw = GasConstants.SPECIES_DATA[species]['molecular_weight'] / 1000.0
                    moles_s = mass_flow_species_kg_s / mw

                    self.moles_stored[species] += moles_s * dt_sec
                    total_moles_in += moles_s * dt_sec

        # Clear buffer
        self._input_buffer = []

        # Heat loss to ambient (negative = heat leaving system)
        if self.heat_loss_coeff > 0:
            Q_loss = -self.heat_loss_coeff * (self.temperature_k - 298.15) * dt_sec
            total_enthalpy_in_J += Q_loss

        self.total_internal_energy_J += total_enthalpy_in_J
        self.cumulative_input_moles += total_moles_in

        # Solve UV flash for new temperature
        try:
            self._perform_uv_flash()
        except Exception as e:
            logger.error(f"UV-flash failed at t={t:.2f}h: {e}")
            self.flash_convergence_failures += 1

        # Pressure relief check
        if self.pressure_pa > self.pressure_relief_pa:
            self._activate_pressure_relief()

    def get_state(self) -> Dict[str, Any]:
        """
        Retrieve the component's current operational state.

        Fulfills the Component Lifecycle Contract state access.

        Returns:
            Dict[str, Any]: State dictionary containing:
                - temperature_k (float): Equilibrium temperature (K).
                - pressure_pa (float): Equilibrium pressure (Pa).
                - total_moles (float): Total mole inventory.
                - vapor_fraction (float): Vapor phase fraction (0-1).
        """
        total_moles = sum(self.moles_stored.values())
        return {
            **super().get_state(),
            'temperature_k': float(self.temperature_k),
            'pressure_pa': float(self.pressure_pa),
            'total_moles': float(total_moles),
            'vapor_fraction': float(self.vapor_fraction),
        }

    def _perform_uv_flash(self) -> None:
        """
        Solve UV flash for equilibrium temperature.

        Given fixed internal energy U and volume V, finds temperature T
        such that U_mixture(T) = U_target using Numba JIT-compiled solver.
        Updates pressure via ideal gas law: P = nRT/V.
        """
        total_moles = sum(self.moles_stored.values())
        if total_moles < 1e-12:
            return

        u_target_molar = self.total_internal_energy_J / total_moles

        mole_fractions = np.zeros(len(self.species_keys))
        for i, s in enumerate(self.species_keys):
            mole_fractions[i] = self.moles_stored[s] / total_moles

        from h2_plant.optimization import numba_ops

        self.temperature_k = numba_ops.solve_uv_flash(
            target_u_molar=u_target_molar,
            volume_m3=self.volume_m3,
            total_moles=total_moles,
            mole_fractions=mole_fractions,
            h_formations=self._h_formations,
            cp_coeffs_matrix=self._cp_coeffs,
            T_guess=self.temperature_k
        )

        # Ideal gas pressure calculation
        self.pressure_pa = (total_moles * GasConstants.R_UNIVERSAL_J_PER_MOL_K *
                           self.temperature_k) / self.volume_m3

    def _calc_internal_energy_vapor(
        self, T: float, P: float, composition: Dict[str, float]
    ) -> float:
        """
        Calculate molar internal energy for vapor phase.

        Uses departure function: U = H - RT for ideal gas.

        Args:
            T (float): Temperature in K.
            P (float): Pressure in Pa (unused for ideal gas).
            composition (Dict[str, float]): Mole fractions by species.

        Returns:
            float: Molar internal energy in J/mol.
        """
        h_mix = self._calculate_molar_enthalpy(T, P, composition, phase='vapor')
        u_mix = h_mix - GasConstants.R_UNIVERSAL_J_PER_MOL_K * T
        return u_mix

    def _calculate_molar_enthalpy(
        self, T: float, P: float, comp: Dict[str, float], phase: str = 'vapor'
    ) -> float:
        """
        Calculate molar enthalpy of mixture using Shomate polynomials.

        H = Σ(y_i × [H_f + ∫Cp dT])

        Args:
            T (float): Temperature in K.
            P (float): Pressure in Pa (unused for ideal gas).
            comp (Dict[str, float]): Mole fractions by species.
            phase (str): Phase identifier ('vapor'). Default: 'vapor'.

        Returns:
            float: Molar enthalpy in J/mol.
        """
        h_mix = 0.0
        for species, mole_frac in comp.items():
            if mole_frac > 1e-12:
                data = GasConstants.SPECIES_DATA[species]
                h_form = data['h_formation']
                delta_h = self._integrate_cp(data['cp_coeffs'], 298.15, T)
                h_mix += mole_frac * (h_form + delta_h)
        return h_mix

    def _integrate_cp(self, coeffs: List[float], T1: float, T2: float) -> float:
        """
        Integrate heat capacity from T1 to T2 using Shomate polynomial.

        Cp = A + BT + CT² + DT³ + E/T²
        ∫Cp dT = AT + BT²/2 + CT³/3 + DT⁴/4 - E/T

        Args:
            coeffs (List[float]): Shomate coefficients [A, B, C, D, E].
            T1 (float): Lower temperature bound in K.
            T2 (float): Upper temperature bound in K.

        Returns:
            float: Enthalpy change in J/mol.
        """
        A, B, C, D, E = coeffs

        def integral(T):
            return A*T + 0.5*B*T**2 + (1/3)*C*T**3 + 0.25*D*T**4 - (E/T if T > 0 else 0)

        return integral(T2) - integral(T1)

    def _activate_pressure_relief(self) -> None:
        """
        Activate pressure relief valve when threshold exceeded.

        Placeholder for venting logic. Full implementation would remove
        mass to reduce pressure below threshold.
        """
        pass

    def _initialize_internal_energy(self) -> None:
        """
        Calculate initial internal energy from temperature and inventory.

        Called during initialization when moles_stored is pre-populated.
        """
        total_moles = sum(self.moles_stored.values())
        if total_moles > 0:
            z = {k: v/total_moles for k, v in self.moles_stored.items()}
            u_molar = self._calc_internal_energy_vapor(self.temperature_k, self.pressure_pa, z)
            self.total_internal_energy_J = u_molar * total_moles

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """
        Define the physical connection ports for this component.

        Returns:
            Dict[str, Dict[str, str]]: Port definitions.
        """
        return {
            'inlet': {'type': 'input', 'resource_type': 'gas', 'units': 'kg/h'},
            'gas_in': {'type': 'input', 'resource_type': 'gas', 'units': 'kg/h'},
            'outlet': {'type': 'output', 'resource_type': 'gas', 'units': 'kg/h'}
        }
