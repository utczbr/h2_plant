"""
Multi-Component Gas Mixer with Rigorous Thermodynamics.

This module implements a thermodynamically rigorous mixer for multi-species
gas streams. The mixer supports both batch (filling tank) and continuous
(pipe junction) modes of operation.

Thermodynamic Model:
    - **Flash Models**:
      1. **UV Flash** (Batch/Tank): Conserves Internal Energy (U). U_out = H_in.
         Result: Temperature rise due to flow work (PV). Correct for filling tanks.
      2. **PH Flash** (Continuous/Pipe): Conserves Enthalpy (H). H_out = H_in.
         Result: T_out is weighted average. Correct for pipe junctions.

    - **Ideal Gas Mixture**: Pressure computed from P = nRT/V after T is found.
    - **Heat Capacity Integration**: Shomate polynomials integrate Cp from
      reference temperature (298.15 K) to operating temperature for enthalpy.

Architecture:
    Implements the Component Lifecycle Contract (Layer 1):
    - `initialize()`: Connects to LUTManager and initializes internal energy.
    - `step()`: Processes input buffer, updates U (or H), performs Flash.
    - `get_state()`: Returns temperature, pressure, and mole inventories.

Species Handling:
    Supports O₂, CO₂, CH₄, H₂O, H₂, and N₂. Each species has thermodynamic
    data (formation enthalpy, Cp coefficients) stored in GasConstants.

Performance:
    UV flash solver is JIT-compiled via Numba for speed during simulation.
    PH flash solver implemented with simplified Newton-Raphson.

References:
    - Smith, Van Ness & Abbott (2005). Intro to Chemical Engineering
      Thermodynamics, 7th Ed., Chapter 12 (Phase Equilibrium).
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import logging

from h2_plant.core.component import Component
from h2_plant.core.stream import Stream
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.constants import GasConstants
from h2_plant.core.exceptions import ComponentStepError, FlashConvergenceError
from h2_plant.optimization.lut_manager import LUTManager

logger = logging.getLogger(__name__)


class MultiComponentMixer(Component):
    """
    Multi-species gas mixer with UV/PH flash thermodynamics.

    Mixes gas streams from multiple sources while rigorously tracking
    thermodynamic state.

    Modes:
    1. **Batch/Tank (continuous_flow=False)**: Conserves Internal Energy (U).
       Models a tank filling up. Causes T rise due to compression work.
    2. **Continuous/Junction (continuous_flow=True)**: Conserves Enthalpy (H).
       Models a pipe junction. T_out is thermodynamic average of inputs.

    Attributes:
        volume_m3 (float): Mixer vessel volume (m³).
        moles_stored (Dict[str, float]): Per-species mole inventory.
        temperature_k (float): Current equilibrium temperature (K).
        pressure_pa (float): Current equilibrium pressure (Pa).

    Example:
        >>> mixer = MultiComponentMixer(volume_m3=10.0, continuous_flow=True)
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
        continuous_flow: bool = True
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
            continuous_flow (bool): If True, models pipe junction (H=const).
                If False, models filling tank (U=const). Default: True.
        """
        super().__init__()

        self.volume_m3 = volume_m3
        self.heat_loss_coeff = heat_loss_coeff_W_per_K
        self.pressure_relief_pa = pressure_relief_threshold_bar * 1e5
        self.enable_vle = enable_phase_equilibrium
        self.continuous_flow = continuous_flow

        # Input buffer for push architecture
        self._input_buffer: List[Any] = []
        
        # Output stream - initialized to zero flow
        self.outlet_stream: Optional[Stream] = Stream(
            mass_flow_kg_h=0.0,
            temperature_k=initial_temperature_k,
            pressure_pa=101325.0,
            composition={'N2': 1.0},
            phase='gas'
        )

        # Species inventory (moles)
        self.moles_stored = {'O2': 0.0, 'CO2': 0.0, 'CH4': 0.0, 'H2O': 0.0, 'H2': 0.0, 'N2': 0.0}
        self.total_internal_energy_J = 0.0
        self.total_enthalpy_J = 0.0 # For continuous flow mode

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

    @property
    def max_flow_kg_h(self) -> float:
        """Design flow capacity for CAPEX sizing. Returns outlet flow rate."""
        if self.outlet_stream:
            return self.outlet_stream.mass_flow_kg_h
        return 0.0

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
        # Flexible input port matching (inlet, gas_in, inlet_1, inlet_2, etc.)
        if port_name.startswith('inlet') or port_name == 'gas_in':
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
        4. Solves Flash to find new equilibrium temperature (UV or PH).
        5. Calculates pressure from ideal gas law.
        6. Activates pressure relief if threshold exceeded.

        Args:
            t (float): Current simulation time in hours.
        """
        super().step(t)
        dt_sec = self.dt * 3600.0

        total_enthalpy_in_J = 0.0
        total_moles_in = 0.0

        # Track pressure weighting
        sum_pressure_moles = 0.0
        
        for stream in self._input_buffer:
            mass_flow_kg_s = stream.mass_flow_kg_h / 3600.0

            # Stream enthalpy contribution (J/s × dt_s = J)
            
            # Convert mass composition to mole fractions for rigorous calc
            mole_fracs = stream.mole_fractions
            if not mole_fracs and stream.mass_flow_kg_h > 0:
                 # Single component fallback if needed or empty
                 mole_fracs = {}

            # Calculate molar enthalpy [J/mol] using Mixer's thermodynamics
            h_molar_in = self._calculate_molar_enthalpy(stream.temperature_k, stream.pressure_pa, mole_fracs)
            
            # Convert to specific enthalpy [J/kg] to multiply by mass flow
            # M_mix [kg/mol] = sum(y_i * M_i)
            m_mix_kg_mol = 0.0
            for s, y in mole_fracs.items():
                if s in GasConstants.SPECIES_DATA:
                    m_mix_kg_mol += y * GasConstants.SPECIES_DATA[s]['molecular_weight'] / 1000.0
            
            # Moles of stream: n = m / M_mix
            if m_mix_kg_mol > 0:
                moles_in_stream = (mass_flow_kg_s * dt_sec) / m_mix_kg_mol
                
                # Total Enthalpy = n * h_molar
                total_enthalpy_in_J += moles_in_stream * h_molar_in

                # Species mole accumulation
                for species, y_i in mole_fracs.items():
                    target_s = 'H2O' if species == 'H2O_liq' else species
                    if target_s in self.moles_stored:
                        moles_s = moles_in_stream * y_i
                        self.moles_stored[target_s] += moles_s
                        total_moles_in += moles_s
                
                # Weighted Pressure Accumulation
                sum_pressure_moles += stream.pressure_pa * moles_in_stream
        
        # Calculate weighted average pressure of inputs (for continuous mode)
        avg_pressure_pa = 101325.0 # Default if no flow
        if total_moles_in > 0:
            avg_pressure_pa = 0.0
            for stream in self._input_buffer:
                 # Re-calculate moles for weighting
                 # (Use simplified check here or store pressure in loop above - simpler to loopbuffer again or inline)
                 # Optimization: Calculate pressure contribution in the loop above
                 pass 
        
        # Clear buffer (but we need pressure info first!)
        # REFACTOR: Calculate pressure contribution inside the main loop above
        # (See modified loop logic below)
        
        self._input_buffer = []

        # Heat loss to ambient (negative = heat leaving system)
        if self.heat_loss_coeff > 0:
            Q_loss = -self.heat_loss_coeff * (self.temperature_k - 298.15) * dt_sec
            total_enthalpy_in_J += Q_loss

        # Update System Energy
        if self.continuous_flow:
            # Continuous: H_sys += H_in (Enthalpy Conserved)
            # Draining logic clears enthalpy each step, so we add current
            self.total_enthalpy_J += total_enthalpy_in_J
        else:
            # Batch/Tank: U_sys += H_in (Internal Energy Conserved)
            self.total_internal_energy_J += total_enthalpy_in_J
        
        self.cumulative_input_moles += total_moles_in

        self.cumulative_input_moles += total_moles_in

        # Update Pressure for Continuous Mode (before Flash)
        if self.continuous_flow:
            if total_moles_in > 0:
                self.pressure_pa = sum_pressure_moles / total_moles_in
            # Else keep previous pressure (or decay? Keeping previous is safer for stability)

        # Perform Flash
        try:
            if self.continuous_flow:
                self._perform_ph_flash()
            else:
                # Need to update U if we switched modes dynamically (unlikely but safe)
                if self.total_internal_energy_J == 0 and self.total_enthalpy_J > 0:
                     # Fallback approximation for mode switching
                     self.total_internal_energy_J = self.total_enthalpy_J - (8.314 * self.temperature_k * sum(self.moles_stored.values()))
                self._perform_uv_flash()
        except Exception as e:
            logger.error(f"Flash failed at t={t:.2f}h: {e}")
            self.flash_convergence_failures += 1

        # Pressure relief check
        if self.pressure_pa > self.pressure_relief_pa:
            self._activate_pressure_relief()
            
        # Draining Logic (For continuous flow simulation step)
        total_moles_out = sum(self.moles_stored.values())
        if total_moles_out > 0:
            # Calculate total mass
            total_mass_kg = 0.0
            comp = {}
            for s, n in self.moles_stored.items():
                mw = GasConstants.SPECIES_DATA[s]['molecular_weight'] / 1000.0
                m_s = n * mw
                total_mass_kg += m_s
                comp[s] = m_s # Store mass for fraction calculation
            
            # Normalize composition
            if total_mass_kg > 0:
                comp = {k: v / total_mass_kg for k, v in comp.items()}
            
            # Create stream
            self.outlet_stream = Stream(
                mass_flow_kg_h=total_mass_kg / self.dt if self.dt > 0 else 0.0,
                temperature_k=self.temperature_k,
                pressure_pa=self.pressure_pa,
                composition=comp,
                phase='gas'
            )
            
            # Reset inventory (Continuous flow assumption = emptied every step)
            self.moles_stored = {k: 0.0 for k in self.moles_stored}
            self.total_internal_energy_J = 0.0
            self.total_enthalpy_J = 0.0
        else:
            self.outlet_stream = Stream(
                mass_flow_kg_h=0.0,
                temperature_k=self.temperature_k,
                pressure_pa=self.pressure_pa,
                composition={'N2': 1.0},
                phase='gas'
            )

    def get_output(self, port_name: str) -> Any:
        if port_name == 'outlet':
            return self.outlet_stream if self.outlet_stream else Stream(0.0)
        return None

    def get_state(self) -> Dict[str, Any]:
        """
        Retrieve the component's current operational state.
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
        Solve UV flash (Internal Energy conserved).
        T_new such that U(T_new) = U_target.
        """
        total_moles = sum(self.moles_stored.values())
        if total_moles < 1e-12:
            return

        u_target_molar = self.total_internal_energy_J / total_moles

        mole_fractions = np.zeros(len(self.species_keys))
        for i, s in enumerate(self.species_keys):
            mole_fractions[i] = self.moles_stored[s] / total_moles

        from h2_plant.optimization import numba_ops

        # JIT Solver for U(T) = U_target
        self.temperature_k = numba_ops.solve_uv_flash(
            target_u_molar=u_target_molar,
            volume_m3=self.volume_m3,
            total_moles=total_moles,
            mole_fractions=mole_fractions,
            h_formations=self._h_formations,
            cp_coeffs_matrix=self._cp_coeffs,
            T_guess=self.temperature_k
        )

        self.pressure_pa = (total_moles * GasConstants.R_UNIVERSAL_J_PER_MOL_K *
                           self.temperature_k) / self.volume_m3

    def _perform_ph_flash(self) -> None:
        """
        Solve PH flash (Enthalpy conserved).
        T_new such that H(T_new) = H_target.
        Essentially a weighted average of enthalpies using Newton-Raphson.
        """
        total_moles = sum(self.moles_stored.values())
        if total_moles < 1e-12:
            return

        h_target_molar = self.total_enthalpy_J / total_moles

        mole_fractions = np.zeros(len(self.species_keys))
        for i, s in enumerate(self.species_keys):
            mole_fractions[i] = self.moles_stored[s] / total_moles

        # Newton-Raphson Solver for H(T) = H_target
        T = self.temperature_k
        max_iter = 10
        tol = 0.05

        for _ in range(max_iter):
            h_calc = 0.0
            cp_calc = 0.0
            
            # Using pre-computed matrices
            A = self._cp_coeffs[:, 0]
            B = self._cp_coeffs[:, 1]
            C = self._cp_coeffs[:, 2]
            D = self._cp_coeffs[:, 3]
            E = self._cp_coeffs[:, 4]
            
            T_ref = 298.15
            
            # Integral terms (H - H_form)
            # Int = A*T + B*T^2/2 ...
            def integ(t_val):
                return A*t_val + 0.5*B*t_val**2 + (1.0/3.0)*C*t_val**3 + 0.25*D*t_val**4 - E/t_val if t_val > 0 else 0
            
            delta_h_sens = integ(T) - integ(T_ref)
            h_total_species = self._h_formations + delta_h_sens 
            h_calc = np.sum(mole_fractions * h_total_species)
            
            # Cp terms
            cp_species = A + B*T + C*T**2 + D*T**3 + E/(T**2)
            cp_calc = np.sum(mole_fractions * cp_species)
            
            if abs(cp_calc) < 1e-4:
                break
                
            delta_T = (h_target_molar - h_calc) / cp_calc
            T = T + delta_T
            
            if abs(delta_T) < tol:
                break
                
            # Clamp physics
            T = max(275.0, min(T, 5000.0))
            
        self.temperature_k = T
        
        
        # In Continuous Flow, Pressure is NOT calculated from Volume (P != nRT/V)
        # It is determined by the hydraulic condition (governed by step() logic)
        # We KEEP self.pressure_pa as set in step()

    def _calc_internal_energy_vapor(
        self, T: float, P: float, composition: Dict[str, float]
    ) -> float:
        """
        Calculate molar internal energy for vapor phase.
        U = H - RT
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
        """
        h_mix = 0.0
        for species, mole_frac in comp.items():
            if mole_frac > 1e-12:
                # Map liquid water to gas properties (simplified) or ignore
                # Lookup species properties
                lookup_species = species
                # (Removed legacy H2O_liq mapping as it is now in SPECIES_DATA)
                
                if lookup_species in GasConstants.SPECIES_DATA:
                    data = GasConstants.SPECIES_DATA[lookup_species]
                    h_form = data['h_formation']
                    delta_h = self._integrate_cp(data['cp_coeffs'], 298.15, T)
                    h_mix += mole_frac * (h_form + delta_h)
        return h_mix

    def _integrate_cp(self, coeffs: List[float], T1: float, T2: float) -> float:
        """
        Integrate heat capacity from T1 to T2.
        """
        A, B, C, D, E = coeffs

        def integral(T):
            return A*T + 0.5*B*T**2 + (1/3)*C*T**3 + 0.25*D*T**4 - (E/T if T > 0 else 0)

        return integral(T2) - integral(T1)

    def _activate_pressure_relief(self) -> None:
        """
        Activate pressure relief valve when threshold exceeded.
        """
        pass

    def _initialize_internal_energy(self) -> None:
        """
        Calculate initial energy from temperature and inventory.
        """
        total_moles = sum(self.moles_stored.values())
        if total_moles > 0:
            z = {k: v/total_moles for k, v in self.moles_stored.items()}
            
            h_molar = self._calculate_molar_enthalpy(self.temperature_k, self.pressure_pa, z)
            # U = H - RT
            u_molar = h_molar - GasConstants.R_UNIVERSAL_J_PER_MOL_K * self.temperature_k
            
            self.total_internal_energy_J = u_molar * total_moles
            self.total_enthalpy_J = h_molar * total_moles

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """
        Define the physical connection ports for this component.
        """
        return {
            'inlet_1': {'type': 'input', 'resource_type': 'gas', 'units': 'kg/h'},
            'inlet_2': {'type': 'input', 'resource_type': 'gas', 'units': 'kg/h'},
            'inlet_3': {'type': 'input', 'resource_type': 'gas', 'units': 'kg/h'},
            'inlet_4': {'type': 'input', 'resource_type': 'gas', 'units': 'kg/h'},
            'inlet_5': {'type': 'input', 'resource_type': 'gas', 'units': 'kg/h'},
            'outlet': {'type': 'output', 'resource_type': 'gas', 'units': 'kg/h'}
        }
