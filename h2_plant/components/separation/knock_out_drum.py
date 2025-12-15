"""
Knock-Out Drum (KOD) Separator Component.

This module implements a vertical gravity separator for removing entrained liquid
water droplets from gas streams. The KOD is a critical component in hydrogen and
oxygen processing trains, typically positioned downstream of a chiller or cooler.

Physical Principles:
    - **Souders-Brown Correlation**: Determines the maximum superficial gas velocity
      to prevent liquid re-entrainment. The correlation balances gravitational settling
      of droplets against drag forces from the gas flow.
    - **Rachford-Rice Equilibrium**: Used for vapor-liquid flash calculations when
      the feed contains condensable species (water vapor). Determines phase split
      based on K-values derived from Antoine or CoolProp saturation data.
    - **Isothermal Flash**: Assumes flash occurs at inlet temperature; pressure drop
      is treated as isenthalpic expansion through the vessel.

Architecture:
    Implements the Component Lifecycle Contract (Layer 1):
    - `initialize()`: Validates configuration and registers with component registry.
    - `step()`: Executes flash physics and prepares output streams per timestep.
    - `get_state()`: Exposes internal state for monitoring and persistence.

References:
    - Souders, M. & Brown, G.G. (1934). Design of Fractionating Columns.
    - Rachford, H.H. & Rice, J.D. (1952). Procedure for Use of Electronic
      Digital Computers in Calculating Flash Vaporization.
"""

from typing import Dict, Any, Optional, TYPE_CHECKING
import math

from h2_plant.core.component import Component
from h2_plant.core.stream import Stream
from h2_plant.core.constants import GasConstants, HenryConstants
from h2_plant.optimization.coolprop_lut import CoolPropLUT
from h2_plant.optimization.numba_ops import solve_rachford_rice_single_condensable

if TYPE_CHECKING:
    from h2_plant.core.component_registry import ComponentRegistry


# ============================================================================
# Physical Constants
# ============================================================================
RHO_L_WATER: float = 1000.0
"""Liquid water density at standard conditions (kg/m³)."""

K_SOUDERS_BROWN: float = 0.08
"""Souders-Brown K-factor for vertical separators with mesh demister (m/s).
Lower values are conservative; typical range 0.05-0.12 depending on demister type."""

R_UNIV: float = 8.31446
"""Universal gas constant (J/(mol·K))."""


class KnockOutDrum(Component):
    """
    Vertical gravity separator for liquid water removal from gas streams.

    The Knock-Out Drum operates as an isothermal flash separator where liquid
    water droplets settle by gravity while gas flows upward. Proper sizing
    ensures gas velocity remains below the Souders-Brown limit to prevent
    liquid carryover.

    This component fulfills the Component Lifecycle Contract (Layer 1):
        - `initialize()`: Validates vessel geometry and registers component.
        - `step()`: Computes flash equilibrium, phase separation, and sizing adequacy.
        - `get_state()`: Returns operational metrics for monitoring and diagnostics.

    Attributes:
        diameter_m (float): Vessel inner diameter in meters.
        delta_p_bar (float): Pressure drop across the vessel in bar.
        gas_species (str): Primary non-condensable gas species ('H2' or 'O2').
        separation_efficiency (float): Fraction of liquid captured (0.0 to 1.0).

    Example:
        >>> kod = KnockOutDrum(diameter_m=1.5, delta_p_bar=0.05, gas_species='H2')
        >>> kod.initialize(dt=1/60, registry=registry)
        >>> kod.receive_input('gas_inlet', wet_h2_stream, 'gas')
        >>> kod.step(t=0.0)
        >>> dry_gas = kod.get_output('gas_outlet')
    """

    def __init__(
        self,
        diameter_m: float = 1.0,
        delta_p_bar: float = 0.05,
        gas_species: str = 'H2'
    ) -> None:
        """
        Initialize the Knock-Out Drum component.

        Constructs a vertical separator with specified geometry and primary gas
        species. The diameter determines superficial velocity and thus the
        Souders-Brown adequacy check during operation.

        Args:
            diameter_m (float): Vessel inner diameter in meters. Must be positive.
                Typical industrial values range from 0.5 to 3.0 m. Default: 1.0 m.
            delta_p_bar (float): Pressure drop across vessel in bar. Represents
                losses through inlets, internals, and outlet. Default: 0.05 bar.
            gas_species (str): Primary non-condensable gas species. Must be 'H2'
                or 'O2'. This determines gas-phase physical properties (density,
                compressibility). Default: 'H2'.

        Raises:
            ValueError: If diameter_m is not positive.
            ValueError: If gas_species is not 'H2' or 'O2'.
        """
        super().__init__()

        if diameter_m <= 0:
            raise ValueError(f"diameter_m must be positive, got {diameter_m}")
        if gas_species not in ('H2', 'O2'):
            raise ValueError(f"gas_species must be 'H2' or 'O2', got {gas_species}")

        self.diameter_m = diameter_m
        self.delta_p_bar = delta_p_bar
        self.gas_species = gas_species
        self.separation_efficiency = 0.97 # Legacy: 0.97, was 0.98

        # Stream state: holds most recent input/output for current timestep
        self._input_stream: Optional[Stream] = None
        self._gas_outlet_stream: Optional[Stream] = None
        self._liquid_drain_stream: Optional[Stream] = None

        # Operational state for monitoring and diagnostics
        self._rho_g: float = 0.0
        self._v_max: float = 0.0
        self._v_real: float = 0.0
        self._separation_status: str = "IDLE"
        self._power_consumption_w: float = 0.0
        self._dissolved_gas_kg_h: float = 0.0

    def initialize(self, dt: float, registry: 'ComponentRegistry') -> None:
        """
        Prepare the component for simulation execution.

        Fulfills the Component Lifecycle Contract initialization phase by validating
        configuration parameters and establishing connections to the component
        registry for cross-component communication.

        Args:
            dt (float): Simulation timestep in hours.
            registry (ComponentRegistry): Central registry for component lookup
                and inter-component communication.

        Raises:
            ValueError: If diameter_m configuration is invalid at runtime.
        """
        super().initialize(dt, registry)

        if self.diameter_m <= 0:
            raise ValueError("diameter_m must be positive")

    def _calculate_dissolved_gas(self, temp_k: float, pressure_pa: float, gas_type: str) -> float:
        """
        Calculate gas solubility in liquid water using Henry's Law.

        Determines equilibrium concentration of dissolved gas in the liquid drain,
        which represents a product loss that must be accounted for in mass balance.

        Formula:
            H(T) = H_298 * exp(C * (1/T - 1/298.15))
            c_mol_L = P_gas_atm / H(T)
            c_mg_kg = c_mol_L * MW * 1000^2

        Args:
            temp_k (float): Temperature in Kelvin.
            pressure_pa (float): Total pressure in Pascals.
            gas_type (str): 'H2' or 'O2'.

        Returns:
            float: Solubility in mg/kg water.
        """
        if gas_type == 'H2':
            H_298 = HenryConstants.H2_H_298_L_ATM_MOL
            C = HenryConstants.H2_DELTA_H_R_K
            MW = HenryConstants.H2_MOLAR_MASS_KG_MOL
        elif gas_type == 'O2':
            H_298 = HenryConstants.O2_H_298_L_ATM_MOL
            C = HenryConstants.O2_DELTA_H_R_K
            MW = HenryConstants.O2_MOLAR_MASS_KG_MOL
        else:
            return 0.0
            
        T0 = 298.15
        if temp_k <= 0:
            return 0.0
        
        # Temperature-corrected Henry constant
        H_T = H_298 * math.exp(C * (1.0/temp_k - 1.0/T0))
        
        # Partial pressure in atm (assume gas phase is dominated by species)
        p_atm = pressure_pa / 101325.0
        
        # Molar concentration (mol/L)
        c_mol_L = p_atm / H_T
        
        # Convert to mass concentration (mg/kg, assuming water density ~1 kg/L)
        mw_g_mol = MW * 1000.0
        c_mg_L = c_mol_L * mw_g_mol * 1000.0
        
        return c_mg_L

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """
        Define the physical connection ports for this component.

        The KOD has one gas inlet and two outlets: a dry gas outlet and a
        liquid drain. Port definitions enable the orchestrator to validate
        and establish flow network connections.

        Returns:
            Dict[str, Dict[str, str]]: Port definitions with keys:
                - 'gas_inlet': Receives wet gas from upstream cooler.
                - 'gas_outlet': Delivers dried gas to downstream processing.
                - 'liquid_drain': Removes separated liquid water.
        """
        return {
            'gas_inlet': {
                'type': 'input',
                'resource_type': 'gas',
                'units': 'kg/h'
            },
            'gas_outlet': {
                'type': 'output',
                'resource_type': 'gas',
                'units': 'kg/h'
            },
            'liquid_drain': {
                'type': 'output',
                'resource_type': 'water',
                'units': 'kg/h'
            }
        }

    def receive_input(self, port_name: str, value: Any, resource_type: str = None) -> float:
        """
        Accept an input stream at the specified port.

        Stores the incoming Stream object for processing during the next step()
        call. Only the 'gas_inlet' port accepts input; other port names are
        silently ignored to allow flexible network topologies.

        Args:
            port_name (str): Target port identifier. Expected: 'gas_inlet'.
            value (Any): Stream object containing mass flow, temperature,
                pressure, and composition data.
            resource_type (str, optional): Resource classification hint.
                Not used for KOD but included for interface consistency.

        Returns:
            float: Mass flow rate accepted (kg/h), or 0.0 if input was rejected.
        """
        if port_name == 'gas_inlet' and isinstance(value, Stream):
            self._input_stream = value
            return value.mass_flow_kg_h
        return 0.0

    def step(self, t: float) -> None:
        """
        Execute one simulation timestep.

        Performs the core separator physics calculation sequence:
        1. Convert mass flow to molar basis for thermodynamic calculations.
        2. Compute vapor-liquid equilibrium using Rachford-Rice flash.
        3. Determine phase compositions and split fractions.
        4. Calculate gas density and superficial velocity for sizing check.
        5. Apply separation efficiency to determine carryover vs. capture.
        6. Prepare output streams for downstream components.

        This method fulfills the Component Lifecycle Contract step phase,
        advancing the component state by one discrete timestep.

        Args:
            t (float): Current simulation time in hours.

        Note:
            The Souders-Brown criterion (V_real < V_max) determines whether
            the separator is adequately sized. If exceeded, `_separation_status`
            is set to "UNDERSIZED", indicating potential liquid carryover.
        """
        super().step(t)

        inlet = self._input_stream
        if inlet is None or inlet.mass_flow_kg_h <= 0:
            self._gas_outlet_stream = None
            self._liquid_drain_stream = None
            self._separation_status = "NO_FLOW"
            self._rho_g = 0.0
            self._v_max = 0.0
            self._v_real = 0.0
            self._power_consumption_w = 0.0
            return

        # ====================================================================
        # A. Molar Basis Conversion
        # ====================================================================
        # Convert mass fractions to molar flows for thermodynamic calculations.
        # Molar basis is required for K-value equilibrium and Rachford-Rice.
        m_dot_in_kg_h = inlet.mass_flow_kg_h
        T_in = inlet.temperature_k
        P_in = inlet.pressure_pa
        composition = inlet.composition

        n_total_mol_s = 0.0
        n_species: Dict[str, float] = {}

        for species, mass_frac in composition.items():
            if species in GasConstants.SPECIES_DATA:
                mw_g_mol = GasConstants.SPECIES_DATA[species]['molecular_weight']
                mw_kg_mol = mw_g_mol / 1000.0
                # n_i = (mass_frac × total_mass_flow) / MW
                n_i = (mass_frac * m_dot_in_kg_h / 3600.0) / mw_kg_mol
                n_species[species] = n_i
                n_total_mol_s += n_i

        if n_total_mol_s <= 0:
            self._separation_status = "NO_FLOW"
            self._gas_outlet_stream = None
            self._liquid_drain_stream = None
            self._rho_g = 0.0
            self._v_max = 0.0
            self._v_real = 0.0
            self._power_consumption_w = 0.0
            return

        z_species: Dict[str, float] = {sp: n / n_total_mol_s for sp, n in n_species.items()}
        z_H2O = z_species.get('H2O', 0.0)

        # ====================================================================
        # B. Flash / Separation Physics
        # ====================================================================
        # Apply pressure drop as isenthalpic expansion, then determine
        # vapor-liquid equilibrium at outlet conditions.
        delta_p_pa = self.delta_p_bar * 1e5
        P_out = P_in - delta_p_pa
        if P_out <= 0:
            P_out = 1e5  # Safety floor at 1 bar

        # Obtain saturation pressure of water at inlet temperature.
        # The K-value (K = P_sat/P) determines condensation behavior.
        try:
            P_sat = CoolPropLUT.PropsSI('P', 'T', T_in, 'Q', 0.0, 'Water')
            if P_sat <= 1e-6 or not math.isfinite(P_sat):
                raise ValueError("CoolProp returned invalid P_sat")
        except Exception:
            # Antoine equation fallback: log10(P_mmHg) = A - B/(C + T_C)
            T_C = T_in - 273.15
            A, B, C = 8.07131, 1730.63, 233.426
            P_sat_mmHg = 10 ** (A - B / (C + T_C))
            P_sat = P_sat_mmHg * 133.322

        K_eq = P_sat / P_out if P_out > 0 else 1.0

        # Rachford-Rice solution for vapor fraction (beta).
        # Water is the only condensable; all other species remain in vapor phase.
        if z_H2O > 1e-12:
            beta = solve_rachford_rice_single_condensable(z_H2O, K_eq)
        else:
            beta = 1.0  # No water → all vapor

        # ====================================================================
        # Phase Composition Determination
        # ====================================================================
        # For beta ≈ 1 (superheated), vapor composition equals feed composition.
        # For beta < 1 (two-phase), vapor water content is limited by saturation.
        if beta >= 1.0 - 1e-9:
            y_H2O = z_H2O
        else:
            y_H2O = min(P_sat / P_out, 1.0) if P_out > 0 else 0.0

        y_H2O = max(0.0, min(y_H2O, 1.0))
        y_gas = 1.0 - y_H2O
        y_gas = max(0.0, min(y_gas, 1.0))

        # Liquid phase is assumed pure water (immiscible gas approximation)
        x_H2O = 1.0

        # ====================================================================
        # C. Fluid Properties & Sizing Check
        # ====================================================================
        # Calculate gas density and compare actual velocity to Souders-Brown limit.
        n_vap_mol_s = beta * n_total_mol_s
        n_liq_mol_s = (1.0 - beta) * n_total_mol_s

        # Mixture molar mass for gas phase
        mw_gas = GasConstants.SPECIES_DATA[self.gas_species]['molecular_weight'] / 1000.0
        mw_h2o = GasConstants.SPECIES_DATA['H2O']['molecular_weight'] / 1000.0
        M_mix = y_gas * mw_gas + y_H2O * mw_h2o

        # Real gas compressibility factor (Z) from equation of state
        try:
            Z = CoolPropLUT.PropsSI('Z', 'T', T_in, 'P', P_out, self.gas_species)
        except Exception:
            Z = 1.0  # Ideal gas approximation

        # Gas density: ρ = PM / (ZRT)
        if Z > 0 and T_in > 0:
            rho_g = (P_out * M_mix) / (Z * R_UNIV * T_in)
        else:
            rho_g = 0.0

        self._rho_g = rho_g

        # Volumetric flow and superficial velocity
        m_vap_kg_s = n_vap_mol_s * M_mix
        if rho_g > 0:
            V_dot_g_m3_s = m_vap_kg_s / rho_g
        else:
            V_dot_g_m3_s = 0.0

        # Souders-Brown maximum velocity: V_max = K * sqrt((ρ_L - ρ_G) / ρ_G)
        # This criterion balances droplet gravitational settling against gas drag.
        if rho_g > 0:
            density_diff = max(0.0, RHO_L_WATER - rho_g)
            v_max = K_SOUDERS_BROWN * math.sqrt(density_diff / rho_g)
        else:
            v_max = float('inf')

        self._v_max = v_max

        # Actual superficial velocity through vessel cross-section
        A_vessel = math.pi * (self.diameter_m / 2.0) ** 2
        if A_vessel > 0:
            v_real = V_dot_g_m3_s / A_vessel
        else:
            v_real = 0.0

        self._v_real = v_real

        # Sizing adequacy check
        if v_real < v_max:
            self._separation_status = "OK"
        else:
            self._separation_status = "UNDERSIZED"

        # ====================================================================
        # D. Power & Output Stream Generation
        # ====================================================================
        # Parasitic power represents theoretical work to restore pressure drop.
        # P = V̇ × ΔP (approximation for pneumatic losses)
        self._power_consumption_w = V_dot_g_m3_s * delta_p_pa

        # Mass flow rates with separation efficiency applied.
        # Efficiency < 1.0 means some liquid is carried over as mist.
        m_liquid_total_kg_h = n_liq_mol_s * mw_h2o * 3600.0

        m_liq_removed_kg_h = m_liquid_total_kg_h * self.separation_efficiency
        m_liq_carryover_kg_h = m_liquid_total_kg_h - m_liq_removed_kg_h

        m_dot_vap_kg_h = m_vap_kg_s * 3600.0

        # Gas outlet includes vapor and any liquid carryover as mist
        m_gas_out_total = m_dot_vap_kg_h + m_liq_carryover_kg_h
        
        # Calculate vapor-phase water mass flow from mole fraction and vapor flow
        m_H2O_vap = y_H2O * m_dot_vap_kg_h

        gas_comp = {}
        if m_gas_out_total > 0:
            # Non-condensables (H2, O2, N2) pass through with same MASS FLOW (not moles).
            # Their mass fraction increases as water is removed.
            # m_species_out = m_species_in (conservation)
            # x_species_out = m_species_in / m_total_out
            
            # Calculate mass of each non-H2O species from inlet
            for species, mass_frac in composition.items():
                if species == 'H2O' or species == 'H2O_liq':
                    continue  # Water handled separately
                # Mass of this species = inlet_mass_frac × inlet_total_mass
                m_species_kg_h = mass_frac * m_dot_in_kg_h
                # New mass fraction = species_mass / outlet_total_mass
                gas_comp[species] = m_species_kg_h / m_gas_out_total
            
            # Add vapor-phase water (equilibrium amount)
            # m_H2O_vap was calculated from flash equilibrium
            gas_comp['H2O'] = m_H2O_vap / m_gas_out_total
            
            # Add liquid carryover (mist)
            gas_comp['H2O_liq'] = m_liq_carryover_kg_h / m_gas_out_total
            
            # Normalize to ensure sum = 1.0 (floating point safety)
            total_comp = sum(gas_comp.values())
            if total_comp > 0 and abs(total_comp - 1.0) > 1e-6:
                gas_comp = {s: f / total_comp for s, f in gas_comp.items()}
        else:
            gas_comp = {self.gas_species: 1.0}

        self._gas_outlet_stream = Stream(
            mass_flow_kg_h=m_gas_out_total,
            temperature_k=T_in,
            pressure_pa=P_out,
            composition=gas_comp,
            phase='gas'
        )

        # Calculate dissolved gas loss using Henry's Law
        dissolved_gas_mg_kg = self._calculate_dissolved_gas(T_in, P_out, self.gas_species)
        dissolved_gas_kg_h = (m_liq_removed_kg_h * dissolved_gas_mg_kg) / 1e6
        self._dissolved_gas_kg_h = dissolved_gas_kg_h
        
        # Subtract dissolved gas from gas outlet flow
        m_gas_out_corrected = max(0.0, m_gas_out_total - dissolved_gas_kg_h)
        
        # Update gas outlet stream with corrected flow
        self._gas_outlet_stream = Stream(
            mass_flow_kg_h=m_gas_out_corrected,
            temperature_k=T_in,
            pressure_pa=P_out,
            composition=gas_comp,
            phase='gas'
        )

        self._liquid_drain_stream = Stream(
            mass_flow_kg_h=m_liq_removed_kg_h + dissolved_gas_kg_h,
            temperature_k=T_in,
            pressure_pa=P_out,
            composition={'H2O': 1.0, f'{self.gas_species}_dissolved': dissolved_gas_kg_h / (m_liq_removed_kg_h + dissolved_gas_kg_h + 1e-12)},
            phase='liquid'
        )

        # Clear input buffer for next timestep
        self._input_stream = None

    def get_output(self, port_name: str) -> Any:
        """
        Retrieve the output stream from a specified port.

        Provides access to the separated gas and liquid streams computed
        during the most recent step() execution.

        Args:
            port_name (str): Port to query. Must be 'gas_outlet' or 'liquid_drain'.

        Returns:
            Stream: Output Stream object containing mass flow, temperature,
                pressure, and composition. Returns None if no flow is available.

        Raises:
            ValueError: If port_name is not a valid output port.
        """
        if port_name == 'gas_outlet':
            return self._gas_outlet_stream if self._gas_outlet_stream else Stream(0.0)
        elif port_name == 'liquid_drain':
            return self._liquid_drain_stream if self._liquid_drain_stream else Stream(0.0)
        else:
            raise ValueError(f"Unknown output port '{port_name}'")

    def get_state(self) -> Dict[str, Any]:
        """
        Retrieve the component's current operational state.

        Fulfills the Component Lifecycle Contract state access, providing
        internal variables for monitoring dashboards, logging, and simulation
        state persistence.

        Returns:
            Dict[str, Any]: State dictionary containing:
                - rho_g (float): Gas phase density (kg/m³).
                - v_max (float): Souders-Brown maximum velocity (m/s).
                - v_real (float): Actual superficial velocity (m/s).
                - separation_status (str): "OK", "UNDERSIZED", "NO_FLOW", or "IDLE".
                - power_consumption_w (float): Parasitic power loss (W).
                - diameter_m (float): Vessel diameter (m).
                - delta_p_bar (float): Pressure drop (bar).
                - gas_species (str): Primary gas species ('H2' or 'O2').
        """
        state = super().get_state()
        state.update({
            'rho_g': self._rho_g,
            'v_max': self._v_max,
            'v_real': self._v_real,
            'separation_status': self._separation_status,
            'power_consumption_w': self._power_consumption_w,
            'dissolved_gas_kg_h': self._dissolved_gas_kg_h,
            'diameter_m': self.diameter_m,
            'delta_p_bar': self.delta_p_bar,
            'gas_species': self.gas_species,
            'water_removed_kg_h': self._liquid_drain_stream.mass_flow_kg_h if self._liquid_drain_stream else 0.0
        })
        return state
