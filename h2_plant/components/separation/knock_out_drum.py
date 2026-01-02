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
import logging

from h2_plant.core.component import Component
from h2_plant.core.stream import Stream
from h2_plant.core.constants import GasConstants, HenryConstants
from h2_plant.optimization.coolprop_lut import CoolPropLUT
from h2_plant.optimization.numba_ops import solve_rachford_rice_single_condensable

# Import mixture thermodynamics for rigorous density calculations
try:
    from h2_plant.optimization import mixture_thermodynamics as mix_thermo
    MIX_THERMO_AVAILABLE = True
except ImportError:
    mix_thermo = None
    MIX_THERMO_AVAILABLE = False

logger = logging.getLogger(__name__)

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

M_H2O: float = 0.018015
"""Water molar mass (kg/mol) - matches CoalescerConstants.M_H2O."""


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
        gas_species: str = 'H2',
        separation_efficiency: float = 0.97
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
            separation_efficiency (float): Fraction of liquid captured (0.0 to 1.0).
                Default: 0.97 (legacy model value).

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
        self.separation_efficiency = separation_efficiency

        # Molar mass of primary gas (kg/mol) - from GasConstants.SPECIES_DATA
        # molecular_weight is in g/mol, divide by 1000 for kg/mol
        self.M_gas = GasConstants.SPECIES_DATA[gas_species]['molecular_weight'] / 1000.0

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
        self.total_drain_removed_kg: float = 0.0  # Cumulative drain tracking
        
        # Mass Balance Tracking (Input vs Output)
        self._last_dissolved_gas_in_kg_h: float = 0.0
        self._last_dissolved_gas_out_kg_h: float = 0.0
        self._total_liquid_in_kg_h: float = 0.0

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
        
        # Reset cumulative tracking
        self.total_drain_removed_kg = 0.0

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
        call. Also calculates the potential dissolved gas (IN) based on inlet
        liquid content and equilibrium conditions.

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
            
            # --- Mass Balance Tracking (IN) ---
            m_dot_total = value.mass_flow_kg_h
            
            # 1. Total Incoming Liquid (Thermodynamic + Entrained)
            liq_frac = value.composition.get('H2O_liq', 0.0)
            m_liq_thermo = m_dot_total * liq_frac
            
            m_liq_entrained = 0.0
            if hasattr(value, 'extra') and value.extra:
                m_liq_entrained = value.extra.get('m_dot_H2O_liq_accomp_kg_s', 0.0) * 3600.0
            
            total_liquid_in = m_liq_thermo + m_liq_entrained
            self._total_liquid_in_kg_h = total_liquid_in
            
            # 2. Calculate Inlet Dissolved Gas Potential (equilibrium at inlet P, T)
            #    Must convert MASS fractions to MOLE fractions for correct partial pressure!
            #    Using mass fractions directly would underestimate p_gas for light H2.
            if total_liquid_in > 0 and value.pressure_pa > 0:
                # Molecular weights (kg/mol)
                MW = {'H2': 2.016e-3, 'O2': 32.00e-3, 'H2O': 18.015e-3, 'H2O_liq': 18.015e-3}
                MW_gas = MW.get(self.gas_species, 2.016e-3)
                
                # Convert mass fractions to mole fractions
                # n_i = x_i / M_i  =>  y_i = n_i / sum(n_i)
                # CRITICAL: Exclude liquid species from gas phase mole fraction!
                n_rel = {}
                for species, x_mass in value.composition.items():
                    if species.endswith('_liq'):  # e.g. 'H2O_liq'
                        continue
                    mw_i = MW.get(species, 28.0e-3)  # Default ~N2 for unknowns
                    n_rel[species] = x_mass / mw_i
                
                total_n = sum(n_rel.values())
                if total_n > 0:
                    # Mole fraction of primary gas
                    y_gas_mol = n_rel.get(self.gas_species, 0.0) / total_n
                    # Partial pressure using MOLE fraction (correct thermodynamics)
                    p_gas_partial = value.pressure_pa * y_gas_mol
                else:
                    p_gas_partial = 0.0
                
                solubility_mg_kg = self._calculate_dissolved_gas(
                    value.temperature_k, p_gas_partial, self.gas_species
                )
                
                self._last_dissolved_gas_in_kg_h = total_liquid_in * (solubility_mg_kg / 1e6)
            else:
                self._last_dissolved_gas_in_kg_h = 0.0
            # -----------------------------------
            
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
            if species == 'H2O_liq':
                continue
            if species in GasConstants.SPECIES_DATA:
                mw_g_mol = GasConstants.SPECIES_DATA[species]['molecular_weight']
                mw_kg_mol = mw_g_mol / 1000.0
                # n_i = (mass_frac × total_mass_flow) / MW
                n_i = (mass_frac * m_dot_in_kg_h / 3600.0) / mw_kg_mol
                n_species[species] = n_i
                n_total_mol_s += n_i

        # IMPORTANT: Capture entrained H2O_liq DIRECTLY from composition.
        # The upstream component (PEM, Chiller) outputs liquid mass directly in
        # composition['H2O_liq']. This is the single source of truth for liquid.
        h2o_liq_mass_frac = composition.get('H2O_liq', 0.0)
        m_entrained_liq_kg_h = h2o_liq_mass_frac * m_dot_in_kg_h
        self._entrained_liq_from_upstream_kg_h = m_entrained_liq_kg_h
        
        # Do NOT add H2O_liq to molar count for flash - it's already liquid!

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
            # Clamp beta to valid range [0, 1] for numerical stability
            beta = max(0.0, min(1.0, beta))
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

        # Mixture molar mass for gas phase - compute from ALL species (not just H2+H2O)
        # M_mix = Σ(y_i × M_i) where y_i is vapor mole fraction
        mw_h2o = GasConstants.SPECIES_DATA['H2O']['molecular_weight'] / 1000.0
        M_mix = 0.0
        for species, z_i in z_species.items():
            if species in GasConstants.SPECIES_DATA:
                mw_i = GasConstants.SPECIES_DATA[species]['molecular_weight'] / 1000.0
                # For non-condensables, vapor mole fraction = feed mole fraction
                # For water, vapor mole fraction = y_H2O (saturated)
                if species == 'H2O':
                    y_i = y_H2O
                else:
                    # Scale non-condensables to fill remaining vapor composition
                    # y_i = z_i × (1 - y_H2O) / (1 - z_H2O) when z_H2O ≠ 1
                    if z_H2O < 1.0:
                        y_i = z_i * (1.0 - y_H2O) / (1.0 - z_H2O)
                    else:
                        y_i = 0.0
                M_mix += y_i * mw_i

        # Real gas compressibility factor (Z) - use LUTManager for consistency
        Z = 1.0  # Default ideal gas
        lut_manager = None
        if hasattr(self, 'registry') and self.registry is not None:
            try:
                lut_manager = self.registry.get('lut_manager')
                if lut_manager is not None:
                    Z = lut_manager.lookup(self.gas_species, 'Z', P_out, T_in)
            except Exception:
                pass 
        if Z == 1.0:  # Fallback to CoolPropLUT if LUTManager unavailable
            try:
                Z = CoolPropLUT.PropsSI('Z', 'T', T_in, 'P', P_out, self.gas_species)
            except Exception:
                Z = 1.0

        # Gas density: PREFER rigorous mixture thermodynamics (Amagat's Law)
        # PERFORMANCE: Skip mix_thermo for near-pure streams (>98% single species)
        rho_g = 0.0
        dominant_z = max(z_species.values()) if z_species else 0
        use_mix_thermo = (dominant_z < 0.98 and MIX_THERMO_AVAILABLE and mix_thermo is not None and lut_manager is not None)
        
        if use_mix_thermo:
            try:
                # Build vapor composition mass fractions from mole fractions
                # y_i → x_i: x_i = y_i * M_i / M_mix
                vapor_comp_mass = {}
                for species, z_i in z_species.items():
                    if species in GasConstants.SPECIES_DATA:
                        mw_i = GasConstants.SPECIES_DATA[species]['molecular_weight'] / 1000.0
                        if species == 'H2O':
                            y_i = y_H2O
                        else:
                            if z_H2O < 1.0:
                                y_i = z_i * (1.0 - y_H2O) / (1.0 - z_H2O)
                            else:
                                y_i = 0.0
                        if M_mix > 0:
                            vapor_comp_mass[species] = y_i * mw_i / M_mix
                
                # Normalize
                total_mass = sum(vapor_comp_mass.values())
                if total_mass > 0:
                    vapor_comp_mass = {s: f/total_mass for s, f in vapor_comp_mass.items()}
                    rho_g = mix_thermo.get_mixture_density(
                        vapor_comp_mass, P_out, T_in, lut_manager
                    )
            except Exception:
                pass  # Fall through to Z-factor method
        
        # Fallback: Z-factor method: ρ = PM / (ZRT)
        if rho_g <= 0:
            if Z > 0 and T_in > 0 and M_mix > 0:
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
        # UPDATED PHYSICS: Use Mist Carryover Limit (mg/Nm³) instead of fixed efficiency.
        # This correctly handles bulk liquid separation where 97% efficiency is non-physical.
        
        m_flash_liquid_kg_h = n_liq_mol_s * mw_h2o * 3600.0
        m_total_liquid_kg_h = m_flash_liquid_kg_h + m_entrained_liq_kg_h

        # Calculate Standard Volumetric Flow (Nm³/h) for Mist Limit
        # Density at STP (0°C, 1 atm)
        rho_stp = 0.0
        if self.gas_species == 'H2':
             rho_stp = 0.08988 # kg/m3
        elif self.gas_species == 'O2':
             rho_stp = 1.429 # kg/m3
        else: # Default air-like
             rho_stp = 1.29 
             
        # m_vap_kg_s is pure vapor mass flow
        m_vap_kg_h = m_vap_kg_s * 3600.0
        if rho_stp > 0:
             Q_gas_nm3_h = m_vap_kg_h / rho_stp
        else:
             Q_gas_nm3_h = 0.0

        # MIST LIMIT: 20 mg/Nm³ (Legacy/Industrial Standard)
        MIST_LIMIT_MG_NM3 = 20.0
        max_mist_carryover_kg_h = Q_gas_nm3_h * (MIST_LIMIT_MG_NM3 / 1e6)
        
        # Apply Logic based on Sizing
        if self._separation_status == "OK":
            # Proper gravity separation: Carryover is limited by mist entrainment
            m_liq_carryover_kg_h = min(m_total_liquid_kg_h, max_mist_carryover_kg_h)
            m_liq_removed_kg_h = m_total_liquid_kg_h - m_liq_carryover_kg_h
        else:
            # Undersized/Flooded: Fallback to poor efficiency (e.g. 50% or fixed factor)
            # Legacy fallback was implicit; here we penalize significantly
            m_liq_removed_kg_h = m_total_liquid_kg_h * 0.50 # 50% carryover penalty
            m_liq_carryover_kg_h = m_total_liquid_kg_h - m_liq_removed_kg_h
            
        m_dot_vap_kg_h = m_vap_kg_h
        
        # Gas outlet includes vapor and any liquid carryover as mist
        m_gas_out_total = m_dot_vap_kg_h + m_liq_carryover_kg_h
        
        # Calculate vapor-phase water mass flow from molar basis
        # y_H2O is mole fraction, so: m_H2O_vap = n_vap × y_H2O × MW_H2O
        n_H2O_vap_mol_s = n_vap_mol_s * y_H2O
        m_H2O_vap = n_H2O_vap_mol_s * mw_h2o * 3600.0  # kg/h

        if m_gas_out_total > 0:
            # Non-condensables (H2, O2, N2) mass balance
            # m_species_out = m_species_in (conservation)
            
            # Calculate mass of each species from inlet
            species_masses = {}
            for species, mass_frac in composition.items():
                species_masses[species] = mass_frac * m_dot_in_kg_h
            
            # Update Water Vapor Mass (from Flash Equilibrium)
            species_masses['H2O'] = m_H2O_vap
            
            # Update Liquid Water Mass in Gas Stream (Carryover Mist)
            # The rest of the liquid water was removed to drain
            species_masses['H2O_liq'] = m_liq_carryover_kg_h
            
            # Reconstruct composition fractions
            gas_comp = {}
            total_mass_check = sum(species_masses.values())
            
            # Sanity check: Total mass should match m_gas_out_total
            # Difference is due to removed liquid (which we excluded from the gas stream's new masses)
            # Actually, m_gas_out_total = m_dot_in - m_liq_removed.
            # And species_masses['H2O_liq'] is now REDUCED to carryover.
            # So sum(species_masses) should roughly equal m_gas_out_total.
            
            if m_gas_out_total > 1e-9:
                for s, m in species_masses.items():
                    gas_comp[s] = m / m_gas_out_total
            
            # Normalize to ensure exactly 1.0
            total_comp = sum(gas_comp.values())
            if total_comp > 0:
                gas_comp = {s: f / total_comp for s, f in gas_comp.items()}
        else:
            gas_comp = {self.gas_species: 1.0}

        # Calculate dissolved gas loss using Henry's Law with PARTIAL PRESSURE
        # Must convert mass fractions to mole fractions for correct partial pressure!
        MW = {'H2': 2.016e-3, 'O2': 32.00e-3, 'H2O': 18.015e-3, 'H2O_liq': 18.015e-3}
        n_rel_out = {}
        for species, x_mass in gas_comp.items():
            mw_i = MW.get(species, 28.0e-3)
            n_rel_out[species] = x_mass / mw_i
        total_n_out = sum(n_rel_out.values())
        if total_n_out > 0:
            y_gas_mol = n_rel_out.get(self.gas_species, 0.0) / total_n_out
            p_gas_pa = P_out * y_gas_mol
        else:
            p_gas_pa = 0.0
        
        dissolved_gas_mg_kg = self._calculate_dissolved_gas(T_in, p_gas_pa, self.gas_species)
        dissolved_gas_kg_h = (m_liq_removed_kg_h * dissolved_gas_mg_kg) / 1e6
        self._dissolved_gas_kg_h = dissolved_gas_kg_h
        self._last_dissolved_gas_out_kg_h = dissolved_gas_kg_h  # Track OUT for mass balance
        
        # Subtract dissolved gas from gas outlet flow
        m_gas_out_corrected = max(0.0, m_gas_out_total - dissolved_gas_kg_h)
        
        # Update gas outlet stream with corrected flow
        self._gas_outlet_stream = Stream(
            mass_flow_kg_h=m_gas_out_corrected,
            temperature_k=T_in,
            pressure_pa=P_out,
            composition=gas_comp,
            phase='gas',
            extra={} # Do not pass carryover in extra, it is already in mass_flow
        )

        # Drain stream: simplify composition if dissolved gas is negligible
        m_drain_total = m_liq_removed_kg_h + dissolved_gas_kg_h
        if dissolved_gas_kg_h < 1e-9 * m_liq_removed_kg_h or m_liq_removed_kg_h <= 0:
            drain_comp = {'H2O_liq': 1.0}
        else:
            h2o_frac = m_liq_removed_kg_h / m_drain_total
            gas_frac = dissolved_gas_kg_h / m_drain_total
            drain_comp = {'H2O_liq': h2o_frac, self.gas_species: gas_frac}

        self._liquid_drain_stream = Stream(
            mass_flow_kg_h=m_drain_total,
            temperature_k=T_in,
            pressure_pa=P_out,
            composition=drain_comp,
            phase='liquid',
            extra={'m_dot_H2O_liq_accomp_kg_s': 0.0}
        )

        # Accumulate drain for tracking
        self.total_drain_removed_kg += m_drain_total * self.dt

        # Mass balance debug logging
        mass_balance_error = m_dot_in_kg_h - m_gas_out_corrected - m_drain_total
        
        # DEBUG: Trace KOD operation if flow is significant
        if m_dot_in_kg_h > 1.0 and abs(mass_balance_error) > 1e-3:
            logger.debug(
                f"KOD Mass Bal: in={m_dot_in_kg_h:.3f}, "
                f"out={m_gas_out_corrected:.3f}+{m_drain_total:.3f}, "
                f"err={mass_balance_error/m_dot_in_kg_h*100:.3f}%"
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
        
        # Calculate liquid carryover (entrainment)
        entrained_liq = 0.0
        if self._gas_outlet_stream:
             entrained_liq = self._gas_outlet_stream.extra.get('m_dot_H2O_liq_accomp_kg_s', 0.0)
        
        state.update({
            'rho_g': self._rho_g,
            'v_max': self._v_max,
            'v_real': self._v_real,
            'separation_status': self._separation_status,
            'power_consumption_w': self._power_consumption_w,
            'dissolved_gas_kg_h': self._dissolved_gas_kg_h,
            'total_drain_removed_kg': self.total_drain_removed_kg,
            'diameter_m': self.diameter_m,
            'delta_p_bar': self.delta_p_bar,
            'gas_species': self.gas_species,
            'separation_efficiency': self.separation_efficiency,
            
            # Rigorous Mass Balance Metrics (IN/OUT)
            'dissolved_gas_in_kg_h': self._last_dissolved_gas_in_kg_h,
            'dissolved_gas_out_kg_h': self._last_dissolved_gas_out_kg_h,
            # Aliases for graph compatibility
            'dissolved_gas_in': self._last_dissolved_gas_in_kg_h,
            'dissolved_gas_out': self._last_dissolved_gas_out_kg_h,
            
            'water_removed_kg_h': self._liquid_drain_stream.mass_flow_kg_h if self._liquid_drain_stream else 0.0,
            'drain_temp_k': self._liquid_drain_stream.temperature_k if self._liquid_drain_stream else 0.0,
            'drain_pressure_bar': self._liquid_drain_stream.pressure_pa / 1e5 if self._liquid_drain_stream else 0.0,
            'dissolved_gas_ppm': (self._dissolved_gas_kg_h / self._liquid_drain_stream.mass_flow_kg_h * 1e6) 
                                 if (self._liquid_drain_stream and self._liquid_drain_stream.mass_flow_kg_h > 0) else 0.0,
            'm_dot_H2O_liq_accomp_kg_s': entrained_liq,
            'outlet_o2_ppm_mol': (self._gas_outlet_stream.get_total_mole_frac('O2') * 1e6) if self._gas_outlet_stream else 0.0
        })
        return state

