"""
Hydrogen Multi-Cyclone Separator Component.

This module implements an axial multi-cyclone separator for gas/liquid separation
in hydrogen production systems. The cyclone uses centrifugal force to separate
entrained liquid droplets from the gas stream, achieving high separation efficiency
with minimal pressure drop.

Physical Principles:
    - **Barth/Muschelknautz Model**: Determines the critical particle cut-size (d₅₀)
      based on the balance between centrifugal settling force and Stokes drag.
    - **Euler Number Correlation**: Estimates pressure drop through the swirl
      generating vanes and cyclone body.
    - **Multi-Tube Design**: Parallel elements maintain high throughput while
      keeping tube Reynolds number in optimal separation regime.

Architecture:
    Implements the Component Lifecycle Contract (Layer 1):
    - `initialize()`: Validates configuration and acquires LUTManager reference.
    - `step()`: Executes separation physics via JIT-compiled Numba kernel.
    - `get_state()`: Exposes operational metrics for monitoring and control.

References:
    - Hoffmann, A.C. & Stein, L.E. (2008). Gas Cyclones and Swirl Tubes.
    - Coker, A.K. (2007). Ludwig's Applied Process Design for Chemical and
      Petrochemical Plants. Vol. 1.
    - Muschelknautz, E. (1972). VDI-Berichte Nr. 363.
"""

import numpy as np
from typing import Dict, Any, Optional, TYPE_CHECKING
import logging

from h2_plant.core.component import Component
from h2_plant.core.stream import Stream
from h2_plant.optimization.numba_ops import solve_cyclone_mechanics, calculate_mixture_density_jit

# Import mixture thermodynamics for rigorous density calculations
try:
    from h2_plant.optimization import mixture_thermodynamics as mix_thermo
    MIX_THERMO_AVAILABLE = True
except ImportError:
    mix_thermo = None
    MIX_THERMO_AVAILABLE = False

if TYPE_CHECKING:
    from h2_plant.core.component_registry import ComponentRegistry

logger = logging.getLogger(__name__)


# ============================================================================
# Physical Constants
# ============================================================================
RHO_L_WATER: float = 1000.0
"""Liquid water density at standard conditions (kg/m³)."""

MU_H2_REF: float = 8.4e-6
"""Reference dynamic viscosity of H2 at ~300K (Pa·s). Perry's Handbook."""


class HydrogenMultiCyclone(Component):
    """
    Axial Multi-Cyclone Separator for Gas/Liquid Separation.

    This component operates within Layer 1 (Component Lifecycle), managing the 
    phase separation of hydrogen gas and entrained liquid water. It utilizes a 
    rigorous Barth/Muschelknautz model to determine separation efficiency based 
    on particle cut-size (d₅₀).

    The multi-tube design uses parallel cyclone elements with fixed-vane swirl
    generators. Gas enters axially, acquires tangential velocity through the
    vanes, and liquid droplets are flung to the walls by centrifugal force
    where they coalesce and drain downward.

    Design Features:
        - **Modular Tube Count**: Automatically sizes the number of active tubes
          to maintain target axial velocity for optimal separation.
        - **Hub Obstruction**: 30% diameter hub ratio per Hoffmann & Stein.
        - **Euler Pressure Drop**: Accounts for swirl generation losses.

    Attributes:
        D_element_m (float): Diameter of individual cyclone tubes [m].
        vane_angle_rad (float): Inlet vane angle [radians].
        target_velocity (float): Design axial velocity for tube sizing [m/s].
        N_tubes (int): Calculated number of active cyclone elements.
        d50_microns (float): Computed cut-size diameter [μm].
        delta_P_pa (float): Pressure drop across separator [Pa].

    Example:
        >>> cyclone = HydrogenMultiCyclone(element_diameter_mm=50.0, vane_angle_deg=45.0)
        >>> cyclone.initialize(dt=1/60, registry=registry)
        >>> cyclone.receive_input('inlet', wet_h2_stream, 'gas')
        >>> cyclone.step(t=0.0)
        >>> dry_gas = cyclone.get_output('outlet')
    """

    def __init__(
        self,
        element_diameter_mm: float = 50.0,
        vane_angle_deg: float = 45.0,
        target_velocity_ms: float = 20.0,
        gas_species: str = 'H2'
    ) -> None:
        """
        Configuration Phase.
        
        Args:
            element_diameter_mm (float): Tube diameter in millimeters.
                Typical values: DN50 (50mm) for standard applications.
                Larger diameters improve capacity but reduce separation efficiency.
            vane_angle_deg (float): Swirl generator vane angle in degrees.
                Higher angles increase tangential velocity and separation efficiency
                but also increase pressure drop. Typical range: 30-60°.
            target_velocity_ms (float): Target axial velocity for tube sizing.
                Controls the number of active tubes calculated. Higher velocity
                means fewer tubes but higher d₅₀. Default: 20 m/s.
            gas_species (str): Primary gas species for property lookup.
                Must be 'H2' or 'O2'. Default: 'H2'.
                
        Raises:
            ValueError: If element_diameter_mm <= 0 or vane_angle not in (0, 90).
        """
        super().__init__()
        
        if element_diameter_mm <= 0:
            raise ValueError(f"element_diameter_mm must be positive, got {element_diameter_mm}")
        if not (0 < vane_angle_deg < 90):
            raise ValueError(f"vane_angle_deg must be in (0, 90), got {vane_angle_deg}")
        if gas_species not in ('H2', 'O2'):
            logger.warning(f"HydrogenMultiCyclone: Non-standard gas species '{gas_species}'. Defaulting to H2-like properties.")
            # raise ValueError(f"gas_species must be 'H2' or 'O2', got {gas_species}")
            
        # Standardize geometry to SI Units (Layer 3 Standard)
        self.D_element_m = element_diameter_mm / 1000.0
        self.vane_angle_rad = np.radians(vane_angle_deg)
        self.target_velocity = target_velocity_ms
        self.gas_species = gas_species
        
        # Geometry Cache (calculated once)
        self.D_hub = 0.3 * self.D_element_m  # 30% hub ratio
        self.Area_annulus = (np.pi / 4.0) * (self.D_element_m**2 - self.D_hub**2)
        
        # State Cache (updated each timestep)
        self.N_tubes: int = 0
        self.d50_microns: float = 0.0
        self.delta_P_pa: float = 0.0
        self.efficiency: float = 0.0
        self.v_axial: float = 0.0
        self.v_tan: float = 0.0
        
        # Stream buffers
        self._input_stream: Optional[Stream] = None
        self._outlet_stream: Optional[Stream] = None
        self._drain_stream: Optional[Stream] = None
        
        # Mass Balance Tracking (Input vs Output)
        self._last_dissolved_gas_in_kg_h: float = 0.0
        self._last_dissolved_gas_out_kg_h: float = 0.0
        
        # LUTManager reference (acquired during initialize)
        # LUTManager reference (acquired during initialize)
        self.lut_manager = None
        
        # JIT Stacked LUTs (cached from LUTManager)
        self._D_luts_stacked: Optional[np.ndarray] = None
        self._P_grid: Optional[np.ndarray] = None
        self._T_grid: Optional[np.ndarray] = None

    def initialize(self, dt: float, registry: 'ComponentRegistry') -> None:
        """
        Lifecycle Phase 1: Initialization.
        
        Acquires references to the Thermodynamic Data Service (LUTManager)
        for high-speed property lookups during simulation.
        
        Args:
            dt (float): Simulation timestep [h].
            registry (ComponentRegistry): System service locator providing
                access to shared components like LUTManager.
                
        Raises:
            RuntimeError: If LUTManager is not available in the registry.
        """
        super().initialize(dt, registry)
        
        # Attempt to acquire LUTManager for thermodynamic lookups
        try:
            self.lut_manager = registry.get('lut_manager')
            if self.lut_manager:
                # Pre-fetch stacked LUTs for JIT density calculation
                self._D_luts_stacked = self.lut_manager.stacked_D
                self._P_grid = self.lut_manager._pressure_grid
                self._T_grid = self.lut_manager._temperature_grid
        except Exception:
            self.lut_manager = None
            
        if self.lut_manager is None:
            logger.warning(
                "HydrogenMultiCyclone: LUTManager not available. "
                "Using fallback viscosity values."
            )

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """
        Define the physical connection ports for this component.
        
        Returns:
            Dict[str, Dict[str, str]]: Port definitions with keys:
                - 'inlet': Receives wet gas from upstream equipment.
                - 'outlet': Delivers dried gas to downstream processing.
                - 'drain': Removes separated liquid.
        """
        return {
            'inlet': {
                'type': 'input',
                'resource_type': 'gas',
                'units': 'kg/h'
            },
            'outlet': {
                'type': 'output',
                'resource_type': 'gas',
                'units': 'kg/h'
            },
            'drain': {
                'type': 'output',
                'resource_type': 'water',
                'units': 'kg/h'
            }
        }

    def receive_input(self, port_name: str, value: Any, resource_type: str = None) -> float:
        """
        Accept an input stream at the specified port.
        Also calculates the potential dissolved gas (IN) based on inlet liquid content.
        
        Args:
            port_name (str): Target port identifier. Expected: 'inlet'.
            value (Any): Stream object containing mass flow, temperature,
                pressure, and composition data.
            resource_type (str, optional): Resource classification hint.
                
        Returns:
            float: Mass flow rate accepted (kg/h), or 0.0 if input was rejected.
        """
        if port_name == 'inlet' and isinstance(value, Stream):
            self._input_stream = value
            
            # --- Mass Balance Tracking (IN) ---
            m_dot_total = value.mass_flow_kg_h
            liq_frac = value.composition.get('H2O_liq', 0.0)
            m_liq_thermo = m_dot_total * liq_frac
            m_liq_entrained = 0.0
            if hasattr(value, 'extra') and value.extra:
                m_liq_entrained = value.extra.get('m_dot_H2O_liq_accomp_kg_s', 0.0) * 3600.0
            total_liquid_in = m_liq_thermo + m_liq_entrained
            
            if total_liquid_in > 0 and value.pressure_pa > 0:
                # Molecular weights (kg/mol) - must convert mass fractions to mole fractions!
                MW = {'H2': 2.016e-3, 'O2': 32.00e-3, 'H2O': 18.015e-3, 'H2O_liq': 18.015e-3}
                
                # Convert mass fractions to mole fractions: n_i = x_i / M_i
                # CRITICAL: Exclude liquid species from gas phase mole fraction!
                n_rel = {}
                for species, x_mass in value.composition.items():
                    if species == 'H2O_liq': continue  # Explicitly skip liquid
                    if species.endswith('_liq'): continue
                    mw_i = MW.get(species, 28.0e-3)  # Default ~N2
                    n_rel[species] = x_mass / mw_i
                
                total_n = sum(n_rel.values())
                if total_n > 0:
                    y_gas_mol = n_rel.get(self.gas_species, 0.0) / total_n
                    p_gas_partial = value.pressure_pa * y_gas_mol
                else:
                    p_gas_partial = 0.0
                
                solubility_in = self._calculate_dissolved_gas(value.temperature_k, p_gas_partial)
                self._last_dissolved_gas_in_kg_h = total_liquid_in * (solubility_in / 1e6)
            else:
                self._last_dissolved_gas_in_kg_h = 0.0
            # -----------------------------------
            
            return value.mass_flow_kg_h
        return 0.0
    
    def _calculate_dissolved_gas(self, temp_k: float, pressure_pa: float) -> float:
        """
        Calculate gas solubility in liquid water using Henry's Law.
        
        Args:
            temp_k: Temperature in Kelvin.
            pressure_pa: Partial pressure of gas in Pascals.
            
        Returns:
            float: Solubility in mg/kg water.
        """
        import math
        from h2_plant.core.constants import HenryConstants
        
        if self.gas_species == 'H2':
            H_298 = HenryConstants.H2_H_298_L_ATM_MOL
            C = HenryConstants.H2_DELTA_H_R_K
            MW = HenryConstants.H2_MOLAR_MASS_KG_MOL
        elif self.gas_species == 'O2':
            H_298 = HenryConstants.O2_H_298_L_ATM_MOL
            C = HenryConstants.O2_DELTA_H_R_K
            MW = HenryConstants.O2_MOLAR_MASS_KG_MOL
        else:
            return 0.0
        
        T0 = 298.15
        if temp_k <= 0 or pressure_pa <= 0:
            return 0.0
        
        H_T = H_298 * math.exp(C * (1.0 / temp_k - 1.0 / T0))
        p_atm = pressure_pa / 101325.0
        c_mol_L = p_atm / H_T
        return c_mol_L * (MW * 1000.0) * 1000.0

    def step(self, t: float) -> None:
        """
        Lifecycle Phase 2: Execution.
        
        Performs the physics simulation step:
        1. **Thermodynamics**: Resolves gas properties via LUTManager (Layer 2).
        2. **Sizing**: Dynamically adjusts active tube count to match flow.
        3. **Mechanics**: Solves separation physics via Numba kernel (Layer 3).
        4. **Efficiency**: Applies cut-size based separation model.
        
        Args:
            t (float): Current simulation time [h].
        """
        super().step(t)
        
        # 1. Input Acquisition (Layer 3 Flow Interface)
        inlet = self._input_stream
        if inlet is None or inlet.mass_flow_kg_h <= 1e-6:
            self._set_idle_state()
            
            # Use inlet conditions if available, else standard ambient
            T_idle = inlet.temperature_k if inlet else 298.15
            P_idle = inlet.pressure_pa if inlet else 101325.0
            
            # Create Zero-Flow Output Stream (fixes graph zeros)
            self._outlet_stream = Stream(
                mass_flow_kg_h=0.0,
                temperature_k=T_idle,
                pressure_pa=P_idle,
                composition={self.gas_species: 1.0},
                phase='gas'
            )
            
            self._drain_stream = Stream(
                mass_flow_kg_h=0.0,
                temperature_k=T_idle,
                pressure_pa=P_idle,
                composition={'H2O': 1.0},
                phase='liquid'
            )
            
            self._input_stream = None
            return

        P_in = inlet.pressure_pa
        T_in = inlet.temperature_k
        
        # Extract vapor mass flow (gas phase only for cyclone sizing)
        vapor_fraction = inlet.composition.get('H2', 0.0) + inlet.composition.get('O2', 0.0)
        if vapor_fraction < 1e-9:
            vapor_fraction = 1.0  # Fallback assumption
            
        m_total_kg_h = inlet.mass_flow_kg_h
        m_gas_kg_s = m_total_kg_h * vapor_fraction / 3600.0
        
        # 2. Thermodynamic Property Lookup (Layer 2)
        # Gas density from LUTManager or CoolProp fallback
        rho_g = self._get_gas_density(P_in, T_in)
        
        # Viscosity is critical for Stokes Law separation calculation
        mu_g = self._get_gas_viscosity(P_in, T_in)
        
        rho_l = RHO_L_WATER  # Liquid water density (incompressible)

        # 3. Continuity & Sizing
        # Q = ṁ / ρ
        if rho_g > 0:
            Q_actual_m3s = m_gas_kg_s / rho_g
        else:
            Q_actual_m3s = 0.0
        
        # Calculate required tubes to maintain target velocity
        # N = ⌈Q / (v_target × A)⌉
        if self.Area_annulus > 0 and self.target_velocity > 0:
            N_required = Q_actual_m3s / (self.target_velocity * self.Area_annulus)
            self.N_tubes = max(1, int(np.ceil(N_required)))
        else:
            self.N_tubes = 1

        # 4. Physics Execution (Hot Path via Numba)
        self.d50_microns, self.delta_P_pa, self.v_axial, self.v_tan = solve_cyclone_mechanics(
            Q_gas_m3s=Q_actual_m3s,
            rho_g=rho_g,
            rho_l=rho_l,
            mu_g=mu_g,
            D_element_m=self.D_element_m,
            vane_angle_rad=self.vane_angle_rad,
            N_tubes=self.N_tubes
        )

        # 5. Separation Efficiency Model
        # Simplified cut-size based efficiency:
        # - Droplets > d₅₀: ~100% captured
        # - Droplets < d₅₀: ~0% captured  
        # Industrial typical mist droplets: 5-50 μm
        # If d₅₀ < 10 μm: excellent separation efficiency
        if self.d50_microns < 5.0:
            self.efficiency = 0.99
        elif self.d50_microns < 10.0:
            self.efficiency = 0.95
        elif self.d50_microns < 20.0:
            self.efficiency = 0.85
        else:
            self.efficiency = 0.70

        # 6. Calculate liquid removal
        # Extract liquid from inlet (H2O_liq composition or extra metadata)
        h2o_liq_frac = inlet.composition.get('H2O_liq', 0.0)
        m_liq_in_kg_h = h2o_liq_frac * m_total_kg_h
        
        # Check Stream.extra for entrained liquid from PEM/chiller
        if hasattr(inlet, 'extra') and inlet.extra:
            m_liq_from_extra_kg_s = inlet.extra.get('m_dot_H2O_liq_accomp_kg_s', 0.0)
            m_liq_in_kg_h += m_liq_from_extra_kg_s * 3600.0
            
        m_liq_removed_kg_h = m_liq_in_kg_h * self.efficiency
        m_liq_carryover_kg_h = m_liq_in_kg_h - m_liq_removed_kg_h

        # 7. Output Stream Formulation
        # NOTE: Removed artificial 1 bar clamp - allow vacuum operation
        P_out = max(P_in - self.delta_P_pa, 1000.0)  # Minimum 10 mbar (avoid zero/negative)
        
        # Gas outlet composition (remove captured liquid)
        m_gas_out = m_total_kg_h - m_liq_removed_kg_h
        
        if m_gas_out > 0:
            out_comp = {}
            for species, frac in inlet.composition.items():
                if species == 'H2O_liq':
                    # Only carryover portion remains
                    if m_liq_carryover_kg_h > 1e-9:
                        out_comp['H2O_liq'] = (m_liq_carryover_kg_h) / m_gas_out
                else:
                    # Mass of species is conserved
                    m_species = frac * m_total_kg_h
                    out_comp[species] = m_species / m_gas_out
                    
            # Normalize composition
            total_comp = sum(out_comp.values())
            if total_comp > 0:
                out_comp = {s: f / total_comp for s, f in out_comp.items()}
        else:
            out_comp = {self.gas_species: 1.0}
            
        self._outlet_stream = Stream(
            mass_flow_kg_h=m_gas_out,
            temperature_k=T_in,  # Approximate isothermal
            pressure_pa=P_out,
            composition=out_comp,
            phase='gas',
            extra={} # Do not pass carryover in extra, it is already in composition['H2O_liq'] and mass_flow
        )
        
        # Calculate dissolved gas in drain (OUT) using MOLE FRACTIONS
        # Must convert mass fractions to mole fractions for correct partial pressure!
        MW = {'H2': 2.016e-3, 'O2': 32.00e-3, 'H2O': 18.015e-3, 'H2O_liq': 18.015e-3}
        n_rel_out = {}
        for species, x_mass in out_comp.items():
            mw_i = MW.get(species, 28.0e-3)
            n_rel_out[species] = x_mass / mw_i
        total_n_out = sum(n_rel_out.values())
        if total_n_out > 0:
            y_gas_mol = n_rel_out.get(self.gas_species, 0.0) / total_n_out
            p_gas_partial = P_out * y_gas_mol
        else:
            p_gas_partial = 0.0
        
        solubility_out = self._calculate_dissolved_gas(T_in, p_gas_partial)
        dissolved_gas_kg_h = m_liq_removed_kg_h * (solubility_out / 1e6)
        self._last_dissolved_gas_out_kg_h = dissolved_gas_kg_h
        
        # Drain stream with dissolved gas
        m_drain_total = m_liq_removed_kg_h + dissolved_gas_kg_h
        drain_comp = {'H2O': m_liq_removed_kg_h / m_drain_total} if m_drain_total > 0 else {'H2O': 1.0}
        if dissolved_gas_kg_h > 0 and m_drain_total > 0:
            drain_comp[self.gas_species] = dissolved_gas_kg_h / m_drain_total
        
        self._drain_stream = Stream(
            mass_flow_kg_h=m_drain_total,
            temperature_k=T_in,
            pressure_pa=P_out,
            composition=drain_comp,
            phase='liquid'
        )
        
        # Clear input buffer for next timestep
        self._input_stream = None

    def get_output(self, port_name: str) -> Any:
        """
        Retrieve the output stream from a specified port.
        
        Args:
            port_name (str): Port to query. Must be 'outlet' or 'drain'.
            
        Returns:
            Stream: Output Stream object, or empty Stream if no flow.
            
        Raises:
            ValueError: If port_name is not a valid output port.
        """
        if port_name == 'outlet':
            return self._outlet_stream if self._outlet_stream else Stream(0.0)
        elif port_name == 'drain':
            return self._drain_stream if self._drain_stream else Stream(0.0)
        else:
            raise ValueError(f"Unknown output port '{port_name}'")

    def get_state(self) -> Dict[str, Any]:
        """
        Lifecycle Phase 3: State Reporting.
        
        Returns:
            Dict[str, Any]: Telemetry dictionary containing:
                - d50_microns: Cut-size diameter (μm)
                - pressure_drop_mbar: Pressure drop (mbar)
                - active_tubes: Number of cyclone elements in use
                - axial_velocity_ms: Gas axial velocity (m/s)
                - tangential_velocity_ms: Gas tangential velocity (m/s)
                - separation_efficiency: Fraction of liquid captured (0-1)
                - outlet_o2_ppm_mol: O2 impurity in outlet stream (ppm molar)
        """
        state = super().get_state()
        
        # Calculate outlet O2 impurity for crossover graph
        outlet_o2_ppm = 0.0
        if self._outlet_stream and self._outlet_stream.mass_flow_kg_h > 0:
            outlet_o2_ppm = self._outlet_stream.get_total_mole_frac('O2') * 1e6
        
        state.update({
            'd50_microns': self.d50_microns,
            'pressure_drop_mbar': self.delta_P_pa / 100.0,
            'pressure_drop_pa': self.delta_P_pa,
            'active_tubes': self.N_tubes,
            'axial_velocity_ms': self.v_axial,
            'tangential_velocity_ms': self.v_tan,
            'separation_efficiency': self.efficiency,
            'element_diameter_mm': self.D_element_m * 1000.0,
            'vane_angle_deg': np.degrees(self.vane_angle_rad),
            'gas_species': self.gas_species,
            'outlet_o2_ppm_mol': outlet_o2_ppm,
            
            # Rigorous Mass Balance Metrics (IN/OUT)
            'dissolved_gas_in_kg_h': self._last_dissolved_gas_in_kg_h,
            'dissolved_gas_out_kg_h': self._last_dissolved_gas_out_kg_h,
            # Aliases for graph compatibility
            'dissolved_gas_in': self._last_dissolved_gas_in_kg_h,
            'dissolved_gas_out': self._last_dissolved_gas_out_kg_h,
            
            'water_removed_kg_h': self._drain_stream.mass_flow_kg_h if self._drain_stream else 0.0,
            'drain_temp_k': self._drain_stream.temperature_k if self._drain_stream else 0.0,
            'drain_pressure_bar': self._drain_stream.pressure_pa / 1e5 if self._drain_stream else 0.0,
            'dissolved_gas_ppm': (self._last_dissolved_gas_out_kg_h / self._drain_stream.mass_flow_kg_h * 1e6) 
                                 if (self._drain_stream and self._drain_stream.mass_flow_kg_h > 0) else 0.0
        })
        return state

    def _set_idle_state(self) -> None:
        """Resets physics state during zero-flow conditions."""
        self.d50_microns = 0.0
        self.delta_P_pa = 0.0
        self.v_axial = 0.0
        self.v_tan = 0.0
        self.N_tubes = 0
        self.efficiency = 0.0

    def _get_gas_density(self, pressure_pa: float, temperature_k: float) -> float:
        """
        Get gas density using rigorous mixture thermodynamics.
        
        Priority:
        1. JIT-compiled mixture density (Amagat's Law)
        2. Python mixture thermodynamics
        3. Pure component LUT lookup
        4. Ideal gas fallback
        
        Args:
            pressure_pa: Pressure in Pascals.
            temperature_k: Temperature in Kelvin.
            
        Returns:
            float: Gas density in kg/m³.
        """
        # 1. JIT Optimized Path
        if (self._D_luts_stacked is not None 
            and self._input_stream is not None
            and calculate_mixture_density_jit is not None):
            try:
                # Use cached composition arrays from inlet stream
                weights, _, _, _ = self._input_stream.get_composition_arrays()
                
                # Check if we need to normalize for gas phase (if liquid present)
                w_sum = np.sum(weights)
                if w_sum > 1e-9:
                    # Create normalized view if needed (avoid modifying cached array)
                    if abs(w_sum - 1.0) > 1e-4:
                        weights_gas = weights / w_sum
                    else:
                        weights_gas = weights
                        
                    rho = calculate_mixture_density_jit(
                        pressure_pa, 
                        temperature_k, 
                        self._P_grid, 
                        self._T_grid, 
                        self._D_luts_stacked, 
                        weights_gas
                    )
                    if rho > 0:
                        return rho
            except Exception:
                pass # Fallback
                
        # 2. Python Mixture Path (Legacy/Fallback)
        if (MIX_THERMO_AVAILABLE and mix_thermo is not None 
            and self._input_stream and self.lut_manager):
            try:
                comp_mass = self._input_stream.composition
                # Only use mixture if more than one species present
                active_species = [s for s, f in comp_mass.items() 
                                   if f > 1e-6 and s not in ('H2O_liq',)]
                if len(active_species) > 1:
                    return mix_thermo.get_mixture_density(
                        comp_mass, pressure_pa, temperature_k, self.lut_manager
                    )
            except Exception:
                pass  # Fall through to pure component
        
        # Pure component LUT lookup
        if self.lut_manager is not None:
            try:
                return self.lut_manager.lookup(self.gas_species, 'D', pressure_pa, temperature_k)
            except Exception:
                pass
                
        # Ideal gas fallback: ρ = PM / (RT)
        M_kg_mol = 0.002016 if self.gas_species == 'H2' else 0.032
        R = 8.314  # J/(mol·K)
        return (pressure_pa * M_kg_mol) / (R * temperature_k)

    def _get_gas_viscosity(self, pressure_pa: float, temperature_k: float) -> float:
        """
        Get gas dynamic viscosity using LUTManager or reference fallback.
        
        Args:
            pressure_pa: Pressure in Pascals.
            temperature_k: Temperature in Kelvin.
            
        Returns:
            float: Dynamic viscosity in Pa·s.
        """
        if self.lut_manager is not None:
            try:
                return self.lut_manager.lookup(self.gas_species, 'V', pressure_pa, temperature_k)
            except (ValueError, KeyError):
                # 'V' (viscosity) may not be in LUT config
                pass
                
        # Fallback reference values (Perry's Handbook at ~300K)
        if self.gas_species == 'H2':
            return 8.4e-6  # Pa·s
        elif self.gas_species == 'O2':
            return 20.4e-6  # Pa·s
        else:
            # Default to H2-like for Syngas/Unknown
            return 8.4e-6
