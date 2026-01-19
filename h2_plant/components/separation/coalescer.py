"""
Coalescer Component for Aerosol Removal.

This module implements a fibrous cartridge filter for removing liquid water
aerosols from H₂ and O₂ gas streams. Coalescers are essential for protecting
downstream equipment (compressors, PSA beds) from liquid carryover.

Physical Model:
    - **Separation Efficiency**: Fixed 99.99% removal of liquid-phase water
      (droplets ≥0.1 μm). Gaseous water vapor passes through unchanged.
    - **Pressure Drop**: Carman-Kozeny equation for fibrous media:
      ΔP = K × μ × L × U_sup
      where K is the permeability constant, μ is gas viscosity, L is element
      length, and U_sup is superficial velocity.
    - **Viscosity Model**: Sutherland power law:
      μ = μ_ref × (T/T_ref)^0.7
    - **Dissolved Gas Removal**: Henry's Law is applied to calculate the amount
      of gas dissolved in the removed liquid water, ensuring stricter mass balance.

Architecture:
    Implements the Component Lifecycle Contract (Layer 1):
    - `initialize()`: Sets up state tracking for liquid removal.
    - `step()`: Updates the time-integrated liquid removal total.
    - `get_state()`: Exposes operational metrics (ΔP, flow rates) to the system.

Process Flow:
    Gas enters via 'inlet' port, passes through fibrous elements where
    liquid droplets coalesce and drain. Dry gas exits via 'outlet' port,
    collected liquid exits via 'drain' port.
"""

from typing import Dict, Any, Optional
import math

from h2_plant.core.component import Component
from h2_plant.core.stream import Stream
from h2_plant.core.constants import CoalescerConstants, ConversionFactors, HenryConstants

# Import mixture thermodynamics for rigorous density calculations
try:
    from h2_plant.optimization import mixture_thermodynamics as mix_thermo
    MIX_THERMO_AVAILABLE = True
except ImportError:
    mix_thermo = None
    MIX_THERMO_AVAILABLE = False


class Coalescer(Component):
    """
    Fibrous cartridge filter for aerosol removal in H₂/O₂ streams.

    Removes liquid water droplets from gas streams using a fibrous coalescing
    element. Models pressure drop via Carman-Kozeny equation and tracks
    cumulative liquid removal.

    This component fulfills the Component Lifecycle Contract (Layer 1):
        - `initialize()`: Resets cumulative liquid counter.
        - `step()`: Integrates removed liquid over simulation timestep.
        - `get_state()`: Returns pressure drop, power loss, and liquid removal.

    The separation process:
    1. Gas stream enters fibrous element at superficial velocity U_sup.
    2. Liquid droplets impact fibers, coalesce, and drain by gravity.
    3. Pressure drop calculated from Carman-Kozeny: ΔP = K × μ × L × U.
    4. Dry gas exits with liquid content reduced by 99.99%.
    5. Dissolved gas loss in the condensate is calculated via Henry's Law.

    Attributes:
        d_shell (float): Vessel shell diameter (m).
        l_elem (float): Coalescing element length (m).
        gas_type (str): Primary gas species ('H2' or 'O2').
        total_liquid_removed_kg (float): Cumulative liquid collected (kg).

    Example:
        >>> coal = Coalescer(d_shell=0.3, l_elem=0.5, gas_type='H2')
        >>> coal.initialize(dt=1/60, registry=registry)
        >>> coal.receive_input('inlet', wet_h2_stream, 'gas_mixture')
        >>> coal.step(t=0.0)
        >>> dry_gas = coal.get_output('outlet')
    """

    def __init__(
        self,
        d_shell: float = CoalescerConstants.D_SHELL_DEFAULT_M,
        l_elem: float = CoalescerConstants.L_ELEM_DEFAULT_M,
        gas_type: str = 'H2',
        volume_m3: Optional[float] = None,
        **kwargs
    ):
        """
        Initialize the coalescer.

        Args:
            d_shell (float): Vessel shell diameter in m. Determines cross-
                sectional area for velocity calculation. Default: from constants.
            l_elem (float): Coalescing element length in m. Longer elements
                provide higher efficiency but greater ΔP. Default: from constants.
            gas_type (str): Primary gas type ('H2' or 'O2'). Currently uses
                H₂ viscosity for both (conservative). Default: 'H2'.
            volume_m3 (float, optional): Explicit vessel volume in m³ for CAPEX.
                If None, calculated from d_shell and l_elem.
            **kwargs: Additional arguments including 'component_id'.

        Raises:
            ValueError: If d_shell or l_elem is non-positive.
        """
        super().__init__(kwargs)
        self.d_shell = d_shell
        self.l_elem = l_elem
        self.gas_type = gas_type
        self._volume_m3 = volume_m3 # Store explicit volume

        if 'component_id' in kwargs:
            self.component_id = kwargs['component_id']
        
        # Permeability override (allow tuning for different pressure regimes)
        # Default to constant if not provided
        self.k_permeability = kwargs.get('k_permeability', CoalescerConstants.K_PERDA)

        # Geometry validation
        if self.d_shell <= 0:
            raise ValueError(f"d_shell must be positive, got {self.d_shell}")
        if self.l_elem <= 0:
            raise ValueError(f"l_elem must be positive, got {self.l_elem}")

        # Pre-calculate geometric constant
        self.area_shell = (math.pi / 4) * (self.d_shell ** 2)

        # Molar mass of primary gas (kg/mol) for rho_mix calculation
        if gas_type == 'H2':
            self.M_gas = CoalescerConstants.M_H2
        elif gas_type == 'O2':
            self.M_gas = CoalescerConstants.M_O2
        else:
            self.M_gas = CoalescerConstants.M_H2  # Default to H2

        # State variables
        self.pressure_drop_bar = 0.0
        self.total_liquid_removed_kg: float = 0.0
        self.total_drain_removed_kg: float = 0.0  # Cumulative drain (liquid + dissolved gas)
        self.current_delta_p_bar: float = 0.0
        self.current_power_loss_w: float = 0.0
        self.output_stream: Optional[Stream] = None
        self.drain_stream: Optional[Stream] = None
        self._step_liq_removed: float = 0.0
        self._step_gas_dissolved: float = 0.0
        
        # Mass Balance Tracking (Input vs Output)
        self._last_dissolved_gas_in_kg_h: float = 0.0
        self._last_dissolved_gas_out_kg_h: float = 0.0

        # EXPOSED FOR HISTORY RECORDING (engine_dispatch.py)
        self.drain_flow_kg_h: float = 0.0
        self.delta_p_bar: float = 0.0

    @property
    def volume_m3(self) -> float:
        """
        Vessel internal volume in m³ for CAPEX sizing.
        
        Returns:
            float: Explicit volume if set, otherwise calculated from shell dimensions.
        """
        if self._volume_m3 is not None:
            return self._volume_m3
        
        # Calculate cylindrical volume from dimensions
        # V = Area_shell * Length
        return self.area_shell * self.l_elem

    @volume_m3.setter
    def volume_m3(self, value: float):
        self._volume_m3 = value


        # Molar mass of primary gas (kg/mol) for rho_mix calculation
        if gas_type == 'H2':
            self.M_gas = CoalescerConstants.M_H2
        elif gas_type == 'O2':
            self.M_gas = CoalescerConstants.M_O2
        else:
            self.M_gas = CoalescerConstants.M_H2  # Default to H2

        # State variables
        self.pressure_drop_bar = 0.0
        self.total_liquid_removed_kg: float = 0.0
        self.total_drain_removed_kg: float = 0.0  # Cumulative drain (liquid + dissolved gas)
        self.current_delta_p_bar: float = 0.0
        self.current_power_loss_w: float = 0.0
        self.output_stream: Optional[Stream] = None
        self.drain_stream: Optional[Stream] = None
        self._step_liq_removed: float = 0.0
        self._step_gas_dissolved: float = 0.0
        
        # Mass Balance Tracking (Input vs Output)
        self._last_dissolved_gas_in_kg_h: float = 0.0
        self._last_dissolved_gas_out_kg_h: float = 0.0

        # EXPOSED FOR HISTORY RECORDING (engine_dispatch.py)
        self.drain_flow_kg_h: float = 0.0
        self.delta_p_bar: float = 0.0

    def initialize(self, dt: float, registry: Any) -> None:
        """
        Prepare the component for simulation execution.

        Fulfills the Component Lifecycle Contract initialization phase.
        Resets cumulative liquid tracking variables to valid starting states.

        Args:
            dt (float): Simulation timestep in hours.
            registry (ComponentRegistry): Central registry for component access.
        """
        super().initialize(dt, registry)
        self.total_liquid_removed_kg = 0.0
        self.total_drain_removed_kg = 0.0
        self._step_liq_removed = 0.0
        self._step_gas_dissolved = 0.0

    def _calculate_dissolved_gas(self, temp_k: float, pressure_pa: float, gas_type: str) -> float:
        """
        Calculate gas solubility in liquid water using Henry's Law.

        Applies the temperature-dependent Henry's Law constant to determine
        equilibrium concentration of gas in the liquid phase.

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
        if temp_k <= 0: return 0.0
        
        # Calculate temperature-corrected Henry constant
        # Justification: Solubility varies significantly with temperature
        H_T = H_298 * math.exp(C * (1.0/temp_k - 1.0/T0))
        
        # Calculate partial pressure in atm
        # Assumption: Gas phase is dominated by the primary species (y ~ 1.0)
        p_atm = pressure_pa / 101325.0
        
        # Calculate molar concentration (mol/L)
        c_mol_L = p_atm / H_T
        
        # Convert to mass concentration (mg/kg)
        # Assumes water density ~1 kg/L
        mw_g_mol = MW * 1000.0
        c_mg_L = c_mol_L * mw_g_mol * 1000.0
        
        return c_mg_L # mg/kg water assuming rho=1

    def receive_input(self, port_name: str, value: Any, resource_type: str = None) -> float:
        """
        Process incoming gas stream: remove liquid water and apply pressure drop.

        This method implements the core physics of the component within the
        push-based data flow architecture. It calculates separation efficiency
        and ensures mass balance closure.

        Args:
            port_name (str): Target port (must be 'inlet').
            value (Any): Stream object containing gas mixture with liquid aerosol.
            resource_type (str, optional): Resource classification hint.

        Returns:
            float: Mass flow accepted (kg/h), or 0.0 if rejected.
        """
        if port_name != 'inlet' or not isinstance(value, Stream):
            return 0.0

        in_stream: Stream = value

        if in_stream.mass_flow_kg_h <= 0:
            self._reset_outputs()
            return 0.0

        # --- Mass Balance Tracking (IN) ---
        m_dot_total = in_stream.mass_flow_kg_h
        liq_frac = in_stream.composition.get('H2O_liq', 0.0)
        m_liq_thermo = m_dot_total * liq_frac
        m_liq_entrained = 0.0
        if hasattr(in_stream, 'extra') and in_stream.extra:
            m_liq_entrained = in_stream.extra.get('m_dot_H2O_liq_accomp_kg_s', 0.0) * 3600.0
        total_liquid_in = m_liq_thermo + m_liq_entrained
        
        if total_liquid_in > 0 and in_stream.pressure_pa > 0:
            # Molecular weights (kg/mol) - must convert mass fractions to mole fractions!
            MW = {'H2': 2.016e-3, 'O2': 32.00e-3, 'H2O': 18.015e-3, 'H2O_liq': 18.015e-3}
            
            # Convert mass fractions to mole fractions: n_i = x_i / M_i
            # CRITICAL: Exclude liquid species from gas phase mole fraction!
            n_rel = {}
            for species, x_mass in in_stream.composition.items():
                if species.endswith('_liq'): continue
                mw_i = MW.get(species, 28.0e-3)  # Default ~N2
                n_rel[species] = x_mass / mw_i
            
            total_n = sum(n_rel.values())
            if total_n > 0:
                y_gas_mol = n_rel.get(self.gas_type, 0.0) / total_n
                p_gas_partial = in_stream.pressure_pa * y_gas_mol
            else:
                p_gas_partial = 0.0
            
            solubility_in = self._calculate_dissolved_gas(in_stream.temperature_k, p_gas_partial, self.gas_type)
            self._last_dissolved_gas_in_kg_h = total_liquid_in * (solubility_in / 1e6)
        else:
            self._last_dissolved_gas_in_kg_h = 0.0
        # -----------------------------------

        # Calculate Viscosity (Sutherland power law)
        # Justification: Viscosity determines pressure drop behavior in porous media
        mu_ref = CoalescerConstants.MU_REF_H2_PA_S
        t_ratio = in_stream.temperature_k / CoalescerConstants.T_REF_K
        mu_g = mu_ref * (t_ratio ** 0.7)

        # Calculate Mixture Density using modelo_coalescedor.py formula
        # rho_mix = (P * M_avg) / (Z * R * T)
        y_H2O_vapor = in_stream.composition.get('H2O', 0.0)  # Mass fraction vapor water
        h2o_liq_frac = in_stream.composition.get('H2O_liq', 0.0)
        y_gas_resto = 1.0 - y_H2O_vapor - h2o_liq_frac  # Mass fraction of primary gas
        
        # Average molar mass of gas phase (kg/mol)
        # FIX: Use harmonic mean of molar masses weighted by mass fractions
        # Formula: M_avg = 1 / Σ(y_i / M_i) - thermodynamically correct for density
        # The previous formula (Σ y_i * M_i) significantly overestimates M_avg for mixtures
        sum_y_over_M = 0.0
        if y_gas_resto > 0 and self.M_gas > 0:
            sum_y_over_M += y_gas_resto / self.M_gas
        if y_H2O_vapor > 0:
            sum_y_over_M += y_H2O_vapor / CoalescerConstants.M_H2O
        
        if sum_y_over_M > 0:
            M_avg = 1.0 / sum_y_over_M
        else:
            M_avg = self.M_gas  # Fallback for pure primary gas
        
        # Get compressibility factor Z from LUT or fallback to Z=1 (ideal gas)
        R_UNIV = 8.31446  # J/(mol·K)
        Z = 1.0  # Default ideal gas assumption
        lut_manager = None
        if hasattr(self, 'registry') and self.registry is not None:
            try:
                lut_manager = self.registry.get('lut_manager')
                if lut_manager is not None:
                    Z = lut_manager.lookup(self.gas_type, 'Z', in_stream.pressure_pa, in_stream.temperature_k)
            except Exception:
                pass  # Use Z=1 fallback
        
        # Compute mixture density: PREFER rigorous mixture thermodynamics (Amagat's Law)
        # PERFORMANCE: Skip mix_thermo for near-pure streams (>98% single species)
        rho_mix = 0.0
        dominant_frac = max(in_stream.composition.values()) if in_stream.composition else 0
        use_mix_thermo = (dominant_frac < 0.98 and MIX_THERMO_AVAILABLE and mix_thermo is not None and lut_manager is not None)
        
        if use_mix_thermo:
            try:
                rho_mix = mix_thermo.get_mixture_density(
                    in_stream.composition, 
                    in_stream.pressure_pa, 
                    in_stream.temperature_k, 
                    lut_manager
                )
            except Exception:
                pass  # Fall through to Z-factor method
        
        # Fallback: Z-factor method: ρ = PM / (ZRT)
        if rho_mix <= 0:
            if in_stream.temperature_k > 0 and M_avg > 0:
                rho_mix = (in_stream.pressure_pa * M_avg) / (Z * R_UNIV * in_stream.temperature_k)
            else:
                rho_mix = in_stream.density_kg_m3  # Fallback to stream property
        
        if rho_mix <= 0:
            self._reset_outputs()
            return 0.0

        # Calculate Superficial Velocity
        q_v_m3_s = (in_stream.mass_flow_kg_h / 3600.0) / rho_mix
        u_sup = q_v_m3_s / self.area_shell

        # Calculate Pressure Drop (Carman-Kozeny)
        # Principle: ΔP = K * μ * L * U
        delta_p_pa = (self.k_permeability *
                      mu_g *
                      self.l_elem *
                      u_sup)

        self.current_delta_p_bar = delta_p_pa * ConversionFactors.PA_TO_BAR
        self.current_power_loss_w = q_v_m3_s * delta_p_pa

        # Perform Liquid Separation (note: h2o_liq_frac already extracted above for rho_mix)
        m_dot_liq_in = in_stream.mass_flow_kg_h * h2o_liq_frac

        m_dot_liq_removed = m_dot_liq_in * CoalescerConstants.ETA_LIQUID_REMOVAL
        m_dot_liq_remaining = m_dot_liq_in - m_dot_liq_removed
        
        # Calculate Dissolved Gas Loss (Henry's Law with partial pressure)
        # Must convert mass fractions to mole fractions for correct partial pressure!
        MW = {'H2': 2.016e-3, 'O2': 32.00e-3, 'H2O': 18.015e-3, 'H2O_liq': 18.015e-3}
        n_rel_out = {}
        for species, x_mass in in_stream.composition.items():
            mw_i = MW.get(species, 28.0e-3)
            n_rel_out[species] = x_mass / mw_i
        total_n_out = sum(n_rel_out.values())
        if total_n_out > 0:
            y_gas_mol = n_rel_out.get(self.gas_type, 0.0) / total_n_out
            p_gas_pa = in_stream.pressure_pa * y_gas_mol
        else:
            p_gas_pa = 0.0
        
        solubility_mg_kg = self._calculate_dissolved_gas(
            in_stream.temperature_k, 
            p_gas_pa,  # Use partial pressure, not total
            self.gas_type
        )
        
        # Calculate mass of gas dissolved in removed liquid
        m_dot_gas_dissolved = m_dot_liq_removed * solubility_mg_kg * 1e-6

        # Construct Output Streams
        out_mass = in_stream.mass_flow_kg_h - m_dot_liq_removed - m_dot_gas_dissolved
        out_pressure = max(in_stream.pressure_pa - delta_p_pa, 1e4)

        # Update Composition with cleanup and validation
        new_comp = in_stream.composition.copy()
        if out_mass > 0:
            liq_frac_new = m_dot_liq_remaining / out_mass
            # Remove H2O_liq key if negligible to avoid floating-point dust
            if liq_frac_new < 1e-9:
                new_comp.pop('H2O_liq', None)
            else:
                new_comp['H2O_liq'] = liq_frac_new
            
            # Renormalize composition
            total_frac = sum(new_comp.values())
            if total_frac > 0:
                new_comp = {k: v / total_frac for k, v in new_comp.items()}
            
            # Validate normalization
            assert abs(sum(new_comp.values()) - 1.0) < 1e-6, "Composition not normalized"

        self.output_stream = Stream(
            mass_flow_kg_h=out_mass,
            temperature_k=in_stream.temperature_k,
            pressure_pa=out_pressure,
            composition=new_comp,
            phase='gas'
        )

        # FIX: Drain stream includes both removed liquid AND dissolved gas
        # This ensures mass balance closure: m_in = m_out + m_drain
        drain_total_mass = m_dot_liq_removed + m_dot_gas_dissolved
        
        # Calculate drain composition (water + dissolved gas)
        if drain_total_mass > 0:
            h2o_frac_drain = m_dot_liq_removed / drain_total_mass
            gas_frac_drain = m_dot_gas_dissolved / drain_total_mass
            drain_comp = {'H2O': h2o_frac_drain}
            if gas_frac_drain > 0:
                drain_comp[self.gas_type] = gas_frac_drain
        else:
            drain_comp = {'H2O': 1.0}
        
        self.drain_stream = Stream(
            mass_flow_kg_h=drain_total_mass,
            temperature_k=in_stream.temperature_k,
            pressure_pa=out_pressure,
            composition=drain_comp,
            phase='liquid'
        )

        self._step_liq_removed = m_dot_liq_removed
        self._step_gas_dissolved = m_dot_gas_dissolved
        self._last_dissolved_gas_out_kg_h = m_dot_gas_dissolved  # Track OUT for mass balance

        # Update exposed attributes for recorder
        self.drain_flow_kg_h = self.drain_stream.mass_flow_kg_h
        self.delta_p_bar = self.current_delta_p_bar

        return in_stream.mass_flow_kg_h
    
    def _reset_outputs(self) -> None:
        """Reset output streams to zero state."""
        self.output_stream = Stream(0.0)
        self.drain_stream = Stream(0.0)
        self.current_delta_p_bar = 0.0
        self.current_power_loss_w = 0.0
        self._step_liq_removed = 0.0
        self._step_gas_dissolved = 0.0
        # Reset exposed attributes
        self.drain_flow_kg_h = 0.0
        self.delta_p_bar = 0.0

    def step(self, t: float) -> None:
        """
        Execute one simulation timestep.

        Implements the time-integration aspect of the component's physics.
        Accumulates material removed during the timestep for reporting.

        Args:
            t (float): Current simulation time in hours.
        """
        super().step(t)
        # Integrate removed liquid flow over timestep to get total mass
        self.total_liquid_removed_kg += self._step_liq_removed * self.dt
        # Integrate total drain (liquid + dissolved gas) for mass balance tracking
        self.total_drain_removed_kg += (self._step_liq_removed + self._step_gas_dissolved) * self.dt
        self._step_liq_removed = 0.0

    def get_output(self, port_name: str) -> Any:
        """
        Retrieve output stream from specified port.

        Args:
            port_name (str): Port identifier ('outlet' or 'drain').

        Returns:
            Stream: Requested output stream.

        Raises:
            ValueError: If port_name is not a valid output port.
        """
        if port_name == 'outlet':
            return self.output_stream if self.output_stream else Stream(0.0)
        elif port_name == 'drain':
            return self.drain_stream if self.drain_stream else Stream(0.0)
        raise ValueError(f"Port '{port_name}' not found in Coalescer")

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """
        Define the physical connection ports for this component.

        Returns:
            Dict[str, Dict[str, str]]: Port definitions including inlet,
                gas outlet, and liquid drain.
        """
        return {
            'inlet': {'type': 'input', 'resource_type': 'gas_mixture'},
            'outlet': {'type': 'output', 'resource_type': 'gas_mixture'},
            'drain': {'type': 'output', 'resource_type': 'water'}
        }

    def get_state(self) -> Dict[str, Any]:
        """
        Retrieve the component's current operational state.

        Fulfills the Component Lifecycle Contract state access requirement.
        Used for system monitoring and dashboard display.

        Returns:
            Dict[str, Any]: State dictionary containing:
                - d_shell_m (float): Shell diameter (m).
                - l_elem_m (float): Element length (m).
                - total_liquid_removed_kg (float): Cumulative liquid (kg).
                - delta_p_bar (float): Current pressure drop (bar).
                - power_loss_w (float): Pressure drop power loss (W).
                - outlet_flow_kg_h (float): Dry gas output rate (kg/h).
                - drain_flow_kg_h (float): Liquid drain rate (kg/h).
        """
        state = super().get_state()
        state.update({
            "d_shell_m": self.d_shell,
            "l_elem_m": self.l_elem,
            "gas_type": self.gas_type,
            "total_liquid_removed_kg": self.total_liquid_removed_kg,
            "total_drain_removed_kg": self.total_drain_removed_kg,
            "step_gas_dissolved_kg_h": self._step_gas_dissolved,
            "delta_p_bar": self.current_delta_p_bar,
            "power_loss_w": self.current_power_loss_w,
            "outlet_flow_kg_h": self.output_stream.mass_flow_kg_h if self.output_stream else 0.0,
            "drain_flow_kg_h": self.drain_stream.mass_flow_kg_h if self.drain_stream else 0.0,
            
            # Rigorous Mass Balance Metrics (IN/OUT)
            "dissolved_gas_in_kg_h": self._last_dissolved_gas_in_kg_h,
            "dissolved_gas_out_kg_h": self._last_dissolved_gas_out_kg_h,
            # Aliases for graph compatibility
            "dissolved_gas_in": self._last_dissolved_gas_in_kg_h,
            "dissolved_gas_out": self._last_dissolved_gas_out_kg_h,
            
            "water_removed_kg_h": self.drain_stream.mass_flow_kg_h if self.drain_stream else 0.0,
            "drain_temp_k": self.drain_stream.temperature_k if self.drain_stream else 0.0,
            "drain_pressure_bar": self.drain_stream.pressure_pa / 1e5 if self.drain_stream else 0.0,
            "dissolved_gas_ppm": (self._step_gas_dissolved / self.drain_stream.mass_flow_kg_h * 1e6) 
                                 if (self.drain_stream and self.drain_stream.mass_flow_kg_h > 0) else 0.0,
            "outlet_o2_ppm_mol": (self.output_stream.get_total_mole_frac('O2') * 1e6) if self.output_stream else 0.0
        })
        return state
