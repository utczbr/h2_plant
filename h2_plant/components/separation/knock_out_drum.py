"""
Knock-Out Drum (KOD) Component for H2Plant Simulation.

Implements a vertical separator vessel for removing liquid water droplets
from gas streams (H2 or O2) following a cooling process. Uses isothermal
flash physics and Souders-Brown sizing criteria.

Physics Reference:
- Souders-Brown equation for maximum gas velocity.
- Rachford-Rice for vapor-liquid equilibrium.
"""

from typing import Dict, Any, Optional, TYPE_CHECKING
import math

from h2_plant.core.component import Component
from h2_plant.core.stream import Stream
from h2_plant.core.constants import GasConstants
from h2_plant.optimization.coolprop_lut import CoolPropLUT
from h2_plant.optimization.numba_ops import solve_rachford_rice_single_condensable

if TYPE_CHECKING:
    from h2_plant.core.component_registry import ComponentRegistry


# ============================================================================
# Physical Constants
# ============================================================================
RHO_L_WATER: float = 1000.0  # Liquid water density, kg/m³
K_SOUDERS_BROWN: float = 0.08  # Max permissible velocity factor, m/s
R_UNIV: float = 8.31446  # Universal Gas Constant, J/(mol·K)


class KnockOutDrum(Component):
    """
    Knock-Out Drum: A vertical separator for liquid water removal from gas.

    The KOD operates as an isothermal flash separator where liquid water
    droplets are removed via gravity. It is typically installed after a
    chiller or cooler in a gas processing train.

    Attributes:
        diameter_m: Vessel inner diameter (m).
        delta_p_bar: Pressure drop across the vessel (bar).
        gas_species: Primary gas species ('H2' or 'O2').
    """

    def __init__(
        self,
        diameter_m: float = 1.0,
        delta_p_bar: float = 0.05,
        gas_species: str = 'H2'
    ) -> None:
        """
        Initialize KnockOutDrum component.

        Args:
            diameter_m: Vessel inner diameter. Default 1.0 m.
            delta_p_bar: Pressure drop across vessel. Default 0.05 bar.
            gas_species: Primary gas species, 'H2' or 'O2'. Default 'H2'.

        Raises:
            ValueError: If diameter is non-positive or gas_species is invalid.
        """
        super().__init__()

        if diameter_m <= 0:
            raise ValueError(f"diameter_m must be positive, got {diameter_m}")
        if gas_species not in ('H2', 'O2'):
            raise ValueError(f"gas_species must be 'H2' or 'O2', got {gas_species}")

        self.diameter_m = diameter_m
        self.delta_p_bar = delta_p_bar
        self.gas_species = gas_species

        # Internal state for last computed timestep
        self._input_stream: Optional[Stream] = None
        self._gas_outlet_stream: Optional[Stream] = None
        self._liquid_drain_stream: Optional[Stream] = None

        # State variables for reporting
        self._rho_g: float = 0.0
        self._v_max: float = 0.0
        self._v_real: float = 0.0
        self._separation_status: str = "IDLE"
        self._power_consumption_w: float = 0.0

    def initialize(self, dt: float, registry: 'ComponentRegistry') -> None:
        """
        Initialize the component before simulation.

        Args:
            dt: Simulation timestep in hours.
            registry: Component registry for accessing other components.
        """
        super().initialize(dt, registry)

        # Validate configuration
        if self.diameter_m <= 0:
            raise ValueError("diameter_m must be positive")

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """
        Define input/output ports for the KOD.

        Returns:
            Port definitions with type and resource metadata.
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
        Receive an input stream at a specified port.

        Args:
            port_name: Port to receive input ('gas_inlet').
            value: Stream object containing gas flow data.
            resource_type: Resource type (optional).

        Returns:
            Mass flow rate received (kg/h), or 0.0 if rejected.
        """
        if port_name == 'gas_inlet' and isinstance(value, Stream):
            self._input_stream = value
            return value.mass_flow_kg_h
        return 0.0

    def step(self, t: float) -> None:
        """
        Execute a single simulation timestep.

        Performs flash separation calculations, sizing checks, and
        prepares output streams for downstream components.

        Args:
            t: Current simulation time in hours.
        """
        super().step(t)

        inlet = self._input_stream
        if inlet is None or inlet.mass_flow_kg_h <= 0:
            # No input or zero flow: reset outputs
            self._gas_outlet_stream = None
            self._liquid_drain_stream = None
            self._separation_status = "NO_FLOW"
            self._rho_g = 0.0
            self._v_max = 0.0
            self._v_real = 0.0
            self._power_consumption_w = 0.0
            return

        # ====================================================================
        # A. Input Processing: Convert mass flow to molar flow
        # ====================================================================
        m_dot_in_kg_h = inlet.mass_flow_kg_h
        T_in = inlet.temperature_k
        P_in = inlet.pressure_pa
        composition = inlet.composition

        # Calculate total molar flow from mass fractions
        # n_i = (mass_frac_i * m_dot) / MW_i
        n_total_mol_s = 0.0
        n_species: Dict[str, float] = {}

        for species, mass_frac in composition.items():
            if species in GasConstants.SPECIES_DATA:
                mw_g_mol = GasConstants.SPECIES_DATA[species]['molecular_weight']
                mw_kg_mol = mw_g_mol / 1000.0
                # mass_i (kg/h) = mass_frac * m_dot_kg_h
                # mol_i (mol/s) = (mass_i (kg/h) / 3600) / (MW kg/mol)
                #               = (mass_frac * m_dot / 3600) / mw_kg_mol
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

        # Mole fractions
        z_species: Dict[str, float] = {sp: n / n_total_mol_s for sp, n in n_species.items()}
        z_H2O = z_species.get('H2O', 0.0)

        # ====================================================================
        # B. Flash / Separation Physics
        # ====================================================================
        # Pressure drop
        delta_p_pa = self.delta_p_bar * 1e5
        P_out = P_in - delta_p_pa
        if P_out <= 0:
            P_out = 1e5  # Fallback to 1 bar

        # Saturation pressure of water at inlet temperature
        try:
            P_sat = CoolPropLUT.PropsSI('P', 'T', T_in, 'Q', 0.0, 'Water')
            # CoolProp might return 0.0 or infinity on failure without raising exception
            if P_sat <= 1e-6 or not math.isfinite(P_sat):
                raise ValueError("CoolProp returned invalid P_sat")
        except Exception:
            # Fallback using Antoine equation approximation (T in K)
            # log10(Psat_mmHg) = A - B/(C+T_C), then convert to Pa
            T_C = T_in - 273.15
            A, B, C = 8.07131, 1730.63, 233.426
            P_sat_mmHg = 10 ** (A - B / (C + T_C))
            P_sat = P_sat_mmHg * 133.322  # mmHg to Pa

        # Equilibrium K-value for water
        K_eq = P_sat / P_out if P_out > 0 else 1.0

        # Vapor fraction (beta) using Rachford-Rice
        if z_H2O > 1e-12:
            beta = solve_rachford_rice_single_condensable(z_H2O, K_eq)
        else:
            beta = 1.0  # All vapor, no water

        # ====================================================================
        # Phase Compositions
        # ====================================================================
        # Robust Logic:
        # If beta == 1.0 (All Vapor), composition matches feed (z).
        # If beta < 1.0 (Vapor + Liquid), vapor is at saturation limit.
        
        if beta >= 1.0 - 1e-9:
            # All vapor phase (superheated or saturated)
            # y_H2O is limited by available water (z_H2O)
            y_H2O = z_H2O
        else:
            # Two-phase region (saturated)
            # y_H2O is determined by equilibrium (P_sat / P_out)
            y_H2O = min(P_sat / P_out, 1.0) if P_out > 0 else 0.0

        # Clamp to valid range [0, 1]
        y_H2O = max(0.0, min(y_H2O, 1.0))

        # Gas phase: primary gas mole fraction
        y_gas = 1.0 - y_H2O
        y_gas = max(0.0, min(y_gas, 1.0))

        # Liquid phase is assumed pure water
        x_H2O = 1.0

        # ====================================================================
        # C. Fluid Properties & Sizing
        # ====================================================================
        # Molar flow of vapor and liquid phases
        n_vap_mol_s = beta * n_total_mol_s
        n_liq_mol_s = (1.0 - beta) * n_total_mol_s

        # Mixture molar mass for gas phase (kg/mol)
        mw_gas = GasConstants.SPECIES_DATA[self.gas_species]['molecular_weight'] / 1000.0
        mw_h2o = GasConstants.SPECIES_DATA['H2O']['molecular_weight'] / 1000.0
        M_mix = y_gas * mw_gas + y_H2O * mw_h2o  # kg/mol

        # Compressibility factor Z (approximate using pure gas)
        try:
            Z = CoolPropLUT.PropsSI('Z', 'T', T_in, 'P', P_out, self.gas_species)
        except Exception:
            Z = 1.0  # Ideal gas fallback

        # Gas density: rho_G = (P_out * M_mix) / (Z * R * T)
        if Z > 0 and T_in > 0:
            rho_g = (P_out * M_mix) / (Z * R_UNIV * T_in)
        else:
            rho_g = 0.0

        self._rho_g = rho_g

        # Volumetric flow: V_dot_G = (n_vap * M_mix) / rho_G (m³/s)
        m_vap_kg_s = n_vap_mol_s * M_mix
        if rho_g > 0:
            V_dot_g_m3_s = m_vap_kg_s / rho_g
        else:
            V_dot_g_m3_s = 0.0

        # Velocity limits (Souders-Brown criterion)
        # Velocity limits (Souders-Brown criterion)
        # V_max = K_sb * sqrt((rho_L - rho_G) / rho_G)
        if rho_g > 0:
            # GUARD: Ensure term inside sqrt is non-negative
            density_diff = max(0.0, RHO_L_WATER - rho_g)
            v_max = K_SOUDERS_BROWN * math.sqrt(density_diff / rho_g)
        else:
            v_max = float('inf')

        self._v_max = v_max

        # Actual velocity: V_real = V_dot / A
        A_vessel = math.pi * (self.diameter_m / 2.0) ** 2
        if A_vessel > 0:
            v_real = V_dot_g_m3_s / A_vessel
        else:
            v_real = 0.0

        self._v_real = v_real

        # Status check
        if v_real < v_max:
            self._separation_status = "OK"
        else:
            self._separation_status = "UNDERSIZED"

        # ====================================================================
        # D. Power & Outputs
        # ====================================================================
        # Parasitic power loss (theoretical work to restore pressure)
        # Power (W) = V_dot (m³/s) * delta_P (Pa)
        self._power_consumption_w = V_dot_g_m3_s * delta_p_pa

        # Mass flows for output streams
        m_dot_vap_kg_h = m_vap_kg_s * 3600.0
        m_dot_liq_kg_h = n_liq_mol_s * mw_h2o * 3600.0

        # Gas outlet stream
        gas_comp = {self.gas_species: y_gas, 'H2O': y_H2O}
        self._gas_outlet_stream = Stream(
            mass_flow_kg_h=m_dot_vap_kg_h,
            temperature_k=T_in,
            pressure_pa=P_out,
            composition=gas_comp,
            phase='gas'
        )

        # Liquid drain stream
        self._liquid_drain_stream = Stream(
            mass_flow_kg_h=m_dot_liq_kg_h,
            temperature_k=T_in,
            pressure_pa=P_out,
            composition={'H2O': 1.0},
            phase='liquid'
        )

        # Clear input for next timestep
        self._input_stream = None

    def get_output(self, port_name: str) -> Any:
        """
        Retrieve the output from a specified port.

        Args:
            port_name: 'gas_outlet' or 'liquid_drain'.

        Returns:
            Stream object for the requested port, or None if unavailable.

        Raises:
            ValueError: If port_name is not a valid output port.
        """
        if port_name == 'gas_outlet':
            return self._gas_outlet_stream
        elif port_name == 'liquid_drain':
            return self._liquid_drain_stream
        else:
            raise ValueError(f"Unknown output port '{port_name}'")

    def get_state(self) -> Dict[str, Any]:
        """
        Return current component state for monitoring/persistence.

        Returns:
            Dictionary with current sizing and operational status.
        """
        state = super().get_state()
        state.update({
            'rho_g': self._rho_g,
            'v_max': self._v_max,
            'v_real': self._v_real,
            'separation_status': self._separation_status,
            'power_consumption_w': self._power_consumption_w,
            'diameter_m': self.diameter_m,
            'delta_p_bar': self.delta_p_bar,
            'gas_species': self.gas_species
        })
        return state
