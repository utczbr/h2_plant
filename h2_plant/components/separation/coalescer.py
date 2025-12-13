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

Architecture:
    Implements the Component Lifecycle Contract (Layer 1):
    - `initialize()`: Prepares component and resets liquid accumulator.
    - `step()`: Integrates removed liquid over timestep.
    - `get_state()`: Returns ΔP, power loss, and cumulative removal.

Process Flow:
    Gas enters via 'inlet' port, passes through fibrous elements where
    liquid droplets coalesce and drain. Dry gas exits via 'outlet' port,
    collected liquid exits via 'drain' port.
"""

from typing import Dict, Any, Optional
import math

from h2_plant.core.component import Component
from h2_plant.core.stream import Stream
from h2_plant.core.constants import CoalescerConstants, ConversionFactors


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
            **kwargs: Additional arguments including 'component_id'.

        Raises:
            ValueError: If d_shell or l_elem is non-positive.
        """
        super().__init__(kwargs)
        self.d_shell = d_shell
        self.l_elem = l_elem
        self.gas_type = gas_type
        if 'component_id' in kwargs:
            self.component_id = kwargs['component_id']

        # Geometry validation
        if self.d_shell <= 0:
            raise ValueError(f"d_shell must be positive, got {self.d_shell}")
        if self.l_elem <= 0:
            raise ValueError(f"l_elem must be positive, got {self.l_elem}")

        # Pre-calculate geometric constant
        self.area_shell = (math.pi / 4) * (self.d_shell ** 2)

        # State variables
        self.pressure_drop_bar = 0.0
        self.total_liquid_removed_kg: float = 0.0
        self.current_delta_p_bar: float = 0.0
        self.current_power_loss_w: float = 0.0
        self.output_stream: Optional[Stream] = None
        self.drain_stream: Optional[Stream] = None
        self._step_liq_removed: float = 0.0

    def initialize(self, dt: float, registry: Any) -> None:
        """
        Prepare the component for simulation execution.

        Fulfills the Component Lifecycle Contract initialization phase.
        Resets cumulative liquid tracking.

        Args:
            dt (float): Simulation timestep in hours.
            registry (ComponentRegistry): Central registry for component access.
        """
        super().initialize(dt, registry)
        self.total_liquid_removed_kg = 0.0
        self._step_liq_removed = 0.0

    def receive_input(self, port_name: str, value: Any, resource_type: str = None) -> float:
        """
        Process incoming gas stream: remove liquid water and apply pressure drop.

        Performs the complete coalescer calculation during input reception
        (push architecture) to ensure output streams are ready for downstream.

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
            self.output_stream = Stream(0.0)
            self.drain_stream = Stream(0.0)
            self.current_delta_p_bar = 0.0
            self.current_power_loss_w = 0.0
            self._step_liq_removed = 0.0
            return 0.0

        # Viscosity via Sutherland power law (H₂ reference for all gases)
        mu_ref = CoalescerConstants.MU_REF_H2_PA_S
        t_ratio = in_stream.temperature_k / CoalescerConstants.T_REF_K
        mu_g = mu_ref * (t_ratio ** 0.7)

        # Gas density from stream
        rho_mix = in_stream.density_kg_m3
        if rho_mix <= 0:
            self.output_stream = Stream(0.0)
            self.drain_stream = Stream(0.0)
            self.current_delta_p_bar = 0.0
            self.current_power_loss_w = 0.0
            self._step_liq_removed = 0.0
            return 0.0

        # Volumetric flow and superficial velocity
        q_v_m3_s = (in_stream.mass_flow_kg_h / 3600.0) / rho_mix
        u_sup = q_v_m3_s / self.area_shell

        # Carman-Kozeny pressure drop: ΔP = K × μ × L × U
        delta_p_pa = (CoalescerConstants.K_PERDA *
                      mu_g *
                      self.l_elem *
                      u_sup)

        self.current_delta_p_bar = delta_p_pa * ConversionFactors.PA_TO_BAR

        # Power loss from pressure drop: P = Q × ΔP
        self.current_power_loss_w = q_v_m3_s * delta_p_pa

        # Liquid separation (99.99% removal of H2O_liq)
        h2o_liq_frac = in_stream.composition.get('H2O_liq', 0.0)
        m_dot_liq_in = in_stream.mass_flow_kg_h * h2o_liq_frac

        m_dot_liq_removed = m_dot_liq_in * CoalescerConstants.ETA_LIQUID_REMOVAL
        m_dot_liq_remaining = m_dot_liq_in - m_dot_liq_removed

        # Construct output streams
        out_mass = in_stream.mass_flow_kg_h - m_dot_liq_removed
        out_pressure = max(in_stream.pressure_pa - delta_p_pa, 1e4)

        # Update composition with reduced liquid fraction
        new_comp = in_stream.composition.copy()
        if out_mass > 0 and h2o_liq_frac > 0:
            new_comp['H2O_liq'] = m_dot_liq_remaining / out_mass
            total_frac = sum(new_comp.values())
            if total_frac > 0:
                for k in new_comp:
                    new_comp[k] /= total_frac

        self.output_stream = Stream(
            mass_flow_kg_h=out_mass,
            temperature_k=in_stream.temperature_k,
            pressure_pa=out_pressure,
            composition=new_comp,
            phase='gas'
        )

        # Drain stream (pure liquid water)
        self.drain_stream = Stream(
            mass_flow_kg_h=m_dot_liq_removed,
            temperature_k=in_stream.temperature_k,
            pressure_pa=out_pressure,
            composition={'H2O': 1.0},
            phase='liquid'
        )

        self._step_liq_removed = m_dot_liq_removed

        return in_stream.mass_flow_kg_h

    def step(self, t: float) -> None:
        """
        Execute one simulation timestep.

        Integrates liquid removal over the timestep duration.
        (Note: Main calculations occur in receive_input for push architecture.)

        Args:
            t (float): Current simulation time in hours.
        """
        super().step(t)
        # Integrate removed liquid (flow rate × dt)
        self.total_liquid_removed_kg += self._step_liq_removed * self.dt
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

        Fulfills the Component Lifecycle Contract state access.

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
            "delta_p_bar": self.current_delta_p_bar,
            "power_loss_w": self.current_power_loss_w,
            "outlet_flow_kg_h": self.output_stream.mass_flow_kg_h if self.output_stream else 0.0,
            "drain_flow_kg_h": self.drain_stream.mass_flow_kg_h if self.drain_stream else 0.0
        })
        return state
