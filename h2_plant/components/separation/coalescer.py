"""
Coalescer component for aerosol removal in H2/O2 streams.

Fibrous cartridge filter removing 99.99% of liquid water aerosols down to 0.1 μm.
Physical model based on CoalescerModel.py with Carman-Kozeny pressure drop.
"""

from typing import Dict, Any, Optional
import math

from h2_plant.core.component import Component
from h2_plant.core.stream import Stream
from h2_plant.core.constants import CoalescerConstants, ConversionFactors


class Coalescer(Component):
    """
    Fibrous cartridge filter for aerosol removal in H2/O2 streams.
    
    Physical Fidelity (CoalescerModel.py):
        - Pressure Drop: Carman-Kozeny (ΔP = K * μ * L * U_sup)
        - Viscosity: Sutherland power law (μ = μ_ref * (T/T_ref)^0.7)
        - Efficiency: Fixed 99.99% removal of liquid phase water
        
    Attributes:
        d_shell: Vessel diameter [m]
        l_elem: Element length [m]
        gas_type: 'H2' or 'O2' for viscosity selection
    """

    def __init__(
        self,
        d_shell: float = CoalescerConstants.D_SHELL_DEFAULT_M,
        l_elem: float = CoalescerConstants.L_ELEM_DEFAULT_M,
        gas_type: str = 'H2',
        **kwargs
    ):
        """
        Initialize Coalescer.
        
        Args:
            d_shell: Vessel diameter [m] (default 0.32)
            l_elem: Element length [m] (default 1.00)
            gas_type: 'H2' or 'O2' for viscosity calibration
        """
        super().__init__(config=kwargs)
        self.d_shell = d_shell
        self.l_elem = l_elem
        self.gas_type = gas_type.upper()
        
        # Pre-calculate geometric constant
        self.area_shell = (math.pi / 4) * (self.d_shell ** 2)
        
        # State variables
        self.total_liquid_removed_kg: float = 0.0
        self.current_delta_p_bar: float = 0.0
        self.current_power_loss_w: float = 0.0
        self.output_stream: Optional[Stream] = None
        self.drain_stream: Optional[Stream] = None
        self._step_liq_removed: float = 0.0

    def initialize(self, dt: float, registry: Any) -> None:
        """Initialize component."""
        super().initialize(dt, registry)
        self.total_liquid_removed_kg = 0.0
        self._step_liq_removed = 0.0

    def receive_input(self, port_name: str, value: Any, resource_type: str = None) -> float:
        """
        Process the gas stream: remove liquid water and apply pressure drop.
        
        Args:
            port_name: Must be 'inlet'
            value: Stream object with gas mixture
            resource_type: Optional resource type hint
            
        Returns:
            Mass flow accepted (kg/h)
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
        
        # 1. Calculate Fluid Properties
        # Viscosity (Sutherland model from CoalescerModel.py:70-71)
        # NOTE: Reference model uses H2 viscosity for BOTH gases as simplification
        # "Modelo de Sutherland Simplificado para H2 (ΔP é proporcional à viscosidade)"
        mu_ref = CoalescerConstants.MU_REF_H2_PA_S  # Always use H2 reference
            
        t_ratio = in_stream.temperature_k / CoalescerConstants.T_REF_K
        mu_g = mu_ref * (t_ratio ** 0.7)

        # Density from stream (ideal gas)
        rho_mix = in_stream.density_kg_m3
        if rho_mix <= 0:
            return 0.0

        # 2. Calculate Hydrodynamics (CoalescerModel.py:86-99)
        # Volumetric Flow (m³/s)
        q_v_m3_s = (in_stream.mass_flow_kg_h / 3600.0) / rho_mix
        
        # Superficial Velocity (m/s)
        u_sup = q_v_m3_s / self.area_shell
        
        # Pressure Drop (Pa) - Linear Carman-Kozeny (CoalescerModel.py:99)
        delta_p_pa = (CoalescerConstants.K_PERDA * 
                      mu_g * 
                      self.l_elem * 
                      u_sup)
        
        self.current_delta_p_bar = delta_p_pa * ConversionFactors.PA_TO_BAR
        
        # Power Loss (W) = Q_V × ΔP
        self.current_power_loss_w = q_v_m3_s * delta_p_pa

        # 3. Calculate Separation (Liquid Removal)
        h2o_liq_frac = in_stream.composition.get('H2O_liq', 0.0)
        m_dot_liq_in = in_stream.mass_flow_kg_h * h2o_liq_frac
        
        # Apply efficiency (CoalescerModel.py:44)
        m_dot_liq_removed = m_dot_liq_in * CoalescerConstants.ETA_LIQUID_REMOVAL
        m_dot_liq_remaining = m_dot_liq_in - m_dot_liq_removed
        
        # 4. Construct Output Streams
        out_mass = in_stream.mass_flow_kg_h - m_dot_liq_removed
        out_pressure = max(in_stream.pressure_pa - delta_p_pa, 1e4)  # Min 0.1 bar
        
        # Update composition: reduce H2O_liq, re-normalize
        new_comp = in_stream.composition.copy()
        if out_mass > 0 and h2o_liq_frac > 0:
            new_comp['H2O_liq'] = m_dot_liq_remaining / out_mass
            # Re-normalize
            total_frac = sum(new_comp.values())
            if total_frac > 0:
                for k in new_comp:
                    new_comp[k] /= total_frac
        
        self.output_stream = Stream(
            mass_flow_kg_h=out_mass,
            temperature_k=in_stream.temperature_k,  # Isothermal
            pressure_pa=out_pressure,
            composition=new_comp,
            phase='gas'
        )

        # Drain Outlet (Pure Water)
        self.drain_stream = Stream(
            mass_flow_kg_h=m_dot_liq_removed,
            temperature_k=in_stream.temperature_k,
            pressure_pa=out_pressure,
            composition={'H2O': 1.0},
            phase='liquid'
        )
        
        # Store for step() accumulation
        self._step_liq_removed = m_dot_liq_removed

        return in_stream.mass_flow_kg_h

    def step(self, t: float) -> None:
        """Execute one timestep: accumulate removed liquid."""
        super().step(t)
        # Integrate removed liquid over timestep (dt in hours, flow in kg/h)
        self.total_liquid_removed_kg += self._step_liq_removed * self.dt
        self._step_liq_removed = 0.0

    def get_output(self, port_name: str) -> Any:
        """Get output from specified port."""
        if port_name == 'outlet':
            return self.output_stream if self.output_stream else Stream(0.0)
        elif port_name == 'drain':
            return self.drain_stream if self.drain_stream else Stream(0.0)
        raise ValueError(f"Port '{port_name}' not found in Coalescer")

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """Return port definitions."""
        return {
            'inlet': {'type': 'input', 'resource_type': 'gas_mixture'},
            'outlet': {'type': 'output', 'resource_type': 'gas_mixture'},
            'drain': {'type': 'output', 'resource_type': 'water'}
        }

    def get_state(self) -> Dict[str, Any]:
        """Return current component state."""
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
