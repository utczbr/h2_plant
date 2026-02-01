"""
Oxygen Mixer Component.

This module implements an oxygen mixing vessel that accumulates O₂ from
multiple sources (PEM cathode, SOEC cathode, external supply) and provides
a unified output stream for downstream consumers (ATR oxidant, storage).

Thermodynamic Model:
    - **Temperature Damping**: Uses first-order lag to smooth temperature
      transients when cold and hot streams mix. Time constant τ = 5 seconds
      provides numerical stability.
    - **Real-Gas Pressure**: Calculates pressure from density and temperature
      using CoolProp equation of state for oxygen, falling back to ideal gas
      if CoolProp is unavailable.

Architecture:
    Implements the Component Lifecycle Contract (Layer 1):
    - `initialize()`: Standard initialization.
    - `step()`: Aggregates buffered inputs, applies mixing, updates state.
    - `get_state()`: Returns mass, pressure, temperature, and fill metrics.

Capacity Calculation:
    Capacity is estimated from volume at target pressure using oxygen density
    at STP scaled by pressure ratio: m = V × ρ_STP × (P_target / P_STP).
"""

from typing import Dict, Any, List, Optional
import numpy as np
import logging

from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.constants import GasConstants
from h2_plant.core.stream import Stream

logger = logging.getLogger(__name__)


class OxygenMixer(Component):
    """
    Multi-source oxygen mixing vessel with real-gas thermodynamics.

    Accumulates oxygen from electrolysis byproduct streams and external
    sources, providing a temperature-stabilized output for downstream use.

    This component fulfills the Component Lifecycle Contract (Layer 1):
        - `initialize()`: Standard component initialization.
        - `step()`: Processes buffered inputs, mixes with stored mass, updates P/T.
        - `get_state()`: Returns mass, pressure, temperature, and capacity metrics.

    The mixing algorithm:
    1. Aggregate all buffered input streams using Stream.mix_with().
    2. Mix combined input with stored mass (virtual stream representation).
    3. Apply first-order temperature damping for numerical stability.
    4. Calculate pressure from CoolProp P(T, ρ) or ideal gas fallback.

    Attributes:
        volume_m3 (float): Vessel internal volume (m³).
        capacity_kg (float): Estimated mass capacity at target pressure (kg).
        mass_kg (float): Current oxygen inventory (kg).
        pressure_pa (float): Current pressure (Pa).
        temperature_k (float): Current temperature (K).

    Example:
        >>> mixer = OxygenMixer(volume_m3=5.0, target_pressure_bar=10.0)
        >>> mixer.initialize(dt=1/60, registry=registry)
        >>> mixer.receive_input('oxygen_in', o2_stream, 'oxygen')
        >>> mixer.step(t=0.0)
    """

    def __init__(
        self,
        volume_m3: float = 1.0,
        target_pressure_bar: float = 5.0,
        target_temperature_c: float = 25.0,
    ):
        """
        Initialize the oxygen mixer.

        Args:
            volume_m3 (float): Internal vessel volume in m³. Default: 1.0.
            target_pressure_bar (float): Operating pressure target in bar.
                Used to estimate mass capacity. Default: 5.0.
            target_temperature_c (float): Operating temperature target in °C.
                Default: 25.0.
        """
        super().__init__()

        self.volume_m3 = volume_m3

        # Capacity estimation: m = V × ρ_STP × (P/P_STP)
        rho_o2_stp = 1.429  # kg/m³ at STP
        self.capacity_kg = self.volume_m3 * rho_o2_stp * (target_pressure_bar / 1.01325)
        self.target_pressure_pa = target_pressure_bar * 1e5
        self.target_pressure_bar = target_pressure_bar
        self.target_temperature_k = target_temperature_c + 273.15

        # State variables
        self.mass_kg = 0.0
        self.pressure_pa = 1e5
        self.temperature_k = 298.15

        # Tracking
        self.output_mass_kg = 0.0
        self.cumulative_input_kg = 0.0
        self.cumulative_output_kg = 0.0
        self.cumulative_vented_kg = 0.0

        # Input buffer for push architecture
        self._input_buffer: List[Stream] = []

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """
        Prepare the component for simulation execution.

        Fulfills the Component Lifecycle Contract initialization phase.

        Args:
            dt (float): Simulation timestep in hours.
            registry (ComponentRegistry): Central registry for component access.
        """
        super().initialize(dt, registry)

    def receive_input(self, port_name: str, value: Any, resource_type: str = None) -> float:
        """
        Accept oxygen stream from upstream component.

        Buffers streams for processing during step(). Multiple inputs can
        be received per timestep.

        Args:
            port_name (str): Target port ('oxygen_in' or 'inlet').
            value (Any): Stream object containing oxygen.
            resource_type (str, optional): Resource classification hint.

        Returns:
            float: Mass flow accepted (kg/h), or 0.0 if rejected.
        """
        if port_name == 'oxygen_in' or port_name == 'inlet':
            if isinstance(value, Stream):
                if value.mass_flow_kg_h > 0:
                    self._input_buffer.append(value)
                return value.mass_flow_kg_h
        return 0.0

    def step(self, t: float) -> None:
        """
        Execute one simulation timestep.

        Processes buffered input streams:
        1. Aggregate all inputs using Stream.mix_with() for correct T/P blending.
        2. Mix combined input with stored mass (represented as virtual stream).
        3. Apply first-order temperature damping (τ = 5s) for stability.
        4. Calculate pressure using CoolProp P(T, ρ) or ideal gas fallback.

        Args:
            t (float): Current simulation time in hours.
        """
        super().step(t)

        # Aggregate buffered inputs
        combined_input_stream: Optional[Stream] = None

        for input_stream in self._input_buffer:
            if combined_input_stream is None:
                combined_input_stream = input_stream
            else:
                combined_input_stream = combined_input_stream.mix_with(input_stream)

        self._input_buffer = []

        mixed_stream = None

        if combined_input_stream is not None:
            input_mass = combined_input_stream.mass_flow_kg_h * self.dt

            if self.mass_kg > 0:
                # Represent stored mass as virtual stream for mixing
                stored_stream = Stream(
                    mass_flow_kg_h=self.mass_kg / self.dt,
                    temperature_k=self.temperature_k,
                    pressure_pa=self.pressure_pa,
                    composition={'O2': 1.0}
                )

                mixed_stream = stored_stream.mix_with(combined_input_stream)
            else:
                mixed_stream = combined_input_stream

            # Update mass inventory
            self.mass_kg += input_mass
            self.cumulative_input_kg += input_mass

            # Temperature damping for numerical stability
            # First-order lag: T_new = α×T_target + (1-α)×T_old
            # α = dt/τ, with τ = 5 seconds
            tau = 5.0
            dt_sec = self.dt * 3600.0
            alpha = min(1.0, dt_sec / tau)

            if mixed_stream is not None:
                target_T = mixed_stream.temperature_k
            else:
                target_T = combined_input_stream.temperature_k

            if self.mass_kg > input_mass:
                # Significant prior mass: apply damping
                self.temperature_k = alpha * target_T + (1 - alpha) * self.temperature_k
            else:
                # Empty tank: take input temperature directly
                self.temperature_k = target_T

            # Pressure calculation using real-gas EOS
            if self.volume_m3 > 0:
                density_kg_m3 = self.mass_kg / self.volume_m3
                try:
                    import CoolProp.CoolProp as CP
                    self.pressure_pa = CP.PropsSI('P', 'T', self.temperature_k, 'D', density_kg_m3, 'Oxygen')
                except Exception:
                    # Ideal gas fallback: P = nRT/V
                    moles_o2 = self.mass_kg * 1000.0 / GasConstants.SPECIES_DATA['O2']['molecular_weight']
                    self.pressure_pa = (moles_o2 * GasConstants.R_UNIVERSAL_J_PER_MOL_K *
                                        self.temperature_k) / self.volume_m3

    def get_state(self) -> Dict[str, Any]:
        """
        Retrieve the component's current operational state.

        Fulfills the Component Lifecycle Contract state access.

        Returns:
            Dict[str, Any]: State dictionary containing:
                - mass_kg (float): Current oxygen inventory (kg).
                - capacity_kg (float): Estimated capacity at target P (kg).
                - pressure_bar (float): Current pressure (bar).
                - temperature_c (float): Current temperature (°C).
                - fill_percentage (float): Inventory as % of capacity.
        """
        return {
            **super().get_state(),
            'mass_kg': float(self.mass_kg),
            'capacity_kg': float(self.capacity_kg),
            'pressure_bar': float(self.pressure_pa / 1e5),
            'temperature_c': float(self.temperature_k - 273.15),
            'fill_percentage': float(self.mass_kg / self.capacity_kg * 100) if self.capacity_kg > 0 else 0.0
        }

    def remove_oxygen(self, mass_kg: float) -> float:
        """
        Withdraw oxygen from the mixer.

        Args:
            mass_kg (float): Requested withdrawal mass in kg.

        Returns:
            float: Actual mass removed (limited by inventory).
        """
        removed = min(mass_kg, self.mass_kg)
        self.mass_kg -= removed
        self.output_mass_kg = removed
        self.cumulative_output_kg += removed
        return removed

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """
        Define the physical connection ports for this component.

        Returns:
            Dict[str, Dict[str, str]]: Port definitions.
        """
        return {
            'oxygen_in': {'type': 'input', 'resource_type': 'oxygen', 'units': 'kg/h'},
            'inlet': {'type': 'input', 'resource_type': 'oxygen', 'units': 'kg/h'},
            'oxygen_out': {'type': 'output', 'resource_type': 'oxygen', 'units': 'kg/h'}
        }
