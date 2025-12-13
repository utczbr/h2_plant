"""
Throttling Valve Component (Joule-Thomson Expansion).

This module implements an isenthalpic throttling valve for pressure reduction
in process streams. The valve models real-gas expansion behavior, correctly
capturing the Joule-Thomson effect for hydrogen.

Thermodynamic Principles:
    - **Isoenthalpic Expansion**: Throttling is an irreversible process where
      enthalpy is conserved (H_in = H_out). No work is extracted and heat
      transfer is negligible for rapid expansion.
    - **Joule-Thomson Effect**: Real gases experience temperature change during
      throttling. The direction depends on whether the gas is above or below
      its inversion temperature.
    - **Hydrogen Behavior**: H₂ has an inversion temperature of ~202 K. Above
      this (typical operating conditions), the Joule-Thomson coefficient is
      negative: expansion causes HEATING, not cooling.

Architecture:
    Implements the Component Lifecycle Contract (Layer 1):
    - `initialize()`: Connects to LUTManager for thermodynamic property lookups.
    - `step()`: Calculates outlet temperature via isoenthalpic flash.
    - `get_state()`: Reports pressure drop and temperature change.

Model Approach:
    The outlet temperature is found by solving H(P_out, T_out) = H_in using
    bisection on temperature. This captures real-gas non-idealities including
    the anomalous heating behavior of hydrogen at room temperature.

References:
    - Smith, Van Ness & Abbott (2005). Introduction to Chemical Engineering
      Thermodynamics, 7th Ed., Section 3.3.
    - NIST Chemistry WebBook: Hydrogen thermophysical properties.
"""

from typing import Any, Dict, Optional
import logging

from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.component_ids import ComponentID
from h2_plant.core.stream import Stream

logger = logging.getLogger(__name__)


class ThrottlingValve(Component):
    """
    Isenthalpic throttling valve for pressure reduction.

    Models irreversible expansion through a valve or restriction, where
    enthalpy is conserved but entropy increases. Uses real-gas properties
    from LUTManager to calculate the temperature change accurately.

    This component fulfills the Component Lifecycle Contract (Layer 1):
        - `initialize()`: Acquires reference to LUTManager for enthalpy lookups.
        - `step()`: Solves isoenthalpic flash to determine outlet temperature.
        - `get_state()`: Returns pressure drop and temperature change metrics.

    Key Physical Behavior:
        For hydrogen above its inversion temperature (~202 K), throttling
        causes heating rather than cooling. At 300 K and 40 bar → 1 bar,
        expect a temperature rise of approximately 5-10 K.

    Attributes:
        P_out_pa (float): Target outlet pressure in Pa.
        fluid (str): Fluid species identifier for property lookups.
        delta_T (float): Temperature change across valve (K), positive = heating.

    Example:
        >>> valve = ThrottlingValve({'P_out_pa': 101325.0, 'fluid': 'H2'})
        >>> valve.initialize(dt=1/60, registry=registry)
        >>> valve.receive_input('inlet', high_pressure_stream, 'gas')
        >>> valve.step(t=0.0)
        >>> outlet = valve.get_output('outlet')
        >>> print(f"ΔT = {valve.delta_T:.2f} K")
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the throttling valve.

        Args:
            config (Dict[str, Any]): Configuration dictionary containing:
                - P_out_pa (float): Target outlet pressure in Pa. Default: 101325 (1 atm).
                - fluid (str): Fluid species for property lookups. Default: 'H2'.
                - component_id (str, optional): Unique component identifier.
        """
        super().__init__(config)
        self.P_out_pa = float(config.get('P_out_pa', 101325.0))
        self.fluid = config.get('fluid', 'H2')
        self.lut_mgr = None

        # Stream state
        self.inlet_stream: Optional[Stream] = None
        self.outlet_stream: Optional[Stream] = None

        # Performance metrics
        self.delta_T = 0.0

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """
        Prepare the valve for simulation execution.

        Fulfills the Component Lifecycle Contract initialization phase by
        acquiring a reference to the LUTManager for thermodynamic property
        lookups during step execution.

        Args:
            dt (float): Simulation timestep in hours.
            registry (ComponentRegistry): Central registry for component access.

        Note:
            If LUTManager is unavailable, the valve falls back to ideal gas
            behavior (constant temperature throttling).
        """
        super().initialize(dt, registry)
        if registry.has(ComponentID.LUT_MANAGER.value):
            self.lut_mgr = registry.get(ComponentID.LUT_MANAGER)
        else:
            logger.warning(
                f"ThrottlingValve {self.component_id}: LUTManager not found. "
                "Thermodynamic calculations will use simplified model."
            )

    def receive_input(self, port_name: str, value: Any, resource_type: str) -> float:
        """
        Accept an input stream at the specified port.

        Args:
            port_name (str): Target port identifier. Expected: 'inlet'.
            value (Any): Stream object containing flow, temperature, and pressure.
            resource_type (str): Resource classification hint.

        Returns:
            float: Mass flow rate accepted (kg/h), or 0.0 if rejected.
        """
        if port_name == 'inlet':
            if isinstance(value, Stream):
                self.inlet_stream = value
                return value.mass_flow_kg_h
        return 0.0

    def step(self, t: float) -> None:
        """
        Execute one simulation timestep.

        Performs isoenthalpic expansion calculation:
        1. If outlet pressure ≥ inlet pressure, no throttling occurs (pass-through).
        2. Otherwise, inlet enthalpy is computed from LUT.
        3. Outlet temperature is found by bisection: solve H(P_out, T) = H_in.

        The resulting temperature change reflects real-gas behavior, including
        the Joule-Thomson heating effect for hydrogen above inversion temperature.

        Args:
            t (float): Current simulation time in hours.
        """
        super().step(t)

        if self.inlet_stream is None or self.inlet_stream.mass_flow_kg_h <= 0:
            self.outlet_stream = None
            self.delta_T = 0.0
            return

        P_in = self.inlet_stream.pressure_pa
        P_target = self.P_out_pa

        # No throttling if outlet pressure >= inlet (valve wide open or backflow)
        if P_target >= P_in:
            self.outlet_stream = Stream(
                mass_flow_kg_h=self.inlet_stream.mass_flow_kg_h,
                temperature_k=self.inlet_stream.temperature_k,
                pressure_pa=P_in,
                composition=self.inlet_stream.composition,
                phase=self.inlet_stream.phase
            )
            self.delta_T = 0.0
            return

        # Isoenthalpic expansion: solve for T_out such that H(P_out, T_out) = H_in
        T_out = self.inlet_stream.temperature_k

        if self.lut_mgr:
            try:
                h_in = self.lut_mgr.lookup(
                    self.fluid, 'H', P_in, self.inlet_stream.temperature_k
                )

                T_out = self._solve_T_isoenthalpic(P_target, h_in)

            except Exception as e:
                logger.error(f"Valve LUT error: {e}")
                # Fallback to constant-T (ideal gas approximation)
                T_out = self.inlet_stream.temperature_k

        self.delta_T = T_out - self.inlet_stream.temperature_k

        self.outlet_stream = Stream(
            mass_flow_kg_h=self.inlet_stream.mass_flow_kg_h,
            temperature_k=T_out,
            pressure_pa=P_target,
            composition=self.inlet_stream.composition,
            phase=self.inlet_stream.phase
        )

    def get_output(self, port_name: str) -> Any:
        """
        Retrieve the output stream from a specified port.

        Args:
            port_name (str): Port identifier. Expected: 'outlet'.

        Returns:
            Stream: Throttled stream at outlet conditions, or None if no flow.
        """
        if port_name == 'outlet':
            return self.outlet_stream
        return None

    def get_state(self) -> Dict[str, Any]:
        """
        Retrieve the component's current operational state.

        Fulfills the Component Lifecycle Contract state access, providing
        valve operating metrics for monitoring and logging.

        Returns:
            Dict[str, Any]: State dictionary containing:
                - inlet_pressure_bar (float): Upstream pressure (bar).
                - outlet_pressure_bar (float): Downstream pressure (bar).
                - flow_rate_kg_h (float): Current mass flow (kg/h).
                - delta_T_K (float): Temperature change (K), positive = heating.
        """
        return {
            **super().get_state(),
            "inlet_pressure_bar": (self.inlet_stream.pressure_pa / 1e5) if self.inlet_stream else 0.0,
            "outlet_pressure_bar": self.P_out_pa / 1e5,
            "flow_rate_kg_h": self.inlet_stream.mass_flow_kg_h if self.inlet_stream else 0.0,
            "delta_T_K": self.delta_T
        }

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """
        Define the physical connection ports for this component.

        Returns:
            Dict[str, Dict[str, str]]: Port definitions with keys:
                - inlet: High-pressure gas feed.
                - outlet: Reduced-pressure gas product.
        """
        return {
            'inlet': {'type': 'input', 'resource_type': 'gas', 'units': 'kg/h'},
            'outlet': {'type': 'output', 'resource_type': 'gas', 'units': 'kg/h'}
        }

    def _solve_T_isoenthalpic(self, P_target_pa: float, h_target: float) -> float:
        """
        Solve for temperature at constant enthalpy using bisection.

        Finds T such that H(P_target, T) = h_target. Uses bisection method
        which is robust for monotonic enthalpy-temperature relationships.

        Args:
            P_target_pa (float): Target pressure in Pa.
            h_target (float): Target enthalpy in J/kg (conserved from inlet).

        Returns:
            float: Solution temperature in K.

        Note:
            Convergence tolerance is 1 J/kg, adequate for engineering accuracy.
            Expands search bounds if initial guess brackets fail.
        """
        T_guess = self.inlet_stream.temperature_k
        T_low = T_guess - 50.0
        T_high = T_guess + 50.0

        # Verify bracketing
        h_low = self.lut_mgr.lookup(self.fluid, 'H', P_target_pa, T_low)
        h_high = self.lut_mgr.lookup(self.fluid, 'H', P_target_pa, T_high)

        # Expand bounds if target not bracketed
        if not (h_low < h_target < h_high):
            T_low = 20.0
            T_high = 1000.0
            h_low = self.lut_mgr.lookup(self.fluid, 'H', P_target_pa, T_low)
            h_high = self.lut_mgr.lookup(self.fluid, 'H', P_target_pa, T_high)

            if not (h_low < h_target < h_high):
                return T_guess

        # Bisection loop (20 iterations yields ~1e-6 relative precision)
        for _ in range(20):
            T_mid = (T_low + T_high) / 2
            h_mid = self.lut_mgr.lookup(self.fluid, 'H', P_target_pa, T_mid)

            if abs(h_mid - h_target) < 1.0:
                return T_mid

            if h_mid < h_target:
                T_low = T_mid
            else:
                T_high = T_mid

        return (T_low + T_high) / 2
