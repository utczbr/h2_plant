"""
Throttling Valve Component (Joule-Thomson Expansion).

This module implements an isenthalpic throttling valve for pressure reduction
in process streams. The valve models real-gas expansion behavior, correctly
capturing the Joule-Thomson effect for hydrogen.

Thermodynamic Principles
------------------------
1.  **Isoenthalpic Expansion**:
    Throttling is modeled as an adiabatic, irreversible process where enthalpy is conserved despite 
    pressure loss and entropy generation.
    $$ H_{in}(P_{in}, T_{in}) = H_{out}(P_{out}, T_{out}) $$

2.  **Joule-Thomson Effect**:
    Real gases experience a temperature change characterized by the Joule-Thomson coefficient 
    $\mu_{JT} = (\partial T / \partial P)_H$.
    $$ \Delta T \approx \int_{P_{in}}^{P_{out}} \mu_{JT} \, dP $$

3.  **Hydrogen Inversion**:
    Hydrogen has an inversion temperature of approx. 202 K. At typical operating temperatures ($> 202$ K), 
    $\mu_{JT} < 0$, meaning expansion ($dP < 0$) causes **heating** ($dT > 0$).

Architecture
------------
*   **Component Lifecycle Contract (Layer 1)**:
    *   `initialize()`: Establishes connection to `LUTManager` (Layer 2).
    *   `step()`: Executes the isoenthalpic flash solve.
    *   `get_state()`: Reports thermodynamic performance ($\Delta T$, $\Delta P$).

Model Approach
--------------
The outlet temperature is determined by numerically solving the energy conservation equation 
$H(P_{out}, T_{out}) - H_{in} = 0$ using the Bisection Method. This captures all real-gas 
non-idealities encoded in the Equation of State.

References
----------
*   Smith, Van Ness & Abbott (2005). Introduction to Chemical Engineering Thermodynamics.
*   NIST Chemistry WebBook: Hydrogen thermophysical properties.
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

    Models irreversible expansion through a restriction. Enthalpy is conserved ($H_{in} = H_{out}$) 
    while entropy increases ($S_{out} > S_{in}$).

    **Key Physical Behavior**:
    For hydrogen at standard conditions ($T > T_{inv} \approx 202 \text{ K}$), the Joule-Thomson 
    coefficient is negative. A pressure drop from 40 bar to 1 bar at 300 K results in a temperature 
    **increase** of approximately 5-10 K.

    **Architecture**:
    *   **Layer 1**: Standard component interface for flow and state.
    *   **Layer 2**: Direct `LUTManager` integration for high-speed property inversion.

    Attributes:
        P_out_pa (float): Target outlet pressure [Pa].
        fluid (str): Fluid species identifier for EOS lookup (e.g., 'H2').
        delta_T (float): Temperature change across valve [K]. Positive indicates heating.
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

        **Lifecycle Contract**:
        Acquires reference to `LUTManager` (Layer 2 service) to enable real-gas property 
        lookups during the `step()` phase. This dependency injection is critical for 
        accurate Joule-Thomson modeling.

        Args:
            dt (float): Simulation timestep [hours].
            registry (ComponentRegistry): Central registry for component access.
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

        **Physics Logic**:
        1.  **Check Pass-Through**: If $P_{target} \ge P_{in}$, no expansion occurs.
        2.  **State Determination**:
            -   Lookup Inlet Enthalpy: $H_{in} = H(P_{in}, T_{in})$ via LUT.
            -   Solve Flash: Find $T_{out}$ such that $H(P_{target}, T_{out}) = H_{in}$.
        3.  **Result**: Updates `outlet_stream` with new state $(P_{target}, T_{out})$.

        Args:
            t (float): Current simulation time [hours].
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
        Solve for temperature at constant enthalpy using Bisection.

        **Root Finding Problem**:
        Find $T$ such that $f(T) = H(P_{target}, T) - h_{target} = 0$.

        **Algorithm (Bisection)**:
        1.  **Bracket**: Establish $[T_{low}, T_{high}]$ such that $f(T_{low}) \cdot f(T_{high}) < 0$. 
            Expands bounds if necessary (robustness).
        2.  **Iterate**: $T_{mid} = (T_{low} + T_{high}) / 2$.
        3.  **Update**: Check sign of $f(T_{mid})$ and contract interval.
        4.  **Converge**: Tolerance $\epsilon_H = 1.0 \text{ J/kg}$ or max iterations.

        Args:
            P_target_pa (float): Target pressure [Pa].
            h_target (float): Conserved enthalpy [J/kg].

        Returns:
            float: Solution temperature [K].
        """
        T_guess = self.inlet_stream.temperature_k
        
        # Determine LUT safe boundaries to avoid nuisance warnings
        T_min_lut = 273.15  # Default fallback
        if hasattr(self.lut_mgr, 'config'):
            T_min_lut = self.lut_mgr.config.temperature_min

        # Initial bracket attempts to stay within LUT validity
        T_low_raw = T_guess - 50.0
        T_low = max(T_low_raw, T_min_lut)  # Clamp to LUT min
        T_high = T_guess + 50.0

        # Verify bracketing
        h_low = self.lut_mgr.lookup(self.fluid, 'H', P_target_pa, T_low)
        h_high = self.lut_mgr.lookup(self.fluid, 'H', P_target_pa, T_high)

        # Robustness Logic:
        # If the target is NOT bracketed by the clamped range, we must check why.
        # CASE A: h_target < h_low. 
        #   This means the required Enthalpy is LOWER than the enthalpy at T_min_lut.
        #   Therefore, the solution temperature is < T_min_lut (Cryogenic/OOB).
        #   We MUST expand downwards and accept the LUT warning/fallback.
        if h_target < h_low:
             T_low = 20.0 # Deep cryogenic search
             h_low = self.lut_mgr.lookup(self.fluid, 'H', P_target_pa, T_low)

        # CASE B: h_target > h_high.
        #   Solution is very hot. Expand upwards.
        elif h_target > h_high:
             T_high = 1500.0
             h_high = self.lut_mgr.lookup(self.fluid, 'H', P_target_pa, T_high)

        # Final check if we failed to bracket even after expansion
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
