"""
Hydrogen Distribution Junction Component.

This module implements a hydrogen distribution junction that combines streams
from multiple production pathways (PEM electrolysis, SOEC electrolysis, and
autothermal reforming) while maintaining separate tracking for certification
and emissions accounting purposes.

Distribution Architecture:
    - **Green Hydrogen**: Electrolysis-derived H₂ (PEM + SOEC) with zero
      direct CO₂ emissions. Eligible for green hydrogen certification.
    - **Blue Hydrogen**: ATR-derived H₂ with associated CO₂ emissions from
      steam methane reforming. Requires carbon capture for certification.

The junction enables:
    1. Flexible production scheduling across pathways based on economics.
    2. Accurate emissions tracking for regulatory compliance.
    3. Downstream blending while preserving source attribution.

Architecture:
    Implements the Component Lifecycle Contract (Layer 1):
    - `initialize()`: Prepares component for simulation.
    - `step()`: Aggregates inlet flows and updates cumulative counters.
    - `get_state()`: Returns flow breakdown and emissions metadata.

Mixing Simplification:
    Temperature and pressure of the mixed stream are assumed constant for
    pure hydrogen streams. Rigorous mixing would require enthalpy balance
    (not implemented as H₂ streams are typically at similar conditions).
"""

from typing import Dict, Any, List

from h2_plant.core.component import Component
from h2_plant.core.stream import Stream


class H2Distribution(Component):
    """
    Hydrogen distribution junction with pathway attribution.

    Combines hydrogen from multiple production sources while tracking
    electrolysis (green) versus reformed (blue) origin for emissions
    accounting and certification purposes.

    This component fulfills the Component Lifecycle Contract (Layer 1):
        - `initialize()`: Standard initialization, marks component ready.
        - `step()`: Aggregates inflows by pathway and updates cumulative totals.
        - `get_state()`: Returns instantaneous flows, cumulative totals, and
          emissions metadata for downstream accounting.

    Port Mapping:
        - Index 0: PEM electrolysis input
        - Index 1: SOEC electrolysis input
        - Index 2: ATR reforming input

    Attributes:
        num_inputs (int): Number of hydrogen input streams.
        total_h2_output_kg_h (float): Combined output flow rate (kg/h).
        electrolysis_flow_kg_h (float): Green H₂ flow (PEM + SOEC, kg/h).
        atr_flow_kg_h (float): Blue H₂ flow (ATR, kg/h).

    Example:
        >>> dist = H2Distribution(component_id='H2_DIST', num_inputs=3)
        >>> dist.initialize(dt=1/60, registry=registry)
        >>> dist.add_inlet_flow(50.0, inlet_index=0)  # PEM
        >>> dist.add_inlet_flow(30.0, inlet_index=1)  # SOEC
        >>> dist.step(t=0.0)
        >>> print(f"Green: {dist.electrolysis_flow_kg_h:.1f} kg/h")
    """

    def __init__(
        self,
        component_id: str = "h2_distribution",
        num_inputs: int = 3
    ):
        """
        Initialize the hydrogen distribution junction.

        Args:
            component_id (str): Unique identifier for this component.
                Default: 'h2_distribution'.
            num_inputs (int): Number of hydrogen input streams.
                Default: 3 (PEM=0, SOEC=1, ATR=2).
        """
        super().__init__()
        self.component_id = component_id
        self.num_inputs = num_inputs

        # Input accumulation buffer (reset each timestep)
        self.inlet_flows_kg_h: List[float] = [0.0] * num_inputs

        # Instantaneous output state
        self.total_h2_output_kg_h = 0.0
        self.electrolysis_flow_kg_h = 0.0
        self.atr_flow_kg_h = 0.0

        # Cumulative tracking for mass balance verification
        self.cumulative_h2_kg = 0.0
        self.cumulative_electrolysis_kg = 0.0
        self.cumulative_atr_kg = 0.0

    def initialize(self, dt: float, registry: Any) -> None:
        """
        Prepare the component for simulation execution.

        Fulfills the Component Lifecycle Contract initialization phase.

        Args:
            dt (float): Simulation timestep in hours.
            registry (ComponentRegistry): Central registry for component access.
        """
        super().initialize(dt, registry)
        self._initialized = True

    def get_output(self, port_name: str) -> Any:
        """
        Retrieve the mixed hydrogen output stream.

        Args:
            port_name (str): Port identifier ('h2_out' or 'out').

        Returns:
            Stream: Mixed hydrogen stream at assumed LP conditions.

        Raises:
            ValueError: If port_name is not a valid output port.
        """
        if port_name in ['h2_out', 'out']:
            return Stream(
                mass_flow_kg_h=self.total_h2_output_kg_h,
                temperature_k=300.0,
                pressure_pa=30e5,
                composition={'H2': 1.0},
                phase='gas'
            )
        else:
            raise ValueError(f"Unknown output port '{port_name}' on {self.component_id}")

    def receive_input(self, port_name: str, value: Any, resource_type: str) -> float:
        """
        Accept hydrogen input from upstream component.

        Accumulates incoming flow to inlet buffer. Source attribution is
        lost when using the generic 'h2_in' port; for proper tracking,
        use add_inlet_flow() with explicit inlet_index.

        Args:
            port_name (str): Target port name.
            value (Any): Stream object or mass flow value.
            resource_type (str): Resource classification hint.

        Returns:
            float: Mass flow accepted (kg/h).

        Note:
            When multiple sources connect via the same port name, flows
            accumulate to index 0 (PEM), which may misclassify sources.
            Future topology should use distinct port names per source.
        """
        flow = 0.0
        if hasattr(value, 'mass_flow_kg_h'):
            flow = value.mass_flow_kg_h
        else:
            flow = float(value)

        # Aggregate to first inlet (limitation of single port name)
        self.inlet_flows_kg_h[0] += flow
        return flow

    def step(self, t: float) -> None:
        """
        Execute one simulation timestep.

        Aggregates inlet flows by source category and updates cumulative
        counters. Resets inlet buffer after processing.

        Args:
            t (float): Current simulation time in hours.
        """
        super().step(t)

        # Extract flows by source category
        pem_flow = self.inlet_flows_kg_h[0] if self.num_inputs > 0 else 0.0
        soec_flow = self.inlet_flows_kg_h[1] if self.num_inputs > 1 else 0.0
        atr_flow = self.inlet_flows_kg_h[2] if self.num_inputs > 2 else 0.0

        # Calculate pathway totals
        self.electrolysis_flow_kg_h = pem_flow + soec_flow
        self.atr_flow_kg_h = atr_flow

        # Combined output
        self.total_h2_output_kg_h = self.electrolysis_flow_kg_h + self.atr_flow_kg_h

        # Update cumulative mass (flow rate × timestep)
        self.cumulative_electrolysis_kg += self.electrolysis_flow_kg_h * self.dt
        self.cumulative_atr_kg += self.atr_flow_kg_h * self.dt
        self.cumulative_h2_kg += self.total_h2_output_kg_h * self.dt

        # Reset buffer for next timestep
        self.inlet_flows_kg_h = [0.0] * self.num_inputs

    def add_inlet_flow(self, flow_kg_h: float, inlet_index: int = 0) -> None:
        """
        Add hydrogen flow from a specific production pathway.

        Direct interface for explicit source attribution, bypassing the
        port system limitations.

        Args:
            flow_kg_h (float): Mass flow rate in kg/h.
            inlet_index (int): Source index (0=PEM, 1=SOEC, 2=ATR).
        """
        if 0 <= inlet_index < self.num_inputs:
            self.inlet_flows_kg_h[inlet_index] = flow_kg_h

    def get_state(self) -> Dict[str, Any]:
        """
        Retrieve the component's current operational state.

        Fulfills the Component Lifecycle Contract state access. Includes
        emissions metadata for downstream carbon accounting.

        Returns:
            Dict[str, Any]: State dictionary containing:
                - inlet_flows_kg_h (List[float]): Per-source flow rates.
                - total_h2_output_kg_h (float): Combined output rate.
                - electrolysis_flow_kg_h (float): Green H₂ rate.
                - atr_flow_kg_h (float): Blue H₂ rate.
                - cumulative_*_kg (float): Total mass by pathway.
                - emissions_metadata (Dict): CO₂ factors by pathway.
        """
        return {
            **super().get_state(),
            "inlet_flows_kg_h": self.inlet_flows_kg_h.copy(),
            "total_h2_output_kg_h": self.total_h2_output_kg_h,
            "electrolysis_flow_kg_h": self.electrolysis_flow_kg_h,
            "atr_flow_kg_h": self.atr_flow_kg_h,
            "cumulative_h2_kg": self.cumulative_h2_kg,
            "cumulative_electrolysis_kg": self.cumulative_electrolysis_kg,
            "cumulative_atr_kg": self.cumulative_atr_kg,
            "emissions_metadata": {
                "electrolysis": {"flow": self.electrolysis_flow_kg_h, "co2_factor": 0.0},
                "atr": {"flow": self.atr_flow_kg_h, "co2_factor": 10.5}
            }
        }

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """
        Define the physical connection ports for this component.

        Returns:
            Dict[str, Dict[str, str]]: Port definitions.
        """
        return {
            'h2_in': {'type': 'input', 'resource_type': 'hydrogen', 'units': 'kg/h'},
            'h2_out': {'type': 'output', 'resource_type': 'hydrogen', 'units': 'kg/h'},
            'out': {'type': 'output', 'resource_type': 'hydrogen', 'units': 'kg/h'}
        }
