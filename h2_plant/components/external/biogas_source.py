"""
Biogas Source Component.

This module provides biogas feedstock for autothermal reforming (ATR) processes.
Biogas is a renewable fuel derived from anaerobic digestion of organic waste,
primarily composed of methane (CH₄) and carbon dioxide (CO₂).

Feedstock Characteristics:
    - **Methane Content**: Typically 50-70% CH₄ depending on source (landfill,
      agricultural waste, wastewater treatment).
    - **CO₂ Ballast**: The CO₂ component participates in reforming reactions
      (dry reforming) and reduces external CO₂ injection requirements.
    - **Trace Contaminants**: Real biogas contains H₂S and siloxanes that
      require upstream cleanup (not modeled here).

Architecture:
    Implements the Component Lifecycle Contract (Layer 1):
    - `initialize()`: Prepares component for simulation.
    - `step()`: Provides biogas at configured utilization rate.
    - `get_state()`: Returns flow rate and cumulative delivery metrics.

Model Simplifications:
    This is a supply-side stub component operating at 50% utilization of
    maximum capacity. Future extensions could implement demand-driven
    modulation based on ATR reactor requirements.
"""

from typing import Dict, Any

from h2_plant.core.component import Component
from h2_plant.core.stream import Stream


class BiogasSource(Component):
    """
    Biogas supply source for autothermal reforming feedstock.

    Provides methane-rich biogas at a configurable flow rate and pressure.
    The composition reflects typical landfill or agricultural biogas.

    This component fulfills the Component Lifecycle Contract (Layer 1):
        - `initialize()`: Standard initialization, marks component ready.
        - `step()`: Calculates biogas delivery for current timestep.
        - `get_state()`: Returns output rate and cumulative totals.

    Attributes:
        max_flow_rate_kg_h (float): Maximum biogas capacity (kg/h).
        methane_content (float): CH₄ mole fraction (0-1).
        pressure_bar (float): Supply pressure (bar gauge).
        biogas_output_kg_h (float): Current output rate (kg/h).

    Example:
        >>> source = BiogasSource(
        ...     component_id='BG_01',
        ...     max_flow_rate_kg_h=1000.0,
        ...     methane_content=0.60
        ... )
        >>> source.initialize(dt=1/60, registry=registry)
        >>> source.step(t=0.0)
        >>> stream = source.get_output('biogas_out')
    """

    def __init__(
        self,
        component_id: str = "biogas_source",
        max_flow_rate_kg_h: float = 1000.0,
        methane_content: float = 0.60,
        pressure_bar: float = 5.0,
        temperature_c: float = 25.0
    ):
        """
        Initialize the biogas source.

        Args:
            component_id (str): Unique component identifier. Default: 'biogas_source'.
            max_flow_rate_kg_h (float): Maximum biogas flow rate in kg/h.
                Represents digester or pipeline capacity. Default: 1000.0.
            methane_content (float): Methane mole fraction (0-1). Typical
                range is 0.50-0.70 depending on feedstock. Default: 0.60.
            pressure_bar (float): Supply pressure in bar gauge. Must exceed
                ATR reactor pressure for flow. Default: 5.0.
            temperature_c (float): Supply temperature in Celsius. Default: 25.0.
        """
        super().__init__()
        self.component_id = component_id
        self.max_flow_rate_kg_h = max_flow_rate_kg_h
        self.methane_content = methane_content
        self.pressure_bar = pressure_bar
        self.temperature_k = temperature_c + 273.15

        # State variables
        self.biogas_output_kg_h = 0.0
        self.cumulative_biogas_kg = 0.0

    def initialize(self, dt: float, registry: Any) -> None:
        """
        Prepare the component for simulation execution.

        Fulfills the Component Lifecycle Contract initialization phase.

        Args:
            dt (float): Simulation timestep in hours.
            registry (ComponentRegistry): Central registry for component access.
        """
        super().initialize(dt, registry)
        
        # Pre-allocate output stream to avoid object creation in inner loops
        self._output_stream = Stream(
            mass_flow_kg_h=0.0,
            temperature_k=self.temperature_k,
            pressure_pa=self.pressure_bar * 1e5,
            composition={'CH4': self.methane_content, 'CO2': 1.0 - self.methane_content},
            phase='gas'
        )
        self._initialized = True

    def step(self, t: float) -> None:
        """
        Execute one simulation timestep.

        Calculates biogas output at a fixed 50% utilization of maximum
        capacity. This represents steady-state digester operation.

        Args:
            t (float): Current simulation time in hours.
        """
        super().step(t)

        # Fixed utilization rate (stub for demand-driven logic)
        self.biogas_output_kg_h = self.max_flow_rate_kg_h * 0.5
        self.cumulative_biogas_kg += self.biogas_output_kg_h * self.dt
        
        # Update cached stream
        self._output_stream.mass_flow_kg_h = self.biogas_output_kg_h

    def get_state(self) -> Dict[str, Any]:
        """
        Retrieve the component's current operational state.

        Fulfills the Component Lifecycle Contract state access.

        Returns:
            Dict[str, Any]: State dictionary containing:
                - biogas_output_kg_h (float): Current output rate (kg/h).
                - cumulative_biogas_kg (float): Total biogas delivered (kg).
                - methane_content (float): CH₄ mole fraction.
                - pressure_bar (float): Supply pressure (bar).
        """
        return {
            **super().get_state(),
            "biogas_output_kg_h": self.biogas_output_kg_h,
            "cumulative_biogas_kg": self.cumulative_biogas_kg,
            "methane_content": self.methane_content,
            "pressure_bar": self.pressure_bar
        }

    def get_output(self, port_name: str) -> Any:
        """
        Retrieve output from a specified port.

        Args:
            port_name (str): Port identifier ('biogas_out' or 'out').

        Returns:
            Stream: Biogas stream with flow, temperature, pressure, and composition.

        Raises:
            ValueError: If port_name is not a valid output port.
        """
        if port_name in ['biogas_out', 'out']:
            return self._output_stream
        else:
            raise ValueError(f"Unknown output port '{port_name}' on {self.component_id}")

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """
        Define the physical connection ports for this component.

        Returns:
            Dict[str, Dict[str, str]]: Port definitions.
        """
        return {
            'biogas_out': {'type': 'output', 'resource_type': 'gas', 'units': 'kg/h'},
            'out': {'type': 'output', 'resource_type': 'gas', 'units': 'kg/h'}
        }
