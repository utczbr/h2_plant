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
        temperature_c: float = 25.0,
        # Proportional Control (Optional)
        reference_component_id: str = None,
        reference_ratio: float = None,
        reference_max_flow_kg_h: float = None  # Max O2 flow for ratio calculation
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
            reference_component_id (str): Optional. ID of the component to track
                for proportional control (e.g., 'O2_Backup_Supply').
            reference_ratio (float): Optional. Scale factor for output flow.
                output_flow = reference_flow * reference_ratio.
            reference_max_flow_kg_h (float): Optional. If reference_ratio is not
                provided but this is, ratio is auto-calculated as
                max_flow_rate_kg_h / reference_max_flow_kg_h.
        """
        super().__init__()
        self.component_id = component_id
        self.max_flow_rate_kg_h = max_flow_rate_kg_h
        self.methane_content = methane_content
        self.pressure_bar = pressure_bar
        self.temperature_k = temperature_c + 273.15
        
        # Proportional control settings
        self.reference_component_id = reference_component_id
        self._reference_component = None  # Resolved at initialize
        
        if reference_ratio is not None:
            self.reference_ratio = reference_ratio
        elif reference_max_flow_kg_h is not None and reference_max_flow_kg_h > 0:
            self.reference_ratio = max_flow_rate_kg_h / reference_max_flow_kg_h
        else:
            self.reference_ratio = None  # No proportional control

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
        
        # Resolve reference component if proportional control is enabled
        if self.reference_component_id and registry:
            self._reference_component = registry.get(self.reference_component_id)
        
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

        If proportional control is enabled, output flow is scaled based on the
        reference component's output. Otherwise, outputs at max capacity.

        Args:
            t (float): Current simulation time in hours.
        """
        super().step(t)

        # Determine output flow
        if self._reference_component and self.reference_ratio:
            # Proportional control active
            ref_flow = getattr(self._reference_component, 'get_output_mass_flow', lambda: 0.0)()
            self.biogas_output_kg_h = min(ref_flow * self.reference_ratio, self.max_flow_rate_kg_h)
        else:
            # Fixed utilization rate (100%)
            self.biogas_output_kg_h = self.max_flow_rate_kg_h
            
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
