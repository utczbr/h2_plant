"""
External Oxygen Source Component.

This module provides oxygen supply from external sources (pipeline, liquid
storage, or on-site ASU) for processes requiring supplemental O₂ beyond
electrolysis byproduct availability.

Supply Modes:
    - **Fixed Flow**: Constant delivery rate regardless of downstream demand.
      Suitable for contracted supply or pipeline delivery.
    - **Pressure Driven**: Modulates flow to maintain target pressure in
      downstream buffer. Suitable for on-demand supply with storage.

Architecture:
    Implements the Component Lifecycle Contract (Layer 1):
    - `initialize()`: Connects to downstream targets (mixer or buffer).
    - `step()`: Delivers oxygen according to configured mode.
    - `get_state()`: Returns flow, cost, and connection metrics.

Economic Considerations:
    External O₂ purchase costs are tracked for techno-economic analysis.
    Typical industrial oxygen costs range from 0.05-0.30 EUR/kg depending
    on purity and delivery method.
"""

import logging
from typing import Dict, Any, Optional

from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.constants import GasConstants

logger = logging.getLogger(__name__)


class ExternalOxygenSource(Component):
    """
    External oxygen supply source for process supplementation.

    Provides oxygen at a configurable flow rate and pressure when electrolysis
    byproduct is insufficient for downstream requirements (e.g., ATR oxidant).

    This component fulfills the Component Lifecycle Contract (Layer 1):
        - `initialize()`: Locates downstream mixer or buffer for connection.
        - `step()`: Delivers oxygen according to fixed_flow or pressure_driven mode.
        - `get_state()`: Returns delivery rate, cumulative totals, and costs.

    Attributes:
        mode (str): Operating mode ('fixed_flow' or 'pressure_driven').
        flow_rate_kg_h (float): Configured flow rate for fixed mode (kg/h).
        pressure_bar (float): Supply pressure (bar).
        cost_per_kg (float): Oxygen purchase cost (EUR/kg).

    Example:
        >>> o2_source = ExternalOxygenSource(
        ...     mode='fixed_flow',
        ...     flow_rate_kg_h=50.0,
        ...     pressure_bar=10.0
        ... )
        >>> o2_source.initialize(dt=1/60, registry=registry)
        >>> o2_source.step(t=0.0)
        >>> print(f"O₂ delivered: {o2_source.o2_output_kg:.2f} kg")
    """

    def __init__(
        self,
        mode: str = "fixed_flow",
        flow_rate_kg_h: float = 0.0,
        pressure_bar: float = 5.0,
        cost_per_kg: float = 0.15,
        max_capacity_kg_h: float = 100.0
    ):
        """
        Initialize the external oxygen source.

        Args:
            mode (str): Operating mode. Options:
                - 'fixed_flow': Constant delivery at flow_rate_kg_h.
                - 'pressure_driven': Modulates to maintain downstream pressure.
                Default: 'fixed_flow'.
            flow_rate_kg_h (float): Flow rate for fixed_flow mode in kg/h.
                Default: 0.0.
            pressure_bar (float): Supply pressure in bar gauge. Must exceed
                downstream system pressure. Default: 5.0.
            cost_per_kg (float): Purchase cost in EUR/kg O₂. Default: 0.15.
            max_capacity_kg_h (float): Maximum supply capacity in kg/h for
                pressure_driven mode. Default: 100.0.
        """
        super().__init__()

        self.mode = mode
        self.flow_rate_kg_h = flow_rate_kg_h
        self.pressure_bar = pressure_bar
        self.cost_per_kg = cost_per_kg
        self.max_capacity_kg_h = max_capacity_kg_h

        # State variables
        self.o2_output_kg = 0.0
        self.cumulative_o2_kg = 0.0
        self.cumulative_cost = 0.0
        self._target_component: Optional[Component] = None

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """
        Prepare the component for simulation execution.

        Fulfills the Component Lifecycle Contract initialization phase.
        Attempts to locate downstream oxygen mixer or buffer for pressure-driven
        mode operation.

        Args:
            dt (float): Simulation timestep in hours.
            registry (ComponentRegistry): Central registry for component access.
        """
        super().initialize(dt, registry)

        # Connect to downstream oxygen handling components
        if registry.has('oxygen_mixer'):
            self._target_component = registry.get('oxygen_mixer')
        elif registry.has('oxygen_buffer'):
            self._target_component = registry.get('oxygen_buffer')

    def step(self, t: float) -> None:
        """
        Execute one simulation timestep.

        Delivers oxygen according to the configured operating mode:
        - Fixed flow: Constant delivery at flow_rate_kg_h.
        - Pressure driven: Estimated flow to maintain downstream pressure.

        Args:
            t (float): Current simulation time in hours.
        """
        super().step(t)

        if self.mode == "fixed_flow":
            self.o2_output_kg = self.flow_rate_kg_h * self.dt

        elif self.mode == "pressure_driven":
            if self._target_component:
                required_flow = self._estimate_required_flow()
                self.o2_output_kg = min(required_flow, self.max_capacity_kg_h * self.dt)
            else:
                self.o2_output_kg = 0.0

        # Update cumulative counters
        self.cumulative_o2_kg += self.o2_output_kg
        self.cumulative_cost += self.o2_output_kg * self.cost_per_kg

    def get_state(self) -> Dict[str, Any]:
        """
        Retrieve the component's current operational state.

        Fulfills the Component Lifecycle Contract state access.

        Returns:
            Dict[str, Any]: State dictionary containing:
                - mode (str): Current operating mode.
                - flow_rate_kg_h (float): Configured flow rate (kg/h).
                - o2_output_kg (float): O₂ delivered this timestep (kg).
                - pressure_bar (float): Supply pressure (bar).
                - cumulative_o2_kg (float): Total O₂ delivered (kg).
                - cumulative_cost (float): Total purchase cost (EUR).
        """
        return {
            **super().get_state(),
            'mode': self.mode,
            'flow_rate_kg_h': float(self.flow_rate_kg_h),
            'o2_output_kg': float(self.o2_output_kg),
            'pressure_bar': float(self.pressure_bar),
            'cumulative_o2_kg': float(self.cumulative_o2_kg),
            'cumulative_cost': float(self.cumulative_cost),
            'flows': {
                'outputs': {
                    'oxygen': {
                        'value': float(self.o2_output_kg),
                        'unit': 'kg',
                        'destination': self._target_component.component_id if self._target_component else 'unconnected'
                    }
                }
            }
        }

    def _estimate_required_flow(self) -> float:
        """
        Estimate O₂ flow required to maintain downstream pressure.

        Placeholder for pressure-driven control logic. A production
        implementation would query the target component's pressure state
        and compute makeup flow via tank pressure dynamics.

        Returns:
            float: Estimated required flow in kg (for this timestep).
        """
        # Placeholder: return nominal flow for stub implementation
        return 10.0

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """
        Define the physical connection ports for this component.

        Returns:
            Dict[str, Dict[str, str]]: Port definitions.
        """
        return {
            'o2_out': {'type': 'output', 'resource_type': 'oxygen', 'units': 'kg/h'}
        }
