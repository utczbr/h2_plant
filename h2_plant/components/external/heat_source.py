"""
External Heat Source Component.

This module provides thermal energy from external waste heat sources for
process integration. Waste heat recovery is critical for improving overall
plant efficiency, particularly for steam generation and feed preheating.

Thermal Integration:
    - **Temperature Quality**: Heat sources are characterized by temperature
      grade. Higher temperatures (>200°C) are suitable for steam generation,
      while lower grades can preheat feed water.
    - **Availability Factor**: Industrial waste heat may be intermittent
      depending on upstream process schedules.
    - **Minimum Turndown**: Heat exchangers require minimum flow to prevent
      thermal stress and maintain effectiveness.

Architecture:
    Implements the Component Lifecycle Contract (Layer 1):
    - `initialize()`: Standard initialization.
    - `step()`: Evaluates availability and delivers heat to match demand.
    - `get_state()`: Returns thermal output and utilization metrics.

Economic Model:
    Heat source may have an associated cost (EUR/kWh) representing either
    direct purchase price or opportunity cost of diverting heat from other
    uses. Zero cost represents free waste heat recovery.
"""

from typing import Dict, Any, Optional
import random

from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry


class ExternalHeatSource(Component):
    """
    External waste heat supply source for process integration.

    Provides thermal energy at a configurable temperature and power level.
    Supports demand-driven modulation with minimum turndown constraints.

    This component fulfills the Component Lifecycle Contract (Layer 1):
        - `initialize()`: Standard component initialization.
        - `step()`: Evaluates availability and matches heat delivery to demand.
        - `get_state()`: Returns output, utilization, and cost metrics.

    Attributes:
        thermal_power_kw (float): Maximum thermal output capacity (kW).
        temperature_c (float): Heat source temperature (°C).
        availability_factor (float): Fraction of time available (0-1).
        current_power_kw (float): Current heat delivery rate (kW).

    Example:
        >>> heat = ExternalHeatSource(
        ...     thermal_power_kw=500.0,
        ...     temperature_c=150.0,
        ...     availability_factor=0.95
        ... )
        >>> heat.initialize(dt=1/60, registry=registry)
        >>> heat.set_demand(300.0)  # Request 300 kW
        >>> heat.step(t=0.0)
        >>> print(f"Delivered: {heat.current_power_kw:.1f} kW")
    """

    def __init__(
        self,
        thermal_power_kw: float = 500.0,
        temperature_c: float = 150.0,
        availability_factor: float = 1.0,
        cost_per_kwh: float = 0.0,
        min_output_fraction: float = 0.2
    ):
        """
        Initialize the external heat source.

        Args:
            thermal_power_kw (float): Maximum thermal capacity in kW.
                Represents total recoverable heat from upstream process.
                Default: 500.0.
            temperature_c (float): Heat source temperature in °C. Determines
                which processes can use this heat (higher = more versatile).
                Default: 150.0.
            availability_factor (float): Probability of availability (0-1).
                Models intermittent upstream processes. Default: 1.0 (always on).
            cost_per_kwh (float): Heat cost in EUR/kWh. Zero for free waste
                heat, positive for purchased heat. Default: 0.0.
            min_output_fraction (float): Minimum turndown ratio (0-1). Heat
                delivery below this fraction is not possible. Default: 0.2.
        """
        super().__init__()

        self.thermal_power_kw = thermal_power_kw
        self.temperature_k = temperature_c + 273.15
        self.temperature_c = temperature_c
        self.availability_factor = availability_factor
        self.cost_per_kwh = cost_per_kwh
        self.min_output_fraction = min_output_fraction

        # State variables
        self.available = True
        self.heat_output_kwh = 0.0
        self.current_power_kw = 0.0
        self.cumulative_heat_kwh = 0.0
        self.cumulative_cost = 0.0
        self.heat_demand_kw = 0.0

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """
        Prepare the component for simulation execution.

        Fulfills the Component Lifecycle Contract initialization phase.

        Args:
            dt (float): Simulation timestep in hours.
            registry (ComponentRegistry): Central registry for component access.
        """
        super().initialize(dt, registry)

    def step(self, t: float) -> None:
        """
        Execute one simulation timestep.

        Evaluates stochastic availability, then delivers heat up to the
        requested demand, respecting capacity and turndown limits.

        Args:
            t (float): Current simulation time in hours.
        """
        super().step(t)

        # Stochastic availability based on upstream process reliability
        self.available = random.random() < self.availability_factor

        if not self.available:
            self.current_power_kw = 0.0
        elif self.heat_demand_kw > 0 and self.heat_demand_kw >= self.thermal_power_kw * self.min_output_fraction:
            # Deliver heat up to demand, capped at capacity
            self.current_power_kw = min(self.heat_demand_kw, self.thermal_power_kw)
        else:
            self.current_power_kw = 0.0

        # Calculate energy for this timestep
        self.heat_output_kwh = self.current_power_kw * self.dt

        # Update cumulative counters
        self.cumulative_heat_kwh += self.heat_output_kwh
        self.cumulative_cost += self.heat_output_kwh * self.cost_per_kwh

    def get_state(self) -> Dict[str, Any]:
        """
        Retrieve the component's current operational state.

        Fulfills the Component Lifecycle Contract state access.

        Returns:
            Dict[str, Any]: State dictionary containing:
                - available (bool): Current availability status.
                - thermal_power_kw (float): Maximum capacity (kW).
                - temperature_c (float): Heat source temperature (°C).
                - current_power_kw (float): Current delivery rate (kW).
                - heat_output_kwh (float): Energy delivered this step (kWh).
                - utilization (float): Fraction of capacity used (0-1).
                - cumulative_heat_kwh (float): Total energy delivered (kWh).
                - cumulative_cost (float): Total cost incurred (EUR).
        """
        return {
            **super().get_state(),
            'available': self.available,
            'thermal_power_kw': float(self.thermal_power_kw),
            'temperature_c': float(self.temperature_c),
            'temperature_k': float(self.temperature_k),
            'current_power_kw': float(self.current_power_kw),
            'heat_output_kwh': float(self.heat_output_kwh),
            'heat_demand_kw': float(self.heat_demand_kw),
            'cumulative_heat_kwh': float(self.cumulative_heat_kwh),
            'cumulative_cost': float(self.cumulative_cost),
            'utilization': float(self.current_power_kw / self.thermal_power_kw) if self.thermal_power_kw > 0 else 0.0,
            'flows': {
                'outputs': {
                    'thermal_energy': {
                        'value': float(self.heat_output_kwh),
                        'unit': 'kWh',
                        'temperature_c': float(self.temperature_c),
                        'destination': 'heat_consumers'
                    }
                }
            }
        }

    def set_demand(self, demand_kw: float) -> None:
        """
        Set heat demand from downstream consuming components.

        Called by heat consumers (steam generators, preheaters) to request
        thermal energy for the upcoming timestep.

        Args:
            demand_kw (float): Requested heat rate in kW.
        """
        self.heat_demand_kw = demand_kw

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """
        Define the physical connection ports for this component.

        Returns:
            Dict[str, Dict[str, str]]: Port definitions.
        """
        return {
            'heat_out': {'type': 'output', 'resource_type': 'thermal', 'units': 'kW'}
        }
