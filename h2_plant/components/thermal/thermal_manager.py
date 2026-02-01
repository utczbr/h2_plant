"""
Centralized Thermal Management System.

This module implements a thermal manager that coordinates heat recovery
and distribution across the hydrogen plant. It collects waste heat from
high-temperature sources and distributes it to heat sinks.

Energy Integration:
    Industrial hydrogen plants generate significant waste heat from:
    - **PEM Electrolysis**: 80-100°C waste heat from stack cooling
    - **Compressors**: Intercooler heat from compression stages

    This heat can be recovered and used for:
    - **SOEC Steam Generation**: Pre-heating water/steam
    - **ATR Pre-heating**: Feedstock heating

    The thermal manager acts as a virtual heat bus, matching available
    heat with demand and tracking utilization efficiency.

Architecture:
    Implements the Component Lifecycle Contract (Layer 1):
    - `initialize()`: Standard initialization.
    - `step()`: Collects heat, matches supply/demand, updates sinks.
    - `get_state()`: Returns heat flows and efficiency metrics.
"""

from typing import Dict, Any, Optional

from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.component_ids import ComponentID


class ThermalManager(Component):
    """
    Centralized thermal management for heat recovery and distribution.

    Collects waste heat from high-temperature sources (PEM, compressors)
    and distributes to heat sinks (SOEC steam gen, ATR pre-heat).
    Tracks thermal efficiency and cumulative heat flows.

    This component fulfills the Component Lifecycle Contract (Layer 1):
        - `initialize()`: Standard component initialization.
        - `step()`: Collects heat sources, satisfies demand, updates sinks.
        - `get_state()`: Returns heat flows and utilization efficiency.

    Heat Balance:
        - Q_available = Σ(heat from sources)
        - Q_demand = Σ(heat required by sinks)
        - Q_utilized = min(Q_available, Q_demand)
        - Q_wasted = Q_available - Q_utilized

    Attributes:
        total_heat_available_kw (float): Current heat supply (kW).
        total_heat_demand_kw (float): Current heat demand (kW).
        heat_utilized_kw (float): Heat successfully transferred (kW).
        heat_wasted_kw (float): Unrecovered heat (kW).

    Example:
        >>> tm = ThermalManager()
        >>> tm.initialize(dt=1/60, registry=registry)
        >>> tm.step(t=0.0)
        >>> print(f"Utilization: {tm.heat_utilized_kw / tm.total_heat_available_kw * 100:.1f}%")
    """

    def __init__(self):
        """
        Initialize the thermal manager.
        """
        super().__init__()

        # Instantaneous state
        self.total_heat_available_kw = 0.0
        self.total_heat_demand_kw = 0.0
        self.heat_utilized_kw = 0.0
        self.heat_wasted_kw = 0.0

        # Cumulative tracking
        self.cumulative_heat_recovered_kwh = 0.0
        self.cumulative_heat_utilized_kwh = 0.0
        self.cumulative_heat_wasted_kwh = 0.0

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

        Performs thermal management cycle:
        1. Collect heat from sources (PEM, compressors).
        2. Identify heat demand from sinks (SOEC steam gen).
        3. Distribute available heat to sinks.
        4. Track wasted heat and update accumulators.

        Args:
            t (float): Current simulation time in hours.
        """
        super().step(t)

        self.total_heat_available_kw = 0.0
        self.total_heat_demand_kw = 0.0
        self.heat_utilized_kw = 0.0

        # Collect heat from PEM electrolyzer
        try:
            pem = self.get_registry_safe(ComponentID.PEM_ELECTROLYZER_DETAILED)
            if hasattr(pem, 'heat_output_kw'):
                self.total_heat_available_kw += pem.heat_output_kw
        except Exception:
            pass

        # Collect heat from compressors
        for comp_id, comp in self._registry._components.items():
            if "compressor" in comp_id and hasattr(comp, 'heat_output_kw'):
                self.total_heat_available_kw += comp.heat_output_kw

        # Identify demand from SOEC steam generator
        steam_gen = None
        try:
            soec = self.get_registry_safe(ComponentID.SOEC_CLUSTER)
            if hasattr(soec, 'steam_gen_hx4'):
                steam_gen = soec.steam_gen_hx4
                if hasattr(steam_gen, 'total_heat_demand_kw'):
                    self.total_heat_demand_kw += steam_gen.total_heat_demand_kw
                elif hasattr(steam_gen, 'heat_input_kw'):
                    self.total_heat_demand_kw += steam_gen.heat_input_kw
        except Exception:
            pass

        # Distribute heat (simple supply-demand matching)
        self.heat_utilized_kw = min(self.total_heat_available_kw, self.total_heat_demand_kw)
        self.heat_wasted_kw = max(0.0, self.total_heat_available_kw - self.heat_utilized_kw)

        # Update heat sink with available external heat
        if steam_gen and hasattr(steam_gen, 'external_heat_input_kw'):
            steam_gen.external_heat_input_kw = self.heat_utilized_kw

        # Update cumulative accumulators
        self.cumulative_heat_recovered_kwh += self.total_heat_available_kw * self.dt
        self.cumulative_heat_utilized_kwh += self.heat_utilized_kw * self.dt
        self.cumulative_heat_wasted_kwh += self.heat_wasted_kw * self.dt

    def get_state(self) -> Dict[str, Any]:
        """
        Retrieve the component's current operational state.

        Fulfills the Component Lifecycle Contract state access.

        Returns:
            Dict[str, Any]: State dictionary containing:
                - total_heat_available_kw (float): Heat supply (kW).
                - total_heat_demand_kw (float): Heat demand (kW).
                - heat_utilized_kw (float): Heat transferred (kW).
                - heat_wasted_kw (float): Unrecovered heat (kW).
                - cumulative_heat_recovered_kwh (float): Total recovered (kWh).
                - cumulative_heat_utilized_kwh (float): Total utilized (kWh).
                - thermal_utilization_efficiency (float): Utilization percentage.
        """
        return {
            **super().get_state(),
            "total_heat_available_kw": self.total_heat_available_kw,
            "total_heat_demand_kw": self.total_heat_demand_kw,
            "heat_utilized_kw": self.heat_utilized_kw,
            "heat_wasted_kw": self.heat_wasted_kw,
            "cumulative_heat_recovered_kwh": self.cumulative_heat_recovered_kwh,
            "cumulative_heat_utilized_kwh": self.cumulative_heat_utilized_kwh,
            "thermal_utilization_efficiency": (
                self.cumulative_heat_utilized_kwh / self.cumulative_heat_recovered_kwh * 100.0
            ) if self.cumulative_heat_recovered_kwh > 0 else 0.0
        }

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """
        Define the physical connection ports for this component.

        Returns:
            Dict[str, Dict[str, str]]: Port definitions.
        """
        return {
            'heat_in': {'type': 'input', 'resource_type': 'heat', 'units': 'kW'},
            'heat_out': {'type': 'output', 'resource_type': 'heat', 'units': 'kW'}
        }
