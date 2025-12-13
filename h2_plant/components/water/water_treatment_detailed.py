"""
Detailed Water Treatment System.

This module implements a composite water treatment system combining a
water purifier and storage tank. Models the complete water supply chain
from external feed to electrolyzer demand.

System Components:
    - **Water Purifier (WP)**: Produces ultrapure water from external feed.
    - **Water Tank (WT)**: Buffer storage for demand smoothing.

Mass Balance:
    Tank accumulation:
    **dm/dt = ṁ_in - ṁ_out**

    Where ṁ_in is purifier output and ṁ_out is total electrolyzer demand.

Architecture:
    Implements the Component Lifecycle Contract (Layer 1) via CompositeComponent:
    - `initialize()`: Initializes all subsystems.
    - `step()`: Coordinates purifier → tank → demand flow.
    - `get_state()`: Returns aggregated summary and subsystem states.
"""

from typing import Dict, Any

from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.composite_component import CompositeComponent


class WaterPurifier(Component):
    """
    Simplified water purifier for composite system.

    Produces ultrapure water from external feed with efficiency losses.

    Attributes:
        max_flow_kg_h (float): Maximum processing capacity (kg/h).
        efficiency (float): Mass recovery efficiency (0-1).
        output_flow_kg_h (float): Current output rate (kg/h).
        power_kw (float): Electrical power consumption (kW).
    """

    def __init__(self, purifier_id: str, max_flow_kg_h: float, efficiency: float = 0.95):
        """
        Initialize the water purifier.

        Args:
            purifier_id (str): Unique identifier.
            max_flow_kg_h (float): Maximum capacity in kg/h.
            efficiency (float): Mass recovery (0-1). Default: 0.95.
        """
        super().__init__()
        self.purifier_id = purifier_id
        self.max_flow_kg_h = max_flow_kg_h
        self.efficiency = efficiency
        self.input_flow_kg_h = 0.0
        self.output_flow_kg_h = 0.0
        self.power_kw = 0.0

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """
        Prepare the component for simulation execution.

        Args:
            dt (float): Simulation timestep in hours.
            registry (ComponentRegistry): Central registry for component access.
        """
        super().initialize(dt, registry)

    def step(self, t: float) -> None:
        """
        Execute one simulation timestep.

        Limits input to capacity and applies efficiency factor.
        Power consumption: 0.5 kWh/m³ ≈ 0.0005 kWh/kg.

        Args:
            t (float): Current simulation time in hours.
        """
        super().step(t)

        processed_flow = min(self.input_flow_kg_h, self.max_flow_kg_h)
        self.output_flow_kg_h = processed_flow * self.efficiency

        # Simplified power: 0.5 kWh/m³
        self.power_kw = (processed_flow * 0.0005) / self.dt if self.dt > 0 else 0.0

    def get_state(self) -> Dict[str, Any]:
        """
        Retrieve the component's current operational state.

        Returns:
            Dict[str, Any]: Output flow and power consumption.
        """
        return {
            **super().get_state(),
            'output_flow_kg_h': float(self.output_flow_kg_h),
            'power_kw': float(self.power_kw)
        }


class WaterTank(Component):
    """
    Atmospheric water storage tank.

    Simple mass-balance tank with capacity limits.

    Attributes:
        capacity_kg (float): Maximum storage capacity (kg).
        current_mass_kg (float): Current inventory (kg).
    """

    def __init__(self, tank_id: str, capacity_kg: float):
        """
        Initialize the water tank.

        Args:
            tank_id (str): Unique identifier.
            capacity_kg (float): Tank capacity in kg.
        """
        super().__init__()
        self.tank_id = tank_id
        self.capacity_kg = capacity_kg
        self.current_mass_kg = 0.0
        self.inlet_flow_kg_h = 0.0
        self.outlet_flow_kg_h = 0.0

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """
        Prepare the component for simulation execution.

        Starts tank at 50% fill.

        Args:
            dt (float): Simulation timestep in hours.
            registry (ComponentRegistry): Central registry for component access.
        """
        super().initialize(dt, registry)
        self.current_mass_kg = self.capacity_kg * 0.5

    def step(self, t: float) -> None:
        """
        Execute one simulation timestep.

        Updates mass based on inlet/outlet flows with capacity clamping.

        Args:
            t (float): Current simulation time in hours.
        """
        super().step(t)

        delta_mass = (self.inlet_flow_kg_h - self.outlet_flow_kg_h) * self.dt
        self.current_mass_kg += delta_mass

        # Clamp to bounds
        if self.current_mass_kg > self.capacity_kg:
            self.current_mass_kg = self.capacity_kg
        elif self.current_mass_kg < 0:
            self.current_mass_kg = 0.0

    def get_state(self) -> Dict[str, Any]:
        """
        Retrieve the component's current operational state.

        Returns:
            Dict[str, Any]: Current mass and fill level.
        """
        return {
            **super().get_state(),
            'current_mass_kg': float(self.current_mass_kg),
            'fill_level': float(self.current_mass_kg / self.capacity_kg)
        }


class DetailedWaterTreatment(CompositeComponent):
    """
    Complete water treatment system with purifier and storage.

    Coordinates purifier output to tank, then allocates to electrolyzer
    demands (PEM, SOEC, ATR).

    This component fulfills the Component Lifecycle Contract (Layer 1):
        - `initialize()`: Initializes purifier and tank subsystems.
        - `step()`: Coordinates flow from purifier → tank → demands.
        - `get_state()`: Returns summary with stored water and demand.

    Attributes:
        external_water_input_kg_h (float): External feed rate (kg/h).
        demand_pem_kg_h (float): PEM electrolyzer demand (kg/h).
        demand_soec_kg_h (float): SOEC electrolyzer demand (kg/h).
        demand_atr_kg_h (float): ATR reformer demand (kg/h).

    Example:
        >>> treatment = DetailedWaterTreatment(max_flow_kg_h=2000.0, tank_capacity_kg=10000.0)
        >>> treatment.initialize(dt=1/60, registry=registry)
        >>> treatment.external_water_input_kg_h = 1500.0
        >>> treatment.demand_pem_kg_h = 500.0
        >>> treatment.step(t=0.0)
    """

    def __init__(self, max_flow_kg_h: float = 2000.0, tank_capacity_kg: float = 10000.0):
        """
        Initialize the detailed water treatment system.

        Args:
            max_flow_kg_h (float): Purifier capacity in kg/h. Default: 2000.0.
            tank_capacity_kg (float): Tank capacity in kg. Default: 10000.0.
        """
        super().__init__()

        self.add_subsystem('purifier_wp', WaterPurifier('WP', max_flow_kg_h))
        self.add_subsystem('tank_wt', WaterTank('WT', tank_capacity_kg))

        # External inputs
        self.external_water_input_kg_h = 0.0
        self.demand_pem_kg_h = 0.0
        self.demand_soec_kg_h = 0.0
        self.demand_atr_kg_h = 0.0

    def step(self, t: float) -> None:
        """
        Execute one simulation timestep.

        Coordinates flow through subsystems:
        1. Feed external water to purifier.
        2. Route purifier output to tank.
        3. Withdraw from tank to meet electrolyzer demands.

        Args:
            t (float): Current simulation time in hours.
        """
        Component.step(self, t)

        # Purification stage
        self.purifier_wp.input_flow_kg_h = self.external_water_input_kg_h
        self.purifier_wp.step(t)

        # Storage stage
        total_demand = self.demand_pem_kg_h + self.demand_soec_kg_h + self.demand_atr_kg_h

        self.tank_wt.inlet_flow_kg_h = self.purifier_wp.output_flow_kg_h
        self.tank_wt.outlet_flow_kg_h = total_demand
        self.tank_wt.step(t)

    def get_state(self) -> Dict[str, Any]:
        """
        Retrieve the component's current operational state.

        Returns:
            Dict[str, Any]: Aggregated state with summary metrics.
        """
        state = super().get_state()
        state['summary'] = {
            'stored_water_kg': self.tank_wt.current_mass_kg,
            'total_demand_kg_h': self.tank_wt.outlet_flow_kg_h
        }
        return state
