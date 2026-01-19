"""
Power Transformer / Rectifier Component.

This module implements a generic efficiency-based power conversion unit.
It abstracts AC/DC conversion physics into a throughput-efficiency model
suitable for techno-economic analysis and thermal load tracking.

Functionality:
    - Scales input power by efficiency factor (P_out = P_in * eta).
    - Calculates thermal losses for cooling system sizing (Q = P_in - P_out).
    - Caps throughput at rated capacity.

Process Integration:
    - Upstream: Grid connection or Power Distribution Unit (PDU).
    - Downstream: Electrolyzer stacks (SOEC/PEM), Compressors, or BOP.
"""

from typing import Dict, Any, Optional

from h2_plant.core.component import Component


class PowerTransformer(Component):
    """
    Efficiency-based power transformer/rectifier.

    Represents the power conditioning stage (Transformer + Rectifier) for
    major plant loads. It applies a fixed efficiency loss to the energy flow,
    generating waste heat that must be managed by the cooling system.

    Attributes:
        component_id (str): Unique identifier.
        efficiency (float): Combined transformer/rectifier efficiency (0.0-1.0).
        rated_power_mw (float): Maximum output power rating (MW).
        system_group (str, optional): Tag for grouping (e.g., 'SOEC', 'PEM').

    Ports:
        - power_in (input): Electricity from grid/source.
        - power_out (output): Conditioned power to load.
        - heat_out (output): Waste heat requiring cooling.
    """

    def __init__(
        self,
        component_id: str = "transformer",
        efficiency: float = 0.95,
        rated_power_mw: float = 20.0,
        system_group: Optional[str] = None
    ):
        """
        Initialize the power transformer.

        Args:
            component_id (str): Component ID.
            efficiency (float): Energy conversion efficiency (0 < eta <= 1).
                                Default 0.95 (includes trafo + rectifier losses).
            rated_power_mw (float): Maximum permitted OUTPUT power (MW).
            system_group (str, optional): Subsystem grouping tag (e.g., 'SOEC').
        """
        super().__init__()
        self.component_id = component_id
        self.efficiency = max(0.01, min(1.0, efficiency))  # Clamp for safety
        self.rated_power_mw = rated_power_mw
        self.system_group = system_group

        # Runtime State
        self.power_in_mw: float = 0.0
        self.power_out_mw: float = 0.0
        self.power_loss_mw: float = 0.0
        self.load_factor: float = 0.0

    def initialize(self, dt: float, registry: Any) -> None:
        """
        Prepare component for simulation.

        Args:
            dt (float): Timestep (h).
            registry (ComponentRegistry): Component registry.
        """
        super().initialize(dt, registry)
        self.initialized = True

    def step(self, t: float) -> None:
        """
        Execute conversion physics.

        Energy Balance: P_in = P_out + P_loss
        - P_out = min(P_in * efficiency, rated_power)
        - P_loss = P_in - P_out
        """
        super().step(t)

        if self.power_in_mw > 0:
            # Calculate theoretical output
            potential_out = self.power_in_mw * self.efficiency

            # Apply capacity constraint
            self.power_out_mw = min(potential_out, self.rated_power_mw)

            # Energy Balance: In = Out + Loss
            self.power_loss_mw = self.power_in_mw - self.power_out_mw

            self.load_factor = self.power_out_mw / self.rated_power_mw
        else:
            self.power_out_mw = 0.0
            self.power_loss_mw = 0.0
            self.load_factor = 0.0

    def receive_input(self, port_name: str, value: Any, resource_type: str = None) -> float:
        """
        Receive power from upstream (Grid/Dispatch).

        Args:
            port_name (str): 'power_in' or 'electricity_in'.
            value (float): Power in MW.

        Returns:
            float: Accepted power (MW).
        """
        if port_name in ["power_in", "electricity_in"]:
            if isinstance(value, (int, float)):
                self.power_in_mw = float(value)
                return self.power_in_mw
        return 0.0

    def get_output(self, port_name: str) -> Any:
        """
        Provide power or heat to downstream components.

        Args:
            port_name (str): 'power_out', 'dc_out' or 'heat_out'.

        Returns:
            float: Power (MW) or Heat (kW).
        """
        if port_name in ["power_out", "dc_out", "electricity_out"]:
            return self.power_out_mw
        elif port_name == "heat_out":
            # Convert MW to kW for thermal consistency with other components
            return self.power_loss_mw * 1000.0
        return 0.0

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """Define component interfaces."""
        return {
            'power_in': {'type': 'input', 'resource_type': 'electricity'},
            'power_out': {'type': 'output', 'resource_type': 'electricity'},
            'heat_out': {'type': 'output', 'resource_type': 'heat'}
        }

    def get_state(self) -> Dict[str, Any]:
        """Return operational state snapshot."""
        return {
            **super().get_state(),
            'component_id': self.component_id,
            'system_group': self.system_group,
            'efficiency': self.efficiency,
            'power_in_mw': self.power_in_mw,
            'power_out_mw': self.power_out_mw,
            'power_loss_mw': self.power_loss_mw,
            'heat_loss_kw': self.power_loss_mw * 1000.0,
            'load_factor': self.load_factor
        }


# Backwards compatibility alias
Rectifier = PowerTransformer
