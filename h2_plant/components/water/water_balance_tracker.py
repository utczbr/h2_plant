"""
Water Balance Tracker Component.

This module implements a plant-wide water balance tracker that monitors
water consumption, recovery, and net demand across all subsystems.
Ensures mass conservation and provides water efficiency metrics.

Water Balance:
    Industrial hydrogen plants consume significant water:
    - PEM electrolysis: ~9 kg H₂O per kg H₂ (stoichiometric)
    - SOEC electrolysis: ~9 kg H₂O per kg H₂ (as steam)
    - ATR reforming: ~3 kg H₂O per kg H₂ (steam-to-carbon ratio)

    Water recovery from separators and condensers reduces net demand.

    **Net Demand = Total Consumption - Total Recovery**

Architecture:
    Implements the Component Lifecycle Contract (Layer 1):
    - `initialize()`: Standard initialization.
    - `step()`: Polls consumers and recoverers, updates balances.
    - `get_state()`: Returns flows, accumulators, and efficiency metrics.
"""

from typing import Dict, Any, List

from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.component_ids import ComponentID


class WaterBalanceTracker(Component):
    """
    Plant-wide water balance tracker for consumption and recovery.

    Monitors water flows from electrolyzers (consumption) and separators
    (recovery) to calculate net water demand and efficiency metrics.

    This component fulfills the Component Lifecycle Contract (Layer 1):
        - `initialize()`: Standard component initialization.
        - `step()`: Polls registered consumers/recoverers, updates balance.
        - `get_state()`: Returns instantaneous flows and cumulative totals.

    Mass Balance:
        net_demand = consumption - recovery
        recovery_ratio = recovery / consumption × 100%

    Attributes:
        total_consumption_kg_h (float): Total water consumption rate (kg/h).
        total_recovery_kg_h (float): Total water recovery rate (kg/h).
        net_demand_kg_h (float): Net freshwater demand (kg/h).
        cumulative_consumption_kg (float): Total consumed water (kg).

    Example:
        >>> tracker = WaterBalanceTracker()
        >>> tracker.initialize(dt=1/60, registry=registry)
        >>> tracker.step(t=0.0)
        >>> efficiency = tracker.get_state()['recovery_ratio']
    """

    def __init__(self):
        """
        Initialize the water balance tracker.
        """
        super().__init__()

        # Instantaneous flows (kg/h)
        self.total_consumption_kg_h = 0.0
        self.total_recovery_kg_h = 0.0
        self.net_demand_kg_h = 0.0
        self.stoichiometric_consumption_kg_h = 0.0  # Actual water consumed by reaction

        # Cumulative tracking (kg)
        self.cumulative_consumption_kg = 0.0
        self.cumulative_recovery_kg = 0.0
        self.cumulative_net_demand_kg = 0.0
        
        # Stoichiometric constant: kg H2O per kg H2 (2H2O -> 2H2 + O2)
        # MW ratio: 2*18.015 / (2*2.016) = 8.936
        self.STOICH_H2O_PER_H2 = 8.936

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

        Polls water consumers (PEM, SOEC) and recovery sources (separators)
        to calculate instantaneous balance and update accumulators.

        Args:
            t (float): Current simulation time in hours.
        """
        super().step(t)

        self.total_consumption_kg_h = 0.0
        self.total_recovery_kg_h = 0.0

        # Track consumption from PEM electrolyzer
        try:
            pem = self.get_registry_safe(ComponentID.PEM_ELECTROLYZER_DETAILED)
            if hasattr(pem, 'water_input_kg_h'):
                self.total_consumption_kg_h += pem.water_input_kg_h
        except Exception:
            pass

        # Track consumption from SOEC cluster
        try:
            soec = self.get_registry_safe(ComponentID.SOEC_CLUSTER)
            if hasattr(soec, 'water_input_kg_h'):
                self.total_consumption_kg_h += soec.water_input_kg_h
        except Exception:
            pass

        # Track recovery from SOEC separator
        try:
            soec = self.get_registry_safe(ComponentID.SOEC_CLUSTER)
            if hasattr(soec, 'separator_sp3'):
                if hasattr(soec.separator_sp3, 'water_return_kg_h'):
                    self.total_recovery_kg_h += soec.separator_sp3.water_return_kg_h
        except Exception:
            pass

        # Calculate stoichiometric water consumed from H2 production
        total_h2_kg_h = 0.0
        try:
            pem = self.get_registry_safe(ComponentID.PEM_ELECTROLYZER_DETAILED)
            if hasattr(pem, 'm_H2_kg_s'):
                total_h2_kg_h += pem.m_H2_kg_s * 3600.0
            elif hasattr(pem, 'last_step_h2_kg'):
                total_h2_kg_h += pem.last_step_h2_kg / self.dt if self.dt > 0 else 0.0
        except Exception:
            pass
        
        try:
            soec = self.get_registry_safe(ComponentID.SOEC_CLUSTER)
            if hasattr(soec, 'last_step_h2_kg'):
                total_h2_kg_h += soec.last_step_h2_kg / self.dt if self.dt > 0 else 0.0
        except Exception:
            pass
        
        self.stoichiometric_consumption_kg_h = total_h2_kg_h * self.STOICH_H2O_PER_H2

        # Calculate net demand
        self.net_demand_kg_h = self.total_consumption_kg_h - self.total_recovery_kg_h

        # Update cumulative accumulators
        self.cumulative_consumption_kg += self.total_consumption_kg_h * self.dt
        self.cumulative_recovery_kg += self.total_recovery_kg_h * self.dt
        self.cumulative_net_demand_kg += self.net_demand_kg_h * self.dt

    def get_state(self) -> Dict[str, Any]:
        """
        Retrieve the component's current operational state.

        Fulfills the Component Lifecycle Contract state access.

        Returns:
            Dict[str, Any]: State dictionary containing:
                - total_consumption_kg_h (float): Consumption rate (kg/h).
                - total_recovery_kg_h (float): Recovery rate (kg/h).
                - net_demand_kg_h (float): Net demand (kg/h).
                - cumulative_consumption_kg (float): Total consumed (kg).
                - cumulative_recovery_kg (float): Total recovered (kg).
                - recovery_ratio (float): Recovery efficiency (%).
        """
        return {
            **super().get_state(),
            "total_consumption_kg_h": self.total_consumption_kg_h,
            "total_recovery_kg_h": self.total_recovery_kg_h,
            "stoichiometric_consumption_kg_h": self.stoichiometric_consumption_kg_h,
            "net_demand_kg_h": self.net_demand_kg_h,
            "cumulative_consumption_kg": self.cumulative_consumption_kg,
            "cumulative_recovery_kg": self.cumulative_recovery_kg,
            "cumulative_net_demand_kg": self.cumulative_net_demand_kg,
            "recovery_ratio": (
                self.cumulative_recovery_kg / self.cumulative_consumption_kg * 100.0
            ) if self.cumulative_consumption_kg > 0 else 0.0
        }

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """
        Define the physical connection ports for this component.

        Returns:
            Dict[str, Dict[str, str]]: Output port for stoichiometric consumption signal.
        """
        return {
            'consumption_out': {'type': 'output', 'resource_type': 'signal', 'units': 'kg/h'}
        }
    
    def get_output(self, port_name: str) -> Any:
        """
        Retrieve output at specified port.
        
        Args:
            port_name: Port identifier ('consumption_out').
            
        Returns:
            float: Stoichiometric water consumption rate (kg/h).
        """
        if port_name == 'consumption_out':
            return self.stoichiometric_consumption_kg_h
        return None
