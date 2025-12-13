"""
Battery Energy Storage System (BESS) Component.

This module implements a battery storage system for grid-connected hydrogen
plants. The battery provides load leveling, renewable energy buffering, and
backup power for electrolyzer operation during grid outages.

Operating Model:
    - **Charging**: When grid power exceeds load, excess energy charges the
      battery up to maximum State of Charge (SoC).
    - **Discharging**: When grid is unavailable or insufficient, battery
      supplies load up to minimum SoC limit.
    - **Round-Trip Efficiency**: Accounts for losses during both charge
      (η_charge ~95%) and discharge (η_discharge ~95%) cycles.

State of Charge Limits:
    - **Min SoC (20%)**: Prevents deep discharge damage and extends cycle life.
    - **Max SoC (95%)**: Prevents overcharge damage and thermal issues.

Architecture:
    Implements the Component Lifecycle Contract (Layer 1):
    - `initialize()`: Standard initialization.
    - `step()`: Manages charge/discharge based on grid availability.
    - `get_state()`: Returns SoC, energy, power, and mode.

Cycle Counting:
    Partial cycles are tracked cumulatively for degradation modeling.
    One full cycle = capacity_kwh charged.
"""

from typing import Dict, Any
from enum import IntEnum
import logging

from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry

logger = logging.getLogger(__name__)


class BatteryMode(IntEnum):
    """
    Battery operating mode enumeration.

    Attributes:
        IDLE: No power transfer.
        CHARGING: Accepting power from grid.
        DISCHARGING: Supplying power to load.
        FULL: At maximum SoC, cannot charge further.
        EMPTY: At minimum SoC, cannot discharge further.
        FAULT: Error condition.
    """
    IDLE = 0
    CHARGING = 1
    DISCHARGING = 2
    FULL = 3
    EMPTY = 4
    FAULT = 5


class BatteryStorage(Component):
    """
    Battery energy storage system for electrolyzer backup and load leveling.

    Manages charge/discharge cycles based on grid availability and load
    demand. Tracks State of Charge, energy stored, and cumulative cycling.

    This component fulfills the Component Lifecycle Contract (Layer 1):
        - `initialize()`: Standard component initialization.
        - `step()`: Evaluates grid status and manages power flow.
        - `get_state()`: Returns capacity, energy, SoC, and operating mode.

    Power flow logic:
    1. If grid available: charge from excess grid power.
    2. If grid unavailable: discharge to meet load demand.
    3. Respect min/max SoC limits at all times.

    Attributes:
        capacity_kwh (float): Total energy capacity (kWh).
        energy_kwh (float): Current stored energy (kWh).
        soc (float): State of Charge (0-1).
        mode (BatteryMode): Current operating mode.

    Example:
        >>> bess = BatteryStorage(capacity_kwh=1000.0, initial_soc=0.50)
        >>> bess.initialize(dt=1/60, registry=registry)
        >>> bess.set_grid_status(available=True, power_kw=500.0)
        >>> bess.step(t=0.0)
        >>> print(f"SoC: {bess.soc*100:.1f}%")
    """

    def __init__(
        self,
        capacity_kwh: float = 1000.0,
        max_charge_power_kw: float = 500.0,
        max_discharge_power_kw: float = 500.0,
        charge_efficiency: float = 0.95,
        discharge_efficiency: float = 0.95,
        min_soc: float = 0.20,
        max_soc: float = 0.95,
        initial_soc: float = 0.50
    ):
        """
        Initialize the battery storage system.

        Args:
            capacity_kwh (float): Total energy capacity in kWh. Default: 1000.0.
            max_charge_power_kw (float): Maximum charging power in kW.
                Default: 500.0.
            max_discharge_power_kw (float): Maximum discharge power in kW.
                Default: 500.0.
            charge_efficiency (float): Charging efficiency (0-1). Energy
                stored = energy input × η. Default: 0.95.
            discharge_efficiency (float): Discharge efficiency (0-1). Energy
                delivered = energy withdrawn × η. Default: 0.95.
            min_soc (float): Minimum State of Charge limit (0-1). Protects
                battery from deep discharge. Default: 0.20.
            max_soc (float): Maximum State of Charge limit (0-1). Prevents
                overcharge. Default: 0.95.
            initial_soc (float): Initial State of Charge (0-1). Default: 0.50.
        """
        super().__init__()

        self.capacity_kwh = capacity_kwh
        self.max_charge_power_kw = max_charge_power_kw
        self.max_discharge_power_kw = max_discharge_power_kw
        self.charge_efficiency = charge_efficiency
        self.discharge_efficiency = discharge_efficiency
        self.min_soc = min_soc
        self.max_soc = max_soc

        # Energy state
        self.energy_kwh = capacity_kwh * initial_soc
        self.soc = initial_soc
        self.mode = BatteryMode.IDLE

        # Power tracking
        self.charge_power_kw = 0.0
        self.discharge_power_kw = 0.0

        # Cumulative metrics
        self.cumulative_charged_kwh = 0.0
        self.cumulative_discharged_kwh = 0.0
        self.charge_cycles = 0.0

        # Grid interface
        self.grid_available = True
        self.grid_power_available_kw = 0.0
        self.load_demand_kw = 0.0

        # Output buffer for flow network
        self._power_output_buffer_kw = 0.0

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

        Manages charge/discharge based on grid availability:
        1. If grid available: charge from excess power.
        2. If grid unavailable: discharge to meet load.
        3. Calculate available discharge power for downstream.
        4. Update SoC and operating mode.

        Args:
            t (float): Current simulation time in hours.
        """
        super().step(t)

        # Reset output buffer
        self._power_output_buffer_kw = 0.0

        if self.grid_available:
            self._charge_from_grid()
        else:
            self._discharge_to_load()

        # Calculate available discharge power
        available_energy = (self.soc - self.min_soc) * self.capacity_kwh
        if available_energy > 0.01:
            max_power = min(
                self.max_discharge_power_kw,
                available_energy / self.dt if self.dt > 0 else 0.0
            ) * self.discharge_efficiency
            self._power_output_buffer_kw = max_power
        else:
            self._power_output_buffer_kw = 0.0

        # Update SoC
        self.soc = self.energy_kwh / self.capacity_kwh if self.capacity_kwh > 0 else 0.0
        self._update_mode()

    def get_state(self) -> Dict[str, Any]:
        """
        Retrieve the component's current operational state.

        Fulfills the Component Lifecycle Contract state access.

        Returns:
            Dict[str, Any]: State dictionary containing:
                - capacity_kwh (float): Total capacity (kWh).
                - energy_kwh (float): Current stored energy (kWh).
                - soc_percentage (float): State of Charge (%).
                - mode (int): Operating mode enumeration value.
                - charge_power_kw (float): Current charge rate (kW).
                - discharge_power_kw (float): Current discharge rate (kW).
        """
        return {
            **super().get_state(),
            'capacity_kwh': float(self.capacity_kwh),
            'energy_kwh': float(self.energy_kwh),
            'soc_percentage': float(self.soc * 100),
            'mode': int(self.mode),
            'charge_power_kw': float(self.charge_power_kw),
            'discharge_power_kw': float(self.discharge_power_kw)
        }

    def _charge_from_grid(self) -> None:
        """
        Charge battery from available grid power.

        Calculates charging energy considering:
        - Available capacity to max SoC
        - Maximum charge power rating
        - Available grid power after load subtraction
        """
        available_capacity_kwh = (self.max_soc - self.soc) * self.capacity_kwh
        if available_capacity_kwh > 0.01:
            max_charge_energy = self.max_charge_power_kw * self.dt
            available_grid_power = self.grid_power_available_kw - self.load_demand_kw
            available_grid_energy = max(0, available_grid_power * self.dt)
            charge_energy = min(available_capacity_kwh, max_charge_energy, available_grid_energy)

            if charge_energy > 0:
                energy_stored = charge_energy * self.charge_efficiency
                self.energy_kwh += energy_stored
                self.charge_power_kw = charge_energy / self.dt if self.dt > 0 else 0.0
                self.cumulative_charged_kwh += charge_energy
                self.charge_cycles += charge_energy / self.capacity_kwh if self.capacity_kwh > 0 else 0.0
                self.mode = BatteryMode.CHARGING
            else:
                self.charge_power_kw = 0.0
                self.mode = BatteryMode.IDLE
        else:
            self.charge_power_kw = 0.0
            self.mode = BatteryMode.FULL

    def _discharge_to_load(self) -> None:
        """
        Discharge battery to meet load demand.

        Calculates discharge energy considering:
        - Available energy above min SoC
        - Maximum discharge power rating
        - Required energy for load demand
        """
        available_discharge_kwh = (self.soc - self.min_soc) * self.capacity_kwh
        if available_discharge_kwh > 0.01:
            required_energy = self.load_demand_kw * self.dt
            max_discharge_energy = self.max_discharge_power_kw * self.dt
            discharge_energy = min(
                available_discharge_kwh,
                max_discharge_energy,
                required_energy / self.discharge_efficiency if self.discharge_efficiency > 0 else float('inf')
            )

            if discharge_energy > 0:
                energy_delivered = discharge_energy * self.discharge_efficiency
                self.energy_kwh -= discharge_energy
                self.discharge_power_kw = energy_delivered / self.dt if self.dt > 0 else 0.0
                self.cumulative_discharged_kwh += discharge_energy
                self.mode = BatteryMode.DISCHARGING
            else:
                self.discharge_power_kw = 0.0
                self.mode = BatteryMode.IDLE
        else:
            self.discharge_power_kw = 0.0
            self.mode = BatteryMode.EMPTY
            logger.warning("Battery depleted - cannot supply load")

    def _update_mode(self) -> None:
        """
        Update operating mode based on current SoC.
        """
        if self.soc >= self.max_soc and self.mode != BatteryMode.DISCHARGING:
            self.mode = BatteryMode.FULL
        elif self.soc <= self.min_soc and self.mode != BatteryMode.CHARGING:
            self.mode = BatteryMode.EMPTY

    def set_grid_status(self, available: bool, power_kw: float = 0.0) -> None:
        """
        Set grid availability status.

        Args:
            available (bool): Whether grid power is available.
            power_kw (float): Available grid power in kW. Default: 0.0.
        """
        self.grid_available = available
        self.grid_power_available_kw = power_kw

    def set_load_demand(self, demand_kw: float) -> None:
        """
        Set current load power demand.

        Args:
            demand_kw (float): Load demand in kW.
        """
        self.load_demand_kw = demand_kw

    def get_available_power(self) -> float:
        """
        Get currently available power from battery or grid.

        Returns:
            float: Available power in kW.
        """
        if self.grid_available:
            return self.grid_power_available_kw
        else:
            available_energy = (self.soc - self.min_soc) * self.capacity_kwh
            max_power = min(
                self.max_discharge_power_kw,
                available_energy / self.dt if self.dt > 0 else 0.0
            ) * self.discharge_efficiency
            return max_power

    def get_output(self, port_name: str) -> Any:
        """
        Retrieve output from specified port.

        Args:
            port_name (str): Port identifier ('electricity_out').

        Returns:
            float: Available discharge power (kW).

        Raises:
            ValueError: If port_name is not a valid output port.
        """
        if port_name == 'electricity_out':
            return self._power_output_buffer_kw
        else:
            raise ValueError(f"Unknown output port '{port_name}' on BatteryStorage")

    def receive_input(self, port_name: str, value: Any, resource_type: str) -> float:
        """
        Accept input at specified port.

        Args:
            port_name (str): Target port ('electricity_in').
            value (Any): Power value in kW.
            resource_type (str): Resource classification hint.

        Returns:
            float: Power accepted for charging (kW).
        """
        if port_name == 'electricity_in':
            if isinstance(value, (int, float)):
                self.grid_power_available_kw = value
                self.grid_available = True
                available_capacity_kwh = (self.max_soc - self.soc) * self.capacity_kwh
                if available_capacity_kwh > 0.01:
                    max_charge_energy = self.max_charge_power_kw * self.dt
                    charge_energy = min(available_capacity_kwh, max_charge_energy, value * self.dt)
                    return charge_energy / self.dt if self.dt > 0 else 0.0
        return 0.0

    def extract_output(self, port_name: str, amount: float, resource_type: str) -> None:
        """
        Acknowledge extraction and deduct energy from battery.

        Args:
            port_name (str): Port from which output was extracted.
            amount (float): Power extracted (kW).
            resource_type (str): Resource classification hint.
        """
        if port_name == 'electricity_out':
            self._power_output_buffer_kw = 0.0
            energy_extracted = amount * self.dt
            internal_energy = energy_extracted / self.discharge_efficiency if self.discharge_efficiency > 0 else 0.0
            self.energy_kwh = max(0.0, self.energy_kwh - internal_energy)
            self.cumulative_discharged_kwh += internal_energy
            self.soc = self.energy_kwh / self.capacity_kwh if self.capacity_kwh > 0 else 0.0

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """
        Define the physical connection ports for this component.

        Returns:
            Dict[str, Dict[str, str]]: Port definitions.
        """
        return {
            'electricity_in': {'type': 'input', 'resource_type': 'electricity', 'units': 'kW'},
            'electricity_out': {'type': 'output', 'resource_type': 'electricity', 'units': 'kW'}
        }
