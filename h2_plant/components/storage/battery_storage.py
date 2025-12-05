"""
Battery energy storage system (BESS) component.
"""

from typing import Dict, Any, Optional
from enum import IntEnum
import logging
from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry

logger = logging.getLogger(__name__)


class BatteryMode(IntEnum):
    """Battery operating modes."""
    IDLE = 0
    CHARGING = 1
    DISCHARGING = 2
    FULL = 3
    EMPTY = 4
    FAULT = 5


class BatteryStorage(Component):
    """
    Battery energy storage system for electrolyzer backup.
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
        super().__init__()
        
        self.capacity_kwh = capacity_kwh
        self.max_charge_power_kw = max_charge_power_kw
        self.max_discharge_power_kw = max_discharge_power_kw
        self.charge_efficiency = charge_efficiency
        self.discharge_efficiency = discharge_efficiency
        self.min_soc = min_soc
        self.max_soc = max_soc
        
        self.energy_kwh = capacity_kwh * initial_soc
        self.soc = initial_soc
        self.mode = BatteryMode.IDLE
        
        self.charge_power_kw = 0.0
        self.discharge_power_kw = 0.0
        
        self.cumulative_charged_kwh = 0.0
        self.cumulative_discharged_kwh = 0.0
        self.charge_cycles = 0.0
        
        self.grid_available = True
        self.grid_power_available_kw = 0.0
        self.load_demand_kw = 0.0
        
        # Output buffer for flow network
        self._power_output_buffer_kw = 0.0
    
    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """Initialize battery system."""
        super().initialize(dt, registry)
    
    def step(self, t: float) -> None:
        """Execute timestep - manage charging/discharging."""
        super().step(t)
        
        # Reset output buffer at start of step
        self._power_output_buffer_kw = 0.0
        
        if self.grid_available:
            self._charge_from_grid()
        else:
            self._discharge_to_load()
        
        # Calculate available discharge power and store in buffer
        # Do this regardless of mode - battery can discharge even while charging stops
        available_energy = (self.soc - self.min_soc) * self.capacity_kwh
        if available_energy > 0.01:
            max_power = min(
                self.max_discharge_power_kw,
                available_energy / self.dt if self.dt > 0 else 0.0
            ) * self.discharge_efficiency
            self._power_output_buffer_kw = max_power
        else:
            self._power_output_buffer_kw = 0.0
        
        self.soc = self.energy_kwh / self.capacity_kwh if self.capacity_kwh > 0 else 0.0
        self._update_mode()
    
    def get_state(self) -> Dict[str, Any]:
        """Return battery state."""
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
        available_discharge_kwh = (self.soc - self.min_soc) * self.capacity_kwh
        if available_discharge_kwh > 0.01:
            required_energy = self.load_demand_kw * self.dt
            max_discharge_energy = self.max_discharge_power_kw * self.dt
            discharge_energy = min(available_discharge_kwh, max_discharge_energy, required_energy / self.discharge_efficiency if self.discharge_efficiency > 0 else float('inf'))
            
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
        if self.soc >= self.max_soc and self.mode != BatteryMode.DISCHARGING:
            self.mode = BatteryMode.FULL
        elif self.soc <= self.min_soc and self.mode != BatteryMode.CHARGING:
            self.mode = BatteryMode.EMPTY
    
    def set_grid_status(self, available: bool, power_kw: float = 0.0) -> None:
        self.grid_available = available
        self.grid_power_available_kw = power_kw
    
    def set_load_demand(self, demand_kw: float) -> None:
        self.load_demand_kw = demand_kw
    
    def get_available_power(self) -> float:
        if self.grid_available:
            return self.grid_power_available_kw
        else:
            available_energy = (self.soc - self.min_soc) * self.capacity_kwh
            max_power = min(self.max_discharge_power_kw, available_energy / self.dt if self.dt > 0 else 0.0) * self.discharge_efficiency
            return max_power
    
    def get_output(self, port_name: str) -> Any:
        """Get output from specific port."""
        if port_name == 'electricity_out':
            # Return available discharge power from buffer
            return self._power_output_buffer_kw
        else:
            raise ValueError(f"Unknown output port '{port_name}' on BatteryStorage")
    
    def receive_input(self, port_name: str, value: Any, resource_type: str) -> float:
        """Receive input."""
        if port_name == 'electricity_in':
            if isinstance(value, (int, float)):
                # Accept power for charging
                self.grid_power_available_kw = value
                self.grid_available = True
                # Calculate how much we can actually accept
                available_capacity_kwh = (self.max_soc - self.soc) * self.capacity_kwh
                if available_capacity_kwh > 0.01:
                    max_charge_energy = self.max_charge_power_kw * self.dt
                    charge_energy = min(available_capacity_kwh, max_charge_energy, value * self.dt)
                    return charge_energy / self.dt if self.dt > 0 else 0.0
        return 0.0
    
    def extract_output(self, port_name: str, amount: float, resource_type: str) -> None:
        """Deduct extracted amount from battery energy."""
        if port_name == 'electricity_out':
            # Clear output buffer
            self._power_output_buffer_kw = 0.0
            # Deduct energy based on amount extracted (power × time)
            energy_extracted = amount * self.dt
            # Account for discharge efficiency
            internal_energy = energy_extracted / self.discharge_efficiency if self.discharge_efficiency > 0 else 0.0
            self.energy_kwh = max(0.0, self.energy_kwh - internal_energy)
            self.cumulative_discharged_kwh += internal_energy
            self.soc = self.energy_kwh / self.capacity_kwh if self.capacity_kwh > 0 else 0.0
    
    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """Return port metadata."""
        return {
            'electricity_in': {'type': 'input', 'resource_type': 'electricity', 'units': 'kW'},
            'electricity_out': {'type': 'output', 'resource_type': 'electricity', 'units': 'kW'}
        }
