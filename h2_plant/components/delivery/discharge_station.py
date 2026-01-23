"""
DischargeStation - Event-Driven Hydrogen Dispensing with Real Compression Energy.

This module implements a truck loading station with:
- Discrete truck arrival/departure events
- Dual-dock operation with cooldown timers
- Isentropic compression work calculation using Real Gas enthalpy
- Demand Signal output for upstream tank communication

Multi-Station Support:
    This component can simulate N identical stations (e.g. 10 Truck Bays)
    aggregated into a single flow component.
    - Input flow is distributed across active stations.
    - Demand signals are summed.
    - Energy consumption is summed.

Physics Model:
    Compression work is calculated using isentropic efficiency:
    
    w_stage = (h_out,s - h_in) / η_isen
    P_electric = (ṁ × Σw_stages) / η_mech
    
    Multi-stage compression is used when pressure ratio > 2.5.

Architecture:
    This component implements the "Demand Signal Propagation Pattern":
    - Calculates demand based on truck status
    - Outputs `demand_signal` for upstream DetailedTankArray
    - Receives physical `h2_in` from tank (fulfilled demand)
    - One-timestep delay ensures causal stability

Reference:
    Logic adapted from H2DemandwithNightShift.py and Outgoingcompressor.py
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import numpy as np
import random
import logging

from h2_plant.core.component import Component
from h2_plant.core.stream import Stream
from h2_plant.core.constants import GasConstants

logger = logging.getLogger(__name__)


@dataclass
class DockState:
    """
    State container for a single truck dock.
    """
    cooldown_min: float = 0.0
    is_active: bool = False
    truck_mass_kg: float = 0.0


class StationUnit:
    """
    Encapsulates the logic for a SINGLE physical station (with Dual Docks A/B).
    
    Managed by the main DischargeStation component.
    """
    def __init__(self, unit_id: int, config: dict):
        self.unit_id = unit_id
        self.config = config
        
        # Unpack config
        self.truck_capacity = config.get('truck_capacity_kg', 1000.0)
        self.cooldown_time = config.get('cooldown_minutes', 150.0)
        self.arrival_prob = config.get('arrival_probability', 0.3)
        self.max_fill_rate = config.get('max_fill_rate_kg_min', 60.0)
        
        # State
        self.dock_a = DockState()
        self.dock_b = DockState()
        self.active_dock_id = 'A'
    
    def get_current_dock(self) -> DockState:
        return self.dock_a if self.active_dock_id == 'A' else self.dock_b
    
    def get_next_dock(self) -> DockState:
        return self.dock_a if self.active_dock_id == 'B' else self.dock_b
        
    def update_cooldowns(self, dt_min: float):
        if self.dock_a.cooldown_min > 0:
            self.dock_a.cooldown_min = max(0, self.dock_a.cooldown_min - dt_min)
        if self.dock_b.cooldown_min > 0:
            self.dock_b.cooldown_min = max(0, self.dock_b.cooldown_min - dt_min)

    def accepts_hydrogen(self) -> bool:
        return self.get_current_dock().is_active

    def fill_truck(self, mass_kg: float):
        dock = self.get_current_dock()
        if dock.is_active:
            dock.truck_mass_kg += mass_kg
            
    def check_departure(self) -> bool:
        """Returns True if a truck just departed."""
        dock = self.get_current_dock()
        if dock.is_active and dock.truck_mass_kg >= self.truck_capacity:
            # Truck leaves
            dock.is_active = False
            dock.cooldown_min = self.cooldown_time
            dock.truck_mass_kg = 0.0
            
            # Switch Docks
            self.active_dock_id = 'B' if self.active_dock_id == 'A' else 'A'
            return True
        return False

    def check_arrival(self) -> bool:
        """Returns True if a truck just arrived."""
        dock = self.get_current_dock()
        # Arrival possible if dock is IDLE and COOLED
        if not dock.is_active and dock.cooldown_min <= 0:
            if random.random() < self.arrival_prob:
                dock.is_active = True
                dock.truck_mass_kg = 0.0
                return True
        return False
        
    def calculate_demand(self, dt_min: float) -> float:
        """Calculate hydrogen demand [kg] for this step."""
        # Demand comes from the dock that will be active next
        # (Usually same as current, unless we switched this step)
        target_dock = self.get_current_dock()
        
        if target_dock.is_active:
            needed = self.truck_capacity - target_dock.truck_mass_kg
            max_step = self.max_fill_rate * dt_min
            return min(needed, max_step)
        
        return 0.0


class DischargeStation(Component):
    """
    Hydrogen discharge station array.
    
    Simulates N identical stations, each with dual-dock logistics.
    
    Args:
        n_stations: Number of identical stations to simulate (default 1)
        truck_capacity_kg: Hydrogen capacity per truck [kg]
        delivery_pressure_bar: Target truck pressure [bar]
        max_fill_rate_kg_min: Maximum filling rate [kg/min]
        isen_efficiency: Isentropic efficiency [0-1]
        mech_efficiency: Mechanical efficiency [0-1]
        cooldown_minutes: Dock cooldown after truck departure [min]
        arrival_probability: Per-step probability of truck arrival [0-1]
    """
    
    def __init__(
        self,
        n_stations: int = 1,
        station_id: int = 1, # Legacy ID param
        truck_capacity_kg: float = 1000.0,
        delivery_pressure_bar: float = 500.0,
        max_fill_rate_kg_min: float = 60.0,
        min_fill_rate_kg_min: float = 10.0,  # New param for schedule
        h_in_day_max: Optional[float] = None, # New param for schedule duration
        isen_efficiency: float = 0.75,
        mech_efficiency: float = 0.95,
        cooldown_minutes: float = 150.0,
        arrival_probability: float = 0.3,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.n_stations = int(n_stations)
        self.station_id = station_id
        
        self.delivery_pressure = delivery_pressure_bar * 1e5
        self.isen_eff = isen_efficiency
        self.mech_eff = mech_efficiency
        
        # Schedule parameters
        self.max_fill_rate = max_fill_rate_kg_min
        self.min_fill_rate = min_fill_rate_kg_min
        self.h_in_day_max = h_in_day_max
        
        # Configuration for sub-units (Stochastic Mode)
        unit_config = {
            'truck_capacity_kg': truck_capacity_kg,
            'max_fill_rate_kg_min': max_fill_rate_kg_min,
            'cooldown_minutes': cooldown_minutes,
            'arrival_probability': arrival_probability
        }
        
        # Initialize Station Units
        self.stations: List[StationUnit] = []
        for i in range(self.n_stations):
            self.stations.append(StationUnit(i, unit_config))
        
        # Input/Output Buffers
        self._h2_in_stream: Optional[Stream] = None
        self._inlet_pressure_pa: float = 101325.0
        self._demand_signal: float = 0.0
        self._h2_out_stream: Optional[Stream] = None
        
        # Statistics
        self.power_consumption_kw: float = 0.0
        self.energy_consumed_mj: float = 0.0
        self.trucks_filled_total: int = 0
        self.total_h2_delivered_kg: float = 0.0
        
        # Use truck capacity for stats estimation in schedule mode
        self.truck_capacity = truck_capacity_kg
        
        self._lut_manager = None
    
    def initialize(self, dt: float, registry) -> None:
        """Initialize the multi-station component."""
        super().initialize(dt, registry)
        self._lut_manager = registry.get("lut_manager") if registry.has("lut_manager") else None
        
        mode = "Scheduled" if self.h_in_day_max is not None else "Stochastic"
        logger.info(
            f"DischargeStation initialized ({mode}): {self.n_stations} units. "
            f"Delivery P={self.delivery_pressure/1e5:.0f} bar"
        )
    
    def receive_input(self, port_name: str, value: Any, **kwargs) -> float:
        """Receive input streams."""
        if isinstance(value, Stream):
             stream = value
        else:
             logger.warning(f"DischargeStation: Port '{port_name}' received non-Stream. Ignoring.")
             return 0.0

        if port_name == 'h2_in':
            self._h2_in_stream = stream
            if stream and stream.pressure_pa is not None:
                self._inlet_pressure_pa = float(stream.pressure_pa)
            else:
                pass # Keep default
            
            return float(stream.mass_flow_kg_h) if stream and stream.mass_flow_kg_h is not None else 0.0
            
        return 0.0
    
    def step(self, t: float) -> None:
        """Execute simulation step for all stations."""
        super().step(t)
        dt_min = self.dt * 60.0
        
        # 1. Distribute Incoming Hydrogen
        total_received_mass = 0.0
        if self._h2_in_stream and self._h2_in_stream.mass_flow_kg_h > 0:
            total_received_mass = self._h2_in_stream.mass_flow_kg_h * self.dt
        
        # 2. Update Demand & State (Scheduled vs Stochastic)
        total_demand_kg = 0.0
        step_trucks_filled = 0
        
        if self.h_in_day_max is not None:
            # --- DETERMINISTIC SCHEDULE MODE ---
            # Operates as a continuous sink with variable rate
            minute_of_day = (t * 60.0) % 1440.0
            day_limit_min = self.h_in_day_max * 60.0
            
            if minute_of_day < day_limit_min:
                current_rate = self.max_fill_rate
            else:
                current_rate = self.min_fill_rate
            
            # Demand is continuous: Rate * dt * N_stations
            total_demand_kg = current_rate * self.n_stations * dt_min
            
            # In schedule mode, we assume stations are always "Ready" to accept flow
            # We don't track individual docks, but we estimate trucks filled for stats
            if total_received_mass > 0:
                # Approximate fractional truck filling
                # Or just count "completed" trucks based on mass accumulation?
                # For now, simplistic:
                pass
            
            # Update stats metric (integer trucks) based on total delivered accumulation
            # We can compare previous total // capacity vs current total // capacity
            prev_trucks = int((self.total_h2_delivered_kg) / self.truck_capacity)
            new_trucks = int((self.total_h2_delivered_kg + total_received_mass) / self.truck_capacity)
            step_trucks_filled = new_trucks - prev_trucks

        else:
            # --- STOCHASTIC DOCK MODE ---
            # Existing logic with StationUnits
            
            # Distribute mass to accepting units
            accepting_units = [s for s in self.stations if s.accepts_hydrogen()]
            if accepting_units and total_received_mass > 0:
                mass_per_unit = total_received_mass / len(accepting_units)
                for unit in accepting_units:
                    unit.fill_truck(mass_per_unit)
            
            # Update units
            for unit in self.stations:
                unit.update_cooldowns(dt_min)
                
                if unit.check_departure():
                    step_trucks_filled += 1
                    logger.debug(f"Station Unit {unit.unit_id}: Truck departed")
                
                if unit.check_arrival():
                        logger.debug(f"Station Unit {unit.unit_id}: Truck arrived")
                
                total_demand_kg += unit.calculate_demand(dt_min)
        
        # Common Updates
        self.total_h2_delivered_kg += total_received_mass
        self.trucks_filled_total += step_trucks_filled
        
        # 3. Calculate Compression Energy (Aggregated)
        # We assume one central compressor system or N identical ones.
        # Work depends on total mass compressed.
        if total_received_mass > 0 and self._inlet_pressure_pa > 101325.0:
            energy_mj = self._calculate_compression_work(
                self._inlet_pressure_pa,
                total_received_mass
            )
            self.energy_consumed_mj += energy_mj
            self.power_consumption_kw = (energy_mj * 1e6) / (self.dt * 3600) / 1000
        else:
            self.power_consumption_kw = 0.0
            
        # 4. Set Output Buffers
        self._demand_signal = total_demand_kg / self.dt if self.dt > 0 else 0.0
     
        self._h2_out_stream = Stream(
            mass_flow_kg_h=total_received_mass / self.dt if self.dt > 0 else 0.0,
            temperature_k=293.15,
            pressure_pa=self.delivery_pressure,
            composition={'H2': 1.0},
            phase='gas'
        )
        
        self._h2_in_stream = None


    def get_output(self, port_name: str = 'demand_signal') -> Optional[Stream]:
        if port_name == 'demand_signal':
            return Stream(
                mass_flow_kg_h=self._demand_signal,
                temperature_k=0.0,
                pressure_pa=0.0,
                composition={},
                phase='signal'
            )
        elif port_name == 'h2_out':
            return self._h2_out_stream
        
        elif port_name == 'inventory':
            # Report TOTAL cumulative delivered hydrogen as "inventory" 
            # so it appears in the Stream Summary Table even when idle.
            return Stream(
                mass_flow_kg_h=self.total_h2_delivered_kg,  # Cumulative Metric
                temperature_k=293.15,
                pressure_pa=self.delivery_pressure,
                composition={'H2': 1.0},
                phase='gas'
            )
            
        else:
            logger.debug(f"DischargeStation: Unknown output port '{port_name}'")
            return None
    
    def get_state(self) -> Dict[str, Any]:
        """Return aggregated state."""
        active_count = sum(1 for s in self.stations if s.get_current_dock().is_active)
        
        return {
            'component_id': self.component_id,
            'n_stations': self.n_stations,
            'active_units': active_count,
            'total_demand_signal_kg_h': self._demand_signal,
            'power_consumption_kw': self.power_consumption_kw,
            'trucks_filled_total': self.trucks_filled_total,
            'total_h2_delivered_kg': self.total_h2_delivered_kg
        }

    def _calculate_compression_work(self, inlet_p_pa: float, mass_kg: float) -> float:
        """
        Calculate compression energy using isentropic efficiency.
        Same helper logic as before.
        """
        if mass_kg <= 0 or inlet_p_pa < 1e5:
            return 0.0
        
        ratio_total = self.delivery_pressure / inlet_p_pa
        if ratio_total <= 1.0: return 0.0
        
        if ratio_total > 10: n_stages = 3
        elif ratio_total > 2.5: n_stages = 2
        else: n_stages = 1
        
        ratio_per_stage = ratio_total ** (1.0 / n_stages)
        total_work_j_kg = 0.0
        current_p = inlet_p_pa
        T_in = 283.15
        
        for _ in range(n_stages):
            p_next = current_p * ratio_per_stage
            if self._lut_manager:
                try:
                    h_in = self._lut_manager.lookup('H2', 'H', current_p, T_in)
                    s_in = self._lut_manager.lookup('H2', 'S', current_p, T_in)
                    h_out_s = self._lut_manager.lookup_from_s('H2', 'H', p_next, s_in)
                    work_ideal = h_out_s - h_in
                    work_real = work_ideal / self.isen_eff
                    total_work_j_kg += work_real
                    current_p = p_next
                    continue
                except: pass
            
            gamma = 1.41
            cp = 14300.0
            work_ideal = cp * T_in * ((ratio_per_stage ** ((gamma - 1) / gamma)) - 1)
            work_real = work_ideal / self.isen_eff
            total_work_j_kg += work_real
            current_p = p_next
            
        energy_j = total_work_j_kg * mass_kg / self.mech_eff
        return energy_j / 1e6
