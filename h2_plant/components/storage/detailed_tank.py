"""
DetailedTankArray - High-Fidelity Hydrogen Storage with Discrete State Machines.

This module implements an advanced tank array model with:
- Individual tank state machines (IDLE, FILLING, EMPTYING)
- Real Gas Equation of State (via LUT) for accurate pressure calculation
- Priority-based fill/discharge allocation (fullest/emptiest first)
- Demand Signal input for causal flow control

Physics Model:
    Tank pressure is calculated using the Helmholtz Energy Equation of State:
    P = f(ρ, T) where ρ = m / V
    
    This is interpolated from pre-computed LUTs for speed.

Architecture:
    This component implements the "Demand Signal Propagation Pattern":
    - Reads `demand_signal` input to determine discharge rate
    - Outputs physical `h2_out` stream based on fulfilled demand
    - Maintains strict forward-only causality with 1-step delay

Reference:
    Logic adapted from HPProcessLogicNoLP.py (Thesis CAS 03.12.2025)
"""

from enum import IntEnum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import logging

from h2_plant.core.component import Component
from h2_plant.core.stream import Stream
from h2_plant.core.constants import GasConstants

logger = logging.getLogger(__name__)


class TankState(IntEnum):
    """
    Discrete operating states for individual tanks.
    
    Enforces mutual exclusivity: a tank cannot fill and empty simultaneously.
    """
    IDLE = 0      # Not actively filling or emptying
    FILLING = 1   # Receiving hydrogen from production
    EMPTYING = 2  # Supplying hydrogen to discharge station


@dataclass
class DetailedTank:
    """
    State container for a single hydrogen storage vessel.
    
    Attributes:
        id: Unique identifier (0 to n_tanks-1)
        volume_m3: Geometric volume [m³]
        mass_kg: Current hydrogen inventory [kg]
        pressure_pa: Current internal pressure [Pa]
        temperature_k: Current gas temperature [K]
        state: Current operating state
    """
    id: int
    volume_m3: float
    mass_kg: float = 0.0
    pressure_pa: float = 101325.0
    temperature_k: float = 293.15
    state: TankState = TankState.IDLE


class DetailedTankArray(Component):
    """
    High-fidelity storage array with individual tank state management.
    
    Implements the "Demand Signal Propagation Pattern":
    - `h2_in`: Incoming hydrogen from production (forward flow)
    - `demand_signal`: Requested discharge rate (backward information)
    - `h2_out`: Physical discharge to station (forward flow)
    
    Tank Allocation Logic:
    - FILLING: Prioritizes tanks with highest mass (top off fullest first)
    - EMPTYING: Prioritizes tanks with lowest mass (clear emptiest first)
    
    Args:
        n_tanks: Number of individual vessels
        volume_per_tank_m3: Geometric volume per vessel [m³]
        max_pressure_bar: Maximum allowable pressure [bar]
        initial_pressure_bar: Starting pressure (optional, default 1 bar)
        ambient_temp_k: Ambient/gas temperature [K]
    
    Example YAML:
        - id: "HP_Storage_Array"
          type: "DetailedTank"
          params:
            n_tanks: 30
            volume_per_tank_m3: 50.0
            max_pressure_bar: 500.0
    """
    
    def __init__(
        self,
        n_tanks: int = 10,
        volume_per_tank_m3: float = 50.0,
        max_pressure_bar: float = 500.0,
        initial_pressure_bar: float = 1.0,
        ambient_temp_k: float = 293.15,
        min_discharge_pressure_bar: float = 30.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.n_tanks = n_tanks
        self.volume_per_tank = volume_per_tank_m3
        self.max_pressure_pa = max_pressure_bar * 1e5
        self.initial_pressure_pa = initial_pressure_bar * 1e5
        self.ambient_temp_k = ambient_temp_k
        self.min_discharge_pressure_pa = min_discharge_pressure_bar * 1e5
        
        # Initialize tank objects
        self.tanks: List[DetailedTank] = []
        for i in range(n_tanks):
            tank = DetailedTank(
                id=i,
                volume_m3=volume_per_tank_m3,
                mass_kg=0.0,
                pressure_pa=self.initial_pressure_pa,
                temperature_k=ambient_temp_k,
                state=TankState.IDLE
            )
            self.tanks.append(tank)
        
        # Output buffers (set during step, read by get_output)
        self._h2_out_stream: Optional[Stream] = None
        self._last_discharge_pressure_pa: float = 101325.0
        
        # Input buffers
        self._demand_signal_rate: float = 0.0  # kg/h requested
        self._h2_in_rate: float = 0.0          # kg/h incoming
        
        # LUT Manager reference (set during initialize)
        self._lut_manager = None
        
        # Statistics
        self.total_filled_kg: float = 0.0
        self.total_discharged_kg: float = 0.0
        
    @property
    def total_mass_kg(self) -> float:
        """Total hydrogen mass in all tanks."""
        return sum(t.mass_kg for t in self.tanks)

    @property
    def avg_pressure_bar(self) -> float:
        """Average pressure across all tanks."""
        if not self.tanks: return 0.0
        return np.mean([t.pressure_pa for t in self.tanks]) / 1e5
        
    def initialize(self, dt: float, registry) -> None:
        """
        Prepare the tank array for simulation.
        
        Pre-computes initial masses from pressure using EOS.
        """
        super().initialize(dt, registry)
        
        # Get LUT Manager for Real Gas properties
        self._lut_manager = registry.get("lut_manager") if registry.has("lut_manager") else None
        
        # Calculate initial mass from initial pressure
        for tank in self.tanks:
            tank.mass_kg = self._mass_from_pressure(
                tank.pressure_pa, 
                tank.volume_m3, 
                tank.temperature_k
            )
        
        logger.info(
            f"DetailedTankArray initialized: {self.n_tanks} tanks, "
            f"{self.volume_per_tank:.1f} m³ each, "
            f"initial P = {self.initial_pressure_pa/1e5:.1f} bar"
        )
    
    def receive_input(self, port_name: str, value: Any, **kwargs) -> float:
        """
        Receive input streams on designated ports.
        
        Ports:
            h2_in: Physical hydrogen inflow from production
            demand_signal: Requested discharge rate (kg/h encoded as mass_flow)
            
        Returns:
            float: Amount accepted (kg/h or similar units matching input)
        """
        if isinstance(value, Stream):
             stream = value
        else:
             logger.warning(f"DetailedTankArray.receive_input: Port '{port_name}' received non-Stream value type '{type(value)}'. Ignoring.")
             return 0.0

        if port_name == 'h2_in':
            val = stream.mass_flow_kg_h if stream else 0.0
            if val is not None:
                self._h2_in_rate = float(val)
                return self._h2_in_rate
            else:
                logger.warning(f"DetailedTankArray: Stream on port '{port_name}' has None mass_flow_kg_h. Defaulting to 0.0.")
                self._h2_in_rate = 0.0
                return 0.0
                
        elif port_name == 'demand_signal':
            # Signal carries requested rate in mass_flow_kg_h field
            val = stream.mass_flow_kg_h if stream else 0.0
            if val is not None:
                self._demand_signal_rate = float(val)
            else:
                logger.debug(f"DetailedTankArray: Signal on port '{port_name}' has None mass_flow_kg_h. Defaulting to 0.0.")
                self._demand_signal_rate = 0.0
            return 0.0 # Signals are not "consumed" resources
        else:
            logger.warning(f"DetailedTankArray: Unknown input port '{port_name}'")
            return 0.0
    
    def step(self, t: float) -> None:
        """
        Execute one simulation timestep.
        
        Order of operations:
        1. Update all tank physics (pressure from mass)
        2. Process incoming hydrogen (filling)
        3. Process demand signal (emptying)
        4. Buffer output stream
        """
        super().step(t)
        
        # 1. Update Physics - Pressure from mass using EOS
        for tank in self.tanks:
            tank.pressure_pa = self._pressure_from_mass(
                tank.mass_kg,
                tank.volume_m3,
                tank.temperature_k
            )
        
        # 2. Distribute Incoming Hydrogen
        if self._h2_in_rate > 0:
            step_mass = self._h2_in_rate * self.dt
            overflow = self._distribute_filling(step_mass)
            if overflow > 0:
                logger.warning(f"DetailedTankArray: {overflow:.2f} kg overflow (all tanks full)")
        
        # 3. Fulfill Demand Signal
        discharge_mass = 0.0
        avg_pressure = 101325.0
        
        if self._demand_signal_rate > 0:
            requested_mass = self._demand_signal_rate * self.dt
            discharge_mass, avg_pressure = self._distribute_emptying(requested_mass)
            self._last_discharge_pressure_pa = avg_pressure
        
        # 4. Buffer Output Stream
        self._h2_out_stream = Stream(
            mass_flow_kg_h=discharge_mass / self.dt if self.dt > 0 else 0.0,
            temperature_k=self.ambient_temp_k,
            pressure_pa=avg_pressure,
            composition={'H2': 1.0},
            phase='gas'
        )
        
        # Reset input buffers for next step
        self._h2_in_rate = 0.0
        self._demand_signal_rate = 0.0
    
    def get_output(self, port_name: str = 'h2_out') -> Optional[Stream]:
        """
        Get output stream for specified port.
        
        Ports:
            h2_out: Physical discharge stream
            inventory: Reporting stream (total mass encoded as flow)
        """
        if port_name == 'h2_out':
            return self._h2_out_stream
        
        elif port_name == 'inventory':
            # Special reporting port for Stream Table
            total_mass = sum(t.mass_kg for t in self.tanks)
            max_pressure = max(t.pressure_pa for t in self.tanks) if self.tanks else 101325.0
            avg_temp = np.mean([t.temperature_k for t in self.tanks]) if self.tanks else 293.15
            
            return Stream(
                mass_flow_kg_h=total_mass,  # Inventory in "Total" column
                temperature_k=avg_temp,
                pressure_pa=max_pressure,
                composition={'H2': 1.0},
                phase='gas'
            )
        
        else:
            # Common reporting behavior probes all ports; use debug to avoid log spam
            logger.debug(f"DetailedTankArray: Unknown output port '{port_name}'")
            return None
    
    def get_state(self) -> Dict[str, Any]:
        """Return component state for checkpointing and monitoring."""
        tank_states = [
            {
                'id': t.id,
                'mass_kg': t.mass_kg,
                'pressure_bar': t.pressure_pa / 1e5,
                'state': t.state.name
            }
            for t in self.tanks
        ]
        
        return {
            'component_id': self.component_id,
            'n_tanks': self.n_tanks,
            'total_mass_kg': sum(t.mass_kg for t in self.tanks),
            'avg_pressure_bar': np.mean([t.pressure_pa for t in self.tanks]) / 1e5,
            'max_pressure_bar': max(t.pressure_pa for t in self.tanks) / 1e5 if self.tanks else 0,
            'min_pressure_bar': min(t.pressure_pa for t in self.tanks) / 1e5 if self.tanks else 0,
            'tanks_filling': sum(1 for t in self.tanks if t.state == TankState.FILLING),
            'tanks_emptying': sum(1 for t in self.tanks if t.state == TankState.EMPTYING),
            'tanks_idle': sum(1 for t in self.tanks if t.state == TankState.IDLE),
            'total_filled_kg': self.total_filled_kg,
            'total_discharged_kg': self.total_discharged_kg,
            'tank_details': tank_states
        }
    
    # -------------------------------------------------------------------------
    # Internal Methods
    # -------------------------------------------------------------------------
    
    def _pressure_from_mass(self, mass_kg: float, volume_m3: float, temp_k: float) -> float:
        """
        Calculate pressure from mass using EOS.
        
        Uses LUT if available, falls back to ideal gas.
        """
        if mass_kg <= 0.1:
            return 101325.0  # Atmospheric minimum
        
        density = mass_kg / volume_m3  # kg/m³
        
        if self._lut_manager:
            try:
                # Real Gas: P = f(ρ, T) from Helmholtz EOS
                pressure = self._lut_manager.lookup('H2', 'P', density, temp_k)
                return max(pressure, 101325.0)
            except Exception:
                pass
        
        # Fallback: Ideal Gas P = ρRT
        R_specific = GasConstants.R_H2  # J/(kg·K)
        pressure = density * R_specific * temp_k
        return max(pressure, 101325.0)
    
    def _mass_from_pressure(self, pressure_pa: float, volume_m3: float, temp_k: float) -> float:
        """
        Calculate mass from pressure using EOS.
        
        Used for initialization.
        """
        if pressure_pa <= 101325.0:
            return 0.0
        
        if self._lut_manager:
            try:
                # Real Gas: ρ = f(P, T)
                density = self._lut_manager.lookup('H2', 'D', pressure_pa, temp_k)
                return density * volume_m3
            except Exception:
                pass
        
        # Fallback: Ideal Gas m = PV/(RT)
        R_specific = GasConstants.R_H2
        mass = (pressure_pa * volume_m3) / (R_specific * temp_k)
        return max(mass, 0.0)
    
    def _max_mass_at_pressure(self, pressure_pa: float, volume_m3: float, temp_k: float) -> float:
        """Calculate maximum mass a tank can hold at given pressure."""
        return self._mass_from_pressure(pressure_pa, volume_m3, temp_k)
    
    def _distribute_filling(self, incoming_mass_kg: float) -> float:
        """
        Distribute incoming hydrogen to tanks.
        
        Priority: FILLING tanks first, sorted by mass DESCENDING (top off fullest).
        If no FILLING tanks, activate IDLE tanks (lowest pressure first).
        
        Args:
            incoming_mass_kg: Total mass to distribute [kg]
            
        Returns:
            Overflow mass if all tanks full [kg]
        """
        remaining = incoming_mass_kg
        loop_count = 0
        max_loops = len(self.tanks) + 2  # Safety break

        # Pre-calculate max mass (optimization: move out of loop)
        max_mass_per_tank = self._max_mass_at_pressure(
            self.max_pressure_pa, 
            self.volume_per_tank, 
            self.ambient_temp_k
        )

        while remaining > 1e-6 and loop_count < max_loops:
            loop_count += 1
            
            # Get candidates: currently FILLING
            candidates = [t for t in self.tanks if t.state == TankState.FILLING]
            
            # If none filling, activate IDLE tanks
            # Prioritize matching pressure or lowest pressure? 
            # Standard: Lowest pressure first (Empty tanks)
            if not candidates:
                idle_tanks = [t for t in self.tanks if t.state == TankState.IDLE]
                if not idle_tanks:
                    # No tanks available at all -> Truly full
                    break
                
                # Sort by pressure (ascending) to fill empty ones first
                idle_tanks.sort(key=lambda x: x.pressure_pa)
                
                # Activate ONE tank (or more?)
                # Just one is sufficient as loop will repeat if it fills instantly
                new_tank = idle_tanks[0]
                new_tank.state = TankState.FILLING
                candidates = [new_tank]
            
            # Sort candidates by mass DESCENDING (top off fullest first)
            # This balances the "Fill Active" vs "Start New" logic?
            # Actually, we just want to fill the active set.
            candidates.sort(key=lambda x: x.mass_kg, reverse=True)
            
            # Try to fill candidates
            progress_made = False
            
            for tank in candidates:
                if remaining <= 1e-6:
                    break
                
                space = max_mass_per_tank - tank.mass_kg
                if space <= 1e-6:
                    # This tank is full, mark IDLE and continue
                    tank.state = TankState.IDLE
                    continue
                
                fill_amount = min(remaining, space)
                
                tank.mass_kg += fill_amount
                remaining -= fill_amount
                self.total_filled_kg += fill_amount
                progress_made = True
                
                # Check if tank is now full
                if tank.mass_kg >= max_mass_per_tank * 0.9999:
                    tank.state = TankState.IDLE
                    # logger.debug(f"Tank {tank.id} filled, switching to IDLE")
            
            if not progress_made:
                # If we iterated candidates and couldn't put mass in, 
                # and presumably they were marked IDLE if full...
                # Check if we have any options left. 
                # If we just marked them IDLE, next loop will look for new IDLE tanks.
                # But if we found NO IDLE tanks earlier, we broke.
                # So if we are here, we might be stuck?
                # Ensure we don't infinite loop if physics says space=0.
                break

        return remaining  # Overflow
    
    def _distribute_emptying(self, demand_kg: float) -> Tuple[float, float]:
        """
        Distribute discharge demand across tanks.
        
        Priority: EMPTYING tanks first, sorted by mass ASCENDING (clear emptiest).
        If no EMPTYING tanks, activate IDLE tanks (fullest first).
        
        Args:
            demand_kg: Total mass requested [kg]
            
        Returns:
            Tuple of (mass_supplied, weighted_average_pressure)
        """
        if demand_kg <= 0:
            return 0.0, 101325.0
        
        # Determine minimum mass (heel) based on min discharge pressure
        heel_mass = self._mass_from_pressure(
            self.min_discharge_pressure_pa,
            self.volume_per_tank,
            self.ambient_temp_k
        )
        
        # Get candidates: currently EMPTYING
        candidates = [t for t in self.tanks if t.state == TankState.EMPTYING]
        
        # If none emptying, activate IDLE tanks (prioritize fullest for best pressure)
        if not candidates:
            # Only consider IDLE tanks that have usable mass > heel
            idle_tanks = [t for t in self.tanks if t.state == TankState.IDLE and t.mass_kg > (heel_mass + 0.1)]
            idle_tanks.sort(key=lambda x: x.mass_kg, reverse=True)  # Fullest first
            
            if idle_tanks:
                idle_tanks[0].state = TankState.EMPTYING
                candidates = [idle_tanks[0]]
            else:
                return 0.0, 101325.0  # No mass available
        
        # Sort by mass ASCENDING (empty lowest first)
        candidates.sort(key=lambda x: x.mass_kg)
        
        supplied = 0.0
        weighted_pressure_sum = 0.0
        remaining = demand_kg
        
        for tank in candidates:
            if remaining <= 0:
                break
            
            # Usable mass is amount above heel
            available = max(0.0, tank.mass_kg - heel_mass)
            pull = min(remaining, available)
            
            if pull > 0:
                # Record pressure BEFORE withdrawal
                weighted_pressure_sum += tank.pressure_pa * pull
                
                tank.mass_kg -= pull
                remaining -= pull
                supplied += pull
                self.total_discharged_kg += pull
            
            # Check if tank reached min pressure (effectively empty for discharge)
            if tank.mass_kg <= heel_mass + 0.01:
                # Don't zero out mass - keep the cushion gas
                tank.state = TankState.IDLE
                logger.debug(f"Tank {tank.id} reached min pressure ({self.min_discharge_pressure_pa/1e5:.1f} bar), transitioning to IDLE")
        
        avg_pressure = weighted_pressure_sum / supplied if supplied > 0 else 101325.0
        return supplied, avg_pressure
    
    def get_total_mass(self) -> float:
        """Get total hydrogen inventory across all tanks."""
        return sum(t.mass_kg for t in self.tanks)
