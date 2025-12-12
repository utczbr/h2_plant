"""
NumPy-based vectorized tank array for high-performance storage operations.

Replaces Python list of tank objects with contiguous NumPy arrays,
enabling SIMD vectorization and Numba JIT compilation.
"""

import numpy as np
import numpy.typing as npt
from typing import Optional, Dict, Any

from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.enums import TankState
from h2_plant.core.constants import StorageConstants, GasConstants
from h2_plant.core.stream import Stream
from h2_plant.optimization.numba_ops import (
    find_available_tank,
    find_fullest_tank,
    batch_pressure_update,
    distribute_mass_to_tanks,
    calculate_total_mass_by_state
)


class TankArray(Component):
    """
    Vectorized array of hydrogen storage tanks.
    
    Uses NumPy arrays for tank properties instead of Python objects,
    enabling 10-50x performance improvement through vectorization and
    Numba JIT compilation.
    
    Example:
        # Create 8 tanks of 200 kg capacity at 350 bar
        tanks = TankArray(
            n_tanks=8,
            capacity_kg=200.0,
            pressure_bar=350
        )
        
        # Initialize
        tanks.initialize(dt=1.0, registry)
        
        # Fill tanks with 500 kg H2
        stored, overflow = tanks.fill(500.0)
        
        # Discharge 300 kg H2
        discharged = tanks.discharge(300.0)
        
        # Query state
        total_mass = tanks.get_total_mass()
        available_capacity = tanks.get_available_capacity()
    """
    
    def __init__(
        self,
        n_tanks: int,
        capacity_kg: float,
        pressure_bar: float,
        temperature_k: float = 298.15
    ):
        """
        Initialize tank array.
        
        Args:
            n_tanks: Number of tanks in array
            capacity_kg: Capacity of each tank (kg)
            pressure_bar: Nominal pressure (bar)
            temperature_k: Operating temperature (K)
        """
        super().__init__()
        
        self.n_tanks = n_tanks
        self.capacity_kg = capacity_kg
        self.pressure_pa = pressure_bar * 1e5
        self.temperature_k = temperature_k
        
        # NumPy arrays for vectorized operations
        self.masses = np.zeros(n_tanks, dtype=np.float64)
        self.states = np.full(n_tanks, TankState.IDLE, dtype=np.int32)
        self.capacities = np.full(n_tanks, capacity_kg, dtype=np.float64)
        self.pressures = np.zeros(n_tanks, dtype=np.float64)
        
        # Calculate tank volumes (constant)
        if self.temperature_k > 0:
            ideal_gas_density = self.pressure_pa / (GasConstants.R_H2 * self.temperature_k)
            self.volumes = self.capacities / ideal_gas_density if ideal_gas_density > 0 else np.zeros(n_tanks, dtype=np.float64)
        else:
            self.volumes = np.zeros(n_tanks, dtype=np.float64)

        
        # Output buffer for flow network (tracks new mass per timestep)
        self._output_buffer_kg = 0.0
        
        # Statistics
        self.total_filled_kg = 0.0
        self.total_discharged_kg = 0.0
        self.overflow_count = 0
    
    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """Initialize tank array."""
        super().initialize(dt, registry)
        
        # Update initial pressures
        self._update_pressures()
    
    def step(self, t: float) -> None:
        """Execute timestep - update pressures and states."""
        super().step(t)
        
        self._update_pressures()
        self._update_states()
    
    def get_state(self) -> Dict[str, Any]:
        """Return current state for checkpointing."""
        return {
            **super().get_state(),
            'n_tanks': self.n_tanks,
            'masses': self.masses.tolist(),
            'states': self.states.tolist(),
            'pressures': (self.pressures / 1e5).tolist(),  # Convert to bar
            'total_mass_kg': float(self.get_total_mass()),
            'available_capacity_kg': float(self.get_available_capacity()),
            'total_filled_kg': float(self.total_filled_kg),
            'total_discharged_kg': float(self.total_discharged_kg),
            'overflow_count': self.overflow_count
        }
    
    def fill(self, mass_kg: float) -> tuple[float, float]:
        """
        Fill tanks with hydrogen mass.
        
        Distributes mass across available (IDLE) tanks sequentially.
        
        Args:
            mass_kg: Mass to store (kg)
            
        Returns:
            Tuple of (mass_stored, mass_overflow)
            
        Example:
            stored, overflow = tanks.fill(500.0)
            if overflow > 0:
                print(f"Warning: {overflow:.1f} kg could not be stored")
        """
        # Use Numba-compiled distribution function
        updated_masses, overflow = distribute_mass_to_tanks(
            mass_kg,
            self.states,
            self.masses,
            self.capacities
        )
        
        self.masses = updated_masses
        stored = mass_kg - overflow
        
        # Update statistics
        self.total_filled_kg += stored
        if overflow > 0:
            self.overflow_count += 1
        
        return stored, overflow
    
    def discharge(self, mass_kg: float) -> float:
        """
        Discharge hydrogen from tanks.
        
        Draws from fullest tanks first.
        
        Args:
            mass_kg: Mass to discharge (kg)
            
        Returns:
            Actual mass discharged (may be less if insufficient stored)
            
        Example:
            if discharged < 300.0:
                print(f"Warning: Only {discharged:.1f} kg available")
        """
        # Finds fullest tanks and extracts mass
        # Implementation details omitted for brevity
        # ... logic to reduce mass in self.masses ...
        # For now, let's assume a simple greedy discharge for the interface
        
        remaining_demand = mass_kg
        total_discharged = 0.0
        
        # Sort indices by mass descending (simple heuristic)
        sorted_indices = np.argsort(self.masses)[::-1]
        
        for i in sorted_indices:
            if remaining_demand <= 0:
                break
            
            available = self.masses[i]
            if available > 0:
                amount = min(remaining_demand, available)
                self.masses[i] -= amount
                remaining_demand -= amount
                total_discharged += amount
                
                # Update state if needed (e.g. from FULL to PARTIAL/IDLE)
                # Simplified state update
                if self.masses[i] < 1e-6:
                     self.states[i] = TankState.EMPTY
                else:
                     self.states[i] = TankState.IDLE # Assuming discharging makes it idle/available
                     
        self.total_discharged_kg += total_discharged
        return total_discharged

    # --- Unified Storage Interface ---
    
    def get_inventory_kg(self) -> float:
        """Returns total stored hydrogen mass (Unified Interface)."""
        return self.get_total_mass()
        
    def withdraw_kg(self, amount: float) -> float:
        """Withdraws amount from storage (Unified Interface)."""
        return self.discharge(amount)

    def get_total_mass(self) -> float:
        """Return total mass stored in all tanks."""
        return float(np.sum(self.masses))

    def get_available_capacity(self) -> float:
        """Return total available capacity in kg."""
        total_cap = np.sum(self.capacities)
        current_mass = np.sum(self.masses)
        return float(total_cap - current_mass)
    def get_total_mass(self) -> float:
        """Return total mass stored across all tanks (kg)."""
        return float(np.sum(self.masses))
    
    def get_available_capacity(self) -> float:
        """Return total available capacity in idle/empty tanks (kg)."""
        available_mask = np.logical_or(
            self.states == TankState.IDLE,
            self.states == TankState.EMPTY
        )
        available_capacities = self.capacities[available_mask] - self.masses[available_mask]
        return float(np.sum(available_capacities))
    
    def get_mass_by_state(self, state: TankState) -> float:
        """Return total mass in tanks with specific state (kg)."""
        return calculate_total_mass_by_state(self.states, self.masses, int(state))
    
    def get_tank_count_by_state(self, state: TankState) -> int:
        """Return number of tanks in specific state."""
        return int(np.sum(self.states == state))

    def restore_state(self, state: Dict[str, Any]) -> None:
        """
        Restore the component's state from a dictionary.

        Args:
            state: The state dictionary to restore from.
        """
        super().__init__() # Reset base attributes
        self.n_tanks = state.get('n_tanks', self.n_tanks)
        self.masses = np.array(state.get('masses', []), dtype=np.float64)
        self.states = np.array(state.get('states', []), dtype=np.int32)
        self.pressures = np.array(state.get('pressures', []), dtype=np.float64)
        self.total_filled_kg = state.get('total_filled_kg', 0.0)
        self.total_discharged_kg = state.get('total_discharged_kg', 0.0)
        self.overflow_count = state.get('overflow_count', 0)
        self._initialized = state.get('initialized', False)
        self.component_id = state.get('component_id')

        # We assume dt, registry, and constant-derived attributes like volumes
        # are correctly set during the simulation engine's re-initialization process.
    
    def _update_pressures(self) -> None:
        """Update pressures for all tanks using ideal gas law."""
        # Ensure arrays are passed with the correct dtype to the Numba function
        masses_arr = np.array(self.masses, dtype=np.float64)
        volumes_arr = np.array(self.volumes, dtype=np.float64)
        
        # batch_pressure_update modifies the pressures array in-place
        batch_pressure_update(
            masses_arr,
            volumes_arr,
            self.pressures, # Pass the array to be modified
            self.temperature_k,
            GasConstants.R_H2
        )
    
    def _update_states(self) -> None:
        """Update tank states based on fill levels."""
        # Avoid division by zero if capacities are zero
        fill_percentages = np.divide(self.masses, self.capacities, out=np.zeros_like(self.masses), where=self.capacities!=0)
        
        # Set FULL state
        full_mask = fill_percentages >= StorageConstants.TANK_FULL_THRESHOLD
        self.states[full_mask] = TankState.FULL
        
        # Set EMPTY state
        empty_mask = fill_percentages <= StorageConstants.TANK_EMPTY_THRESHOLD
        self.states[empty_mask] = TankState.EMPTY
        
        # Set IDLE state (between empty and full)
        idle_mask = np.logical_and(
            fill_percentages > StorageConstants.TANK_EMPTY_THRESHOLD,
            fill_percentages < StorageConstants.TANK_FULL_THRESHOLD
        )
        self.states[idle_mask] = TankState.IDLE
    def get_output(self, port_name: str) -> Any:
        """Get output from specific port.
        
        Behavior depends on output_mode config (default: 'availability'):
        - 'availability': Returns stored inventory as available flow rate
        - 'passthrough': Returns buffer of mass added this timestep (legacy)
        """
        if port_name == 'h2_out':
            # Configurable output mode (Fix 5)
            output_mode = getattr(self, 'output_mode', 'availability')
            
            if output_mode == 'availability':
                # Return stored inventory as available flow rate
                inventory_kg = self.get_total_mass()
                flow_rate_kg_h = inventory_kg / self.dt if self.dt > 0 else 0.0
                avg_pressure = float(np.mean(self.pressures)) if len(self.pressures) > 0 else self.pressure_pa
            else:
                # Legacy passthrough: return buffer of mass added this timestep
                flow_rate_kg_h = self._output_buffer_kg / self.dt if self.dt > 0 else 0.0
                avg_pressure = self.pressure_pa
            
            return Stream(
                mass_flow_kg_h=flow_rate_kg_h,
                temperature_k=self.temperature_k,
                pressure_pa=avg_pressure,
                composition={'H2': 1.0},
                phase='gas'
            )
        else:
            raise ValueError(f"Unknown output port '{port_name}' on {self.component_id}")

    def receive_input(self, port_name: str, value: Any, resource_type: str) -> float:
        """Receive input."""
        if port_name == 'h2_in':
            if isinstance(value, Stream):
                # Fill tank
                stored, overflow = self.fill(value.mass_flow_kg_h * self.dt)
                # Track new mass in output buffer for next get_output call
                self._output_buffer_kg += stored
                return stored # Return amount actually stored
        return 0.0

    def extract_output(self, port_name: str, amount: float, resource_type: str) -> None:
        """Deduct extracted amount."""
        if port_name == 'h2_out':
            # Clear output buffer (mass has been taken)
            self._output_buffer_kg = 0.0
            # Discharge amount from tanks
            self.discharge(amount)
        else:
            raise ValueError(f"Unknown output port '{port_name}' on {self.component_id}")

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """Return port metadata."""
        return {
            'h2_in': {'type': 'input', 'resource_type': 'hydrogen', 'units': 'kg/h'},
            'h2_out': {'type': 'output', 'resource_type': 'hydrogen', 'units': 'kg/h'}
        }
