"""
Vectorized Hydrogen Tank Array Component.

This module implements a high-performance tank array using NumPy vectorized
operations and Numba JIT compilation. Unlike object-oriented tank collections,
this approach uses contiguous memory arrays enabling SIMD optimization.

Performance Architecture:
    - **NumPy Arrays**: Tank properties (mass, pressure, state) stored in
      contiguous float64/int32 arrays for cache efficiency.
    - **Numba JIT**: Core operations (pressure update, mass distribution)
      compiled to native code for 10-50x speedup.
    - **Vectorized State Updates**: Batch processing of all tanks per timestep.

Tank State Machine:
    - IDLE: Available for filling or discharging.
    - FULL: At capacity threshold (>95%).
    - EMPTY: Below empty threshold (<5%).
    - Other states defined in TankState enum.

Architecture:
    Implements the Component Lifecycle Contract (Layer 1):
    - `initialize()`: Pre-computes volumes and initial pressures.
    - `step()`: Updates pressures and states for all tanks.
    - `get_state()`: Returns aggregate metrics and per-tank arrays.

Unified Storage Interface:
    Provides `get_inventory_kg()` and `withdraw_kg()` methods for
    Orchestrator compatibility.
"""

import numpy as np
import numpy.typing as npt
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.enums import TankState
from h2_plant.core.constants import StorageConstants, GasConstants
from h2_plant.core.stream import Stream
from h2_plant.core.stream import Stream
from h2_plant.optimization.numba_ops import (
    find_available_tank,
    find_fullest_tank,
    batch_pressure_update,
    distribute_mass_to_tanks,
    calculate_total_mass_by_state,
    distribute_mass_and_energy,       # New
    apply_heat_loss_batch,            # New
    batch_pressure_update_vector_T    # New
)


class TankArray(Component):
    """
    Vectorized array of hydrogen storage tanks for high-performance simulation.

    Uses NumPy arrays for tank properties instead of Python objects, enabling
    significant performance improvement through vectorization and Numba JIT.

    This component fulfills the Component Lifecycle Contract (Layer 1):
        - `initialize()`: Computes initial pressures from ideal gas law.
        - `step()`: Batch updates pressures and states for all tanks.
        - `get_state()`: Returns aggregate metrics and serialized arrays.

    Fill/Discharge Strategy:
        - Fill: Distributes to IDLE tanks sequentially until full.
        - Discharge: Draws from fullest tanks first (greedy algorithm).

    Attributes:
        n_tanks (int): Number of tanks in the array.
        capacity_kg (float): Capacity per tank (kg).
        masses (np.ndarray): Per-tank mass inventory (kg).
        pressures (np.ndarray): Per-tank pressure (Pa).
        states (np.ndarray): Per-tank state (TankState enum).

    Example:
        >>> tanks = TankArray(n_tanks=8, capacity_kg=200.0, pressure_bar=350)
        >>> tanks.initialize(dt=1/60, registry=registry)
        >>> stored, overflow = tanks.fill(500.0)
        >>> discharged = tanks.discharge(300.0)
    """

    def __init__(
        self,
        n_tanks: int,
        capacity_kg: float,
        pressure_bar: float,
        temperature_k: float = 298.15,
        max_output_flow_kg_h: float = 5000.0
    ):
        """
        Initialize the tank array.

        Args:
            n_tanks (int): Number of tanks in array.
            capacity_kg (float): Capacity of each tank in kg.
            pressure_bar (float): Nominal operating pressure in bar.
            temperature_k (float): Initial temperature in Kelvin.
            max_output_flow_kg_h (float): Maximum physical discharge rate (kg/h).
        """
        super().__init__()
        self.n_tanks = int(n_tanks)
        self.capacity_kg = float(capacity_kg)
        self.pressure_bar = float(pressure_bar)
        self.pressure_pa = self.pressure_bar * 1e5
        self.temperature_k = float(temperature_k)
        self.max_output_flow_kg_h = float(max_output_flow_kg_h)
        self.heat_transfer_coeff_UA = 50.0  # W/K per tank (Added default)

        # Contiguous NumPy arrays for vectorized operations
        self.masses = np.zeros(n_tanks, dtype=np.float64)
        self.states = np.full(n_tanks, TankState.IDLE, dtype=np.int32)
        self.capacities = np.full(n_tanks, capacity_kg, dtype=np.float64)
        self.pressures = np.zeros(n_tanks, dtype=np.float64)
        self.temperatures = np.full(n_tanks, temperature_k, dtype=np.float64) # Per-tank temperature array

        # Calculate tank volumes from ideal gas law: V = m / ρ = mRT/P
        if self.temperature_k > 0:
            ideal_gas_density = self.pressure_pa / (GasConstants.R_H2 * self.temperature_k)
            self.volumes = (self.capacities / ideal_gas_density if ideal_gas_density > 0
                           else np.zeros(n_tanks, dtype=np.float64))
        else:
            self.volumes = np.zeros(n_tanks, dtype=np.float64)

        # Output buffer for flow network
        self._output_buffer_kg = 0.0

        # Statistics
        self.total_filled_kg = 0.0
        self.total_discharged_kg = 0.0
        self.overflow_count = 0

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """
        Prepare the tank array for simulation execution.

        Fulfills the Component Lifecycle Contract initialization phase.
        Computes initial pressures from mass inventory.

        Args:
            dt (float): Simulation timestep in hours.
            registry (ComponentRegistry): Central registry for component access.
        """
        super().initialize(dt, registry)
        self._update_pressures()

    def step(self, t: float) -> None:
        """
        Execute one simulation timestep.

        Updates pressures using ideal gas law and states based on fill level.

        Args:
            t (float): Current simulation time in hours.
        """
        super().step(t)
        
        # Apply Heat Loss (Thermal relaxation)
        # Convert hours to seconds for physics calculation
        dt_seconds = self.dt * 3600.0
        T_ambient = 298.15 # Assumed ambient
        
        apply_heat_loss_batch(
            self.temperatures,
            self.masses,
            T_ambient,
            dt_seconds,
            self.heat_transfer_coeff_UA,
            GasConstants.CV_H2
        )

        self._update_pressures()
        self._update_states()

    def get_state(self) -> Dict[str, Any]:
        """
        Retrieve the component's current operational state.

        Fulfills the Component Lifecycle Contract state access.

        Returns:
            Dict[str, Any]: State dictionary containing:
                - n_tanks (int): Number of tanks.
                - masses (List[float]): Per-tank masses (kg).
                - states (List[int]): Per-tank states.
                - pressures (List[float]): Per-tank pressures (bar).
                - total_mass_kg (float): Aggregate mass (kg).
                - available_capacity_kg (float): Remaining capacity (kg).
        """
        return {
            **super().get_state(),
            'n_tanks': self.n_tanks,
            'masses': self.masses.tolist(),
            'states': self.states.tolist(),
            'pressures': (self.pressures / 1e5).tolist(),
            'total_mass_kg': float(self.get_total_mass()),
            'available_capacity_kg': float(self.get_available_capacity()),
            'total_filled_kg': float(self.total_filled_kg),
            'total_discharged_kg': float(self.total_discharged_kg),
            'overflow_count': self.overflow_count,
            'temperatures': self.temperatures.tolist(),
            'max_temperature_k': float(np.max(self.temperatures)) if self.n_tanks > 0 else 0.0
        }

    def fill(self, mass_kg: float, T_in: float = 298.15) -> tuple[float, float]:
        """
        Fill tanks with hydrogen mass.

        Distributes mass across IDLE tanks using Numba-compiled distribution.

        Args:
            mass_kg (float): Mass to store in kg.
            T_in (float): Incoming gas temperature (K). Default 298.15.

        Returns:
            tuple[float, float]: (mass_stored, mass_overflow).
        """
        updated_masses, updated_temps, overflow = distribute_mass_and_energy(
            mass_kg,
            T_in, # Pass incoming temperature
            self.states,
            self.masses,
            self.temperatures,
            self.capacities,
            GasConstants.GAMMA_H2
        )

        self.masses = updated_masses
        self.temperatures = updated_temps
        stored = mass_kg - overflow

        self.total_filled_kg += stored
        if overflow > 0:
            self.overflow_count += 1

        return stored, overflow

    def discharge(self, mass_kg: float) -> float:
        """
        Discharge hydrogen from tanks.

        Uses greedy algorithm drawing from fullest tanks first.

        Args:
            mass_kg (float): Mass to discharge in kg.

        Returns:
            float: Actual mass discharged (may be less if insufficient).
        """
        remaining_demand = mass_kg
        total_discharged = 0.0

        # Sort indices by mass descending (fullest first)
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

                # Update tank state
                if self.masses[i] < 1e-6:
                    self.states[i] = TankState.EMPTY
                else:
                    self.states[i] = TankState.IDLE

        self.total_discharged_kg += total_discharged
        return total_discharged

    # --- Unified Storage Interface ---

    def get_inventory_kg(self) -> float:
        """
        Return total stored hydrogen mass (Unified Storage Interface).

        Returns:
            float: Total mass across all tanks in kg.
        """
        return self.get_total_mass()

    def withdraw_kg(self, amount: float) -> float:
        """
        Withdraw hydrogen from storage (Unified Storage Interface).

        Args:
            amount (float): Requested withdrawal mass in kg.

        Returns:
            float: Actual mass withdrawn in kg.
        """
        return self.discharge(amount)

    def get_total_mass(self) -> float:
        """
        Return total mass stored across all tanks.

        Returns:
            float: Total mass in kg.
        """
        return float(np.sum(self.masses))

    def get_available_capacity(self) -> float:
        """
        Return available capacity in IDLE and EMPTY tanks.

        Returns:
            float: Available capacity in kg.
        """
        available_mask = np.logical_or(
            self.states == TankState.IDLE,
            self.states == TankState.EMPTY
        )
        available_capacities = self.capacities[available_mask] - self.masses[available_mask]
        return float(np.sum(available_capacities))

    def get_mass_by_state(self, state: TankState) -> float:
        """
        Return total mass in tanks with specific state.

        Args:
            state (TankState): Target state to filter by.

        Returns:
            float: Total mass in kg.
        """
        return calculate_total_mass_by_state(self.states, self.masses, int(state))

    def get_tank_count_by_state(self, state: TankState) -> int:
        """
        Return count of tanks in specific state.

        Args:
            state (TankState): Target state to count.

        Returns:
            int: Number of tanks in state.
        """
        return int(np.sum(self.states == state))

    def restore_state(self, state: Dict[str, Any]) -> None:
        """
        Restore component state from checkpoint dictionary.

        Args:
            state (Dict[str, Any]): State dictionary from get_state().
        """
        super().__init__()
        self.n_tanks = state.get('n_tanks', self.n_tanks)
        self.masses = np.array(state.get('masses', []), dtype=np.float64)
        self.states = np.array(state.get('states', []), dtype=np.int32)
        self.states = np.array(state.get('states', []), dtype=np.int32)
        self.pressures = np.array(state.get('pressures', []), dtype=np.float64)
        
        # Restore temperatures if available, else default to nominal
        temps = state.get('temperatures', [])
        if temps:
             self.temperatures = np.array(temps, dtype=np.float64)
        else:
             self.temperatures = np.full(self.n_tanks, self.temperature_k, dtype=np.float64)
             
        self.total_filled_kg = state.get('total_filled_kg', 0.0)
        self.total_discharged_kg = state.get('total_discharged_kg', 0.0)
        self.overflow_count = state.get('overflow_count', 0)
        self._initialized = state.get('initialized', False)
        self.component_id = state.get('component_id')

    def _update_pressures(self) -> None:
        """
        Update pressures for all tanks using ideal gas law.

        Uses Numba JIT-compiled batch operation for performance.
        P = mRT/V for each tank.
        """
        masses_arr = np.array(self.masses, dtype=np.float64)
        volumes_arr = np.array(self.volumes, dtype=np.float64)
        temps_arr = np.array(self.temperatures, dtype=np.float64)

        batch_pressure_update_vector_T(
            masses_arr,
            volumes_arr,
            self.pressures,
            temps_arr, # Pass temperature array
            GasConstants.R_H2
        )
        
        # --- Relief Valve Logic (PSV) ---
        # If any tank exceeds max_pressure_bar, vent excess mass to clamp at max_pressure.
        # This prevents unphysical over-pressurization when compressor keeps running.
        max_p_pa = self.pressure_pa  # This is the P_max setting in Pa
        
        overpressure_mask = self.pressures > max_p_pa
        if np.any(overpressure_mask):
            # Calculate max allowed mass at current T: m_max = P_max * V / (R * T)
            # We use the array operation for vectorized clamping
            R = GasConstants.R_H2
            m_max = (max_p_pa * volumes_arr[overpressure_mask]) / (R * temps_arr[overpressure_mask])
            
            # Vent the difference
            vented_mass = np.sum(self.masses[overpressure_mask] - m_max)
            # print(f"DEBUG: Venting {vented_mass:.2f} kg from tanks due to overpressure ({np.max(self.pressures[overpressure_mask])/1e5:.1f} bar > {max_p_pa/1e5} bar)")
            
            # Update state
            self.masses[overpressure_mask] = m_max
            self.pressures[overpressure_mask] = max_p_pa

    def _update_states(self) -> None:
        """
        Update tank states based on fill level thresholds.

        State transitions:
        - fill >= 95%: FULL
        - fill <= 5%: EMPTY
        - otherwise: IDLE
        """
        fill_percentages = np.divide(
            self.masses, self.capacities,
            out=np.zeros_like(self.masses),
            where=self.capacities != 0
        )

        full_mask = fill_percentages >= StorageConstants.TANK_FULL_THRESHOLD
        self.states[full_mask] = TankState.FULL

        empty_mask = fill_percentages <= StorageConstants.TANK_EMPTY_THRESHOLD
        self.states[empty_mask] = TankState.EMPTY

        idle_mask = np.logical_and(
            fill_percentages > StorageConstants.TANK_EMPTY_THRESHOLD,
            fill_percentages < StorageConstants.TANK_FULL_THRESHOLD
        )
        self.states[idle_mask] = TankState.IDLE

    def get_output(self, port_name: str) -> Any:
        """
        Retrieve output stream from specified port.

        Output mode determines flow rate:
        - 'availability': Returns current inventory as mass_kg_h for display purposes.
        - 'passthrough': Returns buffer of mass added this timestep.

        Args:
            port_name (str): Port identifier ('h2_out').

        Returns:
            Stream: Output stream at tank conditions.

        Raises:
            ValueError: If port_name is not valid.
        """
        if port_name == 'inventory':
            # Special reporting port: Returns total inventory as 'flow rate' for Stream Table display
            inventory_kg = self.get_total_mass()
            avg_pressure = float(np.max(self.pressures)) if len(self.pressures) > 0 else self.pressure_pa
            avg_temp = float(np.mean(self.temperatures)) if len(self.temperatures) > 0 else self.temperature_k
            
            return Stream(
                mass_flow_kg_h=inventory_kg,  # Display Inventory in 'Total' column
                temperature_k=avg_temp,
                pressure_pa=avg_pressure,
                composition={'H2': 1.0},
                phase=self._get_phase()
            )

        if port_name == 'h2_out':
            output_mode = getattr(self, 'output_mode', 'availability')

            if output_mode == 'availability':
                # Return inventory as available mass, but clamp flow rate to physical limit
                # This prevents downstream components from seeing physically impossible flow rates (e.g. 27 tons/h)
                inventory_kg = self.get_total_mass()
                
                # Physical flow limit (pipe/valve limit) - default to 5000 kg/h or user config
                max_flow = getattr(self, 'max_output_flow_kg_h', 5000.0)
                
                # If interpreting as flow source, don't exceed max_flow
                flow_rate_kg_h = min(inventory_kg / self.dt if self.dt > 0 else inventory_kg, max_flow)
                
                # For pure inventory reporting, we rely on get_state(), not get_output()
                # Use inventory for mass balance check, but flow_rate for stream
                # For stream table display: show inventory directly (not divided by dt!)
                
                avg_pressure = float(np.max(self.pressures)) if len(self.pressures) > 0 else self.pressure_pa
                avg_temp = float(np.mean(self.temperatures)) if len(self.temperatures) > 0 else self.temperature_k
            else:
                # Passthrough mode: show actual flow rate added this timestep
                flow_rate_kg_h = self._output_buffer_kg / self.dt if self.dt > 0 else 0.0
                avg_pressure = float(np.max(self.pressures)) if len(self.pressures) > 0 else self.pressure_pa
                avg_temp = float(np.mean(self.temperatures)) if len(self.temperatures) > 0 else self.temperature_k

            return Stream(
                mass_flow_kg_h=flow_rate_kg_h,
                temperature_k=avg_temp,
                pressure_pa=avg_pressure,
                composition={'H2': 1.0},
                phase='gas'
            )
        else:
            raise ValueError(f"Unknown output port '{port_name}' on {self.component_id}")

    def receive_input(self, port_name: str, value: Any, resource_type: str) -> float:
        """
        Accept hydrogen input from upstream component.

        Args:
            port_name (str): Target port ('h2_in').
            value (Any): Stream object containing hydrogen.
            resource_type (str): Resource classification hint.

        Returns:
            float: Mass actually stored (kg).
        """
        if port_name == 'h2_in':
            if isinstance(value, Stream):
                mass_to_store = value.mass_flow_kg_h * self.dt
                T_in = value.temperature_k
                stored, overflow = self.fill(mass_to_store, T_in)
                self._output_buffer_kg += stored
                
                # Log overflow warning (throttled to prevent spam)
                if overflow > 0.001:
                    fill_pct = 100.0 * self.get_total_mass() / (self.n_tanks * self.capacity_kg)
                    if self.overflow_count <= 10 or self.overflow_count % 100 == 0:
                        logger.warning(
                            f"Storage {self.component_id}: OVERFLOW! "
                            f"Rejected {overflow:.2f} kg, Fill: {fill_pct:.1f}%, "
                            f"Total overflows: {self.overflow_count}"
                        )
                return stored
        return 0.0

    def extract_output(self, port_name: str, amount: float, resource_type: str) -> None:
        """
        Acknowledge extraction and discharge from tanks.

        Args:
            port_name (str): Source port ('h2_out').
            amount (float): Mass to discharge (kg).
            resource_type (str): Resource classification hint.

        Raises:
            ValueError: If port_name is not valid.
        """
        if port_name == 'h2_out':
            self._output_buffer_kg = 0.0
            self.discharge(amount)
        else:
            raise ValueError(f"Unknown output port '{port_name}' on {self.component_id}")

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """
        Define the physical connection ports for this component.

        Returns:
            Dict[str, Dict[str, str]]: Port definitions.
        """
        return {
            'h2_in': {'type': 'input', 'resource_type': 'hydrogen', 'units': 'kg/h'},
            'h2_out': {'type': 'output', 'resource_type': 'hydrogen', 'units': 'kg/h'}
        }
