"""
Drain Recorder Mixer Component.

This module implements a mixer that collects drain streams from multiple
separation units (KOD, Coalescer) and records the water inflow from each
source per timestep for mass balance tracking and visualization.

Purpose:
    In PEM electrolyzer systems, drain streams from knock-out drums and
    coalescers contain recovered water that is recycled to the process.
    This component tracks the contribution from each source, enabling
    accurate water balance accounting and identification of major sources.

Architecture:
    Implements the Component Lifecycle Contract (Layer 1):
    - `initialize()`: Pre-allocates history arrays for the simulation duration.
    - `step()`: Mixes received streams and records per-source contributions.
    - `get_state()`: Returns current mixed stream state and history summary.

References:
    - Legacy: drain_mixer.py
"""

from typing import Dict, Any, List, Optional, TYPE_CHECKING
import numpy as np

from h2_plant.core.component import Component
from h2_plant.core.stream import Stream
from h2_plant.optimization.numba_ops import solve_water_T_from_H_jit

if TYPE_CHECKING:
    from h2_plant.core.component_registry import ComponentRegistry


class DrainRecorderMixer(Component):
    """
    Mixer with per-source recording for drain water accounting.

    Collects drain streams from multiple upstream separators, mixes them
    using mass-weighted temperature blending, and records the contribution
    from each source at every timestep for subsequent analysis.

    This component fulfills the Component Lifecycle Contract (Layer 1):
        - `initialize()`: Creates pre-allocated NumPy arrays for history.
        - `step()`: Performs mass-weighted mixing and records to history.
        - `get_state()`: Returns mixer outlet conditions and recording stats.

    Attributes:
        source_ids (List[str]): List of expected source identifiers.
        outlet_stream (Stream): Mixed outlet stream for downstream.

    Example:
        >>> mixer = DrainRecorderMixer(source_ids=['KOD1', 'KOD2', 'Coalescer1'])
        >>> mixer.initialize(dt=1/60, registry=registry)
        >>> mixer.receive_input('KOD1', kod1_drain_stream)
        >>> mixer.receive_input('KOD2', kod2_drain_stream)
        >>> mixer.step(t=0.0)
        >>> mixed = mixer.get_output('outlet')
    """

    def __init__(self, source_ids: Optional[List[str]] = None):
        """
        Initialize the drain recorder mixer.

        Args:
            source_ids (List[str], optional): List of expected source port
                identifiers. Each source represents an upstream separator
                drain stream. Default: ['drain_1', 'drain_2'].
        """
        super().__init__()
        self.source_ids = source_ids or ['drain_1', 'drain_2']
        
        # Buffer for received streams in current timestep
        self._received_streams: Dict[str, Stream] = {}
        
        # Output stream after mixing
        # Output stream after mixing - initialized to zero flow
        self.outlet_stream: Optional[Stream] = Stream(0.0)
        
        # History arrays (pre-allocated in initialize)
        self._history: Dict[str, np.ndarray] = {}
        self._history_index: int = 0
        self._history_capacity: int = 0
        
        # Running totals for summary
        self._total_mass_kg: Dict[str, float] = {}
        
        # Exposed metrics for optimized recording
        self.dissolved_gas_ppm: float = 0.0
        
        self._lut_manager = None

    def initialize(self, dt: float, registry: 'ComponentRegistry') -> None:
        """
        Prepare the component for simulation execution.

        Pre-allocates NumPy arrays for recording drain contributions from
        each source over the simulation duration. This avoids expensive
        dynamic list appends during the simulation loop.

        Args:
            dt (float): Simulation timestep in hours.
            registry (ComponentRegistry): Central registry for component access.
        """
        super().initialize(dt, registry)
        
        # Estimate capacity based on 1-year simulation at given timestep
        # 8760 hours/year, plus buffer for extra steps
        self._history_capacity = int(8760 / dt) + 1000
        
        # Pre-allocate history arrays (float32 for memory efficiency)
        for source_id in self.source_ids:
            self._history[source_id] = np.zeros(self._history_capacity, dtype=np.float32)
            self._total_mass_kg[source_id] = 0.0
        
        self._history['time_h'] = np.zeros(self._history_capacity, dtype=np.float32)
        self._history_index = 0
        
        if registry.has('lut_manager'):
            self._lut_manager = registry.get('lut_manager')

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """
        Define the physical connection ports for this component.

        Creates input ports dynamically based on source_ids, plus one outlet.

        Returns:
            Dict[str, Dict[str, str]]: Port definitions with one input per
                source and a single 'outlet' output.
        """
        ports = {}
        for source_id in self.source_ids:
            ports[source_id] = {
                'type': 'input',
                'resource_type': 'water',
                'units': 'kg/h'
            }
        ports['outlet'] = {
            'type': 'output',
            'resource_type': 'water',
            'units': 'kg/h'
        }
        return ports

    def receive_input(self, port_name: str, value: Any, resource_type: str = None) -> float:
        """
        Accept a drain stream at the specified port.

        Buffers the stream for mixing during the next step() call.
        Streams are identified by port_name which should match a source_id.

        Args:
            port_name (str): Source identifier (e.g., 'KOD1', 'Coalescer1').
            value (Any): Stream object containing water drain data.
            resource_type (str, optional): Resource classification hint.

        Returns:
            float: Mass flow rate accepted (kg/h), or 0.0 if port unknown.
        """
        if port_name in self.source_ids and isinstance(value, Stream):
            self._received_streams[port_name] = value
            return value.mass_flow_kg_h
        return 0.0

    def step(self, t: float) -> None:
        """
        Execute one simulation timestep.

        Performs mass-weighted mixing of all received drain streams:
        1. Sum total mass flow from all sources.
        2. Calculate mass-weighted average temperature.
        3. Record each source's contribution to history arrays.
        4. Create outlet stream with mixed properties.

        Args:
            t (float): Current simulation time in hours.
        """
        super().step(t)
        
        total_mass_kg_h = 0.0
        weighted_temp_sum = 0.0
        min_pressure = float('inf')
        
        # Sum contributions from all sources
        for source_id in self.source_ids:
            stream = self._received_streams.get(source_id)
            mass_kg_h = stream.mass_flow_kg_h if stream else 0.0
            
            # Record to history
            if self._history_index < self._history_capacity:
                self._history[source_id][self._history_index] = mass_kg_h
            
            # Accumulate for mixing
            total_mass_kg_h += mass_kg_h
            if stream and mass_kg_h > 0:
                # RIGOROUS: Enthalpy Accumulation
                h_in = 0.0
                t_in = stream.temperature_k
                p_in = stream.pressure_pa
                
                if self._lut_manager:
                    try:
                         h_in = self._lut_manager.lookup('H2O', 'H', p_in, t_in)
                    except:
                         # Fallback: Cp * (T - T_ref), T_ref = 273.15K (IAPWS-95)
                         h_in = 4184.0 * (t_in - 273.15)
                else:
                    # Fallback: Cp * (T - T_ref), T_ref = 273.15K (IAPWS-95)
                    h_in = 4184.0 * (t_in - 273.15)
                    
                weighted_temp_sum += mass_kg_h * h_in # Reusing variable name for H sum
                min_pressure = min(min_pressure, p_in)
            
            # Update running totals (kg)
            self._total_mass_kg[source_id] = self._total_mass_kg.get(source_id, 0.0) + (mass_kg_h * self.dt)
        
        # Record time
        if self._history_index < self._history_capacity:
            self._history['time_h'][self._history_index] = t
            self._history_index += 1
        
        # Calculate mixed outlet properties
        if total_mass_kg_h > 0:
            avg_enthalpy = weighted_temp_sum / total_mass_kg_h
            outlet_pressure = min_pressure if min_pressure < float('inf') else 101325.0
            
            # Resolve T from H
            # Use linear T mix guess for solver
            t_guess_mix = (avg_enthalpy / 4184.0) + 298.15 
            mixed_temp_k = solve_water_T_from_H_jit(avg_enthalpy, outlet_pressure, t_guess_mix)
            
            # Weighted Composition Mixing
            mixed_comp = {}
            # Initialize with zero for all species found
            all_species = set()
            for s in self._received_streams.values():
                all_species.update(s.composition.keys())
            
            for species in all_species:
                total_species_mass = 0.0
                for source_id, s in self._received_streams.items():
                    total_species_mass += s.mass_flow_kg_h * s.composition.get(species, 0.0)
                
                if total_mass_kg_h > 0:
                    mixed_comp[species] = total_species_mass / total_mass_kg_h
            
            # Normalize
            total_frac = sum(mixed_comp.values())
            if total_frac > 0:
                mixed_comp = {k: v / total_frac for k, v in mixed_comp.items()}
            
            self.outlet_stream = Stream(
                mass_flow_kg_h=total_mass_kg_h,
                temperature_k=mixed_temp_k,
                pressure_pa=outlet_pressure,
                composition=mixed_comp,
                phase='liquid'
            )
            
            # Update exposed metric for recording
            non_water_mass_frac = sum(
                frac for sp, frac in self.outlet_stream.composition.items()
                if sp not in ('H2O', 'H2O_liq')
            )
            self.dissolved_gas_ppm = non_water_mass_frac * 1e6
            
        else:
            self.outlet_stream = Stream(0.0)
            self.dissolved_gas_ppm = 0.0
        
        # Clear buffer for next timestep
        self._received_streams.clear()

    def get_output(self, port_name: str) -> Any:
        """
        Retrieve the mixed outlet stream.

        Args:
            port_name (str): Port identifier. Expected: 'outlet'.

        Returns:
            Stream: Mixed drain stream, or None if no flow.
        """
        if port_name == 'outlet':
            return self.outlet_stream if self.outlet_stream else Stream(0.0)
        return None

    def get_history(self) -> Dict[str, np.ndarray]:
        """
        Retrieve recorded history of drain contributions.

        Returns the pre-allocated arrays trimmed to actual recorded length.

        Returns:
            Dict[str, np.ndarray]: Dictionary with 'time_h' and per-source
                mass flow arrays, each of length history_index.
        """
        idx = self._history_index
        return {key: arr[:idx].copy() for key, arr in self._history.items()}

    def get_state(self) -> Dict[str, Any]:
        """
        Retrieve the component's current operational state.

        Returns:
            Dict[str, Any]: State dictionary containing:
                - outlet_mass_kg_h (float): Current outlet mass flow.
                - outlet_temp_k (float): Current outlet temperature.
                - total_mass_kg (Dict): Cumulative mass from each source.
                - history_length (int): Number of recorded timesteps.
        """
        state = super().get_state()
        
        # Base metrics
        mass_kg_h = self.outlet_stream.mass_flow_kg_h if self.outlet_stream else 0.0
        temp_k = self.outlet_stream.temperature_k if self.outlet_stream else 0.0
        pressure_pa = self.outlet_stream.pressure_pa if self.outlet_stream else 101325.0
        
        # Calculate dissolved gas PPM from stream composition
        # Sum all non-water species mass fractions
        ppm = 0.0
        if self.outlet_stream and self.outlet_stream.composition:
            non_water_mass_frac = sum(
                frac for sp, frac in self.outlet_stream.composition.items()
                if sp not in ('H2O', 'H2O_liq')
            )
            ppm = non_water_mass_frac * 1e6
        
        state.update({
            # Original keys
            'outlet_mass_kg_h': mass_kg_h,
            'outlet_temp_k': temp_k,
            'total_mass_kg': self._total_mass_kg.copy(),
            'history_length': self._history_index,
            # Aliased keys for visualization compatibility
            'outlet_mass_flow_kg_h': mass_kg_h,
            'outlet_temperature_c': temp_k - 273.15 if temp_k > 0 else 0.0,
            'outlet_temperature_k': temp_k,
            'outlet_pressure_kpa': pressure_pa / 1000.0,
            'dissolved_gas_ppm': ppm,
        })
        return state
