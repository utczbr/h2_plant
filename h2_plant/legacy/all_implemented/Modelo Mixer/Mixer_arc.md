Mixer Component Implementation
System Architecture Overview
This document provides detailed instructions for implementing the Mixer component (multi-stream mass and energy balance) following the established architecture patterns of the hydrogen production system.

1. Core Architecture Principles
1.1 Component Base Class Pattern
All mixer components MUST inherit from 
Component
 and implement:

from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.stream import Stream
class Mixer(Component):
    """
    Multi-stream mixer with mass and energy balance.
    
    Combines multiple inlet streams (gas or liquid) into a single
    outlet stream. Calculates outlet temperature and composition
    using thermodynamic properties from CoolProp/LUT.
    """
    
    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """Initialize mixer - validate configuration."""
        super().initialize(dt, registry)
        
    def step(self, t: float) -> None:
        """Execute timestep - perform mass/energy balance."""
        super().step(t)
        
    def get_state(self) -> Dict[str, Any]:
        """Return current state for monitoring."""
        return {
            **super().get_state(),
            # Your state variables
        }
CRITICAL RULES:

Always call super() methods first
Support variable number of inlet streams
All state must be JSON-serializable
Never hardcode constants (use config files)
2. Legacy Code Analysis
2.1 Current Implementation
File: 
h2_plant/legacy/toimplement/Modelo Mixer/Mixer.py

Key Features:

Function-based model for 3 input streams
Mass balance: m_out = Σ m_in_i
Energy balance: h_out = Σ(m_in_i × h_in_i) / m_out
Uses CoolProp for enthalpy calculations
Determines outlet temperature from energy balance
Physics Equations (must be preserved):

# Mass Balance
m_total = m_1 + m_2 + m_3 + ...
# Energy Balance
E_total = m_1*h_1 + m_2*h_2 + m_3*h_3 + ...
h_out = E_total / m_total
# Temperature from enthalpy
T_out = CoolProp('T', 'H', h_out, 'P', P_out, fluid)
2.2 Current Limitations to Address
Legacy code has:

Fixed 3 streams (should be flexible N streams)
Water-only (should support O2, H2, mixed gases)
No composition tracking for gas mixtures
No error recovery if CoolProp fails
New implementation must:

Support arbitrary number of inlet streams
Handle different fluids (Water, O2, H2, mixtures)
Track composition changes for gas mixtures
Provide simplified fallback if CoolProp unavailable
3. Thermodynamic Calculations
3.1 Mass Balance
def _calculate_mass_balance(self) -> float:
    """
    Calculate total outlet mass flow from inlet streams.
    
    Mass Conservation: Σ m_in = m_out
    
    Returns:
        m_out_kg_s: Total outlet mass flow (kg/s)
    """
    m_total_kg_s = 0.0
    
    for stream in self.inlet_streams.values():
        if stream:
            m_total_kg_s += stream.mass_flow_kg_h / 3600.0
    
    return m_total_kg_s
3.2 Energy Balance with CoolProp
def _calculate_energy_balance_coolprop(self) -> Tuple[float, float]:
    """
    Calculate outlet enthalpy using precise CoolProp properties.
    
    Energy Conservation: Σ(m_i × h_i) = m_out × h_out
    h_out = Σ(m_i × h_i) / m_out
    
    Returns:
        h_out_kj_kg: Outlet specific enthalpy (kJ/kg)
        T_out_k: Outlet temperature (K)
    """
    import CoolProp.CoolProp as CP
    
    total_energy_kw = 0.0  # kJ/s = kW
    total_mass_kg_s = 0.0
    
    # Determine fluid type (assume all same for now)
    fluid = self._get_fluid_name()
    
    for port_name, stream in self.inlet_streams.items():
        if stream is None:
            continue
            
        m_dot_kg_s = stream.mass_flow_kg_h / 3600.0
        T_k = stream.temperature_k
        P_pa = stream.pressure_pa
        
        # Get enthalpy from CoolProp (J/kg → kJ/kg)
        h_i_kj_kg = CP.PropsSI('H', 'T', T_k, 'P', P_pa, fluid) / 1000.0
        
        # Accumulate
        total_mass_kg_s += m_dot_kg_s
        total_energy_kw += m_dot_kg_s * h_i_kj_kg
    
    if total_mass_kg_s <= 0:
        return 0.0, 298.15  # Default
    
    # Outlet enthalpy
    h_out_kj_kg = total_energy_kw / total_mass_kg_s
    
    # Find outlet temperature from h_out and P_out
    h_out_j_kg = h_out_kj_kg * 1000.0
    P_out_pa = self.outlet_pressure_pa
    
    T_out_k = CP.PropsSI('T', 'H', h_out_j_kg, 'P', P_out_pa, fluid)
    
    return h_out_kj_kg, T_out_k
3.3 Simplified Energy Balance (Fallback)
def _calculate_energy_balance_simplified(self) -> float:
    """
    Simplified energy balance using constant Cp.
    
    For liquid water or when CoolProp unavailable.
    
    Returns:
        T_out_k: Outlet temperature (K)
    """
    # Get Cp from configuration
    if self.fluid_type == 'water':
        Cp_kj_kg_k = self.config['fluid_properties']['water']['cp_liquid']
    else:
        Cp_kj_kg_k = 4.18  # Default for water
    
    total_thermal_energy = 0.0  # kJ/s
    total_mass_kg_s = 0.0
    
    for stream in self.inlet_streams.values():
        if stream:
            m_dot = stream.mass_flow_kg_h / 3600.0
            T_k = stream.temperature_k
            
            # Thermal energy relative to 0K (simplified)
            thermal_energy_i = m_dot * Cp_kj_kg_k * (T_k - 273.15)
            
            total_mass_kg_s += m_dot
            total_thermal_energy += thermal_energy_i
    
    if total_mass_kg_s <= 0:
        return 298.15
    
    # Average temperature (°C)
    T_avg_c = total_thermal_energy / (total_mass_kg_s * Cp_kj_kg_k)
    T_out_k = T_avg_c + 273.15
    
    return T_out_k
3.4 Composition Mixing (for Gas Mixtures)
def _calculate_composition_mix(self) -> Dict[str, float]:
    """
    Calculate outlet composition for gas mixtures.
    
    Mass-weighted composition mixing:
    y_i_out = Σ(m_j × y_i_j) / m_total
    
    Returns:
        composition: Dict of species fractions (mole or mass based)
    """
    total_mass_kg_s = 0.0
    species_mass = {}  # kg/s of each species
    
    for stream in self.inlet_streams.values():
        if stream is None:
            continue
            
        m_dot = stream.mass_flow_kg_h / 3600.0
        total_mass_kg_s += m_dot
        
        # Accumulate each species
        for species, fraction in stream.composition.items():
            m_species = m_dot * fraction
            species_mass[species] = species_mass.get(species, 0.0) + m_species
    
    # Normalize to fractions
    composition_out = {}
    if total_mass_kg_s > 0:
        for species, mass in species_mass.items():
            composition_out[species] = mass / total_mass_kg_s
    
    return composition_out
4. Configuration Management
4.1 Add to 
configs/physics_parameters.yaml
mixer:
  # Fluid properties (for simplified calculations)
  fluid_properties:
    water:
      cp_liquid: 4.18  # kJ/(kg·K)
      cp_vapor: 2.08   # kJ/(kg·K)
    
    # Gas properties for fallback
    O2:
      cp: 0.918  # kJ/(kg·K) at 300K
    H2:
      cp: 14.3   # kJ/(kg·K) at 300K
    N2:
      cp: 1.04   # kJ/(kg·K) at 300K
  
  # Operational limits
  limits:
    max_temperature_difference_k: 100.0  # Warn if ΔT > 100K
    min_flow_kg_s: 1e-6                  # Minimum flow to consider
    max_streams: 10                       # Maximum inlet streams
  
  # Pressure handling
  pressure:
    drop_method: 'minimum'  # 'minimum', 'average', 'specified'
    default_drop_fraction: 0.01  # 1% pressure drop if not specified
4.2 Loading Configuration
import yaml
from pathlib import Path
class Mixer(Component):
    def __init__(self, mixer_id: str, fluid_type: str = 'water', ...):
        super().__init__()
        self.mixer_id = mixer_id
        self.fluid_type = fluid_type
        
        # Load configuration
        self.config = self._load_config()
        
        # Dynamic inlet streams
        self.inlet_streams: Dict[str, Optional[Stream]] = {}
        self.max_streams = self.config['limits']['max_streams']
    
    def _load_config(self) -> Dict[str, Any]:
        """Load mixer configuration."""
        config_path = Path(__file__).parent.parent.parent / 'configs' / 'physics_parameters.yaml'
        with open(config_path, 'r') as f:
            full_config = yaml.safe_load(f)
        return full_config.get('mixer', {})
5. Resource & Energy Tracking
5.1 Tracked Properties
The mixer must track:

Mass flows - all inlet streams and total outlet
Energy flows - enthalpy-based energy tracking
Temperatures - all inlets, computed outlet
Compositions - for gas mixtures
Pressure - all inlets, outlet (minimal drop)
5.2 State Reporting with Resource Flows
def get_state(self) -> Dict[str, Any]:
    """Return state with comprehensive resource tracking."""
    # Calculate statistics
    num_active_inlets = sum(1 for s in self.inlet_streams.values() if s is not None)
    
    # Temperature range
    inlet_temps = [s.temperature_k - 273.15 for s in self.inlet_streams.values() if s]
    T_min = min(inlet_temps) if inlet_temps else 0.0
    T_max = max(inlet_temps) if inlet_temps else 0.0
    
    # Prepare inlet flow information
    inlet_flows = {}
    for port_name, stream in self.inlet_streams.items():
        if stream:
            inlet_flows[port_name] = {
                'mass_flow_kg_h': float(stream.mass_flow_kg_h),
                'temperature_c': float(stream.temperature_k - 273.15),
                'pressure_bar': float(stream.pressure_pa / 1e5),
                'enthalpy_kj_kg': float(getattr(self, f'{port_name}_enthalpy', 0.0))
            }
    
    return {
        **super().get_state(),
        'mixer_id': self.mixer_id,
        'fluid_type': self.fluid_type,
        
        # Stream counts
        'num_inlets': num_active_inlets,
        'num_inlets_configured': len(self.inlet_streams),
        
        # Outlet properties
        'outlet_mass_flow_kg_h': float(self.outlet_mass_flow_kg_s * 3600.0),
        'outlet_temperature_c': float(self.outlet_temperature_k - 273.15),
        'outlet_pressure_bar': float(self.outlet_pressure_pa / 1e5),
        'outlet_enthalpy_kj_kg': float(self.outlet_enthalpy_kj_kg),
        
        # Temperature statistics
        'inlet_temperature_min_c': float(T_min),
        'inlet_temperature_max_c': float(T_max),
        'temperature_difference_k': float(T_max - T_min),
        
        # Inlet details
        'inlet_streams': inlet_flows,
        
        # Resource flow tracking
        'flows': {
            'inputs': {
                f'stream_{i+1}': {
                    'value': stream.mass_flow_kg_h if stream else 0.0,
                    'unit': 'kg/h',
                    'temperature_c': stream.temperature_k - 273.15 if stream else 0.0,
                    'pressure_bar': stream.pressure_pa / 1e5 if stream else 0.0,
                    'flowtype': 'FLUID_MASS'
                } for i, stream in enumerate(self.inlet_streams.values())
            },
            'outputs': {
                'mixed_stream': {
                    'value': self.outlet_mass_flow_kg_s * 3600.0,
                    'unit': 'kg/h',
                    'temperature_c': self.outlet_temperature_k - 273.15,
                    'pressure_bar': self.outlet_pressure_pa / 1e5,
                    'flowtype': 'FLUID_MASS'
                }
            }
        }
    }
6. Complete Implementation Example
"""
Mixer component for multi-stream mass and energy balance.
Preserves physics from legacy Mixer.py with architecture integration.
"""
from typing import Dict, Any, Optional, Tuple, List
import logging
from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.stream import Stream
try:
    import CoolProp.CoolProp as CP
    COOLPROP_AVAILABLE = True
except ImportError:
    CP = None
    COOLPROP_AVAILABLE = False
logger = logging.getLogger(__name__)
class Mixer(Component):
    """
    Multi-stream mixer with thermodynamic mixing calculations.
    
    Combines N inlet streams into single outlet stream using:
    - Mass balance: m_out = Σ m_in
    - Energy balance: h_out = Σ(m_in × h_in) / m_out
    - CoolProp for accurate enthalpy and temperature
    
    Preserves exact physics from legacy Mixer.py.
    
    Example:
        mixer = Mixer(
            mixer_id='water_mixer',
            fluid_type='water',
            outlet_pressure_bar=2.0
        )
        
        mixer.initialize(dt=1.0, registry)
        
        # Add inlet streams
        mixer.receive_input('stream_1', cold_stream, 'water')
        mixer.receive_input('stream_2', hot_stream, 'water')
        mixer.receive_input('stream_3', warm_stream, 'water')
        
        mixer.step(t=0.0)
        
        # Get mixed output
        mixed_stream = mixer.get_output('mixed_out')
        T_out = mixed_stream.temperature_k - 273.15  # °C
    """
    
    def __init__(
        self,
        mixer_id: str,
        fluid_type: str = 'water',
        outlet_pressure_bar: Optional[float] = None,
        max_inlet_streams: int = 10
    ):
        """
        Initialize mixer.
        
        Args:
            mixer_id: Unique identifier
            fluid_type: Type of fluid ('water', 'O2', 'H2', 'mixed')
            outlet_pressure_bar: Outlet pressure (bar), if None uses minimum inlet
            max_inlet_streams: Maximum number of inlet streams
        """
        super().__init__()
        
        self.mixer_id = mixer_id
        self.fluid_type = fluid_type
        self.outlet_pressure_bar = outlet_pressure_bar
        self.outlet_pressure_pa = outlet_pressure_bar * 1e5 if outlet_pressure_bar else None
        self.max_inlet_streams = max_inlet_streams
        
        # Load configuration
        self.config = self._load_config()
        
        # Inlet streams (dynamic dictionary)
        self.inlet_streams: Dict[str, Optional[Stream]] = {}
        
        # Outlet
        self.outlet_stream: Optional[Stream] = None
        self.outlet_mass_flow_kg_s: float = 0.0
        self.outlet_temperature_k: float = 298.15
        self.outlet_enthalpy_kj_kg: float = 0.0
        self.outlet_composition: Dict[str, float] = {}
        
        # Intermediate calculations (for state reporting)
        self.total_energy_kw: float = 0.0
    
    def _load_config(self) -> Dict[str, Any]:
        """Load mixer configuration."""
        # Would load from YAML in production
        return {
            'fluid_properties': {
                'water': {'cp_liquid': 4.18},
                'O2': {'cp': 0.918},
                'H2': {'cp': 14.3}
            },
            'limits': {
                'max_temperature_difference_k': 100.0,
                'min_flow_kg_s': 1e-6
            }
        }
    
    def _get_fluid_name(self) -> str:
        """Map fluid type to CoolProp fluid name."""
        fluid_map = {
            'water': 'Water',
            'O2': 'Oxygen',
            'H2': 'Hydrogen',
            'N2': 'Nitrogen'
        }
        return fluid_map.get(self.fluid_type, 'Water')
    
    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """Initialize mixer."""
        super().initialize(dt, registry)
        
        logger.info(
            f"Mixer '{self.mixer_id}': fluid={self.fluid_type}, "
            f"max_inlets={self.max_inlet_streams}, "
            f"P_out={self.outlet_pressure_bar} bar" if self.outlet_pressure_bar else "P_out=auto"
        )
    
    def step(self, t: float) -> None:
        """Execute timestep - perform mixing calculation."""
        super().step(t)
        
        # Reset
        self.outlet_mass_flow_kg_s = 0.0
        self.total_energy_kw = 0.0
        
        # Check if we have any inlet streams
        active_streams = [s for s in self.inlet_streams.values() if s is not None]
        
        if not active_streams:
            # No inputs, no output
            self.outlet_stream = None
            return
        
        # Calculate mass balance
        self.outlet_mass_flow_kg_s = self._calculate_mass_balance()
        
        if self.outlet_mass_flow_kg_s <= 0:
            self.outlet_stream = None
            return
        
        # Calculate energy balance and outlet temperature
        if COOLPROP_AVAILABLE and self.fluid_type in ['water', 'O2', 'H2', 'N2']:
            try:
                self.outlet_enthalpy_kj_kg, self.outlet_temperature_k = \
                    self._calculate_energy_balance_coolprop()
            except Exception as e:
                logger.warning(f"CoolProp failed in mixer: {e}. Using simplified.")
                self.outlet_temperature_k = self._calculate_energy_balance_simplified()
        else:
            self.outlet_temperature_k = self._calculate_energy_balance_simplified()
        
        # Calculate composition (for gas mixtures)
        if self.fluid_type == 'mixed':
            self.outlet_composition = self._calculate_composition_mix()
        else:
            # Single species
            fluid_name = self.fluid_type.upper() if self.fluid_type != 'water' else 'H2O'
            self.outlet_composition = {fluid_name: 1.0}
        
        # Determine outlet pressure
        if self.outlet_pressure_pa is None:
            # Use minimum inlet pressure (conservative)
            inlet_pressures = [s.pressure_pa for s in active_streams]
            self.outlet_pressure_pa = min(inlet_pressures)
        
        # Create outlet stream
        self.outlet_stream = Stream(
            mass_flow_kg_h=self.outlet_mass_flow_kg_s * 3600.0,
            temperature_k=self.outlet_temperature_k,
            pressure_pa=self.outlet_pressure_pa,
            composition=self.outlet_composition,
            phase='liquid' if self.fluid_type == 'water' else 'gas'
        )
    
    def _calculate_mass_balance(self) -> float:
        """Calculate total outlet mass flow."""
        m_total = 0.0
        for stream in self.inlet_streams.values():
            if stream:
                m_total += stream.mass_flow_kg_h / 3600.0
        return m_total
    
    def _calculate_energy_balance_coolprop(self) -> Tuple[float, float]:
        """
        Calculate using CoolProp (exact legacy logic).
        
        Returns:
            h_out_kj_kg, T_out_k
        """
        fluid = self._get_fluid_name()
        
        total_energy_kw = 0.0
        total_mass_kg_s = 0.0
        
        for stream in self.inlet_streams.values():
            if stream is None:
                continue
            
            m_dot = stream.mass_flow_kg_h / 3600.0
            T_k = stream.temperature_k
            P_pa = stream.pressure_pa
            
            # Get enthalpy (J/kg → kJ/kg)
            h_i_kj_kg = CP.PropsSI('H', 'T', T_k, 'P', P_pa, fluid) / 1000.0
            
            total_mass_kg_s += m_dot
            total_energy_kw += m_dot * h_i_kj_kg
        
        # Outlet enthalpy
        h_out_kj_kg = total_energy_kw / total_mass_kg_s
        
        # Outlet temperature
        h_out_j_kg = h_out_kj_kg * 1000.0
        T_out_k = CP.PropsSI('T', 'H', h_out_j_kg, 'P', self.outlet_pressure_pa, fluid)
        
        self.total_energy_kw = total_energy_kw
        
        return h_out_kj_kg, T_out_k
    
    def _calculate_energy_balance_simplified(self) -> float:
        """Simplified calculation using weighted average temperature."""
        Cp = self.config['fluid_properties'].get(
            self.fluid_type, {}
        ).get('cp_liquid', 4.18)
        
        total_thermal = 0.0
        total_mass = 0.0
        
        for stream in self.inlet_streams.values():
            if stream:
                m_dot = stream.mass_flow_kg_h / 3600.0
                T_c = stream.temperature_k - 273.15
                
                total_mass += m_dot
                total_thermal += m_dot * T_c
        
        if total_mass > 0:
            T_avg_c = total_thermal / total_mass
            return T_avg_c + 273.15
        
        return 298.15
    
    def _calculate_composition_mix(self) -> Dict[str, float]:
        """Calculate mass-weighted composition."""
        total_mass = 0.0
        species_mass = {}
        
        for stream in self.inlet_streams.values():
            if stream:
                m_dot = stream.mass_flow_kg_h / 3600.0
                total_mass += m_dot
                
                for species, fraction in stream.composition.items():
                    species_mass[species] = species_mass.get(species, 0.0) + m_dot * fraction
        
        composition = {}
        if total_mass > 0:
            for species, mass in species_mass.items():
                composition[species] = mass / total_mass
        
        return composition
    
    def receive_input(self, port_name: str, value: Any, resource_type: str) -> float:
        """
        Receive input stream.
        
        Dynamically creates inlet ports as streams are received.
        """
        if isinstance(value, Stream):
            # Add to inlet streams dictionary
            if len(self.inlet_streams) < self.max_inlet_streams:
                self.inlet_streams[port_name] = value
                return value.mass_flow_kg_h
            else:
                logger.warning(
                    f"Mixer {self.mixer_id}: Maximum {self.max_inlet_streams} "
                    f"streams reached. Ignoring port {port_name}."
                )
        return 0.0
    
    def get_output(self, port_name: str) -> Any:
        """Get output stream."""
        if port_name == 'mixed_out':
            return self.outlet_stream
        return None
    
    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """Return port metadata (dynamic based on connected streams)."""
        ports = {}
        
        # Input ports
        for port_name in self.inlet_streams.keys():
            ports[port_name] = {
                'type': 'input',
                'resource_type': 'fluid',
                'units': 'kg/h'
            }
        
        # Output port
        ports['mixed_out'] = {
            'type': 'output',
            'resource_type': 'fluid',
            'units': 'kg/h'
        }
        
        return ports
    
    def get_state(self) -> Dict[str, Any]:
        """Return current state."""
        num_active = sum(1 for s in self.inlet_streams.values() if s is not None)
        
        inlet_temps = [
            s.temperature_k - 273.15 
            for s in self.inlet_streams.values() if s
        ]
        
        return {
            **super().get_state(),
            'mixer_id': self.mixer_id,
            'fluid_type': self.fluid_type,
            'num_active_inlets': num_active,
            'outlet_mass_flow_kg_h': float(self.outlet_mass_flow_kg_s * 3600.0),
            'outlet_temperature_c': float(self.outlet_temperature_k - 273.15),
            'outlet_pressure_bar': float(self.outlet_pressure_pa / 1e5) if self.outlet_pressure_pa else 0.0,
            'outlet_enthalpy_kj_kg': float(self.outlet_enthalpy_kj_kg),
            'inlet_temperature_range_c': {
                'min': float(min(inlet_temps)) if inlet_temps else 0.0,
                'max': float(max(inlet_temps)) if inlet_temps else 0.0
            }
        }
7. Validation Testing Against Legacy
7.1 Create Validation Script
"""
Validation test for Mixer against legacy Mixer.py.
"""
import pytest
import sys
sys.path.insert(0, '/path/to/legacy/Modelo Mixer')
from Mixer import mixer_model
from h2_plant.components.mixing.mixer import Mixer
from h2_plant.core.stream import Stream
from h2_plant.core.component_registry import ComponentRegistry
def test_three_stream_water_mixer_validation():
    """Validate 3-stream water mixer matches legacy exactly."""
    # Legacy test case
    legacy_streams = [
        {"m_dot": 0.5, "T": 15.0, "P": 200.0},  # Cold
        {"m_dot": 0.3, "T": 80.0, "P": 220.0},  # Hot
        {"m_dot": 0.2, "T": 50.0, "P": 210.0}   # Warm
    ]
    P_out = 200.0  # kPa
    
    # Run legacy
    _, legacy_output = mixer_model(legacy_streams, P_out)
    legacy_m_out = legacy_output['Output Mass Flow Rate (kg/s)']
    legacy_T_out = legacy_output['Output Temperature (°C)']
    
    # Run new component
    mixer = Mixer(
        mixer_id='test_water_mixer',
        fluid_type='water',
        outlet_pressure_bar=P_out / 100.0  # kPa → bar
    )
    
    registry = ComponentRegistry()
    mixer.initialize(dt=1.0, registry)
    
    # Create streams
    stream_1 = Stream(
        mass_flow_kg_h=legacy_streams[0]['m_dot'] * 3600.0,
        temperature_k=legacy_streams[0]['T'] + 273.15,
        pressure_pa=legacy_streams[0]['P'] * 1000.0,
        composition={'H2O': 1.0},
        phase='liquid'
    )
    
    stream_2 = Stream(
        mass_flow_kg_h=legacy_streams[1]['m_dot'] * 3600.0,
        temperature_k=legacy_streams[1]['T'] + 273.15,
        pressure_pa=legacy_streams[1]['P'] * 1000.0,
        composition={'H2O': 1.0},
        phase='liquid'
    )
    
    stream_3 = Stream(
        mass_flow_kg_h=legacy_streams[2]['m_dot'] * 3600.0,
        temperature_k=legacy_streams[2]['T'] + 273.15,
        pressure_pa=legacy_streams[2]['P'] * 1000.0,
        composition={'H2O': 1.0},
        phase='liquid'
    )
    
    mixer.receive_input('stream_1', stream_1, 'water')
    mixer.receive_input('stream_2', stream_2, 'water')
    mixer.receive_input('stream_3', stream_3, 'water')
    
    mixer.step(t=0.0)
    
    # Validate (tolerance 0.1%)
    assert abs(mixer.outlet_mass_flow_kg_s - legacy_m_out) / legacy_m_out < 0.001
    
    T_out_c = mixer.outlet_temperature_k - 273.15
    assert abs(T_out_c - legacy_T_out) / legacy_T_out < 0.001
    
    print(f"✅ Mixer Validation PASS:")
    print(f"   Mass flow: {mixer.outlet_mass_flow_kg_s:.3f} kg/s (legacy: {legacy_m_out:.3f})")
    print(f"   Temperature: {T_out_c:.2f}°C (legacy: {legacy_T_out:.2f})")
7.2 Individual Unit Tests
"""
Unit tests for Mixer component.
"""
def test_mixer_initialization():
    """Test basic initialization."""
    mixer = Mixer(mixer_id='test', fluid_type='water')
    assert mixer.fluid_type == 'water'
    assert len(mixer.inlet_streams) == 0
def test_mass_balance():
    """Test mass conservation."""
    mixer = Mixer('test', 'water', outlet_pressure_bar=2.0)
    mixer.initialize(dt=1.0, ComponentRegistry())
    
    # Add streams: 1 kg/s + 2 kg/s = 3 kg/s
    stream1 = Stream(3600.0, 298.15, 2e5, {'H2O': 1.0}, 'liquid')
    stream2 = Stream(7200.0, 298.15, 2e5, {'H2O': 1.0}, 'liquid')
    
    mixer.receive_input('s1', stream1, 'water')
    mixer.receive_input('s2', stream2, 'water')
    mixer.step(0.0)
    
    assert abs(mixer.outlet_mass_flow_kg_s - 3.0) < 1e-6
def test_energy_balance_simple():
    """Test energy balance with equal temperatures."""
    # If all inlets at same T, outlet should be same T
    mixer = Mixer('test', 'water', outlet_pressure_bar=2.0)
    mixer.initialize(dt=1.0, ComponentRegistry())
    
    T_in = 298.15  # All at 25°C
    
    stream1 = Stream(3600.0, T_in, 2e5, {'H2O': 1.0}, 'liquid')
    stream2 = Stream(3600.0, T_in, 2e5, {'H2O': 1.0}, 'liquid')
    
    mixer.receive_input('s1', stream1, 'water')
    mixer.receive_input('s2', stream2, 'water')
    mixer.step(0.0)
    
    assert abs(mixer.outlet_temperature_k - T_in) < 0.1  # Within 0.1K
def test_variable_inlet_count():
    """Test mixer handles variable number of inlets."""
    mixer = Mixer('test', 'water', max_inlet_streams=5)
    mixer.initialize(dt=1.0, ComponentRegistry())
    
    # Start with 2 streams
    s1 = Stream(3600.0, 298.15, 2e5, {'H2O': 1.0}, 'liquid')
    s2 = Stream(3600.0, 310.15, 2e5, {'H2O': 1.0}, 'liquid')
    
    mixer.receive_input('inlet_1', s1, 'water')
    mixer.receive_input('inlet_2', s2, 'water')
    mixer.step(0.0)
    
    assert mixer.get_state()['num_active_inlets'] == 2
    
    # Add third stream
    s3 = Stream(3600.0, 305.15, 2e5, {'H2O': 1.0}, 'liquid')
    mixer.receive_input('inlet_3', s3, 'water')
    mixer.step(0.0)
    
    assert mixer.get_state()['num_active_inlets'] == 3
8. Summary Checklist
When implementing Mixer:

 Inherits from 
Component
 Supports variable number of inlet streams
 All constants in YAML config
 CoolProp/LUT for precise enthalpy (with fallback)
 Mass balance: Σm_in = m_out
 Energy balance: Σ(m×h)_in = m_out × h_out
 Composition mixing for gas mixtures
 Dynamic inlet port creation
 Comprehensive state reporting
 Type hints on all methods
 JSON-serializable state
 Error handling and logging
 Unit tests for mass/energy balance
 Validation tests against legacy
 Documentation and examples
9. References
Legacy Model: 
h2_plant/legacy/toimplement/Modelo Mixer/Mixer.py
Component Base: 
h2_plant/core/component.py
Stream Class: 
h2_plant/core/stream.py
Similar Components:
h2_plant/components/compression/compressor_storage.py
 (CoolProp)
h2_plant/components/water/water_pump_thermodynamic.py
 (thermodynamics)
h2_plant/components/cooling/chiller.py (energy balance)