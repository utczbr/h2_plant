"""
External Hydrogen Source Component.

This module provides a configurable hydrogen source for testing hydrogen-dependent
components (compressors, coolers, dryers, etc.) without requiring a complete 
electrolyzer topology.

Features:
    - Fully configurable stream properties (mass flow, T, P, composition).
    - Optional impurities (O2, H2O, N2) for realistic testing.
    - Multiple output modes (fixed_flow, on_demand, time_profile).
    - Energy source output for components requiring electrical input.

Use Cases:
    - Unit testing compressors, heat exchangers, separators.
    - Debugging single component behavior.
    - Creating minimal topologies for integration testing.

Architecture:
    Implements the Component Lifecycle Contract (Layer 1):
    - `initialize()`: Prepares the source for simulation.
    - `step()`: Updates flow counters and time-based profiles.
    - `get_output()`: Returns fully configured H2 stream.
    - `get_state()`: Returns cumulative metrics.
"""

import logging
from typing import Dict, Any, Optional, List
import numpy as np

from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.stream import Stream

logger = logging.getLogger(__name__)


class ExternalH2Source(Component):
    """
    Configurable hydrogen source for testing hydrogen-dependent components.

    This component acts as an "infinite source" of hydrogen with fully
    configurable thermodynamic properties. Designed for component testing
    without requiring complete electrolyzer topology.

    Configuration Options:
        - **Output Mode**: 'fixed_flow', 'on_demand', 'time_profile'
        - **Thermodynamic State**: Temperature (K), Pressure (Pa/bar)
        - **Composition**: H2 purity with optional O2, H2O, N2 impurities
        - **Phase**: 'gas' or 'liquid' (for cryogenic H2)

    Attributes:
        mode (str): Operating mode.
        mass_flow_kg_h (float): Hydrogen mass flow rate (kg/h).
        temperature_k (float): Stream temperature (K).
        pressure_pa (float): Stream pressure (Pa).
        composition (dict): Species mole fractions.
        enthalpy_kj_kg (float, optional): Override enthalpy if specified.

    Example:
        >>> # Create source for compressor testing at 5 bar, 30°C
        >>> h2_source = ExternalH2Source(
        ...     mass_flow_kg_h=100.0,
        ...     temperature_k=303.15,
        ...     pressure_bar=5.0,
        ...     h2_purity=0.9998,
        ...     o2_impurity=0.0002
        ... )
        >>> stream = h2_source.get_output('h2_out')
        >>> print(f"T={stream.temperature_k}K, P={stream.pressure_pa}Pa")
    """

    # Default thermodynamic properties for H2
    DEFAULTS = {
        'mode': 'fixed_flow',
        'mass_flow_kg_h': 100.0,
        'temperature_k': 303.15,  # 30°C - typical PEM output
        'pressure_bar': 30.0,     # Typical PEM stack pressure
        'h2_purity': 0.9998,      # High purity from electrolyzer
        'o2_impurity': 0.0001,    # Typical O2 crossover
        'h2o_impurity': 0.0001,   # Moisture content
        'n2_impurity': 0.0,       # No nitrogen by default
        'phase': 'gas',
        'enthalpy_kj_kg': None,   # Auto-calculate if None
    }

    def __init__(
        self,
        component_id: str = "h2_source",
        mode: str = None,
        mass_flow_kg_h: float = None,
        temperature_k: float = None,
        temperature_c: float = None,
        pressure_pa: float = None,
        pressure_bar: float = None,
        h2_purity: float = None,
        o2_impurity: float = None,
        h2o_impurity: float = None,
        n2_impurity: float = None,
        composition: Dict[str, float] = None,
        phase: str = None,
        enthalpy_kj_kg: float = None,
        time_profile: List[float] = None,
        config: Dict[str, Any] = None
    ):
        """
        Initialize the configurable hydrogen source.

        Args:
            component_id (str): Unique identifier. Default: 'h2_source'.
            mode (str): 'fixed_flow', 'on_demand', 'time_profile'.
            mass_flow_kg_h (float): H2 mass flow rate (kg/h).
            temperature_k (float): Stream temperature in Kelvin.
            temperature_c (float): Stream temperature in Celsius (converted to K).
            pressure_pa (float): Stream pressure in Pascals.
            pressure_bar (float): Stream pressure in bar (converted to Pa).
            h2_purity (float): H2 mole fraction (0-1). Default: 0.9998.
            o2_impurity (float): O2 mole fraction (0-1). Default: 0.0001.
            h2o_impurity (float): H2O mole fraction (0-1). Default: 0.0001.
            n2_impurity (float): N2 mole fraction (0-1). Default: 0.
            composition (dict): Override full composition dict (sum should = 1).
            phase (str): 'gas' or 'liquid'. Default: 'gas'.
            enthalpy_kj_kg (float, optional): Override enthalpy calculation.
            time_profile (list, optional): Hourly flow profile for time_profile mode.
            config (dict, optional): Configuration dictionary (from YAML/topology).
        """
        super().__init__()
        self.component_id = component_id

        # Handle dict config if passed (Pattern used by PlantBuilder)
        if config is not None or (isinstance(component_id, dict)):
            if isinstance(component_id, dict):
                config = component_id
                self.component_id = config.get('component_id', 'h2_source')
            self._apply_config(config)
        else:
            # Apply individual parameters with defaults
            self.mode = mode or self.DEFAULTS['mode']
            self.mass_flow_kg_h = mass_flow_kg_h or self.DEFAULTS['mass_flow_kg_h']
            
            # Temperature: prefer K, fallback to C conversion
            if temperature_k is not None:
                self.temperature_k = temperature_k
            elif temperature_c is not None:
                self.temperature_k = temperature_c + 273.15
            else:
                self.temperature_k = self.DEFAULTS['temperature_k']
            
            # Pressure: prefer Pa, fallback to bar conversion
            if pressure_pa is not None:
                self.pressure_pa = pressure_pa
            elif pressure_bar is not None:
                self.pressure_pa = pressure_bar * 1e5
            else:
                self.pressure_pa = self.DEFAULTS['pressure_bar'] * 1e5
            
            # Composition
            if composition is not None:
                self.composition = composition
            else:
                h2 = h2_purity if h2_purity is not None else self.DEFAULTS['h2_purity']
                o2 = o2_impurity if o2_impurity is not None else self.DEFAULTS['o2_impurity']
                h2o = h2o_impurity if h2o_impurity is not None else self.DEFAULTS['h2o_impurity']
                n2 = n2_impurity if n2_impurity is not None else self.DEFAULTS['n2_impurity']
                self.composition = {'H2': h2, 'O2': o2, 'H2O': h2o, 'N2': n2}
            
            self.phase = phase or self.DEFAULTS['phase']
            self.enthalpy_kj_kg = enthalpy_kj_kg
            self.time_profile = time_profile

        # State variables
        self.h2_output_kg = 0.0
        self.cumulative_h2_kg = 0.0
        self.current_step_index = 0

        # Energy output for power-requiring components
        self.power_available_kw = 1e6  # Infinite power source

    def _apply_config(self, config: Dict[str, Any]) -> None:
        """Apply configuration dictionary."""
        if config is None:
            config = {}
            
        self.mode = config.get('mode', self.DEFAULTS['mode'])
        self.mass_flow_kg_h = float(config.get('mass_flow_kg_h', self.DEFAULTS['mass_flow_kg_h']))
        
        # Temperature
        if 'temperature_k' in config:
            self.temperature_k = float(config['temperature_k'])
        elif 'temperature_c' in config:
            self.temperature_k = float(config['temperature_c']) + 273.15
        else:
            self.temperature_k = self.DEFAULTS['temperature_k']
        
        # Pressure
        if 'pressure_pa' in config:
            self.pressure_pa = float(config['pressure_pa'])
        elif 'pressure_bar' in config:
            self.pressure_pa = float(config['pressure_bar']) * 1e5
        else:
            self.pressure_pa = self.DEFAULTS['pressure_bar'] * 1e5
        
        # Composition and Phase
        self.phase = config.get('phase', self.DEFAULTS['phase'])
        self.enthalpy_kj_kg = config.get('enthalpy_kj_kg', None)
        self.time_profile = config.get('time_profile', None)

        if 'composition' in config:
            self.composition = config['composition']
        else:
            # Inputs are molar fractions (purity/impurity)
            y_h2 = float(config.get('h2_purity', self.DEFAULTS['h2_purity']))
            y_o2 = float(config.get('o2_impurity', self.DEFAULTS['o2_impurity']))
            y_h2o = float(config.get('h2o_impurity', self.DEFAULTS['h2o_impurity']))
            y_n2 = float(config.get('n2_impurity', self.DEFAULTS['n2_impurity']))
            
            # Convert Molar -> Mass Fractions
            # x_i = (y_i * MW_i) / sum(y_j * MW_j)
            from h2_plant.core.constants import GasConstants
            MW = {s: GasConstants.SPECIES_DATA[s]['molecular_weight'] for s in ['H2', 'O2', 'H2O', 'N2']}
            
            w_h2 = y_h2 * MW['H2']
            w_o2 = y_o2 * MW['O2']
            w_h2o = y_h2o * MW['H2O']
            w_n2 = y_n2 * MW['N2']
            w_total = w_h2 + w_o2 + w_h2o + w_n2
            
            if w_total > 0:
                self.composition = {
                    'H2': w_h2 / w_total,
                    'O2': w_o2 / w_total,
                    'H2O': w_h2o / w_total,
                    'N2': w_n2 / w_total
                }
            else:
                self.composition = {'H2': 1.0}  # Fallback

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """Prepare the component for simulation."""
        super().initialize(dt, registry)
        self.current_step_index = 0

    def step(self, t: float) -> None:
        """Execute one simulation timestep."""
        super().step(t)

        # Determine flow based on mode
        if self.mode == 'time_profile' and self.time_profile:
            # Cycle through profile
            idx = self.current_step_index % len(self.time_profile)
            current_flow = float(self.time_profile[idx])
            self.current_step_index += 1
        elif self.mode == 'on_demand':
            # Return whatever was requested (set by receive_input)
            current_flow = getattr(self, '_requested_flow_kg_h', self.mass_flow_kg_h)
        else:
            # Fixed flow
            current_flow = self.mass_flow_kg_h

        # Calculate output
        self.h2_output_kg = current_flow * self.dt
        self.cumulative_h2_kg += self.h2_output_kg

    def receive_input(self, port_name: str, value: Any, resource_type: str = None) -> float:
        """
        Accept input requests (for on_demand mode).

        Args:
            port_name (str): 'flow_request' for requesting specific flow.
            value: Requested flow rate (kg/h) or Stream object.
            resource_type (str): Resource hint (ignored).

        Returns:
            float: Amount accepted.
        """
        if port_name == 'flow_request':
            if isinstance(value, (int, float)):
                self._requested_flow_kg_h = float(value)
                return float(value)
        return 0.0

    def get_output(self, port_name: str) -> Any:
        """
        Retrieve the output stream from a specified port.

        Args:
            port_name (str): Port identifier.
                - 'h2_out': Hydrogen stream with configured properties.
                - 'power_out': Power availability for energy-input ports.

        Returns:
            Stream or float: Output stream or power availability.
        """
        if port_name == 'h2_out':
            # Determine current flow
            if self.mode == 'time_profile' and self.time_profile:
                idx = (self.current_step_index - 1) % len(self.time_profile)
                current_flow = float(self.time_profile[idx]) if self.time_profile else self.mass_flow_kg_h
            else:
                current_flow = getattr(self, '_requested_flow_kg_h', self.mass_flow_kg_h)

            stream = Stream(
                mass_flow_kg_h=current_flow,
                temperature_k=self.temperature_k,
                pressure_pa=self.pressure_pa,
                composition=self.composition.copy(),
                phase=self.phase
            )
            
            # Override enthalpy if specified
            if self.enthalpy_kj_kg is not None:
                stream.enthalpy_kj_kg = self.enthalpy_kj_kg
                
            return stream

        elif port_name == 'power_out':
            # Return available power for energy input ports
            return self.power_available_kw

        else:
            raise ValueError(f"Unknown output port '{port_name}' on {self.component_id}")

    def extract_output(self, port_name: str, amount: float, resource_type: str) -> None:
        """Called by FlowNetwork after downstream accepts the flow."""
        # For an infinite source, no state update needed
        pass

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """Define the physical connection ports."""
        return {
            'h2_out': {
                'type': 'output',
                'resource_type': 'hydrogen',
                'units': 'kg/h',
                'description': 'Hydrogen stream with configurable T, P, composition'
            },
            'power_out': {
                'type': 'output',
                'resource_type': 'electricity',
                'units': 'kW',
                'description': 'Infinite power source for component testing'
            },
            'flow_request': {
                'type': 'input',
                'resource_type': 'control',
                'units': 'kg/h',
                'description': 'Request specific flow rate (on_demand mode)'
            }
        }

    def get_state(self) -> Dict[str, Any]:
        """Retrieve the component's current operational state."""
        return {
            **super().get_state(),
            'mode': self.mode,
            'mass_flow_kg_h': self.mass_flow_kg_h,
            'temperature_k': self.temperature_k,
            'pressure_pa': self.pressure_pa,
            'pressure_bar': self.pressure_pa / 1e5,
            'composition': self.composition,
            'phase': self.phase,
            'h2_output_kg': self.h2_output_kg,
            'cumulative_h2_kg': self.cumulative_h2_kg,
            'h2_purity': self.composition.get('H2', 0.0),
            'o2_impurity': self.composition.get('O2', 0.0),
            'h2o_impurity': self.composition.get('H2O', 0.0),
            # Calculate molar ppm for cross-checking
            'o2_impurity_ppm_mol': (Stream(mass_flow_kg_h=1, composition=self.composition).get_mole_frac('O2') * 1e6)
        }

    @classmethod
    def create_pem_output(
        cls,
        flow_kg_h: float = 100.0,
        pressure_bar: float = 30.0,
        temperature_c: float = 30.0,
        o2_crossover_ppm: float = 100.0
    ) -> 'ExternalH2Source':
        """
        Factory method to create source simulating PEM electrolyzer output.

        Args:
            flow_kg_h (float): H2 mass flow rate.
            pressure_bar (float): Stack output pressure.
            temperature_c (float): Stack output temperature.
            o2_crossover_ppm (float): O2 impurity in ppm.

        Returns:
            ExternalH2Source: Configured for PEM-like output.
        """
        o2_frac = o2_crossover_ppm / 1e6
        return cls(
            mass_flow_kg_h=flow_kg_h,
            pressure_bar=pressure_bar,
            temperature_c=temperature_c,
            h2_purity=1.0 - o2_frac - 0.0005,  # Rest is H2O moisture
            o2_impurity=o2_frac,
            h2o_impurity=0.0005
        )

    @classmethod
    def create_soec_output(
        cls,
        flow_kg_h: float = 100.0,
        pressure_bar: float = 1.5,
        temperature_c: float = 700.0
    ) -> 'ExternalH2Source':
        """
        Factory method to create source simulating SOEC electrolyzer output.

        Args:
            flow_kg_h (float): H2 mass flow rate.
            pressure_bar (float): Stack output pressure (low for SOEC).
            temperature_c (float): Stack output temperature (high-temp).

        Returns:
            ExternalH2Source: Configured for SOEC-like output.
        """
        return cls(
            mass_flow_kg_h=flow_kg_h,
            pressure_bar=pressure_bar,
            temperature_c=temperature_c,
            h2_purity=0.95,  # SOEC output has more steam
            h2o_impurity=0.05
        )
