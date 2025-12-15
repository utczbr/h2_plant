"""
External Oxygen Source Component.

This module provides oxygen supply from external sources (pipeline, liquid
storage, or on-site ASU) for processes requiring supplemental O₂ beyond
electrolysis byproduct availability.

Supply Modes:
    - **Fixed Flow**: Constant delivery rate regardless of downstream demand.
      Suitable for contracted supply or pipeline delivery.
    - **Pressure Driven**: Modulates flow to maintain target pressure in
      downstream buffer. Suitable for on-demand supply with storage.

Architecture:
    Implements the Component Lifecycle Contract (Layer 1):
    - `initialize()`: Connects to downstream targets (mixer or buffer).
    - `step()`: Delivers oxygen according to configured mode.
    - `get_state()`: Returns flow, cost, and connection metrics.

Economic Considerations:
    External O₂ purchase costs are tracked for techno-economic analysis.
    Typical industrial oxygen costs range from 0.05-0.30 EUR/kg depending
    on purity and delivery method.
"""

import logging
from typing import Dict, Any, Optional, List

from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.constants import GasConstants
from h2_plant.core.stream import Stream

logger = logging.getLogger(__name__)


class ExternalOxygenSource(Component):
    """
    Configurable oxygen source for testing oxygen-dependent components.

    This component acts as an "infinite source" of oxygen with fully
    configurable thermodynamic properties. Designed for component testing
    without requiring complete electrolyzer topology or for supplying backup O2.

    Configuration Options:
        - **Output Mode**: 'fixed_flow', 'on_demand', 'time_profile'
        - **Thermodynamic State**: Temperature (K), Pressure (Pa/bar)
        - **Composition**: O2 purity with optional impurities
        - **Phase**: 'gas' or 'liquid'

    Attributes:
        mode (str): Operating mode.
        mass_flow_kg_h (float): Oxygen mass flow rate (kg/h).
        temperature_k (float): Stream temperature (K).
        pressure_pa (float): Stream pressure (Pa).
        composition (dict): Species mole fractions.

    Example:
        >>> o2_source = ExternalOxygenSource(
        ...     mass_flow_kg_h=50.0,
        ...     pressure_bar=10.0,
        ...     o2_purity=0.995
        ... )
        >>> stream = o2_source.get_output('o2_out')
    """

    # Default thermodynamic properties for O2 source
    DEFAULTS = {
        'mode': 'fixed_flow',
        'mass_flow_kg_h': 850.0,  # ~8:1 ration with H2? No, mass ratio is 8:1 but here just default.
        'temperature_k': 333.15,  # 60°C
        'pressure_bar': 40.0,
        'o2_purity': 0.99,
        'h2o_impurity': 0.01,
        'h2_impurity': 0.0,
        'phase': 'gas',
        'enthalpy_kj_kg': None,
    }

    def __init__(
        self,
        component_id: str = "o2_source",
        mode: str = None,
        mass_flow_kg_h: float = None,
        temperature_k: float = None,
        temperature_c: float = None,
        pressure_pa: float = None,
        pressure_bar: float = None,
        o2_purity: float = None,
        h2o_impurity: float = None,
        h2_impurity: float = None,
        composition: Dict[str, float] = None,
        phase: str = None,
        enthalpy_kj_kg: float = None,
        time_profile: List[float] = None,
        config: Dict[str, Any] = None
    ):
        """
        Initialize the configurable oxygen source.
        
        Args:
            component_id (str): Unique identifier.
            mode (str): 'fixed_flow', 'on_demand', 'time_profile'.
            mass_flow_kg_h (float): O2 mass flow rate (kg/h).
            temperature_k (float): Stream temperature (K).
            temperature_c (float): Stream temperature (C).
            pressure_pa (float): Stream pressure (Pa).
            pressure_bar (float): Stream pressure (bar).
            o2_purity (float): O2 mole fraction.
            h2o_impurity (float): H2O mole fraction.
            h2_impurity (float): H2 mole fraction (crossover).
            composition (dict): Full composition override.
            phase (str): 'gas' or 'liquid'.
            enthalpy_kj_kg (float): Enthalpy override.
            time_profile (list): Flow profile.
            config (dict): Config dict.
        """
        super().__init__()
        self.component_id = component_id

        if config is not None or (isinstance(component_id, dict)):
            if isinstance(component_id, dict):
                config = component_id
                self.component_id = config.get('component_id', 'o2_source')
            self._apply_config(config)
        else:
            self.mode = mode or self.DEFAULTS['mode']
            self.mass_flow_kg_h = mass_flow_kg_h or self.DEFAULTS['mass_flow_kg_h']
            
            # Temperature
            if temperature_k is not None:
                self.temperature_k = temperature_k
            elif temperature_c is not None:
                self.temperature_k = temperature_c + 273.15
            else:
                self.temperature_k = self.DEFAULTS['temperature_k']
            
            # Pressure
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
                o2 = o2_purity if o2_purity is not None else self.DEFAULTS['o2_purity']
                h2o = h2o_impurity if h2o_impurity is not None else self.DEFAULTS['h2o_impurity']
                h2 = h2_impurity if h2_impurity is not None else self.DEFAULTS['h2_impurity']
                self.composition = {'O2': o2, 'H2O': h2o, 'H2': h2}
            
            self.phase = phase or self.DEFAULTS['phase']
            self.enthalpy_kj_kg = enthalpy_kj_kg
            self.time_profile = time_profile

        # State variables
        self.o2_output_kg = 0.0
        self.cumulative_o2_kg = 0.0
        self.current_step_index = 0

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
        
        # Composition
        if 'composition' in config:
            self.composition = config['composition']
        else:
            o2 = float(config.get('o2_purity', self.DEFAULTS['o2_purity']))
            h2o = float(config.get('h2o_impurity', self.DEFAULTS['h2o_impurity']))
            h2 = float(config.get('h2_impurity', self.DEFAULTS['h2_impurity']))
            self.composition = {'O2': o2, 'H2O': h2o, 'H2': h2}
        
        self.phase = config.get('phase', self.DEFAULTS['phase'])
        self.enthalpy_kj_kg = config.get('enthalpy_kj_kg', None)
        self.time_profile = config.get('time_profile', None)

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """Prepare the component for simulation execution."""
        super().initialize(dt, registry)
        self.current_step_index = 0

    def step(self, t: float) -> None:
        """Execute one simulation timestep."""
        super().step(t)

        # Determine flow based on mode
        if self.mode == 'time_profile' and self.time_profile:
            idx = self.current_step_index % len(self.time_profile)
            current_flow = float(self.time_profile[idx])
            self.current_step_index += 1
        elif self.mode == 'on_demand':
            current_flow = getattr(self, '_requested_flow_kg_h', self.mass_flow_kg_h)
        else:
            current_flow = self.mass_flow_kg_h

        self.o2_output_kg = current_flow * self.dt
        self.cumulative_o2_kg += self.o2_output_kg

    def get_output(self, port_name: str) -> Any:
        """
        Retrieve the output stream from a specified port.
        
        Args:
             port_name (str): Identifier. 'o2_out' is the primary output.
             
        Returns:
             Stream: Output stream with configured properties.
        """
        if port_name in ['o2_out', 'outlet']:
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
            
            if self.enthalpy_kj_kg is not None:
                stream.enthalpy_kj_kg = self.enthalpy_kj_kg
                
            return stream
        else:
            raise ValueError(f"Unknown output port '{port_name}' on {self.component_id}")

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """Define the physical connection ports."""
        return {
            'o2_out': {
                'type': 'output',
                'resource_type': 'oxygen',
                'units': 'kg/h',
                'description': 'Oxygen stream with configurable properties'
            },
            'flow_request': {
                'type': 'input',
                'resource_type': 'control',
                'description': 'Request specific flow rate'
            }
        }

    def receive_input(self, port_name: str, value: Any, resource_type: str = None) -> float:
        if port_name == 'flow_request':
            if isinstance(value, (int, float)):
                self._requested_flow_kg_h = float(value)
                return float(value)
        return 0.0

    def get_state(self) -> Dict[str, Any]:
        """Retrieve the component's current operational state."""
        return {
            **super().get_state(),
            'mode': self.mode,
            'mass_flow_kg_h': self.mass_flow_kg_h,
            'temperature_k': self.temperature_k,
            'pressure_bar': self.pressure_pa / 1e5,
            'o2_output_kg': self.o2_output_kg,
            'cumulative_o2_kg': self.cumulative_o2_kg
        }
