"""
Stream Splitter Component.

This module implements a process component that divides a single input stream
into two output streams based on a defined split ratio, maintaining 
thermodynamic equilibrium (Isobaric/Isothermal split).
"""

from typing import Dict, Any, Optional
from h2_plant.core.component import Component
from h2_plant.core.stream import Stream

class StreamSplitter(Component):
    """
    Separates an input stream into two distinct streams with defined flow ratios.
    
    Physics:
        - Mass Balance: m_in = m_out1 + m_out2
        - Energy Balance: h_in = h_out1 = h_out2 (Adiabatic split)
        - Momentum: P_in = P_out1 = P_out2 (Negligible pressure drop)
        
    Configuration:
        split_ratio (float): Fraction of mass flow directed to outlet_1 (0.0 to 1.0).
                             Remaining flow (1 - ratio) goes to outlet_2.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs) -> None:
        """
        Initialize the StreamSplitter.

        Args:
            config: Configuration dictionary.
            **kwargs: Legacy arguments (component_id, etc.)
        """
        super().__init__(config, **kwargs)
        
        # Determine split ratio from config or kwargs, default to 50/50
        self.split_ratio = 0.5
        if config:
            self.split_ratio = float(config.get('split_ratio', 0.5))
        elif 'split_ratio' in kwargs:
            self.split_ratio = float(kwargs['split_ratio'])
            
        # Validate ratio
        if not (0.0 <= self.split_ratio <= 1.0):
            raise ValueError(f"StreamSplitter {self.component_id}: split_ratio must be between 0.0 and 1.0")

        # Internal buffers
        self.input_buffer: Dict[str, Stream] = {}
        self.output_buffer: Dict[str, Stream] = {}

    def initialize(self, dt: float, registry: Any) -> None:
        """
        Initialize component state.
        
        Args:
            dt: Timestep in hours.
            registry: Component registry.
        """
        super().initialize(dt, registry)
        self.input_buffer = {}
        self.output_buffer = {}

    def receive_input(self, port_name: str, value: Any, resource_type: str = None) -> float:
        """
        Receive material stream from upstream.
        
        Args:
            port_name: Name of the input port (expects 'inlet').
            value: The Stream object.
            resource_type: Resource identifier.
            
        Returns:
            float: Amount accepted (mass in kg).
        """
        if port_name == 'inlet' and isinstance(value, Stream):
            self.input_buffer['inlet'] = value
            return value.mass_flow_kg_h
            
        # Also accept control signals for dynamic split ratio adjustment
        if port_name == 'split_ratio_setpoint':
            try:
                new_ratio = float(value)
                if 0.0 <= new_ratio <= 1.0:
                    self.split_ratio = new_ratio
                return 1.0
            except (ValueError, TypeError):
                pass
                
        return super().receive_input(port_name, value, resource_type)

    def step(self, t: float) -> None:
        """
        Execute splitting logic for the current timestep.
        
        Args:
            t: Current simulation time in hours.
        """
        super().step(t)

        # 1. Retrieve Input
        in_stream = self.input_buffer.get('inlet')

        # 2. Guard Clause: Handle zero flow / no input
        if not in_stream or in_stream.mass_flow_kg_h <= 1e-9:
            empty_stream = Stream(mass_flow_kg_h=0.0)
            self.output_buffer['outlet_1'] = empty_stream
            self.output_buffer['outlet_2'] = empty_stream
            return

        # 3. Calculate Mass Split
        m_total = in_stream.mass_flow_kg_h
        m_1 = m_total * self.split_ratio
        m_2 = m_total * (1.0 - self.split_ratio)

        # 4. Create Output Streams
        # We copy the inlet stream to preserve P, T, and Composition
        # then explicitly update the mass flow.
        
        # Outlet 1 (Primary)
        stream_1 = in_stream.copy()
        stream_1.mass_flow_kg_h = m_1
        
        # Outlet 2 (Secondary)
        stream_2 = in_stream.copy()
        stream_2.mass_flow_kg_h = m_2

        # 5. Push to Output Buffer
        self.output_buffer['outlet_1'] = stream_1
        self.output_buffer['outlet_2'] = stream_2
        
        # Clear input buffer for next step
        self.input_buffer = {}

    def get_output(self, port_name: str) -> Optional[Stream]:
        """
        Retrieve output stream by port name.
        
        Args:
            port_name: 'outlet_1' or 'outlet_2'.
        """
        return self.output_buffer.get(port_name)

    def get_state(self) -> Dict[str, Any]:
        """
        Return component state for monitoring.
        """
        # Get current flows safely
        s1 = self.output_buffer.get('outlet_1')
        s2 = self.output_buffer.get('outlet_2')
        
        return {
            **super().get_state(),
            "split_ratio": self.split_ratio,
            "flow_in_kg_h": (s1.mass_flow_kg_h + s2.mass_flow_kg_h) if s1 and s2 else 0.0,
            "flow_out1_kg_h": s1.mass_flow_kg_h if s1 else 0.0,
            "flow_out2_kg_h": s2.mass_flow_kg_h if s2 else 0.0,
            "temperature_k": s1.temperature_k if s1 else 298.15,
            "pressure_pa": s1.pressure_pa if s1 else 101325.0
        }

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """Declare ports for the FlowNetwork."""
        return {
            'inlet': {'type': 'input', 'resource_type': 'stream'},
            'split_ratio_setpoint': {'type': 'input', 'resource_type': 'signal'},
            'outlet_1': {'type': 'output', 'resource_type': 'stream'},
            'outlet_2': {'type': 'output', 'resource_type': 'stream'}
        }
