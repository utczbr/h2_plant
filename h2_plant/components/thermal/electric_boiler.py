"""
Continuous Flow Electric Boiler Component.

Refactored from batch regime (modelo_boiler.py) to continuous flow regime
compatible with the H2 Plant Simulation Engine.

The component acts as a thermodynamic transformer:
- Input: Water stream + electricity
- Physics: Enthalpy balance (h_out = h_in + Q_net / m_dot)
- Output: Heated water stream
"""

import numpy as np
from typing import Dict, Any, Optional, TYPE_CHECKING

from h2_plant.core.component import Component
from h2_plant.core.stream import Stream
from h2_plant.optimization.numba_ops import calc_boiler_outlet_enthalpy

if TYPE_CHECKING:
    from h2_plant.core.component_registry import ComponentRegistry


class ElectricBoiler(Component):
    """
    Continuous flow electric boiler simulating thermodynamic heating of water streams.
    
    Implements energy balance: h_out = h_in + Q_net / m_dot.
    Uses LUTManager for property lookups instead of constant Cp.
    
    Adaptations from legacy batch model:
    - Old: t = m×Cp×ΔT/P (time to heat fixed mass)
    - New: h_out = h_in + P×η/m_dot (continuous flow enthalpy balance)
    
    Attributes:
        max_power_w (float): Maximum heating power (Watts)
        efficiency (float): Thermal efficiency (0.0 - 1.0)
        design_pressure_pa (float): Operational pressure limit (Pa)
        current_power_w (float): Applied power in current timestep (W)
        outlet_temp_k (float): Outlet temperature (K)
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs) -> None:
        """
        Initialize boiler configuration.
        
        Args:
            config (dict): Configuration containing:
                - max_power_kw (float): Maximum heating power in kW (default: 1000)
                - efficiency (float): Thermal efficiency 0.0-1.0 (default: 0.99)
                - design_pressure_bar (float): Operational pressure limit in bar (default: 10)
        
        Config -> SI Unit Conversions (Layer 3 Standard):
            - max_power_kw -> max_power_w (×1000)
            - design_pressure_bar -> design_pressure_pa (×1e5)
        """
        super().__init__(config, **kwargs)
        
        config = config or {}
        
        # Convert Configuration to SI Units (Layer 3 Standard)
        self.max_power_w = config.get('max_power_kw', 1000.0) * 1000.0
        self.efficiency = config.get('efficiency', 0.99)  # Default from legacy boiler model
        self.design_pressure_pa = config.get('design_pressure_bar', 10.0) * 1e5
        
        # State tracking
        self.current_power_w = 0.0
        self.outlet_temp_k = 298.15
        
        # Input buffers (populated by receive_input)
        self._input_stream: Optional[Stream] = None
        self._power_setpoint_w = 0.0
        
        # Performance history (pre-allocated during initialize)
        self.history_temp = np.zeros(1)
        self.step_idx = 0

    def initialize(self, dt: float, registry: 'ComponentRegistry') -> None:
        """
        Phase 1: Initialization and memory allocation.
        
        Pre-allocates arrays for history tracking based on expected simulation
        duration (24h default, resized dynamically if needed).
        
        Args:
            dt (float): Simulation timestep in hours
            registry (ComponentRegistry): Access to LUTManager and other components
        """
        super().initialize(dt, registry)
        
        # Access optimized lookup tables (optional, for future Cp lookup)
        self.lut = registry.get('lut_manager') if registry.has('lut_manager') else None
        
        # Pre-allocate arrays for history tracking (Optimization)
        # Assuming 24h simulation with dt=1/60 h (1 min) = 1440 steps
        # Adjust based on actual simulation duration
        steps = int(24 / dt) + 1
        self.history_temp = np.zeros(steps)
        self.step_idx = 0

    def step(self, t: float) -> None:
        """
        Phase 2: Execution of physics.
        
        Performs enthalpy balance to determine outlet temperature.
        Uses Numba-compiled function for core calculation.
        
        Physics:
            h_out = h_in + (Power × Efficiency) / MassFlow
        
        Args:
            t (float): Current simulation time in hours
        """
        super().step(t)

        # 1. Read Inputs (Layer 3 Flow Interface)
        inflow = self._input_stream
        power_setpoint = self._power_setpoint_w
        
        # 2. Apply Operational Limits
        # Why: Equipment protection and physical constraints
        applied_power_w = min(power_setpoint, self.max_power_w)
        
        # 3. Execute Physics (Thermodynamics)
        if inflow is None or inflow.mass_flow_kg_h <= 0:
            # No flow: Output is zero flow stream
            outflow = Stream(
                mass_flow_kg_h=0.0,
                temperature_k=298.15,
                pressure_pa=101325,
                composition={'H2O': 1.0},
                phase='liquid'
            )
            self.current_power_w = 0.0
        else:
            # Use Numba-compiled function for core calculation
            # Why: "Move hot loops to numba_ops.py" (Developer Guide Section 5)
            h_in = inflow.specific_enthalpy_j_kg
            
            h_out = calc_boiler_outlet_enthalpy(
                h_in_j_kg=h_in,
                mass_flow_kg_h=inflow.mass_flow_kg_h,
                power_input_w=applied_power_w,
                efficiency=self.efficiency
            )
            
            # Reconstruct Stream (State Update)
            # Note: Stream class converts h_out back to Temperature
            # using its internal property manager
            outflow = Stream(
                mass_flow_kg_h=inflow.mass_flow_kg_h,
                pressure_pa=inflow.pressure_pa,  # Isobaric heating assumption
                specific_enthalpy_j_kg=h_out,    # Driven by energy balance
                composition=inflow.composition,
                phase='liquid'  # Assumption for boiler (sub-boiling heating)
            )
            
            self.current_power_w = applied_power_w
            self.outlet_temp_k = outflow.temperature_k

        # 4. Store output for downstream
        self._output_stream = outflow
        
        # 5. Record History
        if self.step_idx < len(self.history_temp):
            self.history_temp[self.step_idx] = self.outlet_temp_k
            self.step_idx += 1
        
        # 6. Clear input buffers for next step
        self._input_stream = None
        self._power_setpoint_w = 0.0

    def receive_input(self, port_name: str, value: Any, resource_type: str = None) -> float:
        """
        Receive input into a specific port.
        
        Ports:
            - water_in: Accepts Stream object (water stream to heat)
            - power_in: Accepts float (electrical power setpoint in Watts)
        
        Args:
            port_name: Name of the input port
            value: Stream object (for water_in) or float (for power_in)
            resource_type: Type of resource ('water', 'electricity')
            
        Returns:
            Amount accepted (mass for water, power for electricity)
        """
        if port_name == 'water_in' and isinstance(value, Stream):
            self._input_stream = value
            return value.mass_flow_kg_h
        
        elif port_name == 'power_in':
            # Accept power setpoint (convert to W if needed)
            power_w = float(value)
            self._power_setpoint_w = power_w
            return power_w
        
        return super().receive_input(port_name, value, resource_type)

    def get_output(self, port_name: str = 'water_out') -> Optional[Stream]:
        """
        Get output from a specific port.
        
        Ports:
            - water_out: Returns heated water Stream object
        
        Args:
            port_name: Name of the output port (default: 'water_out')
            
        Returns:
            Stream object representing heated water output
        """
        if port_name == 'water_out':
            return getattr(self, '_output_stream', None)
        
        return None

    def get_state(self) -> Dict[str, Any]:
        """
        Phase 3: State Reporting.
        
        Returns JSON-serializable dictionary of component state for
        monitoring, checkpointing, and visualization.
        
        Returns:
            Dict containing power input, temperatures, and efficiency
        """
        return {
            **super().get_state(),
            "power_input_w": self.current_power_w,
            "power_input_kw": self.current_power_w / 1000.0,
            "power_input_mw": self.current_power_w / 1e6,
            "outlet_temperature_k": self.outlet_temp_k,
            "outlet_temperature_c": self.outlet_temp_k - 273.15,
            "efficiency_actual": self.efficiency,
            "max_power_kw": self.max_power_w / 1000.0
        }

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """
        Metadata for FlowNetwork topology.
        
        Returns:
            Dictionary describing input/output ports and their resource types
        """
        return {
            'water_in': {
                'type': 'input',
                'resource_type': 'water',
                'units': 'kg/h'
            },
            'power_in': {
                'type': 'input',
                'resource_type': 'electricity',
                'units': 'W'
            },
            'water_out': {
                'type': 'output',
                'resource_type': 'water',
                'units': 'kg/h'
            }
        }
