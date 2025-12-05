"""
Thermodynamic water mixer with CoolProp-based calculations.

This component performs rigorous mass and energy balance for mixing
multiple water streams, preserving exact thermodynamic calculations
from the validated legacy Mixer.py implementation.
"""

from typing import Dict, Any, Optional
import logging

from h2_plant.core.component import Component
from h2_plant.core.stream import Stream
from h2_plant.core.component_registry import ComponentRegistry

# CoolProp import with graceful fallback
try:
    import CoolProp.CoolProp as CP
    COOLPROP_AVAILABLE = True
except ImportError:
    CP = None
    COOLPROP_AVAILABLE = False

logger = logging.getLogger(__name__)


class WaterMixer(Component):
    """
    Multi-inlet water mixer with thermodynamically rigorous calculations.
    
    Performs mass and energy balance using CoolProp for exact enthalpy
    calculations. Supports liquid water mixing with any number of inlet streams.
    
    Mass Balance: m_out = Σ m_in
    Energy Balance: h_out = Σ(m_in × h_in) / m_out
    
    Args:
        outlet_pressure_kpa: Output pressure in kPa (default: 200.0)
        fluid_type: Fluid for CoolProp (default: 'Water')
        max_inlet_streams: Maximum number of inlet streams (default: 10)
        
    Attributes:
        inlet_streams: Dictionary of inlet streams by port name
        outlet_stream: Mixed output stream
        outlet_pressure_pa: Output pressure in Pa
    """
    
    def __init__(
        self,
        outlet_pressure_kpa: float = 200.0,
        fluid_type: str = 'Water',
        max_inlet_streams: int = 10
    ):
        super().__init__()
        
        if not COOLPROP_AVAILABLE:
            logger.warning(
                "CoolProp not available. WaterMixer requires CoolProp for "
                "thermodynamic calculations. Install with: pip install CoolProp"
            )
        
        self.outlet_pressure_kpa = outlet_pressure_kpa
        self.outlet_pressure_pa = outlet_pressure_kpa * 1000.0
        self.fluid_type = fluid_type
        self.max_inlet_streams = max_inlet_streams
        
        # Dynamic inlet streams dictionary
        self.inlet_streams: Dict[str, Optional[Stream]] = {}
        
        # Output
        self.outlet_stream: Optional[Stream] = None
        
        # State tracking for monitoring
        self.last_mass_flow_kg_h = 0.0
        self.last_temperature_k = 0.0
        self.last_enthalpy_j_kg = 0.0
        
    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """Initialize mixer component."""
        super().initialize(dt, registry)
        logger.info(
            f"WaterMixer {self.component_id} initialized. "
            f"P_out={self.outlet_pressure_kpa} kPa"
        )
        
        if not COOLPROP_AVAILABLE:
            raise RuntimeError(
                f"WaterMixer {self.component_id} cannot initialize: "
                "CoolProp library not available"
            )
    
    def receive_input(self, port_name: str, value: Any, resource_type: str) -> float:
        """
        Receive input stream on a specific port.
        
        Args:
            port_name: Name of the inlet port
            value: Stream object
            resource_type: Type of resource (e.g., 'water')
            
        Returns:
            Accepted mass flow in kg/h
        """
        if not isinstance(value, Stream):
            logger.warning(
                f"{self.component_id}: Expected Stream object, got {type(value)}"
            )
            return 0.0
        
        if len(self.inlet_streams) < self.max_inlet_streams:
            self.inlet_streams[port_name] = value
            return value.mass_flow_kg_h
        else:
            logger.warning(
                f"{self.component_id}: Max inlet streams ({self.max_inlet_streams}) "
                f"reached. Rejecting stream on port '{port_name}'"
            )
            return 0.0
    
    def get_output(self, port_name: str) -> Any:
        """
        Get output stream from mixer.
        
        Args:
            port_name: Name of output port (typically 'outlet')
            
        Returns:
            Mixed output Stream or None if no valid output
        """
        if port_name in ('outlet', 'mixed_out', 'output'):
            return self.outlet_stream
        
        logger.warning(f"{self.component_id}: Unknown output port '{port_name}'")
        return None
    
    def step(self, t: float) -> None:
        """
        Execute mixing calculations for current timestep.
        
        Performs mass and energy balance using exact CoolProp thermodynamics,
        replicating the validated Mixer.py calculations.
        
        Args:
            t: Current simulation time in hours
        """
        super().step(t)
        
        # Get active inlet streams
        active_streams = [s for s in self.inlet_streams.values() if s is not None]
        
        if not active_streams:
            self.outlet_stream = None
            self.last_mass_flow_kg_h = 0.0
            self.last_temperature_k = 0.0
            self.last_enthalpy_j_kg = 0.0
            return
        
        # --- EXACT MIXER.PY LOGIC ---
        # Variables matching Mixer.py implementation
        total_energy_in = 0.0  # kJ/s
        total_mass_in = 0.0    # kg/s
        
        # Process each inlet stream
        for stream in active_streams:
            # Convert from architecture units to Mixer.py units
            # Stream: kg/h, K, Pa
            # Mixer.py: kg/s, °C, kPa
            m_dot_i = stream.mass_flow_kg_h / 3600.0  # kg/s
            T_i_C = stream.temperature_k - 273.15     # °C
            P_i_kPa = stream.pressure_pa / 1000.0     # kPa
            
            # Convert to CoolProp units (K, Pa) - exactly as Mixer.py does
            T_i_K = T_i_C + 273.15
            P_i_Pa = P_i_kPa * 1000.0
            
            try:
                # Calculate enthalpy using CoolProp
                # Returns J/kg, convert to kJ/kg
                h_i_kJ_kg = CP.PropsSI('H', 'T', T_i_K, 'P', P_i_Pa, self.fluid_type) / 1000.0
                
                # Mass balance: sum of mass flow rates
                total_mass_in += m_dot_i
                
                # Energy balance: sum of (m_dot * h)
                energy_in_i = m_dot_i * h_i_kJ_kg
                total_energy_in += energy_in_i
                
            except Exception as e:
                logger.error(
                    f"{self.component_id}: CoolProp error for stream at "
                    f"T={T_i_C:.2f}°C, P={P_i_kPa:.2f}kPa: {e}"
                )
                self.outlet_stream = None
                return
        
        # Check mass balance
        if total_mass_in <= 0:
            self.outlet_stream = None
            return
        
        # Calculate output enthalpy
        h_out_kJ_kg = total_energy_in / total_mass_in
        
        # Find output temperature from h_out and P_out
        h_out_J_kg = h_out_kJ_kg * 1000.0
        P_out_Pa = self.outlet_pressure_pa
        
        try:
            # Get temperature from enthalpy and pressure
            T_out_K = CP.PropsSI('T', 'H', h_out_J_kg, 'P', P_out_Pa, self.fluid_type)
            
        except Exception as e:
            logger.error(
                f"{self.component_id}: CoolProp error calculating output temperature "
                f"for h={h_out_kJ_kg:.2f}kJ/kg, P={self.outlet_pressure_kpa:.2f}kPa: {e}"
            )
            self.outlet_stream = None
            return
        
        # Store state for monitoring
        self.last_mass_flow_kg_h = total_mass_in * 3600.0  # Convert back to kg/h
        self.last_temperature_k = T_out_K
        self.last_enthalpy_j_kg = h_out_J_kg
        
        # Create output stream (in architecture units: kg/h, K, Pa)
        self.outlet_stream = Stream(
            mass_flow_kg_h=total_mass_in * 3600.0,
            temperature_k=T_out_K,
            pressure_pa=P_out_Pa,
            composition={'H2O': 1.0},  # Pure water
            phase='liquid'
        )
    
    def get_state(self) -> Dict[str, Any]:
        """
        Return current state for monitoring and checkpointing.
        
        Returns:
            Dictionary with component state including flows and thermodynamic properties
        """
        num_inlets = len([s for s in self.inlet_streams.values() if s is not None])
        
        return {
            **super().get_state(),
            'outlet_pressure_kpa': float(self.outlet_pressure_kpa),
            'fluid_type': self.fluid_type,
            'num_active_inlets': num_inlets,
            'max_inlets': self.max_inlet_streams,
            'outlet_mass_flow_kg_h': float(self.last_mass_flow_kg_h),
            'outlet_temperature_k': float(self.last_temperature_k),
            'outlet_temperature_c': float(self.last_temperature_k - 273.15),
            'outlet_enthalpy_j_kg': float(self.last_enthalpy_j_kg),
            'outlet_enthalpy_kj_kg': float(self.last_enthalpy_j_kg / 1000.0),
        }
    
    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """
        Get information about available ports.
        
        Returns:
            Dictionary of port metadata
        """
        ports = {
            'outlet': {
                'type': 'output',
                'resource_type': 'water',
                'units': 'kg/h',
                'phase': 'liquid'
            }
        }
        
        # Add dynamic inlet ports
        for i in range(self.max_inlet_streams):
            port_name = f'inlet_{i}'
            ports[port_name] = {
                'type': 'input',
                'resource_type': 'water',
                'units': 'kg/h',
                'phase': 'liquid'
            }
        
        return ports
