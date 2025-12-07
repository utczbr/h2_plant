
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import logging
from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.constants import GasConstants
from h2_plant.core.stream import Stream

logger = logging.getLogger(__name__)


class OxygenMixer(Component):
    """
    Multi-source oxygen mixing component.
    """
    

    def __init__(
        self,
        volume_m3: float = 1.0, # Changed from capacity_kg
        target_pressure_bar: float = 5.0,
        target_temperature_c: float = 25.0,
    ):
        super().__init__()
        
        self.volume_m3 = volume_m3 
        # Calculate capacity at target pressure (approximate for reporting)
        # Using Ideal Gas approximation for 'capacity' estimation: PV = mRT -> m = PV/RT
        # Or user provided density 1.429 kg/m3 at STP.
        # Let's use user formula: capacity = V * rho_stp * (P_target_bar / 1.01325)
        rho_o2_stp = 1.429 
        self.capacity_kg = self.volume_m3 * rho_o2_stp * (target_pressure_bar / 1.01325)
        self.target_pressure_pa = target_pressure_bar * 1e5
        self.target_pressure_bar = target_pressure_bar
        self.target_temperature_k = target_temperature_c + 273.15
        
        self.mass_kg = 0.0
        self.pressure_pa = 1e5
        self.temperature_k = 298.15
        
        # Outputs
        self.output_mass_kg = 0.0
        self.cumulative_input_kg = 0.0
        self.cumulative_output_kg = 0.0
        self.cumulative_vented_kg = 0.0
        
        # Buffer for push architecture
        self._input_buffer: List[Stream] = []
    
    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """Initialize mixer."""
        super().initialize(dt, registry)
    
    def receive_input(self, port_name: str, value: Any, resource_type: str = None) -> float:
        """
        Receive input stream from upstream component.
        
        Args:
            port_name: 'oxygen_in'
            value: Stream object
            resource_type: 'oxygen' (optional)
        """
        if port_name == 'oxygen_in' or port_name == 'inlet':
            if isinstance(value, Stream):
                if value.mass_flow_kg_h > 0:
                    self._input_buffer.append(value)
                return value.mass_flow_kg_h # Accept all
        return 0.0

    def step(self, t: float) -> None:
        """Execute timestep - mix oxygen from buffered inputs."""
        super().step(t)
        
        # 1. Aggregate all inputs from buffer
        combined_input_stream: Optional[Stream] = None
        
        for input_stream in self._input_buffer:
             if combined_input_stream is None:
                 combined_input_stream = input_stream
             else:
                 combined_input_stream = combined_input_stream.mix_with(input_stream)
        
        # Clear buffer for next step
        self._input_buffer = []

        mixed_stream = None # Initialize scope
        
        # 2. Mix combined input with stored mass
        if combined_input_stream is not None:
            input_mass = combined_input_stream.mass_flow_kg_h * self.dt
            
            if self.mass_kg > 0:
                # Create stream representing current storage
                stored_stream = Stream(
                    mass_flow_kg_h=self.mass_kg / self.dt, # Virtual flow
                    temperature_k=self.temperature_k,
                    pressure_pa=self.pressure_pa,
                    composition={'O2': 1.0} 
                )
                
                # Mix input + stored
                mixed_stream = stored_stream.mix_with(combined_input_stream)
                
                # Update state
                # self.temperature_k updated by Damping logic below
            else:
                # Tank empty, takes input state
                mixed_stream = combined_input_stream # Treat as mixed
                # self.temperature_k updated by Damping logic below
            
            # Update mass
            self.mass_kg += input_mass
            self.cumulative_input_kg += input_mass

            # 2a. Temperature Damping (Stability)
            # Tau = 5.0 seconds standard time constant
            tau = 5.0
            dt_sec = self.dt * 3600.0
            alpha = min(1.0, dt_sec / tau) 
            
            # mixed_stream is defined in lines above
            if mixed_stream is not None:
                target_T = mixed_stream.temperature_k
            else:
                 target_T = combined_input_stream.temperature_k
            
            if self.mass_kg > input_mass: # If we had significant mass before (mass added above)
                 self.temperature_k = alpha * target_T + (1 - alpha) * self.temperature_k
            else:
                 self.temperature_k = target_T

            # 2b. Pressure Upgrade (CoolProp) based on Density
            # P = f(T, Density)
            # Density = mass / volume
            if self.volume_m3 > 0:
                density_kg_m3 = self.mass_kg / self.volume_m3
                try:
                    import CoolProp.CoolProp as CP
                    # PropsSI('P', 'T', T_K, 'D', D_kg_m3, 'Oxygen')
                    self.pressure_pa = CP.PropsSI('P', 'T', self.temperature_k, 'D', density_kg_m3, 'Oxygen')
                except:
                    # Fallback to Ideal Gas if CoolProp fails or missing
                    moles_o2 = self.mass_kg * 1000.0 / GasConstants.SPECIES_DATA['O2']['molecular_weight']
                    self.pressure_pa = (moles_o2 * GasConstants.R_UNIVERSAL_J_PER_MOL_K * self.temperature_k) / self.volume_m3
        
        # 3. Handle overflow
        pass
    
    def get_state(self) -> Dict[str, Any]:
        return {
            **super().get_state(),
            'mass_kg': float(self.mass_kg),
            'capacity_kg': float(self.capacity_kg),
            'pressure_bar': float(self.pressure_pa / 1e5),
            'temperature_c': float(self.temperature_k - 273.15),
            'fill_percentage': float(self.mass_kg / self.capacity_kg * 100) if self.capacity_kg > 0 else 0.0
        }
    
    def remove_oxygen(self, mass_kg: float) -> float:
        removed = min(mass_kg, self.mass_kg)
        self.mass_kg -= removed
        self.output_mass_kg = removed
        self.cumulative_output_kg += removed
        return removed
    
    def get_ports(self) -> Dict[str, Dict[str, str]]:
        return {
            'oxygen_in': {'type': 'input', 'resource_type': 'oxygen', 'units': 'kg/h'},
            'oxygen_out': {'type': 'output', 'resource_type': 'oxygen', 'units': 'kg/h'}
        }

