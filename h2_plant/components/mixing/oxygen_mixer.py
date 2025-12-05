
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
        capacity_kg: float = 1000.0,
        target_pressure_bar: float = 5.0,
        target_temperature_c: float = 25.0,
        input_source_ids: Optional[List[str]] = None
    ):
        super().__init__()
        
        self.capacity_kg = capacity_kg
        self.target_pressure_pa = target_pressure_bar * 1e5
        self.target_pressure_bar = target_pressure_bar
        self.target_temperature_k = target_temperature_c + 273.15
        self.input_source_ids = input_source_ids or []
        
        self.mass_kg = 0.0
        self.pressure_pa = 1e5
        self.temperature_k = 298.15
        
        # Outputs
        self.output_mass_kg = 0.0
        self.cumulative_input_kg = 0.0
        self.cumulative_output_kg = 0.0
        self.cumulative_vented_kg = 0.0
        
        self._input_sources: List[Component] = []
    
    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """Initialize mixer and resolve input sources."""
        super().initialize(dt, registry)
        
        for source_id in self.input_source_ids:
            if registry.has(source_id):
                self._input_sources.append(registry.get(source_id))
            else:
                logger.warning(f"Input source '{source_id}' not found in registry")
    
    def step(self, t: float) -> None:
        """Execute timestep - mix oxygen from all sources."""
        super().step(t)
        
        # 1. Aggregate all inputs into a single input stream
        combined_input_stream: Optional[Stream] = None
        
        for source in self._input_sources:
            source_state = source.get_state()
            
            # Try to get stream from source
            input_stream = None
            
            # Check for o2_stream (Electrolyzer)
            if hasattr(source, 'o2_stream') and source.o2_stream is not None:
                input_stream = source.o2_stream
            # Check for generic output stream (Compressor, etc)
            elif hasattr(source, 'output_stream') and source.output_stream is not None:
                input_stream = source.output_stream
            
            # Fallback: Create stream from legacy outputs
            if input_stream is None:
                o2_output = self._extract_o2_output(source_state)
                if o2_output > 0:
                    input_stream = Stream(
                        mass_flow_kg_h=o2_output / self.dt,
                        temperature_k=self._extract_temperature(source_state),
                        pressure_pa=self._extract_pressure(source_state),
                        composition={'O2': 1.0}
                    )
            
            # Mix into combined stream
            if input_stream is not None and input_stream.mass_flow_kg_h > 0:
                if combined_input_stream is None:
                    combined_input_stream = input_stream
                else:
                    combined_input_stream = combined_input_stream.mix_with(input_stream)
        
        # 2. Mix combined input with stored mass
        if combined_input_stream is not None:
            input_mass = combined_input_stream.mass_flow_kg_h * self.dt
            
            # Create stream representing current storage
            # Treat current mass as a flow over 1 hour for mixing calc (simplified)
            # or better: use mix_with logic manually
            
            if self.mass_kg > 0:
                stored_stream = Stream(
                    mass_flow_kg_h=self.mass_kg / self.dt, # Virtual flow
                    temperature_k=self.temperature_k,
                    pressure_pa=self.pressure_pa,
                    composition={'O2': 1.0} # Assume pure O2 for now
                )
                
                # Mix input + stored
                mixed_stream = stored_stream.mix_with(combined_input_stream)
                
                # Update state
                self.temperature_k = mixed_stream.temperature_k
                # Pressure logic for tank is different (PV=nRT), not just mixing
                # But for now we keep the mixing temp
            else:
                # Tank empty, takes input state
                self.temperature_k = combined_input_stream.temperature_k
            
            # Update mass
            self.mass_kg += input_mass
            self.cumulative_input_kg += input_mass
            
            # Pressure update (Ideal Gas Law: P = mRT/V)
            # V = m_cap * R * T_target / P_target
            # So P = m * P_target / m_cap * (T / T_target)
            if self.capacity_kg > 0:
                # Simplified pressure scaling
                self.pressure_pa = (self.mass_kg / self.capacity_kg) * self.target_pressure_pa * (self.temperature_k / self.target_temperature_k)
        
        # 3. Handle overflow
        if self.mass_kg > self.capacity_kg:
            vented = self.mass_kg - self.capacity_kg
            self.mass_kg = self.capacity_kg
            self.cumulative_vented_kg += vented
            logger.warning(f"Oxygen mixer overflow: {vented:.2f} kg vented")
    
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
    
    def _extract_o2_output(self, state: Dict[str, Any]) -> float:
        if 'o2_output_kg' in state:
            return state['o2_output_kg']
        if 'flows' in state and 'outputs' in state['flows'] and 'oxygen' in state['flows']['outputs']:
            return state['flows']['outputs']['oxygen'].get('value', 0.0)
        return 0.0
    
    def _extract_temperature(self, state: Dict[str, Any]) -> float:
        return state.get('temperature_k', 298.15)
    
    def _extract_pressure(self, state: Dict[str, Any]) -> float:
        return state.get('pressure_pa', 1e5)
