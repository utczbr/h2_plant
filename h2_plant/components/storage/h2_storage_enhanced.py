"""
Enhanced Hydrogen Storage Tank with Pressure Dynamics
"""

from typing import Dict, Any, Optional
from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.stream import Stream
from h2_plant.models.flow_dynamics import GasAccumulatorDynamics

class H2StorageTankEnhanced(Component):
    """
    Storage tank with gas accumulator dynamics.
    Pressure evolves based on dP/dt = (RT/V) * (m_in - m_out).
    """
    
    def __init__(
        self,
        tank_id: str,
        volume_m3: float = 10.0,
        initial_pressure_bar: float = 40.0,
        max_pressure_bar: float = 350.0
    ):
        super().__init__()
        self.tank_id = tank_id
        self.volume_m3 = volume_m3
        self.max_pressure_bar = max_pressure_bar
        
        # Dynamics Model
        self.accumulator = GasAccumulatorDynamics(
            V_tank_m3=volume_m3,
            initial_pressure_pa=initial_pressure_bar * 1e5,
            T_tank_k=298.15
        )
        
        # Flow tracking
        self.m_dot_in_kg_s = 0.0
        self.m_dot_out_kg_s = 0.0
        
        # State
        self.pressure_bar = initial_pressure_bar
        self.mass_kg = self.accumulator.M_kg
        self.fill_fraction = 0.0
        
    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        super().initialize(dt, registry)
        
    def step(self, t: float) -> None:
        super().step(t)
        dt_seconds = self.dt * 3600.0
        
        # Advance accumulator
        P_new_pa = self.accumulator.step(
            dt_s=dt_seconds,
            m_dot_in_kg_s=self.m_dot_in_kg_s,
            m_dot_out_kg_s=self.m_dot_out_kg_s
        )
        
        # Update state
        self.pressure_bar = P_new_pa / 1e5
        self.mass_kg = self.accumulator.M_kg
        
        # Calculate fill fraction (assuming max pressure at 25C defines capacity)
        max_mass = (self.max_pressure_bar * 1e5 * self.volume_m3) / (self.accumulator.R * self.accumulator.T)
        self.fill_fraction = self.mass_kg / max_mass if max_mass > 0 else 0.0
        
        # Reset flows for next step (they are set by receive_input/extract_output)
        self.m_dot_in_kg_s = 0.0
        self.m_dot_out_kg_s = 0.0
        
    def receive_input(self, port_name: str, value: Any, resource_type: str) -> float:
        if port_name == 'h2_in' and isinstance(value, Stream):
            # Convert kg/h to kg/s
            flow_kg_s = value.mass_flow_kg_h / 3600.0
            self.m_dot_in_kg_s += flow_kg_s
            return value.mass_flow_kg_h
        return 0.0
        
    def extract_output(self, port_name: str, amount: float, resource_type: str) -> None:
        if port_name == 'h2_out':
            # amount is usually in kg/h (requested)
            # We need to convert to kg/s
            flow_kg_s = amount / 3600.0
            self.m_dot_out_kg_s += flow_kg_s
            
    def get_output(self, port_name: str) -> Any:
        if port_name == 'h2_out':
            # Return stream with current pressure
            # Flow rate is determined by demand (extract_output), but we can return potential?
            # Usually get_output returns what is available.
            # For storage, we might return a stream with 0 flow but correct pressure, 
            # and let the consumer request amount via extract_output?
            # Or return max available?
            return Stream(
                mass_flow_kg_h=0.0, # Consumer sets flow
                temperature_k=self.accumulator.T,
                pressure_pa=self.accumulator.P,
                composition={'H2': 1.0},
                phase='gas'
            )
        return None

    def get_state(self) -> Dict[str, Any]:
        return {
            **super().get_state(),
            'pressure_bar': self.pressure_bar,
            'mass_kg': self.mass_kg,
            'fill_fraction': self.fill_fraction,
            'flow_in_kg_s': self.m_dot_in_kg_s,
            'flow_out_kg_s': self.m_dot_out_kg_s
        }

    # --- Unified Storage Interface ---
    
    def get_inventory_kg(self) -> float:
        """Returns total stored hydrogen mass (Unified Interface)."""
        return self.mass_kg
        
    def withdraw_kg(self, amount: float) -> float:
        """
        Withdraws amount from storage immediately (Unified Interface).
        NOTE: This bypasses the m_dot_out rate logic used in `step`. 
        This is a bridge for the Orchestrator's push-sweep.
        """
        available = self.mass_kg
        actual = min(amount, available)
        self.mass_kg -= actual
        
        # Also sync the accumulator mass to keep physics consistent
        self.accumulator.M_kg = self.mass_kg
        # Update pressure instantly based on new mass (isochoric)
        # P = mRT/V
        self.accumulator.P = (self.accumulator.M_kg * self.accumulator.R * self.accumulator.T) / self.accumulator.V
        self.pressure_bar = self.accumulator.P / 1e5
        
        return actual
