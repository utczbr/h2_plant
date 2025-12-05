"""
Chiller/Heat Exchanger component for thermal management.

Used for cooling streams in PEM/SOEC electrolysis systems.
"""

from typing import Dict, Any
from h2_plant.core.component import Component
from h2_plant.core.stream import Stream
from h2_plant.models.thermal_inertia import ThermalInertiaModel
from h2_plant.models.flow_dynamics import PumpFlowDynamics
import numpy as np


class Chiller(Component):
    """
    Heat exchanger for cooling fluid streams.
    
    Implements thermal energy balance:
    Q = m_fluid * Cp * (T_in - T_out)
    
    Used in Process Flow as HX-1, HX-2, HX-3, HX-5, HX-6, HX-10, HX-11.
    """
    
    def __init__(
        self,
        component_id: str = "chiller",
        cooling_capacity_kw: float = 100.0,
        efficiency: float = 0.95,
        target_temp_k: float = 298.15  # 25°C default
    ):
        """
        Initialize Chiller.
        
        Args:
            component_id: Unique identifier
            cooling_capacity_kw: Maximum cooling capacity in kW
            efficiency: Heat transfer efficiency (0-1)
            target_temp_k: Target outlet temperature in Kelvin
        """
        super().__init__()
        self.component_id = component_id
        self.cooling_capacity_kw = cooling_capacity_kw
        self.efficiency = efficiency
        self.target_temp_k = target_temp_k
        
        # State variables
        self.inlet_stream: Stream = Stream(0.0)
        self.outlet_stream: Stream = Stream(0.0)
        self.cooling_load_kw: float = 0.0
        self.cooling_water_flow_kg_h: float = 0.0
        self.heat_rejected_kw: float = 0.0
        
        # Dynamics Models
        self.pump = PumpFlowDynamics(
            initial_flow_m3_h=0.0,
            fluid_inertance_kg_m4=1e9 # Stable default
        )
        
        self.coolant_thermal = ThermalInertiaModel(
            C_thermal_J_K=1.0e6,
            h_A_passive_W_K=50.0,
            T_initial_K=293.15,
            max_cooling_kw=cooling_capacity_kw
        )
        
    def initialize(self, dt: float, registry: Any) -> None:
        """Initialize component."""
        super().initialize(dt, registry)
        self.initialized = True
        
    def step(self, t: float) -> None:
        """
        Execute one timestep of chiller operation.
        
        Args:
            t: Current simulation time in hours
        """
        super().step(t)
        
        # If no inlet flow, idle state
        if self.inlet_stream.mass_flow_kg_h <= 0:
            self.outlet_stream = Stream(0.0)
            self.cooling_load_kw = 0.0
            self.cooling_water_flow_kg_h = 0.0
            self.heat_rejected_kw = 0.0
            return
        
        # Calculate required cooling
        # Q = m * Cp * ΔT (simplified, using Cp_water ≈ 4.18 kJ/kg·K)
        Cp = 4.18  # kJ/kg·K
        mass_flow_kg_s = self.inlet_stream.mass_flow_kg_h / 3600.0
        temp_delta_k = max(0, self.inlet_stream.temperature_k - self.target_temp_k)
        
        required_cooling_kw = mass_flow_kg_s * Cp * temp_delta_k
        
        # Apply capacity and efficiency limits
        actual_cooling_kw = min(required_cooling_kw, self.cooling_capacity_kw) * self.efficiency
        
        # Calculate outlet temperature
        if mass_flow_kg_s > 0:
            actual_temp_drop = actual_cooling_kw / (mass_flow_kg_s * Cp)
            outlet_temp = self.inlet_stream.temperature_k - actual_temp_drop
        else:
            outlet_temp = self.inlet_stream.temperature_k
        
        # Update outlet stream
        self.outlet_stream = Stream(
            mass_flow_kg_h=self.inlet_stream.mass_flow_kg_h,
            temperature_k=outlet_temp,
            pressure_pa=self.inlet_stream.pressure_pa,
            composition=self.inlet_stream.composition.copy()
        )
        
        # Update state
        # Dynamics Update
        dt_seconds = self.dt * 3600.0
        
        # 1. Pump Control (Proportional to error)
        temp_error = self.inlet_stream.temperature_k - self.target_temp_k
        pump_speed = np.clip(temp_error / 30.0, 0, 1) if temp_error > 0 else 0.0
        
        # 2. Advance Pump
        Q_cool_m3_h = self.pump.step(dt_s=dt_seconds, pump_speed_fraction=pump_speed)
        
        # 3. Advance Coolant Thermal
        # Heat absorbed = actual cooling provided
        Q_absorbed_W = actual_cooling_kw * 1000.0
        T_coolant_K = self.coolant_thermal.step(
            dt_s=dt_seconds,
            heat_generated_W=Q_absorbed_W,
            T_control_K=self.target_temp_k
        )
        
        # Update state with dynamic values
        self.cooling_load_kw = actual_cooling_kw
        self.heat_rejected_kw = actual_cooling_kw / self.efficiency
        
        # Estimate cooling water flow (ΔT_cooling_water ≈ 10K)
        cooling_water_temp_rise = 10.0
        self.cooling_water_flow_kg_h = (self.heat_rejected_kw / (Cp * cooling_water_temp_rise / 3600.0))
    
    def get_output(self, port_name: str) -> Any:
        """Get output from specified port."""
        if port_name == "fluid_out":
            return self.outlet_stream
        elif port_name == "heat_out":
            return self.heat_rejected_kw
        elif port_name == "cooling_water_out":
            # Return heated cooling water stream
            return Stream(
                mass_flow_kg_h=self.cooling_water_flow_kg_h,
                temperature_k=298.15 + 10.0,  # Heated by ~10K
                pressure_pa=101325.0
            )
        return 0.0
    
    def receive_input(self, port_name: str, value: Any, resource_type: str = None) -> float:
        """Receive input at specified port."""
        if port_name == "fluid_in" and isinstance(value, Stream):
            self.inlet_stream = value
            return value.mass_flow_kg_h
        elif port_name == "cooling_water_in" and isinstance(value, Stream):
            # Cooling water is provided externally, just accept it
            return value.mass_flow_kg_h
        return 0.0
    
    def extract_output(self, port_name: str, amount: float, resource_type: str = None) -> None:
        """Acknowledge extraction of output."""
        pass  # Chiller doesn't store fluid, just passes through
    
    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """Return port definitions."""
        return {
            'fluid_in': {'type': 'input', 'resource_type': 'stream'},
            'cooling_water_in': {'type': 'input', 'resource_type': 'water'},
            'fluid_out': {'type': 'output', 'resource_type': 'stream'},
            'cooling_water_out': {'type': 'output', 'resource_type': 'water'},
            'heat_out': {'type': 'output', 'resource_type': 'heat'}
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Return current component state."""
        return {
            **super().get_state(),
            'component_id': self.component_id,
            'cooling_load_kw': self.cooling_load_kw,
            'outlet_temp_k': self.outlet_stream.temperature_k,
            'heat_rejected_kw': self.heat_rejected_kw,
            'cooling_water_flow_kg_h': self.cooling_water_flow_kg_h
        }
