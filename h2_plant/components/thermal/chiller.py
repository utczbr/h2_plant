"""
Chiller/Heat Exchanger component for thermal management.

Used for cooling streams in PEM/SOEC electrolysis systems.
Aligned with reference model (modelo_chiller.py) for accuracy.
"""

from typing import Dict, Any, Optional
from h2_plant.core.component import Component
from h2_plant.core.stream import Stream
from h2_plant.core.constants import GasConstants
from h2_plant.models.thermal_inertia import ThermalInertiaModel
from h2_plant.models.flow_dynamics import PumpFlowDynamics
import numpy as np
import logging


class Chiller(Component):
    """
    Heat exchanger for cooling fluid streams.
    
    Implements enthalpy-based thermal energy balance:
    Q = m_dot * (h_in - h_out)
    
    With gas-specific Cp fallback for H2/O2 streams.
    Electrical power: W = |Q| / COP
    Pressure drop: P_out = P_in - ΔP
    
    Used in Process Flow as HX-1, HX-2, HX-3, HX-5, HX-6, HX-10, HX-11.
    """
    
    def __init__(
        self,
        component_id: str = "chiller",
        cooling_capacity_kw: float = 100.0,
        efficiency: float = 0.95,
        target_temp_k: float = 298.15,  # 25°C default
        cop: float = 4.0,               # Coefficient of Performance
        pressure_drop_bar: float = 0.2,  # Pressure drop across chiller
        enable_dynamics: bool = False    # Off by default for reference matching
    ):
        """
        Initialize Chiller.
        
        Args:
            component_id: Unique identifier
            cooling_capacity_kw: Maximum cooling capacity in kW
            efficiency: Heat transfer efficiency (0-1)
            target_temp_k: Target outlet temperature in Kelvin
            cop: Coefficient of Performance for electrical consumption
            pressure_drop_bar: Pressure drop across the heat exchanger (bar)
            enable_dynamics: Enable pump/thermal dynamics (False for steady-state)
        """
        super().__init__()
        self.component_id = component_id
        self.cooling_capacity_kw = cooling_capacity_kw
        self.efficiency = efficiency
        self.target_temp_k = target_temp_k
        self.cop = cop
        self.pressure_drop_bar = pressure_drop_bar
        self.enable_dynamics = enable_dynamics
        
        # Logger for fallback warnings
        self.logger = logging.getLogger(f"chiller.{component_id}")
        
        # State variables
        self.inlet_stream: Stream = Stream(0.0)
        self.cooling_water_inlet: Optional[Stream] = None
        self.outlet_stream: Stream = Stream(0.0)
        self.cooling_load_kw: float = 0.0
        self.cooling_water_flow_kg_h: float = 0.0
        self.heat_rejected_kw: float = 0.0
        self.electrical_power_kw: float = 0.0  # COP-based electrical consumption
        
        # Dynamics Models (only initialized if enabled)
        if self.enable_dynamics:
            self.pump = PumpFlowDynamics(
                initial_flow_m3_h=0.0,
                fluid_inertance_kg_m4=1e9  # Stable default
            )
            
            self.coolant_thermal = ThermalInertiaModel(
                C_thermal_J_K=1.0e6,
                h_A_passive_W_K=50.0,
                T_initial_K=293.15,
                max_cooling_kw=cooling_capacity_kw
            )
        else:
            self.pump = None
            self.coolant_thermal = None
        
    def initialize(self, dt: float, registry: Any) -> None:
        """Initialize component."""
        super().initialize(dt, registry)
        
    def _calculate_cooling_fallback(self) -> float:
        """
        Fallback cooling calculation using gas-specific Cp.
        
        Uses H2 Cp (14300 J/kg·K) or O2 Cp (918 J/kg·K) based on
        dominant gas component in the stream.
        
        Returns:
            Cooling load in Watts (positive = heat removed from fluid)
        """
        composition = self.inlet_stream.composition
        
        # Detect dominant gas
        h2_fraction = composition.get('H2', 0.0)
        o2_fraction = composition.get('O2', 0.0)
        
        if h2_fraction > o2_fraction:
            Cp_avg = GasConstants.CP_H2_AVG  # 14300 J/(kg·K)
            gas_type = 'H2'
        else:
            Cp_avg = GasConstants.CP_O2_AVG  # 918 J/(kg·K)
            gas_type = 'O2'
        
        mass_flow_kg_s = self.inlet_stream.mass_flow_kg_h / 3600.0
        delta_T = self.inlet_stream.temperature_k - self.target_temp_k
        
        # Q = m * Cp * ΔT (positive for cooling: T_in > T_target)
        Q_dot_W = mass_flow_kg_s * Cp_avg * delta_T
        
        self.logger.warning(
            f"Chiller {self.component_id} using Cp fallback for {gas_type}: "
            f"Cp={Cp_avg:.0f} J/kg·K, ΔT={delta_T:.1f} K"
        )
        
        return Q_dot_W
        
    def step(self, t: float) -> None:
        """
        Execute one timestep of chiller operation.
        
        Uses enthalpy-based calculation as primary method, with gas-specific
        Cp fallback if enthalpy calculation fails.
        
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
            self.electrical_power_kw = 0.0
            return
        
        mass_flow_kg_s = self.inlet_stream.mass_flow_kg_h / 3600.0
        
        # Calculate outlet pressure with drop
        outlet_pressure_pa = self.inlet_stream.pressure_pa - (self.pressure_drop_bar * 1e5)
        
        # Ensure outlet pressure doesn't go negative
        outlet_pressure_pa = max(outlet_pressure_pa, 1e4)  # Min 0.1 bar
        
        # --- Primary: Enthalpy-based calculation ---
        try:
            h_in = self.inlet_stream.specific_enthalpy_j_kg
            
            # Create target stream at desired outlet conditions
            target_stream = Stream(
                mass_flow_kg_h=self.inlet_stream.mass_flow_kg_h,
                temperature_k=self.target_temp_k,
                pressure_pa=outlet_pressure_pa,
                composition=self.inlet_stream.composition.copy()
            )
            h_target = target_stream.specific_enthalpy_j_kg
            
            # Q = mdot * (h_in - h_target) [W]
            # Positive when cooling (h_in > h_target)
            Q_dot_W = mass_flow_kg_s * (h_in - h_target)
            
        except Exception as e:
            # --- Fallback: Gas-specific Cp calculation ---
            self.logger.debug(f"Enthalpy calc failed, using fallback: {e}")
            Q_dot_W = self._calculate_cooling_fallback()
        
        # Apply Cooling Capacity Cap
        max_Q_W = self.cooling_capacity_kw * 1000.0 * self.efficiency
        final_temp_k = self.target_temp_k
        
        if abs(Q_dot_W) > max_Q_W:
            Q_dot_W = np.sign(Q_dot_W) * max_Q_W
            self.logger.warning(
                f"Chiller {self.component_id}: Capacity exceeded. "
                f"Capped at {self.cooling_capacity_kw} kW (Eff={self.efficiency})"
            )
            # Recalculate outlet temperature based on limited Q
            # h_out = h_in - Q / m
            # Approximate T_out using Cp logic if strict H-T lookup is hard to inline here
            # Or use simplified dT = Q / (m * Cp)
            comp = self.inlet_stream.composition
            Cp_est = 14300.0 if comp.get('H2', 0) > 0.5 else 918.0
            delta_T_real = Q_dot_W / (mass_flow_kg_s * Cp_est)
            final_temp_k = self.inlet_stream.temperature_k - delta_T_real

        # Convert to kW
        cooling_load_kw = Q_dot_W / 1000.0
        
        # --- COP-based electrical power ---
        if self.cop > 0:
            self.electrical_power_kw = abs(cooling_load_kw) / self.cop
        else:
            self.electrical_power_kw = 0.0
        
        # --- Create outlet stream ---
        self.outlet_stream = Stream(
            mass_flow_kg_h=self.inlet_stream.mass_flow_kg_h,
            temperature_k=final_temp_k,
            pressure_pa=outlet_pressure_pa,
            composition=self.inlet_stream.composition.copy()
        )
        
        # --- Update state ---
        self.cooling_load_kw = cooling_load_kw
        # Heat rejected includes extracted heat + electrical work (imperfect COP)
        # Typically Q_rejected = Q_cooling + W_compressor
        self.heat_rejected_kw = abs(cooling_load_kw) + self.electrical_power_kw
        
        # Estimate cooling water flow (ΔT_cooling_water ≈ 10K)
        Cp_water = 4.18  # kJ/kg·K
        cooling_water_temp_rise = 10.0
        if self.heat_rejected_kw > 0:
            # Formula: m_dot_h = (Q_kW / (Cp * dT)) * 3600
            self.cooling_water_flow_kg_h = (
                self.heat_rejected_kw / (Cp_water * cooling_water_temp_rise)
            ) * 3600.0
        else:
            self.cooling_water_flow_kg_h = 0.0
        
        # --- Dynamics update (optional) ---
        if self.enable_dynamics and self.pump is not None and self.coolant_thermal is not None:
            dt_seconds = self.dt * 3600.0
            
            # Pump Control (Proportional to error)
            temp_error = self.inlet_stream.temperature_k - self.target_temp_k
            pump_speed = np.clip(temp_error / 30.0, 0, 1) if temp_error > 0 else 0.0
            
            # Advance Pump
            Q_cool_m3_h = self.pump.step(dt_s=dt_seconds, pump_speed_fraction=pump_speed)
            
            # Advance Coolant Thermal
            Q_absorbed_W = abs(cooling_load_kw) * 1000.0
            T_coolant_K = self.coolant_thermal.step(
                dt_s=dt_seconds,
                heat_generated_W=Q_absorbed_W,
                T_control_K=self.target_temp_k
            )
    
    def get_output(self, port_name: str) -> Any:
        """Get output from specified port."""
        if port_name == "fluid_out":
            return self.outlet_stream
        elif port_name == "heat_out":
            return self.heat_rejected_kw
        elif port_name == "cooling_water_out":
            # Return heated cooling water stream
            if self.cooling_water_inlet:
                # Use actual inlet conditions + heat
                return Stream(
                    mass_flow_kg_h=self.cooling_water_flow_kg_h,
                    temperature_k=self.cooling_water_inlet.temperature_k + 10.0,
                    pressure_pa=self.cooling_water_inlet.pressure_pa,
                    composition=self.cooling_water_inlet.composition
                )
            else:
                # Fallback purely for sizing (phantom stream warning)
                return Stream(
                    mass_flow_kg_h=self.cooling_water_flow_kg_h,
                    temperature_k=298.15 + 10.0,
                    pressure_pa=101325.0
                )
        elif port_name == "electricity_in":
            # Return electrical power demand for this chiller
            return self.electrical_power_kw
        return 0.0
    
    def receive_input(self, port_name: str, value: Any, resource_type: str = None) -> float:
        """Receive input at specified port."""
        if port_name == "fluid_in" and isinstance(value, Stream):
            self.inlet_stream = value
            return value.mass_flow_kg_h
        elif port_name == "cooling_water_in" and isinstance(value, Stream):
            # Cooling water is provided externally
            self.cooling_water_inlet = value
            return value.mass_flow_kg_h
        elif port_name == "electricity_in":
            # Accept electrical power allocation (orchestrator reserves this)
            # Return the actual power demand
            return self.electrical_power_kw
        return 0.0
    
    def extract_output(self, port_name: str, amount: float, resource_type: str = None) -> None:
        """Acknowledge extraction of output."""
        pass  # Chiller doesn't store fluid, just passes through
    
    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """Return port definitions."""
        return {
            'fluid_in': {'type': 'input', 'resource_type': 'stream'},
            'cooling_water_in': {'type': 'input', 'resource_type': 'water'},
            'electricity_in': {'type': 'input', 'resource_type': 'electricity'},
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
            'outlet_pressure_bar': self.outlet_stream.pressure_pa / 1e5,
            'heat_rejected_kw': self.heat_rejected_kw,
            'electrical_power_kw': self.electrical_power_kw,
            'cooling_water_flow_kg_h': self.cooling_water_flow_kg_h,
            'cop': self.cop,
            'pressure_drop_bar': self.pressure_drop_bar
        }
