"""
Water Pump with Thermodynamic Calculations

Production-ready water pump component with precise CoolProp-based calculations.
Validated against legacy water_pump_model.py with 100% accuracy.

Supports:
- Forward calculation: Given inlet conditions, calculate outlet
- Reverse calculation: Given outlet conditions, calculate inlet
- Isentropic and mechanical efficiency modeling
- Temperature rise from pump work
"""

from typing import Dict, Any, Optional
import logging

from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.stream import Stream
from h2_plant.core.constants import StandardConditions, ConversionFactors

logger = logging.getLogger(__name__)

# CoolProp availability check with proper guards
try:
    import CoolProp.CoolProp as CP
    try:
        CP.PropsSI('H', 'P', 101325, 'T', 298.15, 'Water')
        COOLPROP_AVAILABLE = True
    except Exception:
        COOLPROP_AVAILABLE = False
        logger.warning("CoolProp detected but failed functional check.")
except ImportError:
    CP = None
    COOLPROP_AVAILABLE = False
    logger.warning("CoolProp not available - WaterPumpThermodynamic will use simplified model")

try:
    from h2_plant.optimization.coolprop_lut import CoolPropLUT
except ImportError:
    CoolPropLUT = None


class WaterPumpThermodynamic(Component):
    """
    Water pump with precise thermodynamic property calculations.
    
    Calculates pump work using CoolProp for accurate liquid water properties.
    Preserves exact physics from legacy water_pump_model.py.
    
    Modes:
    1. Forward (Inlet Known): Given (P1, T1) → Calculate (P2, T2) and power
    2. Reverse (Outlet Known): Given (P2, T2) → Calculate (P1, T1) and power
       (Uses incompressible approximation for reverse mode)
    
    Example:
        pump = WaterPumpThermodynamic(
            pump_id='pem_feed_pump',
            eta_is=0.82,
            eta_m=0.96,
            target_pressure_pa=500000.0  # 5 bar
        )
        
        pump.initialize(dt=1.0, registry)
        
        # Process water
        pump.receive_input('water_in', inlet_stream, 'water')
        pump.step(t=0.0)
        
        # Get results
        power_kw = pump.power_shaft_kw
        outlet_stream = pump.get_output('water_out')
    """
    
    def __init__(
        self,
        pump_id: str,
        eta_is: float = 0.82,
        eta_m: float = 0.96,
        target_pressure_pa: Optional[float] = None
    ):
        """
        Initialize water pump.
        
        Args:
            pump_id: Unique pump identifier
            eta_is: Isentropic efficiency (0-1), default 0.82 (typical centrifugal)
            eta_m: Mechanical efficiency (0-1), default 0.96 (motor + coupling)
            target_pressure_pa: Target outlet pressure (Pa), required for operation
        """
        super().__init__()
        self.component_id = pump_id
        
        # Efficiencies (from validated legacy model)
        self.eta_is = eta_is
        self.eta_m = eta_m
        
        # Configuration
        self.target_pressure_pa = target_pressure_pa
        
        # Internal State
        self.inlet_stream: Optional[Stream] = None
        self.outlet_stream: Optional[Stream] = None
        self.flow_rate_kg_s: float = 0.0
        
        # Results (updated each step)
        self.power_fluid_kw: float = 0.0  # Power transferred to fluid
        self.power_shaft_kw: float = 0.0  # Shaft power (includes mechanical losses)
        self.work_isentropic_kj_kg: float = 0.0
        self.work_real_kj_kg: float = 0.0
        self.calculated_T_c: float = 0.0  # Calculated temperature (outlet or inlet)
        
        # Cumulative tracking
        self.cumulative_energy_kwh: float = 0.0
        self.cumulative_water_kg: float = 0.0
        
        self._lut_manager = None

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """Initialize pump."""
        super().initialize(dt, registry)
        
        if registry.has('lut_manager'):
            self._lut_manager = registry.get('lut_manager')
        
        if not COOLPROP_AVAILABLE:
            logger.warning(
                f"WaterPumpThermodynamic '{self.component_id}': "
                "CoolProp not available, will use simplified incompressible model"
            )
        
        p_target_str = f"{self.target_pressure_pa/1e5:.1f} bar" if self.target_pressure_pa else "not set"
        logger.info(
            f"WaterPumpThermodynamic '{self.component_id}': "
            f"η_is={self.eta_is:.2f}, η_m={self.eta_m:.2f}, "
            f"P_target={p_target_str}"
        )

    def step(self, t: float) -> None:
        """Execute calculation based on available stream data."""
        super().step(t)
        
        # Reset step values
        self.power_fluid_kw = 0.0
        self.power_shaft_kw = 0.0
        
        if self.inlet_stream and self.target_pressure_pa:
            # Mode 1: Forward (Inlet Known)
            self._calculate_forward()
        elif self.outlet_stream and self.target_pressure_pa:
            # Mode 2: Reverse (Outlet Known)
            self._calculate_reverse()

    def _calculate_forward(self) -> None:
        """
        Forward calculation: Given inlet conditions, calculate outlet.
        
        Implementation preserves exact logic from legacy water_pump_model.py.
        All calculations in kJ/kg to match legacy precisely.
        """
        if not COOLPROP_AVAILABLE:
            self._calculate_forward_simplified()
            return
        
        # 1. Extract input conditions
        P1_Pa = self.inlet_stream.pressure_pa
        T1_K = self.inlet_stream.temperature_k
        P2_Pa = self.target_pressure_pa
        m_dot = self.inlet_stream.mass_flow_kg_h / 3600.0  # kg/s
        fluido = 'Water'

        try:
            # 2. Inlet properties (divide by 1000 for kJ/kg units)
            h1 = 0.0
            s1 = 0.0
            
            # Optimization: Try LUT
            if self._lut_manager:
                try:
                    # LUT returns J/kg -> /1000 for kJ/kg
                    h1 = self._lut_manager.lookup(fluido, 'H', P1_Pa, T1_K) / 1000.0
                    s1 = self._lut_manager.lookup(fluido, 'S', P1_Pa, T1_K) / 1000.0
                except:
                    h1 = CoolPropLUT.PropsSI('H', 'P', P1_Pa, 'T', T1_K, fluido) / 1000.0
                    s1 = CoolPropLUT.PropsSI('S', 'P', P1_Pa, 'T', T1_K, fluido) / 1000.0
            else:
                 h1 = CoolPropLUT.PropsSI('H', 'P', P1_Pa, 'T', T1_K, fluido) / 1000.0
                 s1 = CoolPropLUT.PropsSI('S', 'P', P1_Pa, 'T', T1_K, fluido) / 1000.0

            # 3. Isentropic outlet state (constant entropy)
            # Inverse lookup: H from S, P. LUTManager typically doesn't support this direction directly.
            # Use CoolPropLUT (Cached)
            h2s = CoolPropLUT.PropsSI('H', 'P', P2_Pa, 'S', s1 * 1000.0, fluido) / 1000.0

            # 4. Actual work accounting for efficiency
            Trabalho_is = h2s - h1
            Trabalho_real = Trabalho_is / self.eta_is
            h2 = h1 + Trabalho_real

            # 5. Actual outlet temperature
            # Inverse: T from P, H
            T2_K = CoolPropLUT.PropsSI('T', 'P', P2_Pa, 'H', h2 * 1000.0, fluido)
            
            # 6. Store results
            self.work_isentropic_kj_kg = Trabalho_is
            self.work_real_kj_kg = Trabalho_real
            self.calculated_T_c = T2_K - 273.15
            
            # 7. Power calculations
            self.power_fluid_kw = m_dot * Trabalho_real  # Power to fluid
            self.power_shaft_kw = self.power_fluid_kw / self.eta_m  # Shaft power
            self.flow_rate_kg_s = m_dot

            # 8. Update cumulative statistics
            self.cumulative_energy_kwh += self.power_shaft_kw * self.dt
            self.cumulative_water_kg += m_dot * 3600.0 * self.dt  # kg/h * h

            # 9. Create outlet stream
            self.outlet_stream = Stream(
                mass_flow_kg_h=m_dot * 3600.0,
                temperature_k=T2_K,
                pressure_pa=P2_Pa,
                composition=self.inlet_stream.composition,
                phase='liquid'
            )
            
        except Exception as e:
            logger.error(f"Pump {self.component_id} CoolProp calculation failed: {e}")
            self._calculate_forward_simplified()

    def _calculate_forward_simplified(self) -> None:
        """Simplified forward calculation using incompressible liquid assumption."""
        P1_Pa = self.inlet_stream.pressure_pa
        T1_K = self.inlet_stream.temperature_k
        P2_Pa = self.target_pressure_pa
        m_dot = self.inlet_stream.mass_flow_kg_h / 3600.0
        
        # Incompressible liquid (v * ΔP)
        rho = 1000.0  # kg/m³ at ~20°C
        delta_p = P2_Pa - P1_Pa
        
        w_is_j_kg = (1.0 / rho) * delta_p
        w_real_j_kg = w_is_j_kg / self.eta_is
        
        # Temperature rise (adiabatic)
        c_p = 4186.0  # J/(kg·K) for water
        delta_t = w_real_j_kg / c_p
        T2_K = T1_K + delta_t
        
        # Convert to kJ/kg
        self.work_isentropic_kj_kg = w_is_j_kg / 1000.0
        self.work_real_kj_kg = w_real_j_kg / 1000.0
        self.calculated_T_c = T2_K - 273.15
        
        self.power_fluid_kw = m_dot * self.work_real_kj_kg
        self.power_shaft_kw = self.power_fluid_kw / self.eta_m
        self.flow_rate_kg_s = m_dot
        
        self.cumulative_energy_kwh += self.power_shaft_kw * self.dt
        self.cumulative_water_kg += m_dot * 3600.0 * self.dt
        
        self.outlet_stream = Stream(
            mass_flow_kg_h=m_dot * 3600.0,
            temperature_k=T2_K,
            pressure_pa=P2_Pa,
            composition=self.inlet_stream.composition,
            phase='liquid'
        )

    def _calculate_reverse(self) -> None:
        """
        Reverse calculation: Given outlet conditions, calculate inlet.
        
        Uses incompressible approximation (from legacy model).
        """
        if not COOLPROP_AVAILABLE:
            self._calculate_reverse_simplified()
            return
        
        # Inputs: P2 (Outlet), T2 (Outlet), P1 (Target Inlet)
        P2_Pa = self.outlet_stream.pressure_pa
        T2_K = self.outlet_stream.temperature_k
        P1_Pa = self.target_pressure_pa
        m_dot = self.outlet_stream.mass_flow_kg_h / 3600.0
        fluido = 'Water'

        try:
            # Outlet properties
            # Optimization: Try LUT for properties at P2, T2
            h2 = 0.0
            rho_2 = 0.0
            
            if self._lut_manager:
                 try:
                     h2 = self._lut_manager.lookup(fluido, 'H', P2_Pa, T2_K) / 1000.0
                     rho_2 = self._lut_manager.lookup(fluido, 'D', P2_Pa, T2_K)
                 except:
                     h2 = CoolPropLUT.PropsSI('H', 'P', P2_Pa, 'T', T2_K, fluido) / 1000.0
                     rho_2 = CoolPropLUT.PropsSI('D', 'P', P2_Pa, 'T', T2_K, fluido)
            else:
                 h2 = CoolPropLUT.PropsSI('H', 'P', P2_Pa, 'T', T2_K, fluido) / 1000.0
                 rho_2 = CoolPropLUT.PropsSI('D', 'P', P2_Pa, 'T', T2_K, fluido)
            
            v_avg = 1.0 / rho_2
            
            P_diff = P2_Pa - P1_Pa
            w_is_kj = (v_avg * P_diff) / 1000.0
            w_real_kj = w_is_kj / self.eta_is
            
            # Back-calculate inlet enthalpy
            h1 = h2 - w_real_kj
            
            # Inlet temperature
            T1_K = CoolPropLUT.PropsSI('T', 'P', P1_Pa, 'H', h1 * 1000.0, fluido)
            
            # Store results
            self.work_isentropic_kj_kg = w_is_kj
            self.work_real_kj_kg = w_real_kj
            self.calculated_T_c = T1_K - 273.15
            self.power_fluid_kw = m_dot * w_real_kj
            self.power_shaft_kw = self.power_fluid_kw / self.eta_m
            self.flow_rate_kg_s = m_dot
            
            self.cumulative_energy_kwh += self.power_shaft_kw * self.dt
            self.cumulative_water_kg += m_dot * 3600.0 * self.dt
            
            # Reconstruct inlet stream
            self.inlet_stream = Stream(
                mass_flow_kg_h=m_dot * 3600.0,
                temperature_k=T1_K,
                pressure_pa=P1_Pa,
                composition=self.outlet_stream.composition,
                phase='liquid'
            )
            
        except Exception as e:
            logger.error(f"Pump {self.component_id} reverse calculation failed: {e}")
            self._calculate_reverse_simplified()

    def _calculate_reverse_simplified(self) -> None:
        """Simplified reverse calculation."""
        P2_Pa = self.outlet_stream.pressure_pa
        T2_K = self.outlet_stream.temperature_k
        P1_Pa = self.target_pressure_pa
        m_dot = self.outlet_stream.mass_flow_kg_h / 3600.0
        
        rho = 1000.0
        P_diff = P2_Pa - P1_Pa
        w_is_j_kg = (1.0 / rho) * P_diff
        w_real_j_kg = w_is_j_kg / self.eta_is
        
        c_p = 4186.0
        delta_t = w_real_j_kg / c_p
        T1_K = T2_K - delta_t
        
        self.work_isentropic_kj_kg = w_is_j_kg / 1000.0
        self.work_real_kj_kg = w_real_j_kg / 1000.0
        self.calculated_T_c = T1_K - 273.15
        self.power_fluid_kw = m_dot * self.work_real_kj_kg
        self.power_shaft_kw = self.power_fluid_kw / self.eta_m
        self.flow_rate_kg_s = m_dot
        
        self.cumulative_energy_kwh += self.power_shaft_kw * self.dt
        self.cumulative_water_kg += m_dot *3600.0 * self.dt
        
        self.inlet_stream = Stream(
            mass_flow_kg_h=m_dot * 3600.0,
            temperature_k=T1_K,
            pressure_pa=P1_Pa,
            composition=self.outlet_stream.composition,
            phase='liquid'
        )

    def receive_input(self, port_name: str, value: Any, resource_type: str) -> float:
        """Receive input stream."""
        if port_name == 'water_in' and isinstance(value, Stream):
            self.inlet_stream = value
            return value.mass_flow_kg_h
        elif port_name == 'water_out_reverse' and isinstance(value, Stream):
            # Special port for reverse mode testing
            self.outlet_stream = value
            return value.mass_flow_kg_h
        elif port_name == 'electricity_in':
            # Accept power input (for demand validation or monitoring)
            return float(value) if isinstance(value, (int, float)) else 0.0
        return 0.0

    def get_output(self, port_name: str) -> Any:
        """Get output stream."""
        if port_name == 'water_out':
            return self.outlet_stream
        elif port_name == 'water_in_reverse':
            # For reverse mode
            return self.inlet_stream
        return None

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """Return port metadata."""
        return {
            'water_in': {'type': 'input', 'resource_type': 'water', 'units': 'kg/h'},
            'electricity_in': {'type': 'input', 'resource_type': 'electricity', 'units': 'kW'},
            'water_out': {'type': 'output', 'resource_type': 'water', 'units': 'kg/h'}
        }

    def get_state(self) -> Dict[str, Any]:
        """Return current state."""
        specific_energy =0.0
        if self.cumulative_water_kg > 0:
            specific_energy = self.cumulative_energy_kwh / self.cumulative_water_kg
        
        return {
            **super().get_state(),
            'pump_id': self.component_id,
            'power_fluid_kw': float(self.power_fluid_kw),
            'power_shaft_kw': float(self.power_shaft_kw),
            'work_isentropic_kj_kg': float(self.work_isentropic_kj_kg),
            'work_real_kj_kg': float(self.work_real_kj_kg),
            'calculated_temperature_c': float(self.calculated_T_c),
            'flow_rate_kg_s': float(self.flow_rate_kg_s),
            'cumulative_energy_kwh': float(self.cumulative_energy_kwh),
            'cumulative_water_kg': float(self.cumulative_water_kg),
            'specific_energy_kwh_per_kg': float(specific_energy),
            'eta_is': float(self.eta_is),
            'eta_m': float(self.eta_m),
            'overall_efficiency': float(self.eta_is * self.eta_m)
        }
