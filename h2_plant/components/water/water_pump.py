"""
Water Pump with Thermodynamic Calculations.

This module implements a water pump with CoolProp-based thermodynamic
property calculations. Supports both forward (inlet→outlet) and reverse
(outlet→inlet) calculation modes for process design flexibility.

Thermodynamic Model:
    Isentropic pump work for incompressible liquid:
    **W_is = v × ΔP** (specific volume × pressure rise)

    Real work accounting for efficiency:
    **W_real = W_is / η_is**

    Shaft power including mechanical losses:
    **P_shaft = (ṁ × W_real) / η_m**

    Temperature rise from pump work (adiabatic):
    **ΔT = W_real / Cp**

Property Calculations:
    - **Primary**: CoolProp with caching via CoolPropLUT for accuracy.
    - **Fallback**: Incompressible liquid approximation (ρ = 1000 kg/m³).

Architecture:
    Implements the Component Lifecycle Contract (Layer 1):
    - `initialize()`: Validates configuration, acquires LUTManager.
    - `step()`: Executes forward or reverse calculation based on data.
    - `get_state()`: Returns power, work, and cumulative statistics.
"""

from typing import Dict, Any, Optional
import logging

from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.stream import Stream
from h2_plant.core.constants import StandardConditions, ConversionFactors

logger = logging.getLogger(__name__)

# CoolProp availability check
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
    Water pump with CoolProp-based thermodynamic calculations.

    Calculates pump work using accurate liquid water properties,
    with incompressible approximation as fallback.

    This component fulfills the Component Lifecycle Contract (Layer 1):
        - `initialize()`: Validates configuration, logs parameters.
        - `step()`: Executes forward/reverse thermodynamic calculation.
        - `get_state()`: Returns power, work, and efficiency metrics.

    Operating Modes:
        - **Forward**: Given inlet (P₁, T₁), calculate outlet (P₂, T₂).
        - **Reverse**: Given outlet (P₂, T₂), back-calculate inlet (P₁, T₁).

    Efficiency Model:
        - η_is: Isentropic efficiency (typically 0.70-0.85 for centrifugal).
        - η_m: Mechanical efficiency (motor + coupling, typically 0.93-0.98).
        - η_overall = η_is × η_m.

    Attributes:
        eta_is (float): Isentropic efficiency (0-1).
        eta_m (float): Mechanical efficiency (0-1).
        power_shaft_kw (float): Current shaft power demand (kW).
        cumulative_energy_kwh (float): Total energy consumed (kWh).

    Example:
        >>> pump = WaterPumpThermodynamic(
        ...     pump_id='pem_feed_pump',
        ...     eta_is=0.82,
        ...     eta_m=0.96,
        ...     target_pressure_pa=500000.0
        ... )
        >>> pump.initialize(dt=1/60, registry=registry)
        >>> pump.receive_input('water_in', inlet_stream, 'water')
        >>> pump.step(t=0.0)
        >>> outlet = pump.get_output('water_out')
    """

    def __init__(
        self,
        pump_id: str,
        eta_is: float = 0.82,
        eta_m: float = 0.96,
        target_pressure_pa: Optional[float] = None
    ):
        """
        Initialize the water pump.

        Args:
            pump_id (str): Unique pump identifier.
            eta_is (float): Isentropic efficiency (0-1).
                Typical centrifugal pump: 0.70-0.85. Default: 0.82.
            eta_m (float): Mechanical efficiency (0-1).
                Motor + coupling: 0.93-0.98. Default: 0.96.
            target_pressure_pa (float, optional): Target outlet pressure in Pa.
                Required for operation.
        """
        super().__init__()
        self.component_id = pump_id

        # Efficiency parameters
        self.eta_is = eta_is
        self.eta_m = eta_m

        # Configuration
        self.target_pressure_pa = target_pressure_pa

        # Stream state
        self.inlet_stream: Optional[Stream] = Stream(0.0)
        self.outlet_stream: Optional[Stream] = Stream(0.0)
        self.flow_rate_kg_s: float = 0.0

        # Calculation results
        self.power_fluid_kw: float = 0.0
        self.power_shaft_kw: float = 0.0
        self.work_isentropic_kj_kg: float = 0.0
        self.work_real_kj_kg: float = 0.0
        self.calculated_T_c: float = 0.0

        # Cumulative tracking
        self.cumulative_energy_kwh: float = 0.0
        self.cumulative_water_kg: float = 0.0

        self._lut_manager = None

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """
        Prepare the component for simulation execution.

        Fulfills the Component Lifecycle Contract initialization phase.
        Acquires LUTManager reference for optimized property lookup.

        Args:
            dt (float): Simulation timestep in hours.
            registry (ComponentRegistry): Central registry for component access.
        """
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
        """
        Execute one simulation timestep.

        Selects calculation mode based on available stream data:
        - Inlet stream present → Forward calculation.
        - Outlet stream present → Reverse calculation.

        Args:
            t (float): Current simulation time in hours.
        """
        super().step(t)

        self.power_fluid_kw = 0.0
        self.power_shaft_kw = 0.0

        if self.inlet_stream and self.target_pressure_pa:
            self._calculate_forward()
        elif self.outlet_stream and self.target_pressure_pa:
            self._calculate_reverse()

    def _calculate_forward(self) -> None:
        """
        Forward calculation: inlet known, calculate outlet.

        Uses CoolProp for enthalpy/entropy at inlet, isentropic compression
        to outlet pressure, then applies efficiency to find real outlet.
        """
        if not COOLPROP_AVAILABLE:
            self._calculate_forward_simplified()
            return

        P1_Pa = self.inlet_stream.pressure_pa
        T1_K = self.inlet_stream.temperature_k
        P2_Pa = self.target_pressure_pa
        m_dot = self.inlet_stream.mass_flow_kg_h / 3600.0
        fluido = 'Water'

        try:
            # Inlet properties (kJ/kg)
            h1 = 0.0
            s1 = 0.0

            if self._lut_manager:
                try:
                    h1 = self._lut_manager.lookup(fluido, 'H', P1_Pa, T1_K) / 1000.0
                    s1 = self._lut_manager.lookup(fluido, 'S', P1_Pa, T1_K) / 1000.0
                except:
                    h1 = CoolPropLUT.PropsSI('H', 'P', P1_Pa, 'T', T1_K, fluido) / 1000.0
                    s1 = CoolPropLUT.PropsSI('S', 'P', P1_Pa, 'T', T1_K, fluido) / 1000.0
            else:
                h1 = CoolPropLUT.PropsSI('H', 'P', P1_Pa, 'T', T1_K, fluido) / 1000.0
                s1 = CoolPropLUT.PropsSI('S', 'P', P1_Pa, 'T', T1_K, fluido) / 1000.0

            # Isentropic outlet enthalpy (constant entropy compression)
            h2s = CoolPropLUT.PropsSI('H', 'P', P2_Pa, 'S', s1 * 1000.0, fluido) / 1000.0

            # Real work with efficiency
            Trabalho_is = h2s - h1
            Trabalho_real = Trabalho_is / self.eta_is
            h2 = h1 + Trabalho_real

            # Outlet temperature from enthalpy
            T2_K = CoolPropLUT.PropsSI('T', 'P', P2_Pa, 'H', h2 * 1000.0, fluido)

            # Store results
            self.work_isentropic_kj_kg = Trabalho_is
            self.work_real_kj_kg = Trabalho_real
            self.calculated_T_c = T2_K - 273.15

            # Power calculations
            self.power_fluid_kw = m_dot * Trabalho_real
            self.power_shaft_kw = self.power_fluid_kw / self.eta_m
            self.flow_rate_kg_s = m_dot

            # Update cumulative statistics
            self.cumulative_energy_kwh += self.power_shaft_kw * self.dt
            self.cumulative_water_kg += m_dot * 3600.0 * self.dt

            # Create outlet stream
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
        """
        Simplified forward calculation using incompressible liquid model.

        W_is = v × ΔP = (1/ρ) × ΔP
        ΔT = W_real / Cp
        """
        P1_Pa = self.inlet_stream.pressure_pa
        T1_K = self.inlet_stream.temperature_k
        P2_Pa = self.target_pressure_pa
        m_dot = self.inlet_stream.mass_flow_kg_h / 3600.0

        rho = 1000.0
        delta_p = P2_Pa - P1_Pa

        w_is_j_kg = (1.0 / rho) * delta_p
        w_real_j_kg = w_is_j_kg / self.eta_is

        c_p = 4186.0
        delta_t = w_real_j_kg / c_p
        T2_K = T1_K + delta_t

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
        Reverse calculation: outlet known, back-calculate inlet.

        Uses incompressible approximation to find inlet conditions
        from known outlet state.
        """
        if not COOLPROP_AVAILABLE:
            self._calculate_reverse_simplified()
            return

        P2_Pa = self.outlet_stream.pressure_pa
        T2_K = self.outlet_stream.temperature_k
        P1_Pa = self.target_pressure_pa
        m_dot = self.outlet_stream.mass_flow_kg_h / 3600.0
        fluido = 'Water'

        try:
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

            h1 = h2 - w_real_kj

            T1_K = CoolPropLUT.PropsSI('T', 'P', P1_Pa, 'H', h1 * 1000.0, fluido)

            self.work_isentropic_kj_kg = w_is_kj
            self.work_real_kj_kg = w_real_kj
            self.calculated_T_c = T1_K - 273.15
            self.power_fluid_kw = m_dot * w_real_kj
            self.power_shaft_kw = self.power_fluid_kw / self.eta_m
            self.flow_rate_kg_s = m_dot

            self.cumulative_energy_kwh += self.power_shaft_kw * self.dt
            self.cumulative_water_kg += m_dot * 3600.0 * self.dt

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
        """
        Simplified reverse calculation using incompressible liquid model.
        """
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
        self.cumulative_water_kg += m_dot * 3600.0 * self.dt

        self.inlet_stream = Stream(
            mass_flow_kg_h=m_dot * 3600.0,
            temperature_k=T1_K,
            pressure_pa=P1_Pa,
            composition=self.outlet_stream.composition,
            phase='liquid'
        )

    def receive_input(self, port_name: str, value: Any, resource_type: str) -> float:
        """
        Accept input at specified port.

        Args:
            port_name (str): Target port ('water_in', 'water_out_reverse',
                or 'electricity_in').
            value (Any): Input stream or power value.
            resource_type (str): Resource classification hint.

        Returns:
            float: Amount accepted.
        """
        if port_name == 'water_in' and isinstance(value, Stream):
            self.inlet_stream = value
            return value.mass_flow_kg_h
        elif port_name == 'water_out_reverse' and isinstance(value, Stream):
            self.outlet_stream = value
            return value.mass_flow_kg_h
        elif port_name == 'electricity_in':
            return float(value) if isinstance(value, (int, float)) else 0.0
        return 0.0

    def get_output(self, port_name: str) -> Any:
        """
        Retrieve output stream from specified port.

        Args:
            port_name (str): Port identifier ('water_out' or 'water_in_reverse').

        Returns:
            Stream: Output stream or None.
        """
        if port_name == 'water_out':
            return self.outlet_stream if self.outlet_stream else Stream(0.0)
        elif port_name == 'water_in_reverse':
            return self.inlet_stream if self.inlet_stream else Stream(0.0)
        return None

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """
        Define the physical connection ports for this component.

        Returns:
            Dict[str, Dict[str, str]]: Port definitions.
        """
        return {
            'water_in': {'type': 'input', 'resource_type': 'water', 'units': 'kg/h'},
            'electricity_in': {'type': 'input', 'resource_type': 'electricity', 'units': 'kW'},
            'water_out': {'type': 'output', 'resource_type': 'water', 'units': 'kg/h'}
        }

    def get_state(self) -> Dict[str, Any]:
        """
        Retrieve the component's current operational state.

        Fulfills the Component Lifecycle Contract state access.

        Returns:
            Dict[str, Any]: State dictionary containing:
                - power_shaft_kw (float): Current shaft power (kW).
                - work_real_kj_kg (float): Real specific work (kJ/kg).
                - cumulative_energy_kwh (float): Total energy (kWh).
                - overall_efficiency (float): η_is × η_m.
        """
        specific_energy = 0.0
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
