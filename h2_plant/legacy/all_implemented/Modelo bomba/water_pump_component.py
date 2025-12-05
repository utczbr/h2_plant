import CoolProp.CoolProp as CP
from typing import Dict, Any, Optional, Tuple

# Clean import - now that file is renamed to framework_mocks.py
from framework_mocks import Component, Stream, ComponentRegistry


class WaterPump(Component):
    """
    WaterPump refactored to new architecture but preserving EXACT
    calculations from 'water_pump_model.py'.
    
    Modes:
    1. Forward (Inlet Known): Calculates Output State & Power
    2. Reverse (Outlet Known): Calculates Inlet State & Power (using incompressible approx)
    """

    def __init__(
        self,
        pump_id: str,
        eta_is: float = 0.82,
        eta_m: float = 0.96,
        target_pressure_pa: Optional[float] = None
    ):
        super().__init__()
        self.component_id = pump_id
        
        # Efficiencies from water_pump_model.py
        self.eta_is = eta_is
        self.eta_m = eta_m
        
        # Configuration
        self.target_pressure_pa = target_pressure_pa
        
        # Internal State
        self.inlet_stream: Optional[Stream] = None
        self.outlet_stream: Optional[Stream] = None
        self.flow_rate_kg_s: float = 0.0
        
        # Results
        self.power_fluid_kw: float = 0.0
        self.power_shaft_kw: float = 0.0
        self.work_isentropic_kj_kg: float = 0.0
        self.work_real_kj_kg: float = 0.0
        self.calculated_T_c: float = 0.0

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        super().initialize(dt, registry)

    def step(self, t: float) -> None:
        """Execute calculation based on available stream data."""
        super().step(t)
        
        if self.inlet_stream and self.target_pressure_pa:
            # Mode 1: Forward (Entrada Conhecida)
            self._calculate_forward()
        elif self.outlet_stream and self.target_pressure_pa:
            # Mode 2: Reverse (Saída Conhecida)
            self._calculate_reverse()

    def _calculate_forward(self) -> None:
        """
        Implements logic from water_pump_model.py 'Exemplo 1: Entrada Conhecida'.
        Math is done in kJ/kg to match legacy script exactly.
        """
        # 1. Get Inputs
        P1_Pa = self.inlet_stream.pressure_pa
        T1_K = self.inlet_stream.temperature_k
        P2_Pa = self.target_pressure_pa
        m_dot = self.inlet_stream.mass_flow_kg_s
        fluido = 'Water'

        # 3.1 Propriedades da Entrada (1)
        # Legacy: Divides by 1000.0 to get kJ/kg
        h1 = CP.PropsSI('H', 'P', P1_Pa, 'T', T1_K, fluido) / 1000.0
        s1 = CP.PropsSI('S', 'P', P1_Pa, 'T', T1_K, fluido) / 1000.0

        # 3.2 Propriedades do Estado Isoentrópico de Saída (2s)
        h2s = CP.PropsSI('H', 'P', P2_Pa, 'S', s1 * 1000.0, fluido) / 1000.0

        # 3.3 Cálculo da Entalpia de Saída Real (h2)
        Trabalho_is = h2s - h1
        Trabalho_real = Trabalho_is / self.eta_is
        h2 = h1 + Trabalho_real

        # 3.4 Propriedades da Saída Real (2)
        T2_K = CP.PropsSI('T', 'P', P2_Pa, 'H', h2 * 1000.0, fluido)
        
        # Update State Results
        self.work_isentropic_kj_kg = Trabalho_is
        self.work_real_kj_kg = Trabalho_real
        self.calculated_T_c = T2_K - 273.15
        
        # 4. Potência
        self.power_fluid_kw = m_dot * Trabalho_real
        self.power_shaft_kw = self.power_fluid_kw / self.eta_m
        self.flow_rate_kg_s = m_dot

        # Create Outlet Stream
        self.outlet_stream = Stream(
            mass_flow_kg_h=m_dot * 3600.0,
            temperature_k=T2_K,
            pressure_pa=P2_Pa,
            composition=self.inlet_stream.composition,
            phase='liquid'
        )

    def _calculate_reverse(self) -> None:
        """
        Implements logic from water_pump_model.py 'Exemplo 2: Saída Conhecida'.
        Uses Incompressible Approximation for Work.
        """
        # Inputs: P2 (Outlet), T2 (Outlet), P1 (Target Inlet)
        P2_Pa = self.outlet_stream.pressure_pa
        T2_K = self.outlet_stream.temperature_k
        P1_Pa = self.target_pressure_pa
        m_dot = self.outlet_stream.mass_flow_kg_s
        fluido = 'Water'

        # 3.1 Propriedades da Saída (2)
        h2 = CP.PropsSI('H', 'P', P2_Pa, 'T', T2_K, fluido) / 1000.0
        
        # CÁLCULO REVERSO (Incompressible Approximation)
        rho_2 = CP.PropsSI('D', 'P', P2_Pa, 'T', T2_K, fluido)
        v_avg = 1.0 / rho_2
        
        P_diff = P2_Pa - P1_Pa
        
        # w_is (em J/kg) -> w_is / 1000.0 (em kJ/kg)
        w_is_kj = (v_avg * P_diff) / 1000.0
        
        # Trabalho real w_real = w_is / Eta_is (em kJ/kg)
        w_real_kj = w_is_kj / self.eta_is
        
        # w_real = h2 - h1 => h1 = h2 - w_real (em kJ/kg)
        h1 = h2 - w_real_kj
        
        # 3.2 Propriedades da Entrada Real (1)
        T1_K = CP.PropsSI('T', 'P', P1_Pa, 'H', h1 * 1000.0, fluido)
        
        # Update State Results
        self.work_isentropic_kj_kg = w_is_kj
        self.work_real_kj_kg = w_real_kj
        self.calculated_T_c = T1_K - 273.15

        # 4. Potência
        self.power_fluid_kw = m_dot * w_real_kj
        self.power_shaft_kw = self.power_fluid_kw / self.eta_m
        self.flow_rate_kg_s = m_dot

        # Create Inlet Stream (Reconstructed)
        self.inlet_stream = Stream(
            mass_flow_kg_h=m_dot * 3600.0,
            temperature_k=T1_K,
            pressure_pa=P1_Pa,
            composition=self.outlet_stream.composition,
            phase='liquid'
        )

    def receive_input(self, port_name: str, value: Any, resource_type: str) -> float:
        """Receive input stream - now with proper isinstance check."""
        if port_name == 'water_in' and isinstance(value, Stream):
            self.inlet_stream = value
            return value.mass_flow_kg_h
        elif port_name == 'water_out_reverse' and isinstance(value, Stream):
            self.outlet_stream = value
            return value.mass_flow_kg_h
        return 0.0

    def get_output(self, port_name: str) -> Any:
        if port_name == 'water_out':
            return self.outlet_stream
        elif port_name == 'water_in_reverse':
            return self.inlet_stream
        return None

    def get_state(self) -> Dict[str, Any]:
        return {
            'power_fluid_kw': self.power_fluid_kw,
            'power_shaft_kw': self.power_shaft_kw,
            'calculated_T_c': self.calculated_T_c,
            'work_isentropic_kj_kg': self.work_isentropic_kj_kg,
            'work_real_kj_kg': self.work_real_kj_kg
        }