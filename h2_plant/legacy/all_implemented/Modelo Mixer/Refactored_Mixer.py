import logging
from typing import Dict, Any, Optional, List
import CoolProp.CoolProp as CP
from Framework import Component, Stream, ComponentRegistry

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Mixer(Component):
    """
    Refactored Mixer Component following Mixer_arc.md architecture.
    
    CRITICAL: Internal physics/calculations strictly follow Mixer.py:
    - Mass Balance: sum(m_dot)
    - Energy Balance: sum(m_dot * h) / sum(m_dot)
    - Uses CoolProp for all enthalpy lookups.
    """
    
    def __init__(
        self,
        mixer_id: str,
        fluid_type: str = 'Water',
        outlet_pressure_kpa: float = 200.0, # Default from Mixer.py example
        max_inlet_streams: int = 10
    ):
        super().__init__()
        
        self.mixer_id = mixer_id
        self.fluid_type = fluid_type # 'Water' is hardcoded in Mixer.py logic
        self.outlet_pressure_kpa = outlet_pressure_kpa
        self.max_inlet_streams = max_inlet_streams
        
        # Dynamic inlet streams dictionary
        self.inlet_streams: Dict[str, Optional[Stream]] = {}
        
        # Outputs
        self.outlet_stream: Optional[Stream] = None
        
        # State variables for reporting (matching Mixer.py outputs)
        self.last_calculated_mass_flow_kg_s = 0.0
        self.last_calculated_enthalpy_kj_kg = 0.0
        self.last_calculated_temp_c = 0.0
        
    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """Initialize mixer."""
        super().initialize(dt, registry)
        logger.info(f"Mixer {self.mixer_id} initialized. Target P_out={self.outlet_pressure_kpa} kPa")

    def receive_input(self, port_name: str, value: Any, resource_type: str) -> float:
        """
        Dynamically accepts input streams.
        Returns accepted mass flow (kg/h) to upstream component.
        """
        if isinstance(value, Stream):
            if len(self.inlet_streams) < self.max_inlet_streams:
                self.inlet_streams[port_name] = value
                return value.mass_flow_kg_h
            else:
                logger.warning(f"Max streams reached. Rejecting {port_name}")
        return 0.0

    def get_output(self, port_name: str) -> Any:
        if port_name == 'mixed_out':
            return self.outlet_stream
        return None

    def step(self, t: float) -> None:
        """
        Execute timestep. 
        
        Implementation Note:
        This method adapts the architecture's 'Stream' objects (SI units) 
        to the specific variables and units used in 'Mixer.py' to ensure 
        exact calculation parity.
        """
        super().step(t)
        
        # 1. Prepare data structures exactly like Mixer.py expects
        # Mixer.py iterates inputs and sums mass and energy
        total_energy_in = 0.0  # kJ/s
        total_mass_in = 0.0    # kg/s
        
        active_streams = [s for s in self.inlet_streams.values() if s is not None]
        
        if not active_streams:
            self.outlet_stream = None
            return

        # 2. Perform Input Calculations (Replicating Mixer.py loop)
        for stream in active_streams:
            # CONVERSION: Architecture (Stream) -> Mixer.py Units
            # Stream: kg/h, K, Pa
            # Mixer.py: kg/s, C, kPa
            
            m_dot_i = stream.mass_flow_kg_h / 3600.0  # kg/s
            T_i_C = stream.temperature_k - 273.15     # °C
            P_i_kPa = stream.pressure_pa / 1000.0     # kPa
            
            # --- START EXACT MIXER.PY LOGIC BLOCK ---
            # Unit conversion for CoolProp standard (K and Pa)
            # Note: We converted to C/kPa above just to match Mixer.py inputs, 
            # now we convert back to K/Pa for CoolProp, exactly as Mixer.py does.
            T_i_K_calc = T_i_C + 273.15 
            P_i_Pa_calc = P_i_kPa * 1000 
            
            try:
                # Find Enthalpy (H) in J/kg and convert to kJ/kg
                h_i_kJ_kg = CP.PropsSI('H', 'T', T_i_K_calc, 'P', P_i_Pa_calc, self.fluid_type) / 1000 
                
                # Mass Balance: Sum of mass flow rates
                total_mass_in += m_dot_i
                
                # Energy Balance: Sum of (m_dot_i * h_i)
                energy_in_i = m_dot_i * h_i_kJ_kg
                total_energy_in += energy_in_i
                
            except ValueError as e:
                logger.error(f"CoolProp error: {e}")
                self.outlet_stream = None
                return
            # --- END EXACT MIXER.PY LOGIC BLOCK ---

        # 3. Perform Output Calculations (Replicating Mixer.py final block)
        m_dot_4 = total_mass_in
        
        if m_dot_4 <= 0:
            self.outlet_stream = None
            return

        # Energy Balance: Output Enthalpy (h_4)
        h_4_kJ_kg = total_energy_in / m_dot_4
        
        # Find Output Temperature (T_4) using h_4 and P_4
        h_4_J_kg = h_4_kJ_kg * 1000 
        P_4_Pa = self.outlet_pressure_kpa * 1000   
        
        try:
            T_4_K = CP.PropsSI('T', 'H', h_4_J_kg, 'P', P_4_Pa, self.fluid_type)
            # Mixer.py calculates T_4_C here, we calculate it for reporting parity
            T_4_C = T_4_K - 273.15 
        except ValueError as e:
            logger.error(f"Output calc error: {e}")
            self.outlet_stream = None
            return

        # 4. Update State and Create Output Stream
        # Save calculated values for parity check
        self.last_calculated_mass_flow_kg_s = m_dot_4
        self.last_calculated_enthalpy_kj_kg = h_4_kJ_kg
        self.last_calculated_temp_c = T_4_C
        
        # Create Output Stream (Converting back to Architecture units: kg/h, K, Pa)
        self.outlet_stream = Stream(
            mass_flow_kg_h=m_dot_4 * 3600.0,
            temperature_k=T_4_K,
            pressure_pa=P_4_Pa,
            composition={'H2O': 1.0}, # Assuming pure water as per Mixer.py
            phase='liquid' # Simplified assumption
        )

    def get_state(self) -> Dict[str, Any]:
        """Return state for monitoring."""
        return {
            **super().get_state(),
            'mixer_id': self.mixer_id,
            'calc_mass_flow_kg_s': self.last_calculated_mass_flow_kg_s,
            'calc_temp_c': self.last_calculated_temp_c,
            'calc_enthalpy_kj_kg': self.last_calculated_enthalpy_kj_kg,
            'calc_pressure_kpa': self.outlet_pressure_kpa
        }