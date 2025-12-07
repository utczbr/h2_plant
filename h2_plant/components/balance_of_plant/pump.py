
import logging
from typing import Any, Dict, Optional, List

from h2_plant.core.component import Component
from h2_plant.core.stream import Stream
from h2_plant.core.component_registry import ComponentRegistry

# Try importing CoolProp
try:
    import CoolProp.CoolProp as CP
    COOLPROP_OK = True
except ImportError:
    CP = None
    COOLPROP_OK = False # Keep existing variable for compatibility
    COOLPROP_AVAILABLE = False # New variable as per instruction

try:
    from h2_plant.optimization.coolprop_lut import CoolPropLUT
except ImportError:
    CoolPropLUT = None

logger = logging.getLogger(__name__)

class Pump(Component):
    """
    Water Pump component using rigorous CoolProp thermodynamics.
    
    Implements the "Push" architecture:
    1. Receives input stream via `receive_input`.
    2. Buffers it.
    3. In `step()`, pumps it to `target_pressure_bar`.
    4. Calculates rigorous Enthalpy/Temperature rise and Power consumption.
    """

    def __init__(
        self, 
        target_pressure_bar: float,
        eta_is: float = 0.82,
        eta_m: float = 0.96,
        capacity_kg_h: float = 1000.0
    ):
        super().__init__()
        
        self.target_pressure_pa = target_pressure_bar * 1e5
        self.eta_is = eta_is
        self.eta_m = eta_m
        self.capacity_kg_h = capacity_kg_h
        self.fluid = "Water"
        
        # Buffer for Push Architecture
        self._input_buffer: List[Stream] = []
        
        # State
        self.outlet_stream: Optional[Stream] = None
        self.power_kw = 0.0
        self.last_power_kw = 0.0
        self.last_efficiency = 0.0
        
        self._lut_manager = None
        
    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        super().initialize(dt, registry)
        
        if registry.has('lut_manager'):
            self._lut_manager = registry.get('lut_manager')
        if not COOLPROP_OK:
            logger.warning(f"Pump {self.component_id}: CoolProp not available. Using simplified fallback.")

    def receive_input(self, port_name: str, value: Any, resource_type: str = None) -> float:
        """
        Receive water input from upstream.
        """
        if port_name == 'inlet' or port_name == 'water_in':
            if isinstance(value, Stream):
                if value.mass_flow_kg_h > 0:
                    self._input_buffer.append(value)
                return value.mass_flow_kg_h
        return 0.0

    def get_output(self, port_name: str) -> Any:
        if port_name == 'outlet' or port_name == 'water_out':
            return self.outlet_stream
        return None

    def step(self, t: float) -> None:
        super().step(t)
        
        # 1. Aggregate Inputs
        if not self._input_buffer:
            self.power_kw = 0.0
            self.flow_rate_kg_h = 0.0
            self.outlet_stream = None
            return

        # Mix multiple inputs if necessary (though usually pumps have 1 inlet)
        combined_stream = self._input_buffer[0]
        for s in self._input_buffer[1:]:
            combined_stream = combined_stream.mix_with(s)
        
        self._input_buffer = [] # Clear buffer
        
        self.flow_rate_kg_h = combined_stream.mass_flow_kg_h
        
        # 2. Check if pumping is needed
        if combined_stream.pressure_pa >= self.target_pressure_pa:
            # Flow through without pumping? Or check valve? 
            # For now, let's assume it passes through with 0 power but still exits
            self.power_kw = 0.0
            self.outlet_stream = combined_stream
            self.outlet_temp_c = combined_stream.temperature_k - 273.15
            logger.debug(f"Pump {self.component_id}: Input pressure {combined_stream.pressure_pa} >= Target {self.target_pressure_pa}. No pumping.")
            return

        # 3. Perform Pumping Calculation
        if COOLPROP_OK:
            self._pump_coolprop(combined_stream)
        else:
            self._pump_simplified(combined_stream)

    def _pump_coolprop(self, inlet: Stream) -> None:
        try:
            P1_Pa = inlet.pressure_pa
            T1_K = inlet.temperature_k
            P2_Pa = self.target_pressure_pa
            
            # 1. Inlet Enthalpy/Entropy Optimization Cascade
            h1 = 0.0
            s1 = 0.0
            
            # Tier 1: LUTManager (Fastest)
            if self._lut_manager:
                try:
                    # LUT Manager returns SI units (J/kg)
                    # We keep internal variables in SI (J/kg) to match CoolProp calls
                    # Note: Pump.py logic uses J/kg for H and J/kg/K for S
                    h1 = self._lut_manager.lookup(self.fluid, 'H', P1_Pa, T1_K)
                    s1 = self._lut_manager.lookup(self.fluid, 'S', P1_Pa, T1_K)
                except (ValueError, RuntimeError):
                     # Fallback
                     h1 = CoolPropLUT.PropsSI('H', 'P', P1_Pa, 'T', T1_K, self.fluid)
                     s1 = CoolPropLUT.PropsSI('S', 'P', P1_Pa, 'T', T1_K, self.fluid)
            else:
                # Tier 2: Cached CoolProp
                h1 = CoolPropLUT.PropsSI('H', 'P', P1_Pa, 'T', T1_K, self.fluid)
                s1 = CoolPropLUT.PropsSI('S', 'P', P1_Pa, 'T', T1_K, self.fluid)
            
            # 2. Isentropic Outlet (Inverse lookup: H from P, S)
            # LUTManager usually doesn't support inverse efficiently without custom table
            # So we use CoolPropLUT (Cached)
            h2s = CoolPropLUT.PropsSI('H', 'P', P2_Pa, 'S', s1, self.fluid) # J/kg
            
            # 3. Real Work
            w_s = h2s - h1
            w_real = w_s / self.eta_is # J/kg
            
            h2 = h1 + w_real
            
            # 4. Outlet Temperature (Inverse lookup: T from P, H)
            T2_K = CoolPropLUT.PropsSI('T', 'P', P2_Pa, 'H', h2, self.fluid)
            self.outlet_temp_c = T2_K - 273.15
            
            # --- Thermodynamic Checks ---
            # Entropy check: Real process must increase entropy
            # s2 = CoolPropLUT.PropsSI('S', 'P', P2_Pa, 'H', h2, self.fluid)
            # if s2 < s1: logger.debug("Entropy decrease detected (numerical noise or bad efficiency)")
            
            # 5. Power
            mass_flow_kg_s = self.flow_rate_kg_h / 3600.0
            fluid_power_w = mass_flow_kg_s * w_real
            shaft_power_w = fluid_power_w / self.eta_m
            self.power_kw = shaft_power_w / 1000.0
            
            # 6. Create Output Stream
            self.outlet_stream = Stream(
                mass_flow_kg_h=self.flow_rate_kg_h,
                temperature_k=T2_K,
                pressure_pa=P2_Pa,
                composition=inlet.composition,
                phase="liquid" # Pumps handle liquid water
            )
            
        except Exception as e:
            logger.error(f"Pump {self.component_id} CoolProp error: {e}. Falling back.")
            self._pump_simplified(inlet)

    def _pump_simplified(self, inlet: Stream) -> None:
        """Hydraulic power estimate: W = V * dP"""
        P1_Pa = inlet.pressure_pa
        P2_Pa = self.target_pressure_pa
        dP_Pa = P2_Pa - P1_Pa
        
        # Density approx 1000 kg/m3
        rho = 1000.0
        mass_flow_kg_s = self.flow_rate_kg_h / 3600.0
        vol_flow_m3_s = mass_flow_kg_s / rho
        
        hydraulic_power_w = vol_flow_m3_s * dP_Pa
        self.power_kw = (hydraulic_power_w / (self.eta_is * self.eta_m)) / 1000.0
        
        # Neglect temp rise
        self.outlet_temp_c = inlet.temperature_k - 273.15
        
        self.outlet_stream = Stream(
            mass_flow_kg_h=self.flow_rate_kg_h,
            temperature_k=inlet.temperature_k,
            pressure_pa=P2_Pa,
            composition=inlet.composition,
            phase="liquid"
        )

    def get_state(self) -> Dict[str, Any]:
        return {
            **super().get_state(),
            "power_kw": self.power_kw,
            "flow_rate_kg_h": self.flow_rate_kg_h,
            "outlet_temp_c": self.outlet_temp_c,
            "target_pressure_bar": self.target_pressure_pa / 1e5
        }
