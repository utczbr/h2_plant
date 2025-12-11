from typing import Any, Dict, Optional
from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.component_ids import ComponentID
from h2_plant.core.stream import Stream
import logging

logger = logging.getLogger(__name__)

class ThrottlingValve(Component):
    """
    Throttling Valve (Joule-Thomson Expansion).
    
    Models an isoenthalpic pressure drop (H_in = H_out).
    Uses real-gas properties from LUTManager to calculate
    the temperature change (often cooling for gases, heating for H2
    at certain conditions, though H2 inversion temp is ~200K, 
    so at room temp it actually HEATS up upon expansion!).
    
    Wait, H2 Joule-Thomson inversion temperature is approx 202 K.
    Above this temperature (e.g. at 300K), the Joule-Thomson coefficient is NEGATIVE.
    This means dP < 0 implies dT > 0. Hydrogen HEATS UP when throttled at room temp.
    
    The LUTManager lookup will correctly capture this physical reality.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.P_out_pa = float(config.get('P_out_pa', 101325.0)) # Default 1 atm
        self.fluid = config.get('fluid', 'H2')
        self.lut_mgr = None
        
        # State
        self.inlet_stream: Optional[Stream] = None
        self.outlet_stream: Optional[Stream] = None
        
        # Performance metrics
        self.delta_T = 0.0

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        super().initialize(dt, registry)
        if registry.has(ComponentID.LUT_MANAGER.value):
            self.lut_mgr = registry.get(ComponentID.LUT_MANAGER)
        else:
            logger.warning(f"ThrottlingValve {self.component_id}: LUTManager not found. Thermo will be simplified.")

    def receive_input(self, port_name: str, value: Any, resource_type: str) -> float:
        if port_name == 'inlet':
            if isinstance(value, Stream):
                self.inlet_stream = value
                return value.mass_flow_kg_h
        return 0.0

    def step(self, t: float) -> None:
        super().step(t)
        
        if self.inlet_stream is None or self.inlet_stream.mass_flow_kg_h <= 0:
            self.outlet_stream = None
            self.delta_T = 0.0
            return
            
        # If output pressure is higher than input, we cannot flow (check valve logic implicitly)
        # or we assume we just pass through at P_in? 
        # Usually a throttling valve regulates P_out ONLY if P_in > P_out.
        
        P_in = self.inlet_stream.pressure_pa
        P_target = self.P_out_pa
        
        if P_target >= P_in:
            # Valve wide open or backflow condition (prevented here by just passing P_in)
            # Or effectively no throttling.
            self.outlet_stream = Stream(
                mass_flow_kg_h=self.inlet_stream.mass_flow_kg_h,
                temperature_k=self.inlet_stream.temperature_k,
                pressure_pa=P_in, # No drop
                composition=self.inlet_stream.composition,
                phase=self.inlet_stream.phase
            )
            self.delta_T = 0.0
            return

        # Perform Isoenthalpic Expansion (H_in = H_out)
        T_out = self.inlet_stream.temperature_k # Guess
        
        if self.lut_mgr:
            try:
                # Get Enthalpy at Input State
                h_in = self.lut_mgr.lookup(
                    self.fluid, 'H', P_in, self.inlet_stream.temperature_k
                )
                
                # Solve for T_out such that H(P_target, T_out) = h_in
                T_out = self._solve_T_isoenthalpic(P_target, h_in)
                
            except Exception as e:
                logger.error(f"Valve LUT error: {e}")
                # Fallback: Ideal Gas (T_out = T_in for ideal gas throttling, 
                # but real H2 heats up. We'll stick to T_in if calculation fails)
                T_out = self.inlet_stream.temperature_k
        
        self.delta_T = T_out - self.inlet_stream.temperature_k
        
        self.outlet_stream = Stream(
            mass_flow_kg_h=self.inlet_stream.mass_flow_kg_h,
            temperature_k=T_out,
            pressure_pa=P_target,
            composition=self.inlet_stream.composition,
            phase=self.inlet_stream.phase
        )

    def get_output(self, port_name: str) -> Any:
        if port_name == 'outlet':
            return self.outlet_stream
        return None
        
    def get_state(self) -> Dict[str, Any]:
        return {
            **super().get_state(),
            "inlet_pressure_bar": (self.inlet_stream.pressure_pa / 1e5) if self.inlet_stream else 0.0,
            "outlet_pressure_bar": self.P_out_pa / 1e5,
            "flow_rate_kg_h": self.inlet_stream.mass_flow_kg_h if self.inlet_stream else 0.0,
            "delta_T_K": self.delta_T
        }

    def _solve_T_isoenthalpic(self, P_target_pa: float, h_target: float) -> float:
        """
        Solve H(P_target, T) = h_target for T using bisection.
        Assumes H is monotonic with T (mostly true).
        """
        # Bounds for bisection (Realistically T shouldn't change MASSIVELY)
        # H2 at 300K expanding max 300 bar -> might change 5-10K?
        T_guess = self.inlet_stream.temperature_k
        T_low = T_guess - 50.0
        T_high = T_guess + 50.0
        
        # Refine bounds if needed
        h_low = self.lut_mgr.lookup(self.fluid, 'H', P_target_pa, T_low)
        h_high = self.lut_mgr.lookup(self.fluid, 'H', P_target_pa, T_high)
        
        # Check monotony briefly or expand bounds
        if not (h_low < h_target < h_high):
            # Try wider bounds
            T_low = 20.0 # Near zero
            T_high = 1000.0
            h_low = self.lut_mgr.lookup(self.fluid, 'H', P_target_pa, T_low)
            h_high = self.lut_mgr.lookup(self.fluid, 'H', P_target_pa, T_high)
            
            if not (h_low < h_target < h_high):
                # Out of range
                return T_guess

        # Bisection
        for _ in range(20): # 20 iterations is plenty for float precision
            T_mid = (T_low + T_high) / 2
            h_mid = self.lut_mgr.lookup(self.fluid, 'H', P_target_pa, T_mid)
            
            if abs(h_mid - h_target) < 1.0: # 1 J/kg tolerance? Maybe too tight, but okay.
                return T_mid
            
            if h_mid < h_target:
                T_low = T_mid
            else:
                T_high = T_mid
                
        return (T_low + T_high) / 2
