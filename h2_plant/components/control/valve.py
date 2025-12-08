
import logging
from typing import Dict, Any, Optional

from h2_plant.core.component import Component
from h2_plant.core.stream import Stream
from h2_plant.optimization.coolprop_lut import CoolPropLUT

logger = logging.getLogger(__name__)

class Valve(Component):
    """
    Isenthalpic throttling valve.
    Adiabatic expansion process where Inlet Enthalpy equals Outlet Enthalpy.
    Used for pressure reduction (Joule-Thomson effect).
    """
    def __init__(self, component_id: str = "valve"):
        super().__init__()
        self.component_id = component_id
        self.inlet_stream: Optional[Stream] = None
        self.outlet_stream: Optional[Stream] = None
        
        # Configurable properties
        self.target_pressure_bar: float = 0.0

    def receive_input(self, port_name: str, value: Any, resource_type: str = None) -> Any:
        if port_name == "fluid_in" and isinstance(value, Stream):
            self.inlet_stream = value
            return value.mass_flow_kg_h
        return 0.0

    def step(self, t: float) -> None:
        super().step(t)
        
        # Guard clause: Check inputs
        if not self.inlet_stream or self.inlet_stream.mass_flow_kg_h <= 0:
            self.outlet_stream = None
            return
            
        # Default target if not set (pass-through fallback)
        if self.target_pressure_bar <= 0:
             self.outlet_stream = self.inlet_stream
             return

        P_in = self.inlet_stream.pressure_pa
        P_out_target = self.target_pressure_bar * 1e5
        
        # Throttling requires P_out < P_in
        if P_out_target >= P_in:
            logger.warning(f"Valve {self.component_id}: Target P_out ({P_out_target/1e5:.2f} bar) >= P_in ({P_in/1e5:.2f} bar). Assuming fully open (pass-through).")
            self.outlet_stream = Stream(
                mass_flow_kg_h=self.inlet_stream.mass_flow_kg_h,
                temperature_k=self.inlet_stream.temperature_k,
                pressure_pa=P_in,
                composition=self.inlet_stream.composition,
                phase=self.inlet_stream.phase
            )
            return

        T_in = self.inlet_stream.temperature_k
        
        # Detect fluid for CoolProp
        fluid = "Hydrogen" # Default
        comp = self.inlet_stream.composition
        if comp.get("H2", 0) > 0.5: fluid = "Hydrogen"
        elif comp.get("O2", 0) > 0.5: fluid = "Oxygen"
        elif comp.get("H2O", 0) > 0.5: fluid = "Water"
        elif comp.get("N2", 0) > 0.5: fluid = "Nitrogen"
        
        try:
            # 1. Calc Inlet Enthalpy (H_in)
            # PropsSI(Output, Name1, Value1, Name2, Value2, Fluid)
            H_in = CoolPropLUT.PropsSI('H', 'T', T_in, 'P', P_in, fluid)
            
            if H_in == 0.0 and T_in > 100: # Simple validity check, Cp might return 0 on error
                 logger.warning(f"Valve {self.component_id}: CoolProp returned 0 enthalpy. Using Ideal Gas approximation.")
                 # Fallback: Ideal gas T_out = T_in (dH = Cp*dT = 0 -> dT=0)
                 # H2 has positive Joule-Thomson at ambient (heats up), Ideal gas has 0.
                 T_out = T_in 
            else:
                # 2. Isenthalpic Expansion: H_out = H_in
                H_out = H_in
                
                # 3. Calc Outlet Temperature (T_out)
                T_out = CoolPropLUT.PropsSI('T', 'P', P_out_target, 'H', H_out, fluid)
                
                # Sanity check
                if T_out <= 0:
                    logger.warning(f"Valve {self.component_id}: Failed to solve for T_out. Fallback to T_in.")
                    T_out = T_in

            self.outlet_stream = Stream(
                mass_flow_kg_h=self.inlet_stream.mass_flow_kg_h,
                temperature_k=T_out,
                pressure_pa=P_out_target,
                composition=self.inlet_stream.composition,
                phase='gas' # Throttling usually results in gas or 2-phase. Assuming gas for now.
            )
            
        except Exception as e:
            logger.error(f"Valve {self.component_id} thermo error: {e}")
            # Fallback pass-through to avoid crushing simulation
            self.outlet_stream = self.inlet_stream

    def get_output(self, port_name: str) -> Any:
        if port_name == "fluid_out":
            return self.outlet_stream
        return None
        
    def get_ports(self) -> Dict[str, Dict[str, str]]:
        return {
            'fluid_in': {'type': 'input', 'resource_type': 'stream'},
            'fluid_out': {'type': 'output', 'resource_type': 'stream'}
        }
