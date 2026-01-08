from typing import Dict, Any
import logging
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.stream import Stream
from h2_plant.components.reforming.atr_data_manager import ATRBaseComponent, C_TO_K, KW_TO_W

logger = logging.getLogger(__name__)

class Boiler(ATRBaseComponent):
    """
    Represents Heaters H01, H02, H04.
    """
    def __init__(self, component_id: str = None, lookup_id: str = None):
        super().__init__(component_id)
        self.lookup_id = lookup_id or component_id

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        super().initialize(dt, registry)
        # Validate ID mapping
        if self.lookup_id not in ['H01', 'H02', 'H04']:
            # Soft warning
            pass
        
        # Initialize outlet stream to prevent NoneType downstream
        self.outlet_stream = Stream(mass_flow_kg_h=0.0)
        self._input_stream = None

    def step(self, t: float) -> None:
        # Note: Component.step() signature is usually step(self, t: float)
        # The user provided example used step(self, inputs, dt) which is not standard Layer 1.
        # We must follow Layer 1: Inputs are pulled via self.get_input(), Outputs via self.outlet_stream or similar.
        # Let's adapt the user's logic to the standard Component Lifecycle.
        super().step(t)
        
        # 1. Get Input
        in_stream = self.get_input('inlet')
        if not in_stream:
             return

        # 2. Get Operating Point (F_O2)
        # In Layer 1, streams are passed between components, so we inspect inputs
        # But get_oxygen_flow relies on self.config or specific logic
        # We'll pass a dummy dict for compatibility with the user's logic method signature if needed
        # or just reimplement get_oxygen_flow logic here.
        # Implemented logic:
        
        f_o2 = self.get_oxygen_flow({'inlet': in_stream})
        
        # If this is H02 (Oxygen Heater), attempt reverse calc
        if self.lookup_id == 'H02':
            mass_flow_kg_hr = in_stream.mass_flow_kg_h
            calc_f_o2 = mass_flow_kg_hr / 32.0 
            if 7.125 <= calc_f_o2 <= 23.75:
                f_o2 = calc_f_o2

        # 3. Map Keys
        q_func = f"{self.lookup_id}_Q_func"
        tout_func = f"Tout_{self.lookup_id}_func"
        
        # 4. Clone and Apply
        out_stream = in_stream.copy()
        
        # Function from base class
        # Note: base class _apply_thermal_model signature: (stream, f_o2, q, t)
        self._apply_thermal_model(out_stream, f_o2, q_func, tout_func)
        
        self.outlet_stream = out_stream

    def get_input(self, port_name: str):
        return getattr(self, '_input_stream', None)

    def receive_input(self, port_name: str, value: Any, resource_type: str = None) -> float:
        if port_name == 'inlet' and isinstance(value, Stream):
            self._input_stream = value
            return value.mass_flow_kg_h
        return 0.0

    def get_output(self, port_name: str) -> Any:
        if port_name == 'outlet':
            return getattr(self, 'outlet_stream', None)
        return None
    
    def get_state(self) -> Dict[str, Any]:
        return {
            "component_id": self.component_id,
            "lookup_id": getattr(self, 'lookup_id', None),
            "input_flow_kg_h": getattr(self, '_input_stream', None).mass_flow_kg_h if getattr(self, '_input_stream', None) else 0.0,
            "outlet_temp_k": self.outlet_stream.temperature_k if getattr(self, 'outlet_stream', None) else 0.0
        }

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        return {
            'inlet': {'type': 'input', 'resource_type': 'stream'},
            'outlet': {'type': 'output', 'resource_type': 'stream'}
        }


class HeatExchanger(ATRBaseComponent):
    """
    Represents Coolers H05, etc.
    """
    def __init__(self, component_id: str = None, lookup_id: str = None):
        super().__init__(component_id)
        self.lookup_id = lookup_id or component_id

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        super().initialize(dt, registry)
        self.outlet_stream = Stream(mass_flow_kg_h=0.0)
        self._input_stream = None

    def step(self, t: float) -> None:
        super().step(t)
        in_stream = getattr(self, '_input_stream', None)
        if not in_stream: return
        
        f_o2 = self.get_oxygen_flow({'inlet': in_stream})
        
        q_func = f"{self.lookup_id}_Q_func"
        tout_func = f"Tout_{self.lookup_id}_func"
        
        out_stream = in_stream.copy()
        self._apply_thermal_model(out_stream, f_o2, q_func, tout_func)
        self.outlet_stream = out_stream

    def receive_input(self, port_name: str, value: Any, resource_type: str = None) -> float:
        if port_name == 'inlet' and isinstance(value, Stream):
            self._input_stream = value
            return value.mass_flow_kg_h
        return 0.0

    def get_output(self, port_name: str) -> Any:
        if port_name == 'outlet':
            return getattr(self, 'outlet_stream', None)
        return None
        
    def get_state(self) -> Dict[str, Any]:
        return {
            "component_id": self.component_id,
            "lookup_id": getattr(self, 'lookup_id', None),
            "input_flow_kg_h": getattr(self, '_input_stream', None).mass_flow_kg_h if getattr(self, '_input_stream', None) else 0.0,
            "outlet_temp_k": self.outlet_stream.temperature_k if getattr(self, 'outlet_stream', None) else 0.0
        }

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        return {
            'inlet': {'type': 'input', 'resource_type': 'stream'},
            'outlet': {'type': 'output', 'resource_type': 'stream'}
        }

class HTWGS(ATRBaseComponent):
    """
    High Temperature Water Gas Shift + Cooler H08.
    Uses regression table for Thermal Setpoints (T_out) and assumes Chemical Equilibrium at that T.
    """
    def step(self, t: float) -> None:
        super().step(t)
        in_stream = getattr(self, '_input_stream', None)
        if not in_stream: return

        f_o2 = self.get_oxygen_flow({'inlet': in_stream})
        
        # 1. Lookup Thermal Targets from Table
        duty_kw = self.data_manager.lookup('H08_Q_func', f_o2)
        t_out_c = self.data_manager.lookup('Tout_H08_func', f_o2)
        target_temp_k = t_out_c + C_TO_K

        # 2. Results Dictionary
        self.results['duty_w'] = duty_kw * KW_TO_W
        
        # 3. Solve Chemical Equilibrium at Target Temperature
        # Reaction: CO + H2O <-> CO2 + H2
        # Keq(T) approximation: log10(K) = 2073/T - 2.029
        K_eq = 10**(2073.0 / target_temp_k - 2.029)
        
        # Moles
        comp = in_stream.composition
        total_flow_kmol = in_stream.mass_flow_kg_h / self.calculate_mw(comp) if in_stream.mass_flow_kg_h > 0 else 0
        
        n_co = comp.get('CO', 0) * total_flow_kmol
        n_h2o = comp.get('H2O', 0) * total_flow_kmol
        n_co2 = comp.get('CO2', 0) * total_flow_kmol
        n_h2 = comp.get('H2', 0) * total_flow_kmol
        
        if n_co > 1e-6 and n_h2o > 1e-6:
            # Solve Quadratic: K = ( (CO2+x)(H2+x) ) / ( (CO-x)(H2O-x) )
            # (1 - K)x^2 + (CO2 + H2 + K(CO + H2O))x + (CO2*H2 - K*CO*H2O) = 0
            
            # Coefficients
            A = 1.0 - K_eq
            B = n_co2 + n_h2 + K_eq * (n_co + n_h2o)
            C = n_co2 * n_h2 - K_eq * n_co * n_h2o
            
            if abs(A) < 1e-9:
                # Linear case: Bx + C = 0 -> x = -C/B
                xi = -C / B
            else:
                # Quadratic formula
                delta = B*B - 4*A*C
                if delta >= 0:
                    sqrt_delta = delta**0.5
                    x1 = (-B + sqrt_delta) / (2*A)
                    x2 = (-B - sqrt_delta) / (2*A)
                    
                    # Choose valid root (must calculate positive moles)
                    # Limit: -min(CO2, H2) <= x <= min(CO, H2O)
                    max_x = min(n_co, n_h2o)
                    min_x = -min(n_co2, n_h2)
                    
                    if min_x - 1e-6 <= x1 <= max_x + 1e-6:
                        xi = x1
                    elif min_x - 1e-6 <= x2 <= max_x + 1e-6:
                        xi = x2
                    else:
                        xi = 0.0 # Error fallback
                else:
                    xi = 0.0
            
            # Apply Extent
            n_co_new = n_co - xi
            n_h2o_new = n_h2o - xi
            n_co2_new = n_co2 + xi
            n_h2_new = n_h2 + xi
            
            # DEBUG LOGGING
            if int(t * 60) % 60 == 0:
                 conv = (xi / n_co * 100) if n_co > 0 else 0
                 logger.info(f"WGS [{self.component_id}] T={target_temp_k:.0f}K, xi={xi:.3f}, CO_in={n_co/total_flow_kmol*100:.1f}%, CO_out={n_co_new/total_flow_kmol*100:.1f}%, Conv={conv:.1f}%")
            
        else:
            xi = 0.0
            n_co_new, n_h2o_new, n_co2_new, n_h2_new = n_co, n_h2o, n_co2, n_h2
            
        # 4. Construct Output Stream
        out_comp = comp.copy()
        out_comp['CO'] = n_co_new / total_flow_kmol
        out_comp['H2O'] = n_h2o_new / total_flow_kmol
        out_comp['CO2'] = n_co2_new / total_flow_kmol
        out_comp['H2'] = n_h2_new / total_flow_kmol
        
        # Calculate new Mass Flow (Mass is conserved in WGS, simple check)
        # But MW changes slightly? No, atoms conserved so mass conserved.
        
        out_stream = in_stream.copy()
        out_stream.temperature_k = target_temp_k
        out_stream.composition = out_comp
        
        self.outlet_stream = out_stream
        self.outlet_temp_k = target_temp_k

    def calculate_mw(self, comp):
        MW = {'CO': 28.01, 'H2O': 18.015, 'CO2': 44.01, 'H2': 2.016, 'CH4': 16.04, 'N2': 28.014}
        return sum(frac * MW.get(s, 0) for s, frac in comp.items())

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        super().initialize(dt, registry)
        self.outlet_stream = Stream(mass_flow_kg_h=0.0)
        self._input_stream = None


    def receive_input(self, port_name: str, value: Any, resource_type: str = None) -> float:
        if port_name == 'inlet' and isinstance(value, Stream):
            self._input_stream = value
            return value.mass_flow_kg_h
        return 0.0

    def get_output(self, port_name: str) -> Any:
        if port_name == 'outlet':
            return getattr(self, 'outlet_stream', None)
        return None
        
    def get_state(self) -> Dict[str, Any]:
        return {
            "component_id": self.component_id,
            "type": "HTWGS",
            "duty_kw": self.results.get('duty_w', 0.0) / 1000.0,
            "outlet_temp_k": self.outlet_stream.temperature_k if getattr(self, 'outlet_stream', None) else 0.0
        }

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        return {
            'inlet': {'type': 'input', 'resource_type': 'stream'},
            'outlet': {'type': 'output', 'resource_type': 'stream'}
        }

class LTWGS(ATRBaseComponent):
    """
    Low Temperature Water Gas Shift + Cooler H09.
    Uses regression table for Thermal Setpoints (T_out) and assumes Chemical Equilibrium at that T.
    """
    def step(self, t: float) -> None:
        super().step(t)
        in_stream = getattr(self, '_input_stream', None)
        if not in_stream: return

        f_o2 = self.get_oxygen_flow({'inlet': in_stream})
        
        # 1. Lookup Thermal Targets from Table
        duty_kw = self.data_manager.lookup('H09_Q_func', f_o2)
        t_out_c = self.data_manager.lookup('Tout_H09_func', f_o2)
        target_temp_k = t_out_c + C_TO_K
        
        self.results['duty_w'] = duty_kw * KW_TO_W
        
        # 2. Solve Chemical Equilibrium at Target Temperature
        K_eq = 10**(2073.0 / target_temp_k - 2.029)
        
        comp = in_stream.composition
        total_flow_kmol = in_stream.mass_flow_kg_h / self.calculate_mw(comp) if in_stream.mass_flow_kg_h > 0 else 0
        
        n_co = comp.get('CO', 0) * total_flow_kmol
        n_h2o = comp.get('H2O', 0) * total_flow_kmol
        n_co2 = comp.get('CO2', 0) * total_flow_kmol
        n_h2 = comp.get('H2', 0) * total_flow_kmol
        
        if n_co > 1e-6 and n_h2o > 1e-6:
            A = 1.0 - K_eq
            B = n_co2 + n_h2 + K_eq * (n_co + n_h2o)
            C = n_co2 * n_h2 - K_eq * n_co * n_h2o
            
            if abs(A) < 1e-9:
                xi = -C / B
            else:
                delta = B*B - 4*A*C
                if delta >= 0:
                    sqrt_delta = delta**0.5
                    x1 = (-B + sqrt_delta) / (2*A)
                    x2 = (-B - sqrt_delta) / (2*A)
                    max_x = min(n_co, n_h2o)
                    min_x = -min(n_co2, n_h2)
                    if min_x - 1e-6 <= x1 <= max_x + 1e-6: xi = x1
                    elif min_x - 1e-6 <= x2 <= max_x + 1e-6: xi = x2
                    else: xi = 0.0
                else: xi = 0.0
            
            # Apply Extent
            n_co_new = n_co - xi
            n_h2o_new = n_h2o - xi
            n_co2_new = n_co2 + xi
            n_h2_new = n_h2 + xi

            # DEBUG LOGGING
            if int(t * 60) % 60 == 0:
                 conv = (xi / n_co * 100) if n_co > 0 else 0
                 logger.info(f"WGS [{self.component_id}] T={target_temp_k:.0f}K, xi={xi:.3f}, CO_in={n_co/total_flow_kmol*100:.1f}%, CO_out={n_co_new/total_flow_kmol*100:.1f}%, Conv={conv:.1f}%")
        else:
            n_co_new, n_h2o_new, n_co2_new, n_h2_new = n_co, n_h2o, n_co2, n_h2
            
        out_comp = comp.copy()
        out_comp['CO'] = n_co_new / total_flow_kmol
        out_comp['H2O'] = n_h2o_new / total_flow_kmol
        out_comp['CO2'] = n_co2_new / total_flow_kmol
        out_comp['H2'] = n_h2_new / total_flow_kmol
        
        out_stream = in_stream.copy()
        out_stream.temperature_k = target_temp_k
        out_stream.composition = out_comp
        
        self.outlet_stream = out_stream
        self.outlet_temp_k = target_temp_k

    def calculate_mw(self, comp):
        MW = {'CO': 28.01, 'H2O': 18.015, 'CO2': 44.01, 'H2': 2.016, 'CH4': 16.04, 'N2': 28.014}
        return sum(frac * MW.get(s, 0) for s, frac in comp.items())
        
    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        super().initialize(dt, registry)
        self.outlet_stream = Stream(mass_flow_kg_h=0.0)
        self._input_stream = None

        
    def receive_input(self, port_name: str, value: Any, resource_type: str = None) -> float:
        if port_name == 'inlet' and isinstance(value, Stream):
            self._input_stream = value
            return value.mass_flow_kg_h
        return 0.0

    def get_output(self, port_name: str) -> Any:
        if port_name == 'outlet':
            return getattr(self, 'outlet_stream', None)
        return None
        
    def get_state(self) -> Dict[str, Any]:
        return {
            "component_id": self.component_id,
            "type": "LTWGS",
            "duty_kw": self.results.get('duty_w', 0.0) / 1000.0,
            "outlet_temp_k": self.outlet_stream.temperature_k if getattr(self, 'outlet_stream', None) else 0.0
        }

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        return {
            'inlet': {'type': 'input', 'resource_type': 'stream'},
            'outlet': {'type': 'output', 'resource_type': 'stream'}
        }
