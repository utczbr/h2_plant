
from typing import Dict, Any
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.stream import Stream
from h2_plant.components.reforming.atr_data_manager import ATRBaseComponent, C_TO_K, KW_TO_W

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
    """
    def step(self, t: float) -> None:
        super().step(t)
        in_stream = getattr(self, '_input_stream', None)
        if not in_stream: return

        f_o2 = self.get_oxygen_flow({'inlet': in_stream})
        
        # H08
        duty_kw = self.data_manager.lookup('H08_Q_func', f_o2)
        t_out_c = self.data_manager.lookup('Tout_H08_func', f_o2)
        
        out_stream = in_stream.copy()
        out_stream.temperature_k = t_out_c + C_TO_K
        self.results['duty_w'] = duty_kw * KW_TO_W
        self.outlet_stream = out_stream

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
    """
    def step(self, t: float) -> None:
        super().step(t)
        in_stream = getattr(self, '_input_stream', None)
        if not in_stream: return

        f_o2 = self.get_oxygen_flow({'inlet': in_stream})
        
        # H09
        duty_kw = self.data_manager.lookup('H09_Q_func', f_o2)
        t_out_c = self.data_manager.lookup('Tout_H09_func', f_o2)
        
        out_stream = in_stream.copy()
        out_stream.temperature_k = t_out_c + C_TO_K
        self.results['duty_w'] = duty_kw * KW_TO_W
        self.outlet_stream = out_stream
        
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
