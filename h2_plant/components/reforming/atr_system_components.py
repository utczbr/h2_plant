
from typing import Dict, Any, Optional
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.stream import Stream
from h2_plant.components.reforming.atr_data_manager import ATRBaseComponent, C_TO_K, KW_TO_W

class ATRSystemCompressor(ATRBaseComponent):
    """
    Represents the total work input for the ATR plant (C01, C02, Pumps).
    Maps to 'W_in_func' in the ATR surrogate model.
    """
    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        super().initialize(dt, registry)
        self.lookup_id = "W_in"
        self.outlet_stream = Stream(mass_flow_kg_h=0.0)
        self._input_stream = None

    def step(self, t: float) -> None:
        super().step(t)
        
        # Pass-through for mass flow
        in_stream = getattr(self, '_input_stream', None)
        out_stream = None
        
        if in_stream:
            out_stream = in_stream.copy()
            f_o2 = self.get_oxygen_flow({'inlet': in_stream})
        else:
            # If no stream is connected, we still calculate power based on global O2 setting
            # Assumes config has been set by controller
            f_o2 = float(self.config.get('current_o2_flow_kmol_h', 15.0))

        # Lookup Power
        # W_in_func unit is kW
        power_kw = self.data_manager.lookup('W_in_func', f_o2)
        
        # Log power consumption (positive for consumption)
        self.results['power_consumption_kw'] = power_kw
        
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
            "type": "ATR_SystemCompressor",
            "power_consumption_kw": self.results.get('power_consumption_kw', 0.0)
        }

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        return {
            'inlet': {'type': 'input', 'resource_type': 'stream'},
            'outlet': {'type': 'output', 'resource_type': 'stream'}
        }


class ATRProductSeparator(ATRBaseComponent):
    """
    Combines the logic of the Separator (SEP) and PSA units.
    Splits the reactor output into final product streams based on the surrogate model.
    
    Outputs:
        - 'hydrogen': Pure H2 stream
        - 'offgas': Tail gas with specific composition
        - 'water': Condensate water
    """
    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        super().initialize(dt, registry)
        self.h2_stream = Stream(mass_flow_kg_h=0.0)
        self.offgas_stream = Stream(mass_flow_kg_h=0.0)
        self.water_stream = Stream(mass_flow_kg_h=0.0)
        self._input_stream = None

    def step(self, t: float) -> None:
        super().step(t)
        # We need an input stream to determine connectivity, but the model is driven by F_O2
        in_stream = getattr(self, '_input_stream', None)
        
        # If we have an input stream, we can use it to find F_O2 if implicit
        # Otherwise fallback to config
        if in_stream:
             f_o2 = self.get_oxygen_flow({'inlet': in_stream})
        else:
             f_o2 = float(self.config.get('current_o2_flow_kmol_h', 15.0))
        
        # 1. Lookup Molar Flows (kmol/hr)
        f_h2_kmol = self.data_manager.lookup('F_H2_func', f_o2)
        f_offgas_kmol = self.data_manager.lookup('F_offgas_func', f_o2)
        f_water_kmol = self.data_manager.lookup('F_water_func', f_o2)
        
        # 2. Lookup Offgas Composition (Mole Fractions)
        x_co2 = self.data_manager.lookup('xCO2_offgas_func', f_o2)
        x_h2_off = self.data_manager.lookup('xH2_offgas_func', f_o2)
        x_ch4 = self.data_manager.lookup('xCH4_offgas_func', f_o2)
        # Remainder will be balanced with H2O in composition dictionary
        # remainder = max(0.0, 1.0 - (x_co2 + x_h2_off + x_ch4))
        
        # 3. Create Output Streams
        
        # Stream 1: Hydrogen
        # Assuming 100% purity for F_H2_func stream
        h2_mass_flow = f_h2_kmol * 2.016 # kg/kmol H2
        s_h2 = Stream(mass_flow_kg_h=h2_mass_flow)
        s_h2.composition = {'H2': 1.0}
        
        # Stream 2: Water
        water_mass_flow = f_water_kmol * 18.015 # kg/kmol H2O
        s_water = Stream(mass_flow_kg_h=water_mass_flow)
        s_water.composition = {'H2O': 1.0}
        
        # Stream 3: Offgas
        # Calculate average MW of offgas
        # MW_approx = x_i * MW_i ...
        mw_offgas = (x_co2 * 44.01) + (x_h2_off * 2.016) + (x_ch4 * 16.04) + (remainder * 28.01) # 28.01 for N2/CO
        offgas_mass_flow = f_offgas_kmol * mw_offgas
        
        s_offgas = Stream(mass_flow_kg_h=offgas_mass_flow)
        s_offgas.composition = {
            'CO2': x_co2,
            'H2': x_h2_off,
            'CH4': x_ch4,
            'CO': 0.01,
            'N2': 0.0,
            'H2O': max(0.0, 1.0 - (x_co2 + x_h2_off + x_ch4 + 0.01))
        }
        
        # Update results for UI/Logging
        self.results['H2_production_kg_h'] = h2_mass_flow
        self.results['Water_production_kg_h'] = water_mass_flow
        
        self.h2_stream = s_h2
        self.offgas_stream = s_offgas
        self.water_stream = s_water

    def receive_input(self, port_name: str, value: Any, resource_type: str = None) -> float:
        if port_name == 'inlet' and isinstance(value, Stream):
            self._input_stream = value
            return value.mass_flow_kg_h
        return 0.0

    def get_output(self, port_name: str) -> Any:
        if port_name == 'hydrogen':
            return self.h2_stream
        elif port_name == 'offgas':
            return self.offgas_stream
        elif port_name == 'water':
            return self.water_stream
        return None
        
    def get_state(self) -> Dict[str, Any]:
        return {
            "component_id": self.component_id,
            "type": "ATR_ProductSeparator",
            "h2_production_kg_h": self.results.get('H2_production_kg_h', 0.0),
            "water_production_kg_h": self.results.get('Water_production_kg_h', 0.0)
        }

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        return {
            'inlet': {'type': 'input', 'resource_type': 'stream'},
            'hydrogen': {'type': 'output', 'resource_type': 'stream', 'units': 'kg/h'},
            'offgas': {'type': 'output', 'resource_type': 'stream', 'units': 'kg/h'},
            'water': {'type': 'output', 'resource_type': 'stream', 'units': 'kg/h'}
        }
