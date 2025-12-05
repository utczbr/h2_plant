"""
Detailed ATR (Auto-Thermal Reforming) system.

Models the complete ATR system including:
- Reforming Reactor (ATR)
- Water Gas Shift (WGS) reactors (High and Low Temp)
- Heat recovery network
- Gas separation and purification
"""

from typing import Dict, Any
from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.composite_component import CompositeComponent
from h2_plant.components.production.pem_electrolyzer_detailed import (
    HeatExchanger, SeparationTank, PSAUnit
)
from h2_plant.components.production.soec_electrolyzer_detailed import (
    SteamGenerator, ProcessCompressor
)

# ============================================================================
# SUBSYSTEM: Reactors
# ============================================================================

class ATRReactor(Component):
    """
    Auto-Thermal Reforming Reactor.
    CH4 + H2O -> 3H2 + CO
    CH4 + 2O2 -> CO2 + 2H2O (Combustion for heat)
    """
    def __init__(self, reactor_id: str, max_flow_kg_h: float):
        super().__init__()
        self.reactor_id = reactor_id
        self.max_flow_kg_h = max_flow_kg_h
        self.biogas_input_kg_h = 0.0
        self.steam_input_kg_h = 0.0
        self.syngas_output_kg_h = 0.0
        self.temp_c = 900.0
        
    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        super().initialize(dt, registry)
        
    def step(self, t: float) -> None:
        super().step(t)
        # Simplified mass balance
        # Input: Biogas (CH4) + Steam (H2O)
        # Output: Syngas (H2 + CO + CO2 + H2O unreacted)
        
        total_in = self.biogas_input_kg_h + self.steam_input_kg_h
        self.syngas_output_kg_h = total_in # Mass conservation
        
    def get_state(self) -> Dict[str, Any]:
        return {
            **super().get_state(),
            'syngas_output_kg_h': float(self.syngas_output_kg_h)
        }

class WGSReactor(Component):
    """
    Water Gas Shift Reactor.
    CO + H2O -> CO2 + H2
    """
    def __init__(self, wgs_id: str, conversion_rate: float = 0.7):
        super().__init__()
        self.wgs_id = wgs_id
        self.conversion_rate = conversion_rate
        self.syngas_input_kg_h = 0.0
        self.syngas_output_kg_h = 0.0
        
    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        super().initialize(dt, registry)
        
    def step(self, t: float) -> None:
        super().step(t)
        # Mass is conserved, composition changes
        self.syngas_output_kg_h = self.syngas_input_kg_h
        
    def get_state(self) -> Dict[str, Any]:
        return {
            **super().get_state(),
            'syngas_output_kg_h': float(self.syngas_output_kg_h)
        }

# ============================================================================
# COMPOSITE: Detailed ATR System
# ============================================================================

class DetailedATRProduction(CompositeComponent):
    """
    Complete ATR system matching Process Flow.csv.
    """
    
    def __init__(self, max_biogas_kg_h: float = 500.0):
        super().__init__()
        
        # 1. Feed Preparation
        self.add_subsystem('compressor_c3', ProcessCompressor('C-3', max_biogas_kg_h))
        self.add_subsystem('steam_gen_hx7', SteamGenerator('HX-7', 1000.0))
        
        # 2. Reforming
        self.add_subsystem('atr_reactor', ATRReactor('ATR', max_biogas_kg_h * 3))
        self.add_subsystem('hx8', HeatExchanger('HX-8', 500.0)) # Cooling before WGS
        
        # 3. Shift Conversion
        self.add_subsystem('wgs_ht', WGSReactor('WGS-HT', 0.7))
        self.add_subsystem('hx9', HeatExchanger('HX-9', 400.0))
        self.add_subsystem('wgs_lt', WGSReactor('WGS-LT', 0.9))
        self.add_subsystem('hx10', HeatExchanger('HX-10', 400.0))
        
        # 4. Separation & Purification
        self.add_subsystem('separator_st4', SeparationTank('ST-4', 'Syngas'))
        self.add_subsystem('compressor_c4', ProcessCompressor('C-4', 500.0))
        self.add_subsystem('psa_d4', PSAUnit('D-4', 'H2'))
        
        # Inputs
        self.biogas_input_kg_h = 0.0
        self.water_input_kg_h = 0.0
        
    def step(self, t: float) -> None:
        Component.step(self, t)
        
        # 1. Feed
        self.compressor_c3.input_flow_kg_h = self.biogas_input_kg_h
        self.compressor_c3.step(t)
        
        self.steam_gen_hx7.water_input_kg_h = self.water_input_kg_h
        self.steam_gen_hx7.step(t)
        
        # 2. Reforming
        self.atr_reactor.biogas_input_kg_h = self.compressor_c3.output_flow_kg_h
        self.atr_reactor.steam_input_kg_h = self.steam_gen_hx7.steam_output_kg_h
        self.atr_reactor.step(t)
        
        self.hx8.inlet_flow_kg_h = self.atr_reactor.syngas_output_kg_h
        self.hx8.step(t)
        
        # 3. Shift
        self.wgs_ht.syngas_input_kg_h = self.hx8.inlet_flow_kg_h # Assuming HX passes flow through
        # Note: HeatExchanger in PEM model doesn't explicitly have an 'outlet_flow' property that mirrors inlet
        # but logically flow in = flow out. We'll use inlet_flow for next stage.
        self.wgs_ht.step(t)
        
        self.hx9.inlet_flow_kg_h = self.wgs_ht.syngas_output_kg_h
        self.hx9.step(t)
        
        self.wgs_lt.syngas_input_kg_h = self.hx9.inlet_flow_kg_h
        self.wgs_lt.step(t)
        
        self.hx10.inlet_flow_kg_h = self.wgs_lt.syngas_output_kg_h
        self.hx10.step(t)
        
        # 4. Separation
        self.separator_st4.gas_inlet_kg_h = self.hx10.inlet_flow_kg_h
        self.separator_st4.step(t)
        
        self.compressor_c4.input_flow_kg_h = self.separator_st4.dry_gas_outlet_kg_h
        self.compressor_c4.step(t)
        
        self.psa_d4.feed_gas_kg_h = self.compressor_c4.output_flow_kg_h
        self.psa_d4.step(t)
        
    @property
    def h2_product_kg_h(self):
        return self.psa_d4.product_gas_kg_h
        
    @property
    def tail_gas_kg_h(self):
        return self.psa_d4.waste_gas_kg_h
        
    @property
    def recycled_water_kg_h(self):
        return self.separator_st4.water_return_kg_h

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state['summary'] = {
            'h2_product_kg_h': self.h2_product_kg_h,
            'tail_gas_kg_h': self.tail_gas_kg_h,
            'recycled_water_kg_h': self.recycled_water_kg_h
        }
        return state
