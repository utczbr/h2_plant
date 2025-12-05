"""
Integrated Plant Coordinator.

Orchestrates the entire detailed plant simulation, wiring together:
- Water Treatment
- PEM, SOEC, ATR Production
- CO2 Capture
- Centralized Compression and Storage
"""

from typing import Dict, Any, List
import logging
from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.components.water.water_treatment_detailed import DetailedWaterTreatment
from h2_plant.components.production.pem_electrolyzer_detailed import DetailedPEMElectrolyzer, HeatExchanger
from h2_plant.components.production.soec_electrolyzer_detailed import DetailedSOECElectrolyzer, ProcessCompressor
from h2_plant.components.production.atr_production_detailed import DetailedATRProduction
from h2_plant.components.carbon.co2_capture_detailed import DetailedCO2Capture
from h2_plant.components.compression.filling_compressor import FillingCompressor
from h2_plant.components.storage.tank_array import TankArray

logger = logging.getLogger(__name__)

class IntegratedPlant(Component):
    """
    Coordinator for the full detailed plant.
    """
    
    def __init__(self):
        super().__init__()
        
        # 1. Water System
        self.water_system = DetailedWaterTreatment()
        
        # 2. Production Systems
        self.pem_system = DetailedPEMElectrolyzer(max_power_kw=2500.0)
        self.soec_system = DetailedSOECElectrolyzer(max_power_kw=1000.0)
        self.atr_system = DetailedATRProduction(max_biogas_kg_h=500.0)
        
        # 3. Carbon System
        self.co2_system = DetailedCO2Capture(max_flow_kg_h=200.0)
        
        # 4. Centralized Storage Chain
        # C-6: H2 Distribution -> Compressed H2
        self.compressor_c6 = FillingCompressor(
            max_flow_kg_h=500.0, 
            inlet_pressure_bar=30.0, 
            outlet_pressure_bar=60.0, # Intermediate pressure
            num_stages=1
        )
        
        # HX-11: Cooling
        self.chiller_hx11 = HeatExchanger('HX-11', 500.0)
        
        # LP Storage (H2 Storage LP)
        self.storage_lp = TankArray(n_tanks=5, capacity_kg=200.0, pressure_bar=60.0)
        
        # C-7: LP -> HP
        self.compressor_c7 = FillingCompressor(
            max_flow_kg_h=500.0,
            inlet_pressure_bar=60.0,
            outlet_pressure_bar=350.0,
            num_stages=3
        )
        
        # HP Storage (H2 Storage HP)
        self.storage_hp = TankArray(n_tanks=10, capacity_kg=100.0, pressure_bar=350.0)
        
        # Inputs
        self.external_water_input_kg_h = 2000.0
        self.biogas_input_kg_h = 400.0
        self.electricity_grid_kw = 0.0 # Tracked but not limited here
        
        # Demand
        self.h2_demand_kg_h = 0.0
        
    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        super().initialize(dt, registry)
        # Initialize all subsystems
        self.water_system.initialize(dt, registry)
        self.pem_system.initialize(dt, registry)
        self.soec_system.initialize(dt, registry)
        self.atr_system.initialize(dt, registry)
        self.co2_system.initialize(dt, registry)
        self.compressor_c6.initialize(dt, registry)
        self.chiller_hx11.initialize(dt, registry)
        self.storage_lp.initialize(dt, registry)
        self.compressor_c7.initialize(dt, registry)
        self.storage_hp.initialize(dt, registry)
        
    def step(self, t: float) -> None:
        Component.step(self, t)
        
        # --- 1. Water Supply ---
        self.water_system.external_water_input_kg_h = self.external_water_input_kg_h
        # Set demands (simplified logic: assume full capacity needed or driven by production targets)
        # For now, we'll just assume they take what they need if available, 
        # but in this push-model, we push water to them.
        
        # Distribute water equally or by priority? 
        # Let's give them fixed amounts for this simulation step
        available_water = self.water_system.tank_wt.current_mass_kg
        # Simplified: infinite supply from tank for this step, clamped by pipe size
        self.water_system.demand_pem_kg_h = 1000.0 
        self.water_system.demand_soec_kg_h = 500.0
        self.water_system.demand_atr_kg_h = 500.0
        
        self.water_system.step(t)
        
        # --- 2. Production ---
        
        # PEM
        # FeedwaterInlet in PEM needs to know how much water it gets
        # We need to inject this into the PEM system. 
        # The PEM system pulls from 'feedwater_inlet'.
        self.pem_system.feedwater_inlet.current_flow_kg_h = self.water_system.tank_wt.outlet_flow_kg_h * 0.5 # Split
        self.pem_system.step(t)
        
        # SOEC
        self.soec_system.water_input_kg_h = self.water_system.tank_wt.outlet_flow_kg_h * 0.25
        self.soec_system.step(t)
        
        # ATR
        # Mix fresh biogas with recycled gas
        recycled_gas = self.co2_system.recycled_gas_kg_h
        self.atr_system.biogas_input_kg_h = self.biogas_input_kg_h + recycled_gas
        self.atr_system.water_input_kg_h = self.water_system.tank_wt.outlet_flow_kg_h * 0.25
        self.atr_system.step(t)
        
        # --- 3. Carbon Capture ---
        self.co2_system.tail_gas_input_kg_h = self.atr_system.tail_gas_kg_h
        self.co2_system.step(t)
        
        # --- 4. H2 Collection & Compression ---
        
        # Collect H2 from all sources
        total_h2_production = (
            self.pem_system.h2_product_kg_h +
            self.soec_system.h2_product_kg_h +
            self.atr_system.h2_product_kg_h
        )
        
        # C-6: Compress to LP
        self.compressor_c6.transfer_mass_kg = total_h2_production * self.dt
        self.compressor_c6.step(t)
        
        # HX-11: Cool
        self.chiller_hx11.inlet_flow_kg_h = self.compressor_c6.actual_mass_transferred_kg / self.dt if self.dt > 0 else 0
        self.chiller_hx11.step(t)
        
        # LP Storage
        # Fill with cooled gas
        self.storage_lp.fill(self.compressor_c6.actual_mass_transferred_kg)
        self.storage_lp.step(t)
        
        # C-7: Compress to HP
        # Check if we need to transfer
        lp_mass = self.storage_lp.get_total_mass()
        if lp_mass > 50.0: # Threshold
            transfer_amount = min(lp_mass, 500.0 * self.dt) # Limit by compressor max
            discharged = self.storage_lp.discharge(transfer_amount)
            self.compressor_c7.transfer_mass_kg = discharged
        else:
            self.compressor_c7.transfer_mass_kg = 0.0
            
        self.compressor_c7.step(t)
        
        # HP Storage
        self.storage_hp.fill(self.compressor_c7.actual_mass_transferred_kg)
        
        # Discharge Demand
        if self.h2_demand_kg_h > 0:
            self.storage_hp.discharge(self.h2_demand_kg_h * self.dt)
            
        self.storage_hp.step(t)
        
    def get_state(self) -> Dict[str, Any]:
        return {
            **super().get_state(),
            'subsystems': {
                'water_system': self.water_system.get_state(),
                'pem_system': self.pem_system.get_state(),
                'soec_system': self.soec_system.get_state(),
                'atr_system': self.atr_system.get_state(),
                'co2_system': self.co2_system.get_state(),
                'compressor_c6': self.compressor_c6.get_state(),
                'chiller_hx11': self.chiller_hx11.get_state(),
                'storage_lp': self.storage_lp.get_state(),
                'compressor_c7': self.compressor_c7.get_state(),
                'storage_hp': self.storage_hp.get_state()
            },
            'total_h2_stored_kg': self.storage_hp.get_total_mass() + self.storage_lp.get_total_mass(),
            'production_rate_kg_h': (
                self.pem_system.h2_product_kg_h +
                self.soec_system.h2_product_kg_h +
                self.atr_system.h2_product_kg_h
            ),
            'co2_stored_kg': self.co2_system.storage_co2s.current_mass_kg
        }
