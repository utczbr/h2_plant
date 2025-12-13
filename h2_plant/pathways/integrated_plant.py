"""
Integrated Plant Coordinator.

This module orchestrates the entire detailed plant simulation, wiring
together all subsystems into a unified processing chain.

Subsystem Integration:
    1. **Water Treatment**: Ultrapure water supply to all production paths.
    2. **PEM Electrolysis**: Low-temperature water splitting.
    3. **SOEC Electrolysis**: High-temperature steam electrolysis.
    4. **ATR Production**: Autothermal reforming with CO₂ capture.
    5. **Carbon Capture**: Tail gas processing and CO₂ sequestration.
    6. **Compression**: LP → HP transfer chain.
    7. **Storage**: Staged LP (60 bar) and HP (350 bar) tank arrays.

Component Lifecycle:
    The IntegratedPlant implements the Component Lifecycle Contract (Layer 1):
    - `initialize()`: Propagates to all subsystems.
    - `step()`: Executes causal flow from water → production → storage.
    - `get_state()`: Aggregates all subsystem states.
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
    Coordinator for the full detailed hydrogen production plant.

    Manages subsystem interactions and material flows across the
    complete production chain, from water intake to H₂ delivery.

    Process Flow:
        ```
        Water → PEM/SOEC/ATR → C-6 → LP Storage → C-7 → HP Storage → Delivery
                     ↓
                CO₂ Capture → Sequestration
        ```

    Attributes:
        water_system (DetailedWaterTreatment): Ultrapure water supply.
        pem_system (DetailedPEMElectrolyzer): PEM electrolyzer stack.
        soec_system (DetailedSOECElectrolyzer): SOEC electrolyzer stack.
        atr_system (DetailedATRProduction): Autothermal reformer.
        co2_system (DetailedCO2Capture): Carbon capture unit.
        storage_lp (TankArray): Low-pressure buffer storage (60 bar).
        storage_hp (TankArray): High-pressure delivery storage (350 bar).

    Example:
        >>> plant = IntegratedPlant()
        >>> plant.initialize(dt=1/60, registry=registry)
        >>> plant.step(t=0.0)
        >>> state = plant.get_state()
    """

    def __init__(self):
        """Initialize the integrated plant with all subsystems."""
        super().__init__()

        # Water System
        self.water_system = DetailedWaterTreatment()

        # Production Systems
        self.pem_system = DetailedPEMElectrolyzer(max_power_kw=2500.0)
        self.soec_system = DetailedSOECElectrolyzer(max_power_kw=1000.0)
        self.atr_system = DetailedATRProduction(max_biogas_kg_h=500.0)

        # Carbon Capture System
        self.co2_system = DetailedCO2Capture(max_flow_kg_h=200.0)

        # Compression Chain
        # C-6: Production → LP (30 bar → 60 bar)
        self.compressor_c6 = FillingCompressor(
            max_flow_kg_h=500.0,
            inlet_pressure_bar=30.0,
            outlet_pressure_bar=60.0,
            num_stages=1
        )

        # HX-11: Inter-stage cooling
        self.chiller_hx11 = HeatExchanger('HX-11', 500.0)

        # LP Storage (60 bar buffer)
        self.storage_lp = TankArray(n_tanks=5, capacity_kg=200.0, pressure_bar=60.0)

        # C-7: LP → HP (60 bar → 350 bar)
        self.compressor_c7 = FillingCompressor(
            max_flow_kg_h=500.0,
            inlet_pressure_bar=60.0,
            outlet_pressure_bar=350.0,
            num_stages=3
        )

        # HP Storage (350 bar delivery)
        self.storage_hp = TankArray(n_tanks=10, capacity_kg=100.0, pressure_bar=350.0)

        # External inputs
        self.external_water_input_kg_h = 2000.0
        self.biogas_input_kg_h = 400.0
        self.electricity_grid_kw = 0.0

        # Demand
        self.h2_demand_kg_h = 0.0

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """
        Initialize all subsystems with timestep and registry.

        Fulfills the Component Lifecycle Contract by propagating
        initialization to all internal subsystems.

        Args:
            dt (float): Simulation timestep in hours.
            registry (ComponentRegistry): Component registry.
        """
        super().initialize(dt, registry)

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
        """
        Execute one simulation timestep.

        Fulfills the Component Lifecycle Contract by orchestrating
        subsystem execution in causal order: water → production →
        carbon capture → compression → storage.

        Args:
            t (float): Current simulation time in hours.
        """
        Component.step(self, t)

        # --- 1. Water Supply ---
        self.water_system.external_water_input_kg_h = self.external_water_input_kg_h
        self.water_system.demand_pem_kg_h = 1000.0
        self.water_system.demand_soec_kg_h = 500.0
        self.water_system.demand_atr_kg_h = 500.0

        self.water_system.step(t)

        # --- 2. Production ---

        # PEM electrolyzer
        self.pem_system.feedwater_inlet.current_flow_kg_h = self.water_system.tank_wt.outlet_flow_kg_h * 0.5
        self.pem_system.step(t)

        # SOEC electrolyzer
        self.soec_system.water_input_kg_h = self.water_system.tank_wt.outlet_flow_kg_h * 0.25
        self.soec_system.step(t)

        # ATR with recycled gas integration
        recycled_gas = self.co2_system.recycled_gas_kg_h
        self.atr_system.biogas_input_kg_h = self.biogas_input_kg_h + recycled_gas
        self.atr_system.water_input_kg_h = self.water_system.tank_wt.outlet_flow_kg_h * 0.25
        self.atr_system.step(t)

        # --- 3. Carbon Capture ---
        self.co2_system.tail_gas_input_kg_h = self.atr_system.tail_gas_kg_h
        self.co2_system.step(t)

        # --- 4. H₂ Collection & Compression ---

        # Aggregate production from all sources
        total_h2_production = (
            self.pem_system.h2_product_kg_h +
            self.soec_system.h2_product_kg_h +
            self.atr_system.h2_product_kg_h
        )

        # C-6: Compress to LP pressure
        self.compressor_c6.transfer_mass_kg = total_h2_production * self.dt
        self.compressor_c6.step(t)

        # HX-11: Inter-stage cooling
        self.chiller_hx11.inlet_flow_kg_h = self.compressor_c6.actual_mass_transferred_kg / self.dt if self.dt > 0 else 0
        self.chiller_hx11.step(t)

        # Fill LP storage with cooled gas
        self.storage_lp.fill(self.compressor_c6.actual_mass_transferred_kg)
        self.storage_lp.step(t)

        # C-7: LP → HP transfer (threshold-based)
        lp_mass = self.storage_lp.get_total_mass()
        if lp_mass > 50.0:
            transfer_amount = min(lp_mass, 500.0 * self.dt)
            discharged = self.storage_lp.discharge(transfer_amount)
            self.compressor_c7.transfer_mass_kg = discharged
        else:
            self.compressor_c7.transfer_mass_kg = 0.0

        self.compressor_c7.step(t)

        # Fill HP storage
        self.storage_hp.fill(self.compressor_c7.actual_mass_transferred_kg)

        # Fulfill delivery demand
        if self.h2_demand_kg_h > 0:
            self.storage_hp.discharge(self.h2_demand_kg_h * self.dt)

        self.storage_hp.step(t)

    def get_state(self) -> Dict[str, Any]:
        """
        Retrieve aggregated state from all subsystems.

        Fulfills the Component Lifecycle Contract by providing
        complete plant state including all subsystem states.

        Returns:
            Dict[str, Any]: Complete plant state including:
                - subsystems: Individual subsystem states.
                - total_h2_stored_kg: Combined LP + HP storage.
                - production_rate_kg_h: Total H₂ production rate.
                - co2_stored_kg: Captured CO₂ inventory.
        """
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
