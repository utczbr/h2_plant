import pytest
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.components.water.quality_test import WaterQualityTestBlock
from h2_plant.components.water.treatment import WaterTreatmentBlock
from h2_plant.components.water.storage import UltrapureWaterStorageTank
from h2_plant.components.water.pump import WaterPump

def test_water_quality_test_block():
    comp = WaterQualityTestBlock()
    comp.initialize(1.0, ComponentRegistry())
    comp.inlet_flow_m3h = 10.0
    comp.step(0)
    state = comp.get_state()
    assert state["inlet_flow_m3h"] == 10.0
    assert state["flows"]["outputs"]["tested_water"]["value"] == 10.0

def test_water_treatment_block():
    comp = WaterTreatmentBlock(max_flow_m3h=10.0, power_consumption_kw=20.0)
    comp.initialize(1.0, ComponentRegistry())
    comp.step(0)
    state = comp.get_state()
    assert state["output_flow_kgh"] == 10000.0
    assert state["test_flow_kgh"] == 100.0
    assert state["flows"]["outputs"]["ultrapure_water"]["value"] == 9900.0

def test_ultrapure_water_storage_tank():
    comp = UltrapureWaterStorageTank(capacity_l=5000, initial_fill_ratio=0.5)
    comp.initialize(1.0, ComponentRegistry())
    assert comp.current_mass_kg == 2500.0
    
    comp.fill(1000)
    assert comp.current_mass_kg == 3500.0
    
    comp.withdraw(500)
    assert comp.current_mass_kg == 3000.0

    # Test filling to capacity
    overflow = comp.fill(3000)
    assert comp.current_mass_kg == 5000.0
    assert overflow == 2000.0
    
    # Test withdrawing more than available
    withdrawn = comp.withdraw(6000)
    assert comp.current_mass_kg == 0.0
    assert withdrawn == 5000.0


def test_water_pump():
    comp = WaterPump(pump_id="A", power_kw=0.75, power_source="grid")
    comp.initialize(1.0, ComponentRegistry())
    comp.flow_kgh = 1000.0
    comp.step(0)
    state = comp.get_state()
    assert state["flow_kgh"] == 1000.0
    assert state["flows"]["outputs"]["pressurized_water"]["value"] == 1000.0
