
import pytest
from unittest.mock import MagicMock
from h2_plant.components.mixing.water_mixer import WaterMixer
from h2_plant.components.balance_of_plant.pump import Pump
from h2_plant.components.water.water_pump import WaterPumpThermodynamic
from h2_plant.core.stream import Stream
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.optimization.lut_manager import LUTManager

class MockOrchestrator:
    """Simple mock to facilitate push transfers between components."""
    def push_stream(self, source, output_port, target, input_port, res_type):
        s = source.get_output(output_port)
        if s:
            target.receive_input(input_port, s, res_type)

def test_water_mixer_to_pump_integration():
    """
    Test flow: Source Streams -> WaterMixer -> Pump -> Output
    Verifies:
    1. Mixer aggregates inputs
    2. Mixer pushes (via Orchestrator mock) to Pump
    3. Pump elevates pressure and adds heat
    4. End-to-end mass conservation
    """
    registry = ComponentRegistry()
    
    # 1. Setup Components
    mixer = WaterMixer(outlet_pressure_kpa=100.0, max_inlet_streams=2) # 1 bar out
    pump = Pump(target_pressure_bar=50.0, eta_is=0.8, eta_m=0.95) # 50 bar out
    
    mixer.initialize(1.0, registry)
    pump.initialize(1.0, registry)
    
    # 2. Input Streams to Mixer (Total 1000 kg/h)
    s1 = Stream(mass_flow_kg_h=600.0, temperature_k=300.0, pressure_pa=2e5, composition={'H2O': 1.0})
    s2 = Stream(mass_flow_kg_h=400.0, temperature_k=310.0, pressure_pa=2e5, composition={'H2O': 1.0})
    
    mixer.receive_input("in_1", s1, "water")
    mixer.receive_input("in_2", s2, "water")
    
    # 3. Step Mixer
    mixer.step(0.0)
    
    # Verify Mixer Output
    mixed = mixer.get_output("outlet")
    assert mixed is not None
    assert mixed.mass_flow_kg_h == 1000.0
    # Temp roughly weighted average: 0.6*300 + 0.4*310 = 180 + 124 = 304K
    assert 303.0 < mixed.temperature_k < 305.0
    
    # 4. Push Mixed Stream to Pump
    # Simulate Orchestrator pushing data
    pump.receive_input("inlet", mixed, "water")
    
    # 5. Step Pump
    pump.step(0.0)
    
    # Verify Pump Output
    pumped = pump.get_output("outlet")
    assert pumped is not None
    
    # Mass conservation
    assert pumped.mass_flow_kg_h == 1000.0
    # Pressure target
    assert pumped.pressure_pa == 50.0 * 1e5
    # Temperature rise (Pump physics)
    # 1 bar to 50 bar is significant work. Water heats up slightly.
    assert pumped.temperature_k > mixed.temperature_k
    assert pumped.temperature_k < mixed.temperature_k + 5.0 # Shouldn't boil
    
    # Power check
    assert pump.power_kw > 0.0

def test_water_pump_detailed_integration_with_lut():
    """
    Test WaterPumpThermodynamic with LUT integration.
    """
    registry = ComponentRegistry()
    
    # Real LUT Manager (if available) or Mock
    # We'll use a Mock for stability in this test, but structure covers real integration path
    lut_manager = MagicMock(spec=LUTManager)
    # H(300K, 1bar) ~ 112 kJ/kg = 112000 J/kg
    lut_manager.lookup.side_effect = lambda f, p, P, T: 112000.0 if p == 'H' else 1000.0 # D=1000
    registry.register("lut_manager", lut_manager)
    
    pump = WaterPumpThermodynamic(pump_id="feed_pump", target_pressure_pa=20e5) # 20 bar
    pump.initialize(1.0, registry)
    
    s_in = Stream(mass_flow_kg_h=100.0, temperature_k=298.15, pressure_pa=1e5, composition={'H2O': 1.0})
    
    pump.receive_input("water_in", s_in, "water")
    pump.step(0.0)
    
    assert pump.outlet_stream is not None
    assert pump.outlet_stream.pressure_pa == 20e5
    assert pump.power_shaft_kw > 0
    
    # Verify LUT was accessed (Optimization Check)
    # note: step calls _calculate_forward which tries LUT
    assert lut_manager.lookup.call_count > 0
