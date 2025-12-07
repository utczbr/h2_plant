
import pytest
from h2_plant.components.balance_of_plant.pump import Pump
from h2_plant.core.stream import Stream
from h2_plant.core.component_registry import ComponentRegistry

try:
    import CoolProp.CoolProp as CP
    COOLPROP_OK = True
except:
    COOLPROP_OK = False

def test_pump_push_thermodynamics():
    # Setup: Pump from 1 bar to 50 bar
    pump = Pump(target_pressure_bar=50.0, eta_is=0.80, eta_m=0.95)
    registry = ComponentRegistry()
    pump.initialize(dt=1.0, registry=registry)
    
    # Input: Water at 25C, 1 bar, 3600 kg/h (1 kg/s)
    s_in = Stream(mass_flow_kg_h=3600.0, temperature_k=298.15, pressure_pa=1e5, composition={'H2O': 1.0}, phase='liquid')
    
    # Check Push
    accepted = pump.receive_input("inlet", s_in)
    assert accepted == 3600.0
    
    # Execute
    pump.step(0.0)
    
    # Validation
    s_out = pump.get_output("outlet")
    assert s_out is not None
    assert s_out.pressure_pa == 50e5
    assert s_out.mass_flow_kg_h == 3600.0
    
    # Power Check
    # dP = 49 bar = 49e5 Pa. 
    # V_dot = 1 kg/s / 1000 kg/m3 = 0.001 m3/s
    # Ideal Work roughly V dP = 0.001 * 4900000 = 4900 W = 4.9 kW
    # Real Power = 4.9 / (0.8 * 0.95) ~ 6.45 kW
    # Let's give a reasonable range
    assert 6.0 < pump.power_kw < 7.0
    
    if COOLPROP_OK:
        # Thermodynamic heating check
        # Real work is dissipative, so T should rise slightly.
        # W_loss = W_real * (1 - eta) ... roughly
        # For water at 49 bar compression, expect < 1C rise but definitely a rise.
        # Actually isentropic compression of liquid water rises T slightly too.
        assert pump.outlet_temp_c > 25.0
        assert pump.outlet_temp_c < 26.0 # Small rise for water

def test_pump_zero_flow():
    pump = Pump(target_pressure_bar=10.0)
    registry = ComponentRegistry()
    pump.initialize(dt=1.0, registry=registry)
    
    pump.step(0.0)
    
    assert pump.power_kw == 0.0
    assert pump.get_output("outlet") is None
