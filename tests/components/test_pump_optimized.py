
import pytest
from unittest.mock import MagicMock, patch
from h2_plant.components.balance_of_plant.pump import Pump
from h2_plant.components.water.water_pump import WaterPumpThermodynamic
from h2_plant.core.stream import Stream
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.optimization.lut_manager import LUTManager

def test_pump_optimization_cascade():
    """Verify Pump (Push) uses LUTManager -> CoolPropLUT cascade."""
    registry = ComponentRegistry()
    
    # Mock LUT Manager
    lut_manager = MagicMock(spec=LUTManager)
    # Mock return values for H/S lookup
    # Water at 300K, 1bar: H ~ 112 kJ/kg, S ~ 0.39 kJ/kgK
    lut_manager.lookup.side_effect = lambda f, p, P, T: 112000.0 if p == 'H' else 390.0
    
    registry.register("lut_manager", lut_manager)
    
    pump = Pump(target_pressure_bar=5.0)
    pump.initialize(1.0, registry)
    
    s_in = Stream(mass_flow_kg_h=3600.0, temperature_k=300.0, pressure_pa=1e5, composition={'H2O': 1.0})
    pump.receive_input("inlet", s_in)
    
    # Mock CoolPropLUT to check usage
    with patch('h2_plant.components.balance_of_plant.pump.CoolPropLUT') as MockLUT:
        pump.step(0.0)
        
        # 1. Verify LUT Manager called for Inlet H/S
        # Should be called twice (H and S)
        assert lut_manager.lookup.call_count >= 2
        
        # 2. Verify CoolPropLUT called for Outlet T (Inverse)
        MockLUT.PropsSI.assert_called()
        
def test_water_pump_reverse_optimization():
    """Verify WaterPumpThermodynamic uses logic in Reverse mode."""
    registry = ComponentRegistry()
    lut_manager = MagicMock(spec=LUTManager)
    lut_manager.lookup.return_value = 112000.0 # Just a float
    registry.register("lut_manager", lut_manager)
    
    pump = WaterPumpThermodynamic(pump_id="rp", target_pressure_pa=1e5)
    pump.initialize(1.0, registry)
    
    # Outlet Stream (P=5bar, T=300K)
    s_out = Stream(mass_flow_kg_h=3600.0, temperature_k=300.0, pressure_pa=5e5, composition={'H2O': 1.0})
    pump.receive_input("water_out_reverse", s_out, "water")
    
    pump.step(0.0)
    
    # Should use LUT/CoolPropLUT in reverse calc
    assert lut_manager.lookup.called
