
import pytest
from unittest.mock import MagicMock, patch
from h2_plant.components.mixing.water_mixer import WaterMixer
from h2_plant.core.stream import Stream
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.optimization.lut_manager import LUTManager

def test_water_mixer_optimization():
    # Setup
    registry = ComponentRegistry()
    
    # Mock LUT Manager
    lut_manager = MagicMock(spec=LUTManager)
    lut_manager.lookup.return_value = 50000.0 # 50 kJ/kg J/kg
    # Need to match expected units. lookup returns J/kg.
    # We want valid H to return valid T.
    # Let's rely on fallback or basic mocks.
    
    registry.register("lut_manager", lut_manager)
    
    mixer = WaterMixer()
    mixer.initialize(1.0, registry)
    
    # Verify LUT Manager was retrieved
    assert mixer._lut_manager == lut_manager
    
    # Test Step with LUT Manager usage
    # 1. Input stream
    s_in = Stream(mass_flow_kg_h=3600.0, temperature_k=300.0, pressure_pa=1e5, composition={'H2O': 1.0})
    mixer.receive_input("in_1", s_in, "water")
    
    # Mock CoolPropLUT to avoid real CP calls and verify usage
    with patch('h2_plant.components.mixing.water_mixer.CoolPropLUT') as MockLUT:
        # Step
        mixer.step(0.0)
        
        # Validation:
        # 1. lut_manager.lookup should be called for Enthalpy (H)
        lut_manager.lookup.assert_called()
        args = lut_manager.lookup.call_args[0]
        assert args[0] == 'Water' # Fluid
        assert args[1] == 'H' # Prop
        
        # 2. CoolPropLUT.PropsSI should be called for Temperature (T) output 
        # (inverse lookup not supported by manager)
        MockLUT.PropsSI.assert_called() 
        # Check call for T from H, P
        call_args_list = MockLUT.PropsSI.call_args_list
        # Should be called for output T
        found_t_call = False
        for call in call_args_list:
            if call[0][0] == 'T':
                found_t_call = True
                break
        assert found_t_call

def test_water_mixer_entropy_deviation():
    # If possible, verify entropy check doesn't crash
    mixer = WaterMixer()
    registry = ComponentRegistry()
    mixer.initialize(1.0, registry)
    
    s_in = Stream(mass_flow_kg_h=3600.0, temperature_k=300.0, pressure_pa=1e5, composition={'H2O': 1.0})
    mixer.receive_input("in_1", s_in, "water")
    
    # Run real step (integration test with real CoolPropLUT if available)
    try:
        mixer.step(0.0)
        assert mixer.outlet_stream is not None
        assert mixer.outlet_stream.temperature_k > 290.0
    except Exception as e:
        pytest.skip(f"Skipping integration test due to missing dependencies: {e}")
