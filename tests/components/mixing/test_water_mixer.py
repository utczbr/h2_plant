"""
Unit tests for WaterMixer component.

Tests thermodynamic mixing calculations and validates against
the legacy Mixer.py implementation.
"""

import pytest
import logging
from typing import Dict, Any

from h2_plant.components.mixing.water_mixer import WaterMixer, COOLPROP_AVAILABLE
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.stream import Stream

# Skip all tests if CoolProp not available
pytestmark = pytest.mark.skipif(
    not COOLPROP_AVAILABLE,
    reason="CoolProp not available - required for WaterMixer"
)

logger = logging.getLogger(__name__)


class TestWaterMixerBasic:
    """Basic functionality tests."""
    
    def test_initialization(self):
        """Test mixer initializes with correct default state."""
        mixer = WaterMixer(outlet_pressure_kpa=200.0)
        
        assert mixer.outlet_pressure_kpa == 200.0
        assert mixer.outlet_pressure_pa == 200000.0
        assert mixer.fluid_type == 'Water'
        assert mixer.max_inlet_streams == 10
        assert len(mixer.inlet_streams) == 0
        assert mixer.outlet_stream is None
    
    def test_custom_parameters(self):
        """Test mixer with custom parameters."""
        mixer = WaterMixer(
            outlet_pressure_kpa=150.0,
            fluid_type='Water',
            max_inlet_streams=5
        )
        
        assert mixer.outlet_pressure_kpa == 150.0
        assert mixer.max_inlet_streams == 5
    
    def test_component_lifecycle(self):
        """Test component initialization lifecycle."""
        mixer = WaterMixer(outlet_pressure_kpa=200.0)
        registry = ComponentRegistry()
        
        mixer.set_component_id('test_mixer')
        mixer.initialize(dt=1.0, registry=registry)
        
        assert mixer.component_id == 'test_mixer'
        assert mixer.dt == 1.0
        assert mixer._initialized


class TestWaterMixerOperations:
    """Test mixing operations."""
    
    def test_single_stream_passthrough(self):
        """Test single stream passes through unchanged (except pressure)."""
        mixer = WaterMixer(outlet_pressure_kpa=200.0)
        registry = ComponentRegistry()
        mixer.set_component_id('mixer')
        mixer.initialize(dt=1.0, registry=registry)
        
        # Single input stream
        stream = Stream(
            mass_flow_kg_h=1800.0,
            temperature_k=288.15,  # 15°C
            pressure_pa=200000.0,
            composition={'H2O': 1.0},
            phase='liquid'
        )
        
        mixer.receive_input('inlet_0', stream, 'water')
        mixer.step(t=0.0)
        
        output = mixer.get_output('outlet')
        assert output is not None
        assert output.mass_flow_kg_h == pytest.approx(1800.0, rel=1e-3)
        # Temperature should be approximately the same
        assert output.temperature_k == pytest.approx(288.15, rel=1e-2)
        assert output.pressure_pa == 200000.0
    
    def test_two_stream_mixing(self):
        """Test mixing two streams."""
        mixer = WaterMixer(outlet_pressure_kpa=200.0)
        registry = ComponentRegistry()
        mixer.set_component_id('mixer')
        mixer.initialize(dt=1.0, registry=registry)
        
        # Cold stream
        stream1 = Stream(
            mass_flow_kg_h=1800.0,  # 0.5 kg/s
            temperature_k=288.15,   # 15°C
            pressure_pa=200000.0,
            composition={'H2O': 1.0},
            phase='liquid'
        )
        
        # Hot stream
        stream2 = Stream(
            mass_flow_kg_h=1080.0,  # 0.3 kg/s
            temperature_k=353.15,   # 80°C
            pressure_pa=220000.0,
            composition={'H2O': 1.0},
            phase='liquid'
        )
        
        mixer.receive_input('inlet_0', stream1, 'water')
        mixer.receive_input('inlet_1', stream2, 'water')
        mixer.step(t=0.0)
        
        output = mixer.get_output('outlet')
        assert output is not None
        
        # Mass balance check
        total_mass_in = 1800.0 + 1080.0
        assert output.mass_flow_kg_h == pytest.approx(total_mass_in, rel=1e-3)
        
        # Temperature should be between input temperatures
        assert 288.15 < output.temperature_k < 353.15
        
        # Output pressure
        assert output.pressure_pa == 200000.0
    
    def test_matches_legacy_validation(self):
        """
        CRITICAL TEST: Verify exact match with validated Mixer.py results.
        
        This test uses the exact same inputs from Comparison_script.py and
        validates that the output matches the legacy implementation.
        """
        mixer = WaterMixer(outlet_pressure_kpa=200.0)
        registry = ComponentRegistry()
        mixer.set_component_id('validation_mixer')
        mixer.initialize(dt=1.0, registry=registry)
        
        # Exact inputs from validation case
        # Stream 1: 0.5 kg/s @ 15°C, 200 kPa
        stream1 = Stream(
            mass_flow_kg_h=1800.0,
            temperature_k=288.15,
            pressure_pa=200000.0,
            composition={'H2O': 1.0},
            phase='liquid'
        )
        
        # Stream 2: 0.3 kg/s @ 80°C, 220 kPa
        stream2 = Stream(
            mass_flow_kg_h=1080.0,
            temperature_k=353.15,
            pressure_pa=220000.0,
            composition={'H2O': 1.0},
            phase='liquid'
        )
        
        # Stream 3: 0.2 kg/s @ 50°C, 210 kPa
        stream3 = Stream(
            mass_flow_kg_h=720.0,
            temperature_k=323.15,
            pressure_pa=210000.0,
            composition={'H2O': 1.0},
            phase='liquid'
        )
        
        mixer.receive_input('inlet_0', stream1, 'water')
        mixer.receive_input('inlet_1', stream2, 'water')
        mixer.receive_input('inlet_2', stream3, 'water')
        mixer.step(t=0.0)
        
        output = mixer.get_output('outlet')
        assert output is not None
        
        # Expected results from validation:
        # Mass Flow: 1.00000 kg/s = 3600 kg/h
        # Temperature: 41.51445 °C = 314.66445 K
        # Enthalpy: 174.03301 kJ/kg
        
        expected_mass_kg_h = 3600.0
        expected_temp_c = 41.51445
        expected_temp_k = expected_temp_c + 273.15
        
        # Mass balance (should be exact)
        assert output.mass_flow_kg_h == pytest.approx(expected_mass_kg_h, rel=1e-6)
        
        # Temperature (within 0.01°C tolerance)
        output_temp_c = output.temperature_k - 273.15
        assert output_temp_c == pytest.approx(expected_temp_c, abs=0.01)
        
        # Also check in Kelvin
        assert output.temperature_k == pytest.approx(expected_temp_k, abs=0.01)
        
        logger.info(f"Validation test passed: T_out = {output_temp_c:.5f}°C")


class TestWaterMixerEdgeCases:
    """Test edge cases and error handling."""
    
    def test_no_input_streams(self):
        """Test mixer with no input streams."""
        mixer = WaterMixer(outlet_pressure_kpa=200.0)
        registry = ComponentRegistry()
        mixer.set_component_id('mixer')
        mixer.initialize(dt=1.0, registry=registry)
        
        mixer.step(t=0.0)
        
        output = mixer.get_output('outlet')
        assert output is None
        assert mixer.last_mass_flow_kg_h == 0.0
    
    def test_max_inlets_limit(self):
        """Test that mixer respects max inlet limit."""
        mixer = WaterMixer(outlet_pressure_kpa=200.0, max_inlet_streams=2)
        registry = ComponentRegistry()
        mixer.set_component_id('mixer')
        mixer.initialize(dt=1.0, registry=registry)
        
        stream = Stream(
            mass_flow_kg_h=1000.0,
            temperature_k=300.0,
            pressure_pa=200000.0,
            composition={'H2O': 1.0},
            phase='liquid'
        )
        
        # Accept first two
        result1 = mixer.receive_input('inlet_0', stream, 'water')
        result2 = mixer.receive_input('inlet_1', stream, 'water')
        
        # Reject third
        result3 = mixer.receive_input('inlet_2', stream, 'water')
        
        assert result1 == 1000.0
        assert result2 == 1000.0
        assert result3 == 0.0  # Rejected
        assert len(mixer.inlet_streams) == 2
    
    def test_invalid_input_type(self):
        """Test handling of invalid input type."""
        mixer = WaterMixer(outlet_pressure_kpa=200.0)
        registry = ComponentRegistry()
        mixer.set_component_id('mixer')
        mixer.initialize(dt=1.0, registry=registry)
        
        # Pass non-Stream object
        result = mixer.receive_input('inlet_0', "not a stream", 'water')
        assert result == 0.0


class TestWaterMixerState:
    """Test state reporting and monitoring."""
    
    def test_get_state(self):
        """Test state dictionary structure."""
        mixer = WaterMixer(outlet_pressure_kpa=200.0)
        registry = ComponentRegistry()
        mixer.set_component_id('mixer')
        mixer.initialize(dt=1.0, registry=registry)
        
        state = mixer.get_state()
        
        assert 'component_id' in state
        assert 'outlet_pressure_kpa' in state
        assert 'num_active_inlets' in state
        assert 'outlet_temperature_k' in state
        assert 'outlet_enthalpy_kj_kg' in state
        
        assert state['outlet_pressure_kpa'] == 200.0
        assert state['num_active_inlets'] == 0
    
    def test_state_after_mixing(self):
        """Test state reflects mixing results."""
        mixer = WaterMixer(outlet_pressure_kpa=200.0)
        registry = ComponentRegistry()
        mixer.set_component_id('mixer')
        mixer.initialize(dt=1.0, registry=registry)
        
        stream = Stream(
            mass_flow_kg_h=1800.0,
            temperature_k=300.0,
            pressure_pa=200000.0,
            composition={'H2O': 1.0},
            phase='liquid'
        )
        
        mixer.receive_input('inlet_0', stream, 'water')
        mixer.step(t=0.0)
        
        state = mixer.get_state()
        assert state['num_active_inlets'] == 1
        assert state['outlet_mass_flow_kg_h'] == pytest.approx(1800.0, rel=1e-3)
        assert state['outlet_temperature_k'] > 0
    
    def test_get_ports(self):
        """Test port information."""
        mixer = WaterMixer(outlet_pressure_kpa=200.0, max_inlet_streams=3)
        
        ports = mixer.get_ports()
        
        # Should have outlet port
        assert 'outlet' in ports
        assert ports['outlet']['type'] == 'output'
        assert ports['outlet']['resource_type'] == 'water'
        
        # Should have inlet ports
        assert 'inlet_0' in ports
        assert 'inlet_1' in ports
        assert 'inlet_2' in ports
        assert ports['inlet_0']['type'] == 'input'


class TestWaterMixerIntegration:
    """Integration tests with component registry."""
    
    def test_registry_integration(self):
        """Test mixer works properly with component registry."""
        registry = ComponentRegistry()
        
        mixer = WaterMixer(outlet_pressure_kpa=200.0)
        registry.register('water_mixer_1', mixer)
        
        registry.initialize_all(dt=1.0)
        
        # Verify registered and initialized
        retrieved = registry.get('water_mixer_1')
        assert retrieved is mixer
        assert retrieved._initialized
    
    def test_multiple_timesteps(self):
        """Test mixer operates correctly over multiple timesteps."""
        mixer = WaterMixer(outlet_pressure_kpa=200.0)
        registry = ComponentRegistry()
        mixer.set_component_id('mixer')
        mixer.initialize(dt=1.0, registry=registry)
        
        stream = Stream(
            mass_flow_kg_h=1800.0,
            temperature_k=300.0,
            pressure_pa=200000.0,
            composition={'H2O': 1.0},
            phase='liquid'
        )
        
        # Run multiple timesteps
        for t in range(10):
            mixer.receive_input('inlet_0', stream, 'water')
            mixer.step(t=float(t))
            
            output = mixer.get_output('outlet')
            assert output is not None
            assert output.mass_flow_kg_h == pytest.approx(1800.0, rel=1e-3)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
