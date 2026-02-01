"""
Unit tests for CompressorStorage component.

Tests component lifecycle, configuration, and basic functionality.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock

from h2_plant.components.compression.compressor_storage import CompressorStorage
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.component_ids import ComponentID
from h2_plant.core.enums import CompressorMode


class TestCompressorStorageInitialization:
    """Test component initialization."""
    
    def test_initialization_basic(self):
        """Test basic initialization."""
        compressor = CompressorStorage(
            max_flow_kg_h=100.0,
            inlet_pressure_bar=40.0,
            outlet_pressure_bar=140.0
        )
        
        assert compressor.max_flow_kg_h == 100.0
        assert compressor.inlet_pressure_bar == 40.0
        assert compressor.outlet_pressure_bar == 140.0
        assert compressor.inlet_temperature_c == 10.0  # Default
        assert compressor.max_temperature_c == 85.0    # Default
        assert compressor.isentropic_efficiency == 0.65 # Default
        assert compressor.chiller_cop == 3.0           # Default
    
    def test_initialization_custom_values(self):
        """Test initialization with custom parameters."""
        compressor = CompressorStorage(
            max_flow_kg_h=200.0,
            inlet_pressure_bar=50.0,
            outlet_pressure_bar=500.0,
            inlet_temperature_c=15.0,
            max_temperature_c=90.0,
            isentropic_efficiency=0.70,
            chiller_cop=3.5
        )
        
        assert compressor.inlet_temperature_c == 15.0
        assert compressor.max_temperature_c == 90.0
        assert compressor.isentropic_efficiency == 0.70
        assert compressor.chiller_cop == 3.5
    
    def test_stage_calculation_requires_registry(self):
        """Test that stage calculation is done during initialize()."""
        compressor = CompressorStorage(
            max_flow_kg_h=100.0,
            inlet_pressure_bar=40.0,
            outlet_pressure_bar=140.0
        )
        
        # Before initialization, stages should be 0
        assert compressor.num_stages == 0
        
        # After initialization (with mock registry), stages calculated
        # This is tested more thoroughly in integration tests


class TestCompressorStorageStateMethods:
    """Test state management methods."""
    
    def test_get_state_structure(self):
        """Test get_state returns proper structure."""
        compressor = CompressorStorage(
            max_flow_kg_h=100.0,
            inlet_pressure_bar=40.0,
            outlet_pressure_bar=140.0
        )
        
        state = compressor.get_state()
        
        # Check required fields
        required_fields = [
            'mode', 'num_stages', 'stage_pressure_ratio',
            'actual_mass_transferred_kg', 'compression_work_kwh',
            'chilling_work_kwh', 'energy_consumed_kwh',
            'specific_energy_kwh_kg', 'cumulative_energy_kwh',
            'cumulative_mass_kg', 'inlet_pressure_bar',
            'outlet_pressure_bar'
        ]
        
        for field in required_fields:
            assert field in state, f"Missing field: {field}"
    
    def test_get_state_json_serializable(self):
        """Test all state values are JSON-serializable."""
        import json
        
        compressor = CompressorStorage(
            max_flow_kg_h=100.0,
            inlet_pressure_bar=40.0,
            outlet_pressure_bar=140.0
        )
        
        state = compressor.get_state()
        
        # Should not raise exception
        json_str = json.dumps(state)
        assert isinstance(json_str, str)


class TestCompressorStorageOperation:
    """Test operational behavior."""
    
    def test_idle_mode_when_no_mass(self):
        """Test compressor stays idle when no mass to transfer."""
        compressor = CompressorStorage(
            max_flow_kg_h=100.0,
            inlet_pressure_bar=40.0,
            outlet_pressure_bar=140.0
        )
        
        # Mock step without initialization for this simple test
        compressor._initialized = True
        compressor.transfer_mass_kg = 0.0
        
        compressor.step(t=0.0)
        
        assert compressor.mode == CompressorMode.IDLE
        assert compressor.energy_consumed_kwh == 0.0
        assert compressor.actual_mass_transferred_kg == 0.0
    
    def test_flow_rate_limiting(self):
        """Test that mass transfer is limited by max flow rate."""
        compressor = CompressorStorage(
            max_flow_kg_h=100.0,
            inlet_pressure_bar=40.0,
            outlet_pressure_bar=140.0
        )
        
        compressor._initialized = True
        compressor.dt = 1.0  # 1 hour timestep
        compressor.num_stages = 2  # Mock value
        compressor.specific_energy_kwh_kg = 1.0  # Mock value
        
        # Request more than max flow
        compressor.transfer_mass_kg = 150.0  # More than 100 kg/h
        
        # Mock the physics calculation to avoid registry requirement
        compressor._calculate_compression_physics = MagicMock()
        
        compressor.step(t=0.0)
        
        # Should be limited to max_flow * dt = 100 kg
        assert compressor.actual_mass_transferred_kg == 100.0
        assert compressor.mode == CompressorMode.LP_TO_HP
    
    def test_cumulative_statistics(self):
        """Test cumulative energy and mass tracking."""
        compressor = CompressorStorage(
            max_flow_kg_h=100.0,
            inlet_pressure_bar=40.0,
            outlet_pressure_bar=140.0
        )
        
        compressor._initialized = True
        compressor.dt = 1.0
        compressor.num_stages = 2
        
        # Mock physics calculation
        def mock_physics():
            compressor.specific_energy_kwh_kg = 1.5
            compressor.compression_work_kwh = 1.2 * compressor.actual_mass_transferred_kg
            compressor.chilling_work_kwh = 0.3 * compressor.actual_mass_transferred_kg
        
        compressor._calculate_compression_physics = mock_physics
        
        # First step: 50 kg
        compressor.transfer_mass_kg = 50.0
        compressor.step(t=0.0)
        
        assert compressor.cumulative_mass_kg == 50.0
        assert compressor.cumulative_energy_kwh == 75.0  # 50 kg * 1.5 kWh/kg
        
        # Second step: 30 kg
        compressor.transfer_mass_kg = 30.0
        compressor.step(t=1.0)
        
        assert compressor.cumulative_mass_kg == 80.0
        assert compressor.cumulative_energy_kwh == 120.0  # 80 kg * 1.5 kWh/kg


class TestCompressorStoragePortInterface:
    """Test port interface methods."""
    
    def test_get_ports(self):
        """Test port metadata."""
        compressor = CompressorStorage(
            max_flow_kg_h=100.0,
            inlet_pressure_bar=40.0,
            outlet_pressure_bar=140.0
        )
        
        ports = compressor.get_ports()
        
        assert 'h2_in' in ports
        assert 'electricity_in' in ports
        assert 'h2_out' in ports
        
        assert ports['h2_in']['type'] == 'input'
        assert ports['h2_out']['type'] == 'output'
        assert ports['h2_in']['resource_type'] == 'hydrogen'


class TestCompressorStorageConstants:
    """Test that legacy constants are preserved."""
    
    def test_legacy_constants(self):
        """Test that hardcoded constants match legacy values."""
        compressor = CompressorStorage(
            max_flow_kg_h=100.0,
            inlet_pressure_bar=40.0,
            outlet_pressure_bar=140.0
        )
        
        # These exact values from legacy code
        assert compressor.BAR_TO_PA == 1e5
        assert compressor.J_TO_KWH == 2.7778e-7
        
        # Default physics values from legacy
        assert compressor.inlet_temperature_c == 10.0
        assert compressor.max_temperature_c == 85.0
        assert compressor.isentropic_efficiency == 0.65
        assert compressor.chiller_cop == 3.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
