"""
Unit tests for WaterPumpThermodynamic component.

Tests cover:
- Initialization and validation
- Forward calculation mode
- Reverse calculation mode  
- CoolProp vs simplified model
- State management
- Port interface
- Cumulative statistics
"""

import pytest
import numpy as np
from h2_plant.components.water.water_pump_thermodynamic import WaterPumpThermodynamic
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.stream import Stream


class TestWaterPumpThermodynamicInitialization:
    """Test pump initialization and configuration."""
    
    def test_basic_initialization(self):
        """Test basic pump creation."""
        pump = WaterPumpThermodynamic(
            pump_id='test_pump',
            eta_is=0.82,
            eta_m=0.96,
            target_pressure_pa=500000.0
        )
        
        assert pump.component_id == 'test_pump'
        assert pump.eta_is == 0.82
        assert pump.eta_m == 0.96
        assert pump.target_pressure_pa == 500000.0
    
    def test_default_efficiencies(self):
        """Test default efficiency values."""
        pump = WaterPumpThermodynamic(
            pump_id='default_pump',
            target_pressure_pa=300000.0
        )
        
        assert pump.eta_is == 0.82  # Default isentropic
        assert pump.eta_m == 0.96   # Default mechanical
    
    def test_initialization_with_registry(self):
        """Test initialization with registry."""
        pump = WaterPumpThermodynamic(
            pump_id='registry_pump',
            target_pressure_pa=500000.0
        )
        
        registry = ComponentRegistry()
        pump.initialize(dt=1.0, registry=registry)
        
        assert pump.dt == 1.0


class TestWaterPumpForwardCalculation:
    """Test forward calculation (inlet known → outlet calculated)."""
    
    def test_forward_calculation_basic(self):
        """Test basic forward calculation matches expected physics."""
        pump = WaterPumpThermodynamic(
            pump_id='forward_pump',
            eta_is=0.82,
            eta_m=0.96,
            target_pressure_pa=500000.0  # 5 bar
        )
        
        registry = ComponentRegistry()
        pump.initialize(dt=1.0, registry=registry)
        
        # Create inlet stream: 1 bar, 20°C, 10 kg/s
        inlet_stream = Stream(
            mass_flow_kg_h=36000.0,  # 10 kg/s
            temperature_k=293.15,     # 20°C
            pressure_pa=101325.0,     # 1 atm
            composition={'H2O': 1.0},
            phase='liquid'
        )
        
        pump.receive_input('water_in', inlet_stream, 'water')
        pump.step(t=0.0)
        
        # Verify calculations executed
        assert pump.power_shaft_kw > 0
        assert pump.calculated_T_c > 20.0  # Temperature should rise
        assert pump.outlet_stream is not None
        assert pump.outlet_stream.pressure_pa == 500000.0
    
    def test_forward_matches_legacy_values(self):
        """Test that forward calculation matches validated legacy results."""
        pump = WaterPumpThermodynamic(
            pump_id='legacy_test',
            eta_is=0.82,
            eta_m=0.96,
            target_pressure_pa=500000.0
        )
        
        registry = ComponentRegistry()
        pump.initialize(dt=1.0, registry=registry)
        
        # Exact conditions from validation
        inlet_stream = Stream(
            mass_flow_kg_h=36000.0,
            temperature_k=293.15,
            pressure_pa=101325.0,
            composition={'H2O': 1.0},
            phase='liquid'
        )
        
        pump.receive_input('water_in', inlet_stream, 'water')
        pump.step(t=0.0)
        
        # Expected values from validation (±0.1% tolerance for floating point)
        assert abs(pump.calculated_T_c - 20.02675) < 0.01
        assert abs(pump.work_real_kj_kg - 0.48702) < 0.001
        assert abs(pump.power_shaft_kw - 5.07310) < 0.01
    
    def test_forward_energy_conservation(self):
        """Test that energy is conserved in forward calculation."""
        pump = WaterPumpThermodynamic(
            pump_id='energy_test',
            eta_is=0.80,
            eta_m=0.95,
            target_pressure_pa=1000000.0
        )
        
        registry = ComponentRegistry()
        pump.initialize(dt=1.0, registry=registry)
        
        inlet_stream = Stream(
            mass_flow_kg_h=18000.0,
            temperature_k=298.15,
            pressure_pa=200000.0,
            composition={'H2O': 1.0},
            phase='liquid'
        )
        
        pump.receive_input('water_in', inlet_stream, 'water')
        pump.step(t=0.0)
        
        # Shaft power should be greater than fluid power (mechanical losses)
        assert pump.power_shaft_kw > pump.power_fluid_kw
        
        # Mechanical efficiency relationship
        expected_shaft = pump.power_fluid_kw / 0.95
        assert abs(pump.power_shaft_kw - expected_shaft) < 0.01


class TestWaterPumpReverseCalculation:
    """Test reverse calculation (outlet known → inlet calculated)."""
    
    def test_reverse_calculation_basic(self):
        """Test basic reverse calculation."""
        pump = WaterPumpThermodynamic(
            pump_id='reverse_pump',
            eta_is=0.82,
            eta_m=0.96,
            target_pressure_pa=101325.0  # Target is INLET in reverse mode
        )
        
        registry = ComponentRegistry()
        pump.initialize(dt=1.0, registry=registry)
        
        # Create outlet stream
        outlet_stream = Stream(
            mass_flow_kg_h=36000.0,
            temperature_k=293.20,  # Slightly warmed from pumping
            pressure_pa=500000.0,
            composition={'H2O': 1.0},
            phase='liquid'
        )
        
        pump.receive_input('water_out_reverse', outlet_stream, 'water')
        pump.step(t=0.0)
        
        # Verify calculations executed
        assert pump.power_shaft_kw > 0
        assert pump.inlet_stream is not None
        assert pump.inlet_stream.pressure_pa == 101325.0
    
    def test_reverse_matches_legacy_values(self):
        """Test reverse calculation matches validated legacy results."""
        pump = WaterPumpThermodynamic(
            pump_id='reverse_legacy_test',
            eta_is=0.82,
            eta_m=0.96,
            target_pressure_pa=101325.0
        )
        
        registry = ComponentRegistry()
        pump.initialize(dt=1.0, registry=registry)
        
        # Exact test scenario from validation
        outlet_stream = Stream(
            mass_flow_kg_h=36000.0,
            temperature_k=293.20,  # 20.05°C
            pressure_pa=500000.0,
            composition={'H2O': 1.0},
            phase='liquid'
        )
        
        pump.receive_input('water_out_reverse', outlet_stream, 'water')
        pump.step(t=0.0)
        
        # Expected values from validation
        assert abs(pump.calculated_T_c - 20.02325) < 0.01
        assert abs(pump.work_real_kj_kg - 0.48698) < 0.001
        assert abs(pump.power_shaft_kw - 5.07269) < 0.01


class TestWaterPumpStateManagement:
    """Test state management and reporting."""
    
    def test_get_state_structure(self):
        """Test that get_state returns all required fields."""
        pump = WaterPumpThermodynamic(
            pump_id='state_test',
            target_pressure_pa=300000.0
        )
        
        registry = ComponentRegistry()
        pump.initialize(dt=1.0, registry=registry)
        
        state = pump.get_state()
        
        # Check required fields
        assert 'pump_id' in state
        assert 'power_shaft_kw' in state
        assert 'power_fluid_kw' in state
        assert 'work_real_kj_kg' in state
        assert 'cumulative_energy_kwh' in state
        assert 'cumulative_water_kg' in state
        assert 'eta_is' in state
        assert 'eta_m' in state
    
    def test_cumulative_statistics(self):
        """Test cumulative energy and water tracking."""
        pump = WaterPumpThermodynamic(
            pump_id='cumulative_test',
            eta_is=0.80,
            eta_m=0.95,
            target_pressure_pa=500000.0
        )
        
        registry = ComponentRegistry()
        pump.initialize(dt=1.0, registry=registry)  # 1 hour timestep
        
        inlet_stream = Stream(
            mass_flow_kg_h=1000.0,
            temperature_k=298.15,
            pressure_pa=101325.0,
            composition={'H2O': 1.0},
            phase='liquid'
        )
        
        # Run for 3 timesteps
        for i in range(3):
            pump.receive_input('water_in', inlet_stream, 'water')
            pump.step(t=float(i))
        
        # Verify accumulation
        assert pump.cumulative_water_kg > 0
        assert pump.cumulative_energy_kwh > 0
        
        # Water should be approximately 3000 kg (1000 kg/h * 3 hours)
        assert abs(pump.cumulative_water_kg - 3000.0) < 10.0
        
        # Specific energy should be consistent
        specific_energy = pump.cumulative_energy_kwh / pump.cumulative_water_kg
        assert specific_energy > 0


class TestWaterPumpPortInterface:
    """Test port-based input/output interface."""
    
    def test_get_ports_structure(self):
        """Test port metadata structure."""
        pump = WaterPumpThermodynamic(
            pump_id='port_test', 
            target_pressure_pa=400000.0
        )
        
        ports = pump.get_ports()
        
        assert 'water_in' in ports
        assert 'water_out' in ports
        assert 'electricity_in' in ports
        
        assert ports['water_in']['type'] == 'input'
        assert ports['water_out']['type'] == 'output'
    
    def test_receive_input_water_stream(self):
        """Test receiving water stream input."""
        pump = WaterPumpThermodynamic(
            pump_id='input_test',
            target_pressure_pa=500000.0
        )
        
        stream = Stream(
            mass_flow_kg_h=5000.0,
            temperature_k=293.15,
            pressure_pa=101325.0,
            composition={'H2O': 1.0},
            phase='liquid'
        )
        
        accepted = pump.receive_input('water_in', stream, 'water')
        
        assert accepted == 5000.0
        assert pump.inlet_stream == stream
    
    def test_get_output_water_stream(self):
        """Test getting output stream."""
        pump = WaterPumpThermodynamic(
            pump_id='output_test',
            eta_is=0.82,
            eta_m=0.96,
            target_pressure_pa=500000.0
        )
        
        registry = ComponentRegistry()
        pump.initialize(dt=1.0, registry=registry)
        
        inlet_stream = Stream(
            mass_flow_kg_h=10000.0,
            temperature_k=298.15,
            pressure_pa=150000.0,
            composition={'H2O': 1.0},
            phase='liquid'
        )
        
        pump.receive_input('water_in', inlet_stream, 'water')
        pump.step(t=0.0)
        
        outlet = pump.get_output('water_out')
        
        assert isinstance(outlet, Stream)
        assert outlet.pressure_pa == 500000.0
        assert outlet.temperature_k > 298.15  # Should be warmer
        assert outlet.mass_flow_kg_h == 10000.0


class TestWaterPumpEdgeCases:
    """Test edge cases and error handling."""
    
    def test_zero_flow(self):
        """Test pump behavior with zero flow."""
        pump = WaterPumpThermodynamic(
            pump_id='zero_flow_test',
            target_pressure_pa=500000.0
        )
        
        registry = ComponentRegistry()
        pump.initialize(dt=1.0, registry=registry)
        
        # Step without input
        pump.step(t=0.0)
        
        # Should not crash, power should be zero
        assert pump.power_shaft_kw == 0.0
    
    def test_no_target_pressure(self):
        """Test pump without target pressure set."""
        pump = WaterPumpThermodynamic(
            pump_id='no_target_test'
            # No target_pressure_pa
        )
        
        registry = ComponentRegistry()
        pump.initialize(dt=1.0, registry=registry)
        
        inlet_stream = Stream(
            mass_flow_kg_h=1000.0,
            temperature_k=298.15,
            pressure_pa=101325.0,
            composition={'H2O': 1.0},
            phase='liquid'
        )
        
        pump.receive_input('water_in', inlet_stream, 'water')
        pump.step(t=0.0)
        
        # Should not execute calculations
        assert pump.power_shaft_kw == 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
