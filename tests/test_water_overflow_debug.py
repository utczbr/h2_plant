"""
Debug tests for UltraPure Water Tank overflow issue.
Run with: python -m pytest tests/test_water_overflow_debug.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import MagicMock

from h2_plant.components.water.ultrapure_water_tank import UltraPureWaterTank
from h2_plant.components.external.water_source import ExternalWaterSource
from h2_plant.core.stream import Stream


class TestUltraPureWaterTankZoneLogic:
    """Test Zone C (Stop) logic in UltraPureWaterTank."""
    
    def test_zone_c_threshold_constants(self):
        """Verify Zone C constants are correct."""
        assert UltraPureWaterTank.ZONE_C_ENTER == 0.90, "Zone C should enter at 90%"
        assert UltraPureWaterTank.ZONE_C_MULTIPLIER == 0.0, "Zone C multiplier must be 0"
    
    def test_zone_c_triggers_at_90_percent(self):
        """Tank at 90%+ fill should enter Zone C and request 0 production."""
        tank = UltraPureWaterTank(
            component_id="TestTank",
            capacity_kg=20000.0,
            nominal_production_kg_h=10000.0,
            initial_fill_fraction=0.5  # Start at 50%
        )
        
        # Mock registry and initialize
        registry = MagicMock()
        registry.get_by_type.return_value = []
        tank.initialize(dt=1/60, registry=registry)  # 1 minute timestep
        
        # Manually set mass to 91% (above Zone C threshold)
        tank.mass_kg = 18200.0  # 91% of 20000
        
        # Run step
        tank.step(t=0.0)
        
        # Verify Zone C and zero production request
        assert tank.control_zone == 'C', f"Expected Zone C, got {tank.control_zone}"
        assert tank.requested_production_kg_h == 0.0, \
            f"Expected 0 production request, got {tank.requested_production_kg_h}"
    
    def test_zone_c_at_100_percent(self):
        """Tank at 100% fill should definitely be in Zone C."""
        tank = UltraPureWaterTank(
            component_id="TestTank",
            capacity_kg=20000.0,
            nominal_production_kg_h=10000.0,
            initial_fill_fraction=1.0  # Start full
        )
        
        registry = MagicMock()
        registry.get_by_type.return_value = []
        tank.initialize(dt=1/60, registry=registry)
        
        tank.step(t=0.0)
        
        assert tank.control_zone == 'C', f"Full tank must be Zone C, got {tank.control_zone}"
        assert tank.requested_production_kg_h == 0.0, \
            f"Full tank should request 0, got {tank.requested_production_kg_h}"
    
    def test_control_signal_output_is_zero_in_zone_c(self):
        """Verify get_output('control_signal') returns 0 flow when in Zone C."""
        tank = UltraPureWaterTank(
            component_id="TestTank",
            capacity_kg=20000.0,
            nominal_production_kg_h=10000.0,
            initial_fill_fraction=0.95  # 95% -> Zone C
        )
        
        registry = MagicMock()
        registry.get_by_type.return_value = []
        tank.initialize(dt=1/60, registry=registry)
        tank.step(t=0.0)
        
        signal = tank.get_output('control_signal')
        
        assert signal is not None, "control_signal should return a Stream"
        assert signal.mass_flow_kg_h == 0.0, \
            f"Zone C signal should be 0, got {signal.mass_flow_kg_h}"


class TestExternalWaterSourceSignalControl:
    """Test ExternalWaterSource responds correctly to control signals."""
    
    def test_external_control_mode_initialized(self):
        """Verify source correctly parses external_control mode from dict."""
        params = {
            'mode': 'external_control',
            'flow_rate_kg_h': 12000.0,
            'pressure_bar': 5.0,
            'temperature_c': 25.0
        }
        source = ExternalWaterSource(params)
        
        assert source.mode == 'external_control', \
            f"Expected external_control, got {source.mode}"
    
    def test_zero_signal_produces_zero_flow(self):
        """Source with 0 signal should produce 0 flow."""
        source = ExternalWaterSource(
            mode='external_control',
            flow_rate_kg_h=12000.0
        )
        
        registry = MagicMock()
        source.initialize(dt=1/60, registry=registry)
        
        # Send zero signal
        zero_signal = Stream(
            mass_flow_kg_h=0.0,
            temperature_k=293.15,
            pressure_pa=101325,
            composition={'Signal': 1.0},
            phase='signal'
        )
        source.receive_input('control_signal', zero_signal, 'signal')
        
        # Run step
        source.step(t=0.0)
        
        assert source.current_flow_kg_h == 0.0, \
            f"Expected 0 flow with 0 signal, got {source.current_flow_kg_h}"
    
    def test_no_signal_defaults_to_zero(self):
        """Source with no signal should default to 0 (fail-closed)."""
        source = ExternalWaterSource(
            mode='external_control',
            flow_rate_kg_h=12000.0
        )
        
        registry = MagicMock()
        source.initialize(dt=1/60, registry=registry)
        
        # Do NOT send any signal
        source.step(t=0.0)
        
        assert source.current_flow_kg_h == 0.0, \
            f"Expected 0 flow with no signal (fail-closed), got {source.current_flow_kg_h}"
    
    def test_signal_zero_order_hold(self):
        """Source should remember last signal (Zero-Order Hold)."""
        source = ExternalWaterSource(
            mode='external_control',
            flow_rate_kg_h=12000.0
        )
        
        registry = MagicMock()
        source.initialize(dt=1/60, registry=registry)
        
        # Send signal of 5000
        signal = Stream(
            mass_flow_kg_h=5000.0,
            temperature_k=293.15,
            pressure_pa=101325,
            composition={'Signal': 1.0},
            phase='signal'
        )
        source.receive_input('control_signal', signal, 'signal')
        source.step(t=0.0)
        
        assert source.current_flow_kg_h == 5000.0
        
        # Step again WITHOUT new signal - should hold previous value
        source.step(t=1/60)
        
        assert source.current_flow_kg_h == 5000.0, \
            f"Expected held value 5000, got {source.current_flow_kg_h}"


class TestGraphBuilderCapacity:
    """Test that GraphBuilder correctly passes capacity_kg."""
    
    def test_capacity_kg_from_params(self):
        """Verify capacity_kg parameter is correctly used."""
        from h2_plant.core.graph_builder import PlantGraphBuilder
        from unittest.mock import MagicMock
        
        # Create a minimal node mock
        class MockNode:
            id = "UltraPure_Tank"
            type = "UltraPureWaterTank"
            params = {
                'capacity_kg': 20000.0,
                'nominal_production_kg_h': 10000.0,
                'initial_fill_fraction': 0.7
            }
            connections = []
        
        # Create mock context
        context = MagicMock()
        builder = PlantGraphBuilder(context)
        
        # Call _create_component directly
        tank = builder._create_component(MockNode())
        
        assert tank.capacity_kg == 20000.0, \
            f"Expected capacity 20000, got {tank.capacity_kg}"


class TestFlowNetworkSignalTransfer:
    """Test FlowNetwork correctly transfers signals including zero values."""
    
    def test_zero_signal_is_transferred(self):
        """FlowNetwork must transfer signal streams even with mass_flow_kg_h = 0."""
        from h2_plant.simulation.flow_network import FlowNetwork
        from h2_plant.config.plant_config import ConnectionConfig
        from unittest.mock import MagicMock, patch
        
        # Create mock components
        tank = MagicMock()
        source = MagicMock()
        
        # Tank returns a zero-value signal (Zone C = Stop)
        zero_signal = Stream(
            mass_flow_kg_h=0.0,
            temperature_k=293.15,
            pressure_pa=101325,
            composition={'Signal': 1.0},
            phase='signal'
        )
        tank.get_output.return_value = zero_signal
        tank.get_ports.return_value = {'control_signal': {'type': 'output', 'resource_type': 'signal'}}
        
        source.receive_input.return_value = 0.0
        source.get_ports.return_value = {'control_signal': {'type': 'input', 'resource_type': 'signal'}}
        
        # Create mock registry that returns our mock components
        registry = MagicMock()
        registry.has.return_value = True
        registry.get.side_effect = lambda x: tank if x == "UltraPure_Tank" else source
        
        # Create connection for signal
        conn = ConnectionConfig(
            source_id="UltraPure_Tank",
            source_port="control_signal",
            target_id="Water_Source",
            target_port="control_signal",
            resource_type="signal"
        )
        
        # Create and initialize flow network
        network = FlowNetwork(registry, [conn])
        network.initialize()
        
        # Execute flows
        network.execute_flows(t=0.0)
        
        # Verify the signal was transferred (receive_input was called)
        source.receive_input.assert_called_once()
        call_args = source.receive_input.call_args
        assert call_args[1]['port_name'] == 'control_signal'
        assert call_args[1]['value'].mass_flow_kg_h == 0.0, \
            "Zero-value signal should be transferred to source"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
