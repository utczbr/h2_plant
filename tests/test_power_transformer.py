"""
Unit tests for PowerTransformer component.

Tests the efficiency-based power conversion model for transformer/rectifier units.
"""

import pytest
import numpy as np

from h2_plant.components.power.rectifier import PowerTransformer, Rectifier


def init_trafo(trafo: PowerTransformer) -> PowerTransformer:
    """Helper to initialize a transformer for testing."""
    trafo.initialize(dt=1/60, registry=None)
    return trafo


class TestPowerTransformer:
    """Test suite for PowerTransformer component."""

    def test_initialization_defaults(self):
        """Verify default configuration."""
        trafo = PowerTransformer()
        assert trafo.component_id == "transformer"
        assert trafo.efficiency == 0.95
        assert trafo.rated_power_mw == 20.0
        assert trafo.system_group is None

    def test_initialization_custom(self):
        """Verify custom configuration."""
        trafo = PowerTransformer(
            component_id="SOEC_Trafo",
            efficiency=0.97,
            rated_power_mw=16.0,
            system_group="SOEC"
        )
        assert trafo.component_id == "SOEC_Trafo"
        assert trafo.efficiency == 0.97
        assert trafo.rated_power_mw == 16.0
        assert trafo.system_group == "SOEC"

    def test_efficiency_clamping(self):
        """Efficiency should be clamped between 0.01 and 1.0."""
        trafo_low = PowerTransformer(efficiency=0.0)
        assert trafo_low.efficiency == 0.01

        trafo_high = PowerTransformer(efficiency=1.5)
        assert trafo_high.efficiency == 1.0

        trafo_ok = PowerTransformer(efficiency=0.95)
        assert trafo_ok.efficiency == 0.95

    def test_efficiency_calculation(self):
        """Verify P_out = P_in × η."""
        trafo = init_trafo(PowerTransformer(efficiency=0.95, rated_power_mw=20.0))
        trafo.receive_input('power_in', 10.0)
        trafo.step(0.0)

        assert trafo.power_out_mw == pytest.approx(9.5, rel=1e-3)
        assert trafo.power_loss_mw == pytest.approx(0.5, rel=1e-3)

    def test_energy_balance(self):
        """Verify energy conservation: P_in = P_out + P_loss."""
        trafo = init_trafo(PowerTransformer(efficiency=0.95, rated_power_mw=20.0))
        
        for p_in in [5.0, 10.0, 15.0, 20.0]:
            trafo.receive_input('power_in', p_in)
            trafo.step(0.0)
            
            assert trafo.power_in_mw == pytest.approx(
                trafo.power_out_mw + trafo.power_loss_mw, rel=1e-9
            )

    def test_capacity_limiting(self):
        """Output should be capped at rated power."""
        trafo = init_trafo(PowerTransformer(efficiency=0.95, rated_power_mw=5.0))
        trafo.receive_input('power_in', 10.0)
        trafo.step(0.0)

        assert trafo.power_out_mw == 5.0
        assert trafo.power_loss_mw == 5.0

    def test_load_factor(self):
        """Load factor = P_out / P_rated."""
        trafo = init_trafo(PowerTransformer(efficiency=0.95, rated_power_mw=10.0))
        trafo.receive_input('power_in', 10.0)
        trafo.step(0.0)

        assert trafo.load_factor == pytest.approx(0.95, rel=1e-3)

    def test_zero_input(self):
        """Zero input should yield zero output and losses."""
        trafo = init_trafo(PowerTransformer(efficiency=0.95, rated_power_mw=10.0))
        trafo.receive_input('power_in', 0.0)
        trafo.step(0.0)

        assert trafo.power_out_mw == 0.0
        assert trafo.power_loss_mw == 0.0
        assert trafo.load_factor == 0.0

    def test_heat_output_port(self):
        """Heat output should be in kW (losses × 1000)."""
        trafo = init_trafo(PowerTransformer(efficiency=0.95, rated_power_mw=20.0))
        trafo.receive_input('power_in', 10.0)
        trafo.step(0.0)

        heat_kw = trafo.get_output('heat_out')
        assert heat_kw == pytest.approx(500.0, rel=1e-3)

    def test_power_output_ports(self):
        """Power output ports should return the same value."""
        trafo = init_trafo(PowerTransformer(efficiency=0.95, rated_power_mw=20.0))
        trafo.receive_input('power_in', 10.0)
        trafo.step(0.0)

        assert trafo.get_output('power_out') == pytest.approx(9.5, rel=1e-3)
        assert trafo.get_output('dc_out') == pytest.approx(9.5, rel=1e-3)
        assert trafo.get_output('electricity_out') == pytest.approx(9.5, rel=1e-3)

    def test_input_port_acceptance(self):
        """Input ports should accept power values."""
        trafo = PowerTransformer()

        accepted = trafo.receive_input('power_in', 5.0)
        assert accepted == 5.0
        assert trafo.power_in_mw == 5.0

        accepted = trafo.receive_input('electricity_in', 7.5)
        assert accepted == 7.5
        assert trafo.power_in_mw == 7.5

        accepted = trafo.receive_input('unknown_port', 10.0)
        assert accepted == 0.0

    def test_get_state(self):
        """State snapshot should include all operational parameters."""
        trafo = init_trafo(PowerTransformer(
            component_id="Test_Trafo",
            efficiency=0.96,
            rated_power_mw=15.0,
            system_group="PEM"
        ))
        trafo.receive_input('power_in', 10.0)
        trafo.step(0.0)

        state = trafo.get_state()
        
        assert state['component_id'] == "Test_Trafo"
        assert state['system_group'] == "PEM"
        assert state['efficiency'] == 0.96
        assert state['power_in_mw'] == 10.0
        assert state['power_out_mw'] == pytest.approx(9.6, rel=1e-3)
        assert state['power_loss_mw'] == pytest.approx(0.4, rel=1e-3)
        assert state['heat_loss_kw'] == pytest.approx(400.0, rel=1e-3)

    def test_get_ports(self):
        """Port definitions should include all interfaces."""
        trafo = PowerTransformer()
        ports = trafo.get_ports()

        assert 'power_in' in ports
        assert ports['power_in']['type'] == 'input'
        assert ports['power_in']['resource_type'] == 'electricity'

        assert 'power_out' in ports
        assert ports['power_out']['type'] == 'output'

        assert 'heat_out' in ports
        assert ports['heat_out']['resource_type'] == 'heat'

    def test_backwards_compatibility_alias(self):
        """Rectifier should be an alias for PowerTransformer."""
        assert Rectifier is PowerTransformer

        rect = Rectifier(efficiency=0.97)
        assert isinstance(rect, PowerTransformer)
        assert rect.efficiency == 0.97


class TestTransformerIntegration:
    """Integration tests for transformer in dispatch context."""

    def test_grossup_calculation(self):
        """Verify dispatch gross-up: P_grid = P_stack / η."""
        efficiency = 0.95
        P_stack_target = 14.4
        
        P_grid_required = P_stack_target / efficiency
        
        assert P_grid_required == pytest.approx(15.158, rel=1e-2)
        
        P_stack_delivered = P_grid_required * efficiency
        assert P_stack_delivered == pytest.approx(P_stack_target, rel=1e-6)

    def test_full_round_trip(self):
        """Simulate full dispatch → transformer → stack power flow."""
        trafo = init_trafo(PowerTransformer(efficiency=0.95, rated_power_mw=16.0))
        
        P_stack_target = 14.4
        P_grid = P_stack_target / trafo.efficiency
        
        trafo.receive_input('power_in', P_grid)
        trafo.step(0.0)
        
        P_stack_actual = trafo.get_output('power_out')
        
        assert P_stack_actual == pytest.approx(P_stack_target, rel=1e-3)
        assert trafo.power_loss_mw == pytest.approx(P_grid - P_stack_target, rel=1e-3)
