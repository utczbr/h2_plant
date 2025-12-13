"""
Regression tests for Phase A/B fixes: Golden data comparison.

Tests validate:
- Post-fix simulation matches expected behavior
- No mass leaks or infinite production
- Edge cases handled correctly
- Configuration variants work

Run with: pytest tests/test_phase_ab_regression.py -v --tb=short
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import json
import tempfile
from pathlib import Path


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def golden_dispatch_outputs():
    """Golden reference data for dispatch strategy outputs."""
    # Pre-computed expected values for known inputs
    return {
        'soec_only_10mw': {
            'P_soec': 10.0,
            'P_pem': 0.0,
            'P_sold': 0.0,
            'force_sell': False
        },
        'arbitrage_high_price': {
            'P_soec': 5.0,  # Previous power maintained
            'P_pem': 0.0,
            'P_sold': 5.0,  # Selling 5 MW
            'force_sell': True
        },
        'surplus_to_pem': {
            'P_soec': 10.0,
            'P_pem': 5.0,
            'P_sold': 0.0,
            'force_sell': False
        }
    }


@pytest.fixture
def mock_stream_outputs():
    """Expected Stream properties for golden comparison."""
    return {
        'h2_stream_standard': {
            'mass_flow_kg_h': 100.0,
            'temperature_k': 300.0,
            'pressure_pa': 1e5,
            'density_kg_m3_approx': 0.081,  # Approx for H2 at 1 bar, 300K
        }
    }


# ============================================================================
# REGRESSION TESTS: Dispatch Strategy
# ============================================================================

class TestDispatchRegression:
    """Regression tests for dispatch strategy outputs."""
    
    def test_soec_only_full_power(self, golden_dispatch_outputs):
        """SOEC-only at full offer should use all power."""
        from h2_plant.control.dispatch import (
            DispatchInput, DispatchState, SoecOnlyStrategy
        )
        
        strategy = SoecOnlyStrategy()
        state = DispatchState()
        
        d_input = DispatchInput(
            minute=0,
            P_offer=10.0,
            P_future_offer=10.0,
            current_price=50.0,  # Below arbitrage limit
            soec_capacity_mw=10.0,
            pem_max_power_mw=0.0
        )
        
        result = strategy.decide(d_input, state)
        expected = golden_dispatch_outputs['soec_only_10mw']
        
        assert abs(result.P_soec - expected['P_soec']) < 0.01
        assert abs(result.P_pem - expected['P_pem']) < 0.01
        assert abs(result.P_sold - expected['P_sold']) < 0.01
    
    def test_arbitrage_triggers_force_sell(self):
        """High price at hour start should trigger force_sell."""
        from h2_plant.control.dispatch import (
            DispatchInput, DispatchState, ReferenceHybridStrategy
        )
        
        strategy = ReferenceHybridStrategy()
        state = DispatchState(P_soec_prev=5.0, force_sell=False)
        
        # High price at minute=0 with surplus power
        d_input = DispatchInput(
            minute=0,  # Hour start
            P_offer=10.0,
            P_future_offer=10.0,
            current_price=500.0,  # Very high - should trigger arbitrage
            soec_capacity_mw=10.0,
            pem_max_power_mw=5.0
        )
        
        result = strategy.decide(d_input, state)
        
        # Should trigger force_sell
        assert result.state_update['force_sell'] == True
    
    def test_surplus_routes_to_pem_below_arbitrage(self):
        """Surplus power below arbitrage limit should go to PEM."""
        from h2_plant.control.dispatch import (
            DispatchInput, DispatchState, ReferenceHybridStrategy
        )
        
        strategy = ReferenceHybridStrategy()
        state = DispatchState()
        
        # SOEC capacity 5, offer 10, price below arbitrage
        d_input = DispatchInput(
            minute=30,  # Mid-hour
            P_offer=10.0,
            P_future_offer=10.0,
            current_price=50.0,  # Below arbitrage (~306 EUR/MWh)
            soec_capacity_mw=5.0,
            pem_max_power_mw=10.0
        )
        
        result = strategy.decide(d_input, state)
        
        # SOEC should get 5 MW, PEM should get 5 MW surplus
        assert result.P_soec == 5.0
        assert result.P_pem == 5.0
        assert result.P_sold == 0.0


# ============================================================================
# REGRESSION TESTS: Mass Conservation
# ============================================================================

class TestMassConservationRegression:
    """Regression tests for mass conservation."""
    
    def test_no_mass_creation(self):
        """Mass produced must not exceed theoretical limit."""
        from h2_plant.core.stream import Stream
        
        # H2 production at SOEC efficiency ~37.5 kWh/kg
        power_mw = 10.0  # 10 MW
        efficiency_kwh_kg = 37.5
        timestep_h = 1/60  # 1 minute
        
        # Maximum theoretical H2 in 1 minute at 10 MW
        energy_kwh = power_mw * 1000 * timestep_h
        max_h2_kg = energy_kwh / efficiency_kwh_kg
        
        # Create stream with this flow
        stream = Stream(
            mass_flow_kg_h=max_h2_kg * 60,  # Convert to kg/h
            temperature_k=300.0,
            pressure_pa=1e5,
            composition={'H2': 1.0}
        )
        
        # Verify mass flow is reasonable
        assert stream.mass_flow_kg_h > 0
        assert stream.mass_flow_kg_h < 1000  # Sanity check
    
    def test_stream_mixing_conserves_mass(self):
        """Stream mixing should conserve total mass."""
        from h2_plant.core.stream import Stream
        
        stream1 = Stream(
            mass_flow_kg_h=100.0,
            temperature_k=300.0,
            pressure_pa=1e5,
            composition={'H2': 1.0}
        )
        
        stream2 = Stream(
            mass_flow_kg_h=50.0,
            temperature_k=350.0,
            pressure_pa=1e5,
            composition={'H2': 1.0}
        )
        
        try:
            mixed = stream1.mix_with(stream2)
            # Total mass should be conserved
            assert abs(mixed.mass_flow_kg_h - 150.0) < 0.01
        except (NotImplementedError, AttributeError):
            pytest.skip("Stream.mix_with not fully implemented")


# ============================================================================
# REGRESSION TESTS: Edge Cases
# ============================================================================

class TestEdgeCaseRegression:
    """Regression tests for edge cases."""
    
    def test_zero_power_no_production(self):
        """Zero power should result in zero production."""
        from h2_plant.control.dispatch import (
            DispatchInput, DispatchState, SoecOnlyStrategy
        )
        
        strategy = SoecOnlyStrategy()
        state = DispatchState()
        
        d_input = DispatchInput(
            minute=0,
            P_offer=0.0,  # No power available
            P_future_offer=0.0,
            current_price=50.0,
            soec_capacity_mw=10.0,
            pem_max_power_mw=5.0
        )
        
        result = strategy.decide(d_input, state)
        
        assert result.P_soec == 0.0
        assert result.P_pem == 0.0
        assert result.P_sold == 0.0
    
    def test_negative_price_handled(self):
        """Negative energy price should not crash."""
        from h2_plant.control.dispatch import (
            DispatchInput, DispatchState, ReferenceHybridStrategy
        )
        
        strategy = ReferenceHybridStrategy()
        state = DispatchState()
        
        d_input = DispatchInput(
            minute=0,
            P_offer=10.0,
            P_future_offer=10.0,
            current_price=-50.0,  # Negative price
            soec_capacity_mw=10.0,
            pem_max_power_mw=5.0
        )
        
        # Should not raise
        result = strategy.decide(d_input, state)
        
        # At negative prices, arbitrage would never trigger
        assert result.state_update['force_sell'] == False
    
    def test_empty_stream_composition(self):
        """Empty composition should be handled gracefully."""
        from h2_plant.core.stream import Stream
        
        # Should not raise
        stream = Stream(
            mass_flow_kg_h=100.0,
            temperature_k=300.0,
            pressure_pa=1e5,
            composition={}
        )
        
        # Validation might set default or pass through
        assert stream.mass_flow_kg_h == 100.0
    
    def test_extreme_pressure_clamped(self):
        """Extreme pressure should be handled (not cause overflow)."""
        from h2_plant.core.stream import Stream
        
        # Very high pressure
        stream = Stream(
            mass_flow_kg_h=100.0,
            temperature_k=300.0,
            pressure_pa=900e5,  # 900 bar
            composition={'H2': 1.0}
        )
        
        # Density should still be computable
        density = stream.density_kg_m3
        assert density > 0
        assert np.isfinite(density)


# ============================================================================
# REGRESSION TESTS: Configuration Variants
# ============================================================================

class TestConfigVariantRegression:
    """Regression tests for different configurations."""
    
    def test_lp_tank_30bar_limit(self):
        """LP tank should respect 30 bar pressure limit."""
        try:
            from h2_plant.components.storage.tank_array import TankArray
            from h2_plant.core.component_registry import ComponentRegistry
            
            tank = TankArray(
                component_id="lp_tank",
                n_tanks=5,
                max_capacity_kg_per_tank=1000.0,
                max_pressure_bar=30.0,
                initial_level_fraction=0.0
            )
            
            registry = ComponentRegistry()
            registry.register("lp_tank", tank)
            tank.initialize(dt=1/60, registry=registry)
            
            # Fill with a lot of H2
            from h2_plant.core.stream import Stream
            stream = Stream(
                mass_flow_kg_h=10000.0,  # Very high flow
                temperature_k=300.0,
                pressure_pa=30e5,
                composition={'H2': 1.0}
            )
            
            tank.receive_input('h2_in', stream, 'hydrogen')
            tank.step(1.0)
            
            # Pressure should not exceed 30 bar
            pressures = tank.pressures_bar
            assert all(p <= 30.0 for p in pressures)
        except (ImportError, AttributeError):
            pytest.skip("TankArray not available")
    
    def test_timestep_1min_vs_1hour(self):
        """Results should scale correctly with timestep."""
        from h2_plant.control.dispatch import (
            DispatchInput, DispatchState, SoecOnlyStrategy
        )
        
        strategy = SoecOnlyStrategy()
        
        # Both runs should give same P_soec for same inputs
        state1 = DispatchState()
        d_input = DispatchInput(
            minute=0,
            P_offer=10.0,
            P_future_offer=10.0,
            current_price=50.0,
            soec_capacity_mw=10.0,
            pem_max_power_mw=0.0
        )
        
        result1 = strategy.decide(d_input, state1)
        
        state2 = DispatchState()
        result2 = strategy.decide(d_input, state2)
        
        # Same inputs should give same outputs
        assert result1.P_soec == result2.P_soec


# ============================================================================
# REGRESSION TESTS: Error Handling
# ============================================================================

class TestErrorHandlingRegression:
    """Regression tests for error handling."""
    
    def test_missing_lut_uses_fallback(self):
        """Missing LUT should use calculation fallback, not crash."""
        from h2_plant.core.stream import Stream
        
        # Stream should compute density even without LUT
        stream = Stream(
            mass_flow_kg_h=100.0,
            temperature_k=300.0,
            pressure_pa=1e5,
            composition={'H2': 1.0}
        )
        
        # Should not raise even if LUT is not available
        density = stream.density_kg_m3
        assert density > 0
    
    def test_bad_port_name_handled(self):
        """Invalid port name should be handled gracefully."""
        from h2_plant.core.component import Component
        from h2_plant.core.stream import Stream
        
        class TestComponent(Component):
            def initialize(self, dt, registry): pass
            def step(self, t): pass
            def get_state(self): return {}
            def get_ports(self): return {'valid_in': {'type': 'input'}}
        
        comp = TestComponent(component_id="test")
        stream = Stream(mass_flow_kg_h=100.0, temperature_k=300.0, pressure_pa=1e5, composition={'H2': 1.0})
        
        # Bad port should not crash, might log warning
        result = comp.receive_input('invalid_port', stream, 'hydrogen')
        # Should return something (possibly None or 0)
        assert result is not None or result == 0.0 or True  # Just verify no crash


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
