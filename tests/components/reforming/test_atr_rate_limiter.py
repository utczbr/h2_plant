"""
Unit tests for ATR Rate Limiter.

Tests the Quasi-Steady State rate-limiting model that constrains
O₂ flow rate changes to model physical plant dynamics.
"""

import pytest
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from h2_plant.components.reforming.atr_rate_limiter import (
    ATRRateLimiter,
    ATRRateLimiterConfig
)


class TestATRRateLimiterConfig:
    """Tests for configuration dataclass."""
    
    def test_default_config(self):
        """Default config has expected values."""
        config = ATRRateLimiterConfig()
        assert config.min_o2_kmol_h == 7.125
        assert config.max_o2_kmol_h == 23.75
        assert config.max_ramp_rate_kmol_h_min == 1.6625
        assert config.idle_threshold_kmol_h == 0.1
    
    def test_custom_config(self):
        """Custom config values are respected."""
        config = ATRRateLimiterConfig(
            min_o2_kmol_h=10.0,
            max_o2_kmol_h=20.0,
            max_ramp_rate_kmol_h_min=2.0
        )
        assert config.min_o2_kmol_h == 10.0
        assert config.max_o2_kmol_h == 20.0
        assert config.max_ramp_rate_kmol_h_min == 2.0


class TestATRRateLimiterInitialization:
    """Tests for rate limiter initialization."""
    
    def test_default_initialization(self):
        """Default initialization starts at minimum O₂."""
        limiter = ATRRateLimiter()
        assert limiter.f_o2_internal == 7.125
        assert limiter.target_o2 == 7.125
        assert limiter.is_ramping == False
    
    def test_custom_initial_value(self):
        """Custom initial O₂ value is set correctly."""
        limiter = ATRRateLimiter(initial_o2=15.0)
        assert limiter.f_o2_internal == 15.0
        assert limiter.target_o2 == 15.0
        assert limiter.is_ramping == False
    
    def test_initial_value_clamped_to_bounds(self):
        """Initial value outside bounds is clamped."""
        limiter = ATRRateLimiter(initial_o2=50.0)  # Above max
        assert limiter.f_o2_internal == 23.75


class TestATRRateLimiterUpdate:
    """Tests for the core update algorithm."""
    
    def test_ramp_up_from_min_to_max(self):
        """
        Ramping from 30% to 100% capacity should take 10 minutes.
        
        Physics: Δ = 23.75 - 7.125 = 16.625 kmol/h
                 Rate = 1.6625 kmol/hr/min
                 Time = 16.625 / 1.6625 = 10 minutes
        """
        limiter = ATRRateLimiter(initial_o2=7.125)
        
        # 10 minutes * 60 seconds = 600 seconds
        # Simulate with 1-second timesteps
        for step in range(600):
            result = limiter.update(target_o2=23.75, dt_seconds=1.0)
        
        # After 10 minutes, should be very close to target
        assert limiter.f_o2_internal == pytest.approx(23.75, abs=0.01)
    
    def test_ramp_down_from_max_to_min(self):
        """
        Ramping from 100% to 30% capacity should also take 10 minutes.
        """
        limiter = ATRRateLimiter(initial_o2=23.75)
        
        # Simulate 10 minutes with 1-second timesteps
        for step in range(600):
            result = limiter.update(target_o2=7.125, dt_seconds=1.0)
        
        assert limiter.f_o2_internal == pytest.approx(7.125, abs=0.01)
    
    def test_snap_to_target_small_step(self):
        """Small steps within rate limit should snap immediately."""
        limiter = ATRRateLimiter(initial_o2=15.0)
        
        # One second timestep allows: 1.6625 / 60 = 0.0277 kmol/h step
        # Request 15.01 (within limit)
        result = limiter.update(target_o2=15.01, dt_seconds=1.0)
        
        assert result == pytest.approx(15.01, abs=0.001)
        assert limiter.is_ramping == False
    
    def test_rate_limiting_during_ramp(self):
        """During ramp, should move exactly at rate limit."""
        limiter = ATRRateLimiter(initial_o2=10.0)
        
        # One 60-second step = 1 minute
        # Expected step = 1.6625 kmol/h
        result = limiter.update(target_o2=20.0, dt_seconds=60.0)
        
        expected = 10.0 + 1.6625
        assert result == pytest.approx(expected, abs=0.001)
        assert limiter.is_ramping == True
    
    def test_boundary_clamping_above_max(self):
        """Target above max should clamp to max."""
        limiter = ATRRateLimiter(initial_o2=23.0)
        
        # Request 30.0, well above max
        for _ in range(120):  # 2 minutes
            result = limiter.update(target_o2=30.0, dt_seconds=1.0)
        
        # Should be clamped at max
        assert limiter.f_o2_internal == pytest.approx(23.75, abs=0.01)
    
    def test_boundary_clamping_below_min(self):
        """Target below min should clamp to min."""
        limiter = ATRRateLimiter(initial_o2=10.0)
        
        # Request 0.0, well below min (but not idle)
        result = limiter.update(target_o2=5.0, dt_seconds=60.0)
        
        # Should ramp towards min
        assert limiter.target_o2 == 7.125
    
    def test_idle_handling(self):
        """Idle state (O₂ < 0.1) should set to minimum."""
        limiter = ATRRateLimiter(initial_o2=15.0)
        
        result = limiter.update(target_o2=0.05, dt_seconds=1.0)
        
        assert result == 7.125
        assert limiter.f_o2_internal == 7.125
        assert limiter.is_ramping == False
    
    def test_intermediate_ramp_values(self):
        """Verify intermediate values during ramp."""
        limiter = ATRRateLimiter(initial_o2=7.125)
        
        # After 5 minutes (300 seconds), should be halfway
        for _ in range(300):
            limiter.update(target_o2=23.75, dt_seconds=1.0)
        
        expected_after_5min = 7.125 + (5 * 1.6625)  # 15.4375
        assert limiter.f_o2_internal == pytest.approx(expected_after_5min, abs=0.1)
        assert limiter.is_ramping == True


class TestATRRateLimiterHelpers:
    """Tests for helper methods."""
    
    def test_ramp_time_remaining_at_target(self):
        """At target, remaining time should be zero."""
        limiter = ATRRateLimiter(initial_o2=15.0)
        limiter.update(target_o2=15.0, dt_seconds=1.0)
        
        assert limiter.get_ramp_time_remaining() == 0.0
    
    def test_ramp_time_remaining_during_ramp(self):
        """During ramp, should estimate remaining time."""
        limiter = ATRRateLimiter(initial_o2=7.125)
        limiter.update(target_o2=23.75, dt_seconds=1.0)  # Start ramping
        
        remaining = limiter.get_ramp_time_remaining()
        # Distance ≈ 16.625, rate = 1.6625, time ≈ 10 min
        assert remaining == pytest.approx(10.0, abs=0.5)
    
    def test_reset_to_default(self):
        """Reset to default should go to minimum."""
        limiter = ATRRateLimiter(initial_o2=20.0)
        limiter.update(target_o2=23.75, dt_seconds=60.0)
        
        limiter.reset()
        
        assert limiter.f_o2_internal == 7.125
        assert limiter.target_o2 == 7.125
        assert limiter.is_ramping == False
    
    def test_reset_to_custom_value(self):
        """Reset to custom value."""
        limiter = ATRRateLimiter(initial_o2=10.0)
        limiter.reset(o2_value=18.0)
        
        assert limiter.f_o2_internal == 18.0
        assert limiter.target_o2 == 18.0
    
    def test_ramp_progress_at_target(self):
        """Ramp progress at target should be 1.0."""
        limiter = ATRRateLimiter(initial_o2=15.0)
        assert limiter.ramp_progress == 1.0
    
    def test_ramp_progress_during_ramp(self):
        """Ramp progress during ramp should be fractional."""
        limiter = ATRRateLimiter(initial_o2=7.125)
        
        # Start ramping
        limiter.update(target_o2=23.75, dt_seconds=1.0)
        
        progress = limiter.ramp_progress
        assert 0.0 < progress < 1.0


class TestATRRateLimiterPhysicsValidation:
    """
    Physics validation tests matching the user's specification.
    
    From the specification:
    - O₂ range: 7.125 – 23.75 kmol/hr
    - Max ramp rate: 1.6625 kmol/hr/min (transition 30%→100% in 10 min)
    """
    
    def test_10_minute_full_transition(self):
        """
        Critical test: Full transition from min to max takes exactly 10 minutes.
        
        This validates the core physics constraint from the user's spec.
        """
        limiter = ATRRateLimiter()
        
        # Record O₂ values at each minute
        o2_values = [limiter.f_o2_internal]
        
        for minute in range(10):
            # 60 1-second steps per minute
            for second in range(60):
                limiter.update(target_o2=23.75, dt_seconds=1.0)
            o2_values.append(limiter.f_o2_internal)
        
        # At t=0: 7.125 kmol/h (30%)
        assert o2_values[0] == pytest.approx(7.125, abs=0.01)
        
        # At t=10min: 23.75 kmol/h (100%)
        assert o2_values[10] == pytest.approx(23.75, abs=0.01)
        
        # Verify linear increase (~1.6625 kmol/h per minute)
        for i in range(10):
            expected_increment = 1.6625
            actual_increment = o2_values[i+1] - o2_values[i]
            assert actual_increment == pytest.approx(expected_increment, abs=0.1)
    
    def test_outputs_track_along_curve(self):
        """
        Validate that intermediate O₂ values are always valid for lookup.
        
        This ensures the QSS property: all intermediate states are 
        thermodynamically valid operating points.
        """
        limiter = ATRRateLimiter()
        MIN_O2 = 7.125
        MAX_O2 = 23.75
        
        # Ramp from min to max
        for _ in range(600):  # 10 minutes
            o2 = limiter.update(target_o2=MAX_O2, dt_seconds=1.0)
            
            # Every intermediate value should be within regression domain
            assert MIN_O2 <= o2 <= MAX_O2, f"O₂={o2} outside valid domain"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
