"""
Unit tests for EconomicSpotDispatchStrategy.

Tests verify:
1. Spot purchase activation based on price threshold
2. RFNBO protection (renewable power prioritized for SOEC)
3. Hydrogen classification accuracy
4. Cumulative tracking of RFNBO vs non-RFNBO hydrogen
"""
import pytest
from h2_plant.control.dispatch import (
    DispatchInput, 
    DispatchState, 
    EconomicSpotDispatchStrategy
)


class TestSpotActivation:
    """Test spot purchase activation logic."""
    
    def test_spot_not_activated_when_price_too_high(self):
        """Verify spot purchases disabled when price > threshold.
        
        Threshold = 2.0 EUR/kg / 0.050 MWh/kg = 40 EUR/MWh
        Spot price 60 > 40, should NOT activate
        """
        strategy = EconomicSpotDispatchStrategy()
        
        inputs = DispatchInput(
            minute=0,
            P_offer=50.0,
            P_future_offer=50.0,
            current_price=60.0,  # Too expensive
            soec_capacity_mw=40.0,
            pem_max_power_mw=20.0,
            h2_non_rfnbo_price_eur_kg=2.0,
            pem_h2_kwh_kg=50.0
        )
        
        result = strategy.decide(inputs, DispatchState())
        
        assert result.state_update.get('spot_purchased_mw', 0.0) == 0.0
        assert result.state_update.get('h2_non_rfnbo_kg', 0.0) == 0.0
    
    def test_spot_activated_when_price_below_threshold(self):
        """Verify spot purchases enabled when price < threshold.
        
        Threshold = 2.0 EUR/kg / 0.050 MWh/kg = 40 EUR/MWh
        Spot price 30 < 40, should activate
        """
        strategy = EconomicSpotDispatchStrategy()
        
        inputs = DispatchInput(
            minute=0,
            P_offer=50.0,
            P_future_offer=50.0,
            current_price=30.0,  # Cheap enough
            soec_capacity_mw=40.0,
            pem_max_power_mw=20.0,
            h2_non_rfnbo_price_eur_kg=2.0,
            pem_h2_kwh_kg=50.0,
            p_grid_max_mw=30.0
        )
        
        result = strategy.decide(inputs, DispatchState())
        
        # SOEC takes 40 MW, PEM takes 10 MW renewable
        # PEM still has 10 MW capacity available for spot
        assert result.state_update.get('spot_purchased_mw', 0.0) > 0.0
    
    def test_spot_threshold_calculation(self):
        """Verify spot threshold is calculated correctly."""
        strategy = EconomicSpotDispatchStrategy()
        
        inputs = DispatchInput(
            minute=0,
            P_offer=100.0,
            P_future_offer=100.0,
            current_price=35.0,
            soec_capacity_mw=80.0,
            pem_max_power_mw=30.0,
            h2_non_rfnbo_price_eur_kg=2.0,  # EUR/kg
            pem_h2_kwh_kg=50.0  # kWh/kg = 0.050 MWh/kg
        )
        
        result = strategy.decide(inputs, DispatchState())
        
        # Threshold = 2.0 / 0.050 = 40 EUR/MWh
        assert result.state_update.get('spot_threshold_eur_mwh') == pytest.approx(40.0, rel=0.01)


class TestRFNBOProtection:
    """Test RFNBO compliance protection."""
    
    def test_soec_gets_priority_for_renewable(self):
        """Verify SOEC always consumes renewable power first."""
        strategy = EconomicSpotDispatchStrategy()
        
        inputs = DispatchInput(
            minute=0,
            P_offer=100.0,
            P_future_offer=100.0,
            current_price=30.0,  # Cheap spot
            soec_capacity_mw=80.0,
            pem_max_power_mw=50.0
        )
        
        result = strategy.decide(inputs, DispatchState())
        
        # SOEC should get full capacity from renewable
        assert result.P_soec == 80.0
        # PEM gets remainder of renewable (20 MW) plus potential spot
        assert result.P_pem >= 20.0
    
    def test_renewable_surplus_sold(self):
        """Verify excess renewable power is sold when PEM is at capacity."""
        strategy = EconomicSpotDispatchStrategy()
        
        inputs = DispatchInput(
            minute=0,
            P_offer=150.0,  # More than SOEC + PEM can use
            P_future_offer=150.0,
            current_price=50.0,  # Above threshold, no spot
            soec_capacity_mw=80.0,
            pem_max_power_mw=30.0,
            h2_non_rfnbo_price_eur_kg=2.0,
            pem_h2_kwh_kg=50.0
        )
        
        result = strategy.decide(inputs, DispatchState())
        
        # SOEC: 80 MW, PEM renewable: 30 MW (capped at max)
        # Surplus: 150 - 80 - 30 = 40 MW should be sold
        assert result.P_soec == 80.0
        # No spot because price too high
        assert result.state_update.get('spot_purchased_mw', 0.0) == 0.0
        assert result.P_sold == pytest.approx(40.0, rel=0.01)


class TestH2Classification:
    """Test hydrogen classification accuracy."""
    
    def test_h2_classification_with_spot(self):
        """Verify hydrogen is correctly classified as RFNBO vs non-RFNBO."""
        strategy = EconomicSpotDispatchStrategy()
        
        # Scenario: 100 MW renewable, 80 MW SOEC, 20 MW remaining for PEM
        # Cheap spot enables additional 10 MW grid purchase for PEM
        inputs = DispatchInput(
            minute=0,
            P_offer=100.0,
            P_future_offer=100.0,
            current_price=30.0,  # Cheap = spot purchases enabled
            soec_capacity_mw=80.0,
            pem_max_power_mw=30.0,  # Can take 20 renewable + 10 spot
            p_grid_max_mw=10.0,  # Limit spot to 10 MW
            h2_non_rfnbo_price_eur_kg=2.0,
            pem_h2_kwh_kg=50.0,
            soec_h2_kwh_kg=37.5
        )
        
        result = strategy.decide(inputs, DispatchState())
        
        # RFNBO H2 from: SOEC (80 MW) + PEM renewable (20 MW)
        # Non-RFNBO H2 from: PEM spot (10 MW)
        assert result.state_update.get('h2_rfnbo_kg', 0) > 0
        assert result.state_update.get('h2_non_rfnbo_kg', 0) > 0
        assert result.state_update.get('spot_purchased_mw') == 10.0
    
    def test_all_rfnbo_when_spot_too_expensive(self):
        """Verify all H2 is RFNBO when spot price exceeds threshold."""
        strategy = EconomicSpotDispatchStrategy()
        
        inputs = DispatchInput(
            minute=0,
            P_offer=100.0,
            P_future_offer=100.0,
            current_price=50.0,  # Above threshold (40 EUR/MWh)
            soec_capacity_mw=80.0,
            pem_max_power_mw=30.0,
            h2_non_rfnbo_price_eur_kg=2.0,
            pem_h2_kwh_kg=50.0
        )
        
        result = strategy.decide(inputs, DispatchState())
        
        # No spot purchase
        assert result.state_update.get('spot_purchased_mw', 0.0) == 0.0
        # All H2 is RFNBO
        assert result.state_update.get('h2_rfnbo_kg', 0) > 0
        assert result.state_update.get('h2_non_rfnbo_kg', 0) == 0.0


class TestCumulativeTracking:
    """Test cumulative H2 tracking and compliance percentage."""
    
    def test_cumulative_tracking_over_multiple_steps(self):
        """Verify cumulative counters update correctly."""
        strategy = EconomicSpotDispatchStrategy()
        
        inputs_cheap = DispatchInput(
            minute=0,
            P_offer=100.0,
            P_future_offer=100.0,
            current_price=30.0,  # Cheap
            soec_capacity_mw=80.0,
            pem_max_power_mw=30.0,
            p_grid_max_mw=10.0
        )
        
        inputs_expensive = DispatchInput(
            minute=1,
            P_offer=100.0,
            P_future_offer=100.0,
            current_price=50.0,  # Expensive
            soec_capacity_mw=80.0,
            pem_max_power_mw=30.0,
            p_grid_max_mw=10.0
        )
        
        # Step 1: Cheap spot - should have both RFNBO and non-RFNBO
        result1 = strategy.decide(inputs_cheap, DispatchState())
        h2_rfnbo_step1 = result1.state_update.get('h2_rfnbo_kg', 0)
        h2_non_rfnbo_step1 = result1.state_update.get('h2_non_rfnbo_kg', 0)
        
        # Step 2: Expensive spot - only RFNBO
        result2 = strategy.decide(inputs_expensive, DispatchState())
        h2_rfnbo_step2 = result2.state_update.get('h2_rfnbo_kg', 0)
        h2_non_rfnbo_step2 = result2.state_update.get('h2_non_rfnbo_kg', 0)
        
        # Cumulative should be sum of both steps
        assert strategy.h2_rfnbo_kg == pytest.approx(h2_rfnbo_step1 + h2_rfnbo_step2, rel=0.01)
        assert strategy.h2_non_rfnbo_kg == pytest.approx(h2_non_rfnbo_step1 + h2_non_rfnbo_step2, rel=0.01)
    
    def test_rfnbo_compliance_percentage(self):
        """Verify compliance percentage calculation."""
        strategy = EconomicSpotDispatchStrategy()
        
        # Manually set known values
        strategy.h2_rfnbo_kg = 90.0
        strategy.h2_non_rfnbo_kg = 10.0
        
        compliance = strategy.get_rfnbo_compliance_pct()
        assert compliance == pytest.approx(90.0, rel=0.01)
    
    def test_reset_counters(self):
        """Verify reset() clears all cumulative counters."""
        strategy = EconomicSpotDispatchStrategy()
        
        # Simulate some production
        strategy.h2_rfnbo_kg = 100.0
        strategy.h2_non_rfnbo_kg = 50.0
        strategy.e_rfnbo_used_mwh = 10.0
        strategy.e_spot_purchased_mwh = 5.0
        
        strategy.reset()
        
        assert strategy.h2_rfnbo_kg == 0.0
        assert strategy.h2_non_rfnbo_kg == 0.0
        assert strategy.e_rfnbo_used_mwh == 0.0
        assert strategy.e_spot_purchased_mwh == 0.0


class TestGridMaxLimit:
    """Test grid connection capacity limit."""
    
    def test_spot_limited_by_grid_max(self):
        """Verify spot purchase respects p_grid_max_mw limit."""
        strategy = EconomicSpotDispatchStrategy()
        
        inputs = DispatchInput(
            minute=0,
            P_offer=50.0,
            P_future_offer=50.0,
            current_price=20.0,  # Very cheap
            soec_capacity_mw=50.0,  # Uses all renewable
            pem_max_power_mw=100.0,  # Lots of PEM capacity available
            p_grid_max_mw=30.0,  # But grid limited to 30 MW
            h2_non_rfnbo_price_eur_kg=2.0,
            pem_h2_kwh_kg=50.0
        )
        
        result = strategy.decide(inputs, DispatchState())
        
        # PEM can take up to 100 MW, but grid is limited to 30 MW
        assert result.state_update.get('spot_purchased_mw') == 30.0
