"""
Unit tests for CoolingManager service component.

Tests:
- Glycol loop temperature calculations
- Cooling water loop temperature calculations
- Load accumulation and reset
- Integration with DryCooler and Chiller
"""

import pytest
import numpy as np
from h2_plant.core.cooling_manager import CoolingManager


class TestCoolingManagerUnit:
    """Unit tests for CoolingManager standalone functionality."""
    
    def test_initialization(self):
        """Test default initialization values."""
        cm = CoolingManager()
        assert cm.glycol_supply_temp_c == 25.0
        assert cm.cw_supply_temp_c == 20.0
        assert cm.t_dry_bulb_c == 25.0
        assert cm.t_wet_bulb_c == 18.0
    
    def test_custom_initialization(self):
        """Test custom initialization parameters."""
        cm = CoolingManager(
            initial_glycol_temp_c=30.0,
            initial_cw_temp_c=25.0,
            dc_total_area_m2=3000.0
        )
        assert cm.glycol_supply_temp_c == 30.0
        assert cm.cw_supply_temp_c == 25.0
        assert cm.dc_total_area_m2 == 3000.0
    
    def test_glycol_load_registration(self):
        """Test glycol load accumulation."""
        cm = CoolingManager()
        cm.initialize(dt=1/60, registry=None)
        
        # Register multiple loads
        cm.register_glycol_load(duty_kw=50.0, flow_kg_s=2.0)
        cm.register_glycol_load(duty_kw=30.0, flow_kg_s=1.5)
        
        # Verify accumulation
        assert cm._current_step_glycol_load_kw == 80.0
        assert cm._current_step_glycol_flow_kg_s == 3.5
    
    def test_cw_load_registration(self):
        """Test cooling water load accumulation."""
        cm = CoolingManager()
        cm.initialize(dt=1/60, registry=None)
        
        cm.register_cw_load(duty_kw=100.0, flow_kg_s=5.0)
        cm.register_cw_load(duty_kw=50.0, flow_kg_s=2.0)
        
        assert cm._current_step_cw_load_kw == 150.0
        assert cm._current_step_cw_flow_kg_s == 7.0
    
    def test_step_glycol_temperature(self):
        """Test glycol temperature calculation after step."""
        cm = CoolingManager(initial_glycol_temp_c=25.0)
        cm.initialize(dt=1/60, registry=None)
        
        # Register a load
        cm.register_glycol_load(duty_kw=100.0, flow_kg_s=5.0)
        cm.step(0.0)
        
        # Glycol return temp should be higher (heat added)
        assert cm.glycol_return_temp_c > 25.0
        # Supply temp should change slightly due to inertia
        assert cm.glycol_supply_temp_c != 25.0
        # Duty should be recorded
        assert cm.glycol_duty_kw == 100.0
    
    def test_step_cw_temperature(self):
        """Test cooling water temperature calculation after step."""
        cm = CoolingManager(initial_cw_temp_c=20.0, tower_design_load_kw=1000.0)
        cm.initialize(dt=1/60, registry=None)
        
        # Register a load at 50% design capacity
        cm.register_cw_load(duty_kw=500.0, flow_kg_s=10.0)
        cm.step(0.0)
        
        # CW supply should approach wet bulb + approach
        assert cm.cw_supply_temp_c > 18.0  # Above wet bulb
        assert cm.cw_duty_kw == 500.0
    
    def test_load_reset_after_step(self):
        """Test that accumulators reset after step."""
        cm = CoolingManager()
        cm.initialize(dt=1/60, registry=None)
        
        cm.register_glycol_load(50.0, 2.0)
        cm.register_cw_load(100.0, 3.0)
        cm.step(0.0)
        
        # Accumulators should be reset
        assert cm._current_step_glycol_load_kw == 0.0
        assert cm._current_step_cw_load_kw == 0.0
    
    def test_get_state(self):
        """Test state dictionary contains expected keys."""
        cm = CoolingManager()
        cm.initialize(dt=1/60, registry=None)
        cm.register_glycol_load(50.0, 2.0)
        cm.step(0.0)
        
        state = cm.get_state()
        
        assert 'glycol_supply_temp_c' in state
        assert 'glycol_return_temp_c' in state
        assert 'glycol_duty_total_kw' in state
        assert 'cw_supply_temp_c' in state
        assert 'cw_duty_total_kw' in state
        assert 't_wet_bulb_c' in state


class TestCoolingManagerConvergence:
    """Test thermal convergence over multiple timesteps."""
    
    def test_glycol_steady_state(self):
        """Test glycol loop approaches steady state with constant load."""
        cm = CoolingManager(inertia_alpha=0.5)  # Faster response
        cm.initialize(dt=1/60, registry=None)
        
        temps = []
        for i in range(20):
            cm.register_glycol_load(duty_kw=200.0, flow_kg_s=10.0)
            cm.step(i/60)
            temps.append(cm.glycol_supply_temp_c)
        
        # Temperature should stabilize
        delta = abs(temps[-1] - temps[-2])
        assert delta < 0.5, f"Temperature not converging: delta={delta}"
    
    def test_cw_steady_state(self):
        """Test CW loop approaches steady state with constant load."""
        cm = CoolingManager(inertia_alpha=0.5, tower_design_load_kw=2000.0)
        cm.initialize(dt=1/60, registry=None)
        
        temps = []
        for i in range(20):
            cm.register_cw_load(duty_kw=1000.0, flow_kg_s=20.0)
            cm.step(i/60)
            temps.append(cm.cw_supply_temp_c)
        
        # Temperature should stabilize
        delta = abs(temps[-1] - temps[-2])
        assert delta < 0.5, f"CW temperature not converging: delta={delta}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
