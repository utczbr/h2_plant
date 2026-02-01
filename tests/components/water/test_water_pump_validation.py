"""
Validation tests for WaterPumpThermodynamic.

These tests validate the component against the legacy water_pump_model.py
to ensure physics preservation.

Run separately from unit tests to catch any regressions in thermodynamic calculations.
"""

import pytest
import numpy as np

try:
    import CoolProp.CoolProp as CP
    COOLPROP_AVAILABLE = True
except ImportError:
    COOLPROP_AVAILABLE = False

from h2_plant.components.water.water_pump_thermodynamic import WaterPumpThermodynamic
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.stream import Stream


# Test scenarios from legacy water_pump_model.py
FORWARD_SCENARIO = {
    'P1_kPa': 101.325,
    'T1_C': 20.0,
    'P2_kPa': 500.0,
    'mass_flow_kg_s': 10.0,
    'eta_is': 0.82,
    'eta_m': 0.96,
    # Expected results from legacy
    'expected_T2_C': 20.02675,
    'expected_work_kj_kg': 0.48702,
    'expected_power_kw': 5.07310
}

REVERSE_SCENARIO = {
    'P2_kPa': 500.0,
    'T2_C': 20.05,
    'P1_kPa': 101.325,
    'mass_flow_kg_s': 10.0,
    'eta_is': 0.82,
    'eta_m': 0.96,
    # Expected results from legacy
    'expected_T1_C': 20.02325,
    'expected_work_kj_kg': 0.48698,
    'expected_power_kw': 5.07269
}


def calculate_legacy_forward(scenario):
    """
    Reference calculation from legacy water_pump_model.py.
    Returns expected values for comparison.
    """
    if not COOLPROP_AVAILABLE:
        pytest.skip("CoolProp required for validation tests")
    
    fluido = 'Water'
    P1_Pa = scenario['P1_kPa'] * 1000.0
    T1_K = scenario['T1_C'] + 273.15
    P2_Pa = scenario['P2_kPa'] * 1000.0
    m_dot = scenario['mass_flow_kg_s']
    eta_is = scenario['eta_is']
    eta_m = scenario['eta_m']
    
    # Legacy calculations (exact code)
    h1 = CP.PropsSI('H', 'P', P1_Pa, 'T', T1_K, fluido) / 1000.0
    s1 = CP.PropsSI('S', 'P', P1_Pa, 'T', T1_K, fluido) / 1000.0
    h2s = CP.PropsSI('H', 'P', P2_Pa, 'S', s1 * 1000.0, fluido) / 1000.0
    
    work_is = h2s - h1
    work_real = work_is / eta_is
    h2 = h1 + work_real
    
    T2_K = CP.PropsSI('T', 'P', P2_Pa, 'H', h2 * 1000.0, fluido)
    T2_C = T2_K - 273.15
    
    power_fluid = m_dot * work_real
    power_shaft = power_fluid / eta_m
    
    return {
        'T_final_C': T2_C,
        'work_real_kj_kg': work_real,
        'power_shaft_kw': power_shaft
    }


def calculate_legacy_reverse(scenario):
    """Reference reverse calculation from legacy."""
    if not COOLPROP_AVAILABLE:
        pytest.skip("CoolProp required for validation tests")
    
    fluido = 'Water'
    P2_Pa = scenario['P2_kPa'] * 1000.0
    T2_K = scenario['T2_C'] + 273.15
    P1_Pa = scenario['P1_kPa'] * 1000.0
    m_dot = scenario['mass_flow_kg_s']
    eta_is = scenario['eta_is']
    eta_m = scenario['eta_m']
    
    h2 = CP.PropsSI('H', 'P', P2_Pa, 'T', T2_K, fluido) / 1000.0
    rho_2 = CP.PropsSI('D', 'P', P2_Pa, 'T', T2_K, fluido)
    v_avg = 1.0 / rho_2
    
    P_diff = P2_Pa - P1_Pa
    w_is_kj = (v_avg * P_diff) / 1000.0
    w_real_kj = w_is_kj / eta_is
    h1 = h2 - w_real_kj
    
    T1_K = CP.PropsSI('T', 'P', P1_Pa, 'H', h1 * 1000.0, fluido)
    T1_C = T1_K - 273.15
    
    power_fluid = m_dot * w_real_kj
    power_shaft = power_fluid / eta_m
    
    return {
        'T_final_C': T1_C,
        'work_real_kj_kg': w_real_kj,
        'power_shaft_kw': power_shaft
    }


@pytest.mark.skipif(not COOLPROP_AVAILABLE, reason="CoolProp required")
class TestWaterPumpValidation:
    """Validation tests against legacy model."""
    
    def test_forward_validation(self):
        """Validate forward calculation matches legacy exactly."""
        scenario = FORWARD_SCENARIO
        
        # Calculate legacy reference
        legacy = calculate_legacy_forward(scenario)
        
        # Setup component
        pump = WaterPumpThermodynamic(
            pump_id='validation_forward',
            eta_is=scenario['eta_is'],
            eta_m=scenario['eta_m'],
            target_pressure_pa=scenario['P2_kPa'] * 1000.0
        )
        
        registry = ComponentRegistry()
        pump.initialize(dt=1.0, registry=registry)
        
        # Create inlet stream
        inlet_stream = Stream(
            mass_flow_kg_h=scenario['mass_flow_kg_s'] * 3600.0,
            temperature_k=scenario['T1_C'] + 273.15,
            pressure_pa=scenario['P1_kPa'] * 1000.0,
            composition={'H2O': 1.0},
            phase='liquid'
        )
        
        pump.receive_input('water_in', inlet_stream, 'water')
        pump.step(t=0.0)
        
        # Validate results (tolerance 1e-5 for floating point)
        assert abs(pump.calculated_T_c - legacy['T_final_C']) < 1e-5
        assert abs(pump.work_real_kj_kg - legacy['work_real_kj_kg']) < 1e-5
        assert abs(pump.power_shaft_kw - legacy['power_shaft_kw']) < 1e-5
        
        print(f"\n✅ Forward Validation PASS:")
        print(f"   Temperature: {pump.calculated_T_c:.5f}°C (expected {legacy['T_final_C']:.5f})")
        print(f"   Work: {pump.work_real_kj_kg:.5f} kJ/kg (expected {legacy['work_real_kj_kg']:.5f})")
        print(f"   Power: {pump.power_shaft_kw:.5f} kW (expected {legacy['power_shaft_kw']:.5f})")
    
    def test_reverse_validation(self):
        """Validate reverse calculation matches legacy exactly."""
        scenario = REVERSE_SCENARIO
        
        # Calculate legacy reference
        legacy = calculate_legacy_reverse(scenario)
        
        # Setup component
        pump = WaterPumpThermodynamic(
            pump_id='validation_reverse',
            eta_is=scenario['eta_is'],
            eta_m=scenario['eta_m'],
            target_pressure_pa=scenario['P1_kPa'] * 1000.0  # Target is inlet in reverse
        )
        
        registry = ComponentRegistry()
        pump.initialize(dt=1.0, registry=registry)
        
        # Create outlet stream
        outlet_stream = Stream(
            mass_flow_kg_h=scenario['mass_flow_kg_s'] * 3600.0,
            temperature_k=scenario['T2_C'] + 273.15,
            pressure_pa=scenario['P2_kPa'] * 1000.0,
            composition={'H2O': 1.0},
            phase='liquid'
        )
        
        pump.receive_input('water_out_reverse', outlet_stream, 'water')
        pump.step(t=0.0)
        
        # Validate results
        assert abs(pump.calculated_T_c - legacy['T_final_C']) < 1e-5
        assert abs(pump.work_real_kj_kg - legacy['work_real_kj_kg']) < 1e-5
        assert abs(pump.power_shaft_kw - legacy['power_shaft_kw']) < 1e-5
        
        print(f"\n✅ Reverse Validation PASS:")
        print(f"   Temperature: {pump.calculated_T_c:.5f}°C (expected {legacy['T_final_C']:.5f})")
        print(f"   Work: {pump.work_real_kj_kg:.5f} kJ/kg (expected {legacy['work_real_kj_kg']:.5f})")
        print(f"   Power: {pump.power_shaft_kw:.5f} kW (expected {legacy['power_shaft_kw']:.5f})")
    
    def test_multiple_scenarios_regression(self):
        """Test multiple pressure/temperature scenarios for regression."""
        test_cases = [
            # (P1_bar, T1_C, P2_bar)
            (1.0, 15.0, 3.0),
            (1.0, 25.0, 5.0),
            (2.0, 20.0, 10.0),
            (5.0, 30.0, 15.0),
        ]
        
        for P1_bar, T1_C, P2_bar in test_cases:
            pump = WaterPumpThermodynamic(
                pump_id=f'regression_{P1_bar}_{P2_bar}',
                eta_is=0.80,
                eta_m=0.95,
                target_pressure_pa=P2_bar * 1e5
            )
            
            registry = ComponentRegistry()
            pump.initialize(dt=1.0, registry=registry)
            
            inlet_stream = Stream(
                mass_flow_kg_h=3600.0,
                temperature_k=T1_C + 273.15,
                pressure_pa=P1_bar * 1e5,
                composition={'H2O': 1.0},
                phase='liquid'
            )
            
            pump.receive_input('water_in', inlet_stream, 'water')
            pump.step(t=0.0)
            
            # Basic sanity checks
            assert pump.power_shaft_kw > 0, f"No power for case {P1_bar}->{P2_bar} bar"
            assert pump.calculated_T_c > T1_C, f"No temp rise for {P1_bar}->{P2_bar} bar"
            assert pump.outlet_stream.pressure_pa == P2_bar * 1e5


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
