"""
Tests for PEM Electrolyzer Component

Validates:
- Component initialization and degradation data loading
- Power-to-flow conversion accuracy
- Degradation progression over time
- Mode equivalence (polynomial vs analytical)
- Operational boundaries and edge cases
- Critical sanity checks (efficiency, physics constraints)
"""

import pytest
import numpy as np
from pathlib import Path

from h2_plant.components.electrolysis.pem_electrolyzer import PEMElectrolyzer
from h2_plant.config.constants_physics import PEMConstants
from h2_plant.core.component_registry import ComponentRegistry

CONST = PEMConstants()


class TestPEMElectrolyzerInitialization:
    """Test component instantiation and initialization."""
    
    def test_instantiation(self):
        """Test that PEM electrolyzer can be instantiated."""
        pem = PEMElectrolyzer()
        assert pem.t_op_h == 0.0
        assert pem.last_j == 0.0
        assert pem.T_op == CONST.T_default
        assert pem.P_op == CONST.P_op_default
    
    def test_initialization_with_polynomial_data(self):
        """Test initialization loads polynomial data if available."""
        registry = ComponentRegistry()
        pem = PEMElectrolyzer()
        pem.initialize(dt=1.0/60.0, registry=registry)
        
        # Should load polynomial data if file exists
        if pem.pkl_path.exists():
            assert pem.use_polynomials == True
            assert len(pem.polynomial_models_list) > 0
            print(f"Loaded {len(pem.polynomial_models_list)} polynomial models")
        else:
            assert pem.use_polynomials == False
            assert len(pem.polynomial_models_list) == 0
            print("No polynomial data - using analytical solver")
    
    def test_initialization_without_polynomial_data(self):
        """Test graceful fallback when polynomial data missing."""
        registry = ComponentRegistry()
        pem = PEMElectrolyzer()
        
        # Temporarily point to non-existent file
        pem.pkl_path = Path('/tmp/nonexistent_degradation.pkl')
        pem.initialize(dt=1.0/60.0, registry=registry)
        
        # Should fall back to analytical mode
        assert pem.use_polynomials == False
        assert pem.V_CELL_BOL_NOM > 0.0


class TestPEMPowerToFlowConversion:
    """Test power input to H2/O2/H2O flow conversion."""
    
    @pytest.fixture
    def initialized_pem(self):
        """Create and initialize PEM electrolyzer."""
        pem = PEMElectrolyzer()
        pem.initialize(dt=1.0/60.0)
        return pem
    
    def test_zero_power(self, initialized_pem):
        """Test that zero power produces zero flows."""
        m_H2, m_O2, m_H2O = initialized_pem.step_simulation(P_target_kW=0.0)
        assert m_H2 == 0.0
        assert m_O2 == 0.0
        assert m_H2O == 0.0
    
    def test_nominal_power(self, initialized_pem):
        """Test operation at nominal power."""
        # Run at nominal power (5 MW)
        m_H2, m_O2, m_H2O = initialized_pem.step_simulation(
            P_target_kW=CONST.P_nominal_sistema_kW,
            dt_h=1.0  # 1 hour
        )
        
        # Should produce significant H2
        assert m_H2 > 0.0
        # O2 should be ~8x H2 by mass (stoichiometry)
        assert 7.0 < (m_O2 / m_H2) < 9.0
        # H2O consumed should be ~9x H2 by mass
        assert 8.0 < (m_H2O / m_H2) < 10.0
        
        print(f"Nominal operation: H2={m_H2:.3f}kg/h, O2={m_O2:.3f}kg/h, H2O={m_H2O:.3f}kg/h")

    def test_partial_power(self, initialized_pem):
        """Test operation at partial load (50%)."""
        m_H2_half, _, _ = initialized_pem.step_simulation(
            P_target_kW=CONST.P_nominal_sistema_kW * 0.5,
            dt_h=1.0
        )
        m_H2_full, _, _ = initialized_pem.step_simulation(
            P_target_kW=CONST.P_nominal_sistema_kW,
            dt_h=1.0
        )
        
        # Half power should produce roughly half the H2 (not exact due to nonlinearity)
        ratio = m_H2_half / m_H2_full
        assert 0.4 < ratio < 0.6,  f"Expected ~0.5, got {ratio:.3f}"


class TestPEMDegradation:
    """Test degradation modeling over time."""
    
    def test_degradation_progression(self):
        """Test that voltage increases with operational time."""
        pem = PEMElectrolyzer()
        pem.initialize(dt=1.0/60.0)
        
        # Record degradation at BOL
        U_deg_bol = pem._calculate_U_deg(0.0)
        
        # Record degradation after 1 year
        t_1year = 8760.0  # hours
        U_deg_1year = pem._calculate_U_deg(t_1year)
        
        # Record degradation after 4 years
        t_4years = 4 * 8760.0
        U_deg_4years = pem._calculate_U_deg(t_4years)
        
        # Degradation should increase over time
        assert U_deg_bol == 0.0  # BOL has no degradation
        assert U_deg_1year >= U_deg_bol
        assert U_deg_4years > U_deg_1year
        
        print(f"Degradation: BOL={U_deg_bol:.4f}V, 1yr={U_deg_1year:.4f}V, 4yr={U_deg_4years:.4f}V")
    
    def test_power_increase_with_degradation(self):
        """Test that power consumption increases for same H2 production over time."""
        pem = PEMElectrolyzer()
        pem.initialize(dt=1.0/60.0)
        
        # Target: 100 kg H2 at BOL
        target_h2 = 100.0  # kg
        
        # Find power needed at BOL
        P_bol = 2500.0  # kW (initial guess)
        m_h2, _, _ = pem.step_simulation(P_bol, dt_h=1.0)
        P_bol_actual = P_bol * (target_h2 / m_h2)  # Scale to hit target
        
        # Simulate 4 years of operation
        pem.t_op_h = 4 * 8760.0
        
        # Same power at EOL should produce less H2 (due to degradation)
        m_h2_eol, _, _ = pem.step_simulation(P_bol_actual, dt_h=1.0)
        
        # EOL production should be less than target
        assert m_h2_eol < target_h2, "Degradation should reduce production at same power"
        
        print(f"At {P_bol_actual:.0f} kW: BOL={target_h2:.1f}kg/h, EOL={m_h2_eol:.1f}kg/h")


class TestPEMModeEquivalence:
    """Test polynomial and analytical modes produce equivalent results."""
    
    def test_mode_accuracy(self):
        """Test polynomial mode matches analytical mode within tolerance."""
        # Create two instances
        pem_poly = PEMElectrolyzer()
        pem_anal = PEMElectrolyzer()
        
        # Initialize one with polynomials, one without
        pem_poly.initialize(dt=1.0/60.0)
        
        # Force analytical mode for second instance
        pem_anal.initialize(dt=1.0/60.0)
        pem_anal.use_polynomials = False
        
        # Only proceed if polynomial data exists
        if not pem_poly.use_polynomials:
            pytest.skip("Polynomial data not available")
        
        # Test at various power levels
        test_powers = [500, 1000, 2500, 5000]  # kW
        
        errors = []
        for P_kw in test_powers:
            m_h2_poly, _, _ = pem_poly.step_simulation(P_kw, dt_h=1.0)
            m_h2_anal, _, _ = pem_anal.step_simulation(P_kw, dt_h=1.0)
            
            relative_error = abs(m_h2_poly - m_h2_anal) / m_h2_anal * 100
            errors.append(relative_error)
            
            print(f"{P_kw} kW: poly={m_h2_poly:.4f}, anal={m_h2_anal:.4f}, error={relative_error:.3f}%")
        
        # Average error should be < 0.5%
        avg_error = np.mean(errors)
        assert avg_error < 0.5, f"Average error {avg_error:.3f}% exceeds 0.5% threshold"


class TestPEMSanityChecks:
    """Critical sanity checks for physics and unit consistency."""
    
    def test_efficiency_bounds(self):
        """Test that efficiency doesn't exceed 105% (catches unit errors)."""
        pem = PEMElectrolyzer()
        pem.initialize(dt=1.0/60.0)
        
        # Test at nominal power
        P_kw = CONST.P_nominal_sistema_kW
        m_H2_kg, _, _ = pem.step_simulation(P_kw, dt_h=1.0)
        
        # Calculate theoretical max H2 from energy (using LHV)
        energy_kwh = P_kw * 1.0  # kWh for 1 hour
        theoretical_max_h2 = energy_kwh / CONST.LHVH2_kWh_kg  # kg
        
        # Actual efficiency
        efficiency = m_H2_kg / theoretical_max_h2 * 100
        
        assert 0 < efficiency < 105, \
            f"Efficiency {efficiency:.1f}% invalid! (Unit conversion error?)"
        
        print(f"Efficiency: {efficiency:.1f}% at {P_kw} kW")
        print(f"  Produced: {m_H2_kg:.3f} kg H2")
        print(f"  Theoretical max: {theoretical_max_h2:.3f} kg H2")
    
    def test_non_negative_flows(self):
        """Test that flows are always non-negative."""
        pem = PEMElectrolyzer()
        pem.initialize(dt=1.0/60.0)
        
        test_powers = [0, 100, 1000, 5000]  # kW
        
        for P_kw in test_powers:
            m_H2, m_O2, m_H2O = pem.step_simulation(P_kw, dt_h=1.0)
            
            assert m_H2 >= 0, f"Negative H2 flow at {P_kw} kW"
            assert m_O2 >= 0, f"Negative O2 flow at {P_kw} kW"
            assert m_H2O >= 0, f"Negative H2O flow at {P_kw} kW"
    
    def test_current_density_limits(self):
        """Test that current density stays within physical limits."""
        pem = PEMElectrolyzer()
        pem.initialize(dt=1.0/60.0)
        
        # Try very high power
        pem.step_simulation(P_target_kW=10000, dt_h=1.0)
        
        # Current density should not exceed limit
        assert pem.last_j <= CONST.j_lim, \
            f"Current density {pem.last_j:.2f} exceeds limit {CONST.j_lim}"


class TestPEMStateReporting:
    """Test state reporting and monitoring."""
    
    def test_get_state(self):
        """Test that get_state returns complete information."""
        pem = PEMElectrolyzer()
        pem.initialize(dt=1.0/60.0)
        
        # Run a step
        pem.step_simulation(P_target_kW=2500, dt_h=1.0)
        
        state = pem.get_state()
        
        # Check required fields
        assert 't_op_h' in state
        assert 'last_j_A_cm2' in state
        assert 'mode' in state
        assert state['mode'] in ['Polynomial', 'Analytical']
        assert 'degradation_V' in state
        assert 'T_op_C' in state
        assert 'P_op_bar' in state
        
        print(f"State: {state}")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
