"""
Unit Tests for Dynamic Heat Transfer Correlations (numba_ops.py).

These tests verify that the JIT-compiled physics functions return correct 
values according to the Dittus-Boelter and crossflow correlations.
"""

import pytest
import numpy as np
from h2_plant.optimization import numba_ops


class TestReynoldsFluxCalculation:
    """Tests for calculate_reynolds_flux function."""

    def test_zero_flow_returns_zero(self):
        """Zero mass flow should return Re=0, not cause division by zero."""
        result = numba_ops.calculate_reynolds_flux(
            mass_flow_kg_s=0.0,
            flow_area_m2=0.01,
            d_hydraulic=0.018,
            visc_pa_s=9e-6
        )
        assert result == 0.0

    def test_zero_area_returns_zero(self):
        """Zero flow area should return Re=0, not cause division by zero."""
        result = numba_ops.calculate_reynolds_flux(
            mass_flow_kg_s=1.0,
            flow_area_m2=0.0,
            d_hydraulic=0.018,
            visc_pa_s=9e-6
        )
        assert result == 0.0

    def test_zero_viscosity_returns_zero(self):
        """Zero viscosity should return Re=0, not cause division by zero."""
        result = numba_ops.calculate_reynolds_flux(
            mass_flow_kg_s=1.0,
            flow_area_m2=0.01,
            d_hydraulic=0.018,
            visc_pa_s=0.0
        )
        assert result == 0.0

    def test_standard_case_matches_manual_calculation(self):
        """
        Standard case: Re = (m_dot / A) * D / mu
        m_dot=1.0 kg/s, A=0.01 m², D=0.018 m, mu=9e-6 Pa·s
        Expected: (1.0/0.01) * 0.018 / 9e-6 = 100 * 0.018 / 9e-6 = 200,000
        """
        result = numba_ops.calculate_reynolds_flux(
            mass_flow_kg_s=1.0,
            flow_area_m2=0.01,
            d_hydraulic=0.018,
            visc_pa_s=9e-6
        )
        expected = (1.0 / 0.01) * 0.018 / 9e-6  # = 200,000
        assert abs(result - expected) < 1.0  # Allow small float tolerance

    def test_reynolds_scales_with_mass_flow(self):
        """Doubling mass flow should double Reynolds number."""
        re_1 = numba_ops.calculate_reynolds_flux(1.0, 0.01, 0.018, 9e-6)
        re_2 = numba_ops.calculate_reynolds_flux(2.0, 0.01, 0.018, 9e-6)
        assert abs(re_2 / re_1 - 2.0) < 0.001


class TestNusseltDittusBoelter:
    """Tests for calculate_nusselt_dittus_boelter function."""

    def test_laminar_flow_returns_constant(self):
        """For Re < 2300, Nu should return constant 3.66."""
        nu_lam_1 = numba_ops.calculate_nusselt_dittus_boelter(1000, 0.7, True)
        nu_lam_2 = numba_ops.calculate_nusselt_dittus_boelter(2000, 0.7, False)
        assert nu_lam_1 == 3.66
        assert nu_lam_2 == 3.66

    def test_transition_regime(self):
        """At Re=2300 exactly, should use turbulent correlation (conservative)."""
        nu = numba_ops.calculate_nusselt_dittus_boelter(2300, 0.7, True)
        # Re=2300 triggers turbulent path in our implementation
        # Nu = 0.023 * 2300^0.8 * 0.7^0.4 ≈ 9.75
        assert nu > 3.66  # Should be turbulent, not laminar

    def test_turbulent_regime(self):
        """For Re > 2300, Nu should follow Dittus-Boelter."""
        re = 50000
        pr = 0.7
        # Nu = 0.023 * Re^0.8 * Pr^0.3 (cooling)
        expected_cooling = 0.023 * (re ** 0.8) * (pr ** 0.3)
        result_cooling = numba_ops.calculate_nusselt_dittus_boelter(re, pr, False)
        assert abs(result_cooling - expected_cooling) < 0.01

    def test_heating_vs_cooling_exponent(self):
        """Heating (n=0.4) should give higher Nu than cooling (n=0.3)."""
        re = 50000
        pr = 0.7
        nu_heating = numba_ops.calculate_nusselt_dittus_boelter(re, pr, True)
        nu_cooling = numba_ops.calculate_nusselt_dittus_boelter(re, pr, False)
        
        # With Pr = 0.7:
        # Heating: Pr^0.4 = 0.7^0.4 ≈ 0.859
        # Cooling: Pr^0.3 = 0.7^0.3 ≈ 0.894
        # So cooling actually gives HIGHER Nu for Pr < 1
        # For Pr > 1 (like glycol), heating would give higher Nu
        
        # For gases (Pr < 1), cooling exponent gives higher Nu
        assert nu_cooling > nu_heating  # Valid for Pr < 1

    def test_high_prandtl_heating_vs_cooling(self):
        """For high Pr fluids (glycol), heating gives higher Nu than cooling."""
        re = 10000
        pr = 28.0  # Glycol
        nu_heating = numba_ops.calculate_nusselt_dittus_boelter(re, pr, True)
        nu_cooling = numba_ops.calculate_nusselt_dittus_boelter(re, pr, False)
        
        # For Pr > 1: Pr^0.4 > Pr^0.3, so heating > cooling
        assert nu_heating > nu_cooling


class TestNusseltCrossflow:
    """Tests for calculate_nusselt_crossflow function."""

    def test_very_low_reynolds_returns_minimum(self):
        """For Re < 100, should return minimum Nu = 1.0."""
        assert numba_ops.calculate_nusselt_crossflow(50, 0.71) == 1.0
        assert numba_ops.calculate_nusselt_crossflow(99, 0.71) == 1.0

    def test_standard_air_crossflow(self):
        """
        Standard case: Nu = 0.27 * Re^0.63 * Pr^0.33
        Re=5000, Pr=0.71
        Expected: 0.27 * 5000^0.63 * 0.71^0.33 ≈ 51.6
        """
        result = numba_ops.calculate_nusselt_crossflow(5000, 0.71)
        expected = 0.27 * (5000 ** 0.63) * (0.71 ** 0.33)
        assert abs(result - expected) < 0.1

    def test_scales_with_reynolds(self):
        """Nu should increase with Re (power law)."""
        nu_low = numba_ops.calculate_nusselt_crossflow(1000, 0.71)
        nu_high = numba_ops.calculate_nusselt_crossflow(10000, 0.71)
        assert nu_high > nu_low


class TestDynamicUFouled:
    """Tests for calculate_dynamic_u_fouled function."""

    def test_zero_h_in_returns_zero(self):
        """Infinite convection resistance on inner side returns U=0."""
        result = numba_ops.calculate_dynamic_u_fouled(
            h_in=0.0, h_out=100.0,
            d_in=0.018, d_out=0.022,
            k_wall=50.0, r_foul_in=0.0, r_foul_out=0.0
        )
        assert result == 0.0

    def test_zero_h_out_returns_zero(self):
        """Infinite convection resistance on outer side returns U=0."""
        result = numba_ops.calculate_dynamic_u_fouled(
            h_in=1000.0, h_out=0.0,
            d_in=0.018, d_out=0.022,
            k_wall=50.0, r_foul_in=0.0, r_foul_out=0.0
        )
        assert result == 0.0

    def test_invalid_geometry_returns_zero(self):
        """d_out < d_in should return 0 (invalid geometry)."""
        result = numba_ops.calculate_dynamic_u_fouled(
            h_in=1000.0, h_out=100.0,
            d_in=0.022, d_out=0.018,  # Inverted
            k_wall=50.0, r_foul_in=0.0, r_foul_out=0.0
        )
        assert result == 0.0

    def test_fouling_reduces_u(self):
        """Adding fouling resistance should strictly reduce U."""
        u_clean = numba_ops.calculate_dynamic_u_fouled(
            h_in=1000.0, h_out=100.0,
            d_in=0.018, d_out=0.022,
            k_wall=50.0, r_foul_in=0.0, r_foul_out=0.0
        )
        u_fouled = numba_ops.calculate_dynamic_u_fouled(
            h_in=1000.0, h_out=100.0,
            d_in=0.018, d_out=0.022,
            k_wall=50.0, r_foul_in=0.0002, r_foul_out=0.0005
        )
        assert u_fouled < u_clean
        assert u_fouled > 0  # Still positive

    def test_flat_plate_geometry(self):
        """Equal d_in and d_out should work (flat plate assumption)."""
        result = numba_ops.calculate_dynamic_u_fouled(
            h_in=100.0, h_out=100.0,
            d_in=0.02, d_out=0.02,  # Equal
            k_wall=200.0, r_foul_in=0.0002, r_foul_out=0.0005
        )
        assert result > 0
        # For equal h values and equal diameters with thin wall:
        # 1/U ≈ 1/h_in + r_foul_in + r_wall + r_foul_out + 1/h_out
        # 1/U ≈ 1/100 + 0.0002 + 0.002/200 + 0.0005 + 1/100
        # 1/U ≈ 0.01 + 0.0002 + 0.00001 + 0.0005 + 0.01 = 0.02071
        # U ≈ 48.3
        assert 40 < result < 55

    def test_u_bounded_by_limiting_coefficient(self):
        """U should be less than the minimum of h_in and h_out."""
        u = numba_ops.calculate_dynamic_u_fouled(
            h_in=1000.0, h_out=50.0,  # h_out is limiting
            d_in=0.018, d_out=0.022,
            k_wall=50.0, r_foul_in=0.0, r_foul_out=0.0
        )
        # U should be less than h_out (the limiting resistance)
        assert u < 50.0

    def test_realistic_gas_glycol_exchanger(self):
        """Realistic values for gas-to-glycol exchanger."""
        # h_gas ≈ 500-2000 W/m²K, h_glycol ≈ 500-2000 W/m²K
        u = numba_ops.calculate_dynamic_u_fouled(
            h_in=800.0, h_out=600.0,
            d_in=0.018, d_out=0.022,
            k_wall=50.0, r_foul_in=0.0, r_foul_out=0.0002
        )
        # Expected U for shell & tube: 50-500 W/m²K typical
        assert 50 < u < 500


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
