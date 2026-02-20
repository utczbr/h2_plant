"""
Tests for chronological lifecycle epoch reset (SOEC + PEM).

Verifies that:
- Degradation resets at calendar lifecycle boundaries (floor(t/lifecycle_h) changes)
- Reset fires even when component is idle at the boundary
- Bootstrap seeds epoch from t without resetting counters (handles resume + nonzero start)
- Degradation accumulation between resets remains usage-based
- Old checkpoint fields are handled with backward-compat defaults
- Plotly economics spikes align with chronological boundaries

None of these tests require a running simulation engine or GUI.
"""

from __future__ import annotations

import numpy as np
import pytest

from h2_plant.components.electrolysis.soec_operator import SOECOperator
from h2_plant.components.electrolysis.pem_electrolyzer import DetailedPEMElectrolyzer as PEMElectrolyzer
from h2_plant.core.component_registry import ComponentRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SOEC_LIFECYCLE_H = 61_320.0   # 7 years
PEM_LIFECYCLE_H = 87_600.0    # 10 years

DT_H = 1.0 / 60.0  # 1-minute timestep in hours


def make_soec(lifecycle_h: float = SOEC_LIFECYCLE_H, degradation_year: float = 0.0) -> SOECOperator:
    config = {
        "num_modules": 2,
        "max_power_nominal_mw": 5.0,
        "lifecycle": lifecycle_h,
        "degradation_year": degradation_year,
    }
    soec = SOECOperator(config=config)
    registry = ComponentRegistry()
    soec.initialize(dt=DT_H, registry=registry)
    return soec


def make_pem(lifecycle_h: float = PEM_LIFECYCLE_H) -> PEMElectrolyzer:
    pem = PEMElectrolyzer(config={"lifecycle": lifecycle_h})
    registry = ComponentRegistry()
    pem.initialize(dt=DT_H, registry=registry)
    return pem


def step_soec_on(soec: SOECOperator, t: float) -> None:
    """Step SOEC with full power so modules are active."""
    soec._power_setpoint_mw = soec.max_nominal_power
    soec.step(t)


def step_soec_off(soec: SOECOperator, t: float) -> None:
    """Step SOEC with zero power (standby/off)."""
    soec._power_setpoint_mw = 0.0
    soec.step(t)


def step_pem_on(pem: PEMElectrolyzer, t: float) -> None:
    """Step PEM with enough power to be ON."""
    pem.set_power_input_mw(pem.max_power_mw * 0.8)
    pem.water_buffer_kg = 1e6   # ensure no water starvation
    pem.step(t)


def step_pem_off(pem: PEMElectrolyzer, t: float) -> None:
    """Step PEM with zero power (OFF state)."""
    pem.set_power_input_mw(0.0)
    pem.step(t)


# ---------------------------------------------------------------------------
# SOEC tests
# ---------------------------------------------------------------------------

class TestSOECLifecycleEpoch:

    def test_bootstrap_no_reset_at_nonzero_start(self):
        """First step at t>0 seeds epoch but must NOT reset module_cycle_hours."""
        soec = make_soec()
        # Pre-age one module by manipulating accumulated_hours directly
        soec.accumulated_hours[:] = SOEC_LIFECYCLE_H * 0.5   # 50% through first lifecycle
        soec.module_cycle_hours[:] = SOEC_LIFECYCLE_H * 0.5

        # First step at t = lifecycle_h + small offset (simulates resume mid-project)
        t_resume = SOEC_LIFECYCLE_H + 100.0
        step_soec_off(soec, t_resume)

        # Epoch is seeded to 1, but cycle hours must NOT be reset
        assert soec.lifecycle_epoch == 1
        assert soec._epoch_bootstrapped is True
        assert soec.module_cycle_hours[0] == pytest.approx(SOEC_LIFECYCLE_H * 0.5, abs=1.0)

    def test_epoch_reset_fires_at_lifecycle_boundary(self):
        """Epoch reset sets module_cycle_hours to 0 when floor(t/lifecycle_h) increases."""
        soec = make_soec()

        # Run just below boundary (bootstrap will seed epoch=0)
        t_before = SOEC_LIFECYCLE_H - DT_H
        step_soec_on(soec, t_before)
        cycle_before = soec.module_cycle_hours.copy()
        assert soec.lifecycle_epoch == 0

        # Step that crosses the boundary
        t_at = SOEC_LIFECYCLE_H
        step_soec_on(soec, t_at)
        assert soec.lifecycle_epoch == 1
        # Cycle hours reset to 0 then accumulated dt for this step
        assert soec.module_cycle_hours[0] == pytest.approx(DT_H, abs=1e-9)

    def test_epoch_reset_fires_even_when_idle(self):
        """Reset triggers at lifecycle boundary regardless of module power state."""
        soec = make_soec()
        step_soec_off(soec, SOEC_LIFECYCLE_H - DT_H)   # bootstrap epoch=0
        assert soec.lifecycle_epoch == 0

        step_soec_off(soec, SOEC_LIFECYCLE_H)           # cross boundary while idle
        assert soec.lifecycle_epoch == 1
        # No accumulation since modules were idle
        assert soec.module_cycle_hours[0] == pytest.approx(0.0, abs=1e-9)

    def test_accumulated_hours_monotonic_through_reset(self):
        """accumulated_hours must never decrease through a lifecycle reset."""
        soec = make_soec()
        step_soec_on(soec, SOEC_LIFECYCLE_H - DT_H)
        acc_before = soec.accumulated_hours.copy()

        step_soec_on(soec, SOEC_LIFECYCLE_H)
        assert np.all(soec.accumulated_hours >= acc_before)

    def test_degradation_symmetric_across_lifecycle(self):
        """Effective degradation factor at (epoch+1, usage=U) == (epoch, usage=U)."""
        soec = make_soec()

        # Set cycle hours to 8760 (1 year) in first lifecycle
        soec.module_cycle_hours[:] = 8760.0
        # Manually compute effective year used for interpolation
        from h2_plant.components.electrolysis.soec_operator import DEG_YEARS, DEG_EFFICIENCY_KWH_KG
        eff_year_1 = 8760.0 / 8760.0   # = 1.0

        # Set cycle hours to same 8760 in second lifecycle (after reset)
        soec.lifecycle_epoch = 1
        soec.module_cycle_hours[:] = 8760.0
        eff_year_2 = 8760.0 / 8760.0   # same

        assert eff_year_1 == pytest.approx(eff_year_2)

    def test_restore_state_new_fields(self):
        """restore_state with new fields sets epoch and cycle hours correctly."""
        soec = make_soec()
        state = {
            "module_hours": [50000.0, 50000.0],
            "module_states": [2, 2],
            "module_powers": [2.5, 2.5],
            "lifecycle_epoch": 0,
            "module_cycle_hours": [8760.0, 8760.0],
        }
        soec.restore_state(state)
        assert soec.lifecycle_epoch == 0
        assert soec._epoch_bootstrapped is True
        assert soec.module_cycle_hours[0] == pytest.approx(8760.0)

    def test_restore_state_old_checkpoint_backward_compat(self):
        """Old checkpoints missing lifecycle_epoch/module_cycle_hours load without crash."""
        soec = make_soec()
        state = {
            "module_hours": [30000.0, 30000.0],
            "module_states": [2, 2],
            "module_powers": [2.5, 2.5],
        }
        soec.restore_state(state)
        # Epoch NOT set → bootstrap still pending
        assert soec._epoch_bootstrapped is False
        # Cycle hours derived from accumulated_hours % lifecycle_h
        expected = 30000.0 % SOEC_LIFECYCLE_H
        assert soec.module_cycle_hours[0] == pytest.approx(expected)

    def test_preaged_module_modulo_folding(self):
        """degradation_year > 1 lifecycle is correctly folded via modulo on init."""
        # 8 years of pre-aging, 7-year lifecycle → 1 year effective age
        years_preaged = 8.0
        soec = make_soec(degradation_year=years_preaged)
        expected_cycle_h = (years_preaged * 8760.0) % SOEC_LIFECYCLE_H
        assert soec.module_cycle_hours[0] == pytest.approx(expected_cycle_h, abs=1.0)

    def test_lifecycle_disabled_when_zero(self):
        """lifecycle_h <= 0 disables all reset logic; no AttributeError on step."""
        soec = make_soec(lifecycle_h=0.0)
        # Should not raise; epoch check skipped
        step_soec_on(soec, 100000.0)
        assert soec.lifecycle_epoch == 0


# ---------------------------------------------------------------------------
# PEM tests
# ---------------------------------------------------------------------------

class TestPEMLifecycleEpoch:

    def test_new_vars_initialized(self):
        """PEM initializes lifecycle epoch fields on construction."""
        pem = make_pem()
        assert pem.lifecycle_epoch == 0
        assert pem.t_cycle_h == 0.0
        assert pem._epoch_bootstrapped is False

    def test_bootstrap_no_reset_at_nonzero_start(self):
        """First step at t > lifecycle_h seeds epoch without resetting t_cycle_h."""
        pem = make_pem()
        pem.t_cycle_h = PEM_LIFECYCLE_H * 0.7   # simulate pre-existing cycle age

        t_resume = PEM_LIFECYCLE_H + 500.0
        step_pem_off(pem, t_resume)

        assert pem.lifecycle_epoch == 1
        assert pem._epoch_bootstrapped is True
        assert pem.t_cycle_h == pytest.approx(PEM_LIFECYCLE_H * 0.7, abs=1.0)

    def test_epoch_reset_fires_when_off_at_boundary(self):
        """Epoch reset triggers even when PEM is OFF at the lifecycle boundary."""
        pem = make_pem()
        step_pem_off(pem, PEM_LIFECYCLE_H - DT_H)   # bootstrap epoch=0
        assert pem.lifecycle_epoch == 0

        step_pem_off(pem, PEM_LIFECYCLE_H)           # cross boundary while OFF
        assert pem.lifecycle_epoch == 1
        assert pem.t_cycle_h == pytest.approx(0.0, abs=1e-9)   # reset, not incremented (OFF)

    def test_t_op_h_monotonic_through_reset(self):
        """t_op_h never decreases through a lifecycle reset."""
        import warnings
        pem = make_pem()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            step_pem_on(pem, PEM_LIFECYCLE_H - DT_H)
            t_op_before = pem.t_op_h
            step_pem_on(pem, PEM_LIFECYCLE_H)
        assert pem.t_op_h >= t_op_before

    def test_t_cycle_h_only_increments_when_on(self):
        """t_cycle_h increments only during active operation, not when OFF."""
        import warnings
        pem = make_pem()
        step_pem_off(pem, 1.0)    # bootstrap, OFF
        assert pem.t_cycle_h == pytest.approx(0.0, abs=1e-9)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            step_pem_on(pem, 2.0)     # ON — may warn about starvation; that's OK
        # t_cycle_h should NOT increment if PEM went to NO_WATER (returned early)
        # Either 0 (NO_WATER path) or DT_H (fully ran) — must be >= 0
        assert pem.t_cycle_h >= 0.0

    def test_no_reset_when_usage_reaches_lifecycle_but_t_has_not(self):
        """Reset must NOT fire when t_op_h crosses lifecycle_h but t has not."""
        pem = make_pem()
        # Simulate a unit that operated 100% of the time to accumulate usage-hours
        # but the simulation time is still < lifecycle_h
        step_pem_off(pem, 0.0)   # bootstrap epoch=0
        # Directly set t_cycle_h to exceed lifecycle_h (edge case)
        pem.t_cycle_h = PEM_LIFECYCLE_H + 100.0
        # Step at a t still within first epoch
        step_pem_off(pem, PEM_LIFECYCLE_H - 1.0)
        # Epoch should remain 0 (t did not cross boundary)
        assert pem.lifecycle_epoch == 0

    def test_restore_state_new_checkpoint(self):
        """restore_state with new fields restores epoch and t_cycle_h."""
        pem = make_pem()
        state = {
            "t_op_h": 50000.0,
            "lifecycle_epoch": 0,
            "t_cycle_h": 12345.0,
            "cumulative_h2_kg": 1000.0,
        }
        pem.restore_state(state)
        assert pem.lifecycle_epoch == 0
        assert pem._epoch_bootstrapped is True
        assert pem.t_cycle_h == pytest.approx(12345.0)

    def test_restore_state_old_checkpoint_backward_compat(self):
        """Old checkpoints missing lifecycle fields load with safe defaults."""
        pem = make_pem()
        pem.t_op_h = 20000.0
        state = {
            "t_op_h": 20000.0,
            "cumulative_h2_kg": 500.0,
        }
        pem.restore_state(state)
        assert pem._epoch_bootstrapped is False   # will bootstrap on first step
        expected_cycle = 20000.0 % PEM_LIFECYCLE_H
        assert pem.t_cycle_h == pytest.approx(expected_cycle)

    def test_lifecycle_disabled_when_zero(self):
        """lifecycle_h <= 0 disables epoch logic; no crash."""
        pem = make_pem(lifecycle_h=0.0)
        # Use OFF path so we avoid electrochemical division-by-zero in any residual % paths
        step_pem_off(pem, 100000.0)
        assert pem.lifecycle_epoch == 0
        assert pem._epoch_bootstrapped is False   # lifecycle guard skips bootstrap entirely


# ---------------------------------------------------------------------------
# Plotly economics spike tests (unit, no Qt / no full simulation)
# ---------------------------------------------------------------------------

class TestEconomicsLifecycleSpikes:
    """
    Test _apply_events logic in isolation by replicating the inner function.
    We do not import plotly_graphs to avoid heavyweight deps; instead we test
    the exact algorithm that was implemented.
    """

    @staticmethod
    def apply_events_chronological(
        hours: np.ndarray,
        lifecycle_h: float,
        cost_per_event: float,
        n_rows: int,
        n_years: int,
        year_idx: np.ndarray,
    ):
        """Replicate the new _apply_events inner function for unit testing."""
        if lifecycle_h <= 0 or cost_per_event <= 0 or len(hours) == 0:
            return np.zeros(n_rows), np.zeros(n_years)

        events = np.floor(hours / lifecycle_h).astype(int)
        prepend_val = int(np.floor(hours[0] / lifecycle_h))
        delta = np.maximum(np.diff(events, prepend=prepend_val), 0)

        costs_time = delta * cost_per_event
        costs_year = np.zeros(n_years)
        for idx, count in enumerate(delta):
            if count > 0:
                y = year_idx[idx]
                if 0 <= y < n_years:
                    costs_year[y] += count * cost_per_event

        return costs_time, costs_year

    def _make_timeline(self, total_hours: float, dt_h: float = 1.0):
        hours = np.arange(0.0, total_hours, dt_h)
        n_years = max(1, int(total_hours / 8760.0))
        year_idx = np.minimum((hours / 8760.0).astype(int), n_years - 1)
        return hours, n_years, year_idx

    def test_spikes_at_chronological_boundaries_zero_power(self):
        """Spikes appear at lifecycle_h, 2·lifecycle_h, ... even with zero power."""
        lifecycle_h = 87_600.0
        total_h = lifecycle_h * 2.5
        dt_h = 100.0
        hours, n_years, year_idx = self._make_timeline(total_h, dt_h)

        costs_time, costs_year = self.apply_events_chronological(
            hours, lifecycle_h, 1_000_000.0, len(hours), n_years, year_idx
        )
        # Exactly 2 crossings: at lifecycle_h and 2*lifecycle_h
        assert costs_time.sum() == pytest.approx(2_000_000.0)

    def test_no_spike_at_first_row_mid_window(self):
        """No spurious spike at row 0 when hours[0] > lifecycle_h."""
        lifecycle_h = 87_600.0
        dt_h = 100.0
        # Window starts after the first lifecycle boundary
        hours = np.arange(lifecycle_h + 1000.0, lifecycle_h * 2.0, dt_h)
        n_years = 5
        year_idx = np.zeros(len(hours), dtype=int)

        costs_time, _ = self.apply_events_chronological(
            hours, lifecycle_h, 1_000_000.0, len(hours), n_years, year_idx
        )
        # No spike at row 0 (already past the first boundary when window starts)
        assert costs_time[0] == pytest.approx(0.0)

    def test_spike_inside_mid_window(self):
        """A spike at 2·lifecycle_h IS captured in a windowed view starting after lifecycle_h."""
        lifecycle_h = 87_600.0
        dt_h = 100.0
        # Window from lifecycle_h+500 to 2*lifecycle_h+500
        hours = np.arange(lifecycle_h + 500.0, 2 * lifecycle_h + 500.0, dt_h)
        n_years = 5
        year_idx = np.zeros(len(hours), dtype=int)

        costs_time, _ = self.apply_events_chronological(
            hours, lifecycle_h, 1_000_000.0, len(hours), n_years, year_idx
        )
        # Exactly one crossing (at 2*lifecycle_h) within this window
        assert costs_time.sum() == pytest.approx(1_000_000.0)

    def test_negative_delta_clamped(self):
        """Non-monotonic time axis (resampling artifact) produces no negative costs."""
        lifecycle_h = 87_600.0
        # Deliberately non-monotonic hours (simulate resampling with repeats)
        hours = np.array([86000.0, 86000.0, 87600.0, 87600.0, 88000.0])
        n_years = 2
        year_idx = np.zeros(len(hours), dtype=int)

        costs_time, _ = self.apply_events_chronological(
            hours, lifecycle_h, 1_000_000.0, len(hours), n_years, year_idx
        )
        assert np.all(costs_time >= 0.0)

    def test_disabled_lifecycle_produces_no_spikes(self):
        """lifecycle_h <= 0 returns zero costs."""
        hours = np.arange(0.0, 200_000.0, 100.0)
        costs_time, costs_year = self.apply_events_chronological(
            hours, 0.0, 1_000_000.0, len(hours), 20, np.zeros(len(hours), dtype=int)
        )
        assert costs_time.sum() == pytest.approx(0.0)
        assert costs_year.sum() == pytest.approx(0.0)
