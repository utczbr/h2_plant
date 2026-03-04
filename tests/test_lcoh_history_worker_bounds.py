import os

import pytest

from h2_plant.economics.lcoh_calculator import LcohCalculator


def test_history_worker_resolution_defaults_and_caps(monkeypatch):
    calc = LcohCalculator()
    monkeypatch.setattr(os, "cpu_count", lambda: 16)

    assert calc._resolve_history_scan_workers(
        requested_workers=0,
        max_memory_mb=None,
        n_chunks=8,
        scan_mode="auto",
    ) == 2
    assert calc._resolve_history_scan_workers(
        requested_workers=8,
        max_memory_mb=None,
        n_chunks=3,
        scan_mode="parallel",
    ) == 3
    assert calc._resolve_history_scan_workers(
        requested_workers=6,
        max_memory_mb=64,
        n_chunks=8,
        scan_mode="parallel",
    ) == 1
    assert calc._resolve_history_scan_workers(
        requested_workers=6,
        max_memory_mb=1024,
        n_chunks=8,
        scan_mode="serial",
    ) == 1


def test_history_worker_resolution_rejects_invalid_mode():
    calc = LcohCalculator()
    with pytest.raises(ValueError, match="Unknown history_scan_mode"):
        calc._resolve_history_scan_workers(
            requested_workers=1,
            max_memory_mb=None,
            n_chunks=2,
            scan_mode="unknown",
        )
