import sys
import types
from pathlib import Path

import pandas as pd

import tools.regenerate_lcoh as regenerate_lcoh


class _VariantReport:
    def __init__(self) -> None:
        self.discount_rate = 0.08
        self.project_lifetime_years = 20
        self.discount_factor_sum = 1.0
        self.capex_total = 1.0
        self.opex_annual_total = 1.0
        self.annual_h2_total_kg = 1.0
        self.lcoh_total = 1.0
        self.lcoh_weighted_plant = 1.0
        self.annual_h2_by_pathway = {"pem": 1.0, "soec": 0.0, "atr": 0.0}
        self.capex_by_pathway = {"pem": 1.0, "soec": 0.0, "atr": 0.0}
        self.opex_by_pathway = {"pem": 1.0, "soec": 0.0, "atr": 0.0}
        self.lcoh_by_pathway = {"pem": 1.0, "soec": 0.0, "atr": 0.0}
        self.lcoh_breakdown = {"capex": 1.0, "opex": 0.0}


class _FakeReport:
    def __init__(self) -> None:
        v = _VariantReport()
        self.variant_order = ["low", "base", "high"]
        self.variants = {"low": v, "base": v, "high": v}
        self.warnings = []

    def model_dump_json(self, indent=2) -> str:
        return "{}"


def _write_minimal_fixture(sim_dir: Path) -> None:
    sim_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir = sim_dir / "history_chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "minute": [0.0, 60.0],
            "H2_pem_kg": [1.0, 1.0],
            "H2_soec_kg": [0.0, 0.0],
            "H2_atr_kg": [0.0, 0.0],
        }
    ).to_parquet(chunks_dir / "chunk_0000.parquet")
    (sim_dir / "capex_report.json").write_text("{}")
    (sim_dir / "opex_report.json").write_text("{}")


def _run_main_and_capture_inputs(monkeypatch, tmp_path: Path, extra_args: list[str]):
    sim_dir = tmp_path / "simulation_output"
    _write_minimal_fixture(sim_dir)
    out_dir = tmp_path / "out"
    captured = {}

    class _DummyCalculator:
        def generate_variants(self, inputs):
            captured["inputs"] = inputs
            return _FakeReport()

    monkeypatch.setattr(regenerate_lcoh, "_ensure_economics_imported", lambda: None)
    monkeypatch.setattr(regenerate_lcoh, "_load_json_report", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(regenerate_lcoh, "CapexReport", object())
    monkeypatch.setattr(regenerate_lcoh, "OpexReport", object())
    monkeypatch.setattr(regenerate_lcoh, "LcohInputs", lambda **kwargs: types.SimpleNamespace(**kwargs))
    monkeypatch.setattr(regenerate_lcoh, "LcohCalculator", lambda: _DummyCalculator())

    monkeypatch.setattr(
        sys,
        "argv",
        ["regenerate_lcoh.py", str(sim_dir), "--output-dir", str(out_dir), *extra_args],
    )
    regenerate_lcoh.main()
    return captured["inputs"]


def test_regenerate_lcoh_parallel_controls_defaults(monkeypatch, tmp_path):
    inputs = _run_main_and_capture_inputs(monkeypatch, tmp_path, [])
    assert inputs.history_scan_workers == 0
    assert inputs.history_scan_max_memory_mb is None
    assert inputs.history_scan_mode == "auto"


def test_regenerate_lcoh_parallel_controls_override(monkeypatch, tmp_path):
    inputs = _run_main_and_capture_inputs(
        monkeypatch,
        tmp_path,
        ["--workers", "4", "--max-memory-mb", "1024", "--history-scan-mode", "parallel"],
    )
    assert inputs.history_scan_workers == 4
    assert inputs.history_scan_max_memory_mb == 1024
    assert inputs.history_scan_mode == "parallel"
