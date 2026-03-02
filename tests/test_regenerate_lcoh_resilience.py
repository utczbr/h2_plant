import logging
import sys

import pandas as pd
import pytest

import h2_plant.economics.lcoh_calculator as lcoh_module
import tools.regenerate_lcoh as regenerate_lcoh
from h2_plant.economics.lcoh_calculator import LcohCalculator


def _write_minimal_chunk(path):
    df = pd.DataFrame(
        {
            "minute": [0.0, 1.0, 2.0],
            "H2_pem_kg": [1.0, 1.0, 1.0],
            "H2_soec_kg": [2.0, 2.0, 2.0],
            "H2_atr_kg": [3.0, 3.0, 3.0],
        }
    )
    df.to_parquet(path)


def test_lcoh_calculator_handles_errno_107_during_schema_read(tmp_path, monkeypatch):
    chunks_dir = tmp_path / "history_chunks"
    chunks_dir.mkdir()
    _write_minimal_chunk(chunks_dir / "chunk_1.parquet")

    import pyarrow.parquet as pq

    def _raise_errno_107(*_args, **_kwargs):
        raise OSError(107, "Transport endpoint is not connected")

    monkeypatch.setattr(pq, "read_schema", _raise_errno_107)

    calc = LcohCalculator()
    with pytest.raises(ValueError, match="disconnected mount"):
        calc._load_h2_totals(chunks_dir)


def test_lcoh_calculator_handles_errno_107_during_chunk_read(tmp_path, monkeypatch):
    chunks_dir = tmp_path / "history_chunks"
    chunks_dir.mkdir()
    _write_minimal_chunk(chunks_dir / "chunk_1.parquet")

    def _raise_errno_107(*_args, **_kwargs):
        raise OSError(107, "Transport endpoint is not connected")

    monkeypatch.setattr(lcoh_module.pd, "read_parquet", _raise_errno_107)

    calc = LcohCalculator()
    with pytest.raises(ValueError, match="disconnected mount"):
        calc._load_h2_totals(chunks_dir)


def test_resolve_history_chunks_supports_parent_or_direct_path(tmp_path):
    sim_dir = tmp_path / "simulation_output"
    direct_chunks = sim_dir / "history_chunks"
    direct_chunks.mkdir(parents=True)
    (direct_chunks / "chunk_1.parquet").touch()

    alt_parent = tmp_path / "alt_source"
    alt_chunks = alt_parent / "history_chunks"
    alt_chunks.mkdir(parents=True)
    (alt_chunks / "chunk_1.parquet").touch()

    resolved_default = regenerate_lcoh._resolve_history_chunks(sim_dir, None)
    resolved_parent = regenerate_lcoh._resolve_history_chunks(sim_dir, str(alt_parent))
    resolved_direct = regenerate_lcoh._resolve_history_chunks(sim_dir, str(alt_chunks))

    assert resolved_default == direct_chunks
    assert resolved_parent == alt_chunks
    assert resolved_direct == alt_chunks


def test_regenerate_lcoh_main_exits_cleanly_on_value_error(tmp_path, monkeypatch, caplog):
    sim_dir = tmp_path / "simulation_output"
    hist_dir = sim_dir / "history_chunks"
    hist_dir.mkdir(parents=True)
    (hist_dir / "chunk_1.parquet").touch()

    econ_dir = sim_dir / "Economics"
    econ_dir.mkdir()
    (econ_dir / "capex_report.json").write_text("{}")
    (econ_dir / "opex_report.json").write_text("{}")

    monkeypatch.setattr(regenerate_lcoh, "_load_json_report", lambda *_args, **_kwargs: object())

    class _FailingCalculator:
        def generate_variants(self, _inputs):
            raise ValueError("History chunks are on a disconnected mount (Errno 107).")

    monkeypatch.setattr(regenerate_lcoh, "LcohCalculator", lambda: _FailingCalculator())
    monkeypatch.setattr(
        sys,
        "argv",
        ["regenerate_lcoh.py", str(sim_dir), "--project-years", "20"],
    )

    caplog.set_level(logging.ERROR)
    with pytest.raises(SystemExit) as exc:
        regenerate_lcoh.main()

    assert exc.value.code == 1
    assert "Failed to generate LCOH report" in caplog.text
