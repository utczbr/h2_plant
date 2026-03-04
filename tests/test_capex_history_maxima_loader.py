from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import h2_plant.economics.capex_generator as capex_module
from h2_plant.economics.capex_generator import CapexGenerator
from h2_plant.economics.models import EquipmentMapping


class DummyRegistry:
    def __init__(self, components):
        self._components = components

    def has(self, component_id: str) -> bool:
        return component_id in self._components

    def get(self, component_id: str):
        return self._components[component_id]


def _make_mapping(topology_id: str, capacity_mode: str | None = None) -> EquipmentMapping:
    return EquipmentMapping(
        tag="TST-001",
        name="Test Equipment",
        topology_ids=[topology_id],
        component_type="Centrifugal Pump",
        capacity_variable="power_kw",
        capacity_unit="kW",
        capacity_aggregation="sum",
        capacity_mode=capacity_mode,
        cost_source="fixed",
        vendor_quote_eur=1_000.0,
    )


def test_generate_keeps_history_loading_lazy_when_not_needed(monkeypatch, tmp_path):
    generator = CapexGenerator()
    generator.capacity_mode = "design"
    generator.mappings = [_make_mapping("COMP_A")]

    def _should_not_run(*_args, **_kwargs):
        raise AssertionError("history maxima loader should remain lazy")

    monkeypatch.setattr(generator, "_load_history_maxima", _should_not_run)

    registry = DummyRegistry({"COMP_A": SimpleNamespace(max_power_kw=12.0)})
    report = generator.generate(
        registry=registry,
        monitoring=None,
        output_dir=tmp_path / "out",
    )

    assert report.entries
    assert report.entries[0].C_BM == pytest.approx(1_000.0)


def test_generate_loads_history_once_when_history_mode_needs_it(monkeypatch, tmp_path):
    generator = CapexGenerator()
    generator.capacity_mode = "history"
    generator.mappings = [_make_mapping("COMP_A", capacity_mode="history")]

    calls = {"count": 0}

    def _fake_load_history_maxima(*_args, **_kwargs):
        calls["count"] += 1
        return {"COMP_A_power_kw": 33.0}

    monkeypatch.setattr(generator, "_load_history_maxima", _fake_load_history_maxima)

    registry = DummyRegistry({"COMP_A": SimpleNamespace()})
    report = generator.generate(
        registry=registry,
        monitoring=None,
        output_dir=tmp_path / "out",
    )

    assert calls["count"] == 1
    assert report.entries
    assert report.entries[0].design_capacity == pytest.approx(33.0)


def test_load_history_maxima_csv_respects_required_columns(monkeypatch, tmp_path):
    csv_path = tmp_path / "simulation_history.csv"
    pd.DataFrame(
        {
            "COMP_A_power_kw": [1.0, 5.0, 3.0],
            "COMP_B_power_kw": [200.0, 300.0, 400.0],
            "non_numeric": ["a", "b", "c"],
        }
    ).to_csv(csv_path, index=False)

    generator = CapexGenerator()

    real_read_csv = capex_module.pd.read_csv
    seen_usecols = []

    def _spy_read_csv(*args, **kwargs):
        seen_usecols.append(kwargs.get("usecols"))
        return real_read_csv(*args, **kwargs)

    monkeypatch.setattr(capex_module.pd, "read_csv", _spy_read_csv)

    maxima = generator._load_history_maxima(
        output_dir=tmp_path,
        required_columns={"COMP_A_power_kw"},
        workers=1,
    )

    assert set(maxima.keys()) == {"COMP_A_power_kw"}
    assert maxima["COMP_A_power_kw"] == pytest.approx(5.0)
    assert any(
        usecols is not None and set(usecols) == {"COMP_A_power_kw"}
        for usecols in seen_usecols
    )


def test_load_history_maxima_parquet_stats_mode_avoids_dataframe_read(monkeypatch, tmp_path):
    chunks_dir = tmp_path / "history_chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "COMP_A_power_kw": [2.0, 4.0, 10.0],
            "COMP_B_power_kw": [1.0, 1.0, 1.0],
        }
    ).to_parquet(chunks_dir / "chunk_0000.parquet", index=False)

    generator = CapexGenerator()

    def _fail_read_parquet(*_args, **_kwargs):
        raise AssertionError("pd.read_parquet should not run in stats mode")

    monkeypatch.setattr(capex_module.pd, "read_parquet", _fail_read_parquet)

    maxima = generator._load_history_maxima(
        output_dir=tmp_path,
        required_columns={"COMP_A_power_kw"},
        workers=1,
        scan_mode="stats",
    )

    assert maxima["COMP_A_power_kw"] == pytest.approx(10.0)


def test_load_history_maxima_parquet_read_mode_falls_back_to_column_scan(monkeypatch, tmp_path):
    chunks_dir = tmp_path / "history_chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "COMP_A_power_kw": [2.0, 4.0, 11.0],
            "COMP_B_power_kw": [1.0, 1.0, 1.0],
        }
    ).to_parquet(chunks_dir / "chunk_0000.parquet", index=False)

    generator = CapexGenerator()

    real_read_parquet = capex_module.pd.read_parquet
    calls = {"count": 0}

    def _spy_read_parquet(*args, **kwargs):
        calls["count"] += 1
        return real_read_parquet(*args, **kwargs)

    monkeypatch.setattr(capex_module.pd, "read_parquet", _spy_read_parquet)

    maxima = generator._load_history_maxima(
        output_dir=tmp_path,
        required_columns={"COMP_A_power_kw"},
        workers=1,
        scan_mode="read",
    )

    assert calls["count"] >= 1
    assert maxima["COMP_A_power_kw"] == pytest.approx(11.0)
