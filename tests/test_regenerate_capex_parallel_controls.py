import sys
from pathlib import Path

import pytest

import tools.regenerate_capex as regen_capex


def _make_minimal_scenario_tree(tmp_path: Path) -> Path:
    scenarios_dir = tmp_path / "scenarios"
    scenarios_dir.mkdir(parents=True, exist_ok=True)

    (scenarios_dir / "physics_parameters.yaml").write_text("{}\n", encoding="utf-8")
    (scenarios_dir / "simulation_config.yaml").write_text("{}\n", encoding="utf-8")
    (scenarios_dir / "economics_parameters.yaml").write_text("{}\n", encoding="utf-8")
    (scenarios_dir / "plant_topology.yaml").write_text("scenario_name: test\n", encoding="utf-8")
    return scenarios_dir


def test_main_passes_parallel_defaults(monkeypatch, tmp_path):
    scenarios_dir = _make_minimal_scenario_tree(tmp_path)
    captured = {}

    def _fake_regenerate_capex(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(regen_capex, "regenerate_capex", _fake_regenerate_capex)
    monkeypatch.setattr(
        sys,
        "argv",
        ["regenerate_capex.py", str(scenarios_dir)],
    )

    with pytest.raises(SystemExit) as exc:
        regen_capex.main()

    assert exc.value.code == 0
    assert captured["workers"] == 0
    assert captured["max_memory_mb"] is None
    assert captured["verbose_components"] is False


def test_main_passes_parallel_overrides(monkeypatch, tmp_path):
    scenarios_dir = _make_minimal_scenario_tree(tmp_path)
    history_dir = tmp_path / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    captured = {}

    def _fake_regenerate_capex(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(regen_capex, "regenerate_capex", _fake_regenerate_capex)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "regenerate_capex.py",
            str(scenarios_dir),
            "--workers",
            "4",
            "--max-memory-mb",
            "2048",
            "--verbose-components",
            "--history-dir",
            str(history_dir),
        ],
    )

    with pytest.raises(SystemExit) as exc:
        regen_capex.main()

    assert exc.value.code == 0
    assert captured["workers"] == 4
    assert captured["max_memory_mb"] == 2048
    assert captured["verbose_components"] is True
    assert captured["history_dir"] == history_dir.resolve()
