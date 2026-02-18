"""
Tests for scenario workspace staging helpers.
"""

from pathlib import Path

import pytest
import yaml

from h2_plant.gui.core.scenario_workspace import (
    DEFAULT_ECONOMICS_FILE,
    DEFAULT_EQUIPMENT_FILE,
    DEFAULT_OPEX_FILE,
    DEFAULT_PHYSICS_FILE,
    DEFAULT_SIMULATION_FILE,
    DEFAULT_TOPOLOGY_FILE,
    create_workspace_from_sources,
    load_yaml_preview,
    resolve_manifest_file,
)


def _write_yaml(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _build_source_tree(base: Path, include_opex: bool = True):
    _write_yaml(base / DEFAULT_TOPOLOGY_FILE, {"scenario_name": "Source", "nodes": []})
    _write_yaml(base / DEFAULT_PHYSICS_FILE, {"pem_system": {}, "soec_cluster": {}})
    _write_yaml(base / DEFAULT_ECONOMICS_FILE, {"h2_price_eur_kg": 9.6})
    _write_yaml(
        base / DEFAULT_SIMULATION_FILE,
        {
            "timestep_hours": 0.0167,
            "duration_hours": 24,
            "start_hour": 0,
            "checkpoint_interval_hours": 120,
            "energy_price_file": "../h2_plant/data/NL_Prices_2024_15min.csv",
            "wind_data_file": "../h2_plant/data/producao_horaria_turbina.csv",
            "dispatch_strategy": "ECONOMIC_SPOT",
            "storage_control_mode": "SCHMITT_TRIGGER",
        },
    )
    _write_yaml(base / DEFAULT_EQUIPMENT_FILE, {"equipment": []})
    if include_opex:
        _write_yaml(base / DEFAULT_OPEX_FILE, {"scenario_name": "Test", "opex_items": []})


def test_create_workspace_copies_expected_files(tmp_path):
    source_dir = tmp_path / "source"
    _build_source_tree(source_dir, include_opex=True)

    source_manifest = {
        "scenarios_dir": str(source_dir),
        "topology_file": DEFAULT_TOPOLOGY_FILE,
        "physics_file": DEFAULT_PHYSICS_FILE,
        "economics_file": DEFAULT_ECONOMICS_FILE,
        "simulation_config_file": DEFAULT_SIMULATION_FILE,
        "equipment_file": DEFAULT_EQUIPMENT_FILE,
        "opex_file": DEFAULT_OPEX_FILE,
    }

    manifest = create_workspace_from_sources(
        source_manifest,
        workspace_root=tmp_path / "generated",
    )

    workspace_dir = Path(manifest["scenarios_dir"])
    assert workspace_dir.exists()
    assert (workspace_dir / DEFAULT_TOPOLOGY_FILE).exists()
    assert (workspace_dir / DEFAULT_PHYSICS_FILE).exists()
    assert (workspace_dir / DEFAULT_ECONOMICS_FILE).exists()
    assert (workspace_dir / DEFAULT_SIMULATION_FILE).exists()
    assert (workspace_dir / DEFAULT_EQUIPMENT_FILE).exists()
    assert (workspace_dir / DEFAULT_OPEX_FILE).exists()
    assert manifest["source_scenarios_dir"] == str(source_dir)

    # Copy-only behavior: mutating workspace copy does not alter source file.
    (workspace_dir / DEFAULT_ECONOMICS_FILE).write_text("h2_price_eur_kg: 11.0\n", encoding="utf-8")
    original = load_yaml_preview(source_dir / DEFAULT_ECONOMICS_FILE)
    assert original["h2_price_eur_kg"] == 9.6


def test_create_workspace_allows_missing_optional_opex(tmp_path):
    source_dir = tmp_path / "source"
    _build_source_tree(source_dir, include_opex=False)

    source_manifest = {
        "scenarios_dir": str(source_dir),
        "topology_file": DEFAULT_TOPOLOGY_FILE,
        "physics_file": DEFAULT_PHYSICS_FILE,
        "economics_file": DEFAULT_ECONOMICS_FILE,
        "simulation_config_file": DEFAULT_SIMULATION_FILE,
        "equipment_file": DEFAULT_EQUIPMENT_FILE,
        "opex_file": DEFAULT_OPEX_FILE,
    }

    manifest = create_workspace_from_sources(
        source_manifest,
        workspace_root=tmp_path / "generated",
    )
    workspace_dir = Path(manifest["scenarios_dir"])

    assert (workspace_dir / DEFAULT_EQUIPMENT_FILE).exists()
    assert not (workspace_dir / DEFAULT_OPEX_FILE).exists()
    assert "opex_file" not in manifest
    assert manifest["source_scenarios_dir"] == str(source_dir)


def test_create_workspace_preserves_existing_source_scenarios_dir(tmp_path):
    source_dir = tmp_path / "source"
    _build_source_tree(source_dir, include_opex=True)
    original_source_dir = tmp_path / "original_scenarios"
    original_source_dir.mkdir(parents=True, exist_ok=True)

    source_manifest = {
        "scenarios_dir": str(source_dir),
        "source_scenarios_dir": str(original_source_dir),
        "topology_file": DEFAULT_TOPOLOGY_FILE,
        "physics_file": DEFAULT_PHYSICS_FILE,
        "economics_file": DEFAULT_ECONOMICS_FILE,
        "simulation_config_file": DEFAULT_SIMULATION_FILE,
        "equipment_file": DEFAULT_EQUIPMENT_FILE,
        "opex_file": DEFAULT_OPEX_FILE,
    }

    manifest = create_workspace_from_sources(
        source_manifest,
        workspace_root=tmp_path / "generated",
    )

    assert manifest["source_scenarios_dir"] == str(original_source_dir)


def test_resolve_manifest_file_handles_relative_and_absolute_paths(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    absolute_target = tmp_path / "abs.yaml"
    absolute_target.write_text("k: v\n", encoding="utf-8")

    manifest = {
        "scenarios_dir": str(source_dir),
        "simulation_config_file": "simulation_config.yaml",
        "physics_file": str(absolute_target),
    }

    assert resolve_manifest_file(
        manifest,
        "simulation_config_file",
    ) == (source_dir / "simulation_config.yaml")
    assert resolve_manifest_file(manifest, "physics_file") == absolute_target


def test_load_yaml_preview_requires_mapping(tmp_path):
    path = tmp_path / "invalid.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_yaml_preview(path)
