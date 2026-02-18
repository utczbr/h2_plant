from pathlib import Path

import yaml

from h2_plant.gui.core.economics_editor import (
    apply_general_econ_info,
    extract_general_econ_info,
    validate_capex_yaml_text,
    validate_opex_yaml_text,
)
from h2_plant.gui.core.scenario_workspace import (
    DEFAULT_ECONOMICS_FILE,
    DEFAULT_EQUIPMENT_FILE,
    DEFAULT_OPEX_FILE,
    DEFAULT_PHYSICS_FILE,
    DEFAULT_SIMULATION_FILE,
    DEFAULT_TOPOLOGY_FILE,
    create_workspace_from_sources,
    resolve_manifest_file,
)


def _write_yaml(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def _build_source_tree(base: Path):
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
    _write_yaml(
        base / DEFAULT_EQUIPMENT_FILE,
        {
            "cepci": {
                "base_year": 2001,
                "base_index": 397.0,
                "current_year": 2025,
                "current_index": 797.0,
            },
            "capacity_mode": "history",
            "equipment": [
                {
                    "tag": "EQ-1",
                    "block": "General",
                    "name": "Main Compressor",
                    "topology_ids": ["comp_1"],
                    "component_type": "Compressor",
                }
            ],
        },
    )
    _write_yaml(
        base / DEFAULT_OPEX_FILE,
        {
            "scenario_name": "Test",
            "opex_items": [
                {
                    "name": "Electricity",
                    "category": "Variable",
                    "strategy": "variable",
                    "price": 0.25,
                }
            ],
        },
    )


def test_capex_opex_edits_change_workspace_copy_only(tmp_path):
    source_dir = tmp_path / "source"
    _build_source_tree(source_dir)

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

    source_capex = (source_dir / DEFAULT_EQUIPMENT_FILE).read_text(encoding="utf-8")
    source_opex = (source_dir / DEFAULT_OPEX_FILE).read_text(encoding="utf-8")

    capex_path = resolve_manifest_file(manifest, "equipment_file", DEFAULT_EQUIPMENT_FILE)
    opex_path = resolve_manifest_file(manifest, "opex_file", DEFAULT_OPEX_FILE)
    assert capex_path is not None and capex_path.exists()
    assert opex_path is not None and opex_path.exists()

    capex_data = validate_capex_yaml_text(capex_path.read_text(encoding="utf-8"))
    info = extract_general_econ_info(capex_data)
    info["current_year"] = 2026
    merged_capex = apply_general_econ_info(capex_data, info)
    merged_capex["equipment"][0]["name"] = "Edited Compressor"
    capex_path.write_text(
        yaml.safe_dump(merged_capex, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    opex_data = validate_opex_yaml_text(opex_path.read_text(encoding="utf-8"))
    opex_data["opex_items"][0]["price"] = 0.30
    opex_path.write_text(
        yaml.safe_dump(opex_data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    # Source files remain untouched.
    assert (source_dir / DEFAULT_EQUIPMENT_FILE).read_text(encoding="utf-8") == source_capex
    assert (source_dir / DEFAULT_OPEX_FILE).read_text(encoding="utf-8") == source_opex

    # Workspace copies reflect edits.
    assert "Edited Compressor" in capex_path.read_text(encoding="utf-8")
    assert "0.3" in opex_path.read_text(encoding="utf-8")
