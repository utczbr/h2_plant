from pathlib import Path

import yaml

from h2_plant.run_integrated_simulation import _inject_replacement_metrics_for_net_profit


def _write_topology(path: Path) -> None:
    payload = {
        "nodes": [
            {"id": "pem_1", "type": "PEM", "params": {"lifecycle": 87600}},
            {"id": "soec_1", "type": "SOEC", "params": {"lifecycle": 61320}},
        ]
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_opex_config(path: Path) -> None:
    payload = {
        "opex_items": [
            {"name": "Stack Replacement Reserve (PEM)", "price": 0.015},
            {"name": "Stack Replacement Reserve (SOEC)", "price": 0.02},
        ]
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_inject_replacement_metrics_extracts_lifecycle_and_reserve_values(tmp_path):
    scenarios_dir = tmp_path / "scenarios"
    scenarios_dir.mkdir(parents=True, exist_ok=True)
    _write_topology(scenarios_dir / "plant_topology.yaml")
    _write_opex_config(scenarios_dir / "Economics" / "opex_config.yaml")

    output_dir = scenarios_dir / "simulation_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = _inject_replacement_metrics_for_net_profit(
        {},
        scenarios_dir=str(scenarios_dir),
        output_dir=output_dir,
    )

    assert metrics["pem_lifecycle_h"] == 87600.0
    assert metrics["soec_lifecycle_h"] == 61320.0
    assert metrics["pem_reserve_pct"] == 0.015
    assert metrics["soec_reserve_pct"] == 0.02


def test_inject_replacement_metrics_supports_pem_soec_topology_filename(tmp_path):
    scenarios_dir = tmp_path / "scenarios"
    scenarios_dir.mkdir(parents=True, exist_ok=True)
    _write_topology(scenarios_dir / "plant_topology_PEM+SOEC.yaml")
    _write_opex_config(scenarios_dir / "Economics" / "opex_config.yaml")

    output_dir = scenarios_dir / "simulation_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = _inject_replacement_metrics_for_net_profit(
        {"pem_lifecycle_h": 99999.0},
        scenarios_dir=str(scenarios_dir),
        output_dir=output_dir,
    )

    # Existing metrics are preserved (setdefault semantics).
    assert metrics["pem_lifecycle_h"] == 99999.0
    assert metrics["soec_lifecycle_h"] == 61320.0
    assert metrics["pem_reserve_pct"] == 0.015
    assert metrics["soec_reserve_pct"] == 0.02
