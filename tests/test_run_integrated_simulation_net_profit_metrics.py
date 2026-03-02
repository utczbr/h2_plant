import sys
import types
from pathlib import Path

import pandas as pd
import yaml

import h2_plant.run_integrated_simulation as runner
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


def test_generate_graphs_calls_net_profit_regeneration(monkeypatch, tmp_path):
    scenarios_dir = tmp_path / "scenarios"
    scenarios_dir.mkdir(parents=True, exist_ok=True)
    (scenarios_dir / "visualization_config.yaml").write_text(
        "visualization:\n  graphs: {}\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "simulation_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    class DummyExecutor:
        def __init__(self, _catalog, _output_dir):
            self.catalog = types.SimpleNamespace(get_enabled=lambda: [])

        def configure_from_yaml(self, _cfg):
            return None

        def load_data(self, history=None, resample_freq=None, chunks_dir=None):
            return pd.DataFrame(
                {
                    "minute": [0.0, 60.0],
                    "cumulative_h2_kg": [0.0, 5.0],
                    "cumulative_grid_revenue_eur": [0.0, 10.0],
                    "P_pem": [1.0, 1.0],
                    "P_soec_actual": [0.0, 0.0],
                    "P_sold": [0.0, 0.0],
                    "spot_price": [50.0, 55.0],
                }
            )

        def execute(self, _df, timeout_seconds=60, resample_freq=None):
            return {"dummy_graph": types.SimpleNamespace(status="success", error=None)}

    fake_regen_calls = {}

    def fake_regen_net_profit(**kwargs):
        fake_regen_calls.update(kwargs)
        return 0

    monkeypatch.setitem(
        sys.modules,
        "h2_plant.visualization.graph_catalog",
        types.SimpleNamespace(GRAPH_REGISTRY=object()),
    )
    monkeypatch.setitem(
        sys.modules,
        "h2_plant.visualization.unified_executor",
        types.SimpleNamespace(UnifiedGraphExecutor=DummyExecutor),
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.regenerate_net_profit_plotly",
        types.SimpleNamespace(regenerate_net_profit_plotly=fake_regen_net_profit),
    )

    runner.generate_graphs(
        history={"minute": [0.0, 60.0]},
        scenarios_dir=str(scenarios_dir),
        output_dir=output_dir,
    )

    assert fake_regen_calls["output_dir"] == output_dir
    assert fake_regen_calls["graphs_dir"] == output_dir / "graphs"
    assert fake_regen_calls["downsample_factor"] == 60
