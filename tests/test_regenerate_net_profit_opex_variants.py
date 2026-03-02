import json
import logging
import sys
import types
from pathlib import Path

import pandas as pd
import yaml

import tools.regenerate_net_profit_plotly as regen


def _write_capex_report(path: Path):
    payload = {
        "total_installed_cost_low": 100.0,
        "total_installed_cost": 200.0,
        "total_installed_cost_high": 300.0,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _run_regen(
    monkeypatch,
    tmp_path,
    opex_payload,
    opex_variant=None,
    topology_filename="plant_topology.yaml",
):
    output_dir = tmp_path / "simulation_output"
    output_dir.mkdir()

    _write_capex_report(output_dir / "capex_report.json")
    (output_dir / "opex_report.json").write_text(json.dumps(opex_payload), encoding="utf-8")

    topology_payload = {
        "nodes": [
            {"id": "pem_1", "type": "PEM", "params": {"lifecycle": 87600}},
            {"id": "soec_1", "type": "SOEC", "params": {"lifecycle": 61320}},
        ]
    }
    (tmp_path / topology_filename).write_text(
        yaml.safe_dump(topology_payload, sort_keys=False),
        encoding="utf-8",
    )

    economics_dir = output_dir / "Economics"
    economics_dir.mkdir(parents=True, exist_ok=True)
    opex_cfg_payload = {
        "opex_items": [
            {"name": "Stack Replacement Reserve (PEM)", "price": 0.015},
            {"name": "Stack Replacement Reserve (SOEC)", "price": 0.02},
        ]
    }
    (economics_dir / "opex_config.yaml").write_text(
        yaml.safe_dump(opex_cfg_payload, sort_keys=False),
        encoding="utf-8",
    )

    history_chunks = output_dir / "history_chunks"
    history_chunks.mkdir()

    dummy_df = pd.DataFrame(
        {
            "minute": [0.0, 60.0],
            "cumulative_h2_kg": [0.0, 10.0],
            "cumulative_grid_revenue_eur": [0.0, 200.0],
            "P_pem": [1.0, 1.0],
            "P_soec_actual": [0.0, 0.0],
        }
    )

    monkeypatch.setattr(regen, "_resolve_history_chunks", lambda *_args, **_kwargs: history_chunks)
    monkeypatch.setattr(regen, "_load_minimal_history", lambda *_args, **_kwargs: dummy_df)

    written_names = []
    used_opex = []
    used_capex = []
    call_kwargs = []

    class DummyFig:
        def write_html(self, path, **_kwargs):
            out = Path(path)
            out.write_text("<html></html>", encoding="utf-8")
            written_names.append(out.name)

    def fake_plot_cumulative_net_profit(_df, **_kwargs):
        used_opex.append(_kwargs.get("opex"))
        used_capex.append(_kwargs.get("capex"))
        call_kwargs.append(dict(_kwargs))
        return DummyFig()

    monkeypatch.setitem(
        sys.modules,
        "h2_plant.visualization.plotly_graphs",
        types.SimpleNamespace(plot_cumulative_net_profit=fake_plot_cumulative_net_profit),
    )

    regen_kwargs = dict(
        output_dir=output_dir,
        downsample_factor=60,
    )
    if opex_variant is not None:
        regen_kwargs["opex_variant"] = opex_variant
    rc = regen.regenerate_net_profit_plotly(**regen_kwargs)
    return rc, output_dir / "graphs", written_names, used_opex, used_capex, call_kwargs


def test_regenerate_net_profit_default_pairs_capex_and_opex_variants(monkeypatch, tmp_path):
    rc, graphs_dir, names, opex_values, capex_values, _calls = _run_regen(
        monkeypatch,
        tmp_path,
        opex_payload={
            "total_opex": 1000.0,
            "total_opex_low": 900.0,
            "total_opex_high": 1100.0,
        },
    )
    assert rc == 0
    assert sorted(names) == sorted(
        [
            "Cumulative_Net_Profit_(Interactive)_Capex_Low_Opex_Low.html",
            "Cumulative_Net_Profit_(Interactive)_Capex_Base_Opex_Base.html",
            "Cumulative_Net_Profit_(Interactive)_Capex_High_Opex_High.html",
        ]
    )
    assert opex_values == [900.0, 1000.0, 1100.0]
    assert capex_values == [100.0, 200.0, 300.0]
    assert len(list(graphs_dir.glob("*.html"))) == 3


def test_regenerate_net_profit_low_opex_variant_keeps_legacy_uniform_mode(monkeypatch, tmp_path):
    rc, graphs_dir, names, opex_values, capex_values, _calls = _run_regen(
        monkeypatch,
        tmp_path,
        opex_payload={"total_opex_low": 900.0},
        opex_variant="low",
    )
    assert rc == 0
    assert sorted(names) == sorted(
        [
            "Cumulative_Net_Profit_(Interactive)_Capex_Low_Opex_Low.html",
            "Cumulative_Net_Profit_(Interactive)_Capex_Base_Opex_Low.html",
            "Cumulative_Net_Profit_(Interactive)_Capex_High_Opex_Low.html",
        ]
    )
    assert opex_values == [900.0, 900.0, 900.0]
    assert capex_values == [100.0, 200.0, 300.0]
    assert len(list(graphs_dir.glob("*.html"))) == 3


def test_regenerate_net_profit_low_opex_missing_fails_fast(monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.ERROR)
    rc, graphs_dir, names, _opex_values, _capex_values, _calls = _run_regen(
        monkeypatch,
        tmp_path,
        opex_payload={"total_opex": 1000.0},
        opex_variant="low",
    )
    assert rc == 1
    assert names == []
    assert len(list(graphs_dir.glob("*.html"))) == 0
    assert "Requested OPEX variant 'low' not found" in caplog.text


def test_regenerate_net_profit_paired_mode_missing_variant_fails_fast(monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.ERROR)
    rc, graphs_dir, names, _opex_values, _capex_values, _calls = _run_regen(
        monkeypatch,
        tmp_path,
        opex_payload={"total_opex": 1000.0, "total_opex_low": 900.0},
        opex_variant=None,
    )
    assert rc == 1
    assert names == []
    assert len(list(graphs_dir.glob("*.html"))) == 0
    assert "Paired CAPEX/OPEX mode requires all OPEX variants" in caplog.text


def test_regenerate_net_profit_passes_replacement_inputs_to_plot_function(monkeypatch, tmp_path):
    rc, _graphs_dir, _names, _opex_values, _capex_values, calls = _run_regen(
        monkeypatch,
        tmp_path,
        opex_payload={
            "total_opex": 1000.0,
            "total_opex_low": 900.0,
            "total_opex_high": 1100.0,
        },
    )
    assert rc == 0
    assert len(calls) == 3
    for kwargs in calls:
        assert kwargs.get("pem_lifecycle_h") == 87600.0
        assert kwargs.get("soec_lifecycle_h") == 61320.0
        assert kwargs.get("pem_reserve_pct") == 0.015
        assert kwargs.get("soec_reserve_pct") == 0.02


def test_regenerate_net_profit_uses_pem_soec_topology_fallback(monkeypatch, tmp_path):
    rc, _graphs_dir, _names, _opex_values, _capex_values, calls = _run_regen(
        monkeypatch,
        tmp_path,
        opex_payload={
            "total_opex": 1000.0,
            "total_opex_low": 900.0,
            "total_opex_high": 1100.0,
        },
        topology_filename="plant_topology_PEM+SOEC.yaml",
    )
    assert rc == 0
    assert len(calls) == 3
    for kwargs in calls:
        assert kwargs.get("pem_lifecycle_h") == 87600.0
        assert kwargs.get("soec_lifecycle_h") == 61320.0
