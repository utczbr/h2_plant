import json
import logging
import sys
import types
from pathlib import Path

import pandas as pd

import tools.regenerate_net_profit_plotly as regen


def _write_capex_report(path: Path):
    payload = {
        "total_installed_cost_low": 100.0,
        "total_installed_cost": 200.0,
        "total_installed_cost_high": 300.0,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _run_regen(monkeypatch, tmp_path, opex_payload, opex_variant="base"):
    output_dir = tmp_path / "simulation_output"
    output_dir.mkdir()

    _write_capex_report(output_dir / "capex_report.json")
    (output_dir / "opex_report.json").write_text(json.dumps(opex_payload), encoding="utf-8")

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

    class DummyFig:
        def write_html(self, path, **_kwargs):
            out = Path(path)
            out.write_text("<html></html>", encoding="utf-8")
            written_names.append(out.name)

    def fake_plot_cumulative_net_profit(_df, **_kwargs):
        return DummyFig()

    monkeypatch.setitem(
        sys.modules,
        "h2_plant.visualization.plotly_graphs",
        types.SimpleNamespace(plot_cumulative_net_profit=fake_plot_cumulative_net_profit),
    )

    rc = regen.regenerate_net_profit_plotly(
        output_dir=output_dir,
        downsample_factor=60,
        opex_variant=opex_variant,
    )
    return rc, output_dir / "graphs", written_names


def test_regenerate_net_profit_default_uses_base_opex_and_legacy_filenames(monkeypatch, tmp_path):
    rc, graphs_dir, names = _run_regen(
        monkeypatch,
        tmp_path,
        opex_payload={"total_opex": 1000.0},
        opex_variant="base",
    )
    assert rc == 0
    assert sorted(names) == sorted(
        [
            "Cumulative_Net_Profit_(Interactive)_Capex_Low.html",
            "Cumulative_Net_Profit_(Interactive)_Capex_Base.html",
            "Cumulative_Net_Profit_(Interactive)_Capex_High.html",
        ]
    )
    assert len(list(graphs_dir.glob("*.html"))) == 3


def test_regenerate_net_profit_low_opex_variant_adds_filename_suffix(monkeypatch, tmp_path):
    rc, graphs_dir, names = _run_regen(
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
    assert len(list(graphs_dir.glob("*.html"))) == 3


def test_regenerate_net_profit_low_opex_missing_fails_fast(monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.ERROR)
    rc, graphs_dir, names = _run_regen(
        monkeypatch,
        tmp_path,
        opex_payload={"total_opex": 1000.0},
        opex_variant="low",
    )
    assert rc == 1
    assert names == []
    assert len(list(graphs_dir.glob("*.html"))) == 0
    assert "Requested OPEX variant 'low' not found" in caplog.text
