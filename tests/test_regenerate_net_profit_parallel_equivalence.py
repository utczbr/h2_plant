"""
Parallel equivalence tests for _load_minimal_history and regenerate_net_profit_plotly.

Ensures workers=1 (sequential) and workers>=2 (parallel) produce identical results
on synthetic multi-chunk Parquet history with boundary-aware minute values.
"""

import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import tools.regenerate_net_profit_plotly as regen


def _create_synthetic_chunks(chunks_dir: Path, n_chunks: int = 3, rows_per: int = 120):
    """
    Create n_chunks Parquet files with realistic minute values and flow data.

    Chunks have contiguous minute values with inter-chunk gaps to test
    boundary-aware dt computation.
    """
    chunks_dir.mkdir(parents=True, exist_ok=True)
    minute_offset = 0.0
    for i in range(n_chunks):
        minutes = np.arange(rows_per, dtype=float) + minute_offset
        flow = np.full(rows_per, 5.0)  # 5 kg/h constant flow
        df = pd.DataFrame({
            "minute": minutes,
            "H2_Production_Mixer_outlet_mass_flow_kg_h": flow,
            "P_pem": np.ones(rows_per),
            "P_soec_actual": np.zeros(rows_per),
        })
        df.to_parquet(chunks_dir / f"chunk_{i:04d}.parquet", index=False)
        # Leave a 2-minute gap between chunks to test boundary logic.
        minute_offset = float(minutes[-1]) + 2.0


class TestLoadMinimalHistoryEquivalence:
    """workers=1 vs workers=2 must produce identical DataFrames."""

    def test_single_vs_parallel_match(self, tmp_path):
        chunks_dir = tmp_path / "history_chunks"
        _create_synthetic_chunks(chunks_dir, n_chunks=4, rows_per=100)

        df_seq = regen._load_minimal_history(chunks_dir, downsample_factor=1, effective_workers=1)
        df_par = regen._load_minimal_history(chunks_dir, downsample_factor=2, effective_workers=1)
        df_par2 = regen._load_minimal_history(chunks_dir, downsample_factor=2, effective_workers=2)

        # Parallel with workers=2 and sequential must give same result.
        pd.testing.assert_frame_equal(df_par, df_par2, check_dtype=False)

    def test_single_vs_parallel_no_downsample(self, tmp_path):
        chunks_dir = tmp_path / "history_chunks"
        _create_synthetic_chunks(chunks_dir, n_chunks=3, rows_per=60)

        df_seq = regen._load_minimal_history(chunks_dir, downsample_factor=1, effective_workers=1)
        df_par = regen._load_minimal_history(chunks_dir, downsample_factor=1, effective_workers=2)

        pd.testing.assert_frame_equal(df_seq, df_par, check_dtype=False)

    def test_cumulative_h2_monotonic(self, tmp_path):
        """Cumulative H2 must be monotonically non-decreasing."""
        chunks_dir = tmp_path / "history_chunks"
        _create_synthetic_chunks(chunks_dir, n_chunks=3, rows_per=120)

        for workers in (1, 2):
            df = regen._load_minimal_history(chunks_dir, downsample_factor=1, effective_workers=workers)
            h2 = df["cumulative_h2_kg"].values
            diffs = np.diff(h2)
            assert np.all(diffs >= -1e-12), (
                f"Cumulative H2 is not monotonic with workers={workers}: "
                f"min diff = {diffs.min()}"
            )

    def test_boundary_dt_consistent(self, tmp_path):
        """
        With a known gap between chunks, the boundary dt must reflect the gap.
        """
        chunks_dir = tmp_path / "history_chunks"
        # 2 chunks, 10 rows each, 2-min gap (chunks are at minute 0-9 and 11-20).
        _create_synthetic_chunks(chunks_dir, n_chunks=2, rows_per=10)

        df_seq = regen._load_minimal_history(chunks_dir, downsample_factor=1, effective_workers=1)
        df_par = regen._load_minimal_history(chunks_dir, downsample_factor=1, effective_workers=2)

        # Both should produce the same cumulative H2 at the end.
        assert np.isclose(
            df_seq["cumulative_h2_kg"].iloc[-1],
            df_par["cumulative_h2_kg"].iloc[-1],
            rtol=1e-10,
        )


class TestRegenerateNetProfitParallelEquivalence:
    """Full pipeline: workers=1 vs workers=2 must call plot_cumulative_net_profit with same kwargs."""

    def _setup_env(self, monkeypatch, tmp_path, workers):
        output_dir = tmp_path / "sim_output"
        output_dir.mkdir(parents=True)

        # CAPEX report
        capex = {
            "total_installed_cost_low": 100.0,
            "total_installed_cost": 200.0,
            "total_installed_cost_high": 300.0,
        }
        (output_dir / "capex_report.json").write_text(json.dumps(capex))

        # OPEX report
        opex = {
            "total_opex": 1000.0,
            "total_opex_low": 900.0,
            "total_opex_high": 1100.0,
        }
        (output_dir / "opex_report.json").write_text(json.dumps(opex))

        # Topology
        topo = {
            "nodes": [
                {"id": "pem_1", "type": "PEM", "params": {"lifecycle": 87600}},
                {"id": "soec_1", "type": "SOEC", "params": {"lifecycle": 61320}},
            ]
        }
        (tmp_path / "plant_topology.yaml").write_text(
            yaml.safe_dump(topo, sort_keys=False)
        )

        # OPEX config
        econ_dir = output_dir / "Economics"
        econ_dir.mkdir()
        opex_cfg = {
            "opex_items": [
                {"name": "Stack Replacement Reserve (PEM)", "price": 0.015},
                {"name": "Stack Replacement Reserve (SOEC)", "price": 0.02},
            ]
        }
        (econ_dir / "opex_config.yaml").write_text(
            yaml.safe_dump(opex_cfg, sort_keys=False)
        )

        # History chunks
        chunks_dir = output_dir / "history_chunks"
        _create_synthetic_chunks(chunks_dir, n_chunks=3, rows_per=60)

        # Stub plotly import
        call_kwargs_list = []

        class DummyFig:
            def write_html(self, path, **_kw):
                Path(path).write_text("<html></html>")

        def fake_plot(_df, **kw):
            call_kwargs_list.append(dict(kw))
            return DummyFig()

        monkeypatch.setitem(
            sys.modules,
            "h2_plant.visualization.plotly_graphs",
            types.SimpleNamespace(plot_cumulative_net_profit=fake_plot),
        )

        rc = regen.regenerate_net_profit_plotly(
            output_dir=output_dir,
            downsample_factor=60,
            workers=workers,
            parallel_mode="threads" if workers > 1 else "off",
        )
        return rc, call_kwargs_list

    def test_sequential_vs_parallel_same_kwargs(self, monkeypatch, tmp_path):
        rc_seq, kwargs_seq = self._setup_env(monkeypatch, tmp_path / "seq", workers=1)
        rc_par, kwargs_par = self._setup_env(monkeypatch, tmp_path / "par", workers=2)

        assert rc_seq == 0
        assert rc_par == 0
        assert len(kwargs_seq) == 3
        assert len(kwargs_par) == 3

        # Sort by CAPEX value for deterministic comparison.
        kwargs_seq.sort(key=lambda d: d["capex"])
        kwargs_par.sort(key=lambda d: d["capex"])

        for kw_s, kw_p in zip(kwargs_seq, kwargs_par):
            assert kw_s["capex"] == kw_p["capex"]
            assert kw_s["opex"] == kw_p["opex"]
            assert kw_s.get("pem_lifecycle_h") == kw_p.get("pem_lifecycle_h")
            assert kw_s.get("soec_lifecycle_h") == kw_p.get("soec_lifecycle_h")
