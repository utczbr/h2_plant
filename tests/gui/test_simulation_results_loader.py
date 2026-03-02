"""
Unit tests for simulation results loader helpers.
"""

from __future__ import annotations

import json
from pathlib import Path

from h2_plant.gui.core.simulation_results_loader import (
    load_capex_data,
    load_lcoh_data,
    resolve_report_path,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_resolve_report_path_uses_deterministic_priority(tmp_path: Path) -> None:
    output_dir = tmp_path / "simulation_output"
    scenarios_dir = tmp_path / "scenarios"

    _write_json(scenarios_dir / "Economics" / "capex_report.json", {"source": "scenarios"})
    _write_json(output_dir.parent / "Economics" / "capex_report.json", {"source": "parent"})
    _write_json(output_dir / "Economics" / "capex_report.json", {"source": "output_econ"})

    resolved = resolve_report_path("capex_report.json", output_dir, scenarios_dir)
    assert resolved == output_dir / "Economics" / "capex_report.json"

    _write_json(output_dir / "capex_report.json", {"source": "output"})
    resolved = resolve_report_path("capex_report.json", output_dir, scenarios_dir)
    assert resolved == output_dir / "capex_report.json"


def test_load_capex_data_missing_file_returns_missing_status(tmp_path: Path) -> None:
    output_dir = tmp_path / "simulation_output"
    result = load_capex_data(output_dir, None)
    assert result.status == "missing"
    assert "not found" in result.message.lower()
    assert result.summary_rows == []
    assert result.tables == {}


def test_load_capex_data_invalid_json_returns_error_status(tmp_path: Path) -> None:
    output_dir = tmp_path / "simulation_output"
    report_path = output_dir / "capex_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("{invalid json", encoding="utf-8")

    result = load_capex_data(output_dir, None)
    assert result.status == "error"
    assert "invalid json" in result.message.lower()
    assert result.source_path == report_path


def test_load_lcoh_data_parses_variant_report_shape(tmp_path: Path) -> None:
    output_dir = tmp_path / "simulation_output"
    payload = {
        "discount_rate": 0.08,
        "project_lifetime_years": 20,
        "discount_factor_sum": 9.8,
        "capex_total": 200.0,
        "opex_annual_total": 50.0,
        "annual_h2_total_kg": 1000.0,
        "lcoh_total": 2.3,
        "lcoh_weighted_plant": 2.2,
        "variant_order": ["low", "base", "high"],
        "warnings": ["top-level warning"],
        "variants": {
            "low": {
                "capex_total": 100.0,
                "opex_annual_total": 30.0,
                "annual_h2_total_kg": 1000.0,
                "lcoh_total": 1.8,
                "lcoh_weighted_plant": 1.7,
                "annual_h2_by_pathway": {"pem": 500.0},
                "capex_by_pathway": {"pem": 70.0},
                "opex_by_pathway": {"pem": 30.0},
                "lcoh_by_pathway": {"pem": 1.8},
                "lcoh_breakdown": {"capex": 1.0, "opex": 0.8},
                "warnings": ["variant warning"],
            },
            "base": {
                "capex_total": 200.0,
                "opex_annual_total": 50.0,
                "annual_h2_total_kg": 1000.0,
                "lcoh_total": 2.3,
                "lcoh_weighted_plant": 2.2,
                "annual_h2_by_pathway": {"pem": 500.0},
                "capex_by_pathway": {"pem": 120.0},
                "opex_by_pathway": {"pem": 50.0},
                "lcoh_by_pathway": {"pem": 2.3},
                "lcoh_breakdown": {"capex": 1.5, "opex": 0.8},
                "warnings": [],
            },
            "high": {
                "capex_total": 300.0,
                "opex_annual_total": 70.0,
                "annual_h2_total_kg": 1000.0,
                "lcoh_total": 3.0,
                "lcoh_weighted_plant": 2.9,
                "annual_h2_by_pathway": {"pem": 500.0},
                "capex_by_pathway": {"pem": 180.0},
                "opex_by_pathway": {"pem": 70.0},
                "lcoh_by_pathway": {"pem": 3.0},
                "lcoh_breakdown": {"capex": 2.2, "opex": 0.8},
                "warnings": [],
            },
        },
    }
    _write_json(output_dir / "lcoh_report.json", payload)

    result = load_lcoh_data(output_dir, None)
    assert result.status == "ok"
    assert len(result.tables["variants"].rows) == 3
    assert result.tables["variants"].rows[0][0] == "low"
    assert len(result.tables["pathways"].rows) >= 1
    assert len(result.tables["breakdown"].rows) >= 1
    warnings = [row[0] for row in result.tables["warnings"].rows]
    assert "top-level warning" in warnings
    assert "variant warning" in warnings


def test_load_lcoh_data_parses_legacy_base_shape(tmp_path: Path) -> None:
    output_dir = tmp_path / "simulation_output"
    payload = {
        "discount_rate": 0.08,
        "project_lifetime_years": 20,
        "discount_factor_sum": 9.8,
        "capex_total": 200.0,
        "opex_annual_total": 50.0,
        "annual_h2_total_kg": 1000.0,
        "lcoh_total": 2.3,
        "lcoh_weighted_plant": 2.2,
        "annual_h2_by_pathway": {"pem": 500.0},
        "capex_by_pathway": {"pem": 120.0},
        "opex_by_pathway": {"pem": 50.0},
        "lcoh_by_pathway": {"pem": 2.3},
        "lcoh_breakdown": {"capex": 1.5, "opex": 0.8},
        "warnings": ["legacy warning"],
    }
    _write_json(output_dir / "lcoh_report.json", payload)

    result = load_lcoh_data(output_dir, None)
    assert result.status == "ok"
    assert len(result.tables["variants"].rows) == 1
    assert result.tables["variants"].rows[0][0] == "base"
    assert [row[0] for row in result.tables["warnings"].rows] == ["legacy warning"]
