from pathlib import Path

import pytest
import yaml

from h2_plant.economics.models import AACECostClass, CapexReport
from h2_plant.economics.opex_generator import OpexGenerator
from tools.regenerate_net_profit_plotly import _load_opex_reserves


def _write_config(path: Path, items):
    data = {
        "scenario_name": "test",
        "opex_items": items,
    }
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def test_opex_generator_does_not_overwrite_labor_base_with_laboratory_items(tmp_path):
    config_path = tmp_path / "opex_config.yaml"
    _write_config(
        config_path,
        [
            {
                "name": "Operating Labor (Turton)",
                "category": "Fixed",
                "strategy": "turton_labor",
                "turton_P": 0,
                "turton_Nnp": 75,
                "shifts": 4.8,
                "hours_per_year": 2080,
                "price": 25.0,
            },
            {
                "name": "Supervisory & Technical Staff",
                "category": "Fixed",
                "strategy": "factor",
                "base_reference": "Labor",
                "price": 0.25,
            },
            {
                "name": "Laboratory & QC",
                "category": "Fixed",
                "strategy": "factor",
                "base_reference": "Labor",
                "price": 0.10,
            },
            {
                "name": "Administration & Overhead",
                "category": "Fixed",
                "strategy": "factor",
                "base_reference": "Labor",
                "price": 0.60,
            },
        ],
    )

    report = OpexGenerator().generate(
        config_path=str(config_path),
        capex_report=None,
        history_df=None,
        output_dir=None,
        simulation_hours=8760.0,
    )

    by_name = {item.name: item.annual_cost for item in report.items}
    operating_labor = by_name["Operating Labor (Turton)"]

    assert report.labor_cost == pytest.approx(operating_labor, abs=0.01)
    assert by_name["Supervisory & Technical Staff"] == pytest.approx(operating_labor * 0.25, abs=0.01)
    assert by_name["Laboratory & QC"] == pytest.approx(operating_labor * 0.10, abs=0.01)
    assert by_name["Administration & Overhead"] == pytest.approx(operating_labor * 0.60, abs=0.01)


def test_replacement_items_keep_atr_annual_and_stack_reserves_available_for_spikes(tmp_path):
    config_path = tmp_path / "opex_config.yaml"
    _write_config(
        config_path,
        [
            {
                "name": "Catalyst Replacement (ATR)",
                "category": "Maintenance",
                "strategy": "fixed",
                "price": 2000.0,
            },
            {
                "name": "Stack Replacement Reserve (PEM)",
                "category": "Maintenance",
                "strategy": "factor",
                "base_reference": "FCI",
                "price": 0.015,
            },
            {
                "name": "Stack Replacement Reserve (SOEC)",
                "category": "Maintenance",
                "strategy": "factor",
                "base_reference": "FCI",
                "price": 0.02,
            },
        ],
    )

    capex_report = CapexReport(
        generated_at="2026-01-01T00:00:00",
        overall_cost_class=AACECostClass.CLASS_3,
        total_installed_cost=100_000.0,
        total_installed_cost_low=80_000.0,
        total_installed_cost_high=120_000.0,
    )

    report = OpexGenerator().generate(
        config_path=str(config_path),
        capex_report=capex_report,
        history_df=None,
        output_dir=None,
        simulation_hours=8760.0,
    )

    by_name = {item.name: item.annual_cost for item in report.items}
    assert by_name["Catalyst Replacement (ATR)"] == pytest.approx(2000.0, abs=0.01)
    assert by_name["Stack Replacement Reserve (PEM)"] == pytest.approx(1500.0, abs=0.01)
    assert by_name["Stack Replacement Reserve (SOEC)"] == pytest.approx(2000.0, abs=0.01)

    assert report.total_opex == pytest.approx(5500.0, abs=0.01)
    assert report.total_opex_low == pytest.approx(4800.0, abs=0.01)
    assert report.total_opex_high == pytest.approx(6200.0, abs=0.01)

    pem_pct, soec_pct = _load_opex_reserves(config_path)
    assert pem_pct == pytest.approx(0.015, abs=1e-9)
    assert soec_pct == pytest.approx(0.02, abs=1e-9)
