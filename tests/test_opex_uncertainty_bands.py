import logging

import yaml

from h2_plant.economics.models import AACECostClass, CapexReport
from h2_plant.economics.opex_generator import OpexGenerator
from h2_plant.economics.opex_models import OpexReport


def _write_minimal_opex_config(path, fixed_cost=100.0):
    data = {
        "scenario_name": "test",
        "opex_items": [
            {
                "name": "Fixed OPEX",
                "category": "Fixed",
                "strategy": "fixed",
                "price": fixed_cost,
                "unit": "EUR/year",
            }
        ],
    }
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def _write_fci_sensitive_opex_config(path, fixed_cost=100.0, fci_factor=0.1):
    data = {
        "scenario_name": "test",
        "opex_items": [
            {
                "name": "Fixed OPEX",
                "category": "Fixed",
                "strategy": "fixed",
                "price": fixed_cost,
                "unit": "EUR/year",
            },
            {
                "name": "FCI-linked OPEX",
                "category": "Maintenance",
                "strategy": "factor",
                "base_reference": "FCI",
                "price": fci_factor,
                "unit": "fraction",
            },
        ],
    }
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def test_opex_generator_recomputes_low_high_from_capex_variants_and_exports_csv(tmp_path):
    config_path = tmp_path / "opex_config.yaml"
    _write_fci_sensitive_opex_config(config_path, fixed_cost=100.0, fci_factor=0.1)

    capex_report = CapexReport(
        generated_at="2026-01-01T00:00:00",
        overall_cost_class=AACECostClass.CLASS_3,
        total_installed_cost=1_000.0,
        total_installed_cost_low=800.0,
        total_installed_cost_high=1_200.0,
    )

    output_dir = tmp_path / "out"
    generator = OpexGenerator()
    report = generator.generate(
        config_path=str(config_path),
        capex_report=capex_report,
        history_df=None,
        output_dir=str(output_dir),
        simulation_hours=8760.0,
    )

    assert report.total_opex == 200.0
    assert report.total_opex_low == 180.0
    assert report.total_opex_high == 220.0

    csv_text = (output_dir / "opex_report.csv").read_text(encoding="utf-8")
    assert "TOTAL OPEX LOW" in csv_text
    assert "TOTAL OPEX HIGH" in csv_text


def test_opex_generator_falls_back_to_aace_without_capex_variants(tmp_path, caplog):
    config_path = tmp_path / "opex_config.yaml"
    _write_minimal_opex_config(config_path, fixed_cost=50.0)

    caplog.set_level(logging.WARNING)
    capex_report = CapexReport(
        generated_at="2026-01-01T00:00:00",
        overall_cost_class=AACECostClass.CLASS_3,
        total_installed_cost=1_000.0,
        total_installed_cost_low=0.0,
        total_installed_cost_high=0.0,
    )

    generator = OpexGenerator()
    report = generator.generate(
        config_path=str(config_path),
        capex_report=capex_report,
        history_df=None,
        output_dir=None,
        simulation_hours=8760.0,
    )

    assert report.total_opex == 50.0
    assert report.total_opex_low == 40.0
    assert report.total_opex_high == 60.0
    assert "falling back to AACE-based" in caplog.text


def test_opex_generator_omits_uncertainty_without_capex_and_logs_warning(tmp_path, caplog):
    config_path = tmp_path / "opex_config.yaml"
    _write_minimal_opex_config(config_path, fixed_cost=50.0)

    caplog.set_level(logging.WARNING)
    generator = OpexGenerator()
    report = generator.generate(
        config_path=str(config_path),
        capex_report=None,
        history_df=None,
        output_dir=None,
        simulation_hours=8760.0,
    )

    assert report.total_opex == 50.0
    assert report.total_opex_low is None
    assert report.total_opex_high is None
    assert "uncertainty bands (low/high) will be omitted" in caplog.text


def test_opex_report_backward_compatibility_without_low_high_fields():
    legacy = {
        "scenario_name": "legacy",
        "total_opex": 123.0,
    }
    parsed = OpexReport.model_validate(legacy)
    assert parsed.total_opex == 123.0
    assert parsed.total_opex_low is None
    assert parsed.total_opex_high is None
