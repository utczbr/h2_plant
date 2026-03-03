import logging

import pandas as pd
import pytest
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


def _write_variable_opex_config(path, resource_id: str, metric: str = "sum", price: float = 1.0):
    data = {
        "scenario_name": "test",
        "opex_items": [
            {
                "name": "Variable item",
                "category": "Variable",
                "strategy": "variable",
                "resource_id": resource_id,
                "metric": metric,
                "price": price,
                "unit": "kg",
            }
        ],
    }
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def test_opex_generator_raises_when_history_exists_and_configured_signal_is_missing(tmp_path):
    config_path = tmp_path / "opex_config.yaml"
    _write_variable_opex_config(config_path, resource_id="electricity_consumption_kwh_step")

    history_df = pd.DataFrame(
        {
            "minute": [0.0, 60.0],
            "unrelated_signal": [1.0, 2.0],
        }
    )

    generator = OpexGenerator()
    with pytest.raises(ValueError, match="electricity_consumption_kwh_step"):
        generator.generate(
            config_path=str(config_path),
            capex_report=None,
            history_df=history_df,
            output_dir=None,
            simulation_hours=2.0,
        )


def test_opex_generator_prefers_exact_column_name_over_substring_match(tmp_path):
    config_path = tmp_path / "opex_config.yaml"
    _write_variable_opex_config(config_path, resource_id="electricity_consumption_kwh_step")

    history_df = pd.DataFrame(
        {
            "minute": [0.0, 60.0],
            "legacy_electricity_consumption_kwh_step_backup": [100.0, 100.0],
            "electricity_consumption_kwh_step": [1.0, 1.0],
        }
    )

    report = OpexGenerator().generate(
        config_path=str(config_path),
        capex_report=None,
        history_df=history_df,
        output_dir=None,
        simulation_hours=2.0,
    )

    variable_item = report.items[0]
    # sum([1,1]) * annualization(8760/2)
    assert variable_item.annual_quantity == pytest.approx(8760.0, abs=1e-9)
    assert variable_item.annual_cost == pytest.approx(8760.0, abs=1e-9)
    assert variable_item.source == "simulation:electricity_consumption_kwh_step"


def test_streaming_parquet_annualization_uses_inferred_minute_span(tmp_path):
    config_path = tmp_path / "opex_config.yaml"
    _write_variable_opex_config(config_path, resource_id="electricity_consumption_kwh_step")

    chunks_dir = tmp_path / "history_chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "minute": [0.0, 60.0, 120.0],
            "electricity_consumption_kwh_step": [10.0, 10.0, 10.0],
            "cumulative_h2_kg": [0.0, 5.0, 10.0],
        }
    ).to_parquet(chunks_dir / "chunk_0000.parquet")

    report = OpexGenerator().generate_streaming_parquet(
        config_path=str(config_path),
        chunks_dir=chunks_dir,
        capex_report=None,
        output_dir=None,
        simulation_hours=8760.0,  # intentionally mismatched
    )

    # Inferred from minute: 3 hours -> annualization factor = 2920
    assert report.simulation_hours == pytest.approx(3.0, abs=1e-9)
    assert report.annualization_factor == pytest.approx(2920.0, abs=1e-9)
    assert report.total_variable_cost == pytest.approx(30.0 * 2920.0, abs=1e-9)


def _write_dynamic_pricing_config(path):
    data = {
        "scenario_name": "dynamic",
        "opex_items": [
            {
                "name": "Electricity Consumption",
                "category": "Variable",
                "strategy": "variable",
                "resource_id": "electricity_consumption_kwh_step",
                "price_resource_id": "ppa_price_effective_eur_mwh",
                "pathway_driver_resource_ids": {
                    "pem": "pem_electricity_consumption_kwh_step",
                    "soec": "soec_electricity_consumption_kwh_step",
                    "atr": "bop_electricity_consumption_kwh_step",
                },
                "metric": "sum",
                "price": 1.0,
                "cost_multiplier": 0.001,
                "unit": "kWh",
                "lcoh_component": "energy",
            },
            {
                "name": "Electricity Sale Credit",
                "category": "Variable",
                "strategy": "variable",
                "resource_id": "sold_energy_mwh_step",
                "price_resource_id": "spot_price",
                "metric": "sum",
                "price": 1.0,
                "cost_multiplier": -1.0,
                "unit": "MWh",
                "is_credit": True,
                "lcoh_component": "energy",
            },
        ],
    }
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def test_opex_generator_dynamic_pricing_and_sale_credit_cashflow_semantics(tmp_path):
    config_path = tmp_path / "opex_config.yaml"
    _write_dynamic_pricing_config(config_path)

    history_df = pd.DataFrame(
        {
            "minute": [0.0, 60.0],
            "electricity_consumption_kwh_step": [1000.0, 1000.0],
            "ppa_price_effective_eur_mwh": [100.0, 200.0],
            "pem_electricity_consumption_kwh_step": [600.0, 600.0],
            "soec_electricity_consumption_kwh_step": [300.0, 300.0],
            "bop_electricity_consumption_kwh_step": [100.0, 100.0],
            "sold_energy_mwh_step": [0.5, 0.25],
            "spot_price": [80.0, 120.0],
            "cumulative_h2_kg": [0.0, 1.0],
        }
    )

    report = OpexGenerator().generate(
        config_path=str(config_path),
        capex_report=None,
        history_df=history_df,
        output_dir=None,
        simulation_hours=2.0,
    )

    annualization_factor = 8760.0 / 2.0
    purchase_period_eur = (1000.0 * 100.0 + 1000.0 * 200.0) * 0.001
    purchase_annual_eur = purchase_period_eur * annualization_factor
    credit_period_eur = (0.5 * 80.0 + 0.25 * 120.0) * -1.0
    credit_annual_eur = credit_period_eur * annualization_factor

    electricity_item = next(item for item in report.items if item.name == "Electricity Consumption")
    credit_item = next(item for item in report.items if item.name == "Electricity Sale Credit")

    assert electricity_item.annual_cost == pytest.approx(round(purchase_annual_eur, 2), abs=1e-9)
    assert credit_item.annual_cost == pytest.approx(round(credit_annual_eur, 2), abs=1e-9)
    assert credit_item.is_credit is True
    assert report.total_opex == pytest.approx(round(purchase_annual_eur + credit_annual_eur, 2), abs=1e-9)
    assert report.total_credit_cost == pytest.approx(round(credit_annual_eur, 2), abs=1e-9)
    assert report.total_opex_cashflow == pytest.approx(round(purchase_annual_eur, 2), abs=1e-9)
    assert electricity_item.pathway_shares is not None
    assert electricity_item.pathway_shares["pem"] == pytest.approx(0.6, abs=1e-9)
    assert electricity_item.pathway_shares["soec"] == pytest.approx(0.3, abs=1e-9)
    assert electricity_item.pathway_shares["atr"] == pytest.approx(0.1, abs=1e-9)


def test_opex_generator_raises_when_dynamic_price_signal_is_missing(tmp_path):
    config_path = tmp_path / "opex_config.yaml"
    _write_dynamic_pricing_config(config_path)

    history_df = pd.DataFrame(
        {
            "minute": [0.0, 60.0],
            "electricity_consumption_kwh_step": [1000.0, 1000.0],
            "pem_electricity_consumption_kwh_step": [600.0, 600.0],
            "soec_electricity_consumption_kwh_step": [300.0, 300.0],
            "bop_electricity_consumption_kwh_step": [100.0, 100.0],
            "sold_energy_mwh_step": [0.5, 0.25],
            "spot_price": [80.0, 120.0],
        }
    )

    with pytest.raises(ValueError, match="ppa_price_effective_eur_mwh"):
        OpexGenerator().generate(
            config_path=str(config_path),
            capex_report=None,
            history_df=history_df,
            output_dir=None,
            simulation_hours=2.0,
        )


def test_opex_generator_raises_when_pathway_driver_signal_is_missing(tmp_path):
    config_path = tmp_path / "opex_config.yaml"
    _write_dynamic_pricing_config(config_path)

    history_df = pd.DataFrame(
        {
            "minute": [0.0, 60.0],
            "electricity_consumption_kwh_step": [1000.0, 1000.0],
            "ppa_price_effective_eur_mwh": [100.0, 200.0],
            "pem_electricity_consumption_kwh_step": [600.0, 600.0],
            "soec_electricity_consumption_kwh_step": [300.0, 300.0],
            "sold_energy_mwh_step": [0.5, 0.25],
            "spot_price": [80.0, 120.0],
        }
    )

    with pytest.raises(ValueError, match="bop_electricity_consumption_kwh_step"):
        OpexGenerator().generate(
            config_path=str(config_path),
            capex_report=None,
            history_df=history_df,
            output_dir=None,
            simulation_hours=2.0,
        )
