from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from h2_plant.economics.opex_generator import OpexGenerator


def _write_dynamic_config(config_path: Path) -> None:
    config = {
        "scenario_name": "parallel-equivalence",
        "opex_items": [
            {
                "name": "Electricity Consumption",
                "category": "Variable",
                "strategy": "variable",
                "resource_id": "electricity_consumption_kwh_step",
                "price_resource_id": "ppa_price_effective_eur_mwh",
                "metric": "sum",
                "price": 1.0,
                "cost_multiplier": 0.001,
                "unit": "kWh",
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
            },
            {
                "name": "Cooling Water",
                "category": "Variable",
                "strategy": "variable",
                "resource_id": "water_makeup_kg_step",
                "metric": "sum",
                "price": 0.0015,
                "cost_multiplier": 1.0,
                "unit": "kg",
            },
            {
                "name": "Fixed Overhead",
                "category": "Fixed",
                "strategy": "fixed",
                "price": 1200.0,
                "unit": "EUR/year",
            },
        ],
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")


def _write_history_chunks(chunks_dir: Path, n_chunks: int = 4, rows_per_chunk: int = 8) -> None:
    chunks_dir.mkdir(parents=True, exist_ok=True)

    cumulative_h2 = 0.0
    minute = 0.0
    for idx in range(n_chunks):
        minutes = []
        electricity = []
        ppa_price = []
        sold_energy = []
        spot_price = []
        water_makeup = []
        cumulative_h2_series = []

        for step in range(rows_per_chunk):
            minutes.append(minute)
            electricity.append(800.0 + 10.0 * idx + 2.0 * step)
            ppa_price.append(70.0 + (idx % 3) * 5.0 + step * 0.5)
            sold_energy.append(0.20 + 0.01 * idx + 0.005 * step)
            spot_price.append(90.0 + (step % 4) * 4.0)
            water_makeup.append(500.0 + 15.0 * idx + 3.0 * step)
            cumulative_h2 += 4.5 + 0.2 * idx + 0.1 * step
            cumulative_h2_series.append(cumulative_h2)
            minute += 60.0

        df = pd.DataFrame(
            {
                "minute": minutes,
                "electricity_consumption_kwh_step": electricity,
                "ppa_price_effective_eur_mwh": ppa_price,
                "sold_energy_mwh_step": sold_energy,
                "spot_price": spot_price,
                "water_makeup_kg_step": water_makeup,
                "cumulative_h2_kg": cumulative_h2_series,
            }
        )
        df.to_parquet(chunks_dir / f"chunk_{idx:04d}.parquet", index=False)


def _assert_reports_equivalent(left, right) -> None:
    scalar_fields = [
        "simulation_hours",
        "annualization_factor",
        "total_variable_cost",
        "total_fixed_cost",
        "total_maintenance_cost",
        "total_opex",
        "total_credit_cost",
        "total_opex_cashflow",
        "annual_h2_production_kg",
        "opex_per_kg_h2",
    ]
    for field in scalar_fields:
        assert getattr(left, field) == pytest.approx(getattr(right, field), abs=1e-9)

    left_items = {item.name: item.annual_cost for item in left.items}
    right_items = {item.name: item.annual_cost for item in right.items}
    assert left_items.keys() == right_items.keys()
    for key in left_items:
        assert left_items[key] == pytest.approx(right_items[key], abs=1e-9)

    yearly_fields = [
        "year_index",
        "year_hours",
        "total_variable_cost_by_year",
        "total_fixed_cost_by_year",
        "total_maintenance_cost_by_year",
        "total_opex_by_year",
        "total_opex_cashflow_by_year",
        "annual_h2_production_kg_by_year",
    ]
    for field in yearly_fields:
        left_value = getattr(left, field)
        right_value = getattr(right, field)
        if left_value is None or right_value is None:
            assert left_value == right_value
            continue
        assert np.allclose(np.array(left_value, dtype=float), np.array(right_value, dtype=float), atol=1e-9)

    assert left.item_annual_cost_by_year is not None
    assert right.item_annual_cost_by_year is not None
    assert left.item_annual_cost_by_year.keys() == right.item_annual_cost_by_year.keys()
    for key in left.item_annual_cost_by_year:
        assert np.allclose(
            np.array(left.item_annual_cost_by_year[key], dtype=float),
            np.array(right.item_annual_cost_by_year[key], dtype=float),
            atol=1e-9,
        )


def test_generate_streaming_parquet_parallel_matches_single_worker(tmp_path):
    config_path = tmp_path / "opex_config.yaml"
    chunks_dir = tmp_path / "history_chunks"

    _write_dynamic_config(config_path)
    _write_history_chunks(chunks_dir)

    single = OpexGenerator().generate_streaming_parquet(
        config_path=str(config_path),
        chunks_dir=chunks_dir,
        capex_report=None,
        output_dir=None,
        simulation_hours=8760.0,
        workers=1,
    )

    parallel = OpexGenerator().generate_streaming_parquet(
        config_path=str(config_path),
        chunks_dir=chunks_dir,
        capex_report=None,
        output_dir=None,
        simulation_hours=8760.0,
        workers=4,
        max_memory_mb=4096,
    )

    _assert_reports_equivalent(single, parallel)
