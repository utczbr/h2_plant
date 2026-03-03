from pathlib import Path

import pandas as pd
import pytest

from h2_plant.economics.lcoh_calculator import LcohCalculator, LcohInputs
from h2_plant.economics.models import BlockCostSummary, CapexReport
from h2_plant.economics.opex_models import OpexCategory, OpexReport, OpexResult


def _write_history_chunk(chunks_dir: Path) -> None:
    chunks_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "minute": [0.0, 60.0, 120.0],
            "H2_pem_kg": [1.0, 1.0, 1.0],
            "H2_soec_kg": [1.0, 1.0, 1.0],
            "H2_atr_kg": [1.0, 1.0, 1.0],
        }
    ).to_parquet(chunks_dir / "chunk_0000.parquet")


def _capex_report() -> CapexReport:
    return CapexReport(
        generated_at="2026-01-01T00:00:00",
        total_installed_cost_low=800.0,
        total_installed_cost=1_000.0,
        total_installed_cost_high=1_200.0,
        block_summaries=[
            BlockCostSummary(
                block_name="PEM",
                total_installed_cost_low=200.0,
                total_installed_cost=300.0,
                total_installed_cost_high=350.0,
            ),
            BlockCostSummary(
                block_name="SOEC",
                total_installed_cost_low=300.0,
                total_installed_cost=350.0,
                total_installed_cost_high=450.0,
            ),
            BlockCostSummary(
                block_name="ATR",
                total_installed_cost_low=300.0,
                total_installed_cost=350.0,
                total_installed_cost_high=400.0,
            ),
        ],
    )


def _opex_report_with_components() -> OpexReport:
    items = [
        OpexResult(
            name="Electricity",
            category=OpexCategory.VARIABLE,
            annual_cost=60.0,
            pathway_shares={"pem": 0.7, "soec": 0.2, "atr": 0.1},
            lcoh_component="energy",
        ),
        OpexResult(
            name="Biogas",
            category=OpexCategory.VARIABLE,
            annual_cost=20.0,
            pathway_shares={"pem": 0.0, "soec": 0.0, "atr": 1.0},
            lcoh_component="energy",
        ),
        OpexResult(
            name="Water",
            category=OpexCategory.VARIABLE,
            annual_cost=10.0,
            lcoh_component="water",
        ),
        OpexResult(
            name="Cooling",
            category=OpexCategory.VARIABLE,
            annual_cost=5.0,
            lcoh_component="compression",
        ),
        OpexResult(
            name="Labor",
            category=OpexCategory.FIXED,
            annual_cost=25.0,
            lcoh_component=None,
        ),
    ]

    report = OpexReport(
        scenario_name="test",
        simulation_hours=8760.0,
        items=items,
        total_opex_low=90.0,
        total_opex=120.0,
        total_opex_high=180.0,
    )
    report.calculate_totals()
    report.total_opex_low = 90.0
    report.total_opex_high = 180.0
    return report


def test_lcoh_breakdown_components_are_present_and_reconcile_to_opex(tmp_path):
    chunks_dir = tmp_path / "history_chunks"
    _write_history_chunk(chunks_dir)

    report = LcohCalculator().generate_variants(
        LcohInputs(
            capex_report=_capex_report(),
            opex_report=_opex_report_with_components(),
            history_chunks_dir=chunks_dir,
            discount_rate=0.08,
            project_years=20,
        )
    )

    for variant in report.variant_order:
        breakdown = report.variants[variant].lcoh_breakdown
        assert "capex" in breakdown
        assert "opex" in breakdown
        assert "energy" in breakdown
        assert "water" in breakdown
        assert "compression" in breakdown
        assert "other_opex" in breakdown

        variable_sum = (
            breakdown["energy"]
            + breakdown["water"]
            + breakdown["compression"]
            + breakdown["other_opex"]
        )
        assert variable_sum == pytest.approx(breakdown["opex"], rel=1e-9, abs=1e-12)

    # Backward-compatible top-level keys are still present on combined report.
    assert "capex" in report.lcoh_breakdown
    assert "opex" in report.lcoh_breakdown

    base_opex_by = report.variants["base"].opex_by_pathway
    assert base_opex_by["pem"] == pytest.approx(55.3333333333, abs=1e-9)
    assert base_opex_by["soec"] == pytest.approx(25.3333333333, abs=1e-9)
    assert base_opex_by["atr"] == pytest.approx(39.3333333333, abs=1e-9)
    assert sum(base_opex_by.values()) == pytest.approx(report.variants["base"].opex_annual_total, abs=1e-9)


def test_lcoh_raises_for_nonzero_variable_item_with_zero_pathway_shares(tmp_path):
    chunks_dir = tmp_path / "history_chunks"
    _write_history_chunk(chunks_dir)

    report = OpexReport(
        scenario_name="invalid",
        simulation_hours=8760.0,
        items=[
            OpexResult(
                name="Invalid Variable",
                category=OpexCategory.VARIABLE,
                annual_cost=10.0,
                pathway_shares={"pem": 0.0, "soec": 0.0, "atr": 0.0},
            )
        ],
        total_opex_low=10.0,
        total_opex=10.0,
        total_opex_high=10.0,
    )
    report.calculate_totals()
    report.total_opex_low = 10.0
    report.total_opex_high = 10.0

    with pytest.raises(ValueError, match="sum to zero"):
        LcohCalculator().generate_variants(
            LcohInputs(
                capex_report=_capex_report(),
                opex_report=report,
                history_chunks_dir=chunks_dir,
                discount_rate=0.08,
                project_years=20,
            )
        )
