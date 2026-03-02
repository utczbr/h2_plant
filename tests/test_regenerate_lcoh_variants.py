from pathlib import Path

import pandas as pd
import pytest

from h2_plant.economics.lcoh_calculator import LcohCalculator, LcohInputs
from h2_plant.economics.models import BlockCostSummary, CapexReport
from h2_plant.economics.opex_models import OpexReport


def _write_history_chunk(chunks_dir: Path) -> None:
    chunks_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "minute": [0.0, 60.0, 120.0],
            "H2_pem_kg": [1.0, 1.0, 1.0],
            "H2_soec_kg": [2.0, 2.0, 2.0],
            "H2_atr_kg": [3.0, 3.0, 3.0],
        }
    )
    df.to_parquet(chunks_dir / "chunk_0000.parquet")


def _make_capex_report(
    *,
    total_low: float = 300.0,
    total_base: float = 600.0,
    total_high: float = 960.0,
) -> CapexReport:
    return CapexReport(
        generated_at="2026-01-01T00:00:00",
        total_installed_cost_low=total_low,
        total_installed_cost=total_base,
        total_installed_cost_high=total_high,
        block_summaries=[
            BlockCostSummary(
                block_name="PEM",
                total_installed_cost_low=30.0,
                total_installed_cost=100.0,
                total_installed_cost_high=200.0,
            ),
            BlockCostSummary(
                block_name="SOEC",
                total_installed_cost_low=120.0,
                total_installed_cost=200.0,
                total_installed_cost_high=260.0,
            ),
            BlockCostSummary(
                block_name="ATR",
                total_installed_cost_low=150.0,
                total_installed_cost=300.0,
                total_installed_cost_high=500.0,
            ),
        ],
    )


def _make_opex_report(
    *,
    total_low=90.0,
    total_base=120.0,
    total_high=180.0,
) -> OpexReport:
    return OpexReport(
        scenario_name="test",
        simulation_hours=8760.0,
        total_opex_low=total_low,
        total_opex=total_base,
        total_opex_high=total_high,
    )


def test_lcoh_generate_variants_pairs_capex_and_opex_and_preserves_base_top_level(tmp_path):
    chunks_dir = tmp_path / "history_chunks"
    _write_history_chunk(chunks_dir)

    calc = LcohCalculator()
    report = calc.generate_variants(
        LcohInputs(
            capex_report=_make_capex_report(),
            opex_report=_make_opex_report(),
            history_chunks_dir=chunks_dir,
            discount_rate=0.08,
            project_years=20,
        )
    )

    assert report.variant_order == ["low", "base", "high"]
    assert report.variants["low"].capex_total == pytest.approx(300.0, abs=1e-9)
    assert report.variants["base"].capex_total == pytest.approx(600.0, abs=1e-9)
    assert report.variants["high"].capex_total == pytest.approx(960.0, abs=1e-9)
    assert report.variants["low"].opex_annual_total == pytest.approx(90.0, abs=1e-9)
    assert report.variants["base"].opex_annual_total == pytest.approx(120.0, abs=1e-9)
    assert report.variants["high"].opex_annual_total == pytest.approx(180.0, abs=1e-9)

    base = report.variants["base"]
    assert report.discount_factor_sum == pytest.approx(base.discount_factor_sum, abs=1e-12)
    assert report.capex_total == pytest.approx(base.capex_total, abs=1e-12)
    assert report.opex_annual_total == pytest.approx(base.opex_annual_total, abs=1e-12)
    assert report.annual_h2_total_kg == pytest.approx(base.annual_h2_total_kg, abs=1e-12)
    assert report.lcoh_total == pytest.approx(base.lcoh_total, abs=1e-12)
    assert report.lcoh_weighted_plant == pytest.approx(base.lcoh_weighted_plant, abs=1e-12)
    assert report.annual_h2_by_pathway == base.annual_h2_by_pathway
    assert report.capex_by_pathway == base.capex_by_pathway
    assert report.opex_by_pathway == base.opex_by_pathway
    assert report.lcoh_by_pathway == base.lcoh_by_pathway
    assert report.lcoh_breakdown == base.lcoh_breakdown


def test_lcoh_generate_variants_uses_block_level_low_high_directly(tmp_path):
    chunks_dir = tmp_path / "history_chunks"
    _write_history_chunk(chunks_dir)

    calc = LcohCalculator()
    report = calc.generate_variants(
        LcohInputs(
            capex_report=_make_capex_report(),
            opex_report=_make_opex_report(),
            history_chunks_dir=chunks_dir,
            discount_rate=0.08,
            project_years=20,
        )
    )

    low_capex_by = report.variants["low"].capex_by_pathway
    high_capex_by = report.variants["high"].capex_by_pathway
    assert low_capex_by["pem"] == pytest.approx(30.0, abs=1e-9)
    assert low_capex_by["soec"] == pytest.approx(120.0, abs=1e-9)
    assert low_capex_by["atr"] == pytest.approx(150.0, abs=1e-9)
    assert high_capex_by["pem"] == pytest.approx(200.0, abs=1e-9)
    assert high_capex_by["soec"] == pytest.approx(260.0, abs=1e-9)
    assert high_capex_by["atr"] == pytest.approx(500.0, abs=1e-9)


def test_lcoh_generate_variants_fails_when_capex_variant_missing_or_nonpositive(tmp_path):
    chunks_dir = tmp_path / "history_chunks"
    _write_history_chunk(chunks_dir)

    capex_report = _make_capex_report(total_low=0.0)
    opex_report = _make_opex_report()

    with pytest.raises(ValueError, match="total_installed_cost_low"):
        LcohCalculator().generate_variants(
            LcohInputs(
                capex_report=capex_report,
                opex_report=opex_report,
                history_chunks_dir=chunks_dir,
                discount_rate=0.08,
                project_years=20,
            )
        )


def test_lcoh_generate_variants_fails_when_opex_variant_missing(tmp_path):
    chunks_dir = tmp_path / "history_chunks"
    _write_history_chunk(chunks_dir)

    capex_report = _make_capex_report()
    opex_report = OpexReport(
        scenario_name="test",
        simulation_hours=8760.0,
        total_opex_low=90.0,
        total_opex=120.0,
        total_opex_high=None,
    )

    with pytest.raises(ValueError, match="total_opex_high"):
        LcohCalculator().generate_variants(
            LcohInputs(
                capex_report=capex_report,
                opex_report=opex_report,
                history_chunks_dir=chunks_dir,
                discount_rate=0.08,
                project_years=20,
            )
        )
