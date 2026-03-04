from pathlib import Path

import pandas as pd
import pytest

from h2_plant.economics.lcoh_calculator import LcohCalculator


def _write_chunk(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def test_lcoh_history_parallel_scan_matches_serial(tmp_path):
    chunks_dir = tmp_path / "history_chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    # Empty first chunk verifies deterministic first-non-empty minute origin handling.
    _write_chunk(
        chunks_dir / "chunk_0000.parquet",
        pd.DataFrame(columns=["minute", "H2_pem_kg", "H2_soec_kg", "H2_atr_kg"]),
    )
    _write_chunk(
        chunks_dir / "chunk_0001.parquet",
        pd.DataFrame(
            {
                "minute": [100.0, 160.0, 220.0],
                "H2_pem_kg": [1.0, 1.5, 2.0],
                "H2_soec_kg": [0.5, 1.0, 1.5],
                "H2_atr_kg": [0.0, 0.5, 0.5],
            }
        ),
    )
    _write_chunk(
        chunks_dir / "chunk_0002.parquet",
        pd.DataFrame(
            {
                "minute": [525700.0, 525760.0, 525850.0],
                "H2_pem_kg": [2.0, 2.5, 3.0],
                "H2_soec_kg": [1.0, 1.0, 1.0],
                "H2_atr_kg": [0.25, 0.25, 0.25],
            }
        ),
    )

    calc = LcohCalculator()
    serial = calc._load_h2_totals(chunks_dir, workers=1, scan_mode="serial")
    parallel = calc._load_h2_totals(chunks_dir, workers=4, scan_mode="parallel")

    assert serial["totals"]["pem"] == pytest.approx(parallel["totals"]["pem"], abs=1e-12)
    assert serial["totals"]["soec"] == pytest.approx(parallel["totals"]["soec"], abs=1e-12)
    assert serial["totals"]["atr"] == pytest.approx(parallel["totals"]["atr"], abs=1e-12)
    for key in ("pem", "soec", "atr"):
        assert serial["totals_by_year"][key] == pytest.approx(parallel["totals_by_year"][key], abs=1e-12)
    assert serial["year_indices"] == parallel["year_indices"]
    assert serial["year_hours"] == pytest.approx(parallel["year_hours"], abs=1e-12)
    assert serial["simulation_hours"] == pytest.approx(parallel["simulation_hours"], abs=1e-12)
