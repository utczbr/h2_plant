import numpy as np
import pandas as pd

from h2_plant.visualization.plotly_graphs import plot_cumulative_net_profit


def _build_history(include_power: bool) -> pd.DataFrame:
    hours = np.arange(0.0, 20.0 * 8760.0 + 8760.0, 8760.0)
    payload = {
        "minute": hours * 60.0,
        "cumulative_h2_kg": np.linspace(0.0, 1_000_000.0, len(hours)),
        "cumulative_grid_revenue_eur": np.linspace(0.0, 500_000.0, len(hours)),
    }
    if include_power:
        payload["P_pem"] = np.full(len(hours), 1.0)
        payload["P_soec_actual"] = np.full(len(hours), 1.0)
    return pd.DataFrame(payload)


def _trace_y(fig, trace_name: str) -> np.ndarray:
    for trace in fig.data:
        if getattr(trace, "name", "") == trace_name:
            return np.asarray(trace.y, dtype=float)
    raise AssertionError(f"Trace not found: {trace_name}")


def test_net_profit_replacement_spikes_nonzero_on_20_year_horizon():
    df = _build_history(include_power=True)
    fig = plot_cumulative_net_profit(
        df,
        capex=1_000_000.0,
        opex=180_000.0,
        pem_lifecycle_h=87_600.0,
        soec_lifecycle_h=61_320.0,
        pem_reserve_pct=0.015,
        soec_reserve_pct=0.02,
    )

    pem_replacement = _trace_y(fig, "PEM Replacement")
    soec_replacement = _trace_y(fig, "SOEC Replacement")
    assert np.any(np.abs(pem_replacement) > 0.0)
    assert np.any(np.abs(soec_replacement) > 0.0)


def test_net_profit_replacement_spikes_do_not_require_power_columns():
    df = _build_history(include_power=False)
    assert "P_pem" not in df.columns
    assert "P_soec_actual" not in df.columns

    fig = plot_cumulative_net_profit(
        df,
        capex=1_000_000.0,
        opex=180_000.0,
        pem_lifecycle_h=87_600.0,
        soec_lifecycle_h=61_320.0,
        pem_reserve_pct=0.015,
        soec_reserve_pct=0.02,
    )

    pem_replacement = _trace_y(fig, "PEM Replacement")
    soec_replacement = _trace_y(fig, "SOEC Replacement")
    assert np.any(np.abs(pem_replacement) > 0.0)
    assert np.any(np.abs(soec_replacement) > 0.0)


def test_net_profit_replacement_spikes_zero_when_lifecycle_or_reserve_missing(caplog):
    caplog.set_level("WARNING")
    df = _build_history(include_power=False)

    fig = plot_cumulative_net_profit(
        df,
        capex=1_000_000.0,
        opex=180_000.0,
    )

    pem_replacement = _trace_y(fig, "PEM Replacement")
    soec_replacement = _trace_y(fig, "SOEC Replacement")
    assert np.allclose(pem_replacement, 0.0)
    assert np.allclose(soec_replacement, 0.0)
    assert "replacement spikes set to zero due to missing lifecycle/reserve" in caplog.text.lower()
