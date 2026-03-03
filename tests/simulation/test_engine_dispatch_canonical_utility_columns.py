from types import SimpleNamespace

import pytest

from h2_plant.components.external.biogas_source import BiogasSource
from h2_plant.components.external.water_source import ExternalWaterSource
from h2_plant.control.engine_dispatch import HybridArbitrageEngineStrategy


class _Registry:
    def __init__(self, components):
        self._components = components

    def has(self, component_id):
        return component_id in self._components

    def get(self, component_id):
        return self._components[component_id]

    def list_components(self):
        return list(self._components.items())


class _FakePEM:
    def __init__(self):
        self.component_id = "PEM_Unit"
        self.V_cell = 1.9
        self.P_consumed_W = 1_000_000.0
        self.h2_output_kg = 0.5
        self.o2_impurity_ppm_mol = 0.0
        self.water_consumption_kg = 0.0
        self.o2_output_kg = 0.0

    def get_state(self):
        return {}


class _FakeCoolingManager:
    def __init__(self):
        self.glycol_supply_temp_c = 18.0
        self.glycol_duty_kw = 80.0
        self.cw_supply_temp_c = 25.0
        self.cw_duty_kw = 20.0
        self.tower_fan_power_kw = 4.0
        self.glycol_fan_power_kw = 6.0
        self.power_kw = 10.0


class _FakeBOP:
    def __init__(self):
        self.power_kw = 200.0


def _make_context(dt_hours: float) -> SimpleNamespace:
    return SimpleNamespace(
        physics=SimpleNamespace(
            soec_cluster=SimpleNamespace(
                num_modules=6,
                max_power_nominal_mw=2.0,
                optimal_limit=0.8,
            ),
            pem_system=SimpleNamespace(max_power_mw=5.0),
        ),
        economics=SimpleNamespace(
            bop_pricing_mode="fixed",
            bop_fixed_price_eur_mwh=80.0,
        ),
        simulation=SimpleNamespace(
            timestep_hours=dt_hours,
        ),
    )


def test_record_post_step_writes_canonical_utility_columns_with_consistent_units():
    dt_hours = 0.5
    biogas = BiogasSource(component_id="Biogas_Source", max_flow_rate_kg_h=40.0)
    water = ExternalWaterSource(flow_rate_kg_h=30.0)
    water.component_id = "Water_Source"

    components = {
        "PEM_Unit": _FakePEM(),
        "cooling_manager": _FakeCoolingManager(),
        "BOP_Load": _FakeBOP(),
        "Biogas_Source": biogas,
        "Water_Source": water,
    }
    registry = _Registry(components)

    # Ensure source outputs are initialized and non-zero for this step.
    biogas.initialize(dt_hours, registry)
    biogas.step(0.0)
    water.initialize(dt_hours, registry)
    water.step(0.0)

    strategy = HybridArbitrageEngineStrategy()
    strategy.initialize(registry=registry, context=_make_context(dt_hours), total_steps=2)
    strategy.record_post_step()

    history = strategy._history
    idx = 0
    expected_columns = [
        "sold_energy_mwh_step",
        "pem_electricity_consumption_kwh_step",
        "soec_electricity_consumption_kwh_step",
        "bop_electricity_consumption_kwh_step",
        "total_electric_load_mw",
        "electricity_consumption_kwh_step",
        "total_cooling_duty_kw",
        "cooling_duty_kwh_th_step",
        "biogas_feed_kg_step",
        "water_makeup_kg_step",
    ]
    for col in expected_columns:
        assert col in history
        assert history[col][idx] >= 0.0

    assert "Biogas_Source_outlet_mass_flow_kg_h" in history
    assert "Water_Source_outlet_mass_flow_kg_h" in history

    expected_total_electric_mw = (
        history["P_soec_grid_mw"][idx]
        + history["P_pem_grid_mw"][idx]
        + history["P_bop_grid_usage_mw"][idx]
    )
    assert history["total_electric_load_mw"][idx] == pytest.approx(expected_total_electric_mw, abs=1e-9)
    assert history["electricity_consumption_kwh_step"][idx] == pytest.approx(
        expected_total_electric_mw * 1000.0 * dt_hours,
        abs=1e-9,
    )
    assert history["sold_energy_mwh_step"][idx] == pytest.approx(
        history["P_sold"][idx] * dt_hours,
        abs=1e-9,
    )
    assert history["pem_electricity_consumption_kwh_step"][idx] == pytest.approx(
        history["P_pem_grid_mw"][idx] * 1000.0 * dt_hours,
        abs=1e-9,
    )
    assert history["soec_electricity_consumption_kwh_step"][idx] == pytest.approx(
        history["P_soec_grid_mw"][idx] * 1000.0 * dt_hours,
        abs=1e-9,
    )
    assert history["bop_electricity_consumption_kwh_step"][idx] == pytest.approx(
        history["P_bop_grid_usage_mw"][idx] * 1000.0 * dt_hours,
        abs=1e-9,
    )

    expected_total_cooling_kw = (
        history["cooling_manager_glycol_duty_kw"][idx]
        + history["cooling_manager_cw_duty_kw"][idx]
    )
    assert history["total_cooling_duty_kw"][idx] == pytest.approx(expected_total_cooling_kw, abs=1e-9)
    assert history["cooling_duty_kwh_th_step"][idx] == pytest.approx(
        expected_total_cooling_kw * dt_hours,
        abs=1e-9,
    )

    assert history["biogas_feed_kg_step"][idx] == pytest.approx(
        history["Biogas_Source_outlet_mass_flow_kg_h"][idx] * dt_hours,
        abs=1e-9,
    )
    assert history["water_makeup_kg_step"][idx] == pytest.approx(
        history["Water_Source_outlet_mass_flow_kg_h"][idx] * dt_hours,
        abs=1e-9,
    )
