from types import SimpleNamespace

import numpy as np
import pytest


class _FakeRegistry:
    def __init__(self):
        self._components = {}

    def register(self, component_id, component):
        self._components[component_id] = component

    def has(self, component_id):
        return component_id in self._components


class _FakeBuilder:
    component_ids = ("A",)

    def __init__(self, context):
        self.context = context

    def build(self):
        return {cid: object() for cid in self.component_ids}


class _FakeEngine:
    instances = []

    def __init__(self, registry, config, output_dir, topology, dispatch_strategy):
        self.registry = registry
        self.config = config
        self.output_dir = output_dir
        self.topology = topology
        self.dispatch_strategy = dispatch_strategy
        self.prices = None
        self.wind = None
        self.initialize_dispatch_args = ()
        self.initialize_dispatch_kwargs = None
        self.total_dispatch_steps = 0
        self.run_start_hour = None
        self.run_end_hour = None
        self.steps_run = 0
        self.dispatch_step_idx = 0
        self.last_storage_action_update_step = -1
        self.synthetic_overflow_count = 0
        _FakeEngine.instances.append(self)

    def set_dispatch_data(self, prices, wind):
        self.prices = prices
        self.wind = wind

    def initialize(self):
        return None

    def initialize_dispatch_strategy(self, *args, **kwargs):
        self.initialize_dispatch_args = args
        self.initialize_dispatch_kwargs = kwargs
        if len(args) >= 2:
            self.total_dispatch_steps = int(args[1])
        elif "total_steps" in kwargs:
            self.total_dispatch_steps = int(kwargs["total_steps"])

    def run(self, start_hour, end_hour, **kwargs):
        self.run_start_hour = start_hour
        self.run_end_hour = end_hour
        self.steps_run = int((end_hour - start_hour) * 60)
        self.dispatch_step_idx = min(self.steps_run, self.total_dispatch_steps)
        self.last_storage_action_update_step = self.dispatch_step_idx - 1
        self.synthetic_overflow_count = max(0, self.steps_run - self.dispatch_step_idx)
        return {"simulation": {"duration_hours": self.steps_run / 60.0}}

    def get_dispatch_history(self):
        if self.steps_run <= 0:
            return {}
        action = np.zeros(self.steps_run, dtype=float)
        if self.dispatch_step_idx > 0:
            action[:self.dispatch_step_idx] = np.linspace(
                1.0, 0.0, self.dispatch_step_idx, endpoint=False
            )
            if self.dispatch_step_idx < self.steps_run:
                action[self.dispatch_step_idx:] = action[self.dispatch_step_idx - 1]
        return {"storage_action_factor": action}


class _FakeLoader:
    def __init__(self, base_dir):
        self.base_dir = base_dir

    def load_data(self, *args, **kwargs):
        # Deliberately short to force cyclic profile tiling.
        return np.array([50.0]), np.array([1.0])


class _FakeSimConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def _make_context(duration_hours=24, arbitrage_enabled=False):
    conn = SimpleNamespace(
        source_port="h2_out",
        target_name="Tank_1",
        target_port="h2_in",
        resource_type="hydrogen",
    )
    node = SimpleNamespace(id="PEM_1", connections=[conn])

    return SimpleNamespace(
        topology=SimpleNamespace(nodes=[node]),
        simulation=SimpleNamespace(
            timestep_hours=1.0 / 60.0,
            duration_hours=duration_hours,
            energy_price_file="NL_Prices_2024_15min.csv",
            wind_data_file="producao_horaria_turbina.csv",
            dispatch_strategy="REFERENCE_HYBRID",
            checkpoint_interval_hours=24,
        ),
        economics=SimpleNamespace(arbitrage_enabled=arbitrage_enabled),
    )


@pytest.fixture(autouse=True)
def _patch_dispatch_runner_dependencies(monkeypatch):
    import h2_plant.core.component_registry as registry_mod
    import h2_plant.core.graph_builder as builder_mod
    import h2_plant.data.price_loader as loader_mod
    import h2_plant.simulation.engine as engine_mod
    import h2_plant.config.plant_config as plant_config_mod

    _FakeEngine.instances.clear()

    monkeypatch.setattr(registry_mod, "ComponentRegistry", _FakeRegistry)
    monkeypatch.setattr(builder_mod, "PlantGraphBuilder", _FakeBuilder)
    monkeypatch.setattr(loader_mod, "EnergyPriceLoader", _FakeLoader)
    monkeypatch.setattr(engine_mod, "SimulationEngine", _FakeEngine)
    monkeypatch.setattr(plant_config_mod, "SimulationConfig", _FakeSimConfig)


def test_dispatch_horizon_matches_engine_runtime_for_extended_run(tmp_path):
    from h2_plant.run_integrated_simulation import run_with_dispatch_context

    context = _make_context(duration_hours=24, arbitrage_enabled=False)
    run_with_dispatch_context(
        context,
        hours=96,
        data_dir=".",
        output_dir=tmp_path,
        allow_graph_dispatch_fallback=True,
        return_registry=False,
    )

    assert len(_FakeEngine.instances) == 1
    engine = _FakeEngine.instances[0]
    assert engine.total_dispatch_steps == 5760
    assert engine.steps_run == 5760
    assert engine.dispatch_step_idx == engine.steps_run
    assert engine.run_start_hour == 0
    assert engine.run_end_hour == 96


def test_profiles_are_tiled_to_run_horizon(tmp_path):
    from h2_plant.run_integrated_simulation import run_with_dispatch_context

    context = _make_context(duration_hours=24, arbitrage_enabled=False)
    run_with_dispatch_context(
        context,
        hours=96,
        data_dir=".",
        output_dir=tmp_path,
        allow_graph_dispatch_fallback=True,
        return_registry=False,
    )

    engine = _FakeEngine.instances[0]
    assert len(engine.prices) == 5760
    assert len(engine.wind) == 5760


def test_storage_action_updates_continue_past_1440_steps(tmp_path):
    from h2_plant.run_integrated_simulation import run_with_dispatch_context

    context = _make_context(duration_hours=24, arbitrage_enabled=False)
    history = run_with_dispatch_context(
        context,
        hours=96,
        data_dir=".",
        output_dir=tmp_path,
        allow_graph_dispatch_fallback=True,
        return_registry=False,
    )

    engine = _FakeEngine.instances[0]
    action = history["storage_action_factor"]
    assert len(action) == 5760
    assert engine.last_storage_action_update_step >= 1441
    assert action[1440] != action[1441]
    assert engine.synthetic_overflow_count == 0


def test_economics_report_generation_hook_is_conditional(tmp_path, monkeypatch):
    import h2_plant.run_integrated_simulation as runner_mod

    calls = []

    def _fake_generate_economics_reports(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(runner_mod, "_generate_economics_reports", _fake_generate_economics_reports)

    context = _make_context(duration_hours=2, arbitrage_enabled=False)
    runner_mod.run_with_dispatch_context(
        context,
        data_dir=".",
        output_dir=tmp_path,
        allow_graph_dispatch_fallback=True,
        return_registry=False,
        generate_economics_reports=False,
        reports_scenarios_dir=str(tmp_path / "scenario_disabled"),
    )
    assert calls == []

    reports_scenarios_dir = str(tmp_path / "scenario_enabled")
    history = runner_mod.run_with_dispatch_context(
        context,
        data_dir=".",
        output_dir=tmp_path,
        allow_graph_dispatch_fallback=True,
        return_registry=False,
        generate_economics_reports=True,
        reports_scenarios_dir=reports_scenarios_dir,
    )

    assert len(calls) == 1
    call = calls[0]
    assert call["context"] is context
    assert call["history"] is history
    assert call["output_dir"] == tmp_path
    assert call["scenarios_dir"] == reports_scenarios_dir
    assert call["simulation_hours"] == 2
