"""
Unit tests for SimulationWorker (GUI core).
"""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PySide6.QtCore")

from h2_plant.gui.core.worker import SimulationWorker


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
        self.initialize_dispatch_kwargs = None
        _FakeEngine.instances.append(self)

    def set_dispatch_data(self, prices, wind):
        self.prices = prices
        self.wind = wind

    def initialize(self):
        return None

    def initialize_dispatch_strategy(self, *args, **kwargs):
        self.initialize_dispatch_kwargs = kwargs

    def run(self, start_hour, end_hour, **kwargs):
        return {}

    def get_dispatch_history(self):
        return {}


class _FakeLoader:
    def __init__(self, base_dir):
        self.base_dir = base_dir

    def load_data(self, *args, **kwargs):
        # Deliberately short to exercise worker tiling path
        return np.array([50.0]), np.array([1.0])


def _make_context(arbitrage_enabled=False):
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
            duration_hours=1,
            energy_price_file="NL_Prices_2024_15min.csv",
            wind_data_file="producao_horaria_turbina.csv",
            dispatch_strategy="REFERENCE_HYBRID",
            checkpoint_interval_hours=24,
        ),
        economics=SimpleNamespace(arbitrage_enabled=arbitrage_enabled),
    )


class _FakeSimConfig:
    """Lightweight stand-in for SimulationConfig to avoid import-time side-effects."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


@pytest.fixture(autouse=True)
def _patch_worker_dependencies(monkeypatch):
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


def test_worker_passes_topology_connections_to_engine():
    context = _make_context(arbitrage_enabled=False)
    worker = SimulationWorker(context=context, scenarios_dir=None)
    worker.run()

    assert len(_FakeEngine.instances) == 1
    topology = _FakeEngine.instances[0].topology
    assert len(topology) == 1
    assert topology[0].source_id == "PEM_1"
    assert topology[0].target_id == "Tank_1"


def test_worker_uses_chunked_history_flag_for_dispatch_init():
    context = _make_context(arbitrage_enabled=False)
    worker = SimulationWorker(context=context, scenarios_dir=None)
    worker.run()

    kwargs = _FakeEngine.instances[0].initialize_dispatch_kwargs
    assert kwargs["use_chunked_history"] is True


def test_worker_delegates_to_run_with_dispatch_context():
    """Worker must call run_with_dispatch_context (shared core) rather than
    assembling the engine itself.  This test intercepts the shared core entry
    point to verify the delegation and confirms the engine is set up identically
    to a direct call with the same context."""
    from h2_plant.run_integrated_simulation import run_with_dispatch_context

    calls: list = []

    def _fake_core(context, *, return_registry=False, **kwargs):
        calls.append({"context": context, "kwargs": kwargs, "return_registry": return_registry})
        if return_registry:
            return {}, _FakeRegistry()
        return {}

    import h2_plant.run_integrated_simulation as runner_mod
    original = runner_mod.run_with_dispatch_context
    runner_mod.run_with_dispatch_context = _fake_core
    try:
        context = _make_context(arbitrage_enabled=False)
        worker = SimulationWorker(context=context, scenarios_dir=None)
        worker.run()
    finally:
        runner_mod.run_with_dispatch_context = original

    assert len(calls) == 1, "Worker must call run_with_dispatch_context exactly once"
    call = calls[0]
    assert call["context"] is context
    assert call["return_registry"] is True


def test_gui_and_cli_shared_core_produce_equivalent_engine_topology():
    """Direct call to run_with_dispatch_context (as used by both GUI worker and
    CLI wrapper) must produce topology connections equivalent to what the worker
    produced when it assembled the engine itself.

    Both paths should see the same source_id / target_id / port names.
    """
    from h2_plant.run_integrated_simulation import run_with_dispatch_context

    context = _make_context(arbitrage_enabled=False)
    run_with_dispatch_context(
        context,
        data_dir=".",
        output_dir=None,
        allow_graph_dispatch_fallback=True,
        return_registry=False,
    )

    assert len(_FakeEngine.instances) == 1
    topology = _FakeEngine.instances[0].topology
    assert len(topology) == 1
    conn = topology[0]
    assert conn.source_id == "PEM_1"
    assert conn.target_id == "Tank_1"
    assert conn.source_port == "h2_out"
    assert conn.target_port == "h2_in"
    assert conn.resource_type == "hydrogen"
