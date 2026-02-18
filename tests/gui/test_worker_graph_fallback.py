"""
Graph-mode dispatch fallback behavior for SimulationWorker.
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
    # Missing SOEC_Transformer / PEM_Transformer / BOP_Transformer on purpose.
    component_ids = ("PEM_1", "Tank_1")

    def __init__(self, context):
        self.context = context

    def build(self):
        return {cid: object() for cid in self.component_ids}


class _FakeStrategy:
    def __init__(self):
        self._strategy_override = None


class _FakeEngine:
    instances = []

    def __init__(self, registry, config, output_dir, topology, dispatch_strategy):
        self.dispatch_strategy = dispatch_strategy
        _FakeEngine.instances.append(self)

    def set_dispatch_data(self, prices, wind):
        return None

    def initialize(self):
        return None

    def initialize_dispatch_strategy(self, *args, **kwargs):
        return None

    def run(self, start_hour, end_hour):
        return {}

    def get_dispatch_history(self):
        return {}


class _FakeLoader:
    def __init__(self, base_dir):
        self.base_dir = base_dir

    def load_data(self, *args, **kwargs):
        return np.array([60.0]), np.array([2.0])


def _make_context():
    node = SimpleNamespace(id="PEM_1", connections=[])
    return SimpleNamespace(
        topology=SimpleNamespace(nodes=[node]),
        simulation=SimpleNamespace(
            timestep_hours=1.0 / 60.0,
            duration_hours=1,
            energy_price_file="NL_Prices_2024_15min.csv",
            wind_data_file="producao_horaria_turbina.csv",
            dispatch_strategy="ECONOMIC_SPOT",
            checkpoint_interval_hours=24,
        ),
        economics=SimpleNamespace(arbitrage_enabled=True),
    )


@pytest.fixture(autouse=True)
def _patch_dependencies(monkeypatch):
    import h2_plant.control.engine_dispatch as dispatch_mod
    import h2_plant.core.component_registry as registry_mod
    import h2_plant.core.graph_builder as builder_mod
    import h2_plant.data.price_loader as loader_mod
    import h2_plant.simulation.engine as engine_mod

    _FakeEngine.instances.clear()

    monkeypatch.setattr(registry_mod, "ComponentRegistry", _FakeRegistry)
    monkeypatch.setattr(builder_mod, "PlantGraphBuilder", _FakeBuilder)
    monkeypatch.setattr(loader_mod, "EnergyPriceLoader", _FakeLoader)
    monkeypatch.setattr(engine_mod, "SimulationEngine", _FakeEngine)
    monkeypatch.setattr(dispatch_mod, "HybridArbitrageEngineStrategy", _FakeStrategy)


def test_graph_mode_missing_dispatch_components_falls_back_to_physics_only(caplog):
    context = _make_context()
    worker = SimulationWorker(
        context=context,
        strategy_override="ECONOMIC_SPOT",
        scenarios_dir=None,  # Graph mode
    )

    with caplog.at_level("WARNING"):
        worker.run()

    assert len(_FakeEngine.instances) == 1
    assert _FakeEngine.instances[0].dispatch_strategy is None
    assert "Falling back to physics-only mode" in caplog.text
