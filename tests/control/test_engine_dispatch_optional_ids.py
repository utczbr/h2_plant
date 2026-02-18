"""
Regression tests for optional component lookups in HybridArbitrageEngineStrategy.
"""

from types import SimpleNamespace

from h2_plant.control.engine_dispatch import HybridArbitrageEngineStrategy


class _StrictRegistry:
    """
    Registry stub that fails on unsafe direct get() for missing IDs.
    """

    def __init__(self, components=None):
        self._components = components or {}

    def has(self, component_id):
        return component_id in self._components

    def get(self, component_id):
        if component_id not in self._components:
            raise AssertionError(f"unsafe get() for missing component '{component_id}'")
        return self._components[component_id]

    def list_components(self):
        return list(self._components.items())


def _make_context():
    return SimpleNamespace(
        physics=SimpleNamespace(
            soec_cluster=SimpleNamespace(
                num_modules=6,
                max_power_nominal_mw=2.4,
                optimal_limit=0.8,
            ),
            pem_system=SimpleNamespace(max_power_mw=5.0),
        ),
        economics=SimpleNamespace(
            bop_pricing_mode="fixed",
            bop_fixed_price_eur_mwh=80.0,
        ),
        simulation=SimpleNamespace(
            timestep_hours=1.0 / 60.0,
        ),
    )


def test_initialize_and_record_post_step_with_missing_optional_ids():
    strategy = HybridArbitrageEngineStrategy()
    registry = _StrictRegistry()  # no transformers, mixer, PSAs, cooling_manager
    context = _make_context()

    strategy.initialize(registry=registry, context=context, total_steps=2)
    strategy.record_post_step()

    # Missing transformers should not crash; defaults to unity efficiency.
    assert strategy._η_soec_trafo == 1.0
    assert strategy._η_pem_trafo == 1.0
    assert strategy._η_bop_trafo == 1.0
    # record_post_step should complete and advance state.
    assert strategy._state.step_idx == 1
