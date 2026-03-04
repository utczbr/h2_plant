"""Tests for lazy import behavior of h2_plant.visualization package."""

import importlib
import sys
import types


def test_visualization_package_import_does_not_trigger_heavy_submodules():
    """Importing h2_plant.visualization should not eagerly load plotly_graphs or static_graphs."""
    # Remove cached visualization modules so reimport is clean.
    mods_to_remove = [
        k for k in sys.modules
        if k.startswith("h2_plant.visualization")
    ]
    saved = {}
    for mod in mods_to_remove:
        saved[mod] = sys.modules.pop(mod)

    try:
        import h2_plant.visualization  # noqa: F811

        # After package import, heavy submodules should NOT be in sys.modules.
        assert "h2_plant.visualization.plotly_graphs" not in sys.modules, (
            "plotly_graphs was eagerly imported by h2_plant.visualization"
        )
        assert "h2_plant.visualization.static_graphs" not in sys.modules, (
            "static_graphs was eagerly imported by h2_plant.visualization"
        )
        assert "h2_plant.visualization.graph_catalog" not in sys.modules, (
            "graph_catalog was eagerly imported by h2_plant.visualization"
        )
        assert "h2_plant.visualization.graph_generator" not in sys.modules, (
            "graph_generator was eagerly imported by h2_plant.visualization"
        )
    finally:
        # Restore original module state.
        for mod in list(sys.modules):
            if mod.startswith("h2_plant.visualization"):
                sys.modules.pop(mod, None)
        sys.modules.update(saved)


def test_visualization_dir_lists_public_names():
    """__dir__ should report all public names."""
    import h2_plant.visualization as viz

    names = dir(viz)
    assert "GraphCatalog" in names
    assert "GRAPH_REGISTRY" in names
    assert "GraphOrchestrator" in names
    assert "MetricsCollector" in names
    assert "GraphGenerator" in names


def test_lazy_getattr_raises_for_unknown():
    """Accessing an unknown attribute should raise AttributeError."""
    import h2_plant.visualization as viz

    try:
        _ = viz.NonExistentAttribute
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass
