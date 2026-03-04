"""
Visualization module for H2 Plant simulation.

Provides interactive and static graph generation with configurable output.

MIGRATION NOTE (2025-12):
    - MetricsCollector and GraphGenerator are DEPRECATED.
    - Use GraphOrchestrator with visualization_config.yaml instead.
    - The main simulation loop (run_integrated_simulation.py) generates graphs
      using the history DataFrame and GraphOrchestrator.

PERFORMANCE NOTE (2026-03):
    - All public names are lazily imported on first access to avoid
      pulling heavy dependencies (Plotly, Matplotlib, CoolProp) at
      package-import time.
"""

import importlib as _importlib

# Map each public name to (submodule, attribute).
_LAZY_IMPORTS = {
    "MetricsCollector": ("h2_plant.visualization.metrics_collector", "MetricsCollector"),
    "GraphGenerator":   ("h2_plant.visualization.graph_generator",   "GraphGenerator"),
    "GraphCatalog":     ("h2_plant.visualization.graph_catalog",     "GraphCatalog"),
    "GRAPH_REGISTRY":   ("h2_plant.visualization.graph_catalog",     "GRAPH_REGISTRY"),
    "GraphOrchestrator": ("h2_plant.visualization.graph_orchestrator", "GraphOrchestrator"),
}

__all__ = list(_LAZY_IMPORTS.keys())


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        module = _importlib.import_module(module_path)
        value = getattr(module, attr)
        # Cache on the module so subsequent accesses skip __getattr__.
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__
