"""
Visualization module for H2 Plant simulation.

Provides interactive and static graph generation with configurable output.

MIGRATION NOTE (2025-12):
    - MetricsCollector and GraphGenerator are DEPRECATED.
    - Use GraphOrchestrator with visualization_config.yaml instead.
    - The main simulation loop (run_integrated_simulation.py) generates graphs
      using the history DataFrame and GraphOrchestrator.
"""

from h2_plant.visualization.metrics_collector import MetricsCollector
from h2_plant.visualization.graph_generator import GraphGenerator
from h2_plant.visualization.graph_catalog import GraphCatalog, GRAPH_REGISTRY
from h2_plant.visualization.graph_orchestrator import GraphOrchestrator  # NEW

__all__ = [
    'MetricsCollector',  # Deprecated
    'GraphGenerator',    # Deprecated
    'GraphCatalog',
    'GRAPH_REGISTRY',
    'GraphOrchestrator',  # Recommended
]

