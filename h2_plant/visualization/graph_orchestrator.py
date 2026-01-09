"""
Graph Orchestrator.
Central dispatcher for graph generation based on configuration.
"""
import logging
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional
from pathlib import Path

# Import specialized graph modules
from h2_plant.visualization.graphs import (
    profiles,
    thermal,
    separation,
    economics,
    production,
    performance,
    soec,
    storage
)

# Import legacy wrappers (for backward compatibility)
try:
    from h2_plant.visualization.graphs.legacy import LEGACY_HANDLERS, LEGACY_AVAILABLE
except ImportError:
    LEGACY_HANDLERS = {}
    LEGACY_AVAILABLE = False

logger = logging.getLogger(__name__)


class GraphOrchestrator:
    """
    Central dispatcher for graph generation.
    Reads visualization_config.yaml and calls specialized graph modules.
    """
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Registry mapping config keys to handler functions
        self.handlers = {
            # Profile/Train
            'process_train_profile': profiles.plot_profile,
            'temperature_profile': profiles.plot_temperature_profile,
            'pressure_profile': profiles.plot_pressure_profile,
            'flow_profile': profiles.plot_flow_profile,
            
            # Thermal & Separation
            'thermal_load_breakdown': thermal.plot_load_breakdown,
            'water_removal_bar': separation.plot_water_removal,
            
            # Economics/Dispatch
            'dispatch_stack': economics.plot_dispatch,
            'economics_time_series': economics.plot_time_series,
            'economics_pie': economics.plot_pie,
            'economics_scatter': economics.plot_arbitrage,
            'effective_ppa': economics.plot_effective_ppa,

            # Production
            'production_time_series': production.plot_time_series,
            'production_stacked': production.plot_stacked,
            'production_cumulative': production.plot_cumulative,

            # Performance
            'performance_time_series': performance.plot_time_series,
            'performance_scatter': performance.plot_scatter,

            # SOEC
            'soec_modules_time_series': soec.plot_active_modules,
            'soec_heatmap': soec.plot_module_heatmap,
            'soec_stats': soec.plot_module_stats,

            # Storage
            'storage_levels': storage.plot_tank_levels,
            'compressor_power': storage.plot_compressor_power,
            'storage_apc': storage.plot_apc,
            'storage_inventory': storage.plot_inventory,
            'storage_pressure_heatmap': storage.plot_pressure_heatmap
        }
        
        # Register legacy handlers (prefixed with 'legacy_')
        if LEGACY_AVAILABLE:
            self.handlers.update(LEGACY_HANDLERS)
            logger.debug(f"Registered {len(LEGACY_HANDLERS)} legacy handlers")

    def generate_all(self, df: pd.DataFrame, config: Dict[str, Any]) -> int:
        """
        Main entry point. Iterates through config and dispatches jobs.
        
        Args:
            df: The simulation history DataFrame.
            config: The loaded dictionary from visualization_config.yaml.
            
        Returns:
            Number of graphs generated successfully.
        """
        # Try new orchestrated_graphs first, fallback to graphs
        graphs_config = config.get('visualization', {}).get('orchestrated_graphs', {})
        if not graphs_config:
            graphs_config = config.get('visualization', {}).get('graphs', {})
        
        if not graphs_config:
            logger.warning("No graph configuration found.")
            return 0

        generated_count = 0

        for graph_type, settings in graphs_config.items():
            # Skip if disabled or if it's a boolean (legacy format)
            if isinstance(settings, bool):
                continue
            if not settings.get('enabled', False):
                continue

            handler = self.handlers.get(graph_type)
            if not handler:
                logger.debug(f"No handler for: {graph_type}")
                continue

            plots = settings.get('plots', [])
            if not plots:
                continue

            for i, plot_config in enumerate(plots):
                try:
                    title = plot_config.get('title', f"{graph_type}_{i}")
                    components = plot_config.get('components', [])
                    
                    logger.info(f"Generating: {title}")
                    
                    fig = handler(df, components, title, plot_config)
                    
                    if fig:
                        safe_title = "".join([c if c.isalnum() else "_" for c in title])
                        filename = self.output_dir / f"{safe_title}.png"
                        fig.savefig(filename, dpi=100, bbox_inches='tight')
                        plt.close(fig)
                        generated_count += 1
                        
                except Exception as e:
                    logger.error(f"Failed: {title}: {e}", exc_info=True)
                    
        return generated_count


def create_all_graphs(df: pd.DataFrame, config: Dict[str, Any], output_dir: Path) -> int:
    """Facade function for simple usage."""
    orchestrator = GraphOrchestrator(output_dir)
    return orchestrator.generate_all(df, config)
