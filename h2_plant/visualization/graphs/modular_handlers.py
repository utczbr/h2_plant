"""
Modular Graph Handlers Registry.

This module provides the registry of modular graph handlers (profiles, thermal, etc.)
and wrapper functions to adapt them to the unified executor interface.
"""
from typing import Callable, Dict, Any
import pandas as pd
from matplotlib.figure import Figure

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

# Registry mapping config keys to handler functions (migrated from GraphOrchestrator)
MODULAR_HANDLERS = {
    # Profile/Train
    'process_train_profile': profiles.plot_profile,
    'temperature_profile': profiles.plot_temperature_profile,
    'pressure_profile': profiles.plot_pressure_profile,
    'flow_profile': profiles.plot_flow_profile,
    
    # Thermal & Separation
    'thermal_load_breakdown': thermal.plot_load_breakdown,
    'thermal_time_series': thermal.plot_thermal_time_series,
    'water_removal_bar': separation.plot_water_removal,
    'central_cooling_performance': thermal.plot_central_cooling_performance,
    
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


def create_modular_wrapper(
    handler_func: Callable, 
    components: list, 
    title: str, 
    config: dict
) -> Callable[[pd.DataFrame, int], Figure]:
    """
    Create a wrapper function that adapts the modular graph handler signature
    (df, components, title, config) to the unified executor signature (df, dpi).
    """
    def wrapper(df: pd.DataFrame, dpi: int = 100) -> Figure:
        # Note: Modular handlers typically handle their own figure creation and don't accept dpi arg explicitly,
        # but they return a Figure which UnifiedGraphExecutor then saves with the correct DPI.
        return handler_func(df, components, title, config)
    
    return wrapper
