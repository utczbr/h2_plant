"""
Storage Graph Module.
Generates tank level and compressor power charts.
"""
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
import logging

from h2_plant.visualization import utils

logger = logging.getLogger(__name__)


def plot_tank_levels(df: pd.DataFrame, component_ids: list, title: str, config: dict) -> Figure:
    """Plots H2 tank levels over time with optional pressure overlay."""
    fig = Figure(figsize=(12, 6), constrained_layout=True)
    ax = fig.add_subplot(111)
    
    df_ds = utils.downsample_dataframe(df, max_points=2000)
    x = df_ds['minute'] / 60.0 if 'minute' in df_ds.columns else df_ds.index
    
    has_data = False
    
    if 'tank_level_kg' in df_ds.columns:
        ax.plot(x, df_ds['tank_level_kg'], label='Main Tank Level', color='tab:blue', linewidth=2)
        has_data = True
        
        if 'tank_pressure_bar' in df_ds.columns:
            ax2 = ax.twinx()
            ax2.plot(x, df_ds['tank_pressure_bar'], color='tab:red', linestyle='--', alpha=0.6, label='Pressure')
            ax2.set_ylabel("Pressure (bar)", color='tab:red')
            ax2.tick_params(axis='y', labelcolor='tab:red')

    # Look for component-specific columns
    for comp_id in component_ids:
        col = f"{comp_id}_level_kg"
        if col in df_ds.columns:
            ax.plot(x, df_ds[col], label=comp_id.replace('_', ' '), linewidth=1.5)
            has_data = True

    if not has_data:
        ax.text(0.5, 0.5, 'No tank level data', ha='center', va='center', transform=ax.transAxes)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Level (kg)")
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    return fig


def plot_compressor_power(df: pd.DataFrame, component_ids: list, title: str, config: dict) -> Figure:
    """Plots compressor power consumption stacked or individual."""
    fig = Figure(figsize=(12, 6), constrained_layout=True)
    ax = fig.add_subplot(111)
    
    df_ds = utils.downsample_dataframe(df, max_points=2000)
    x = df_ds['minute'] / 60.0 if 'minute' in df_ds.columns else df_ds.index
    
    has_data = False
    stack_data = []
    labels = []
    
    # Collect breakdown data
    for comp_id in component_ids:
        # Try multiple suffixes
        for suffix in ['power_kw', 'shaft_power_kw', 'electric_power_kw']:
            col = f"{comp_id}_{suffix}"
            if col in df_ds.columns:
                stack_data.append(df_ds[col].values)
                # Pretty label: Compressor_S1 -> S1
                label = comp_id.replace('Compressor_', '').replace('_', ' ')
                labels.append(label)
                has_data = True
                break
    
    # Plot Logic
    if stack_data:
        # We have breakdown data - use stackplot
        ax.stackplot(x, stack_data, labels=labels, alpha=0.8)
        
        # Add Total as a line for reference
        if 'compressor_power_kw' in df_ds.columns:
            ax.plot(x, df_ds['compressor_power_kw'], color='black', linestyle='--', linewidth=1.5, label='Total')
        
        # Calculate approximate energy for title (using mean diff of downsampled time, or original if provided)
        # Using full df for accurate energy calc
        if 'minute' in df.columns:
            total_kwh_full = 0
            # Re-sum from full df
            for comp_id in component_ids:
                for suffix in ['power_kw', 'shaft_power_kw', 'electric_power_kw']:
                     col_full = f"{comp_id}_{suffix}"
                     if col_full in df.columns:
                         total_kwh_full += df[col_full].sum() * (df['minute'].diff().mean() / 60.0)
                         break
            title += f" (Total: {total_kwh_full/1000:.2f} MWh)"

    elif 'compressor_power_kw' in df_ds.columns:
        # Only total available - use area fill
        ax.fill_between(x, 0, df_ds['compressor_power_kw'], color='tab:gray', alpha=0.6, label='Total')
        has_data = True
        
    if has_data:
        ax.legend(loc='upper left', fontsize=8, ncol=2)
    else:
        ax.text(0.5, 0.5, 'No compressor power data', ha='center', va='center', transform=ax.transAxes)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Power (kW)")
    ax.grid(True, alpha=0.3)
    return fig


def plot_apc(df: pd.DataFrame, component_ids: list, title: str, config: dict) -> Figure:
    """
    Plots Storage APC control status.
    Wraps create_storage_apc_figure from static_graphs.
    """
    from h2_plant.visualization.static_graphs import create_storage_apc_figure
    return create_storage_apc_figure(df)


def plot_inventory(df: pd.DataFrame, component_ids: list, title: str, config: dict) -> Figure:
    """
    Plots storage inventory as percentage of capacity.
    Wraps create_storage_inventory_figure from static_graphs.
    """
    from h2_plant.visualization.static_graphs import create_storage_inventory_figure
    return create_storage_inventory_figure(df)


def plot_pressure_heatmap(df: pd.DataFrame, component_ids: list, title: str, config: dict) -> Figure:
    """
    Plots per-tank pressure heatmap.
    Wraps create_storage_pressure_heatmap_figure from static_graphs.
    
    Note: This function requires matrix data from simulation_matrices.npz.
    If called with DataFrame only, it will attempt to load the matrix data
    from the default output location.
    """
    from h2_plant.visualization.static_graphs import create_storage_pressure_heatmap_figure
    from pathlib import Path
    
    # Try to load matrix data from a known location
    # The matrices are typically saved in simulation_output/
    matrices = {}
    try:
        # Check if matrix path is in config
        matrix_path = config.get('matrix_path')
        if matrix_path:
            matrices = np.load(matrix_path)
        else:
            # Try default location (relative to current working directory)
            default_paths = [
                Path("simulation_output/simulation_matrices.npz"),
                Path("scenarios/simulation_output/simulation_matrices.npz"),
            ]
            for path in default_paths:
                if path.exists():
                    matrices = np.load(str(path))
                    break
    except Exception as e:
        logger.warning(f"Could not load matrix data: {e}")
    
    return create_storage_pressure_heatmap_figure(matrices)

