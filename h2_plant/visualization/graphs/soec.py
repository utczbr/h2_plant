"""
SOEC Graph Module.
Generates SOEC module activity, heatmaps, and wear statistics.
"""
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
import logging

from h2_plant.visualization import utils

logger = logging.getLogger(__name__)


def plot_active_modules(df: pd.DataFrame, component_ids: list, title: str, config: dict) -> Figure:
    """Plots number of active SOEC modules over time."""
    fig = Figure(figsize=(12, 5), constrained_layout=True)
    ax = fig.add_subplot(111)
    
    df_ds = utils.downsample_dataframe(df, max_points=2000)
    x = df_ds['minute'] / 60.0 if 'minute' in df_ds.columns else df_ds.index
    
    if 'soec_active_modules' in df_ds.columns:
        ax.step(x, df_ds['soec_active_modules'], where='post', color='tab:orange', linewidth=1.5)
        ax.set_ylabel("Active Modules")
        ax.set_ylim(bottom=0)
    else:
        ax.text(0.5, 0.5, 'No SOEC module data', ha='center', va='center', transform=ax.transAxes)
        
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Time (hours)")
    ax.grid(True, alpha=0.3)
    return fig


def plot_module_heatmap(df: pd.DataFrame, component_ids: list, title: str, config: dict) -> Figure:
    """Visualizes individual module power as heatmap."""
    cols = sorted([c for c in df.columns if 'soec_module_powers_' in c])
    
    if not cols:
        fig = Figure(figsize=(12, 4), constrained_layout=True)
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, 'No module power data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return fig

    data = df[cols].values.T  # Shape: (Num_Modules, Time)
    
    fig = Figure(figsize=(12, 4), constrained_layout=True)
    ax = fig.add_subplot(111)
    
    # Downsample for visual clarity
    stride = max(1, data.shape[1] // 1000)
    
    im = ax.imshow(data[:, ::stride], aspect='auto', cmap='magma', interpolation='nearest')
    
    ax.set_ylabel("Module Index")
    ax.set_xlabel("Time Step (downsampled)")
    ax.set_title(title, fontsize=14, fontweight='bold')
    fig.colorbar(im, ax=ax, label="Power (MW)")
    return fig


def plot_module_stats(df: pd.DataFrame, component_ids: list, title: str, config: dict) -> Figure:
    """Plots module wear/runtime statistics as bar chart."""
    cols = sorted([c for c in df.columns if 'soec_module_powers_' in c])
    
    if not cols:
        fig = Figure(figsize=(10, 6), constrained_layout=True)
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, 'No module data', ha='center', va='center', transform=ax.transAxes)
        return fig

    # Calculate runtime (hours where power > 0)
    dt = 1.0 / 60.0  # Assume 1-minute steps
    if 'minute' in df.columns and len(df) > 1:
        dt = (df['minute'].iloc[1] - df['minute'].iloc[0]) / 60.0
    
    runtimes = []
    for col in cols:
        runtime_hours = (df[col] > 0.01).sum() * dt
        runtimes.append(runtime_hours)
    
    fig = Figure(figsize=(10, 6), constrained_layout=True)
    ax = fig.add_subplot(111)
    
    module_ids = [f"M{i}" for i in range(len(cols))]
    bars = ax.bar(module_ids, runtimes, color='tab:orange', alpha=0.8)
    
    ax.set_xlabel("Module")
    ax.set_ylabel("Runtime (hours)")
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)
    
    return fig
