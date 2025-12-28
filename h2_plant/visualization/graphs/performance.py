"""
Performance Graph Module.
Generates efficiency, voltage, and scatter plots.
"""
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
import logging

logger = logging.getLogger(__name__)


def plot_time_series(df: pd.DataFrame, component_ids: list, title: str, config: dict) -> Figure:
    """Plots performance metrics over time."""
    variable = config.get('variable', 'voltage')
    fig = Figure(figsize=(12, 6), constrained_layout=True)
    ax = fig.add_subplot(111)
    x = df['minute'] / 60.0 if 'minute' in df.columns else df.index
    has_data = False

    for comp_id in component_ids:
        col = None
        if variable == 'voltage':
            if "PEM" in comp_id and 'pem_V_cell' in df.columns:
                col = df['pem_V_cell']
        elif variable == 'efficiency':
            if "PEM" in comp_id and 'pem_V_cell' in df.columns:
                # Efficiency ~ 1.23 / V_cell (HHV basis approximation)
                col = 1.23 / df['pem_V_cell'].replace(0, np.nan)
        
        if col is not None and len(col) > 0:
            ax.plot(x, col, label=comp_id, linewidth=1.5)
            has_data = True

    if not has_data:
        ax.text(0.5, 0.5, f'No data for {variable}', ha='center', va='center', transform=ax.transAxes)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel(variable.title())
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def plot_scatter(df: pd.DataFrame, component_ids: list, title: str, config: dict) -> Figure:
    """Generates scatter plots (e.g., Price vs Power)."""
    x_key = config.get('x_axis', 'spot_price')
    y_key = config.get('y_axis', 'total_power')
    
    fig = Figure(figsize=(8, 8), constrained_layout=True)
    ax = fig.add_subplot(111)
    
    x_data = df.get(x_key, df.get('spot_price', df.get('Spot')))
    
    if y_key == 'total_power':
        y_data = df.get('P_soec_actual', pd.Series(np.zeros(len(df)))) + df.get('P_pem', pd.Series(np.zeros(len(df))))
    else:
        y_data = df.get(y_key, pd.Series([]))

    if x_data is not None and len(x_data) > 0 and len(y_data) > 0:
        # Downsample if too many points
        stride = max(1, len(x_data) // 2000)
        ax.scatter(x_data.iloc[::stride], y_data.iloc[::stride], alpha=0.5, s=10, c='tab:blue')
        ax.set_xlabel(x_key.replace('_', ' ').title())
        ax.set_ylabel(y_key.replace('_', ' ').title())
    else:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    return fig
