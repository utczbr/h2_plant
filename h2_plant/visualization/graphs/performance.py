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
    """Generates scatter plots with binned average trend line (e.g., Price vs Power)."""
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
        # 1. Raw Scatter (Background context)
        # Downsample if too many points for the scatter background
        stride = max(1, len(x_data) // 5000)
        ax.scatter(x_data.iloc[::stride], y_data.iloc[::stride], 
                  alpha=0.15, s=15, c='silver', label='Raw Data') # Lighter, more transparent

        # 2. Binned Average (Trend Line)
        try:
            # Create bins for X axis (e.g., Price)
            # Use ~20 bins spanning the range
            x_min, x_max = x_data.min(), x_data.max()
            if x_max > x_min:
                bins = np.linspace(x_min, x_max, 21) # 20 intervals
                # Cut data into bins
                df_temp = pd.DataFrame({'x': x_data, 'y': y_data})
                df_temp['bin'] = pd.cut(df_temp['x'], bins)
                
                # Calculate mean per bin
                binned_stats = df_temp.groupby('bin', observed=True)['y'].agg(['mean', 'count']).reset_index()
                # Calculate bin centers for plotting
                binned_stats['x_center'] = binned_stats['bin'].apply(lambda b: b.mid).astype(float)
                
                # Filter empty bins
                binned_stats = binned_stats[binned_stats['count'] > 0]
                
                # Plot Trend Line
                ax.plot(binned_stats['x_center'], binned_stats['mean'], 
                       color='tab:blue', linewidth=2.5, marker='o', markersize=6, 
                       label='Average Trend')
        except Exception as e:
            logger.warning(f"Failed to calculate binned average for scatter: {e}")
            # Fallback to simple scatter if binning fails
            pass

        ax.set_xlabel(x_key.replace('_', ' ').title())
        ax.set_ylabel(y_key.replace('_', ' ').title())
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    return fig
