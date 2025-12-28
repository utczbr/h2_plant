"""
Economics Graph Module.
Generates dispatch and power-related charts.
"""
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
import logging

logger = logging.getLogger(__name__)


def plot_dispatch(df: pd.DataFrame, component_ids: list, title: str, config: dict) -> Figure:
    """
    Generates system-level power dispatch stack chart.
    
    Args:
        df: Simulation history DataFrame.
        component_ids: Ignored for system-level data.
        title: Plot title.
        config: Additional plot configuration.
        
    Returns:
        matplotlib Figure object.
    """
    # Check if necessary columns exist
    if 'P_soec' not in df.columns and 'P_soec_actual' not in df.columns:
        logger.warning(f"[{title}] No SOEC power data found.")
        return None
        
    p_soec = df.get('P_soec', df.get('P_soec_actual', pd.Series(np.zeros(len(df)))))
    p_pem = df.get('P_pem', pd.Series(np.zeros(len(df))))
    p_sold = df.get('P_sold', pd.Series(np.zeros(len(df))))
    p_bop = df.get('P_bop_mw', pd.Series(np.zeros(len(df))))
    
    # Downsample for performance
    limit = 2000
    stride = max(1, len(df) // limit)
    
    if 'minute' in df.columns:
        x = df['minute'].iloc[::stride] / 60.0  # Hours
    else:
        x = np.arange(0, len(df), stride)

    y1 = p_soec.iloc[::stride].values
    y2 = p_pem.iloc[::stride].values
    y3 = p_bop.iloc[::stride].values
    y4 = p_sold.iloc[::stride].values
    
    fig = Figure(figsize=(12, 6), constrained_layout=True)
    ax = fig.add_subplot(111)
    
    ax.stackplot(x, y1, y2, y3, y4, 
                 labels=['SOEC', 'PEM', 'BoP', 'Grid Sale'],
                 colors=['tab:orange', 'tab:blue', 'tab:gray', 'tab:green'],
                 alpha=0.8)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Power (MW)")
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_time_series(df: pd.DataFrame, component_ids: list, title: str, config: dict) -> Figure:
    """Plots economics time series (e.g., energy price)."""
    variable = config.get('variable', 'energy_price')
    fig = Figure(figsize=(12, 6), constrained_layout=True)
    ax = fig.add_subplot(111)
    x = df['minute'] / 60.0 if 'minute' in df.columns else df.index
    
    # Map variable to column
    col_map = {
        'energy_price': ['spot_price', 'Spot', 'energy_price'],
    }
    candidates = col_map.get(variable, [variable])
    
    col_name = None
    for cand in candidates:
        if cand in df.columns:
            col_name = cand
            break
    
    if col_name:
        ax.plot(x, df[col_name], color='tab:green', linewidth=1.5)
        ax.set_ylabel("Price (€/MWh)")
    else:
        ax.text(0.5, 0.5, f'No data for {variable}', ha='center', va='center', transform=ax.transAxes)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Time (hours)")
    ax.grid(True, alpha=0.3)
    return fig


def plot_pie(df: pd.DataFrame, component_ids: list, title: str, config: dict) -> Figure:
    """Generates power consumption breakdown pie chart."""
    fig = Figure(figsize=(8, 8), constrained_layout=True)
    ax = fig.add_subplot(111)
    
    # Calculate average power per category
    data = {}
    data['SOEC'] = df.get('P_soec_actual', df.get('P_soec', pd.Series([0]))).mean()
    data['PEM'] = df.get('P_pem', pd.Series([0])).mean()
    data['Compressors'] = df.get('compressor_power_kw', pd.Series([0])).mean() / 1000  # Convert to MW
    data['Chillers'] = df.get('Chiller_1_cooling_load_kw', pd.Series([0])).mean() / 1000
    
    # Filter zero values
    labels = [k for k, v in data.items() if v > 0.01]
    sizes = [data[k] for k in labels]
    
    if sizes:
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    else:
        ax.text(0.5, 0.5, 'No power data', ha='center', va='center', transform=ax.transAxes)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    return fig


def plot_arbitrage(df: pd.DataFrame, component_ids: list, title: str, config: dict) -> Figure:
    """Generates price vs production scatter for arbitrage analysis."""
    fig = Figure(figsize=(8, 8), constrained_layout=True)
    ax = fig.add_subplot(111)
    
    price = df.get('spot_price', df.get('Spot'))
    h2 = df.get('H2_soec_kg', df.get('H2_pem_kg', pd.Series([0])))
    
    if price is not None and len(price) > 0:
        stride = max(1, len(price) // 2000)
        scatter = ax.scatter(price.iloc[::stride], h2.iloc[::stride], 
                             c=df['minute'].iloc[::stride] if 'minute' in df.columns else None,
                             alpha=0.5, s=10, cmap='viridis')
        ax.set_xlabel("Spot Price (€/MWh)")
        ax.set_ylabel("H2 Production (kg/step)")
        if 'minute' in df.columns:
            fig.colorbar(scatter, ax=ax, label='Time (min)')
    else:
        ax.text(0.5, 0.5, 'No price data', ha='center', va='center', transform=ax.transAxes)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    return fig

