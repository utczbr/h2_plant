"""
Economics Graph Module.
Generates dispatch and power-related charts.
"""
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
import logging

from h2_plant.visualization import utils

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
    
    x = utils.get_time_axis_hours(df).iloc[::stride] if hasattr(utils.get_time_axis_hours(df), 'iloc') else utils.get_time_axis_hours(df)[::stride]

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
    x = utils.get_time_axis_hours(df)
    
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
    """
    Generates power consumption breakdown as a Ranked Horizontal Bar Chart.
    (Formerly a Pie chart, updated for engineering visibility of small loads).
    """
    fig = Figure(figsize=(10, 6), constrained_layout=True)
    ax = fig.add_subplot(111)
    
    # Calculate average power per category (MW)
    data = {}
    data['SOEC'] = df.get('P_soec_actual', df.get('P_soec', pd.Series([0]))).mean()
    data['PEM'] = df.get('P_pem', pd.Series([0])).mean()
    data['Compressors'] = df.get('compressor_power_kw', pd.Series([0])).mean() / 1000
    data['Chillers'] = df.get('Chiller_1_cooling_load_kw', pd.Series([0])).mean() / 1000 # Wait, load != power. Need electrical power (W = Q/COP)
    # Correct logic for Chillers/DryCoolers if electrical power columns exist
    # Since we might not have exact columns, we stick to what was there but add warnings
    
    # Try to find auxiliary power columns
    aux_keyword = 'power_kw'
    for col in df.columns:
        if aux_keyword in col and 'compressor' not in col and 'soec' not in col.replace('P_soec', '') and 'pem' not in col.replace('P_pem', ''):
             # Add other aux loads (e.g. pumps)
             name = col.replace('_power_kw', '').replace('_', ' ').title()
             data[name] = df[col].mean() / 1000

    # Sort and Filter
    total_power = sum(data.values())
    if total_power < 0.001:
        ax.text(0.5, 0.5, 'No power data', ha='center')
        return fig

    # Sort descending
    sorted_items = sorted(data.items(), key=lambda x: x[1], reverse=True)
    
    names = []
    values = []
    
    # Bucketize < 1% if strictly needed, but bar chart handles small bars better than pie slices
    # We will show all, but maybe group very small ones if too many
    
    for k, v in sorted_items:
        if v > 0:
            names.append(k)
            values.append(v)
            
    # Horizontal Bar
    y_pos = np.arange(len(names))
    ax.barh(y_pos, values, align='center', color='steelblue', alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()  # Labels read top-to-bottom
    ax.set_xlabel('Average Power (MW)')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='x', linestyle='--', alpha=0.5)

    # Annotate percentages
    for i, v in enumerate(values):
        pct = (v / total_power) * 100
        label_text = f"{v:.2f} MW ({pct:.1f}%)"
        ax.text(v, i, f" {label_text}", va='center', fontweight='bold', fontsize=9)
        
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


def plot_effective_ppa(df: pd.DataFrame, component_ids: list, title: str, config: dict) -> Figure:
    """
    Generates Effective PPA Price chart showing dual pricing dynamics.
    
    Shows:
    - Effective (weighted average) PPA price over time
    - Spot price overlay for arbitrage comparison
    - Contract and variable price reference lines
    
    Args:
        df: Simulation history DataFrame.
        component_ids: Ignored for system-level data.
        title: Plot title.
        config: Additional plot configuration.
        
    Returns:
        matplotlib Figure object.
    """
    fig = Figure(figsize=(12, 6), constrained_layout=True)
    ax = fig.add_subplot(111)
    
    # Check for required data
    if 'ppa_price_effective_eur_mwh' not in df.columns:
        ax.text(0.5, 0.5, 'No effective PPA data\n(ppa_price_effective_eur_mwh column not found)', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        return fig
    
    # Time axis
    if 'minute' in df.columns:
        x = df['minute'] / 60.0  # Hours
        x_label = 'Time (hours)'
    else:
        x = df.index
        x_label = 'Timestep'
    
    # Downsample for performance
    limit = 2000
    stride = max(1, len(df) // limit)
    x_ds = x.iloc[::stride]
    
    # Effective PPA
    ppa_eff = df['ppa_price_effective_eur_mwh'].iloc[::stride]
    ax.plot(x_ds, ppa_eff, color='#1976D2', linewidth=2, label='Effective PPA')
    ax.fill_between(x_ds, 0, ppa_eff, color='#1976D2', alpha=0.2)
    
    # Spot price overlay
    spot_col = None
    for col in ['Spot', 'spot_price']:
        if col in df.columns:
            spot_col = col
            break
    if spot_col:
        spot_ds = df[spot_col].iloc[::stride]
        ax.plot(x_ds, spot_ds, color='#FF5722', linewidth=1.5, linestyle='--', 
                label='Spot Price', alpha=0.8)
    
    # Reference lines (from config if available)
    ppa_contract = config.get('ppa_contract_price', 80.0)
    ppa_variable = config.get('ppa_variable_price', 55.0)
    
    ax.axhline(y=ppa_contract, color='#D32F2F', linestyle=':', linewidth=1.5, 
               label=f'Contract ({ppa_contract:.0f} €/MWh)')
    ax.axhline(y=ppa_variable, color='#388E3C', linestyle=':', linewidth=1.5, 
               label=f'Variable ({ppa_variable:.0f} €/MWh)')
    
    # Statistics annotation
    avg_ppa = df['ppa_price_effective_eur_mwh'].mean()
    min_ppa = df['ppa_price_effective_eur_mwh'].min()
    max_ppa = df['ppa_price_effective_eur_mwh'].max()
    
    stats_text = f'Avg: {avg_ppa:.2f} €/MWh\nMin: {min_ppa:.2f} | Max: {max_ppa:.2f}'
    ax.annotate(stats_text, xy=(0.98, 0.98), xycoords='axes fraction',
               ha='right', va='top', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(x_label)
    ax.set_ylabel('Price (EUR/MWh)')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    return fig

