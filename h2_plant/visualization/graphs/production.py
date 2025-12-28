"""
Production Graph Module.
Generates H2/O2/Water production time series, stacked, and cumulative charts.
"""
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
import logging

logger = logging.getLogger(__name__)


def plot_time_series(df: pd.DataFrame, component_ids: list, title: str, config: dict) -> Figure:
    """Plots simple time series for specified components and variable."""
    variable = config.get('variable', 'h2_production')
    fig = Figure(figsize=(12, 6), constrained_layout=True)
    ax = fig.add_subplot(111)
    
    # Map friendly variable names to dataframe columns
    col_map = {
        'h2_production': ['h2_production_kg_h', 'H2_kg', 'H2_soec_kg', 'H2_pem_kg'],
        'o2_production': ['o2_production_kg_h', 'O2_kg', 'O2_pem_kg'],
        'water_consumption': ['water_consumption_kg_h', 'H2O_in_kg', 'steam_soec_kg']
    }
    candidates = col_map.get(variable, [])

    x = df['minute'] / 60.0 if 'minute' in df.columns else df.index
    has_data = False

    for comp_id in component_ids:
        col_name = None
        for cand in candidates:
            if f"{comp_id}_{cand}" in df.columns:
                col_name = f"{comp_id}_{cand}"
            elif cand in df.columns and len(component_ids) == 1:
                col_name = cand
            
            # Special case for raw history arrays
            if not col_name and comp_id == "PEM" and "H2_pem_kg" in df.columns:
                col_name = "H2_pem_kg"
            if not col_name and "SOEC" in comp_id and "H2_soec_kg" in df.columns:
                col_name = "H2_soec_kg"
            
            if col_name:
                break
        
        if col_name and col_name in df.columns:
            ax.plot(x, df[col_name], label=comp_id, linewidth=1.5)
            has_data = True

    if not has_data:
        ax.text(0.5, 0.5, f'No data found for {variable}', ha='center', va='center', transform=ax.transAxes)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel(variable.replace('_', ' ').title())
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    return fig


def plot_stacked(df: pd.DataFrame, component_ids: list, title: str, config: dict) -> Figure:
    """Plots stacked area chart for total production."""
    variable = config.get('variable', 'h2_production')
    fig = Figure(figsize=(12, 6), constrained_layout=True)
    ax = fig.add_subplot(111)
    
    x = df['minute'] / 60.0 if 'minute' in df.columns else df.index
    stack_data = []
    labels = []
    
    for comp_id in component_ids:
        val = None
        if variable == 'h2_production':
            if "PEM" in comp_id:
                val = df.get('H2_pem_kg')
            elif "SOEC" in comp_id:
                val = df.get('H2_soec_kg')
        elif variable == 'o2_production':
            if "PEM" in comp_id:
                val = df.get('O2_pem_kg')
            elif "SOEC" in comp_id:
                val = df.get('O2_soec_kg')
        elif variable == 'water_consumption':
            if "PEM" in comp_id:
                val = df.get('H2O_pem_kg')
            elif "SOEC" in comp_id:
                val = df.get('steam_soec_kg')

        if val is not None and hasattr(val, 'values'):
            stack_data.append(val.values)
            labels.append(comp_id)

    if stack_data:
        ax.stackplot(x, stack_data, labels=labels, alpha=0.7)
        ax.legend(loc='upper left')
        ax.set_ylabel("Rate (kg/step)")
    else:
        ax.text(0.5, 0.5, f'No data for {variable}', ha='center', va='center', transform=ax.transAxes)
        
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Time (hours)")
    ax.grid(True, alpha=0.3)
    return fig


def plot_cumulative(df: pd.DataFrame, component_ids: list, title: str, config: dict) -> Figure:
    """Plots cumulative production."""
    fig = Figure(figsize=(12, 6), constrained_layout=True)
    ax = fig.add_subplot(111)
    x = df['minute'] / 60.0 if 'minute' in df.columns else df.index
    
    if 'cumulative_h2_kg' in df.columns:
        ax.plot(x, df['cumulative_h2_kg'], label='Total System', color='black', linewidth=2)
        ax.fill_between(x, 0, df['cumulative_h2_kg'], alpha=0.1)
    else:
        # Calculate cumulative from individual components
        total = np.zeros(len(df))
        for comp_id in component_ids:
            if "PEM" in comp_id and 'H2_pem_kg' in df.columns:
                total = total + df['H2_pem_kg'].cumsum().values
            elif "SOEC" in comp_id and 'H2_soec_kg' in df.columns:
                total = total + df['H2_soec_kg'].cumsum().values
        ax.plot(x, total, label='Calculated Total')
        ax.fill_between(x, 0, total, alpha=0.1)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Cumulative Production (kg)")
    ax.legend()
    ax.grid(True)
    return fig
