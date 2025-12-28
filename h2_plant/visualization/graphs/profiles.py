"""
Profile Graph Module.
Generates T/P/Flow/Composition stacked panel plots for ordered component lists.
"""
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
import logging

logger = logging.getLogger(__name__)


def plot_profile(df: pd.DataFrame, component_ids: list, title: str, config: dict) -> Figure:
    """
    Generates a T/P/Flow/H2O profile for a specific ordered list of components.
    
    Args:
        df: Simulation history DataFrame (time-series).
        component_ids: Ordered list of component IDs to include.
        title: Plot title.
        config: Additional plot configuration (unused currently).
        
    Returns:
        matplotlib Figure object.
    """
    if not component_ids:
        logger.warning(f"[{title}] No components provided for profile plot.")
        return None

    # 1. Data Extraction (Snapshot: Time Average)
    profile_data = []
    
    for comp_id in component_ids:
        temp = _get_mean(df, comp_id, ['outlet_temp_c', 'temperature_c', 'temp_c', 'T_c'])
        if temp == 0:
            temp_k = _get_mean(df, comp_id, ['outlet_temp_k', 'temperature_k', 'temp_k'])
            if temp_k > 0:
                temp = temp_k - 273.15
                
        press = _get_mean(df, comp_id, ['outlet_pressure_bar', 'pressure_bar', 'P_bar'])
        if press == 0:
            press_pa = _get_mean(df, comp_id, ['outlet_pressure_pa', 'pressure_pa'])
            if press_pa > 0:
                press = press_pa / 1e5
                
        flow = _get_mean(df, comp_id, ['outlet_mass_flow_kg_h', 'mass_flow_kg_h'])
        
        # H2O Content
        h2o_frac = _get_mean(df, comp_id, ['mass_fraction_h2o', 'y_h2o', 'w_h2o']) * 100.0
        if h2o_frac == 0:
            h2o_ppm = _get_mean(df, comp_id, ['h2o_ppm'])
            if h2o_ppm > 0:
                h2o_frac = h2o_ppm / 10000.0

        profile_data.append({
            'Component': comp_id,
            'Temperature': temp,
            'Pressure': press,
            'Flow': flow,
            'H2O_pct': h2o_frac
        })
    
    plot_df = pd.DataFrame(profile_data)
    
    # 2. Plotting (3 Panels)
    fig = Figure(figsize=(14, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, hspace=0.1, height_ratios=[1, 1, 1])
    
    x = range(len(component_ids))
    
    # Ax1: Temperature (Red)
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(x, plot_df['Temperature'], 'o-', color='tab:red', linewidth=2)
    ax1.set_ylabel('Temperature (°C)', color='tab:red', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticklabels([])
    ax1.set_title(title, fontsize=14, fontweight='bold')
    
    for i, v in enumerate(plot_df['Temperature']):
        if pd.notnull(v) and abs(v) > 0.1:
            ax1.annotate(f"{v:.1f}", (i, v), xytext=(0, 5), textcoords='offset points', ha='center', fontsize=8)

    # Ax2: Pressure (Blue)
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(x, plot_df['Pressure'], 's-', color='tab:blue', linewidth=2)
    ax2.set_ylabel('Pressure (bar)', color='tab:blue', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticklabels([])

    for i, v in enumerate(plot_df['Pressure']):
        if pd.notnull(v) and abs(v) > 0.01:
            ax2.annotate(f"{v:.1f}", (i, v), xytext=(0, 5), textcoords='offset points', ha='center', fontsize=8)

    # Ax3: Flow & H2O (Green Bar + Purple Line)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    
    bars = ax3.bar(x, plot_df['Flow'], color='tab:green', alpha=0.4, label='Mass Flow')
    ax3.set_ylabel('Mass Flow (kg/h)', color='tab:green', fontweight='bold')
    
    ax3_twin = ax3.twinx()
    ax3_twin.plot(x, plot_df['H2O_pct'], 'x--', color='purple', linewidth=1.5, label='H2O Content')
    ax3_twin.set_ylabel('H2O Content (%)', color='purple', fontweight='bold')
    
    if plot_df['H2O_pct'].max() > 1.0 and plot_df['H2O_pct'].min() > 0 and plot_df['H2O_pct'].min() < 0.01:
        ax3_twin.set_yscale('log')

    ax3.set_xticks(x)
    ax3.set_xticklabels(component_ids, rotation=45, ha='right')
    ax3.grid(True, axis='x', alpha=0.3)
    
    return fig


def _get_mean(df: pd.DataFrame, comp_id: str, suffixes: list) -> float:
    """Try to find a column matching comp_id + suffix and return mean."""
    for suffix in suffixes:
        col_name = f"{comp_id}_{suffix}"
        if col_name in df.columns:
            return df[col_name].mean()
    return 0.0
