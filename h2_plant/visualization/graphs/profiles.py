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

    # 1. Data Extraction
    plot_df = _extract_profile_data(df, component_ids)
    
    # 2. Plotting (3 Panels)
    fig = Figure(figsize=(14, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, hspace=0.1, height_ratios=[1, 1, 1])
    
    x = range(len(component_ids))
    
    # Ax1: Temperature
    _plot_temperature_subplot(fig.add_subplot(gs[0]), x, plot_df, title)
    
    # Ax2: Pressure
    _plot_pressure_subplot(fig.add_subplot(gs[1], sharex=fig.axes[0]), x, plot_df)

    # Ax3: Flow & H2O
    _plot_flow_subplot(fig.add_subplot(gs[2], sharex=fig.axes[0]), x, plot_df, component_ids)
    
    return fig


def plot_temperature_profile(df: pd.DataFrame, component_ids: list, title: str, config: dict) -> Figure:
    """Generates only the Temperature profile."""
    if not component_ids: return None
    plot_df = _extract_profile_data(df, component_ids)
    
    fig = Figure(figsize=(14, 6), constrained_layout=True)
    ax = fig.add_subplot(111)
    
    x = range(len(component_ids))
    _plot_temperature_subplot(ax, x, plot_df, title)
    
    ax.set_xticks(x)
    ax.set_xticklabels(component_ids, rotation=45, ha='right')
    
    return fig


def plot_pressure_profile(df: pd.DataFrame, component_ids: list, title: str, config: dict) -> Figure:
    """Generates only the Pressure profile."""
    if not component_ids: return None
    plot_df = _extract_profile_data(df, component_ids)
    
    fig = Figure(figsize=(14, 6), constrained_layout=True)
    ax = fig.add_subplot(111)
    
    x = range(len(component_ids))
    _plot_pressure_subplot(ax, x, plot_df, title)
    
    ax.set_xticks(x)
    ax.set_xticklabels(component_ids, rotation=45, ha='right')
    
    return fig


def plot_flow_profile(df: pd.DataFrame, component_ids: list, title: str, config: dict) -> Figure:
    """Generates only the Flow & Composition profile."""
    if not component_ids: return None
    plot_df = _extract_profile_data(df, component_ids)
    
    fig = Figure(figsize=(14, 6), constrained_layout=True)
    ax = fig.add_subplot(111)
    
    x = range(len(component_ids))
    _plot_flow_subplot(ax, x, plot_df, component_ids)
    
    # Override subplot x-axis settings for standalone plot
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    return fig


def _extract_profile_data(df: pd.DataFrame, component_ids: list) -> pd.DataFrame:
    """Extracts T/P/Flow/Composition data for components."""
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
        
        h2o_frac = _get_mean(df, comp_id, ['outlet_h2o_frac', 'mass_fraction_h2o', 'y_h2o', 'w_h2o']) * 100.0
        if h2o_frac == 0:
            h2o_ppm = _get_mean(df, comp_id, ['h2o_ppm', 'outlet_h2o_ppm'])
            if h2o_ppm > 0:
                h2o_frac = h2o_ppm / 10000.0

        profile_data.append({
            'Component': comp_id,
            'Temperature': temp,
            'Pressure': press,
            'Flow': flow,
            'H2O_pct': h2o_frac
        })
    
    # DEBUG: Print mass flow for each component
    # print("\n" + "="*70)
    # print("DEBUG: H2 Main Train Mass Flow Profile")
    # print("="*70)
    # for entry in profile_data:
    #     print(f"  {entry['Component']:25s} | Flow: {entry['Flow']:8.2f} kg/h | T: {entry['Temperature']:6.1f}°C | P: {entry['Pressure']:6.2f} bar | H2O: {entry['H2O_pct']:6.3f}%")
    # print("="*70 + "\n")
    
    return pd.DataFrame(profile_data)


def _plot_temperature_subplot(ax, x, plot_df, title=None):
    ax.plot(x, plot_df['Temperature'], 'o-', color='tab:red', linewidth=2)
    ax.set_ylabel('Temperature (°C)', color='tab:red', fontweight='bold')
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    
    for i, v in enumerate(plot_df['Temperature']):
        if pd.notnull(v) and abs(v) > 0.1:
            ax.annotate(f"{v:.1f}", (i, v), xytext=(0, 5), textcoords='offset points', ha='center', fontsize=8)


def _plot_pressure_subplot(ax, x, plot_df, title=None):
    ax.plot(x, plot_df['Pressure'], 's-', color='tab:blue', linewidth=2)
    ax.set_ylabel('Pressure (bar)', color='tab:blue', fontweight='bold')
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')

    for i, v in enumerate(plot_df['Pressure']):
        if pd.notnull(v) and abs(v) > 0.01:
            ax.annotate(f"{v:.1f}", (i, v), xytext=(0, 5), textcoords='offset points', ha='center', fontsize=8)


def _plot_flow_subplot(ax, x, plot_df, component_ids):
    bars = ax.bar(x, plot_df['Flow'], color='tab:green', alpha=0.4, label='Mass Flow')
    ax.set_ylabel('Mass Flow (kg/h)', color='tab:green', fontweight='bold')
    
    ax_twin = ax.twinx()
    ax_twin.plot(x, plot_df['H2O_pct'], 'x--', color='purple', linewidth=1.5, label='H2O Content')
    ax_twin.set_ylabel('H2O Content (%)', color='purple', fontweight='bold')
    
    if plot_df['H2O_pct'].max() > 1.0 and plot_df['H2O_pct'].min() > 0 and plot_df['H2O_pct'].min() < 0.01:
        ax_twin.set_yscale('log')

    ax.set_xticks(x)
    ax.set_xticklabels(component_ids, rotation=45, ha='right')
    ax.grid(True, axis='x', alpha=0.3)


def _get_mean(df: pd.DataFrame, comp_id: str, suffixes: list) -> float:
    """Try to find a column matching comp_id + suffix and return last timestep value."""
    for suffix in suffixes:
        col_name = f"{comp_id}_{suffix}"
        if col_name in df.columns:
            series = df[col_name]
            # Use last timestep value for steady-state snapshot
            if len(series) > 0:
                return series.iloc[-1]
            return 0.0
    return 0.0
