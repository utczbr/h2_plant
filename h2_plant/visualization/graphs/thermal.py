"""
Thermal Graph Module.
Generates thermal load breakdown charts.
"""
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
import logging

import logging
from h2_plant.visualization import utils

logger = logging.getLogger(__name__)


def plot_load_breakdown(df: pd.DataFrame, component_ids: list, title: str, config: dict) -> Figure:
    """
    Generates a bar chart of average thermal loads for specified components.
    
    Args:
        df: Simulation history DataFrame.
        component_ids: List of component IDs to include.
        title: Plot title.
        config: Additional plot configuration.
        
    Returns:
        matplotlib Figure object.
    """
    sensible_values = []
    latent_values = []
    comp_names = []
    
    for comp_id in component_ids:
        total_load = 0.0
        latent_load = 0.0
        
        # 1. Determine Total Load (as before)
        for suffix in ['cooling_load_kw', 'heat_rejected_kw', 'heat_removed_kw', 'tqc_duty_kw', 'dc_duty_kw', 'q_transferred_kw']:
            col = f"{comp_id}_{suffix}"
            if col in df.columns:
                val = df[col].mean()
                if val > 0: 
                    total_load = val
                    break
        
        # 2. Determine Latent Load (new)
        latent_col = f"{comp_id}_latent_heat_kw"
        if latent_col in df.columns:
             latent_val = df[latent_col].mean()
             if latent_val > 0:
                 latent_load = latent_val
        
        # 3. Calculate Sensible Load
        # Ensure non-negative sensible load (Total should be >= Latent)
        sensible_load = max(0.0, total_load - latent_load)
        
        comp_names.append(comp_id)
        sensible_values.append(sensible_load)
        latent_values.append(latent_load)
        
    fig = Figure(figsize=(12, 6), constrained_layout=True)
    ax = fig.add_subplot(111)
    
    indices = range(len(comp_names))
    width = 0.6
    
    # Plot Stacked Bars
    p1 = ax.bar(indices, sensible_values, width, label='Sensible Heat', color='#87CEFA', edgecolor='black', alpha=0.9)
    p2 = ax.bar(indices, latent_values, width, bottom=sensible_values, label='Latent Heat', color='#FFA500', edgecolor='black', alpha=0.9)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel("Average Thermal Load (kW)")
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Annotate Total Values
    total_values = [s + l for s, l in zip(sensible_values, latent_values)]
    for i, total in enumerate(total_values):
        if total > 0:
            ax.annotate(f'{total:.1f}', xy=(i, total),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xticks(indices)
    ax.set_xticklabels(comp_names, rotation=30, ha='right')
    ax.legend()
    
    return fig


def plot_central_cooling_performance(df: pd.DataFrame, component_ids: list, title: str, config: dict) -> Figure:
    """
    Generates a 2-panel plot for Central Cooling performance (Glycol Loop + Cooling Water Loop).
    
    Args:
        df: Simulation history DataFrame.
        component_ids: Ignored (uses fixed CoolingManager columns).
        title: Plot title.
        config: Additional plot configuration.
        
    Returns:
        matplotlib Figure object.
    """
    fig = Figure(figsize=(12, 10), constrained_layout=True)
    ax1, ax2 = fig.subplots(2, 1, sharex=True)
    
    # Extract data
    df_ds = utils.downsample_dataframe(df, max_points=2000)
    hours_axis = utils.get_time_axis_hours(df_ds)
    x_label = "Simulation Time [Hours]"
    
    # Cooling Manager Data
    glycol_temp = df_ds.get('cooling_manager_glycol_supply_temp_c', np.zeros_like(hours_axis))
    glycol_duty = df_ds.get('cooling_manager_glycol_duty_kw', np.zeros_like(hours_axis))
    cw_temp = df_ds.get('cooling_manager_cw_supply_temp_c', np.zeros_like(hours_axis))
    cw_duty = df_ds.get('cooling_manager_cw_duty_kw', np.zeros_like(hours_axis))
    
    # --- Plot 1: Glycol Loop (Dry Cooler Bank) ---
    ax1.set_title(f"{title} - System 1: Central Glycol Loop (Dry Cooler Bank)", fontsize=12)
    
    # Left axis: Duty
    ln1 = ax1.plot(hours_axis, glycol_duty, 'b-', label='Total Glycol Duty (kW)', alpha=0.8)
    ax1.set_ylabel("Heat Load [kW]", color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)
    
    # Right axis: Temperature
    ax1r = ax1.twinx()
    ln2 = ax1r.plot(hours_axis, glycol_temp, 'r-', label='Supply Temp (°C)', linewidth=2)
    ax1r.set_ylabel("Glycol Supply Temp [°C]", color='r')
    ax1r.tick_params(axis='y', labelcolor='r')
    
    # Combined legend
    lns1 = ln1 + ln2
    labs1 = [l.get_label() for l in lns1]
    ax1.legend(lns1, labs1, loc='upper left')
    
    # --- Plot 2: Cooling Water Loop (Cooling Tower) ---
    ax2.set_title(f"{title} - System 2: Central Cooling Water Loop (Cooling Tower)", fontsize=12)
    
    # Left axis: Duty
    ln3 = ax2.plot(hours_axis, cw_duty, 'g-', label='Total CW Duty (kW)', alpha=0.8)
    ax2.set_ylabel("Heat Load [kW]", color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    ax2.grid(True, alpha=0.3)
    
    # Right axis: Temperature
    ax2r = ax2.twinx()
    ln4 = ax2r.plot(hours_axis, cw_temp, 'm-', label='Supply Temp (°C)', linewidth=2)
    ax2r.set_ylabel("CW Supply Temp [°C]", color='m')
    ax2r.tick_params(axis='y', labelcolor='m')
    
    # Combined legend
    lns2 = ln3 + ln4
    labs2 = [l.get_label() for l in lns2]
    ax2.legend(lns2, labs2, loc='upper left')
    
    ax2.set_xlabel(x_label)
    
    return fig


def plot_thermal_time_series(df: pd.DataFrame, component_ids: list, title: str, config: dict) -> Figure:
    """
    Plots thermal load (kW) over time for specified components.
    
    Searches for standard thermal columns (cooling_load, heat_rejected, etc.)
    """
    variable = config.get('variable', 'thermal_load')
    fig = Figure(figsize=(12, 6), constrained_layout=True)
    ax = fig.add_subplot(111)
    
    df_ds = utils.downsample_dataframe(df, max_points=2000)
    x = utils.get_time_axis_hours(df_ds)
    has_data = False
    
    # Define suffixes based on variable
    if variable == 'outlet_temperature':
         suffixes = ['outlet_temp_c', 'outlet_temperature_c', 'temp_c', 'temperature_c']
         y_label = "Outlet Temperature (°C)"
    else:
         # Default to thermal load
         suffixes = ['cooling_load_kw', 'heat_rejected_kw', 'heat_removed_kw', 'tqc_duty_kw', 'dc_duty_kw', 'duty_kw', 'q_transferred_kw']
         y_label = "Thermal Load (kW)"

    for comp_id in component_ids:
        col_name = None
        for suffix in suffixes:
            candidate = f"{comp_id}_{suffix}"
            if candidate in df.columns:
                col_name = candidate
                break
        
        if col_name:
            # Re-fetch column from downsampled dataframe by name
            if col_name in df_ds.columns:
                ax.plot(x, df_ds[col_name], label=comp_id, linewidth=1.5)
                has_data = True
            
    if not has_data:
        ax.text(0.5, 0.5, f'No {variable} data found for specified components', 
                ha='center', va='center', transform=ax.transAxes)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel(y_label)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    return fig
