"""
Thermal Graph Module.
Generates thermal load breakdown charts.
"""
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
import logging

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
    data = {}
    for comp_id in component_ids:
        load = 0.0
        for suffix in ['cooling_load_kw', 'heat_rejected_kw', 'heat_removed_kw', 'tqc_duty_kw', 'dc_duty_kw']:
            col = f"{comp_id}_{suffix}"
            if col in df.columns:
                val = df[col].mean()
                if val > 0: 
                    load = val
                    break
        data[comp_id] = load
        
    fig = Figure(figsize=(10, 6), constrained_layout=True)
    ax = fig.add_subplot(111)
    
    names = list(data.keys())
    values = list(data.values())
    
    bars = ax.bar(names, values, color='salmon', edgecolor='black', alpha=0.8)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel("Average Thermal Load (kW)")
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    if len(names) > 5:
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha='right')
        

    
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
    if 'minute' in df.columns:
        hours_axis = df['minute'] / 60.0
        x_label = "Simulation Time [Hours]"
    else:
        hours_axis = df.index
        x_label = "Timestep"
    
    # Cooling Manager Data
    glycol_temp = df.get('cooling_manager_glycol_supply_temp_c', np.zeros_like(hours_axis))
    glycol_duty = df.get('cooling_manager_glycol_duty_kw', np.zeros_like(hours_axis))
    cw_temp = df.get('cooling_manager_cw_supply_temp_c', np.zeros_like(hours_axis))
    cw_duty = df.get('cooling_manager_cw_duty_kw', np.zeros_like(hours_axis))
    
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
