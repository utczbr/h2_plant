"""
Storage Graph Module.
Generates tank level and compressor power charts.
"""
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
import logging

logger = logging.getLogger(__name__)


def plot_tank_levels(df: pd.DataFrame, component_ids: list, title: str, config: dict) -> Figure:
    """Plots H2 tank levels over time with optional pressure overlay."""
    fig = Figure(figsize=(12, 6), constrained_layout=True)
    ax = fig.add_subplot(111)
    x = df['minute'] / 60.0 if 'minute' in df.columns else df.index
    
    has_data = False
    
    if 'tank_level_kg' in df.columns:
        ax.plot(x, df['tank_level_kg'], label='Main Tank Level', color='tab:blue', linewidth=2)
        has_data = True
        
        if 'tank_pressure_bar' in df.columns:
            ax2 = ax.twinx()
            ax2.plot(x, df['tank_pressure_bar'], color='tab:red', linestyle='--', alpha=0.6, label='Pressure')
            ax2.set_ylabel("Pressure (bar)", color='tab:red')
            ax2.tick_params(axis='y', labelcolor='tab:red')

    # Look for component-specific columns
    for comp_id in component_ids:
        col = f"{comp_id}_level_kg"
        if col in df.columns:
            ax.plot(x, df[col], label=comp_id.replace('_', ' '), linewidth=1.5)
            has_data = True

    if not has_data:
        ax.text(0.5, 0.5, 'No tank level data', ha='center', va='center', transform=ax.transAxes)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Level (kg)")
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    return fig


def plot_compressor_power(df: pd.DataFrame, component_ids: list, title: str, config: dict) -> Figure:
    """Plots compressor power consumption stacked or individual."""
    fig = Figure(figsize=(12, 6), constrained_layout=True)
    ax = fig.add_subplot(111)
    x = df['minute'] / 60.0 if 'minute' in df.columns else df.index
    
    has_data = False
    stack_data = []
    labels = []
    
    # Collect breakdown data
    for comp_id in component_ids:
        # Try multiple suffixes
        for suffix in ['power_kw', 'shaft_power_kw', 'electric_power_kw']:
            col = f"{comp_id}_{suffix}"
            if col in df.columns:
                stack_data.append(df[col].values)
                # Pretty label: Compressor_S1 -> S1
                label = comp_id.replace('Compressor_', '').replace('_', ' ')
                labels.append(label)
                has_data = True
                break
    
    # Plot Logic
    if stack_data:
        # We have breakdown data - use stackplot
        ax.stackplot(x, stack_data, labels=labels, alpha=0.8)
        
        # Add Total as a line for reference
        if 'compressor_power_kw' in df.columns:
            ax.plot(x, df['compressor_power_kw'], color='black', linestyle='--', linewidth=1.5, label='Total')
        
        # Calculate approximate energy for title
        total_kwh = sum([np.sum(arr) for arr in stack_data]) * (df['minute'].diff().mean() / 60.0)
        title += f" (Total: {total_kwh/1000:.2f} MWh)"

    elif 'compressor_power_kw' in df.columns:
        # Only total available - use area fill
        ax.fill_between(x, 0, df['compressor_power_kw'], color='tab:gray', alpha=0.6, label='Total')
        has_data = True
        
    if has_data:
        ax.legend(loc='upper left', fontsize=8, ncol=2)
    else:
        ax.text(0.5, 0.5, 'No compressor power data', ha='center', va='center', transform=ax.transAxes)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Power (kW)")
    ax.grid(True, alpha=0.3)
    return fig
