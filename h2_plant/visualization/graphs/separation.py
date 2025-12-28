"""
Separation Graph Module.
Generates water removal and separation efficiency charts.
"""
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
import logging

logger = logging.getLogger(__name__)


def plot_water_removal(df: pd.DataFrame, component_ids: list, title: str, config: dict) -> Figure:
    """
    Generates a bar chart of total water removed (kg) per component.
    
    Args:
        df: Simulation history DataFrame.
        component_ids: List of component IDs to include.
        title: Plot title.
        config: Additional plot configuration.
        
    Returns:
        matplotlib Figure object.
    """
    total_water_map = {}
    
    # Determine timestep (dt) in hours
    dt = 1.0 / 60.0  # Default: 1-minute steps
    if 'minute' in df.columns and len(df) > 1:
        dt = (df['minute'].iloc[1] - df['minute'].iloc[0]) / 60.0

    for comp_id in component_ids:
        total_kg = 0.0
        for suffix in ['water_removed_kg_h', 'drain_flow_kg_h', 'water_condensed_kg_h', 'liquid_removed_kg_h']:
            col = f"{comp_id}_{suffix}"
            if col in df.columns:
                total_kg = df[col].sum() * dt
                if total_kg > 0:
                    break
        
        total_water_map[comp_id] = total_kg

    fig = Figure(figsize=(10, 6), constrained_layout=True)
    ax = fig.add_subplot(111)
    
    names = list(total_water_map.keys())
    values = list(total_water_map.values())
    
    bars = ax.bar(names, values, color='cornflowerblue', edgecolor='black', alpha=0.8)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel("Total Water Removed (kg)")
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
