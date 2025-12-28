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
