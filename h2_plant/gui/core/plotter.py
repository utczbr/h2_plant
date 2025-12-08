"""
Interactive Matplotlib Figure Generators for Simulation Reports.

This module provides functions that create and return Matplotlib Figure objects
for embedding in the Qt GUI. No files are saved to disk.

EXTENSIBILITY:
- Each function follows the pattern: create_*_figure(df) -> Figure
- New graphs can be added by creating a new function and registering it in GRAPH_REGISTRY
- Component-specific graphs can be grouped using the folder system in SimulationReportWidget
"""

from matplotlib.figure import Figure
import pandas as pd
import numpy as np
from typing import Dict, Any, Callable, Optional

# ==============================================================================
# COLOR PALETTE
# ==============================================================================
COLORS = {
    'offer': 'black',
    'soec': '#4CAF50',    # Green
    'pem': '#2196F3',     # Blue
    'sold': '#FF9800',    # Orange
    'price': 'black',
    'ppa': 'red',
    'limit': 'blue',
    'h2_total': 'black',
    'water_total': 'navy',
    'oxygen': 'purple',
    'compressor': '#9C27B0',  # Purple
    'tank': '#00BCD4',    # Cyan
    'pump': '#795548',    # Brown
    'chiller': '#00BCD4',  # Cyan
    'coalescer': '#8BC34A',  # Light Green
    'kod': '#42A5F5',        # Light Blue (Knock-Out Drum)
}

# ==============================================================================
# PERFORMANCE CONSTANTS
# ==============================================================================
DPI_FAST = 72   # Fast initial rendering for quick display
DPI_HIGH = 100  # High quality for focused viewing

def downsample_for_plot(data, max_points: int = 500):
    """
    Downsample data for faster plotting while preserving visual shape.
    
    Uses stride-based downsampling which is fast and maintains the overall
    trend of the data. For datasets with <= max_points, returns unchanged.
    
    Args:
        data: Array-like data (pandas Series, numpy array, or list)
        max_points: Maximum number of points to keep (default: 500)
        
    Returns:
        Downsampled data of the same type as input
    """
    if len(data) <= max_points:
        return data
    stride = max(1, len(data) // max_points)
    return data[::stride]

# ==============================================================================
# DATA NORMALIZATION
# ==============================================================================
def normalize_history(history: Dict[str, Any]) -> pd.DataFrame:
    """
    Normalizes the history dictionary to ensure consistent keys for plotting.
    Maps backend keys to those expected by the plotting logic.
    """
    df = pd.DataFrame(history)
    
    # Power
    if 'P_soec_actual' in df.columns and 'P_soec' not in df.columns:
        df['P_soec'] = df['P_soec_actual']
    if 'P_soec' not in df.columns: df['P_soec'] = 0.0
    if 'P_pem' not in df.columns: df['P_pem'] = 0.0
    if 'P_sold' not in df.columns: df['P_sold'] = 0.0
    if 'P_offer' not in df.columns: df['P_offer'] = 0.0
    
    # Prices
    if 'spot_price' in df.columns and 'Spot' not in df.columns:
        df['Spot'] = df['spot_price']
    if 'Spot' not in df.columns: df['Spot'] = 0.0
    
    # Hydrogen
    if 'H2_soec_kg' in df.columns and 'H2_soec' not in df.columns:
        df['H2_soec'] = df['H2_soec_kg']
    if 'H2_soec' not in df.columns: df['H2_soec'] = 0.0
    if 'H2_pem_kg' in df.columns and 'H2_pem' not in df.columns:
        df['H2_pem'] = df['H2_pem_kg']
    if 'H2_pem' not in df.columns: df['H2_pem'] = 0.0
    
    # Water
    if 'steam_soec_kg' in df.columns and 'Steam_soec' not in df.columns:
        df['Steam_soec'] = df['steam_soec_kg']
    if 'Steam_soec' not in df.columns: df['Steam_soec'] = 0.0
    if 'H2O_pem_kg' in df.columns and 'H2O_pem' not in df.columns:
        df['H2O_pem'] = df['H2O_pem_kg']
    if 'H2O_pem' not in df.columns: df['H2O_pem'] = 0.0
    
    # Time: Each row is 1 minute, so minute equals index
    if 'minute' not in df.columns:
        df['minute'] = df.index
    
    # Preserve module powers as metadata (2D array not suitable for DataFrame columns)
    # We store it as a private attribute that can be accessed via df.attrs
    if 'soec_module_powers' in history:
        df.attrs['soec_module_powers'] = history['soec_module_powers']
        
    return df

# ==============================================================================
# FIGURE CREATION FUNCTIONS
# ==============================================================================

def create_dispatch_figure(df: pd.DataFrame, dpi: int = DPI_FAST) -> Figure:
    """Create power dispatch stacked area chart."""
    fig = Figure(figsize=(10, 5), dpi=dpi, constrained_layout=True)
    ax = fig.add_subplot(111)
    
    # Downsample for performance
    minutes = downsample_for_plot(df['minute'])
    P_soec = downsample_for_plot(df['P_soec'])
    P_pem = downsample_for_plot(df['P_pem'])
    P_sold = downsample_for_plot(df['P_sold'])
    P_offer = downsample_for_plot(df['P_offer'])
    
    ax.fill_between(minutes, 0, P_soec, label='SOEC Consumption', color=COLORS['soec'], alpha=0.6)
    ax.fill_between(minutes, P_soec, P_soec + P_pem, label='PEM Consumption', color=COLORS['pem'], alpha=0.6)
    ax.fill_between(minutes, P_soec + P_pem, P_soec + P_pem + P_sold, label='Grid Sale', color=COLORS['sold'], alpha=0.6)
    ax.plot(minutes, P_offer, label='Offered Power', color=COLORS['offer'], linestyle='--', linewidth=1.5)
    
    ax.set_title('Hybrid Dispatch: SOEC + PEM + Arbitrage', fontsize=12)
    ax.set_xlabel('Time (Minutes)')
    ax.set_ylabel('Power (MW)')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    return fig


def create_arbitrage_figure(df: pd.DataFrame, dpi: int = DPI_FAST) -> Figure:
    """Create price scenario chart."""
    fig = Figure(figsize=(10, 5), dpi=dpi, constrained_layout=True)
    ax = fig.add_subplot(111)
    
    minutes = downsample_for_plot(df['minute'])
    spot_price = downsample_for_plot(df['Spot'])
    
    PPA_PRICE = 50.0
    H2_EQUIV_PRICE = 192.0  # ~9.6 EUR/kg / 0.05 MWh/kg
    
    ax.plot(minutes, spot_price, label='Spot Price (EUR/MWh)', color=COLORS['price'])
    ax.axhline(y=PPA_PRICE, color=COLORS['ppa'], linestyle='--', label='PPA Price')
    ax.axhline(y=H2_EQUIV_PRICE, color='green', linestyle='-.', label=f'H2 Breakeven (~{H2_EQUIV_PRICE:.0f})')
    
    if 'sell_decision' in df.columns:
        sell_idx = df.index[df['sell_decision'] == 1].tolist()
        if sell_idx and len(sell_idx) > 0:
            # Use original data for scatter points (not downsampled)
            ax.scatter(df['minute'].iloc[sell_idx], df['Spot'].iloc[sell_idx], 
                       color='red', zorder=5, label='Sell Decision', s=20)
    
    ax.set_title('Price Scenario & H2 Opportunity Cost', fontsize=12)
    ax.set_xlabel('Time (Minutes)')
    ax.set_ylabel('Price (EUR/MWh)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return fig


def create_h2_production_figure(df: pd.DataFrame, dpi: int = DPI_FAST) -> Figure:
    """Create hydrogen production chart."""
    fig = Figure(figsize=(10, 5), dpi=dpi, constrained_layout=True)
    ax = fig.add_subplot(111)
    
    # Downsample for performance
    minutes = downsample_for_plot(df['minute'])
    H2_soec = downsample_for_plot(df['H2_soec'])
    H2_pem = downsample_for_plot(df['H2_pem'])
    H2_total = H2_soec + H2_pem
    
    ax.fill_between(minutes, 0, H2_soec, label='H2 SOEC', color=COLORS['soec'], alpha=0.5)
    ax.fill_between(minutes, H2_soec, H2_total, label='H2 PEM', color=COLORS['pem'], alpha=0.5)
    ax.plot(minutes, H2_total, color=COLORS['h2_total'], linestyle='--', label='Total H2')
    
    ax.set_title('Hydrogen Production Rate (kg/min)', fontsize=12)
    ax.set_ylabel('Production Rate (kg/min)')
    ax.set_xlabel('Time (Minutes)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return fig


def create_oxygen_figure(df: pd.DataFrame, dpi: int = DPI_FAST) -> Figure:
    """Create oxygen production chart."""
    fig = Figure(figsize=(10, 5), dpi=dpi, constrained_layout=True)
    ax = fig.add_subplot(111)
    
    # Compute O2 before downsampling
    O2_pem_full = df.get('O2_pem_kg', df['H2_pem'] * 8.0)
    O2_soec_full = df['H2_soec'] * 8.0
    O2_total_full = O2_soec_full + O2_pem_full
    
    # Downsample for performance
    minutes = downsample_for_plot(df['minute'])
    O2_soec = downsample_for_plot(O2_soec_full)
    O2_total = downsample_for_plot(O2_total_full)
    
    ax.fill_between(minutes, 0, O2_soec, label='O2 SOEC', color=COLORS['soec'], alpha=0.5)
    ax.fill_between(minutes, O2_soec, O2_total, label='O2 PEM', color=COLORS['oxygen'], alpha=0.5)
    ax.plot(minutes, O2_total, color='black', linestyle='--', label='Total O2')
    
    ax.set_title('Oxygen Co-Production (kg/min)', fontsize=12)
    ax.set_ylabel('Production Rate (kg/min)')
    ax.set_xlabel('Time (Minutes)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return fig


def create_water_figure(df: pd.DataFrame, dpi: int = DPI_FAST) -> Figure:
    """Create water consumption chart."""
    fig = Figure(figsize=(10, 5), dpi=dpi, constrained_layout=True)
    ax = fig.add_subplot(111)
    
    # Compute water before downsampling
    water_soec_full = df['Steam_soec'] * 1.10
    water_pem_full = df['H2O_pem'] * 1.02
    total_full = water_soec_full + water_pem_full
    
    # Downsample for performance
    minutes = downsample_for_plot(df['minute'])
    water_soec = downsample_for_plot(water_soec_full)
    total = downsample_for_plot(total_full)
    
    ax.fill_between(minutes, 0, water_soec, label='H2O SOEC', color=COLORS['soec'], alpha=0.5)
    ax.fill_between(minutes, water_soec, total, label='H2O PEM', color='brown', alpha=0.5)
    ax.plot(minutes, total, color=COLORS['water_total'], linestyle='--', label='Total H2O')
    
    ax.set_title('Water Consumption (Including Losses)', fontsize=12)
    ax.set_ylabel('Water Flow (kg/min)')
    ax.set_xlabel('Time (Minutes)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return fig


def create_energy_pie_figure(df: pd.DataFrame, dpi: int = DPI_FAST) -> Figure:
    """Create energy distribution pie chart."""
    fig = Figure(figsize=(8, 6), dpi=dpi, constrained_layout=True)
    ax = fig.add_subplot(111)
    
    E_soec = df['P_soec'].sum() / 60.0
    E_pem = df['P_pem'].sum() / 60.0
    E_sold = df['P_sold'].sum() / 60.0
    E_total = E_soec + E_pem + E_sold
    
    sizes = [E_soec, E_pem, E_sold]
    labels = ['SOEC', 'PEM', 'Grid Sale']
    colors = [COLORS['soec'], COLORS['pem'], COLORS['sold']]
    
    valid = [(s, l, c) for s, l, c in zip(sizes, labels, colors) if s > 0.01]
    
    if valid:
        valid_sizes, valid_labels, valid_colors = zip(*valid)
        explode = [0.05 if 'Sale' in l else 0 for l in valid_labels]
        
        wedges, texts, autotexts = ax.pie(
            valid_sizes, explode=explode, labels=valid_labels, colors=valid_colors,
            autopct='%1.1f%%', pctdistance=0.85, startangle=140,
            wedgeprops=dict(width=0.4, edgecolor='w')
        )
        
        ax.text(0, 0, f"TOTAL\n{E_total:.1f}\nMWh", ha='center', va='center', fontsize=10, fontweight='bold')
        ax.set_title('Energy Distribution', fontsize=12)
    else:
        ax.text(0.5, 0.5, 'No energy data', ha='center', va='center', transform=ax.transAxes)
    
    return fig


def create_histogram_figure(df: pd.DataFrame, dpi: int = DPI_FAST) -> Figure:
    """Create price histogram."""
    fig = Figure(figsize=(10, 5), dpi=dpi, constrained_layout=True)
    ax = fig.add_subplot(111)
    
    spot_price = df['Spot']
    ax.hist(spot_price, bins=30, color='gray', edgecolor='black', alpha=0.7)
    
    mean_price = spot_price.mean()
    max_price = spot_price.max()
    
    ax.axvline(mean_price, color='blue', linestyle='--', linewidth=2, label=f'Mean: {mean_price:.1f} EUR')
    ax.axvline(max_price, color='red', linestyle='--', linewidth=2, label=f'Max: {max_price:.1f} EUR')
    
    ax.set_title('Spot Price Distribution', fontsize=12)
    ax.set_xlabel('Price (EUR/MWh)')
    ax.set_ylabel('Frequency')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return fig


def create_dispatch_curve_figure(df: pd.DataFrame, dpi: int = DPI_FAST) -> Figure:
    """Create dispatch curve scatter plot (P vs H2)."""
    fig = Figure(figsize=(10, 5), dpi=dpi, constrained_layout=True)
    ax = fig.add_subplot(111)
    
    # Compute and downsample for performance (Higher resolution for scatter)
    P_total = downsample_for_plot(df['P_soec'] + df['P_pem'], max_points=10000)
    H2_total = downsample_for_plot(df['H2_soec'] + df['H2_pem'], max_points=10000)
    
    ax.scatter(P_total, H2_total, alpha=0.5, c='blue', edgecolors='none', s=10)
    
    ax.set_title('Dispatch Curve: H2 Production vs Power', fontsize=12)
    ax.set_xlabel('Total Power Input (MW)')
    ax.set_ylabel('H2 Production (kg/min)')
    ax.grid(True, alpha=0.3)
    return fig


# ==============================================================================
# NEW CHARTS - Cumulative, Efficiency, Revenue, Power Balance
# ==============================================================================

def create_cumulative_h2_figure(df: pd.DataFrame, dpi: int = DPI_FAST) -> Figure:
    """Create cumulative hydrogen production chart."""
    fig = Figure(figsize=(10, 5), dpi=dpi, constrained_layout=True)
    ax = fig.add_subplot(111)
    
    # Compute cumulative before downsampling to preserve final values
    H2_soec_cum_full = df['H2_soec'].cumsum()
    H2_pem_cum_full = df['H2_pem'].cumsum()
    H2_total_cum_full = H2_soec_cum_full + H2_pem_cum_full
    final_total = H2_total_cum_full.iloc[-1]
    
    # Downsample for plotting
    minutes = downsample_for_plot(df['minute'])
    H2_soec_cum = downsample_for_plot(H2_soec_cum_full)
    H2_total_cum = downsample_for_plot(H2_total_cum_full)
    
    ax.fill_between(minutes, 0, H2_soec_cum, label='SOEC (Cumulative)', 
                    color=COLORS['soec'], alpha=0.6)
    ax.fill_between(minutes, H2_soec_cum, H2_total_cum, label='PEM (Cumulative)', 
                    color=COLORS['pem'], alpha=0.6)
    ax.plot(minutes, H2_total_cum, color='black', linestyle='--', 
            label=f'Total: {final_total:.1f} kg', linewidth=1.5)
    
    ax.set_title('Cumulative Hydrogen Production', fontsize=12)
    ax.set_xlabel('Time (Minutes)')
    ax.set_ylabel('Cumulative H2 (kg)')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)
    return fig


def create_cumulative_energy_figure(df: pd.DataFrame, dpi: int = DPI_FAST) -> Figure:
    """Create cumulative energy consumption chart."""
    fig = Figure(figsize=(10, 5), dpi=dpi, constrained_layout=True)
    ax = fig.add_subplot(111)
    
    # Compute cumulative before downsampling to preserve final values
    E_soec_cum_full = (df['P_soec'] / 60.0).cumsum()
    E_pem_cum_full = (df['P_pem'] / 60.0).cumsum()
    E_sold_cum_full = (df['P_sold'] / 60.0).cumsum()
    E_total_cum_full = E_soec_cum_full + E_pem_cum_full
    final_sold = E_sold_cum_full.iloc[-1]
    final_total = E_total_cum_full.iloc[-1]
    
    # Downsample for plotting
    minutes = downsample_for_plot(df['minute'])
    E_soec_cum = downsample_for_plot(E_soec_cum_full)
    E_total_cum = downsample_for_plot(E_total_cum_full)
    E_sold_cum = downsample_for_plot(E_sold_cum_full)
    
    ax.fill_between(minutes, 0, E_soec_cum, label='SOEC Energy', 
                    color=COLORS['soec'], alpha=0.6)
    ax.fill_between(minutes, E_soec_cum, E_total_cum, label='PEM Energy', 
                    color=COLORS['pem'], alpha=0.6)
    ax.plot(minutes, E_sold_cum, color=COLORS['sold'], linestyle='--', 
            label=f'Grid Sale: {final_sold:.1f} MWh', linewidth=1.5)
    ax.plot(minutes, E_total_cum, color='black', linestyle='-', 
            label=f'Total Used: {final_total:.1f} MWh', linewidth=1.5)
    
    ax.set_title('Cumulative Energy Consumption', fontsize=12)
    ax.set_xlabel('Time (Minutes)')
    ax.set_ylabel('Energy (MWh)')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)
    return fig


def create_efficiency_curve_figure(df: pd.DataFrame, dpi: int = DPI_FAST) -> Figure:
    """Create system efficiency over time chart."""
    fig = Figure(figsize=(10, 5), dpi=dpi, constrained_layout=True)
    ax = fig.add_subplot(111)
    
    P_total = df['P_soec'] + df['P_pem']
    H2_total = df['H2_soec'] + df['H2_pem']
    
    # Efficiency = H2 energy / Input power
    # H2 LHV = 33.33 kWh/kg = 0.03333 MWh/kg
    H2_energy = H2_total * 0.03333  # MWh
    
    # Avoid division by zero
    efficiency = np.where(P_total > 0.01, (H2_energy / (P_total / 60.0)) * 100, 0)
    
    # Use rolling average for smoothing
    window = min(30, len(efficiency) // 10) if len(efficiency) > 30 else 5
    eff_smooth = pd.Series(efficiency).rolling(window=window, min_periods=1).mean()
    
    # Downsample AFTER smoothing for accurate display
    minutes = downsample_for_plot(df['minute'])
    eff_plot = downsample_for_plot(eff_smooth)
    
    ax.plot(minutes, eff_plot, color='green', linewidth=2, label='System Efficiency')
    ax.fill_between(minutes, 0, eff_plot, color='green', alpha=0.1)
    
    avg_eff = eff_smooth[eff_smooth > 0].mean() if (eff_smooth > 0).any() else 0
    ax.axhline(y=avg_eff, color='darkgreen', linestyle='--', 
               label=f'Average: {avg_eff:.1f}%', alpha=0.8)
    
    ax.set_title('System Efficiency Over Time (LHV Basis)', fontsize=12)
    ax.set_xlabel('Time (Minutes)')
    ax.set_ylabel('Efficiency (%)')
    ax.set_ylim(0, min(100, max(eff_smooth) * 1.2) if max(eff_smooth) > 0 else 100)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return fig


def create_revenue_analysis_figure(df: pd.DataFrame, dpi: int = DPI_FAST) -> Figure:
    """Create revenue analysis chart showing grid sales revenue."""
    fig = Figure(figsize=(10, 5), dpi=dpi, constrained_layout=True)
    ax = fig.add_subplot(111)
    
    # Compute cumulative values before downsampling
    spot_price = df['Spot']
    P_sold = df['P_sold']
    revenue_per_min = (P_sold * spot_price) / 60.0
    cumulative_revenue_full = revenue_per_min.cumsum()
    
    H2_total = df['H2_soec'] + df['H2_pem']
    H2_PRICE = 9.6  # EUR/kg
    h2_value_per_min = H2_total * H2_PRICE
    cumulative_h2_value_full = h2_value_per_min.cumsum()
    
    # Preserve final values for labels
    final_revenue = cumulative_revenue_full.iloc[-1]
    final_h2_value = cumulative_h2_value_full.iloc[-1]
    
    # Downsample for plotting
    minutes = downsample_for_plot(df['minute'])
    cumulative_revenue = downsample_for_plot(cumulative_revenue_full)
    cumulative_h2_value = downsample_for_plot(cumulative_h2_value_full)
    
    ax.plot(minutes, cumulative_revenue, color=COLORS['sold'], linewidth=2, 
            label=f'Grid Revenue: €{final_revenue:.0f}')
    ax.plot(minutes, cumulative_h2_value, color=COLORS['h2_total'], linewidth=2, 
            linestyle='--', label=f'H2 Value: €{final_h2_value:.0f}')
    
    ax.fill_between(minutes, 0, cumulative_revenue, color=COLORS['sold'], alpha=0.2)
    ax.fill_between(minutes, 0, cumulative_h2_value, color=COLORS['h2_total'], alpha=0.1)
    
    ax.set_title('Cumulative Revenue Analysis', fontsize=12)
    ax.set_xlabel('Time (Minutes)')
    ax.set_ylabel('Value (EUR)')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)
    return fig


def create_temporal_averages_figure(df: pd.DataFrame, dpi: int = DPI_FAST) -> Figure:
    """
    Create temporal averages chart showing hourly aggregated data.
    Shows 3 subplots: Spot Price, Power Dispatch, H2 Production.
    """
    fig = Figure(figsize=(12, 10), dpi=dpi, constrained_layout=True)
    
    # Create time-indexed DataFrame for resampling
    df_indexed = df.copy()
    start_date = "2024-01-01 00:00"
    df_indexed.index = pd.date_range(start=start_date, periods=len(df_indexed), freq='min')
    
    # Compute H2 total
    df_indexed['H2_total'] = df_indexed['H2_soec'] + df_indexed['H2_pem']
    
    # Resample to hourly averages
    try:
        df_hourly = df_indexed.resample('h').mean()
    except ValueError:
        df_hourly = df_indexed.resample('H').mean()  # Fallback for older pandas
    
    # Skip if not enough data
    if len(df_hourly) < 2:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, 'Simulation too short for hourly averages', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        return fig
    
    # === Subplot 1: Spot Price ===
    ax1 = fig.add_subplot(311)
    ax1.plot(df_hourly.index, df_hourly['Spot'], color='black', marker='.', 
             linestyle='-', linewidth=1, label='Avg Spot Price')
    ax1.set_ylabel('Price (EUR/MWh)')
    ax1.set_title('Hourly Average: Spot Market Prices', fontsize=11)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # === Subplot 2: Power Dispatch (Stacked) ===
    ax2 = fig.add_subplot(312, sharex=ax1)
    ax2.stackplot(df_hourly.index, 
                  df_hourly['P_soec'], 
                  df_hourly['P_pem'], 
                  df_hourly['P_sold'],
                  labels=['SOEC Power', 'PEM Power', 'Sold Power'],
                  colors=[COLORS['soec'], COLORS['pem'], COLORS['sold']], 
                  alpha=0.7)
    ax2.set_ylabel('Avg Power (MW)')
    ax2.set_title('Hourly Average: Power Dispatch Distribution', fontsize=11)
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # === Subplot 3: H2 Production ===
    ax3 = fig.add_subplot(313, sharex=ax1)
    ax3.plot(df_hourly.index, df_hourly['H2_total'], color=COLORS['h2_total'], 
             linewidth=2, label='Avg H2 Rate')
    ax3.fill_between(df_hourly.index, 0, df_hourly['H2_total'], 
                     color=COLORS['h2_total'], alpha=0.1)
    ax3.set_ylabel('Production (kg/min)')
    ax3.set_xlabel('Simulation Time')
    ax3.set_title('Hourly Average: Hydrogen Production Rate', fontsize=11)
    ax3.legend(loc='upper left', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    fig.suptitle('Temporal Averages (Hourly)', fontsize=14)
    return fig




# ==============================================================================
# PHYSICS-BASED CHARTS - PEM Polarization, Degradation, Compressor T-s
# ==============================================================================

def create_polarization_figure(df: pd.DataFrame, dpi: int = DPI_FAST) -> Figure:
    """
    Create Combined Polarization Curve (PEM + SOEC).
    PEM: Uses physics parameters from constants.
    SOEC: Uses theoretical high-temp parameters (800C).
    """
    fig = Figure(figsize=(12, 6), dpi=dpi, constrained_layout=True)
    
    # === SUBPLOT 1: PEM ===
    ax1 = fig.add_subplot(121)
    
    # Physics constants (PEM)
    R = 8.314
    F = 96485.33
    T = 333.15  # 60°C
    P_op = 40.0e5
    P_ref = 1.0e5
    z = 2
    alpha = 0.5
    j0 = 1.0e-6
    j_lim = 4.0
    delta_mem = 100e-4
    sigma_base = 0.1
    j_nom_pem = 2.91
    
    # Degradation table
    YEARS_TABLE = np.array([1.0, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 9.0, 10.0])
    V_STACK_TABLE = np.array([171, 172, 176, 178, 178, 180, 181, 183, 184, 187, 190, 193, 197])
    N_cells = 85
    V_CELL_TABLE = V_STACK_TABLE / N_cells
    T_OP_H_TABLE = YEARS_TABLE * 8760
    
    j_range = np.linspace(0.01, j_lim * 0.95, 200)
    
    def calculate_pem_vcell(j, U_deg=0.0):
        U_rev = 1.229 - 0.9e-3 * (T - 298.15) + (R * T) / (z * F) * np.log((P_op / P_ref)**1.5)
        eta_act = (R * T) / (alpha * z * F) * np.log(np.maximum(j, 1e-10) / j0)
        eta_ohm = j * (delta_mem / sigma_base)
        eta_conc = np.where(
            j >= j_lim * 0.99, 100.0,
            (R * T) / (z * F) * np.log(j_lim / np.maximum(j_lim - j, 1e-6))
        )
        return U_rev + eta_act + eta_ohm + eta_conc + U_deg
    
    V_bol = calculate_pem_vcell(j_range, U_deg=0.0)
    
    # EOL calc
    V_bol_nom = calculate_pem_vcell(j_nom_pem, U_deg=0.0)
    V_eol_ref = np.interp(87600, T_OP_H_TABLE, V_CELL_TABLE)
    U_deg_eol = max(0, V_eol_ref - V_bol_nom)
    V_eol = calculate_pem_vcell(j_range, U_deg=U_deg_eol)
    
    ax1.plot(j_range, V_bol, 'g--', linewidth=1.5, alpha=0.7, label='BOL (Year 0)')
    ax1.plot(j_range, V_eol, 'r--', linewidth=1.5, alpha=0.7, label='EOL (Year 10)')
    ax1.axvline(x=j_nom_pem, color='black', linestyle=':', alpha=0.5, label=f'Nominal ({j_nom_pem:.2f})')
    
    ax1.set_title('PEM Polarization (60°C)', fontsize=12)
    ax1.set_xlabel('Current Density (A/cm²)')
    ax1.set_ylabel('Cell Voltage (V)')
    ax1.set_xlim(0, j_lim)
    ax1.set_ylim(1.4, 2.4)
    ax1.legend(fontsize=8, loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # === SUBPLOT 2: SOEC (Theoretical) ===
    ax2 = fig.add_subplot(122)
    
    # Physics constants (SOEC 800C)
    T_soec = 1073.15 # 800C
    # Nernst potential at 800C with ~50% steam/H2 mix is lower, around 0.8-0.9V
    # Using standard theoretical value
    E_nernst = 0.95 
    
    # Area Specific Resistance (ASR) - Dominant factor at high temp
    # Typical SOEC ASR: 0.3 - 0.5 Ohm*cm2
    ASR_bol = 0.35
    ASR_eol = 0.60 # Degradation increases resistance
    
    j_nom_soec = 1.0 # Typical SOEC nominal (0.5 - 1.5 A/cm2)
    j_range_soec = np.linspace(0.01, 2.0, 100)
    
    # V = E_nernst + j * ASR (Linear approximation for high T)
    V_soec_bol = E_nernst + j_range_soec * ASR_bol
    V_soec_eol = E_nernst + j_range_soec * ASR_eol
    
    ax2.plot(j_range_soec, V_soec_bol, color=COLORS['soec'], linestyle='--', linewidth=1.5, label='BOL')
    ax2.plot(j_range_soec, V_soec_eol, 'r--', linewidth=1.5, label='EOL (High ASR)')
    ax2.axvline(x=j_nom_soec, color='black', linestyle=':', alpha=0.5, label=f'Nominal ({j_nom_soec} A/cm²)')
    
    ax2.set_title('SOEC Polarization (Theoretical 800°C)', fontsize=12)
    ax2.set_xlabel('Current Density (A/cm²)')
    # ax2.set_ylabel('Cell Voltage (V)') # Shared Y label usually, but ranges differ massively
    
    # PEM is 1.4-2.4V, SOEC is 0.9-1.6V. Better to keep separate scales.
    ax2.set_ylim(0.8, 1.8)
    
    ax2.legend(fontsize=8, loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle("Electrolyzer Physics: V-I Curves", fontsize=14)
    return fig


def create_physics_efficiency_figure(df: pd.DataFrame, dpi: int = DPI_FAST) -> Figure:
    """
    Create physics-based efficiency chart showing System vs Stack-only efficiency.
    Based on legacy plot_physics_efficiency, using self-contained constants.
    """
    fig = Figure(figsize=(10, 6), dpi=dpi, constrained_layout=True)
    ax = fig.add_subplot(111)
    
    # Physics constants (PEM - same as polarization)
    R = 8.314
    F = 96485.33
    T = 333.15  # 60°C
    P_op = 40.0e5
    P_ref = 1.0e5
    z = 2
    alpha = 0.5
    j0 = 1.0e-6
    j_lim = 4.0
    delta_mem = 100e-4
    sigma_base = 0.1
    j_nom = 2.91
    
    # BoP (Balance of Plant) parameters
    P_bop_fixed = 5000.0  # W fixed BoP consumption
    k_bop_var = 0.05  # 5% variable BoP
    Area_Total = 10000.0  # cm² total active area
    
    j_range = np.linspace(0.01, j_lim * 0.95, 200)
    
    # Calculate cell voltage (BOL)
    U_rev = 1.229 - 0.9e-3 * (T - 298.15) + (R * T) / (z * F) * np.log((P_op / P_ref)**1.5)
    eta_act = (R * T) / (alpha * z * F) * np.log(np.maximum(j_range, 1e-10) / j0)
    eta_ohm = j_range * (delta_mem / sigma_base)
    eta_conc = np.where(
        j_range >= j_lim * 0.99, 100.0,
        (R * T) / (z * F) * np.log(j_lim / np.maximum(j_lim - j_range, 1e-6))
    )
    V_total = U_rev + eta_act + eta_ohm + eta_conc
    
    # Power and efficiency calculations
    I_total = j_range * Area_Total  # Total current (A)
    P_stack = I_total * V_total  # Stack power (W)
    P_system = P_stack + P_bop_fixed + (k_bop_var * P_stack)  # Total system power
    
    # H2 chemical power (thermoneutral voltage = 1.254 V)
    P_hydrogen_chemical = I_total * 1.254
    
    # Efficiencies
    efficiency_system = np.divide(P_hydrogen_chemical, P_system, 
                                   out=np.zeros_like(P_system), where=P_system != 0) * 100
    efficiency_stack = (1.254 / V_total) * 100  # Stack-only efficiency
    
    # Plot
    ax.plot(j_range, efficiency_system, color='green', linewidth=2, 
            label='System Efficiency (Stack + BoP)')
    ax.plot(j_range, efficiency_stack, color='gray', linestyle='--', 
            alpha=0.6, linewidth=1.5, label='Stack-Only Efficiency')
    ax.fill_between(j_range, efficiency_system, efficiency_stack, 
                    color='red', alpha=0.1, label='BoP Losses')
    
    ax.axvline(x=j_nom, color='black', linestyle=':', alpha=0.5, 
               label=f'Nominal ({j_nom} A/cm²)')
    
    # Find and annotate peak efficiency
    peak_idx = np.argmax(efficiency_system)
    peak_j = j_range[peak_idx]
    peak_eff = efficiency_system[peak_idx]
    ax.scatter([peak_j], [peak_eff], color='green', s=80, zorder=5, marker='*')
    ax.annotate(f'Peak: {peak_eff:.1f}%', xy=(peak_j, peak_eff), 
                xytext=(peak_j + 0.3, peak_eff - 5), fontsize=9)
    
    ax.set_title('PEM Electrolyzer Efficiency: System vs Stack', fontsize=12)
    ax.set_xlabel('Current Density (A/cm²)')
    ax.set_ylabel('Efficiency (% LHV)')
    ax.set_xlim(0, j_lim)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8, loc='lower left')
    ax.grid(True, alpha=0.3)
    return fig




def create_degradation_figure(df: pd.DataFrame, dpi: int = DPI_FAST) -> Figure:
    """Create degradation projection chart showing voltage evolution over years."""
    fig = Figure(figsize=(10, 5), dpi=dpi, constrained_layout=True)
    ax = fig.add_subplot(111)
    
    # Degradation table
    YEARS_TABLE = np.array([1.0, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 9.0, 10.0])
    V_STACK_TABLE = np.array([171, 172, 176, 178, 178, 180, 181, 183, 184, 187, 190, 193, 197])
    N_cells = 85
    V_CELL_TABLE = V_STACK_TABLE / N_cells
    
    # Plot degradation curve
    ax.plot(YEARS_TABLE, V_CELL_TABLE, 'b-o', linewidth=2, markersize=6, label='Model Prediction')
    
    # Reference lines
    V_bol = V_CELL_TABLE[0]  # BOL = Year 1 value
    V_eol_threshold = 2.2  # Typical EOL threshold
    
    ax.axhline(y=V_bol, color='green', linestyle='--', alpha=0.7, label=f'BOL: {V_bol:.3f} V')
    ax.axhline(y=V_eol_threshold, color='red', linestyle='--', alpha=0.7, label=f'EOL Threshold: {V_eol_threshold} V')
    
    # Current state marker (estimate)
    t_sim_hours = len(df) / 60.0
    current_years = min(t_sim_hours / 8760, 10)
    current_V = np.interp(max(1, current_years), YEARS_TABLE, V_CELL_TABLE)
    
    ax.scatter([max(1, current_years)], [current_V], color='orange', s=150, zorder=5, 
               label=f'Current: ~{current_years:.1f} yr', marker='*')
    
    # Fill degradation region
    ax.fill_between(YEARS_TABLE, V_bol, V_CELL_TABLE, alpha=0.2, color='blue', label='Degradation')
    
    ax.set_title('PEM Cell Voltage Degradation Over Operating Time', fontsize=12)
    ax.set_xlabel('Operating Time (Years)')
    ax.set_ylabel('Cell Voltage @ Nominal Current (V)')
    ax.set_xlim(0, 11)
    ax.set_ylim(1.9, 2.4)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)
    return fig


def create_compressor_ts_figure(df: pd.DataFrame, dpi: int = DPI_FAST) -> Figure:
    """
    Create T-s diagram for hydrogen compression.
    Uses CoolProp for real thermodynamic properties with fallback to simplified model.
    """
    fig = Figure(figsize=(10, 6), dpi=dpi, constrained_layout=True)
    ax = fig.add_subplot(111)
    
    # Compressor parameters (typical H2 storage scenario)
    P_in_bar = 30.0    # Inlet pressure (from electrolyzer)
    P_out_bar = 350.0  # Storage pressure
    T_in_K = 313.15    # Inlet temperature (40°C)
    N_stages = 3       # Multi-stage compression
    eta_isen = 0.75    # Isentropic efficiency
    T_intercool_K = 313.15  # Intercooling temperature (40°C)
    
    # Try CoolProp for accurate thermodynamic properties
    use_coolprop = False
    try:
        from h2_plant.optimization.coolprop_lut import CoolPropLUT
        # Test if CoolProp works
        s_test = CoolPropLUT.PropsSI('S', 'P', P_in_bar * 1e5, 'T', T_in_K, 'Hydrogen')
        if s_test > 0:
            use_coolprop = True
    except Exception:
        pass
    
    T_points = []  # Temperature in °C
    S_points = []  # Entropy in kJ/(kg·K)
    
    if use_coolprop:
        # Real thermodynamic calculation with CoolProp
        pressure_ratios = np.power(P_out_bar / P_in_bar, 1.0 / N_stages)
        P_current = P_in_bar
        T_current = T_in_K
        
        # Initial point
        s_init = CoolPropLUT.PropsSI('S', 'P', P_current * 1e5, 'T', T_current, 'Hydrogen') / 1000
        T_points.append(T_current - 273.15)
        S_points.append(s_init)
        
        for i in range(N_stages):
            P_out_stage = P_current * pressure_ratios
            
            # Isentropic compression (ideal)
            s_in = CoolPropLUT.PropsSI('S', 'P', P_current * 1e5, 'T', T_current, 'Hydrogen')
            h_in = CoolPropLUT.PropsSI('H', 'P', P_current * 1e5, 'T', T_current, 'Hydrogen')
            T_out_isen = CoolPropLUT.PropsSI('T', 'P', P_out_stage * 1e5, 'S', s_in, 'Hydrogen')
            h_out_isen = CoolPropLUT.PropsSI('H', 'P', P_out_stage * 1e5, 'T', T_out_isen, 'Hydrogen')
            
            # Real compression (with efficiency)
            h_out_real = h_in + (h_out_isen - h_in) / eta_isen
            T_out_real = CoolPropLUT.PropsSI('T', 'P', P_out_stage * 1e5, 'H', h_out_real, 'Hydrogen')
            s_out = CoolPropLUT.PropsSI('S', 'P', P_out_stage * 1e5, 'T', T_out_real, 'Hydrogen') / 1000
            
            T_points.append(T_out_real - 273.15)
            S_points.append(s_out)
            
            if i < N_stages - 1:  # Intercooling
                s_cool = CoolPropLUT.PropsSI('S', 'P', P_out_stage * 1e5, 'T', T_intercool_K, 'Hydrogen') / 1000
                T_points.append(T_intercool_K - 273.15)
                S_points.append(s_cool)
                T_current = T_intercool_K
            else:
                T_current = T_out_real
            
            P_current = P_out_stage
        
        title_suffix = "(CoolProp)"
    else:
        # Simplified thermodynamic model (fallback)
        # Using ideal gas approximation: T2/T1 = (P2/P1)^((gamma-1)/gamma)
        gamma = 1.41  # For H2
        pressure_ratios = np.power(P_out_bar / P_in_bar, 1.0 / N_stages)
        
        T_current = T_in_K
        S_current = 0.0  # Reference entropy
        T_points.append(T_current - 273.15)
        S_points.append(S_current)
        
        # Simplified entropy change: ds = Cp * ln(T2/T1) - R * ln(P2/P1)
        R_H2 = 4.124  # kJ/(kg·K) for H2
        Cp_H2 = 14.3  # kJ/(kg·K) at ~300K
        
        for i in range(N_stages):
            # Isentropic temperature ratio
            T_out_isen = T_current * np.power(pressure_ratios, (gamma - 1) / gamma)
            # Real temperature with efficiency
            T_out_real = T_current + (T_out_isen - T_current) / eta_isen
            # Entropy change for real process
            ds = Cp_H2 * np.log(T_out_real / T_current) - R_H2 * np.log(pressure_ratios)
            S_current += ds
            
            T_points.append(T_out_real - 273.15)
            S_points.append(S_current)
            
            if i < N_stages - 1:  # Intercooling
                ds_cool = Cp_H2 * np.log(T_intercool_K / T_out_real)
                S_current += ds_cool
                T_points.append(T_intercool_K - 273.15)
                S_points.append(S_current)
                T_current = T_intercool_K
            else:
                T_current = T_out_real
        
        title_suffix = "(Ideal Gas Model)"
    
    # Plot real process
    ax.plot(S_points, T_points, 'r-', linewidth=2, marker='o', markersize=6, label='Compression Path')
    
    # Intercooling reference
    ax.axhline(y=T_intercool_K - 273.15, color='cyan', linestyle='-.', alpha=0.5, 
               label=f'Intercooling ({T_intercool_K - 273.15:.0f}°C)')
    
    # Annotations
    ax.scatter([S_points[0]], [T_points[0]], color='green', s=100, zorder=5, marker='^', label='Inlet')
    ax.scatter([S_points[-1]], [T_points[-1]], color='red', s=100, zorder=5, marker='v', label='Outlet')
    
    ax.set_title(f'Compressor T-s Diagram: {P_in_bar:.0f} → {P_out_bar:.0f} bar {title_suffix}', 
                 fontsize=12)
    ax.set_xlabel('Specific Entropy (kJ/kg·K)')
    ax.set_ylabel('Temperature (°C)')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)
    return fig


def create_module_power_figure(df: pd.DataFrame, dpi: int = DPI_FAST) -> Figure:
    """
    Create module power distribution chart for SOEC cluster.
    Uses real module data from simulation when available.
    """
    fig = Figure(figsize=(10, 6), dpi=dpi, constrained_layout=True)
    ax = fig.add_subplot(111)
    
    minutes = df['minute']
    
    # Try to get real module power data from DataFrame attrs
    module_powers = df.attrs.get('soec_module_powers', None)
    data_source = "Real Data"
    
    if module_powers is not None:
        # Real data available - convert to numpy array
        module_powers = np.array(module_powers)
        NUM_MODULES = module_powers.shape[1] if len(module_powers.shape) > 1 else 1
    else:
        # Fallback: Reconstruct from total SOEC power
        # This shows approximate distribution, not actual wear leveling
        P_soec = df['P_soec'].values
        NUM_MODULES = 6
        MAX_MODULE_MW = 0.4
        
        module_powers = np.zeros((len(P_soec), NUM_MODULES))
        for i, p_total in enumerate(P_soec):
            if p_total > 0:
                # Simple equal distribution (approximation)
                base_power = min(p_total / NUM_MODULES, MAX_MODULE_MW)
                module_powers[i] = base_power
        
        data_source = "Estimated (no module data)"
    
    MAX_MODULE_MW = 0.4  # For reference line
    
    # Create stacked area plot
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, NUM_MODULES))
    
    cumulative = np.zeros(len(minutes))
    for m in range(NUM_MODULES):
        ax.fill_between(minutes, cumulative, cumulative + module_powers[:, m], 
                       alpha=0.7, label=f'Module {m+1}', color=colors[m])
        cumulative += module_powers[:, m]
    
    # Reference line for max capacity
    ax.axhline(y=MAX_MODULE_MW * NUM_MODULES, color='black', linestyle='--', 
               alpha=0.5, label=f'Max Capacity ({MAX_MODULE_MW * NUM_MODULES:.1f} MW)')
    
    ax.set_title(f'SOEC Module Power Distribution ({data_source})', fontsize=12)
    ax.set_xlabel('Time (Minutes)')
    ax.set_ylabel('Power (MW)')
    ax.legend(fontsize=8, loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3)
    return fig


def create_module_stats_figure(df: pd.DataFrame, dpi: int = DPI_FAST) -> Figure:
    """
    Create SOEC module statistics bar chart showing wear and cycle counts.
    Based on legacy plot_modules_bars.
    """
    fig = Figure(figsize=(14, 6), dpi=dpi, constrained_layout=True)
    
    # Try to get module data from DataFrame attrs
    module_wear = df.attrs.get('soec_module_wear', None)
    cycle_counts = df.attrs.get('soec_cycle_counts', None)
    
    # Fallback: Estimate from power data if not available
    if module_wear is None or cycle_counts is None:
        P_soec = df['P_soec'].values
        NUM_MODULES = 6
        
        # Estimate total wear per module (uniform distribution)
        total_wear = np.sum(P_soec)
        module_wear = np.full(NUM_MODULES, total_wear / NUM_MODULES)
        
        # Estimate cycles (count transitions)
        cycles_estimate = max(1, np.sum(np.diff(P_soec > 0.1).astype(int) > 0))
        cycle_counts = np.full(NUM_MODULES, cycles_estimate // NUM_MODULES)
        
        data_source = "Estimated"
    else:
        module_wear = np.array(module_wear)
        cycle_counts = np.array(cycle_counts)
        NUM_MODULES = len(module_wear)
        data_source = "Simulation Data"
    
    x_base = np.arange(NUM_MODULES)
    labels = [f'Mod {i+1}' for i in range(NUM_MODULES)]
    
    wear_mean = np.mean(module_wear)
    cycle_mean = np.mean(cycle_counts)
    
    # === Subplot 1: Total Wear ===
    ax1 = fig.add_subplot(121)
    bars1 = ax1.bar(x_base, module_wear, color=COLORS['sold'], alpha=0.8, edgecolor='black')
    ax1.axhline(wear_mean, color='black', linestyle='--', linewidth=1.5, 
                label=f'Mean: {wear_mean:.1f}')
    ax1.set_title('Total Accumulated Wear (Processed Energy)', fontsize=12)
    ax1.set_ylabel('MW·min')
    ax1.set_xticks(x_base)
    ax1.set_xticklabels(labels)
    ax1.legend(fontsize=8)
    ax1.grid(axis='y', alpha=0.3)
    
    # === Subplot 2: Cycle Counts ===
    ax2 = fig.add_subplot(122)
    bars2 = ax2.bar(x_base, cycle_counts, color=COLORS['pem'], alpha=0.8, edgecolor='black')
    ax2.axhline(cycle_mean, color='black', linestyle='--', linewidth=1.5, 
                label=f'Mean: {cycle_mean:.1f}')
    ax2.set_title('Startup Cycle Count (Hot Standby → Ramp Up)', fontsize=12)
    ax2.set_ylabel('Cycles (#)')
    ax2.set_xticks(x_base)
    ax2.set_xticklabels(labels)
    ax2.legend(fontsize=8)
    ax2.grid(axis='y', alpha=0.3)
    
    fig.suptitle(f'SOEC Module Status ({NUM_MODULES} Units) - {data_source}', fontsize=14)
    return fig


# Need to import plt for colors
import matplotlib.pyplot as plt


# ==============================================================================
# THERMAL & SEPARATION CHARTS - Chiller & Coalescer
# ==============================================================================

def create_chiller_cooling_figure(df: pd.DataFrame, dpi: int = DPI_FAST) -> Figure:
    """
    Create chiller cooling load chart.
    Shows cooling duty and electrical consumption over time.
    """
    fig = Figure(figsize=(10, 6), dpi=dpi, constrained_layout=True)
    ax = fig.add_subplot(111)
    
    minutes = df['minute']
    
    # Try to get chiller data from simulation (may be stored in attrs or columns)
    cooling_load = df.get('chiller_cooling_kw', None)
    electrical_power = df.get('chiller_electrical_kw', None)
    
    if cooling_load is None:
        # Estimate from total power (fallback for demo)
        P_total = df['P_soec'] + df['P_pem']
        # Assume ~2% of total power goes to cooling
        cooling_load = P_total * 0.02 * 1000  # Convert MW to kW
        electrical_power = cooling_load / 4.0  # COP = 4.0
        data_source = "Estimated"
    else:
        data_source = "Simulation Data"
    
    ax.plot(minutes, cooling_load, color=COLORS['chiller'], linewidth=2, 
            label='Cooling Load (kW)')
    ax.fill_between(minutes, 0, cooling_load, color=COLORS['chiller'], alpha=0.2)
    
    if electrical_power is not None:
        ax.plot(minutes, electrical_power, color='red', linewidth=1.5, linestyle='--',
                label='Electrical Consumption (kW)')
    
    # Add average line
    avg_cooling = cooling_load.mean() if hasattr(cooling_load, 'mean') else np.mean(cooling_load)
    ax.axhline(y=avg_cooling, color='black', linestyle=':', alpha=0.5, 
               label=f'Average: {avg_cooling:.1f} kW')
    
    ax.set_title(f'Chiller Cooling Performance ({data_source})', fontsize=12)
    ax.set_xlabel('Time (Minutes)')
    ax.set_ylabel('Power (kW)')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    return fig


def create_coalescer_separation_figure(df: pd.DataFrame, dpi: int = DPI_FAST) -> Figure:
    """
    Create coalescer separation performance chart.
    Shows pressure drop and liquid removal over time.
    """
    fig = Figure(figsize=(10, 6), dpi=dpi, constrained_layout=True)
    ax1 = fig.add_subplot(111)
    
    minutes = df['minute']
    
    # Try to get coalescer data from simulation
    delta_p = df.get('coalescer_delta_p_bar', None)
    liquid_removed = df.get('coalescer_liquid_kg_h', None)
    
    if delta_p is None:
        # Estimate from H2 production (fallback for demo)
        H2_total = df['H2_soec'] + df['H2_pem']
        # Estimate pressure drop proportional to flow (typically 0.01-0.05 bar)
        delta_p = np.clip(H2_total * 0.005, 0.01, 0.1)
        # Estimate liquid removal (small fraction of flow)
        liquid_removed = H2_total * 0.001 * 60  # kg/h
        data_source = "Estimated"
    else:
        data_source = "Simulation Data"
    
    # Primary axis: pressure drop
    color1 = COLORS['coalescer']
    ax1.set_xlabel('Time (Minutes)')
    ax1.set_ylabel('Pressure Drop (bar)', color=color1)
    line1 = ax1.plot(minutes, delta_p, color=color1, linewidth=2, label='ΔP (bar)')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.fill_between(minutes, 0, delta_p, color=color1, alpha=0.1)
    
    # Secondary axis: liquid removed
    ax2 = ax1.twinx()
    color2 = 'blue'
    ax2.set_ylabel('Liquid Removed (kg/h)', color=color2)
    line2 = ax2.plot(minutes, liquid_removed, color=color2, linewidth=1.5, 
                     linestyle='--', label='Liquid (kg/h)')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Cumulative liquid removed
    if hasattr(liquid_removed, 'cumsum'):
        cumulative = (liquid_removed / 60).cumsum()  # Convert to kg
    else:
        cumulative = np.cumsum(np.array(liquid_removed) / 60)
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=8)
    
    avg_dp = delta_p.mean() if hasattr(delta_p, 'mean') else np.mean(delta_p)
    total_liquid = cumulative.iloc[-1] if len(cumulative) > 0 else 0
    
    ax1.set_title(f'Coalescer Performance ({data_source})\n'
                  f'Avg ΔP: {avg_dp:.4f} bar | Total Liquid: {total_liquid:.2f} kg', 
                  fontsize=12)
    ax1.grid(True, alpha=0.3)
    return fig


def create_kod_separation_figure(df: pd.DataFrame, dpi: int = DPI_FAST) -> Figure:
    """
    Create Knock-Out Drum separation performance chart.
    Shows gas density, velocity status, and liquid drainage over time.
    """
    fig = Figure(figsize=(10, 6), dpi=dpi, constrained_layout=True)
    ax1 = fig.add_subplot(111)
    
    minutes = df['minute']
    
    # Try to get KOD data from simulation
    rho_g = df.get('kod_rho_g', None)
    v_real = df.get('kod_v_real', None)
    v_max = df.get('kod_v_max', None)
    liquid_drain = df.get('kod_liquid_drain_kg_h', None)
    
    if rho_g is None:
        # Estimate from H2 production (fallback for demo)
        H2_total = df['H2_soec'] + df['H2_pem']
        # Estimate gas density (typical for H2 at 40 bar)
        rho_g = np.clip(2.0 + H2_total * 0.01, 2.0, 4.0)
        # Estimate velocities
        v_real = np.clip(H2_total * 0.1, 0.1, 1.5)
        v_max = 1.5  # Typical V_max for H2
        # Estimate liquid drainage
        liquid_drain = H2_total * 0.005 * 60  # kg/h
        data_source = "Estimated"
    else:
        data_source = "Simulation Data"
    
    color1 = COLORS['kod']
    
    # Primary axis: Gas density
    ax1.set_xlabel('Time (Minutes)')
    ax1.set_ylabel('Gas Density (kg/m³)', color=color1)
    line1 = ax1.plot(minutes, rho_g, color=color1, linewidth=2, label='ρ_g (kg/m³)')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.fill_between(minutes, 0, rho_g, color=color1, alpha=0.1)
    
    # Secondary axis: Velocities
    ax2 = ax1.twinx()
    color2 = 'green'
    color3 = 'red'
    ax2.set_ylabel('Velocity (m/s)', color='black')
    
    line2 = ax2.plot(minutes, v_real, color=color2, linewidth=1.5, 
                     linestyle='-', label='V_real')
    if isinstance(v_max, (int, float)):
        ax2.axhline(y=v_max, color=color3, linestyle='--', linewidth=1.5, 
                    label=f'V_max = {v_max:.2f} m/s')
        line3 = [ax2.lines[-1]]
    else:
        line3 = ax2.plot(minutes, v_max, color=color3, linewidth=1.5, 
                         linestyle='--', label='V_max')
    
    ax2.tick_params(axis='y', labelcolor='black')
    
    # Combined legend
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', fontsize=8)
    
    # Statistics
    avg_rho = rho_g.mean() if hasattr(rho_g, 'mean') else np.mean(rho_g)
    avg_v = v_real.mean() if hasattr(v_real, 'mean') else np.mean(v_real)
    
    # Check sizing status
    if isinstance(v_max, (int, float)):
        max_v_real = v_real.max() if hasattr(v_real, 'max') else np.max(v_real)
        status = "OK" if max_v_real < v_max else "UNDERSIZED"
    else:
        status = "OK"
    
    ax1.set_title(f'Knock-Out Drum Performance ({data_source})\n'
                  f'Avg ρ_g: {avg_rho:.2f} kg/m³ | Avg V: {avg_v:.3f} m/s | Status: {status}', 
                  fontsize=12)
    ax1.grid(True, alpha=0.3)
    return fig


def create_dry_cooler_figure(df: pd.DataFrame, dpi: int = DPI_FAST) -> Figure:
    """
    Create Dry Cooler performance chart.
    Shows heat duty, fan power, and temperatures.
    """
    fig = Figure(figsize=(10, 6), dpi=dpi, constrained_layout=True)
    ax1 = fig.add_subplot(111)
    
    minutes = df['minute']
    
    # Try to get Dry Cooler data (assuming dry_cooler_0 for now or sum)
    # Search for keys starting with 'dry_cooler_'
    
    heat_duty_cols = [c for c in df.columns if 'dry_cooler' in c and 'heat_duty_kw' in c]
    fan_power_cols = [c for c in df.columns if 'dry_cooler' in c and 'fan_power_kw' in c]
    temp_out_cols = [c for c in df.columns if 'dry_cooler' in c and 'outlet_temp_c' in c]
    
    data_source = "Simulation"
    
    if not heat_duty_cols:
        # Fallback for demo/empty
        ax1.text(0.5, 0.5, 'No Dry Cooler data available', 
                 ha='center', va='center', transform=ax1.transAxes)
        return fig
        
    # Aggregate or use first
    if len(heat_duty_cols) > 1:
        total_heat_duty = df[heat_duty_cols].sum(axis=1)
        total_fan_power = df[fan_power_cols].sum(axis=1) if fan_power_cols else 0
        avg_temp_out = df[temp_out_cols].mean(axis=1) if temp_out_cols else 0
        data_source = f"Aggregate ({len(heat_duty_cols)} units)"
    else:
        total_heat_duty = df[heat_duty_cols[0]]
        total_fan_power = df[fan_power_cols[0]] if fan_power_cols else 0
        avg_temp_out = df[temp_out_cols[0]] if temp_out_cols else 0
        data_source = heat_duty_cols[0].split('_heat_duty')[0]
        
    ax1.plot(minutes, total_heat_duty, label='Heat Rejection (kW)', color='orange')
    ax1.plot(minutes, total_fan_power, label='Fan Power (kW)', color='blue', linestyle='--')
    
    ax2 = ax1.twinx()
    ax2.plot(minutes, avg_temp_out, label='Outlet Temp (°C)', color='red', linestyle=':')
    ax2.set_ylabel('Temperature (°C)')
    
    ax1.set_xlabel('Time (Minutes)')
    ax1.set_ylabel('Power (kW)')
    ax1.set_title(f'Dry Cooler Performance ({data_source})', fontsize=12)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    return fig


# ==============================================================================
# GRAPH REGISTRY - Extensible configuration for all available graphs
# ==============================================================================
# To add a new graph:
# 1. Create a function create_*_figure(df) -> Figure
# 2. Add an entry to GRAPH_REGISTRY below
# 3. Optionally assign it to a folder in GRAPH_HIERARCHY (in SimulationReportWidget)

GRAPH_REGISTRY: Dict[str, Dict[str, Any]] = {
    'dispatch': {
        'name': 'Power Dispatch',
        'func': create_dispatch_figure,
        'description': 'Stacked power consumption and grid sales'
    },
    'arbitrage': {
        'name': 'Price Scenario',
        'func': create_arbitrage_figure,
        'description': 'Spot prices, PPA, and H2 breakeven'
    },
    'h2_production': {
        'name': 'H2 Production Rate',
        'func': create_h2_production_figure,
        'description': 'Hydrogen production rates by source'
    },
    'oxygen_production': {
        'name': 'O2 Production',
        'func': create_oxygen_figure,
        'description': 'Oxygen co-production rates'
    },
    'water_consumption': {
        'name': 'Water Consumption',
        'func': create_water_figure,
        'description': 'Total water usage including losses'
    },
    'energy_pie': {
        'name': 'Energy Distribution',
        'func': create_energy_pie_figure,
        'description': 'Donut chart of energy breakdown'
    },
    'price_histogram': {
        'name': 'Price Histogram',
        'func': create_histogram_figure,
        'description': 'Distribution of spot prices'
    },
    'dispatch_curve': {
        'name': 'Dispatch Curve',
        'func': create_dispatch_curve_figure,
        'description': 'H2 output vs power input scatter'
    },
    # --- NEW CHARTS ---
    'cumulative_h2': {
        'name': 'Cumulative H2',
        'func': create_cumulative_h2_figure,
        'description': 'Total hydrogen produced over time'
    },
    'cumulative_energy': {
        'name': 'Cumulative Energy',
        'func': create_cumulative_energy_figure,
        'description': 'Total energy consumed and sold'
    },
    'efficiency_curve': {
        'name': 'Efficiency Curve',
        'func': create_efficiency_curve_figure,
        'description': 'System efficiency (LHV) over time'
    },
    'revenue_analysis': {
        'name': 'Revenue Analysis',
        'func': create_revenue_analysis_figure,
        'description': 'Grid revenue vs H2 value comparison'
    },
    'polarization': {
        'name': 'Polarization Curves',
        'func': create_polarization_figure,
        'description': 'PEM + SOEC V-I Curves (BOL/EOL)'
    },
    'degradation': {
        'name': 'Degradation Projection',
        'func': create_degradation_figure,
        'description': 'Cell voltage degradation over operating years'
    },
    'compressor_ts': {
        'name': 'Compressor T-s Diagram',
        'func': create_compressor_ts_figure,
        'description': 'Temperature-entropy diagram for compression'
    },
    'module_power': {
        'name': 'Module Power Distribution',
        'func': create_module_power_figure,
        'description': 'Power distribution across SOEC modules'
    },
    # --- THERMAL & SEPARATION CHARTS ---
    'chiller_cooling': {
        'name': 'Chiller Cooling Load',
        'func': create_chiller_cooling_figure,
        'description': 'Cooling load and electrical consumption'
    },
    'coalescer_separation': {
        'name': 'Coalescer Separation',
        'func': create_coalescer_separation_figure,
        'description': 'Pressure drop and liquid removal'
    },
    'kod_separation': {
        'name': 'Knock-Out Drum',
        'func': create_kod_separation_figure,
        'description': 'Gas density, velocity, and liquid drainage'
    },
    'dry_cooler_performance': {
        'name': 'Dry Cooler Performance',
        'func': create_dry_cooler_figure,
        'description': 'Heat rejection, fan power, and outlet temperature'
    },
    # --- NEW LEGACY CHARTS ---
    'temporal_averages': {
        'name': 'Temporal Averages',
        'func': create_temporal_averages_figure,
        'description': 'Hourly aggregated price, power, H2 data'
    },
    'physics_efficiency': {
        'name': 'Physics Efficiency',
        'func': create_physics_efficiency_figure,
        'description': 'System vs Stack-only efficiency curves'
    },
    'module_stats': {
        'name': 'Module Statistics',
        'func': create_module_stats_figure,
        'description': 'Wear and cycle counts per SOEC module'
    },
}


def get_graph_function(graph_id: str) -> Optional[Callable]:
    """Get the figure creation function for a graph ID."""
    entry = GRAPH_REGISTRY.get(graph_id)
    return entry['func'] if entry else None


def get_all_graph_ids() -> list:
    """Get list of all available graph IDs."""
    return list(GRAPH_REGISTRY.keys())


def create_figure(graph_id: str, history: Dict[str, Any]) -> Optional[Figure]:
    """
    Create a figure by graph ID.
    
    Args:
        graph_id: The graph identifier (key in GRAPH_REGISTRY)
        history: Simulation history dictionary
        
    Returns:
        Matplotlib Figure object or None if graph_id not found
    """
    func = get_graph_function(graph_id)
    if func is None:
        return None
    
    df = normalize_history(history)
    return func(df)


def generate_all_graphs_to_files(
    history: Dict[str, Any],
    output_dir: str,
    graph_ids: Optional[list] = None,
    dpi: int = DPI_HIGH,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> Dict[str, str]:
    """
    Generate all graphs and save as PNG files.
    
    This is the preferred approach for GUI display - generates high-quality
    static images that can be displayed quickly using QLabel/QPixmap.
    
    Args:
        history: Simulation history dictionary
        output_dir: Directory to save PNG files
        graph_ids: Optional list of graph IDs to generate (default: all)
        dpi: DPI for output images (default: DPI_HIGH=100)
        progress_callback: Optional callback(current, total, graph_name)
        
    Returns:
        Dict mapping graph_id -> file path
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    df = normalize_history(history)
    
    if graph_ids is None:
        graph_ids = list(GRAPH_REGISTRY.keys())
    
    result = {}
    total = len(graph_ids)
    
    for i, graph_id in enumerate(graph_ids):
        entry = GRAPH_REGISTRY.get(graph_id)
        if not entry:
            continue
            
        func = entry['func']
        name = entry.get('name', graph_id)
        
        if progress_callback:
            progress_callback(i + 1, total, name)
        
        try:
            # Generate figure with high DPI
            fig = func(df, dpi=dpi)
            
            # Save to file with white background
            filepath = os.path.join(output_dir, f"{graph_id}.png")
            fig.savefig(filepath, dpi=dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            
            # Close figure to free memory
            fig.clear()
            plt.close(fig)
            
            result[graph_id] = filepath
            
        except Exception as e:
            print(f"Error generating {graph_id}: {e}")
            continue
    
    return result

