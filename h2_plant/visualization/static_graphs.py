"""
Static (Matplotlib) Graph Implementations.

This module consolidates all Matplotlib-based figure generators.
It replaces the legacy `h2_plant/gui/core/plotter.py` and `h2_plant/visualization/profile_plotter.py`.
"""

from matplotlib.figure import Figure
import matplotlib.patches as mpatches
from matplotlib.cm import get_cmap
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging

# Imports for Profile Reconstitution
from h2_plant.core.stream import Stream
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.components.purification.deoxo_reactor import DeoxoReactor

logger = logging.getLogger(__name__)


def log_graph_errors(func):
    """
    Decorator to wrap graph generation functions with error logging.
    
    Catches exceptions during graph generation and logs them instead of
    failing silently. Also logs entry for debugging data availability.
    """
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        try:
            logger.debug(f"Generating graph: {func_name}")
            result = func(*args, **kwargs)
            logger.debug(f"Graph generated successfully: {func_name}")
            return result
        except KeyError as e:
            logger.warning(f"[{func_name}] Missing data column: {e}")
            raise
        except ValueError as e:
            logger.warning(f"[{func_name}] Value error: {e}")
            raise
        except Exception as e:
            logger.error(f"[{func_name}] Failed to generate graph: {e}", exc_info=True)
            raise
    return wrapper

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

# ==============================================================================
# CONFIG ACCESS HELPER
# ==============================================================================
def get_config(df: pd.DataFrame, key: str, default=None):
    """
    Retrieve configuration value from df.attrs['config'].
    
    Config values are sourced from simulation context (economics_parameters.yaml, 
    topology, etc.) and stored in df.attrs during DataFrame creation.
    
    Args:
        df: DataFrame with attrs containing 'config' dict.
        key: Configuration key (e.g., 'ppa_price_eur_mwh', 'h2_price_eur_kg').
        default: Fallback value if key not found.
    
    Returns:
        Config value or default.
    """
    config = df.attrs.get('config', {})
    return config.get(key, default)


def calculate_h2_equiv_price(df: pd.DataFrame) -> float:
    """
    Calculate H2 equivalent electricity price using dispatch formula.
    
    Formula: h2_equiv = (1000 / η_H2) × h2_price_eur_kg
    
    Where η_H2 is the electrolyzer efficiency in kWh/kg.
    
    Returns:
        H2 equivalent price in EUR/MWh, or 0.0 if config values are missing.
    """
    h2_price = get_config(df, 'h2_price_eur_kg')
    efficiency = get_config(df, 'soec_h2_kwh_kg', 37.5)  # Default SOEC efficiency
    
    if h2_price is None:
        return 0.0
    
    return (1000.0 / efficiency) * h2_price


# ==============================================================================
# TEMPERATURE DETECTION
# ==============================================================================
# Threshold to distinguish Kelvin from Celsius in auto-detection.
# Used only when column names don't specify units (_k or _c suffix).
# Most components now expose both variants; prefer _c columns when available.
KELVIN_DETECTION_THRESHOLD = 200.0

def downsample_for_plot(data, max_points: int = 500):
    """
    Downsample data for faster plotting while preserving visual shape.
    """
    if len(data) <= max_points:
        return data
    stride = max(1, len(data) // max_points)
    return data[::stride]


# ==============================================================================
# DATA NORMALIZATION
# ==============================================================================

def _detect_component_stream_type(df: pd.DataFrame, comp_name: str) -> str:
    """
    Detect if component processes primarily H2 or O2 based on history composition.
    
    Args:
        df: History DataFrame
        comp_name: Component name (e.g., 'KOD_1')
        
    Returns:
        'H2', 'O2', or 'Mixed'
    """
    # Try molar fractions
    h2_cols = [c for c in df.columns if comp_name in c and ('molar_fraction_H2' in c or 'MassFrac_H2' in c)]
    o2_cols = [c for c in df.columns if comp_name in c and ('molar_fraction_O2' in c or 'MassFrac_O2' in c)]
    
    if not h2_cols and not o2_cols:
        return 'Mixed'
        
    # Calculate average composition
    h2_mean = df[h2_cols].mean().mean() if h2_cols else 0.0
    o2_mean = df[o2_cols].mean().mean() if o2_cols else 0.0
    
    if h2_mean > 0.5:
        return 'H2'
    elif o2_mean > 0.5:
        return 'O2'
        
    # If no dominant species found by composition (e.g. water line), try inference from upstream/downstream
    # For now, return Mixed
    return 'Mixed'

def normalize_history(history: Dict[str, Any]) -> pd.DataFrame:
    """
    Normalizes the history dictionary to ensure consistent keys for plotting.
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
    
    # Time
    if 'minute' not in df.columns:
        df['minute'] = df.index
    
    if 'soec_module_powers' in history:
        df.attrs['soec_module_powers'] = history['soec_module_powers']
        
    return df

# ==============================================================================
# FIGURE CREATION FUNCTIONS
# ==============================================================================

@log_graph_errors
def create_dispatch_figure(df: pd.DataFrame, dpi: int = DPI_FAST) -> Figure:
    """Create power dispatch stacked area chart."""
    fig = Figure(figsize=(10, 5), dpi=dpi, constrained_layout=True)
    ax = fig.add_subplot(111)
    minutes = downsample_for_plot(df['minute'])
    P_soec = downsample_for_plot(df['P_soec'])
    P_pem = downsample_for_plot(df['P_pem'])
    P_sold = downsample_for_plot(df['P_sold'])
    P_offer = downsample_for_plot(df['P_offer'])
    
    # Get auxiliary power (convert kW to MW)
    if 'auxiliary_power_kw' in df.columns:
        P_aux = downsample_for_plot(df['auxiliary_power_kw'] / 1000.0)
    else:
        P_aux = np.zeros_like(P_soec)
    
    ax.fill_between(minutes, 0, P_soec, label='SOEC Consumption', color=COLORS['soec'], alpha=0.6)
    ax.fill_between(minutes, P_soec, P_soec + P_pem, label='PEM Consumption', color=COLORS['pem'], alpha=0.6)
    ax.fill_between(minutes, P_soec + P_pem, P_soec + P_pem + P_aux, label='Auxiliary', color='#9467bd', alpha=0.6)
    ax.fill_between(minutes, P_soec + P_pem + P_aux, P_soec + P_pem + P_aux + P_sold, label='Grid Sale', color=COLORS['sold'], alpha=0.6)
    ax.plot(minutes, P_offer, label='Offered Power', color=COLORS['offer'], linestyle='--', linewidth=1.5)
    
    ax.set_title('Hybrid Dispatch: SOEC + PEM + Auxiliary + Arbitrage', fontsize=12)
    ax.set_xlabel('Time (Minutes)')
    ax.set_ylabel('Power (MW)')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    return fig

@log_graph_errors
def create_arbitrage_figure(df: pd.DataFrame, dpi: int = DPI_FAST) -> Figure:
    """Create price scenario chart."""
    fig = Figure(figsize=(10, 5), dpi=dpi, constrained_layout=True)
    ax = fig.add_subplot(111)
    minutes = downsample_for_plot(df['minute'])
    spot_price = downsample_for_plot(df['Spot'])
    
    # Get prices from config (or show message if not available)
    ppa_price = get_config(df, 'ppa_price_eur_mwh')
    h2_equiv_price = calculate_h2_equiv_price(df)
    
    ax.plot(minutes, spot_price, label='Spot Price (EUR/MWh)', color=COLORS['price'])
    
    if ppa_price is not None:
        ax.axhline(y=ppa_price, color=COLORS['ppa'], linestyle='--', label=f'PPA Price ({ppa_price:.0f})')
    
    if h2_equiv_price > 0:
        ax.axhline(y=h2_equiv_price, color='green', linestyle='-.', label=f'H2 Breakeven (~{h2_equiv_price:.0f})')
    
    ax.set_title('Price Scenario & H2 Opportunity Cost', fontsize=12)
    ax.set_xlabel('Time (Minutes)')
    ax.set_ylabel('Price (EUR/MWh)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return fig

@log_graph_errors
def create_h2_production_figure(df: pd.DataFrame, dpi: int = DPI_FAST) -> Figure:
    """Create hydrogen production chart."""
    fig = Figure(figsize=(10, 5), dpi=dpi, constrained_layout=True)
    ax = fig.add_subplot(111)
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

@log_graph_errors
def create_oxygen_figure(df: pd.DataFrame, dpi: int = DPI_FAST) -> Figure:
    """Create oxygen production chart."""
    fig = Figure(figsize=(10, 5), dpi=dpi, constrained_layout=True)
    ax = fig.add_subplot(111)
    O2_pem_full = df.get('O2_pem_kg', df['H2_pem'] * 8.0)
    O2_soec_full = df['H2_soec'] * 8.0
    O2_total_full = O2_soec_full + O2_pem_full
    
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

@log_graph_errors
def create_water_figure(df: pd.DataFrame, dpi: int = DPI_FAST) -> Figure:
    """Create water consumption chart."""
    fig = Figure(figsize=(10, 5), dpi=dpi, constrained_layout=True)
    ax = fig.add_subplot(111)
    water_soec_full = df['Steam_soec'] * 1.10
    water_pem_full = df['H2O_pem'] * 1.02
    total_full = water_soec_full + water_pem_full
    
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

@log_graph_errors
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
        ax.pie(valid_sizes, explode=explode, labels=valid_labels, colors=valid_colors, autopct='%1.1f%%', pctdistance=0.85, startangle=140, wedgeprops=dict(width=0.4, edgecolor='w'))
        ax.text(0, 0, f"TOTAL\n{E_total:.1f}\nMWh", ha='center', va='center', fontsize=10, fontweight='bold')
        ax.set_title('Energy Distribution', fontsize=12)
    else:
        ax.text(0.5, 0.5, 'No energy data', ha='center', va='center', transform=ax.transAxes)
    return fig

@log_graph_errors
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

@log_graph_errors
def create_dispatch_curve_figure(df: pd.DataFrame, dpi: int = DPI_FAST) -> Figure:
    """Create dispatch curve scatter plot (P vs H2)."""
    fig = Figure(figsize=(10, 5), dpi=dpi, constrained_layout=True)
    ax = fig.add_subplot(111)
    P_total = downsample_for_plot(df['P_soec'] + df['P_pem'], max_points=10000)
    H2_total = downsample_for_plot(df['H2_soec'] + df['H2_pem'], max_points=10000)
    ax.scatter(P_total, H2_total, alpha=0.5, c='blue', edgecolors='none', s=10)
    ax.set_title('Dispatch Curve: H2 Production vs Power', fontsize=12)
    ax.set_xlabel('Total Power Input (MW)')
    ax.set_ylabel('H2 Production (kg/min)')
    ax.grid(True, alpha=0.3)
    return fig

@log_graph_errors
def create_cumulative_h2_figure(df: pd.DataFrame, dpi: int = DPI_FAST) -> Figure:
    """Create cumulative hydrogen production chart."""
    fig = Figure(figsize=(10, 5), dpi=dpi, constrained_layout=True)
    ax = fig.add_subplot(111)
    H2_soec_cum_full = df['H2_soec'].cumsum()
    H2_pem_cum_full = df['H2_pem'].cumsum()
    H2_total_cum_full = H2_soec_cum_full + H2_pem_cum_full
    final_total = H2_total_cum_full.iloc[-1] if not H2_total_cum_full.empty else 0
    
    minutes = downsample_for_plot(df['minute'])
    H2_soec_cum = downsample_for_plot(H2_soec_cum_full)
    H2_total_cum = downsample_for_plot(H2_total_cum_full)
    
    ax.fill_between(minutes, 0, H2_soec_cum, label='SOEC (Cumulative)', color=COLORS['soec'], alpha=0.6)
    ax.fill_between(minutes, H2_soec_cum, H2_total_cum, label='PEM (Cumulative)', color=COLORS['pem'], alpha=0.6)
    ax.plot(minutes, H2_total_cum, color='black', linestyle='--', label=f'Total: {final_total:.1f} kg', linewidth=1.5)
    
    ax.set_title('Cumulative Hydrogen Production', fontsize=12)
    ax.set_xlabel('Time (Minutes)')
    ax.set_ylabel('Cumulative H2 (kg)')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)
    return fig

@log_graph_errors
def create_cumulative_energy_figure(df: pd.DataFrame, dpi: int = DPI_FAST) -> Figure:
    """Create cumulative energy consumption chart."""
    fig = Figure(figsize=(10, 5), dpi=dpi, constrained_layout=True)
    ax = fig.add_subplot(111)
    E_soec_cum_full = (df['P_soec'] / 60.0).cumsum()
    E_total_cum_full = E_soec_cum_full + (df['P_pem'] / 60.0).cumsum()
    E_sold_cum_full = (df['P_sold'] / 60.0).cumsum()
    final_sold = E_sold_cum_full.iloc[-1] if not E_sold_cum_full.empty else 0
    final_total = E_total_cum_full.iloc[-1] if not E_total_cum_full.empty else 0
    
    minutes = downsample_for_plot(df['minute'])
    E_soec_cum = downsample_for_plot(E_soec_cum_full)
    E_total_cum = downsample_for_plot(E_total_cum_full)
    E_sold_cum = downsample_for_plot(E_sold_cum_full)
    
    ax.fill_between(minutes, 0, E_soec_cum, label='SOEC Energy', color=COLORS['soec'], alpha=0.6)
    ax.fill_between(minutes, E_soec_cum, E_total_cum, label='PEM Energy', color=COLORS['pem'], alpha=0.6)
    ax.plot(minutes, E_sold_cum, color=COLORS['sold'], linestyle='--', label=f'Grid Sale: {final_sold:.1f} MWh', linewidth=1.5)
    ax.plot(minutes, E_total_cum, color='black', linestyle='-', label=f'Total Used: {final_total:.1f} MWh', linewidth=1.5)
    
    ax.set_title('Cumulative Energy Consumption', fontsize=12)
    ax.set_xlabel('Time (Minutes)')
    ax.set_ylabel('Energy (MWh)')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)
    return fig

@log_graph_errors
def create_efficiency_curve_figure(df: pd.DataFrame, dpi: int = DPI_FAST) -> Figure:
    """Create system efficiency over time chart."""
    fig = Figure(figsize=(10, 5), dpi=dpi, constrained_layout=True)
    ax = fig.add_subplot(111)
    
    P_total = df['P_soec'] + df['P_pem']
    H2_total = df['H2_soec'] + df['H2_pem']
    H2_energy = H2_total * 0.03333  # MWh
    
    efficiency = np.where(P_total > 0.01, (H2_energy / (P_total / 60.0)) * 100, 0)
    window = min(30, len(efficiency) // 10) if len(efficiency) > 30 else 5
    eff_smooth = pd.Series(efficiency).rolling(window=window, min_periods=1).mean()
    
    minutes = downsample_for_plot(df['minute'])
    eff_plot = downsample_for_plot(eff_smooth)
    
    ax.plot(minutes, eff_plot, color='green', linewidth=2, label='System Efficiency')
    ax.fill_between(minutes, 0, eff_plot, color='green', alpha=0.1)
    avg_eff = eff_smooth[eff_smooth > 0].mean() if (eff_smooth > 0).any() else 0
    ax.axhline(y=avg_eff, color='darkgreen', linestyle='--', label=f'Average: {avg_eff:.1f}%', alpha=0.8)
    
    ax.set_title('System Efficiency Over Time (LHV Basis)', fontsize=12)
    ax.set_xlabel('Time (Minutes)')
    ax.set_ylabel('Efficiency (%)')
    ax.set_ylim(0, min(100, max(eff_smooth) * 1.2) if max(eff_smooth) > 0 else 100)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return fig

@log_graph_errors
def create_revenue_analysis_figure(df: pd.DataFrame, dpi: int = DPI_FAST) -> Figure:
    """Create revenue analysis chart."""
    fig = Figure(figsize=(10, 5), dpi=dpi, constrained_layout=True)
    ax = fig.add_subplot(111)
    
    revenue_per_min = (df['P_sold'] * df['Spot']) / 60.0
    cumulative_revenue_full = revenue_per_min.cumsum()
    
    cumulative_h2_value_full = ((df['H2_soec'] + df['H2_pem']) * get_config(df, 'h2_price_eur_kg', 9.6)).cumsum()
    
    final_revenue = cumulative_revenue_full.iloc[-1] if not cumulative_revenue_full.empty else 0
    final_h2_value = cumulative_h2_value_full.iloc[-1] if not cumulative_h2_value_full.empty else 0
    
    minutes = downsample_for_plot(df['minute'])
    cumulative_revenue = downsample_for_plot(cumulative_revenue_full)
    cumulative_h2_value = downsample_for_plot(cumulative_h2_value_full)
    
    ax.plot(minutes, cumulative_revenue, color=COLORS['sold'], linewidth=2, label=f'Grid Revenue: €{final_revenue:.0f}')
    ax.plot(minutes, cumulative_h2_value, color=COLORS['h2_total'], linewidth=2, linestyle='--', label=f'H2 Value: €{final_h2_value:.0f}')
    ax.fill_between(minutes, 0, cumulative_revenue, color=COLORS['sold'], alpha=0.2)
    
    ax.set_title('Cumulative Revenue Analysis', fontsize=12)
    ax.set_xlabel('Time (Minutes)')
    ax.set_ylabel('Value (EUR)')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)
    return fig

@log_graph_errors
def create_temporal_averages_figure(df: pd.DataFrame, dpi: int = DPI_FAST) -> Figure:
    """Create temporal averages chart."""
    fig = Figure(figsize=(12, 10), dpi=dpi, constrained_layout=True)
    
    df_indexed = df.copy()
    df_indexed.index = pd.date_range(start="2024-01-01 00:00", periods=len(df_indexed), freq='min')
    df_indexed['H2_total'] = df_indexed['H2_soec'] + df_indexed['H2_pem']
    
    numeric_cols = df_indexed.select_dtypes(include=[np.number]).columns
    df_hourly = df_indexed[numeric_cols].resample('h').mean()
    
    if len(df_hourly) < 2:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, 'Simulation too short for hourly averages', ha='center', va='center', transform=ax.transAxes, fontsize=12)
        return fig
    
    ax1 = fig.add_subplot(311)
    ax1.plot(df_hourly.index, df_hourly['Spot'], color='black', marker='.', linestyle='-', linewidth=1, label='Avg Spot Price')
    ax1.set_ylabel('Price (EUR/MWh)')
    ax1.set_title('Hourly Average: Spot Market Prices', fontsize=11)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(312, sharex=ax1)
    ax2.stackplot(df_hourly.index, df_hourly['P_soec'], df_hourly['P_pem'], df_hourly['P_sold'], labels=['SOEC', 'PEM', 'Sold'], colors=[COLORS['soec'], COLORS['pem'], COLORS['sold']], alpha=0.7)
    ax2.set_ylabel('Avg Power (MW)')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(313, sharex=ax1)
    ax3.plot(df_hourly.index, df_hourly['H2_total'], color=COLORS['h2_total'], linewidth=2, label='Avg H2 Rate')
    ax3.fill_between(df_hourly.index, 0, df_hourly['H2_total'], color=COLORS['h2_total'], alpha=0.1)
    ax3.set_ylabel('Rate (kg/min)')
    ax3.legend(loc='upper left', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    fig.suptitle('Temporal Averages (Hourly)', fontsize=14)
    return fig

def _prepare_monthly_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Helper to prepare monthly aggregated data for performance graphs."""
    # Create time index
    df_indexed = df.copy()
    start_date = df.attrs.get('start_date', "2024-01-01 00:00")
    df_indexed.index = pd.date_range(start=start_date, periods=len(df_indexed), freq='min')
    
    # Check sufficiency
    if (df_indexed.index[-1] - df_indexed.index[0]).days < 30:
        return {'valid': False, 'reason': 'Simulation too short (< 1 month)'}

    # Resample Monthly
    cols_to_sum = ['P_soec', 'P_pem', 'H2_soec', 'H2_pem']
    for col in cols_to_sum:
        if col not in df_indexed.columns:
            df_indexed[col] = 0.0

    df_monthly = df_indexed[cols_to_sum].resample('ME').sum()
    
    # Energy MWh = Power(MW) / 60
    df_monthly['E_soec_MWh'] = df_monthly['P_soec'] / 60.0
    df_monthly['E_pem_MWh'] = df_monthly['P_pem'] / 60.0
    df_monthly['E_total_MWh'] = df_monthly['E_soec_MWh'] + df_monthly['E_pem_MWh']
    
    # H2 Energy (LHV)
    df_monthly['H2_E_soec'] = df_monthly['H2_soec'] * 0.03333
    df_monthly['H2_E_pem'] = df_monthly['H2_pem'] * 0.03333
    df_monthly['H2_E_total'] = df_monthly['H2_E_soec'] + df_monthly['H2_E_pem']

    # Filter empty
    df_monthly = df_monthly[df_monthly['E_total_MWh'] > 0.001].copy()
    if df_monthly.empty:
        return {'valid': False, 'reason': 'No energy data'}

    # Efficiency
    df_monthly['Eff_Sys'] = (df_monthly['H2_E_total'] / df_monthly['E_total_MWh']) * 100
    df_monthly['Eff_SOEC'] = (df_monthly['H2_E_soec'] / df_monthly['E_soec_MWh']) * 100
    df_monthly['Eff_PEM'] = (df_monthly['H2_E_pem'] / df_monthly['E_pem_MWh']) * 100
    
    # Capacity Factor
    rated_soec = get_config(df, 'soec_capacity_mw')
    if rated_soec is None: rated_soec = df['P_soec'].max() or 1.0
    
    rated_pem = get_config(df, 'pem_capacity_mw')
    if rated_pem is None: rated_pem = df['P_pem'].max() or 1.0
    
    hours_in_month = df_monthly.index.days_in_month * 24.0
    df_monthly['CF_SOEC'] = (df_monthly['E_soec_MWh'] / hours_in_month / rated_soec) * 100
    df_monthly['CF_PEM'] = (df_monthly['E_pem_MWh'] / hours_in_month / rated_pem) * 100
    
    # Module Data
    mod_cols = [c for c in df_indexed.columns if 'soec_module_powers_' in c]
    mod_cols.sort(key=lambda s: int(s.split('_')[-1]))
    
    df_mod_monthly = None
    if mod_cols:
        df_mod_monthly = df_indexed[mod_cols].resample('ME').sum() / 60.0 # MWh
    
    return {
        'valid': True,
        'df_monthly': df_monthly,
        'df_mod_monthly': df_mod_monthly,
        'mod_cols': mod_cols,
        'rated_soec': rated_soec,
        'rated_pem': rated_pem,
        'x_labels': df_monthly.index.strftime('%b'),
        'hours_in_month': hours_in_month
    }

@log_graph_errors
def create_monthly_performance_figure(df: pd.DataFrame, dpi: int = DPI_FAST) -> Figure:
    """Combined monthly performance figure (Efficiency + CF + Heatmap)."""
    data = _prepare_monthly_data(df)
    fig = Figure(figsize=(12, 10), dpi=dpi, constrained_layout=True)
    
    if not data['valid']:
        fig.text(0.5, 0.5, data.get('reason', 'Invalid Data'), ha='center')
        return fig
        
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])
    ax_eff = fig.add_subplot(gs[0, 0])
    ax_cf = fig.add_subplot(gs[0, 1])
    ax_mod = fig.add_subplot(gs[1, :])
    
    # Plot Efficiency
    _plot_monthly_efficiency(ax_eff, data)
    
    # Plot Capacity Factor
    _plot_monthly_cf(ax_cf, data)
    
    # Plot Heatmap
    _plot_soec_heatmap(ax_mod, data, fig)
    
    fig.suptitle('Monthly Performance Indicators', fontsize=14)
    return fig

@log_graph_errors
def create_monthly_efficiency_figure(df: pd.DataFrame, dpi: int = DPI_FAST) -> Figure:
    """Separate figure for Monthly Efficiency."""
    data = _prepare_monthly_data(df)
    fig = Figure(figsize=(8, 6), dpi=dpi, constrained_layout=True)
    
    if not data['valid']:
        fig.text(0.5, 0.5, data.get('reason', 'Invalid Data'), ha='center')
        return fig
        
    ax = fig.add_subplot(111)
    _plot_monthly_efficiency(ax, data)
    fig.suptitle('Monthly Efficiency Trends', fontsize=14)
    return fig

@log_graph_errors
def create_monthly_capacity_factor_figure(df: pd.DataFrame, dpi: int = DPI_FAST) -> Figure:
    """Separate figure for Monthly Capacity Factor."""
    data = _prepare_monthly_data(df)
    fig = Figure(figsize=(8, 6), dpi=dpi, constrained_layout=True)
    
    if not data['valid']:
        fig.text(0.5, 0.5, data.get('reason', 'Invalid Data'), ha='center')
        return fig
        
    ax = fig.add_subplot(111)
    _plot_monthly_cf(ax, data)
    fig.suptitle('Monthly Capacity Factor', fontsize=14)
    return fig

@log_graph_errors
def create_soec_module_heatmap_figure(df: pd.DataFrame, dpi: int = DPI_FAST) -> Figure:
    """Separate figure for SOEC Module Heatmap."""
    data = _prepare_monthly_data(df)
    fig = Figure(figsize=(10, 6), dpi=dpi, constrained_layout=True)
    
    if not data['valid']:
        fig.text(0.5, 0.5, data.get('reason', 'Invalid Data'), ha='center')
        return fig
    
    # Only useful if module data exists
    if data['df_mod_monthly'] is None:
        fig.text(0.5, 0.5, 'No individual module data available', ha='center')
        return fig

    ax = fig.add_subplot(111)
    _plot_soec_heatmap(ax, data, fig)
    fig.suptitle('SOEC Module Utilization', fontsize=14)
    return fig

# --- Plotting Helpers ---
def _plot_monthly_efficiency(ax, data):
    df = data['df_monthly']
    x_labels = data['x_labels']
    x = np.arange(len(x_labels))
    
    ax.plot(x, df['Eff_Sys'], 'k-o', label='System', linewidth=2)
    ax.plot(x, df['Eff_SOEC'], 'r--s', label='SOEC', linewidth=1.5, alpha=0.7)
    ax.plot(x, df['Eff_PEM'], 'b--^', label='PEM', linewidth=1.5, alpha=0.7)
    
    ax.set_ylabel('Efficiency LHV (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3)

def _plot_monthly_cf(ax, data):
    df = data['df_monthly']
    x_labels = data['x_labels']
    x = np.arange(len(x_labels))
    width = 0.35
    
    ax.bar(x - width/2, df['CF_SOEC'], width, label='SOEC', color='salmon', alpha=0.8)
    ax.bar(x + width/2, df['CF_PEM'], width, label='PEM', color='skyblue', alpha=0.8)
    
    ax.set_ylabel('Capacity Factor (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_title(f"Rated: SOEC {data['rated_soec']:.1f}MW, PEM {data['rated_pem']:.1f}MW", fontsize=10)

def _plot_soec_heatmap(ax, data, fig):
    df_mod = data['df_mod_monthly']
    mod_cols = data['mod_cols']
    x_labels = data['x_labels']
    
    if df_mod is None:
        ax.text(0.5, 0.5, 'No individual module data available', ha='center')
        return

    n_modules = len(mod_cols)
    rated_mod = data['rated_soec'] / n_modules if n_modules > 0 else 1.0
    
    cf_matrix = np.zeros((n_modules, len(data['df_monthly'])))
    
    for i, col in enumerate(mod_cols):
        cf_col = (df_mod[col] / data['hours_in_month'] / rated_mod) * 100
        cf_matrix[i, :] = cf_col.values # .values to ensure alignment

    im = ax.imshow(cf_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)
    
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticks(np.arange(n_modules))
    ax.set_yticklabels([f"Mod {i+1}" for i in range(n_modules)])
    ax.set_xlabel('Month')
    
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label('CF (%)')

@log_graph_errors
def create_soec_module_power_stacked_figure(df: pd.DataFrame, dpi: int = DPI_FAST) -> Figure:
    """
    Stacked area chart of power consumption by each SOEC module.
    
    Visualizes the load distribution and total SOEC power composition.
    """
    fig = Figure(figsize=(12, 6), dpi=dpi, constrained_layout=True)
    ax = fig.add_subplot(111)

    # Prepare data
    df_indexed = df.copy()
    if 'time' not in df_indexed.columns:
        if 'minute' in df_indexed.columns:
            df_indexed['time'] = df_indexed['minute'] / 60.0
        else:
            df_indexed['time'] = df_indexed.index  # Fallback
            
    # Find module columns
    mod_cols = [c for c in df_indexed.columns if 'soec_module_powers_' in c]
    mod_cols.sort(key=lambda s: int(s.split('_')[-1]))
    
    if not mod_cols:
        ax.text(0.5, 0.5, 'No individual module power data available', ha='center')
        return fig

    # Downsample if needed for performance
    if len(df_indexed) > 10000:
        factor = len(df_indexed) // 5000
        df_plot = df_indexed.iloc[::factor]
    else:
        df_plot = df_indexed

    x = df_plot['time']
    y_stack = [df_plot[col] for col in mod_cols]
    labels = [f"Mod {col.split('_')[-1]}" for col in mod_cols]
    
    # Use a colormap if many modules
    cmap = get_cmap('tab20c') if len(mod_cols) > 10 else None
    colors = [cmap(i) for i in np.linspace(0, 1, len(mod_cols))] if cmap else None

    if colors:
        ax.stackplot(x, *y_stack, labels=labels, colors=colors, alpha=0.8)
    else:
        ax.stackplot(x, *y_stack, labels=labels, alpha=0.8)
    
    ax.set_title('SOEC Module Power Consumption Breakdown')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Power (MW)')
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(bottom=0)
    
    # Legend - might be crowded if many modules
    if len(mod_cols) <= 12:
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Modules')
    else:
        # Simplified legend for many modules
        ax.text(1.02, 0.5, f"Total Modules: {len(mod_cols)}\n(Legend hidden for clarity)", 
                transform=ax.transAxes, va='center')
                
    ax.grid(True, alpha=0.3)
    return fig

    ax.grid(True, alpha=0.3)
    return fig

@log_graph_errors
def create_soec_module_wear_figure(df: pd.DataFrame, dpi: int = DPI_FAST) -> Figure:
    """
    Bar charts of SOEC module wear statistics:
    1. Total Energy Produced (MWh) per module.
    2. Number of Start/Stop Cycles per module.
    """
    fig = Figure(figsize=(10, 8), dpi=dpi, constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1])
    ax_energy = fig.add_subplot(gs[0])
    ax_cycles = fig.add_subplot(gs[1])
    
    # 1. Identify Module Columns
    mod_cols = [c for c in df.columns if 'soec_module_powers_' in c]
    mod_cols.sort(key=lambda s: int(s.split('_')[-1]))
    
    if not mod_cols:
        fig.text(0.5, 0.5, 'No individual module power data available', ha='center')
        return fig
    
    # 2. Calculate Metrics
    energies_mwh = []
    cycles_count = []
    labels = []
    
    threshold_mw = 0.01 # Minimal power to consider "ON"
    
    for col in mod_cols:
        series = df[col]
        
        # Energy: Integral of Power. Assuming 1-minute steps (standard).
        # Safe check: if 'minute' exists, check dt? Assuming constant 1 min for now consistent with other graphs.
        total_mwh = series.sum() / 60.0
        energies_mwh.append(total_mwh)
        
        # Cycles: 0 -> 1 transitions
        # Boolean series of ON/OFF
        is_on = series > threshold_mw
        # Shift to compare with previous step - avoid FutureWarning
        shifted = is_on.shift(1)
        prev_on = shifted.where(shifted.notna(), False)
        starts = is_on & (~prev_on)
        cycles = starts.sum()
        cycles_count.append(cycles)
        
        labels.append(f"Mod {col.split('_')[-1]}")
    
    x = np.arange(len(labels))
    
    # 3. Plot Energy
    # Use a gradient or robust color
    ax_energy.bar(x, energies_mwh, color='#4CAF50', alpha=0.8, edgecolor='darkgreen')
    ax_energy.set_ylabel('Total Energy (MWh)')
    ax_energy.set_title('Total Energy Operated (Integrity/Wear Proxy)')
    ax_energy.set_xticks(x)
    ax_energy.set_xticklabels(labels)
    ax_energy.grid(True, axis='y', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(energies_mwh):
        ax_energy.text(i, v, f"{v:.1f}", ha='center', va='bottom', fontsize=8)
        
    # 4. Plot Cycles
    ax_cycles.bar(x, cycles_count, color='#FF9800', alpha=0.8, edgecolor='darkorange')
    ax_cycles.set_ylabel('Start/Stop Cycles (#)')
    ax_cycles.set_title('Number of Activations (Thermal Cycling Proxy)')
    ax_cycles.set_xticks(x)
    ax_cycles.set_xticklabels(labels)
    ax_cycles.grid(True, axis='y', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(cycles_count):
        ax_cycles.text(i, v, f"{int(v)}", ha='center', va='bottom', fontsize=8)

    fig.suptitle('SOEC Module Wear Statistics', fontsize=14)
    return fig

@log_graph_errors
def create_q_breakdown_figure(df: pd.DataFrame, dpi: int = DPI_FAST) -> Figure:
    """
    Thermal Load Breakdown (Q_dot) by Component and Phase (Average kW).
    
    Replicates legacy 'plot_q_breakdown' logic:
    - Plots Average Cooling Duty (kW) for each component.
    - Categorizes by 'Total', 'Gas' (Sensible), 'Liquid/Latent'.
    - Splits into H2-line and O2-line subplots if identifiable, otherwise Combined.
    """
    fig = Figure(figsize=(10, 10), dpi=dpi, constrained_layout=True)
    
    # 1. Identify Components and Calculate Averages
    # Pattern: {cid: {'Total': avg_kw, 'Sensible': avg_kw, 'Latent': avg_kw}}
    data = {}
    
    # Chillers (Full Split available)
    chiller_cols = [c for c in df.columns if '_cooling_load_kw' in c and 'Chiller' in c]
    for col in chiller_cols:
        cid = col.replace('_cooling_load_kw', '')
        total = df[col].mean()
        # Try to find split
        sens = df.get(f"{cid}_sensible_heat_kw", pd.Series([0])).mean()
        lat = df.get(f"{cid}_latent_heat_kw", pd.Series([0])).mean()
        
        # If sensible/latent columns missing/empty, assume majority is Sensible or use Total
        if sens == 0 and lat == 0:
             # Legacy fallback: Use total as 'Sensible' for gas coolers, or try to guess
             sens = total
             
        data[cid] = {'Total': total, 'Sensible': sens, 'Latent': lat}

    # Dry Coolers (Total only/Sensible)
    dc_cols = [c for c in df.columns if '_heat_rejected_kw' in c and 'DryCooler' in c]
    for col in dc_cols:
        cid = col.replace('_heat_rejected_kw', '')
        # Note: heat_rejected is usually positive for cooling
        val = df[col].mean()
        data[cid] = {'Total': val, 'Sensible': val, 'Latent': 0.0}
        
    # Heat Exchangers (Total only)
    hx_cols = [c for c in df.columns if '_heat_removed_kw' in c] # We just added this
    for col in hx_cols:
        cid = col.replace('_heat_removed_kw', '')
        val = df[col].mean()
        # Logic: If val > 0 it's cooling.
        data[cid] = {'Total': val, 'Sensible': val, 'Latent': 0.0}

    if not data:
        fig.text(0.5, 0.5, 'No thermal load data found', ha='center')
        return fig

    # 2. Grouping (H2 vs O2)
    # Heuristic: Check CID string
    h2_group = {}
    o2_group = {}
    other_group = {}
    
    for cid, vals in data.items():
        # Heuristics based on topology naming conventions
        lower_id = cid.lower()
        if 'h2' in lower_id or 'recup' in lower_id or 'interchanger' in lower_id:
            h2_group[cid] = vals
        elif 'o2' in lower_id or 'oxygen' in lower_id:
            o2_group[cid] = vals
        else:
            other_group[cid] = vals
            
    # Merge 'other' into H2 if small or generic, or keep separate? 
    # Let's create subplots based on what we found.
    
    subplots = []
    if h2_group: subplots.append(('Fluxo H₂', h2_group))
    if o2_group: subplots.append(('Fluxo O₂', o2_group))
    if other_group: subplots.append(('Outros', other_group))
    
    n_plots = len(subplots)
    if n_plots == 0: return fig
    
    # 3. Plotting
    gs = fig.add_gridspec(n_plots, 1)
    
    for i, (title, group) in enumerate(subplots):
        ax = fig.add_subplot(gs[i])
        
        labels = list(group.keys())
        # Sort labels to stabilize order?
        labels.sort()
        
        x = np.arange(len(labels))
        width = 0.4
        
        lat_vals = [group[k]['Latent'] for k in labels]
        sens_vals = [group[k]['Sensible'] for k in labels]
        
        # Stacked Bar: Latent (Bottom), Sensible (Top)
        # Colors matching legacy: Latent (H2O) = skyblue/salmon, Sensible (Gas) = blue/red
        
        # Base colors (H2-ish vs O2-ish)
        is_o2 = 'O₂' in title
        color_lat = 'salmon' if is_o2 else 'skyblue'
        color_sens = 'red' if is_o2 else 'blue'
        label_lat = 'H2O (Vapor + Líquido)'
        label_sens = f"{'O2' if is_o2 else 'H2'} (Gás Principal)"
        
        p1 = ax.bar(x, lat_vals, width, color=color_lat, edgecolor='grey', label=label_lat)
        p2 = ax.bar(x, sens_vals, width, bottom=lat_vals, color=color_sens, edgecolor='grey', label=label_sens)
        
        ax.set_title(title)
        ax.set_ylabel('Carga Térmica (kW)')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15 if len(labels) > 4 else 0)
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        ax.axhline(0, color='black', linewidth=0.5)
        
        # Value Labels
        for j, k in enumerate(labels):
            total = group[k]['Total']
            if abs(total) > 0.01:
                ax.text(j + width/2 + 0.02, total, f"{total:.1f}", ha='left', va='center', fontsize=8)

        if i == 0:
            ax.legend(loc='upper right') # Only legend on top plot to save space? Or all?

    fig.suptitle('Carga Térmica Média (Q dot) por Componente', fontsize=14)
    return fig

@log_graph_errors
def create_drain_line_properties_figure(df: pd.DataFrame, dpi: int = DPI_FAST) -> Figure:
    """
    Drain Line Properties (T, P, Flow, Dissolved Gas).
    
    Compares aggregated 'IN' state (sum/avg of upstream drains) vs 'OUT' state (Mixer).
    Filters for H2 and O2 lines based on ID naming.
    """
    fig = Figure(figsize=(14, 12), dpi=dpi, constrained_layout=True)
    
    # 1. Identify Drain Lines (Mixers)
    # Look for DrainRecorderMixer columns
    mixer_cols = [c for c in df.columns if '_outlet_mass_flow_kg_h' in c and 'Drain' in c]
    mixers = [c.replace('_outlet_mass_flow_kg_h', '') for c in mixer_cols]
    
    if not mixers:
        fig.text(0.5, 0.5, 'No DrainRecorderMixer data found', ha='center')
        return fig

    # 2. Collect Data for each Line
    lines_data = {} # {line_name: {'IN': dict, 'OUT': dict}}
    
    for mix_id in mixers:
        # Determine Line Type (H2 or O2)
        line_type = 'H2' if 'H2' in mix_id else ('O2' if 'O2' in mix_id else 'Unknown')
        
        # --- OUT State (Mixer) ---
        out_flow = df[f"{mix_id}_outlet_mass_flow_kg_h"].mean()
        out_temp = df[f"{mix_id}_outlet_temperature_c"].mean()
        # Pressure: stored in kPa, convert to bar
        out_pres_bar = df[f"{mix_id}_outlet_pressure_kpa"].mean() / 100.0
        out_gas = df[f"{mix_id}_dissolved_gas_ppm"].mean()
        
        # --- IN State (Aggregation) ---
        # Find upstream components belonging to this line type
        # Heuristic: Components matching the line_type string
        upstream_flow = 0.0
        upstream_enthalpy_product = 0.0 # Flow * T (Approx for mixing T)
        upstream_gas_product = 0.0 # Flow * Conc
        upstream_pressures = []
        
        # List of potential upstream component types and their flow columns
        # (CID_Suffix, Flow_Col_Suffix, Temp_K_Suffix, Pres_Bar_Suffix)
        
        # We need to scan all columns to find matching components
        # Helper to scan:
        def scan_components(comp_marker, flow_suffix, temp_k_suffix='_drain_temp_k', pres_bar_suffix='_drain_pressure_bar'):
             nonlocal upstream_flow, upstream_enthalpy_product, upstream_gas_product, upstream_pressures
             cols = [c for c in df.columns if flow_suffix in c and comp_marker in c]
             for c in cols:
                 cid = c.replace(flow_suffix, '')
                 # Filter by line type 
                 if line_type not in cid: continue
                 
                 flow = df[c].mean()
                 if flow < 0.001: continue
                 
                 # Temp (K -> C)
                 t_k = df.get(f"{cid}{temp_k_suffix}", pd.Series([298.15])).mean()
                 t_c = t_k - 273.15
                 
                 # Pressure
                 p_bar = df.get(f"{cid}{pres_bar_suffix}", pd.Series([1.0])).mean()
                 
                 # Dissolved Gas (ppm)
                 # Note: Coalescer/KOD track 'dissolved_gas_ppm' or 'outlet_o2_ppm_mol'? 
                 # 'dissolved_gas_ppm' is tracked for KOD/Coalescer/Mixer
                 gas_ppm = df.get(f"{cid}_dissolved_gas_ppm", pd.Series([0.0])).mean()
                 
                 upstream_flow += flow
                 upstream_enthalpy_product += flow * t_c
                 upstream_gas_product += flow * gas_ppm
                 upstream_pressures.append(p_bar)

        # Scan KODs
        scan_components('KOD', '_water_removed_kg_h')
        # Scan Coalescers
        scan_components('Coalescer', '_drain_flow_kg_h')
        # Scan Cyclones
        scan_components('Cyclone', '_water_removed_kg_h') # Using water_removed based on verify
        
        # Calculate Aggregated IN
        if upstream_flow > 0:
            in_temp = upstream_enthalpy_product / upstream_flow
            in_gas = upstream_gas_product / upstream_flow
            in_pres = np.mean(upstream_pressures) if upstream_pressures else 1.0
        else:
            # Fallback if no upstream found (e.g. simple test)
            in_temp, in_gas, in_pres = out_temp, out_gas, out_pres_bar

        lines_data[line_type] = {
            'IN': {'Flow': upstream_flow, 'Temp': in_temp, 'Pres': in_pres, 'Gas': in_gas},
            'OUT': {'Flow': out_flow, 'Temp': out_temp, 'Pres': out_pres_bar, 'Gas': out_gas}
        }

    # 3. Plotting
    # 4 Rows (Props) x 2 Columns (H2, O2)
    gs = fig.add_gridspec(4, 2)
    props = [
        ('Flow', 'Vazão (kg/h)', 'tab:green'),
        ('Temp', 'Temperatura (°C)', 'tab:orange'),
        ('Pres', 'Pressão (bar)', 'tab:purple'),
        ('Gas', 'Gás Dissolvido (ppm)', 'tab:brown')
    ]
    
    x_indices = [0, 1]
    x_labels = ['Agregação IN', 'Mixer/Tank OUT']
    
    for col_idx, line_key in enumerate(['H2', 'O2']):
        data = lines_data.get(line_key)
        
        for row_idx, (prop_key, prop_label, color) in enumerate(props):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            
            if data:
                vals = [data['IN'][prop_key], data['OUT'][prop_key]]
                ax.plot(x_indices, vals, marker='o', linestyle='-', color=color, linewidth=2, markersize=8)
                
                # Value labels
                for i, v in zip(x_indices, vals):
                     ax.text(i, v, f"{v:.1f}", ha='center', va='bottom', fontsize=9, fontweight='bold', color=color)
                     
                # Limits padding
                min_v, max_v = min(vals), max(vals)
                margin = (max_v - min_v) * 0.2 if max_v != min_v else 1.0
                ax.set_ylim(min_v - margin, max_v + margin)
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center')

            if row_idx == 0:
                ax.set_title(f"Fluxo {line_key}")
            
            ax.set_ylabel(prop_label)
            ax.set_xticks(x_indices)
            ax.set_xticklabels(x_labels)
            ax.grid(True, linestyle='--', alpha=0.5)

    fig.suptitle('Propriedades da Linha de Drenos (Média)', fontsize=14)
    return fig

@log_graph_errors
def create_deoxo_profile_figure(df: pd.DataFrame, dpi: int = DPI_FAST) -> Figure:
    """
    Deoxo Reactor Internal Profile (Temperature & Conversion).
    
    METHOD: Profile Reconstitution.
    Since internal profiles are not stored in history, this function:
    1. Calculates average inlet conditions from history.
    2. Instantiates a temporary DeoxoReactor.
    3. Runs a single step to reproduce the steady-state profile.
    4. Plots T(z) and Conversion(z).
    """
    fig = Figure(figsize=(10, 6), dpi=dpi, constrained_layout=True)
    
    # 1. Gather Average Inlet Data
    # We look for columns like 'Deoxo_inlet_mass_flow_kg_h' etc. if tracked, 
    # OR deduce from upstream if necessary. 
    # Assuming 'Deoxo' is the standard ID, or we search for it.
    
    # Try to find a Deoxo ID in columns (now looking for tracked columns)
    deoxo_id = None
    for col in df.columns:
        # Look for any Deoxo tracked column
        if 'Deoxo' in col and any(suffix in col for suffix in ['_inlet_temp_c', '_peak_temp_c', '_o2_in_kg_h']):
            # Extract component ID by removing the suffix
            for suffix in ['_inlet_temp_c', '_peak_temp_c', '_o2_in_kg_h']:
                if col.endswith(suffix):
                    deoxo_id = col.replace(suffix, '')
                    break
            if deoxo_id:
                break
            
    if not deoxo_id:
        fig.text(0.5, 0.5, 'No Deoxo data found in history.\n(Ensure Deoxo tracking is enabled)', ha='center')
        return fig

    # Get Averages
    # Note: 'inlet' props might not be explicitly tracked in history if not requested.
    # However, 'Internal' props often track 'o2_in_kg_h'.
    # A safe bet is to look for 'Deoxo_Last_Inlet_Temp' if we added it, 
    # OR approximate from 'Deoxo_outlet_temperature_c' - delta_T if inlet missing.
    # BEST: Use the 'Stream' tracking if available, or just use what we have.
    
    # Let's assume standard names or fallback to defaults for reconstruction demonstration
    # Ideally, we should have tracked inlet conditions. 
    # If not, we can try to guess from the previous component (e.g. Heater or PSA).
    # FOR NOW: Let's reconstruction from valid tracked metrics OR use a sane default.
    
    # Valid tracked metrics for Deoxo usually: 
    # - outlet_o2_ppm_mol
    # - peak_temperature_c 
    # - conversion_o2_percent
    # - o2_in_kg_h
    
    # We can back-calculate:
    # Mass Flow: Not explicitly tracked? -> Check 'total_mass_flow_kg_h' or similar global?
    # Actually 'Deoxo' usually has 'input_stream.mass_flow' which we might not have.
    # But checking 'run_simulation.py', we often track 'mass_flow_total'.
    # Now we track these explicitly for each Deoxo
    avg_mass_flow = df.get(f"{deoxo_id}_mass_flow_kg_h", pd.Series([1000.0])).mean()
    if avg_mass_flow <= 0:
        avg_mass_flow = df.get('total_mass_flow_kg_h', pd.Series([1000.0])).mean()
    
    # O2 In (tracked directly):
    avg_o2_in_kg_h = df.get(f"{deoxo_id}_o2_in_kg_h", pd.Series([0.0])).mean()
    # If O2 in is 0, no profile to show.
    if avg_o2_in_kg_h <= 1e-6:
        fig.text(0.5, 0.5, 'Negligible O2 Input to Deoxo (Average)', ha='center')
        return fig
        
    y_o2 = avg_o2_in_kg_h / avg_mass_flow if avg_mass_flow > 0 else 0
    
    # Inlet Temperature (tracked):
    t_in_c = df.get(f"{deoxo_id}_inlet_temp_c", pd.Series([70.0])).mean()
    t_in_k = t_in_c + 273.15
    
    # Inlet Pressure (tracked):
    p_in_bar = df.get(f"{deoxo_id}_inlet_pressure_bar", pd.Series([30.0])).mean()
    p_in_pa = p_in_bar * 1e5

    # 2. Instantiate & Simulate
    # Create Registry for initialization (mock)
    registry = ComponentRegistry()
    
    deoxo = DeoxoReactor(deoxo_id)
    registry.register(deoxo_id, deoxo)  # Pass component_id first
    deoxo.initialize(dt=1/3600, registry=registry)
    
    # Construct Inlet Stream
    # Composition: H2 (bal), O2 (calculated), H2O (trace)
    comp = {'H2': 1.0 - y_o2, 'O2': y_o2, 'H2O': 0.0}
    
    in_stream = Stream(
        mass_flow_kg_h=avg_mass_flow,
        temperature_k=t_in_k,
        pressure_pa=p_in_pa,
        composition=comp,
        phase='gas'
    )
    
    deoxo.receive_input('inlet', in_stream)
    deoxo.step(0.0) # Run physics
    
    # 3. Extract Profiles
    profiles = deoxo.get_last_profiles()
    L_data = profiles['L']
    T_data = profiles['T'] - 273.15 # K -> C
    X_data = profiles['X']
    
    if len(L_data) == 0:
        fig.text(0.5, 0.5, 'Deoxo Profile Generation Failed (No Data)', ha='center')
        return fig

    # 4. Plot
    ax1 = fig.add_subplot(111)
    
    # Conversion (Left Axis)
    ax1.set_xlabel('Reactor Length (m)')
    ax1.set_ylabel('Conversion O₂ (Fraction)', color='tab:blue')
    line1, = ax1.plot(L_data, X_data, color='tab:blue', linewidth=2, label='Conversion')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Temperature (Right Axis)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Temperature (°C)', color='tab:red')
    line2, = ax2.plot(L_data, T_data, color='tab:red', linewidth=2, linestyle='--', label='Temperature')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    # Title & Legend
    lns = [line1, line2]
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='center right')
    
    ax1.set_title(f"Deoxo Internal Profile (Reconstituted from Avg Conditions)\nInlet O₂: {y_o2*1e6:.1f} ppm | Flow: {avg_mass_flow:.1f} kg/h")
    
    return fig

    ax1.set_title(f"Deoxo Internal Profile (Reconstituted from Avg Conditions)\nInlet O₂: {y_o2*1e6:.1f} ppm | Flow: {avg_mass_flow:.1f} kg/h")
    
    return fig

@log_graph_errors
def create_drain_mixer_figure(df: pd.DataFrame, dpi: int = DPI_FAST) -> Figure:
    """
    Drain Mixer Balance (Static Bar Chart).
    
    Visualizes Mass & Energy Balance for Drain Mixers.
    Compares 'Calculated IN' (Sum of upstream) vs 'Recorded OUT'.
    """
    fig = Figure(figsize=(10, 12), dpi=dpi, constrained_layout=True)
    
    # 1. Identify Mixers
    mixer_cols = [c for c in df.columns if '_outlet_mass_flow_kg_h' in c and 'Drain' in c]
    mixers = [c.replace('_outlet_mass_flow_kg_h', '') for c in mixer_cols]
    
    if not mixers:
        fig.text(0.5, 0.5, 'No DrainRecorderMixer data found', ha='center')
        return fig

    # 2. Collect Data (IN vs OUT)
    # Structure: {label: {'Flow': val, 'Temp': val, 'Pres': val}}
    plot_items = {} 
    
    for mix_id in mixers:
        # Determine Line Type
        line_type = 'H2' if 'H2' in mix_id else ('O2' if 'O2' in mix_id else 'Other')
        label_base = f"{line_type} Drain"

        # --- OUT (Recorded) ---
        out_flow = df.get(f"{mix_id}_outlet_mass_flow_kg_h", pd.Series([0])).mean()
        out_temp = df.get(f"{mix_id}_outlet_temperature_c", pd.Series([0])).mean()
        out_pres = df.get(f"{mix_id}_outlet_pressure_kpa", pd.Series([100])).mean() / 100.0 # bar
        
        # --- IN (Calculated Sum) ---
        upstream_flow = 0.0
        upstream_enthalpy_prod = 0.0
        upstream_pressures = []

        # Helper to scan upstream (similar to drain_line_props but simpler sum)
        def scan_components(comp_marker, flow_suffix, temp_k_suffix='_drain_temp_k', pres_bar_suffix='_drain_pressure_bar'):
             nonlocal upstream_flow, upstream_enthalpy_prod, upstream_pressures
             cols = [c for c in df.columns if flow_suffix in c and comp_marker in c]
             for c in cols:
                 cid = c.replace(flow_suffix, '')
                 # Filter by line type AND ensuring it belongs to this mixer logic
                 # Heuristic: If mix_id is 'DrainRecorderMixer_H2', we want 'KOD_H2'.
                 # If mix_id is generic, we might capture too much. 
                 # Assuming 1 mixer per line type for now.
                 if line_type not in cid and line_type != 'Other': continue
                 
                 flow = df[c].mean()
                 if flow < 1e-4: continue
                 
                 t_k = df.get(f"{cid}{temp_k_suffix}", pd.Series([298.15])).mean()
                 t_c = t_k - 273.15
                 p_bar = df.get(f"{cid}{pres_bar_suffix}", pd.Series([1.0])).mean()
                 
                 upstream_flow += flow
                 upstream_enthalpy_prod += flow * t_c
                 upstream_pressures.append(p_bar)

        # Scan all potential sources
        scan_components('KOD', '_water_removed_kg_h')
        scan_components('Coalescer', '_drain_flow_kg_h')
        scan_components('Cyclone', '_water_removed_kg_h')
        
        # Calc IN averages
        if upstream_flow > 0:
            in_temp = upstream_enthalpy_prod / upstream_flow
            in_pres = np.mean(upstream_pressures) if upstream_pressures else 1.0
        else:
            in_temp, in_pres = 0.0, 0.0
            
        # Store Comparison Pair
        plot_items[f"{label_base} IN (Calc)"] = {'Flow': upstream_flow, 'Temp': in_temp, 'Pres': in_pres, 'Color': 'tab:blue', 'Alpha': 0.6}
        plot_items[f"{label_base} OUT (Mixer)"] = {'Flow': out_flow, 'Temp': out_temp, 'Pres': out_pres, 'Color': 'tab:green', 'Alpha': 1.0}

    # 3. Plotting
    # 3 Rows: Temp, Pres, Flow
    gs = fig.add_gridspec(3, 1)
    
    # Prepare X axis
    labels = list(plot_items.keys())
    x = np.arange(len(labels))
    
    # Metrics map
    metrics = [
        ('Temp', 'Temperatura (°C)', 0),
        ('Pres', 'Pressão (bar)', 1),
        ('Flow', 'Vazão Mássica (kg/h)', 2)
    ]
    
    for metric_key, ylabel, row_idx in metrics:
        ax = fig.add_subplot(gs[row_idx])
        
        vals = [plot_items[k][metric_key] for k in labels]
        colors = [plot_items[k]['Color'] for k in labels]
        alphas = [plot_items[k]['Alpha'] for k in labels]
        
        bars = ax.bar(x, vals, color=colors, alpha=0.9, edgecolor='black')
        
        # Apply alphas manually if needed or just rely on color diff
        for bar, alpha in zip(bars, alphas):
            bar.set_alpha(alpha)
            
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha='right')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        # Value Labels
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.1f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
            
    fig.suptitle('Balanço dos Mixers de Dreno (Média)', fontsize=14)
    return fig

            
    fig.suptitle('Balanço dos Mixers de Dreno (Média)', fontsize=14)
    return fig

@log_graph_errors
def create_drain_scheme_schematic(df: pd.DataFrame, dpi: int = DPI_FAST) -> Figure:
    """
    Drain System Schematic (Static Diagram).
    
    Visualizes the topology of the drain system:
    - Inputs: PEM Drains, KODs
    - Processing: Valve -> Flash Drum
    - Aggregation: Final Mixer
    - Adapts to show O2 branch only if O2 data exists.
    """
    fig = Figure(figsize=(14, 8), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.set_title('Esquema do Processo de Tratamento de Água de Dreno', fontsize=16, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Detect O2 Branch Existence
    # Check if we have O2 columns in the dataframe
    has_o2 = any('O2' in c or 'o2' in c for c in df.columns if 'mass_flow' in c)
    # Or strict check for O2 drain mixer
    has_o2_drain = any('Drain' in c and 'O2' in c for c in df.columns)
    
    draw_o2 = has_o2 or has_o2_drain

    # --- 1. Definição de Componentes ---
    
    # H2 Flow (Top)
    x_h2 = 10; y_h2 = 80
    componentes_h2 = [
        ("PEM Dreno Recirc. (H₂)", x_h2 + 5, y_h2),
        ("KOD 1 (H₂)", x_h2 + 25, y_h2),
        ("KOD 2 (H₂)", x_h2 + 45, y_h2),
    ]
    
    # O2 Flow (Bottom) - Only if exists
    x_o2 = 10; y_o2 = 25
    componentes_o2 = [
        ("PEM Dreno Recirc. (O₂)", x_o2 + 5, y_o2),
        ("KOD 1 (O₂)", x_o2 + 25, y_o2),
        ("KOD 2 (O₂)", x_o2 + 45, y_o2),
    ] if draw_o2 else []

    # Common Components
    x_valve = 65
    y_valve_h2 = 80
    y_valve_o2 = 25
    
    x_flash = 78
    y_flash_h2 = 80
    y_flash_o2 = 25
    
    x_mixer = 85
    y_mixer = 52.5 if draw_o2 else y_h2 # If no O2, mixer is inline or lower? 
    # Actually if no O2, maybe just straight line? But let's keep structure centered if O2 usually exists.
    if not draw_o2:
        y_mixer = y_h2
        
    # --- Helper Draw Functions ---
    def draw_drenos_brutos(comps, color):
        for i, (name, x, y) in enumerate(comps):
            # Circle
            ax.plot(x, y, 'o', color=color, markersize=8, zorder=5)
            ax.text(x, y + 4, name, ha='center', fontsize=8, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle="round,pad=0.2"))
            
            # Connection Line
            if i > 0:
                prev_x = comps[i-1][1]
                ax.plot([prev_x, x], [y, y], color=color, linestyle=':', linewidth=1.5, zorder=1)
                
            # Vertical Aggregation Line (Stylized input)
            ax.plot([x, x], [y, y + 10], color='gray', linestyle='--', linewidth=1)
                
            # Arrow
            ax.arrow(x - 2, y, 4, 0, head_width=0, head_length=0, fc=color, ec=color) # Just marker line

    def draw_valve(x, y, color):
        rect = mpatches.Rectangle((x - 1.5, y - 3), 3, 6, facecolor='lightgray', edgecolor=color, linewidth=2, zorder=3)
        ax.add_patch(rect)
        ax.text(x, y + 5, "Válvula", ha='center', fontsize=8)

    def draw_flash(x, y, color):
        rect = mpatches.Rectangle((x - 3, y - 6), 6, 12, facecolor='mistyrose' if color == 'firebrick' else 'lightblue', edgecolor=color, linewidth=2, zorder=3)
        ax.add_patch(rect)
        ax.text(x, y + 8, "Flash Drum", ha='center', fontsize=8)
        # Vent arrow
        ax.arrow(x, y + 6, 0, 8, head_width=2, head_length=2, fc='gray', ec='gray', zorder=4)
        ax.text(x + 2, y + 14, "Vent", fontsize=7, color='gray')

    # Draw H2 Branch
    draw_drenos_brutos(componentes_h2, 'firebrick')
    # Aggregation line
    if componentes_h2:
        end_h2 = componentes_h2[-1]
        ax.plot([end_h2[1], x_valve], [y_h2, y_h2], color='firebrick', linewidth=2, zorder=2)
        ax.arrow(x_valve - 5, y_h2, 2, 0, head_width=1.5, head_length=1.5, fc='firebrick', ec='firebrick')
    
    draw_valve(x_valve, y_valve_h2, 'firebrick')
    
    ax.plot([x_valve + 1.5, x_flash - 3], [y_h2, y_h2], color='firebrick', linewidth=2, zorder=2)
    
    draw_flash(x_flash, y_flash_h2, 'firebrick')

    # Draw O2 Branch
    if draw_o2 and componentes_o2:
        draw_drenos_brutos(componentes_o2, 'navy')
        end_o2 = componentes_o2[-1]
        ax.plot([end_o2[1], x_valve], [y_o2, y_o2], color='navy', linewidth=2, zorder=2)
        ax.arrow(x_valve - 5, y_o2, 2, 0, head_width=1.5, head_length=1.5, fc='navy', ec='navy')
        
        draw_valve(x_valve, y_valve_o2, 'navy')
        
        ax.plot([x_valve + 1.5, x_flash - 3], [y_o2, y_o2], color='navy', linewidth=2, zorder=2)
        
        draw_flash(x_flash, y_flash_o2, 'navy')

    # Mixer
    mixer_circle = mpatches.Circle((x_mixer, y_mixer), radius=5, facecolor='lightgray', edgecolor='black', linewidth=2, zorder=4)
    ax.add_patch(mixer_circle)
    ax.text(x_mixer, y_mixer, "Mixer", ha='center', va='center', fontsize=9, fontweight='bold')

    # H2 to Mixer
    ax.plot([x_flash + 3, x_mixer], [y_h2, y_mixer + (5 if draw_o2 else 0)], color='firebrick', linewidth=2, zorder=3)
    
    # O2 to Mixer
    if draw_o2:
        ax.plot([x_flash + 3, x_mixer], [y_o2, y_mixer - 5], color='navy', linewidth=2, zorder=3)

    # Output
    ax.plot([x_mixer + 5, x_mixer + 20], [y_mixer, y_mixer], color='black', linewidth=3, zorder=4)
    ax.arrow(x_mixer + 20, y_mixer, 0.01, 0, head_width=2, head_length=2, fc='black', ec='black')
    ax.text(x_mixer + 15, y_mixer + 3, "Água Recuperada", fontsize=9, fontweight='bold')

    return fig

    ax.text(x_mixer + 15, y_mixer + 3, "Água Recuperada", fontsize=9, fontweight='bold')

    return fig

@log_graph_errors
def create_energy_flow_figure(df: pd.DataFrame, dpi: int = DPI_FAST) -> Figure:
    """
    Energy Flow & Consumption by Component.
    
    Subplot 1: Thermal Load (Q_dot) - Comparisons H2 vs O2.
    Subplot 2: Electrical Power (W_dot) - Consumption.
    """
    fig = Figure(figsize=(10, 10), dpi=dpi, constrained_layout=True)
    
    # 1. Identify Components with Energy Data
    # Power: *_power_kw
    # Heat: *_heat_duty_kw OR *_heat_removed_kw
    
    # helper to find and classify
    def get_comp_data(suffix):
        cols = [c for c in df.columns if c.endswith(suffix)]
        data = {}
        for c in cols:
            comp_id = c.replace(suffix, '')
            val = df[c].mean()
            # Filter low values to reduce noise
            if abs(val) > 0.01:
                data[comp_id] = val
        return data

    power_data = get_comp_data('_power_kw')
    # Also capture Chiller electrical power and DryCooler fan power
    power_data.update(get_comp_data('_electrical_power_kw'))
    power_data.update(get_comp_data('_fan_power_kw'))
    
    # Merge all heat tracking styles
    heat_data = get_comp_data('_heat_duty_kw')
    heat_data.update(get_comp_data('_heat_removed_kw'))
    heat_data.update(get_comp_data('_heat_rejected_kw'))  # DryCooler
    heat_data.update(get_comp_data('_cooling_load_kw'))   # Chiller
    # Also DryCooler tqc
    heat_data.update(get_comp_data('_tqc_duty_kw'))

    if not power_data and not heat_data:
        fig.text(0.5, 0.5, 'No Energy Data (Power/Heat) found.', ha='center')
        return fig

    # 2. Categorize (H2 vs O2 vs Other)
    def categorize(cid):
        if 'H2' in cid: return 'H2'
        if 'O2' in cid: return 'O2'
        return 'Common'

    # Prepare Lists for Plotting
    # We want grouped bars: H2 Group, O2 Group.
    
    # --- Thermal Plot ---
    ax1 = fig.add_subplot(211)
    
    # Sort keys for consistency
    heat_keys = sorted(heat_data.keys())
    h2_heat = [k for k in heat_keys if categorize(k) == 'H2']
    o2_heat = [k for k in heat_keys if categorize(k) == 'O2']
    common_heat = [k for k in heat_keys if categorize(k) == 'Common'] # Treat Common as H2 side or separate?

    # Let's just plot all H2 then all O2 for clarity
    
    # X locations
    x_h2 = np.arange(len(h2_heat))
    x_o2 = np.arange(len(o2_heat)) + (len(h2_heat) + 1) if o2_heat else []
    
    # H2 Bars
    if h2_heat:
        vals = [heat_data[k] for k in h2_heat]
        bars_h2 = ax1.bar(x_h2, vals, color='tab:blue', edgecolor='black', label='H2 - Calor (kW)')
        ax1.bar_label(bars_h2, fmt='%.1f', padding=3)
        
    # O2 Bars
    if o2_heat:
        vals = [heat_data[k] for k in o2_heat]
        bars_o2 = ax1.bar(x_o2, vals, color='tab:red', edgecolor='black', label='O2 - Calor (kW)')
        ax1.bar_label(bars_o2, fmt='%.1f', padding=3)

    # Ticks
    all_ticks = list(x_h2) + list(x_o2)
    all_labels = h2_heat + o2_heat
    ax1.set_xticks(all_ticks)
    ax1.set_xticklabels(all_labels, rotation=15, ha='right')
    ax1.set_ylabel('Carga Térmica (kW)')
    ax1.set_title('Fluxo Térmico por Componente (Média)')
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    
    # --- Electrical Plot ---
    ax2 = fig.add_subplot(212)
    
    power_keys = sorted(power_data.keys())
    h2_pow = [k for k in power_keys if categorize(k) == 'H2']
    o2_pow = [k for k in power_keys if categorize(k) == 'O2']
    # Common (e.g. Pump_Recirc usually O2 side or Common)
    comm_pow = [k for k in power_keys if categorize(k) == 'Common']
    
    # Merge H2 and Common for visual simplicity, or keep distinct?
    # Let's append Common to the end
    
    current_x = 0
    ticks = []
    labels = []
    
    # H2
    if h2_pow:
        vals = [power_data[k] for k in h2_pow]
        x = np.arange(len(h2_pow)) + current_x
        bars = ax2.bar(x, vals, color='skyblue', edgecolor='black', label='H2 - Potência (kW)')
        ax2.bar_label(bars, fmt='%.1f', padding=3)
        ticks.extend(x)
        labels.extend(h2_pow)
        current_x += len(h2_pow) + 0.5
        
    # O2
    if o2_pow:
        vals = [power_data[k] for k in o2_pow]
        x = np.arange(len(o2_pow)) + current_x
        bars = ax2.bar(x, vals, color='salmon', edgecolor='black', label='O2 - Potência (kW)')
        ax2.bar_label(bars, fmt='%.1f', padding=3)
        ticks.extend(x)
        labels.extend(o2_pow)
        current_x += len(o2_pow) + 0.5

    # Common
    if comm_pow:
        vals = [power_data[k] for k in comm_pow]
        x = np.arange(len(comm_pow)) + current_x
        bars = ax2.bar(x, vals, color='lightgreen', edgecolor='black', label='Comum - Potência (kW)')
        ax2.bar_label(bars, fmt='%.1f', padding=3)
        ticks.extend(x)
        labels.extend(comm_pow)
        
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(labels, rotation=15, ha='right')
    ax2.set_ylabel('Potência Elétrica (kW)')
    ax2.set_title('Consumo Elétrico por Componente (Média)')
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

    fig.suptitle('Fluxos de Energia Globais', fontsize=14)
    return fig

    fig.suptitle('Fluxos de Energia Globais', fontsize=14)
    return fig

@log_graph_errors
def create_process_scheme_schematic(df: pd.DataFrame, dpi: int = DPI_FAST) -> Figure:
    """
    Process Scheme Schematic (Dynamic PFD).
    
    Dynamically discovers components and draws the Process Flow Diagram.
    Annotates Energy (Heat/Power) and Mass (Water Removal) interactions.
    """
    fig = Figure(figsize=(20, 8), dpi=dpi)
    ax = fig.add_subplot(111)
    
    # 1. Discover Components (Topology)
    # Heuristic: Define a canonical order, check if each exists in columns.
    canonical_order_h2 = [
        'KOD_1', 'KOD1', 
        'DryCooler', 'DryCooler1', 
        'Chiller_H2', 'Chiller1', 
        'KOD_2', 'KOD2', 
        'Coalescer_H2', 'Coalescer1',
        'HydrogenMultiCyclone',
        'Deoxo', 
        'PSA'
    ]
    # Simple mapping to Display Names
    display_names = {
        'KOD_1': 'KOD 1', 'KOD1': 'KOD 1',
        'DryCooler': 'Dry Cooler', 'DryCooler1': 'Dry Cooler',
        'Chiller_H2': 'Chiller H2', 'Chiller1': 'Chiller',
        'KOD_2': 'KOD 2', 'KOD2': 'KOD 2',
        'Coalescer_H2': 'Coalescer', 'Coalescer1': 'Coalescer',
        'HydrogenMultiCyclone': 'Cyclone',
        'Deoxo': 'Deoxo Reactor',
        'PSA': 'PSA Unit'
    }
    
    comps_present = []
    
    # Check existence by looking for ANY column starting with ID
    # This is a loose check but effective.
    for cid in canonical_order_h2:
        # Check if any column starts with this ID (plus underscore to avoid partial match like KOD_1 matching KOD_10)
        # Or exact match
        found = False
        for col in df.columns:
            if col.startswith(cid + '_') or col == cid:
                found = True
                break
        if found:
            comps_present.append(cid)
            
    if not comps_present:
        ax.text(0.5, 0.5, 'No Components Found for PFD', ha='center')
        return fig

    # 2. Layout Calculation
    spacing_x = 3.0
    y_pos = 0
    positions = {cid: (i * spacing_x, y_pos) for i, cid in enumerate(comps_present)}
    
    # Scale limits
    ax.set_xlim(-1, len(comps_present) * spacing_x)
    ax.set_ylim(-2, 2)
    ax.axis('off')
    
    # 3. Draw Components
    for cid, (x, y) in positions.items():
        name = display_names.get(cid, cid)
        
        # Box
        rect = mpatches.Rectangle((x - 0.6, y - 0.4), 1.2, 0.8, facecolor='lightgray', edgecolor='black', zorder=2)
        ax.add_patch(rect)
        ax.text(x, y, name, ha='center', va='center', fontsize=9, fontweight='bold', zorder=3)
        
        # Connect to Next
        # Find index
        idx = comps_present.index(cid)
        if idx < len(comps_present) - 1:
            next_cid = comps_present[idx+1]
            next_x = positions[next_cid][0]
            
            # Arrow
            ax.annotate('', xy=(next_x - 0.6, y), xytext=(x + 0.6, y),
                        arrowprops=dict(facecolor='blue', shrink=0.0, width=2, headwidth=8))
            if idx == 0: # Label first flow
                ax.text((x + 0.6 + next_x - 0.6)/2, y + 0.1, 'Process Gas', ha='center', va='bottom', fontsize=8, color='blue')

    # 4. Annotations (Energy/Mass)
    # Define rules based on Type
    for cid in comps_present:
        x, y = positions[cid]
        
        # Classification
        is_cooler = 'Cooler' in cid or 'Chiller' in cid
        is_separator = 'KOD' in cid or 'Coalescer' in cid or 'Cyclone' in cid
        is_deoxo = 'Deoxo' in cid
        is_comp = 'Compressor' in cid # Not in canonical list above but might be
        
        # Annotate Heat Removal (Red, Bottom)
        if is_cooler:
            ax.annotate('', xy=(x, y - 0.8), xytext=(x, y - 0.4),
                        arrowprops=dict(facecolor='red', width=1.5, headwidth=6))
            ax.text(x, y - 1.0, 'Q (Heat)', ha='center', va='top', fontsize=7, color='red')
            
        # Annotate Water Removal (Brown, Bottom-Right)
        if is_separator:
            ax.annotate('', xy=(x + 0.3, y - 0.8), xytext=(x + 0.3, y - 0.4),
                        arrowprops=dict(facecolor='saddlebrown', width=1.5, headwidth=6))
            ax.text(x + 0.3, y - 1.0, 'H₂O (Liq)', ha='center', va='top', fontsize=7, color='saddlebrown')
            
        # Annotate Deoxo Heat (Exothermic -> Heat Out usually, or Jacket)
        if is_deoxo:
            ax.annotate('', xy=(x, y - 0.8), xytext=(x, y - 0.4),
                        arrowprops=dict(facecolor='red', width=1.5, headwidth=6))
            ax.text(x, y - 1.0, 'Q (Cooling)', ha='center', va='top', fontsize=7, color='red')
            # Water production? Internal.
            
    fig.suptitle('Esquema do Processo (Discovered Topology)', fontsize=14)
    return fig

# ... (Include other simple physics/module charts here, omitted for brevity but should be copied from plotter.py) ...
# I will include the critical ones requested/verified recently: Water charts

@log_graph_errors
def create_water_removal_total_figure(df: pd.DataFrame, dpi: int = DPI_FAST) -> Figure:
    """
    Create Water Removal Bar Chart (Cumulative).
    
    Enhanced to show H2 vs O2 stream sources separately,
    replicating legacy behavior while keeping cumulative mass metric.
    """
    fig = Figure(figsize=(12, 6), dpi=dpi, constrained_layout=True)
    ax = fig.add_subplot(111)
    
    water_cols = [c for c in df.columns if 'water_removed' in c.lower() or 'water_condensed' in c.lower()]
    
    if not water_cols:
        ax.text(0.5, 0.5, 'No water removal data available.', ha='center', va='center', transform=ax.transAxes)
        return fig
    
    dt_hours = 1.0 / 60.0
    
    # Organize data by stream type
    data = {
        'H2': {},
        'O2': {},
        'Mixed': {}
    }
    
    for col in water_cols:
        # Extract component name
        # Heuristic: Remove standard suffixes
        parts = col.replace('_water_removed_kg_h', '').replace('_water_condensed_kg_h', '')
        # Handle cases where component name has underscores (e.g. KOD_1)
        # Usually checking exact column name logic is better
        # Let's try to extract standard component ID which is usually at start
        # Assume format: {ComponentID}_{Property}
        # But here properties are long.
        comp_name = parts # Simplification for now, works if property is suffix
        
        # Calculate total mass
        total_kg = df[col].sum() * dt_hours
        
        if total_kg > 0.01:
            stream_type = _detect_component_stream_type(df, comp_name)
            data[stream_type][comp_name] = total_kg
    
    # Flatten for plotting
    groups = []
    
    for stream, items in data.items():
        if items:
            sorted_items = sorted(items.items(), key=lambda x: x[1], reverse=True)
            for comp, val in sorted_items:
                groups.append({
                    'comp': comp,
                    'val': val,
                    'type': stream,
                    'color': 'tab:blue' if stream == 'H2' else 'tab:red' if stream == 'O2' else 'tab:gray'
                })
    
    if not groups:
        ax.text(0.5, 0.5, 'No significant water removal (> 0.01 kg).', ha='center', va='center')
        return fig
        
    # Plotting
    names = [g['comp'] for g in groups]
    values = [g['val'] for g in groups]
    colors = [g['color'] for g in groups]
    
    bars = ax.bar(names, values, color=colors, edgecolor='black', alpha=0.8)
    
    # Legend
    legend_elements = [
        mpatches.Rectangle((0,0),1,1, color='tab:blue', label='H2 Stream'),
        mpatches.Rectangle((0,0),1,1, color='tab:red', label='O2 Stream'),
        mpatches.Rectangle((0,0),1,1, color='tab:gray', label='Combined/Other')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Annotations
    for bar, val in zip(bars, values):
         ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                 f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
    ax.set_ylabel('Total Water Removed (kg)')
    ax.set_title('Cumulative Water Removal by Component', fontsize=12, pad=15)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    return fig

@log_graph_errors
def create_drains_discarded_figure(df: pd.DataFrame, dpi: int = DPI_FAST) -> Figure:
    """Create multi-panel plot for discarded drains."""
    fig = Figure(figsize=(12, 10), dpi=dpi, constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    
    drain_cols = [c for c in df.columns if 'water_removed_kg_h' in c.lower()]
    components = []
    total_mass = []
    avg_temp = []
    avg_press = []
    dt_hours = 1.0 / 60.0
    
    for col in drain_cols:
        comp_name = col.replace('_water_removed_kg_h', '').upper()
        mass_kg = df[col].sum() * dt_hours
        
        temp_col = [c for c in df.columns if comp_name.lower() in c.lower() and 'temp' in c.lower()]
        t_avg = df[temp_col[0]].mean() if temp_col else 25.0
        
        press_col = [c for c in df.columns if comp_name.lower() in c.lower() and 'press' in c.lower()]
        p_avg = df[press_col[0]].mean() if press_col else 1.0
        
        if mass_kg > 0:
            components.append(comp_name)
            total_mass.append(mass_kg)
            avg_temp.append(t_avg)
            avg_press.append(p_avg)
            
    if not components:
        logger.warning("No active discarded drains found.")
        ax1.text(0.5, 0.5, 'No active discarded drains found.', ha='center', va='center')
        return fig
    
    x = np.arange(len(components))
    width = 0.6
    
    ax1.bar(x, total_mass, width, color='darkred', alpha=0.7)
    ax1.set_ylabel('Total Mass (kg)')
    ax1.set_title('Discarded Drains Overview')
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    
    ax2.bar(x, avg_temp, width, color='orange', alpha=0.7)
    ax2.set_ylabel('Avg Temp (°C)')
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    
    ax3.bar(x, avg_press, width, color='purple', alpha=0.7)
    ax3.set_ylabel('Avg Pressure (bar)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(components, rotation=45, ha='right')
    ax3.grid(axis='y', linestyle='--', alpha=0.5)
    
    return fig


# ==============================================================================
# ENERGY & THERMAL ANALYSIS GRAPHS (From Plots.csv specification)
# ==============================================================================

@log_graph_errors
def create_energy_flows_figure(df: pd.DataFrame, dpi: int = DPI_FAST) -> Figure:
    """
    plot_fluxos_energia: Energy Flows & Consumption by component.
    
    Shows Heat (Q) and Work (W) for thermal/compression components.
    Grouped bar chart with H2/O2 stream distinction where applicable.
    """
    fig = Figure(figsize=(14, 8), dpi=dpi, constrained_layout=True)
    
    # Collect energy data from component columns
    heat_data = {}  # component -> total kWh
    work_data = {}  # component -> total kWh
    
    dt_hours = 1.0 / 60.0  # Minutes to hours
    
    # Heat sources: Chiller, HeatExchanger, DryCooler
    for col in df.columns:
        col_lower = col.lower()
        if 'cooling_load_kw' in col_lower or 'heat_removed_kw' in col_lower:
            comp = col.split('_')[0]
            heat_data[comp] = df[col].sum() * dt_hours
        elif 'tqc_duty_kw' in col_lower or 'dc_duty_kw' in col_lower:
            comp = col.split('_')[0]
            heat_data[comp] = heat_data.get(comp, 0) + df[col].sum() * dt_hours
            
    # Work sources: Compressor, Chiller electrical, Pump
    for col in df.columns:
        col_lower = col.lower()
        if 'compression_work_kwh' in col_lower or 'energy_consumed_kwh' in col_lower:
            comp = col.split('_')[0]
            work_data[comp] = df[col].sum()
        elif 'electrical_power_kw' in col_lower or 'fan_power_kw' in col_lower:
            comp = col.split('_')[0]
            work_data[comp] = work_data.get(comp, 0) + df[col].sum() * dt_hours
        elif col_lower.endswith('_power_kw'):
            # Capture compressor power_kw and convert to kWh
            comp = col.rsplit('_power_kw', 1)[0]
            work_data[comp] = work_data.get(comp, 0) + df[col].sum() * dt_hours
    
    if not heat_data and not work_data:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, 'No energy flow data in history.\nEnsure thermal/compression components expose get_state() metrics.',
                ha='center', va='center', transform=ax.transAxes, fontsize=12, color='gray')
        ax.set_title('Energy Flows & Consumption')
        return fig
    
    # Combine all components
    all_comps = sorted(set(heat_data.keys()) | set(work_data.keys()))
    x = np.arange(len(all_comps))
    width = 0.35
    
    ax = fig.add_subplot(111)
    
    heat_vals = [heat_data.get(c, 0) for c in all_comps]
    work_vals = [work_data.get(c, 0) for c in all_comps]
    
    bars_q = ax.bar(x - width/2, heat_vals, width, label='Heat Q (kWh)', color='salmon', edgecolor='darkred')
    bars_w = ax.bar(x + width/2, work_vals, width, label='Work W (kWh)', color='steelblue', edgecolor='darkblue')
    
    # Add value labels
    for bar in bars_q:
        if bar.get_height() > 0.1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.1f}',
                    ha='center', va='bottom', fontsize=8)
    for bar in bars_w:
        if bar.get_height() > 0.1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.1f}',
                    ha='center', va='bottom', fontsize=8)
    
    ax.set_ylabel('Energy (kWh)')
    ax.set_xlabel('Component')
    ax.set_title('Energy Flows & Consumption (Heat vs Work)')
    ax.set_xticks(x)
    ax.set_xticklabels(all_comps, rotation=30, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    return fig


@log_graph_errors
def create_plant_balance_schematic(df: pd.DataFrame, dpi: int = DPI_FAST) -> Figure:
    """
    Plant Balance Diagram (Control Volume).
    
    Replaces legacy `plot_esquema_planta_completa.py`.
    Shows aggregated mass/energy balance as a system-level schematic.
    Dynamically shows inputs/outputs based on available data.
    """
    fig = Figure(figsize=(14, 10), dpi=dpi, constrained_layout=True)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    dt_hours = 1.0 / 60.0  # Minutes to hours
    
    # --- Aggregate Metrics ---
    # H2 Product (sum of all H2 production columns)
    h2_cols = [c for c in df.columns if 'h2_production_kg' in c.lower() or 'h2_soec' in c.lower() or 'h2_pem' in c.lower()]
    h2_total = sum(df[c].sum() for c in h2_cols) if h2_cols else 0.0
    
    # O2 Product (from external O2 sources or O2 stream finals)
    o2_cols = [c for c in df.columns if 'o2_production' in c.lower() or 'outlet_mass' in c.lower() and 'o2' in c.lower()]
    o2_total = sum(df[c].sum() * dt_hours for c in o2_cols) if o2_cols else 0.0
    
    # Electric Power (SOEC + PEM consumption)
    power_cols = [c for c in df.columns if 'p_soec' in c.lower() or 'p_pem' in c.lower()]
    power_total = sum(df[c].sum() * dt_hours / 60.0 for c in power_cols) if power_cols else 0.0  # MWh
    
    # Heat Rejected
    heat_cols = [c for c in df.columns if 'cooling_load_kw' in c.lower() or 'heat_removed_kw' in c.lower()]
    heat_total = sum(df[c].sum() * dt_hours for c in heat_cols) if heat_cols else 0.0  # kWh
    
    # Water Makeup (from water source or tank inlet)
    water_in_cols = [c for c in df.columns if 'makeup' in c.lower() or 'water_source' in c.lower()]
    water_makeup = sum(df[c].sum() * dt_hours for c in water_in_cols) if water_in_cols else 0.0
    
    # Water Removed (condensed)
    water_out_cols = [c for c in df.columns if 'water_removed' in c.lower() or 'water_condensed' in c.lower()]
    water_removed = sum(df[c].sum() * dt_hours for c in water_out_cols) if water_out_cols else 0.0
    
    # --- Draw Control Volume ---
    cv_rect = mpatches.FancyBboxPatch((15, 25), 70, 55, boxstyle='round,pad=0.02',
                                       edgecolor='black', facecolor='#f0f8ff', linewidth=3)
    ax.add_patch(cv_rect)
    ax.text(50, 78, 'PLANT CONTROL VOLUME', ha='center', fontsize=14, fontweight='bold')
    
    # --- Inputs (Left Side) ---
    input_y = 65
    
    # Power Input
    if power_total > 0:
        ax.annotate('', xy=(15, input_y), xytext=(2, input_y),
                    arrowprops=dict(arrowstyle='->', color='gold', lw=2))
        ax.text(2, input_y + 2, f'Power: {power_total:.1f} MWh', fontsize=10, color='darkorange', fontweight='bold')
        input_y -= 12
    
    # Water Makeup
    if water_makeup > 0:
        ax.annotate('', xy=(15, input_y), xytext=(2, input_y),
                    arrowprops=dict(arrowstyle='->', color='cyan', lw=2))
        ax.text(2, input_y + 2, f'Makeup: {water_makeup:.1f} kg', fontsize=10, color='darkcyan', fontweight='bold')
    
    # --- Outputs (Right Side) ---
    output_y = 65
    
    # H2 Product
    if h2_total > 0:
        ax.annotate('', xy=(98, output_y), xytext=(85, output_y),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=3))
        ax.text(86, output_y + 2, f'H₂: {h2_total:.1f} kg', fontsize=11, color='darkblue', fontweight='bold')
        output_y -= 12
    
    # O2 Product
    if o2_total > 0:
        ax.annotate('', xy=(98, output_y), xytext=(85, output_y),
                    arrowprops=dict(arrowstyle='->', color='red', lw=3))
        ax.text(86, output_y + 2, f'O₂: {o2_total:.1f} kg', fontsize=11, color='darkred', fontweight='bold')
        output_y -= 12
    
    # --- Bottom Outputs ---
    # Heat Rejected
    if heat_total > 0:
        ax.annotate('', xy=(35, 10), xytext=(35, 25),
                    arrowprops=dict(arrowstyle='->', color='salmon', lw=2))
        ax.text(35, 5, f'Heat: {heat_total:.0f} kWh', ha='center', fontsize=10, color='red')
    
    # Water Removed
    if water_removed > 0:
        ax.annotate('', xy=(65, 10), xytext=(65, 25),
                    arrowprops=dict(arrowstyle='->', color='lightblue', lw=2))
        ax.text(65, 5, f'Water Out: {water_removed:.1f} kg', ha='center', fontsize=10, color='teal')
    
    # --- Title ---
    ax.set_title('Plant Mass & Energy Balance', fontsize=16, pad=20)
    
    # --- Legend ---
    legend_text = (
        f"Simulation Period: {len(df)} minutes\n"
        f"Total H₂ Produced: {h2_total:.2f} kg\n"
        f"Total Heat Rejected: {heat_total:.1f} kWh"
    )
    ax.text(50, 45, legend_text, ha='center', va='center', fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    return fig

@log_graph_errors
def create_mixer_comparison_figure(df: pd.DataFrame, dpi: int = DPI_FAST) -> Figure:
    """
    plot_drenos_mixer: Drain Mixer input/output comparison.
    
    Shows T, P, Flow properties from MultiComponentMixer or DrainRecorderMixer.
    Compares H2/O2 inputs vs final mixed output.
    """
    fig = Figure(figsize=(12, 10), dpi=dpi, constrained_layout=True)
    
    # Find mixer data
    mixer_temp = _find_component_columns(df, 'Mixer', 'temperature_k')
    mixer_press = _find_component_columns(df, 'Mixer', 'pressure_pa')
    mixer_flow = _find_component_columns(df, 'Mixer', 'outlet_mass_kg_h')
    
    # Also check DrainRecorder
    if not mixer_temp:
        mixer_temp = _find_component_columns(df, 'DrainRecorder', 'outlet_temp_k')
    if not mixer_flow:
        mixer_flow = _find_component_columns(df, 'DrainRecorder', 'outlet_mass_kg_h')
        
    # Check WaterMixer (Drain_Collector)
    if not mixer_temp:
        mixer_temp = _find_component_columns(df, 'Drain_Collector', 'outlet_temperature_k')
        if not mixer_temp:
             mixer_temp = _find_component_columns(df, 'WaterMixer', 'outlet_temperature_k')
             
    if not mixer_press:
        # Note: WaterMixer outputs kPa, convert to Pa for consistency
        press_kpa = _find_component_columns(df, 'Drain_Collector', 'outlet_pressure_kpa')
        if not press_kpa:
             press_kpa = _find_component_columns(df, 'WaterMixer', 'outlet_pressure_kpa')
        
        if press_kpa:
             mixer_press = {k: v * 1000.0 for k, v in press_kpa.items()}

    if not mixer_flow:
        mixer_flow = _find_component_columns(df, 'Drain_Collector', 'outlet_mass_flow_kg_h')
        if not mixer_flow:
             mixer_flow = _find_component_columns(df, 'WaterMixer', 'outlet_mass_flow_kg_h')
    
    if not mixer_temp and not mixer_flow:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, 'No mixer data in history.\nEnsure Mixer/DrainRecorderMixer components expose get_state() metrics.',
                ha='center', va='center', transform=ax.transAxes, fontsize=12, color='gray')
        ax.set_title('Drain Mixer Comparison')
        return fig
    
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312, sharex=ax1)
    ax3 = fig.add_subplot(313, sharex=ax1)
    
    x = df['minute'] if 'minute' in df.columns else df.index
    
    # Temperature
    for comp_id, data in mixer_temp.items():
        # Convert K to C if needed
        data_c = data - 273.15 if data.mean() > KELVIN_DETECTION_THRESHOLD else data
        ax1.plot(downsample_for_plot(x), downsample_for_plot(data_c), label=comp_id, linewidth=1.5)
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title('Mixer Outlet Temperature')
    if mixer_temp:
        ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Pressure
    for comp_id, data in mixer_press.items():
        # Convert Pa to bar
        data_bar = data / 1e5 if data.mean() > 1000 else data
        ax2.plot(downsample_for_plot(x), downsample_for_plot(data_bar), label=comp_id, linewidth=1.5)
    ax2.set_ylabel('Pressure (bar)')
    ax2.set_title('Mixer Operating Pressure')
    if mixer_press:
        ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # Flow
    for comp_id, data in mixer_flow.items():
        ax3.fill_between(downsample_for_plot(x), 0, downsample_for_plot(data), 
                         label=comp_id, alpha=0.6)
    ax3.set_ylabel('Mass Flow (kg/h)')
    ax3.set_xlabel('Time (Minutes)')
    ax3.set_title('Mixer Outlet Flow Rate')
    if mixer_flow:
        ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, linestyle='--', alpha=0.5)
    
    return fig


# ==============================================================================
# THERMAL & SEPARATION GRAPHS (Real implementations)
# ==============================================================================

@log_graph_errors
def _find_component_columns(df: pd.DataFrame, component_type: str, metric: str) -> Dict[str, pd.Series]:
    """
    Find all columns matching a component type and metric pattern.
    
    Args:
        df: History dataframe
        component_type: e.g., 'Chiller', 'KOD', 'Coalescer'
        metric: e.g., 'cooling_load_kw', 'water_removed_kg_h'
    
    Returns:
        Dict mapping component_id to data series
    """
    result = {}
    metric_lower = metric.lower()
    for col in df.columns:
        col_lower = col.lower()
        if component_type.lower() in col_lower and metric_lower in col_lower:
            # Extract component name using rsplit (more robust than replace)
            # e.g., "KOD_1_dissolved_gas_ppm" → "KOD_1"
            parts = col.rsplit(f"_{metric}", 1)
            if len(parts) == 2:
                comp_name = parts[0]
            else:
                # Try case-insensitive match
                parts = col_lower.rsplit(f"_{metric_lower}", 1)
                comp_name = col[:len(parts[0])] if len(parts) == 2 else col.split('_')[0]
            result[comp_name] = df[col]
    return result


@log_graph_errors
def create_individual_drains_figure(df: pd.DataFrame, dpi: int = DPI_FAST) -> Figure:
    """
    plot_drenos_individuais: Individual Drain Properties.
    
    Shows Mass Flow, Temperature, and Pressure for each component drain.
    Uses multi-panel layout with dual Y-axis for flow if magnitudes differ significantly.
    """
    fig = Figure(figsize=(12, 12), dpi=dpi, constrained_layout=True)
    
    # 1. Collect drain data
    # KOD
    kod_flow = _find_component_columns(df, 'KOD', 'water_removed_kg_h')
    kod_temp = _find_component_columns(df, 'KOD', 'drain_temp_k')
    kod_press = _find_component_columns(df, 'KOD', 'drain_pressure_bar')
    
    # Coalescer
    coal_flow = _find_component_columns(df, 'Coalescer', 'drain_flow_kg_h')
    coal_temp = _find_component_columns(df, 'Coalescer', 'drain_temp_k')
    coal_press = _find_component_columns(df, 'Coalescer', 'drain_pressure_bar')
    
    # Combine
    all_flows = {**kod_flow, **coal_flow}
    all_temps = {**kod_temp, **coal_temp}
    all_press = {**kod_press, **coal_press}
    
    if not all_flows:
        ax = fig.add_subplot(111)
        logger.warning("No individual drain data found.")
        ax.text(0.5, 0.5, 'No individual drain data found.\nEnsure KOD/Coalescer expose drain metrics (flow, T, P).',
                ha='center', va='center', transform=ax.transAxes, fontsize=12, color='gray')
        ax.set_title('Individual Drain Properties')
        return fig

    # 2. Setup plotting
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312, sharex=ax1)
    ax3 = fig.add_subplot(313, sharex=ax1)
    
    x = df['minute'] if 'minute' in df.columns else df.index
    
    # Panel 1: Mass Flow (Dual Axis possible, but starting with single for clarity)
    for comp_id, data in all_flows.items():
        if data.sum() > 0:
            ax1.plot(downsample_for_plot(x), downsample_for_plot(data), label=comp_id, linewidth=1.5)
    ax1.set_ylabel('Drain Flow (kg/h)')
    ax1.set_title('Drain Mass Flow Rate')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Panel 2: Temperature
    for comp_id, data in all_temps.items():
        # Convert K to C
        data_c = data - 273.15 if data.mean() > KELVIN_DETECTION_THRESHOLD else data
        if not data.isna().all() and data.sum() != 0:
             ax2.plot(downsample_for_plot(x), downsample_for_plot(data_c), label=comp_id, linewidth=1.5)
    ax2.set_ylabel('Temperature (°C)')
    ax2.set_title('Drain Stream Temperature')
    # ax2.legend(loc='upper right', fontsize=8) # Optional if too crowded
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # Panel 3: Pressure
    for comp_id, data in all_press.items():
        if not data.isna().all() and data.sum() != 0:
            ax3.plot(downsample_for_plot(x), downsample_for_plot(data), label=comp_id, linewidth=1.5)
    ax3.set_ylabel('Pressure (bar)')
    ax3.set_xlabel('Time (Minutes)')
    ax3.set_title('Drain Stream Pressure')
    # ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, linestyle='--', alpha=0.5)
    
    return fig

@log_graph_errors
def create_dissolved_gas_figure(df: pd.DataFrame, dpi: int = DPI_FAST) -> Figure:
    """
    plot_concentracao_dreno: Dissolved Gas Removal Efficiency.
    
    Plots the concentration of dissolved gases (ppm) in the drain streams.
    Comparing across different separation stages (KOD, Coalescer).
    """
    fig = Figure(figsize=(10, 6), dpi=dpi, constrained_layout=True)
    
    # Collect PPM data
    kod_ppm = _find_component_columns(df, 'KOD', 'dissolved_gas_ppm')
    coal_ppm = _find_component_columns(df, 'Coalescer', 'dissolved_gas_ppm')
    
    all_ppm = {**kod_ppm, **coal_ppm}
    
    if not all_ppm:
        ax = fig.add_subplot(111)
        logger.warning("No dissolved gas data found.")
        ax.text(0.5, 0.5, 'No dissolved gas data found.\nEnsure KOD/Coalescer calculate and expose ppm.',
                ha='center', va='center', transform=ax.transAxes, fontsize=12, color='gray')
        ax.set_title('Dissolved Gas Concentration')
        return fig
    
    ax = fig.add_subplot(111)
    x = df['minute'] if 'minute' in df.columns else df.index
    
    for comp_id, data in all_ppm.items():
        if data.sum() > 0:
            ax.plot(downsample_for_plot(x), downsample_for_plot(data), label=comp_id, linewidth=1.5)
            
    ax.set_ylabel('Concentration (ppm mg/kg)')
    ax.set_xlabel('Time (Minutes)')
    ax.set_title('Dissolved Gas Concentration in Drains')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    # Use log scale with lower limit to avoid zero issues
    ax.set_ylim(bottom=0.01)  # Clip to avoid log(0)
    ax.set_yscale('log')
    
    return fig


@log_graph_errors
def create_drain_concentration_figure(df: pd.DataFrame, dpi: int = DPI_FAST) -> Figure:
    """
    plot_concentracao_dreno: Dissolved Gas Removal Efficiency & Mixer.
    
    Compares:
    1. Aggregated IN (Pre-Flash)
    2. Aggregated OUT (Post-Flash/Separation)
    3. Final Mixer Output
    
    Shows removal efficiency and final quality.
    """
    fig = Figure(figsize=(10, 7), dpi=dpi, constrained_layout=True)
    ax = fig.add_subplot(111)
    
    # helper for finding mean
    def get_mean(pattern):
        cols = [c for c in df.columns if pattern in c]
        if not cols: return 0.0
        return sum(df[c].mean() for c in cols) # Sum if multiple parallel lines? Or mean of mean?
        # Logic: We are comparing concentration. 
        # Ideally weighted average by flow, but simple mean is legacy proxy if flows similar.
        # Let's use max found to be safe or average of active ones.
        vals = [df[c].mean() for c in cols if df[c].mean() > 0]
        return sum(vals) / len(vals) if vals else 0.0

    # 1. IN Concentration (Aggregated)
    # Search for known inlet concentration columns
    c_in_h2 = get_mean('H2_inlet_dissolved_ppm')
    c_in_o2 = get_mean('O2_inlet_dissolved_ppm')
    
    # If explicit columns missing, try to infer from upstream components?
    # Legacy hardcoded this. For dynamic, we rely on standard tracking.
    if c_in_h2 == 0 and c_in_o2 == 0:
        # Try finding KOD/Coalescer drain ppm as "IN" to the flash system
        # Actually in new KOD, output IS the drain. 
        # So KOD Drain PPM = IN to Mixer? 
        # Let's assume KOD output is "IN" to the treatment system.
        c_in_h2 = get_mean('KOD_1_dissolved_gas_ppm') + get_mean('Coalescer_H2_dissolved_gas_ppm')
        # Average it?
        c_in_h2 /= 2 if c_in_h2 > 0 else 1

    # 2. OUT Concentration (Post Flash / Tank)
    # In legacy this was after "Flash Drum".
    # In new topology, this might be "DrainRecorderMixer" BEFORE final mix?
    # Or just "WaterMixer" output?
    
    # Let's look for Mixer Output as "Final"
    # And maybe "Tank" output as "Intermediate"?
    
    # If tracking is sparse, we might only have Component Drains (IN) and Mixer Output (Final).
    # Legacy had 3 bars.
    
    # Let's compare: 
    # IN (Separators) -> OUT (Mixer)
    # If we have intermediate "Flash", use it.
    
    c_out_h2 = 0.0 # Placeholder for intermediate
    c_out_o2 = 0.0
    
    # 3. Final Mixer
    c_final_h2 = get_mean('Mixer_dissolved_H2_ppm')
    if c_final_h2 == 0: c_final_h2 = get_mean('dissolved_gas_ppm') # generic check? dangerous
    
    # Specific Mixer check
    mix_cols = [c for c in df.columns if 'Mixer' in c and 'dissolved' in c]
    if mix_cols:
         # Assume last mixer is final
         # Sort by name length or something?
         pass

    # RE-EVALUATION:
    # Use simpler approach matching typical data availability.
    # Group By GASEOUS SPECIES (H2 vs O2).
    
    # H2 Data
    h2_sources = [c for c in df.columns if 'H2' in c and 'dissolved_gas_ppm' in c and 'Mixer' not in c]
    h2_in_val = df[h2_sources].max(axis=1).mean() if h2_sources else 0.0 # Use max conc source?
    
    h2_mixer = [c for c in df.columns if 'H2' in c and 'dissolved' in c and 'Mixer' in c]
    h2_final_val = df[h2_mixer[0]].mean() if h2_mixer else 0.0
    
    # O2 Data
    o2_sources = [c for c in df.columns if 'O2' in c and 'dissolved_gas_ppm' in c and 'Mixer' not in c]
    o2_in_val = df[o2_sources].max(axis=1).mean() if o2_sources else 0.0
    
    o2_mixer = [c for c in df.columns if 'O2' in c and 'dissolved' in c and 'Mixer' in c]
    o2_final_val = df[o2_mixer[0]].mean() if o2_mixer else 0.0
    
    # Assemble Plot Data
    labels = ['H2 Dissolved', 'O2 Dissolved']
    in_vals = [h2_in_val, o2_in_val]
    mid_vals = [h2_in_val * 0.1, o2_in_val * 0.1] # Mock intermediate (Flash efficiency assumption if data missing)
    # Actually, if we don't have Flash data, maybe skip bar 2? 
    # Legacy had 'OUT' (Flash).
    # New system might strictly be Separator -> Mixer.
    # Let's plot Separator (IN) vs Mixer (OUT).
    
    x = np.arange(len(labels))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, in_vals, width, label='Component Drain (IN)', color='#1f77b4')
    bars2 = ax.bar(x + width/2,  [h2_final_val, o2_final_val], width, label='Final Mixer (OUT)', color='#2ca02c')
    
    # Max input for limit
    max_val = max(max(in_vals), max([h2_final_val, o2_final_val]))
    if max_val > 0:
        ax.set_ylim(0, max_val * 1.5)
        
    ax.set_ylabel('Concentration (ppm)')
    ax.set_title('Dissolved Gas Concentration (Source vs Discharge)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Labels
    def label_bars(bars):
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(f'{h:.4f}', xy=(bar.get_x() + bar.get_width()/2, h),
                           xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
    
    label_bars(bars1)
    label_bars(bars2)
    
    # Efficiency Label
    for i in range(2):
        inn = in_vals[i]
        out = [h2_final_val, o2_final_val][i]
        if inn > 0:
            eff = (inn - out) / inn
            ax.annotate(f'Removal: {eff:.1%}', xy=(x[i], max(inn, out)), xytext=(0, 15),
                       textcoords='offset points', ha='center', fontweight='bold', color='green')

    return fig
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    return fig

@log_graph_errors
def create_crossover_impurities_figure(df: pd.DataFrame, dpi: int = DPI_FAST, metadata: Dict[str, Any] = None, **kwargs) -> Figure:
    """
    Stream Impurity Profile Graph.
    
    Plots **time-averaged** impurity concentrations along a single process train.
    
    Args:
        df: DataFrame with simulation history.
        dpi: Plot resolution.
        stream_type: 'H2' or 'O2' - determines which impurities to show.
        components: Ordered list of component IDs.
    
    H2 stream shows: O2 impurity
    O2 stream shows: H2 impurity
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Extract configuration
    stream_type = kwargs.get('stream_type', 'H2')
    components = kwargs.get('components', [])
    
    # Fallback to legacy h2_components/o2_components for backward compat
    if not components:
        if stream_type == 'H2':
            components = kwargs.get('h2_components', [])
        else:
            components = kwargs.get('o2_components', [])
    
    fig = Figure(figsize=(12, 6), dpi=dpi, constrained_layout=True)
    
    # Default empty metadata if not provided
    if metadata is None:
        metadata = {}
    
    ax = fig.add_subplot(111)
    
    # -------------------------------------------------------------------------
    # 1. Configure based on stream_type
    # -------------------------------------------------------------------------
    if stream_type == 'H2':
        impurity_name = 'O₂'
        impurity_color = 'darkgreen'
        suffixes = ['_o2_impurity_ppm_mol', '_outlet_o2_ppm_mol', '_o2_ppm_mol', '_o2_ppm']
        limit_ppm = 5.0
        limit_label = 'Deoxo Limit (5 ppm)'
    else:
        impurity_name = 'H₂'
        impurity_color = 'darkorange'
        suffixes = ['_h2_impurity_ppm_mol', '_outlet_h2_ppm_mol', '_h2_ppm_mol', '_h2_ppm']
        limit_ppm = None
        limit_label = None
    
    # -------------------------------------------------------------------------
    # 2. Collect impurity data for specified components
    # -------------------------------------------------------------------------
    impurity_data = {}
    
    for comp_id in components:
        for suffix in suffixes:
            col = f"{comp_id}{suffix}"
            if col in df.columns:
                avg_val = df[col].mean()
                if avg_val >= 0:
                    impurity_data[comp_id] = avg_val
                break
    
    # -------------------------------------------------------------------------
    # 3. Handle empty data
    # -------------------------------------------------------------------------
    if not impurity_data:
        logger.warning(f"No {impurity_name} impurity data found.")
        ax.text(0.5, 0.5, f'No {impurity_name} impurity data found.\n'
                          f'Ensure components expose ppm_mol metrics.',
                ha='center', va='center', transform=ax.transAxes, fontsize=12, color='gray')
        ax.set_title(f'{impurity_name} Impurity in {stream_type} Stream')
        return fig
    
    # -------------------------------------------------------------------------
    # 4. Plot the impurity profile
    # -------------------------------------------------------------------------
    ordered_comps = [c for c in components if c in impurity_data]
    values = [impurity_data[c] for c in ordered_comps]
    
    x_pos = np.arange(len(ordered_comps))
    plot_values = [max(v, 1e-3) for v in values]
    
    ax.plot(x_pos, plot_values, marker='o', linewidth=2, color=impurity_color, 
            label=f'{impurity_name} Impurity (ppm)')
    
    if limit_ppm is not None:
        ax.axhline(limit_ppm, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=limit_label)
    
    for i, (comp, val) in enumerate(zip(ordered_comps, values)):
        if val < 0.01:
            label = '~0' if val == 0 else f'{val:.1e}'
        elif val < 1.0:
            label = f'{val:.2f}'
        else:
            label = f'{val:.1f}'
        ax.text(i, plot_values[i] * 1.1, label, ha='center', va='bottom', fontsize=8, color=impurity_color)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(ordered_comps, rotation=35, ha='right', fontsize=9)
    ax.set_ylabel(f'{impurity_name} Concentration (ppm molar)')
    ax.set_xlabel('Component (Process Order)')
    ax.set_title(f'{impurity_name} Impurity Profile in {stream_type} Stream')
    ax.grid(True, which='major', linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', fontsize=9)
    
    if values:
        y_max = max(values) * 1.4 if max(values) > 0 else 10
        ax.set_ylim(bottom=-0.5, top=y_max)
    
    return fig

@log_graph_errors
def create_drain_line_properties_figure(df: pd.DataFrame, dpi: int = DPI_FAST) -> Figure:
    """
    plot_propriedades_linha_dreno: Mixed Drain Line Properties.
    
    Plots the properties of the aggregated drain stream (after collection):
    - Mass Flow (Top)
    - Temperature (Middle)
    - Pressure (Bottom)
    
    Target Component: 'Drain_Collector' (WaterMixer)
    """
    fig = Figure(figsize=(10, 10), dpi=dpi, constrained_layout=True)
    
    # 1. Collect Data using helper
    mass_data = _find_component_columns(df, 'Drain_Collector', 'outlet_mass_flow_kg_h')
    temp_data = _find_component_columns(df, 'Drain_Collector', 'outlet_temperature_c')
    pres_data = _find_component_columns(df, 'Drain_Collector', 'outlet_pressure_kpa')
    
    # Also try finding by Drain_Mixer
    if not mass_data:
         mass_data = _find_component_columns(df, 'Drain_Mixer', 'outlet_mass_flow_kg_h')
         temp_data = _find_component_columns(df, 'Drain_Mixer', 'outlet_temperature_c')
         pres_data = _find_component_columns(df, 'Drain_Mixer', 'outlet_pressure_kpa')
         
    if not mass_data or not temp_data:
        ax = fig.add_subplot(111)
        logger.warning("No aggregated drain data found.")
        ax.text(0.5, 0.5, 'No aggregated drain data found.\nEnsure "Drain_Collector" (WaterMixer) exists and has flow.',
                ha='center', va='center', transform=ax.transAxes, fontsize=12, color='gray')
        ax.set_title('Drain Line Properties')
        return fig
    
    x = df['minute'] if 'minute' in df.columns else df.index
    
    # 2. Plotting (3 Panels)
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312, sharex=ax1)
    ax3 = fig.add_subplot(313, sharex=ax1)
    
    # Mass Flow
    for cid, data in mass_data.items():
        if data.any():
            ax1.plot(downsample_for_plot(x), downsample_for_plot(data), label=f"{cid} Flow", color='blue')
    ax1.set_ylabel('Mass Flow (kg/h)')
    ax1.set_title('Aggregated Drain Flow')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()
    
    # Temperature
    for cid, data in temp_data.items():
        if data.any():
            ax2.plot(downsample_for_plot(x), downsample_for_plot(data), label=f"{cid} Temp", color='red')
    ax2.set_ylabel('Temperature (°C)')
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # Pressure
    for cid, data in pres_data.items():
        if data.any():
            # Convert kPa to bar
            data_bar = data / 100.0
            ax3.plot(downsample_for_plot(x), downsample_for_plot(data_bar), label=f"{cid} Pressure", color='green')
    ax3.set_ylabel('Pressure (bar)')
    ax3.set_xlabel('Time (Minutes)')
    ax3.grid(True, linestyle='--', alpha=0.5)
    
    return fig

@log_graph_errors
def create_thermal_load_breakdown_figure(df: pd.DataFrame, dpi: int = DPI_FAST) -> Figure:
    """
    plot_q_breakdown: Thermal Load Breakdown.
    
    Plots:
    1. Cooling Load by Component (Stacked)
    2. Sensible vs Latent Heat Split (Chillers)
    """
    fig = Figure(figsize=(10, 8), dpi=dpi, constrained_layout=True)
    
    # 1. Collect Data
    # Chillers
    chiller_load = _find_component_columns(df, 'Chiller', 'cooling_load_kw')
    chiller_sensible = _find_component_columns(df, 'Chiller', 'sensible_heat_kw')
    chiller_latent = _find_component_columns(df, 'Chiller', 'latent_heat_kw')
    
    # Dry Coolers (Count as Sensible)
    dc_tqc = _find_component_columns(df, 'DryCooler', 'tqc_duty_kw')
    
    if not chiller_load and not dc_tqc:
        ax = fig.add_subplot(111)
        logger.warning("No thermal load data found.")
        ax.text(0.5, 0.5, 'No thermal load data found (Chiller/DryCooler).',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Thermal Load Breakdown')
        return fig
    
    x = df['minute'] if 'minute' in df.columns else df.index
    
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212, sharex=ax1)
    
    # Panel 1: Component Breakdown (Stacked Area)
    # Combine all sources
    all_loads = {}
    for k, v in chiller_load.items():
        all_loads[f"{k} (Chiller)"] = v
    for k, v in dc_tqc.items():
        all_loads[f"{k} (TQC)"] = v
    
    if all_loads:
        labels = list(all_loads.keys())
        data = [downsample_for_plot(all_loads[k]) for k in labels]
        x_ds = downsample_for_plot(x)
        
        ax1.stackplot(x_ds, *data, labels=labels, alpha=0.7)
        ax1.set_ylabel('Cooling Load (kW)')
        ax1.set_title('Process Cooling Load by Component')
        ax1.legend(loc='upper left')
        ax1.grid(True, linestyle='--', alpha=0.5)
        
    # Panel 2: Sensible vs Latent (System Total)
    total_sensible = pd.Series(0.0, index=df.index)
    total_latent = pd.Series(0.0, index=df.index)
    
    # Sum Chiller components
    for v in chiller_sensible.values():
        total_sensible = total_sensible.add(v, fill_value=0)
    for v in chiller_latent.values():
        total_latent = total_latent.add(v, fill_value=0)
        
    # Add Dry Cooler (Pure Sensible)
    for v in dc_tqc.values():
        total_sensible = total_sensible.add(v, fill_value=0)
        
    if total_sensible.any() or total_latent.any():
        x_ds = downsample_for_plot(x)
        y_sens = downsample_for_plot(total_sensible)
        y_lat = downsample_for_plot(total_latent)
        
        ax2.stackplot(x_ds, y_sens, y_lat, labels=['Sensible', 'Latent'], 
                      colors=['tab:blue', 'tab:orange'], alpha=0.7)
        ax2.set_ylabel('Heat Type (kW)')
        ax2.set_xlabel('Time (Minutes)')
        ax2.set_title('System Heat Removal: Sensible vs Latent')
        ax2.legend(loc='upper left')
        ax2.grid(True, linestyle='--', alpha=0.5)
        
    return fig

@log_graph_errors
def create_drain_concentration_figure(df: pd.DataFrame, dpi: int = DPI_FAST) -> Figure:
    """
    plot_concentracao_linha_dreno: Dissolved Gas Tracking.
    
    Plots the concentration of dissolved gases (ppm) in the aggregated drain line.
    """
    fig = Figure(figsize=(10, 6), dpi=dpi, constrained_layout=True)
    
    # 1. Collect Data
    # Look for Drain_Collector or fallback to Drain_Mixer
    ppm_data = _find_component_columns(df, 'Drain_Collector', 'dissolved_gas_ppm')
    if not ppm_data:
        ppm_data = _find_component_columns(df, 'Drain_Mixer', 'dissolved_gas_ppm')
    
    if not ppm_data:
        # Fallbacks
        ppm_data = _find_component_columns(df, 'WaterMixer', 'dissolved_gas_ppm')
        
    x = df['minute'] if 'minute' in df.columns else df.index
    
    ax = fig.add_subplot(111)
    
    if not ppm_data:
        logger.warning("No dissolved gas PPM data found in Drain_Collector.")
        ax.text(0.5, 0.5, 'No dissolved gas PPM data found in Drain_Collector.',
                ha='center', va='center', transform=ax.transAxes, color='gray')
    else:
        for cid, data in ppm_data.items():
            if data.any():
                ax.plot(downsample_for_plot(x), downsample_for_plot(data), label=f"{cid} Dissolved Gas")
                
        ax.set_ylabel('Concentration (ppm)')
        ax.set_xlabel('Time (Minutes)')
        ax.set_title('Dissolved Gas in Aggregated Drain')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        
    return fig

@log_graph_errors
def create_recirculation_comparison_figure(df: pd.DataFrame, dpi: int = DPI_FAST) -> Figure:
    """
    plot_recirculacao_mixer: Water Recovery System Comparison.
    
    Compares the state of water BEFORE recirculation (Drain_Collector output)
    and AFTER replenishment (WaterTank/Mixer output).
    
    Categorical Snapshot (Recovered vs Recirculated).
    """
    fig = Figure(figsize=(8, 10), dpi=dpi, constrained_layout=True)
    
    # 1. Collect Data (Averages)
    
    # --- State 1: Recovered (Drain Collector Output) ---
    rec_flow = 0.0
    rec_temp = 25.0
    rec_press = 1.0 # bar
    
    # Flow
    rf = _find_component_columns(df, 'Drain_Collector', 'outlet_mass_flow_kg_h')
    if not rf: rf = _find_component_columns(df, 'Drain_Mixer', 'outlet_mass_flow_kg_h')
    if rf: rec_flow = sum(v.mean() for v in rf.values() if v.mean() > 0)
    
    # Temp
    rt = _find_component_columns(df, 'Drain_Collector', 'outlet_temperature_c')
    if not rt: rt = _find_component_columns(df, 'Drain_Mixer', 'outlet_temperature_c')
    # Fallback K
    if not rt:
        rt_k = _find_component_columns(df, 'Drain_Collector', 'outlet_temperature_k')
        if rt_k: rt = {k: v - 273.15 for k, v in rt_k.items()}
    if rt: rec_temp = sum(v.mean() for v in rt.values()) / len(rt)
    
    # Pressure (kPa usually for water mixer)
    rp = _find_component_columns(df, 'Drain_Collector', 'outlet_pressure_kpa')
    if not rp: rp = _find_component_columns(df, 'Drain_Mixer', 'outlet_pressure_kpa')
    if rp: rec_press = (sum(v.mean() for v in rp.values()) / len(rp)) / 100.0 # kPa -> bar
    else: 
        # Check for bar
         rp_b = _find_component_columns(df, 'Drain_Collector', 'pressure_bar')
         if rp_b: rec_press = sum(v.mean() for v in rp_b.values()) / len(rp_b)

    # --- State 2: Recirculated (Feed Tank / Makeup Out) ---
    feed_flow = 0.0
    feed_temp = 25.0
    feed_press = 1.0
    
    ff = _find_component_columns(df, 'WaterTank', 'mass_flow_out_kg_h')
    if not ff: ff = _find_component_columns(df, 'Feed_Tank', 'mass_flow_out_kg_h')
    if not ff: ff = _find_component_columns(df, 'MakeupMixer', 'outlet_mass_flow_kg_h')
    if ff: feed_flow = sum(v.mean() for v in ff.values() if v.mean() > 0)
    else: feed_flow = rec_flow * 1.05 # Mock makeup if missing? Or just 0.
    
    ft = _find_component_columns(df, 'WaterTank', 'temperature_c')
    if not ft: ft = _find_component_columns(df, 'MakeupMixer', 'outlet_temperature_c')
    if ft: feed_temp = sum(v.mean() for v in ft.values()) / len(ft)
    
    fp_b = _find_component_columns(df, 'WaterTank', 'pressure_bar')
    if fp_b: feed_press = sum(v.mean() for v in fp_b.values()) / len(fp_b)
    
    # Prepare Plot Data
    states = ['Recovered (Drain)', 'Recirculated (Feed)']
    x = np.arange(len(states))
    
    flows = [rec_flow, feed_flow]
    temps = [rec_temp, feed_temp]
    press = [rec_press, feed_press]

    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312, sharex=ax1)
    ax3 = fig.add_subplot(313, sharex=ax1)
    
    # 1. Flow
    ax1.plot(x, flows, marker='o', linestyle='-', color='darkblue', linewidth=2, markersize=8)
    ax1.set_ylabel('Mass Flow (kg/h)')
    ax1.set_title('Water Recovery: Mass Flow')
    ax1.grid(True, linestyle='--', alpha=0.5)
    for i, v in enumerate(flows):
        ax1.annotate(f'{v:.1f}', (i, v), xytext=(0, 5), textcoords='offset points', ha='center', fontweight='bold')

    # 2. Pressure
    ax2.plot(x, press, marker='s', linestyle='-', color='darkorange', linewidth=2, markersize=8)
    ax2.set_ylabel('Pressure (bar)')
    ax2.set_title('Water Recovery: Pressure')
    ax2.grid(True, linestyle='--', alpha=0.5)
    for i, v in enumerate(press):
        ax2.annotate(f'{v:.2f}', (i, v), xytext=(0, 5), textcoords='offset points', ha='center')

    # 3. Temperature
    ax3.plot(x, temps, marker='^', linestyle='-', color='darkgreen', linewidth=2, markersize=8)
    ax3.set_ylabel('Temperature (°C)')
    ax3.set_title('Water Recovery: Temperature')
    ax3.set_xticks(x)
    ax3.set_xticklabels(states)
    ax3.grid(True, linestyle='--', alpha=0.5)
    for i, v in enumerate(temps):
        ax3.annotate(f'{v:.1f}', (i, v), xytext=(0, 5), textcoords='offset points', ha='center')

    fig.suptitle('Recirculation System Comparison (Average)', fontsize=14)
    return fig

@log_graph_errors
def create_entrained_liquid_figure(df: pd.DataFrame, dpi: int = DPI_FAST) -> Figure:
    """
    plot_vazao_liquida_acompanhante: Entrained Liquid Flow.
    
    Plots the entrained liquid mass flow (kg/h) in gas streams, typically
    exiting separators like KODs or Coalescers (Liquid Carryover).
    """
    fig = Figure(figsize=(10, 6), dpi=dpi, constrained_layout=True)
    
    x = df['minute'] if 'minute' in df.columns else df.index
    
    # 1. Collect Data
    # Search for 'm_dot_H2O_liq_accomp_kg_s' in ANY component
    # Convert to kg/h for plotting
    
    entrained_liq = {}
    
    # Identify all columns ending in m_dot_H2O_liq_accomp_kg_s
    # Format usually: {ComponentID}_m_dot_H2O_liq_accomp_kg_s
    suffix = 'm_dot_H2O_liq_accomp_kg_s'
    
    for col in df.columns:
        if col.endswith(suffix):
            # Extract component ID
            # Assuming col is like "KOD_1_m_dot_H2O_liq_accomp_kg_s"
            # But graph_builder flattens dicts. 
            # If KOD returns {..., 'm_dot_H2O_liq_accomp_kg_s': ...}, 
            # column should be "{CID}_{suffix}" or similar based on flattening logic.
            
            # Let's clean the name for the legend
            cid = col.replace(f"_{suffix}", "")
            entrained_liq[cid] = df[col] * 3600.0 # Convert kg/s -> kg/h
            
    ax = fig.add_subplot(111)
    
    if not entrained_liq:
        logger.warning("No entrained liquid data found.")
        ax.text(0.5, 0.5, 'No entrained liquid data found (m_dot_H2O_liq_accomp_kg_s).',
                ha='center', va='center', transform=ax.transAxes, color='gray')
    else:
        for cid, data in entrained_liq.items():
            if data.any():
                ax.plot(downsample_for_plot(x), downsample_for_plot(data), 
                        label=f"{cid} Entrainment", marker='.', linestyle='-')
                
        ax.set_ylabel('Liquid Flow (kg/h)')
        ax.set_xlabel('Time (Minutes)')
        ax.set_title('Entrained Liquid Carryover')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        # Use scientific notation if values are very small
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
        
    return fig

@log_graph_errors
def create_chiller_cooling_figure(df: pd.DataFrame, dpi: int = DPI_FAST) -> Figure:
    """
    Chiller cooling load and electrical consumption over time.
    
    Displays cooling load (kW) and electrical power for all chillers found in history.
    """
    fig = Figure(figsize=(12, 8), dpi=dpi, constrained_layout=True)
    
    # Find chiller data columns
    cooling_data = _find_component_columns(df, 'Chiller', 'cooling_load_kw')
    elec_data = _find_component_columns(df, 'Chiller', 'electrical_power_kw')
    
    if not cooling_data:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, 'No chiller data in history.\nEnsure Chiller components expose get_state() metrics.', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12, color='gray')
        ax.set_title('Chiller Cooling Load')
        return fig
    
    # Create 2-panel plot
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212, sharex=ax1)
    
    x = df['minute'] if 'minute' in df.columns else df.index
    
    # Panel 1: Cooling Load
    for comp_id, data in cooling_data.items():
        ax1.plot(downsample_for_plot(x), downsample_for_plot(data), 
                 label=comp_id, linewidth=1.5)
    ax1.set_ylabel('Cooling Load (kW)')
    ax1.set_title('Chiller Cooling Load Over Time')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Panel 2: Electrical Power
    for comp_id, data in elec_data.items():
        ax2.plot(downsample_for_plot(x), downsample_for_plot(data), 
                 label=comp_id, linestyle='--', linewidth=1.5)
    ax2.set_ylabel('Electrical Power (kW)')
    ax2.set_xlabel('Time (Minutes)')
    ax2.set_title('Chiller Electrical Consumption (COP-based)')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    return fig


@log_graph_errors
def create_coalescer_separation_figure(df: pd.DataFrame, dpi: int = DPI_FAST) -> Figure:
    """
    Coalescer pressure drop and liquid removal over time.
    """
    fig = Figure(figsize=(12, 8), dpi=dpi, constrained_layout=True)
    
    # Find coalescer data columns
    delta_p_data = _find_component_columns(df, 'Coalescer', 'delta_p_bar')
    drain_data = _find_component_columns(df, 'Coalescer', 'drain_flow_kg_h')
    
    if not delta_p_data and not drain_data:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, 'No coalescer data in history.\nEnsure Coalescer components expose get_state() metrics.', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12, color='gray')
        ax.set_title('Coalescer Separation')
        return fig
    
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212, sharex=ax1)
    
    x = df['minute'] if 'minute' in df.columns else df.index
    
    # Panel 1: Pressure Drop
    for comp_id, data in delta_p_data.items():
        ax1.plot(downsample_for_plot(x), downsample_for_plot(data), 
                 label=comp_id, linewidth=1.5, color=COLORS.get('coalescer', 'green'))
    ax1.set_ylabel('Pressure Drop (bar)')
    ax1.set_title('Coalescer Pressure Drop')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Panel 2: Drain Flow
    for comp_id, data in drain_data.items():
        ax2.fill_between(downsample_for_plot(x), 0, downsample_for_plot(data), 
                         label=comp_id, alpha=0.5)
    ax2.set_ylabel('Liquid Drain (kg/h)')
    ax2.set_xlabel('Time (Minutes)')
    ax2.set_title('Coalescer Liquid Removal')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    return fig


@log_graph_errors
def create_kod_separation_figure(df: pd.DataFrame, dpi: int = DPI_FAST) -> Figure:
    """
    Knock-Out Drum gas density, velocity, and water drainage.
    """
    fig = Figure(figsize=(12, 10), dpi=dpi, constrained_layout=True)
    
    # Find KOD data columns
    rho_data = _find_component_columns(df, 'KOD', 'rho_g')
    v_data = _find_component_columns(df, 'KOD', 'v_real')
    water_data = _find_component_columns(df, 'KOD', 'water_removed_kg_h')
    
    if not rho_data and not v_data and not water_data:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, 'No KOD data in history.\nEnsure KnockOutDrum components expose get_state() metrics.', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12, color='gray')
        ax.set_title('Knock-Out Drum Separation')
        return fig
    
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312, sharex=ax1)
    ax3 = fig.add_subplot(313, sharex=ax1)
    
    x = df['minute'] if 'minute' in df.columns else df.index
    
    # Panel 1: Gas Density
    for comp_id, data in rho_data.items():
        ax1.plot(downsample_for_plot(x), downsample_for_plot(data), 
                 label=comp_id, linewidth=1.5)
    ax1.set_ylabel('Gas Density (kg/m³)')
    ax1.set_title('KOD Gas Phase Density')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Panel 2: Velocity
    for comp_id, data in v_data.items():
        ax2.plot(downsample_for_plot(x), downsample_for_plot(data), 
                 label=comp_id, linewidth=1.5, color=COLORS.get('kod', 'blue'))
    ax2.set_ylabel('Superficial Velocity (m/s)')
    ax2.set_title('KOD Actual Velocity vs Design')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # Panel 3: Water Removal
    for comp_id, data in water_data.items():
        ax3.fill_between(downsample_for_plot(x), 0, downsample_for_plot(data), 
                         label=comp_id, alpha=0.6, color=COLORS.get('kod', 'lightblue'))
    ax3.set_ylabel('Water Removed (kg/h)')
    ax3.set_xlabel('Time (Minutes)')
    ax3.set_title('KOD Liquid Drainage Rate')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, linestyle='--', alpha=0.5)
    
    return fig


@log_graph_errors
def create_dry_cooler_figure(df: pd.DataFrame, dpi: int = DPI_FAST) -> Figure:
    """
    Dry cooler performance: heat rejection and outlet temperature.
    
    Note: Dry cooler may use similar state keys to Chiller. 
    Adjust column patterns if component uses different naming.
    """
    fig = Figure(figsize=(12, 8), dpi=dpi, constrained_layout=True)
    
    # Try to find DryCooler specific columns, or fall back to generic cooler patterns
    heat_rejected = _find_component_columns(df, 'DryCooler', 'heat_rejected_kw')
    outlet_temp = _find_component_columns(df, 'DryCooler', 'outlet_temp_k')
    
    if not heat_rejected and not outlet_temp:
        # Fallback: check for generic 'dry' or 'cooler' patterns
        heat_rejected = _find_component_columns(df, 'Dry', 'heat')
        outlet_temp = _find_component_columns(df, 'Dry', 'temp')
    
    if not heat_rejected and not outlet_temp:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, 'No dry cooler data in history.\nEnsure DryCooler components expose get_state() metrics.', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12, color='gray')
        ax.set_title('Dry Cooler Performance')
        return fig
    
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212, sharex=ax1)
    
    x = df['minute'] if 'minute' in df.columns else df.index
    
    # Panel 1: Heat Rejected
    for comp_id, data in heat_rejected.items():
        ax1.plot(downsample_for_plot(x), downsample_for_plot(data), 
                 label=comp_id, linewidth=1.5, color='darkorange')
    ax1.set_ylabel('Heat Rejected (kW)')
    ax1.set_title('Dry Cooler Heat Rejection')
    if heat_rejected:  # Only add legend if we plotted data
        ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Panel 2: Outlet Temperature
    for comp_id, data in outlet_temp.items():
        # Convert K to °C if needed
        data_c = data - 273.15 if data.mean() > KELVIN_DETECTION_THRESHOLD else data
        ax2.plot(downsample_for_plot(x), downsample_for_plot(data_c), 
                 label=comp_id, linewidth=1.5, color='teal')
    ax2.set_ylabel('Outlet Temperature (°C)')
    ax2.set_xlabel('Time (Minutes)')
    ax2.set_title('Dry Cooler Outlet Temperature')
    if outlet_temp:  # Only add legend if we plotted data
        ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    return fig


# ==============================================================================
# PROFILE PLOTTING (Merged from profile_plotter.py)
# ==============================================================================

@log_graph_errors
def create_process_train_profile_figure(df: pd.DataFrame, dpi: int = DPI_HIGH) -> Figure:
    """
    Generates a multi-panel line graph describing properties after each component.
    
    Args:
        df: DataFrame containing profile data (NOT history data).
            Expected keys: 'Component', 'T_c', 'P_bar', 'H_kj_kg', 'S_kj_kgK', 
            'MassFrac_H2', 'MassFrac_O2', 'MassFrac_H2O'.
    """
    # Profile data is passed as df, but typically contains one row per component.
    # It is DIFFERENT from history df which is time-series.
    # The caller must provide the correct DataFrame.
    
    if df.empty:
        fig = Figure(figsize=(10, 6), dpi=dpi)
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, 'No profile data available.', ha='center', va='center', transform=ax.transAxes, fontsize=12, color='gray')
        ax.set_title('Process Train Profile')
        return fig

    components = df['Component'].tolist()
    x_indices = range(len(components))
    scenario_name = df.attrs.get('scenario_name', 'Scenario')

    fig = Figure(figsize=(12, 12), dpi=dpi)
    # Replicate layout: 3 rows share x
    gs = fig.add_gridspec(3, 1)
    ax_t = fig.add_subplot(gs[0])
    ax_thermo = fig.add_subplot(gs[1], sharex=ax_t)
    ax_comp = fig.add_subplot(gs[2], sharex=ax_t)
    
    # Helper to annotate
    def annotate_points(ax, x_vals, y_vals, fmt="{:.1f}", offset=(0, 5), color='black'):
        for x, y in zip(x_vals, y_vals):
             if pd.notnull(y):
                ax.annotate(fmt.format(y), xy=(x, y), xytext=offset, 
                            textcoords='offset points', ha='center', va='bottom', 
                            fontsize=8, color=color, bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"))

    # --- Panel 1: Temperature ---
    ax_t.plot(x_indices, df['T_c'], marker='o', linestyle='-', color='tab:red', linewidth=2)
    annotate_points(ax_t, x_indices, df['T_c'], fmt="{:.1f}", color='tab:red')
    ax_t.set_ylabel('Temperature (°C)')
    ax_t.set_title(f'Temperature Profile - {scenario_name}')
    ax_t.grid(True, linestyle='--', alpha=0.7)
    
    # --- Panel 2: Thermodynamics (P, H, S) ---
    ln1 = ax_thermo.plot(x_indices, df['P_bar'], marker='s', color='tab:blue', label='Pressure (bar)')
    annotate_points(ax_thermo, x_indices, df['P_bar'], fmt="{:.1f}", offset=(0, 5), color='tab:blue')
    ax_thermo.set_ylabel('Pressure (bar)', color='tab:blue')
    ax_thermo.tick_params(axis='y', labelcolor='tab:blue')
    
    ax_thermo2 = ax_thermo.twinx()
    ln2 = ax_thermo2.plot(x_indices, df['H_kj_kg'], marker='^', color='tab:green', label='Enthalpy (kJ/kg)')
    annotate_points(ax_thermo2, x_indices, df['H_kj_kg'], fmt="{:.0f}", offset=(0, 15), color='tab:green')
    
    ln3 = ax_thermo2.plot(x_indices, df['S_kj_kgK'], marker='x', linestyle='--', color='tab:purple', label='Entropy (kJ/kg·K)')
    annotate_points(ax_thermo2, x_indices, df['S_kj_kgK'], fmt="{:.2f}", offset=(0, -15), color='tab:purple')
    
    ax_thermo2.set_ylabel('Enthalpy / Entropy', color='tab:green')
    ax_thermo2.tick_params(axis='y', labelcolor='tab:green')
    
    lns = ln1+ln2+ln3
    labs = [l.get_label() for l in lns]
    ax_thermo.legend(lns, labs, loc='upper right')
    ax_thermo.set_title('Thermodynamic Profile')
    ax_thermo.grid(True, linestyle='--', alpha=0.7)

    # --- Panel 3: Composition ---
    ax_comp.plot(x_indices, df['MassFrac_H2']*100, marker='.', label='H2 %')
    ax_comp.plot(x_indices, df['MassFrac_O2']*100, marker='.', label='O2 %')
    ax_comp.plot(x_indices, df['MassFrac_H2O']*100, marker='.', label='H2O %')
    
    annotate_points(ax_comp, x_indices, df['MassFrac_H2O']*100, fmt="{:.2f}", offset=(0, 5), color='brown')
    
    if df['MassFrac_O2'].mean() > 0.5:
         annotate_points(ax_comp, x_indices, df['MassFrac_O2']*100, fmt="{:.1f}", offset=(0, -15), color='orange')
    else:
         annotate_points(ax_comp, x_indices, df['MassFrac_H2']*100, fmt="{:.1f}", offset=(0, -15), color='blue')

    ax_comp.set_ylabel('Mass Fraction (%)')
    ax_comp.set_title('Composition')
    ax_comp.legend()
    ax_comp.grid(True, linestyle='--', alpha=0.7)
    
    ax_comp.set_xticks(x_indices)
    ax_comp.set_xticklabels(components, rotation=15, ha='right')
    
    fig.tight_layout()
    return fig


@log_graph_errors
def create_water_vapor_tracking_figure(df: pd.DataFrame, dpi: int = DPI_FAST) -> Figure:
    """
    plot_vazao_agua_separada: Water Vapor Flow Tracking.
    
    Tracks water vapor mass flow (kg/h) across components with PPM concentration labels.
    Includes limit line for purity specification (100 ppm).
    
    Uses time-averaged values from dynamic simulation.
    """
    fig = Figure(figsize=(12, 6), dpi=dpi, constrained_layout=True)
    ax = fig.add_subplot(111)
    
    # Find water vapor columns
    vapor_cols = [c for c in df.columns if 'h2o_vapor' in c.lower() and 'kg_h' in c.lower()]
    molar_cols = [c for c in df.columns if 'molar_fraction_h2o' in c.lower()]
    
    if not vapor_cols:
        ax.text(0.5, 0.5, 'No water vapor data available.\nEnsure components expose h2o_vapor_kg_h metric.', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12, color='gray')
        ax.set_title('Water Vapor Tracking')
        return fig
    
    # Extract component names and values (time-averaged)
    components = []
    vapor_flows = []
    ppm_values = []
    
    for col in sorted(vapor_cols):
        comp_name = col.replace('_h2o_vapor_kg_h', '').replace('_H2O_vapor_kg_h', '')
        components.append(comp_name)
        vapor_flows.append(df[col].mean())  # Time-average
        
        # Find matching molar fraction column
        matching_ppm = [c for c in molar_cols if comp_name in c]
        if matching_ppm:
            ppm_values.append(df[matching_ppm[0]].mean() * 1e6)  # Convert to ppm
        else:
            ppm_values.append(None)
    
    x = range(len(components))
    
    # Plot vapor flow line
    stream_color = 'tab:blue'  # Default to H2 stream color
    ax.plot(x, vapor_flows, marker='o', linestyle='-', color=stream_color, linewidth=2, markersize=8)
    
    # Add PPM labels
    for i, (flow, ppm) in enumerate(zip(vapor_flows, ppm_values)):
        if ppm is not None and ppm > 0:
            label = f'{ppm:.1f} ppm' if ppm >= 1 else f'{ppm:.2e} ppm'
            ax.annotate(label, xy=(i, flow), xytext=(0, 10), textcoords='offset points',
                       ha='center', fontsize=8, color='purple')
    
    # Add 100 ppm limit line (typical VSA/Dryer exit spec)
    # Calculate equivalent flow at 100 ppm for reference
    LIMIT_PPM = 100
    ax.axhline(y=0, color='green', linestyle='--', alpha=0.7, label=f'Target: {LIMIT_PPM} ppm')
    
    ax.set_xticks(x)
    ax.set_xticklabels(components, rotation=30, ha='right')
    ax.set_ylabel('Water Vapor Flow (kg/h)')
    ax.set_xlabel('Component')
    ax.set_title('Water Vapor Mass Flow Tracking (with PPM Labels)', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    return fig


@log_graph_errors
def create_total_mass_flow_figure(df: pd.DataFrame, dpi: int = DPI_FAST) -> Figure:
    """
    plot_vazao_massica_total_e_removida: Total Mass Flow Comparison.
    
    Compares gas, vapor, liquid, and total mass flows across components.
    Uses time-averaged values from dynamic simulation.
    """
    fig = Figure(figsize=(12, 7), dpi=dpi, constrained_layout=True)
    ax = fig.add_subplot(111)
    
    # Find flow columns - use 'mass_flow' (actual pattern, not 'gas_flow')
    gas_cols = [c for c in df.columns if 'mass_flow' in c.lower() and 'kg_h' in c.lower()]
    vapor_cols = [c for c in df.columns if 'h2o_vapor' in c.lower() and 'kg_h' in c.lower()]
    liquid_cols = [c for c in df.columns if 'm_dot_h2o_liq_accomp' in c.lower()]
    
    if not gas_cols:
        ax.text(0.5, 0.5, 'No mass flow data available.\nEnsure components expose mass_flow_kg_h metric.',
                ha='center', va='center', transform=ax.transAxes, fontsize=12, color='gray')
        ax.set_title('Total Mass Flow Comparison')
        return fig
    
    # Build component-wise data using proper name extraction
    components = []
    for col in gas_cols:
        # Extract component name using rsplit (e.g., "O2_Source_mass_flow_kg_h" → "O2_Source")
        parts = col.rsplit('_mass_flow_kg_h', 1)
        if len(parts) == 2 and parts[0]:
            comp_name = parts[0]
            if comp_name not in components:
                components.append(comp_name)
    x = range(len(components))
    
    gas_vals = []
    vapor_vals = []
    liquid_vals = []
    total_vals = []
    
    for comp in components:
        # Find matching columns - use startswith for exact prefix match
        g = [c for c in gas_cols if c.startswith(f"{comp}_")]
        v = [c for c in vapor_cols if c.startswith(f"{comp}_")]
        l = [c for c in liquid_cols if c.startswith(f"{comp}_")]
        
        gas = df[g[0]].mean() if g else 0
        vapor = df[v[0]].mean() if v else 0
        # Convert kg/s to kg/h for liquid columns
        liquid = df[l[0]].mean() * 3600 if l else 0
        
        gas_vals.append(gas)
        vapor_vals.append(vapor)
        liquid_vals.append(liquid)
        total_vals.append(gas + vapor + liquid)
    
    # Plot lines
    ax.plot(x, total_vals, marker='o', linestyle='-', color='purple', linewidth=2.5, markersize=8, label='TOTAL (Gas+Vapor+Liquid)')
    ax.plot(x, gas_vals, marker='s', linestyle='--', color='blue', linewidth=2, markersize=6, label='Main Gas')
    ax.plot(x, vapor_vals, marker='^', linestyle=':', color='red', linewidth=2, markersize=6, label='H2O Vapor')
    ax.plot(x, liquid_vals, marker='d', linestyle='-', color='brown', linewidth=1.5, markersize=6, label='Entrained Liquid')
    
    # Value labels (offset to avoid overlap)
    for i, (t, g, v, l) in enumerate(zip(total_vals, gas_vals, vapor_vals, liquid_vals)):
        if t > 0:
            ax.annotate(f'{t:.1f}', xy=(i, t), xytext=(0, 8), textcoords='offset points',
                       ha='center', fontsize=8, color='purple', fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(components, rotation=30, ha='right')
    ax.set_ylabel('Mass Flow (kg/h)')
    ax.set_xlabel('Component')
    ax.set_title('Mass Flow Comparison (Gas + Vapor + Liquid)', fontsize=12)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    return fig


@log_graph_errors
def _build_train_profile_df(df: pd.DataFrame, metadata: Dict[str, Any], train_tag: str) -> pd.DataFrame:
    """
    Constructs a profile DataFrame for a specific process train (H2 or O2).
    
    Averages time-series history to create a static snapshot of properties
    (T, P, Enthalpy, Composition) at each component's OUTLET.
    """
    # 1. Identify Components in Train
    train_comps = []
    for cid, meta in metadata.items():
        if meta.get('system_group') == train_tag:
            train_comps.append((cid, meta.get('process_step', 999)))
            
    # Sort by step
    train_comps.sort(key=lambda x: x[1])
    components = [x[0] for x in train_comps]
    
    if not components:
        logger.warning(f"No components found for train: {train_tag}")
        return pd.DataFrame()
        
    # 2. Collect Data
    data_rows = []
    
    for cid in components:
        row = {'Component': cid}
        
        # Helper to find column mean
        def get_mean(pattern_candidates):
            for pat in pattern_candidates:
                # Direct check
                key = f"{cid}_{pat}"
                if key in df.columns: return df[key].mean()
                
                # Suffix check
                cols = [c for c in df.columns if c.endswith(f"_{pat}") and cid in c]
                if cols: return df[cols[0]].mean()
            return None


        # Temperature - Note: engine_dispatch uses 'outlet_temp_c' for many components
        t_c = get_mean(['outlet_temp_c', 'outlet_temperature_c', 'temperature_c', 'temp_c', 'T_c'])
        if t_c is None:
            t_k = get_mean(['outlet_temp_k', 'outlet_temperature_k', 'temperature_k', 'temp_k'])
            if t_k: t_c = t_k - 273.15
        row['T_c'] = t_c

        # Pressure
        p_bar = get_mean(['pressure_bar', 'outlet_pressure_bar', 'P_bar'])
        if p_bar is None:
            p_pa = get_mean(['pressure_pa', 'outlet_pressure_pa'])
            if p_pa: p_bar = p_pa / 1e5
        row['P_bar'] = p_bar
        
        # Enthalpy - Now tracked as outlet_enthalpy_kj_kg
        h_kj = get_mean(['outlet_enthalpy_kj_kg', 'specific_enthalpy_kj_kg'])
        if h_kj:
            row['H_kj_kg'] = h_kj
            row['H_mix_J_kg'] = h_kj * 1000.0
        else:
            h_spec = get_mean(['specific_enthalpy_j_kg', 'enthalpy_j_kg', 'h_mix_j_kg'])
            if h_spec:
                 row['H_kj_kg'] = h_spec / 1000.0
                 row['H_mix_J_kg'] = h_spec
            else:
                 row['H_kj_kg'] = 0.0
                 row['H_mix_J_kg'] = 0.0
             
        # Entropy
        s_spec = get_mean(['specific_entropy_j_kgk', 'entropy_j_kgk'])
        row['S_kj_kgK'] = s_spec / 1000.0 if s_spec else 0.0
        
        # Composition - Now tracked as outlet_h2o_frac
        row['MassFrac_H2O'] = get_mean(['outlet_h2o_frac', 'mass_fraction_h2o', 'y_h2o', 'w_h2o']) or 0.0
        row['MassFrac_H2'] = get_mean(['mass_fraction_h2', 'y_h2', 'w_h2']) or 0.0
        row['MassFrac_O2'] = get_mean(['mass_fraction_o2', 'y_o2', 'w_o2']) or 0.0
        
        # Alias for legacy compatibility
        row['w_H2O'] = row['MassFrac_H2O']
        
        data_rows.append(row)
        
    return pd.DataFrame(data_rows)


@log_graph_errors
def create_h2_stacked_properties(df: pd.DataFrame, dpi: int = DPI_HIGH, metadata: Dict[str, Any] = None) -> Figure:
    """Wrapper for H2 Train Stacked Properties."""
    if metadata is None: metadata = {}
    profile_df = _build_train_profile_df(df, metadata, 'H2_Train')
    profile_df.attrs['scenario_name'] = 'H2 Stream'
    return create_stacked_properties_figure(profile_df, 'H2', dpi)


@log_graph_errors
def create_o2_stacked_properties(df: pd.DataFrame, dpi: int = DPI_HIGH, metadata: Dict[str, Any] = None) -> Figure:
    """Wrapper for O2 Train Stacked Properties."""
    if metadata is None: metadata = {}
    profile_df = _build_train_profile_df(df, metadata, 'O2_Train')
    profile_df.attrs['scenario_name'] = 'O2 Stream'
    return create_stacked_properties_figure(profile_df, 'O2', dpi)


@log_graph_errors
def create_stacked_properties_figure(df: pd.DataFrame, gas_fluido: str, dpi: int = DPI_FAST) -> Figure:
    """
    Replica of plot_propriedades_empilhadas.py.
    
    4 Panels:
    1. Temperature (°C)
    2. Pressure (bar)
    3. Mass Fraction H2O (%)
    4. Specific Enthalpy (kJ/kg)
    """
    if df.empty:
        # Avoid error if no data found
        return None
    
    fig = Figure(figsize=(10, 14), dpi=dpi, constrained_layout=True)
    # 4 rows, share x
    gs = fig.add_gridspec(4, 1)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    
    x_labels = df['Component']
    x = range(len(x_labels))
    
    # 1. Temperature
    ax1.plot(x, df['T_c'], marker='o', color='blue', label=f'{gas_fluido} - Temp')
    ax1.set_ylabel('T (°C)')
    ax1.grid(True, linestyle='--')
    for i, v in enumerate(df['T_c']):
        if pd.notnull(v):
            ax1.annotate(f'{v:.1f}', (i, v), xytext=(0, 5), textcoords='offset points', ha='center', fontsize=8)
            
    # 2. Pressure
    ax2.plot(x, df['P_bar'], marker='o', color='red', label=f'{gas_fluido} - Press')
    ax2.set_ylabel('P (bar)')
    ax2.grid(True, linestyle='--')
    for i, v in enumerate(df['P_bar']):
         if pd.notnull(v):
            ax2.annotate(f'{v:.2f}', (i, v), xytext=(0, 5), textcoords='offset points', ha='center', fontsize=8)
            
    # 3. H2O Fraction
    w_h2o = df['MassFrac_H2O'] * 100
    ax3.plot(x, w_h2o, marker='o', color='green', label=f'{gas_fluido} - H2O%')
    ax3.set_ylabel('w_H2O (%)')
    ax3.grid(True, linestyle='--')
    for i, v in enumerate(w_h2o):
        if pd.notnull(v):
            # Only annotate significant values or endpoints
            if abs(v) > 1e-4:
                fmt = '{:.4f}' if abs(v) >= 0.0001 else '{:.2e}'
                ax3.annotate(fmt.format(v), (i, v), xytext=(0, 5), textcoords='offset points', ha='center', fontsize=8)
            
    # 4. Enthalpy
    ax4.plot(x, df['H_kj_kg'], marker='o', color='purple', label='Enthalpy (kJ/kg)')
    ax4.set_ylabel('H (kJ/kg)')
    ax4.set_xlabel('Component')
    ax4.grid(True, linestyle='--')
    for i, v in enumerate(df['H_kj_kg']):
        if pd.notnull(v):
            ax4.annotate(f'{v:.1f}', (i, v), xytext=(0, 5), textcoords='offset points', ha='center', fontsize=8)
            
    # X-Axis Labels
    ax4.set_xticks(x)
    ax4.set_xticklabels(x_labels, rotation=45, ha='right')
    
    fig.suptitle(f'Property Profile: {gas_fluido} Stream', fontsize=14)
    return fig
