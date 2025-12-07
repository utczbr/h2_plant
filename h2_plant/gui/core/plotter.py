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
}

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
    
    # Time
    if 'minute' not in df.columns:
        df['minute'] = df.index * 60
    
    # Preserve module powers as metadata (2D array not suitable for DataFrame columns)
    # We store it as a private attribute that can be accessed via df.attrs
    if 'soec_module_powers' in history:
        df.attrs['soec_module_powers'] = history['soec_module_powers']
        
    return df

# ==============================================================================
# FIGURE CREATION FUNCTIONS
# ==============================================================================

def create_dispatch_figure(df: pd.DataFrame) -> Figure:
    """Create power dispatch stacked area chart."""
    fig = Figure(figsize=(10, 5), dpi=100)
    ax = fig.add_subplot(111)
    
    minutes = df['minute']
    P_soec = df['P_soec']
    P_pem = df['P_pem']
    P_sold = df['P_sold']
    P_offer = df['P_offer']
    
    ax.fill_between(minutes, 0, P_soec, label='SOEC Consumption', color=COLORS['soec'], alpha=0.6)
    ax.fill_between(minutes, P_soec, P_soec + P_pem, label='PEM Consumption', color=COLORS['pem'], alpha=0.6)
    ax.fill_between(minutes, P_soec + P_pem, P_soec + P_pem + P_sold, label='Grid Sale', color=COLORS['sold'], alpha=0.6)
    ax.plot(minutes, P_offer, label='Offered Power', color=COLORS['offer'], linestyle='--', linewidth=1.5)
    
    ax.set_title('Hybrid Dispatch: SOEC + PEM + Arbitrage', fontsize=12)
    ax.set_xlabel('Time (Minutes)')
    ax.set_ylabel('Power (MW)')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def create_arbitrage_figure(df: pd.DataFrame) -> Figure:
    """Create price scenario chart."""
    fig = Figure(figsize=(10, 5), dpi=100)
    ax = fig.add_subplot(111)
    
    minutes = df['minute']
    spot_price = df['Spot']
    
    PPA_PRICE = 50.0
    H2_EQUIV_PRICE = 192.0  # ~9.6 EUR/kg / 0.05 MWh/kg
    
    ax.plot(minutes, spot_price, label='Spot Price (EUR/MWh)', color=COLORS['price'])
    ax.axhline(y=PPA_PRICE, color=COLORS['ppa'], linestyle='--', label='PPA Price')
    ax.axhline(y=H2_EQUIV_PRICE, color='green', linestyle='-.', label=f'H2 Breakeven (~{H2_EQUIV_PRICE:.0f})')
    
    if 'sell_decision' in df.columns:
        sell_idx = df.index[df['sell_decision'] == 1].tolist()
        if sell_idx:
            ax.scatter(minutes.iloc[sell_idx], spot_price.iloc[sell_idx], 
                       color='red', zorder=5, label='Sell Decision', s=20)
    
    ax.set_title('Price Scenario & H2 Opportunity Cost', fontsize=12)
    ax.set_xlabel('Time (Minutes)')
    ax.set_ylabel('Price (EUR/MWh)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def create_h2_production_figure(df: pd.DataFrame) -> Figure:
    """Create hydrogen production chart."""
    fig = Figure(figsize=(10, 5), dpi=100)
    ax = fig.add_subplot(111)
    
    minutes = df['minute']
    H2_soec = df['H2_soec']
    H2_pem = df['H2_pem']
    H2_total = H2_soec + H2_pem
    
    ax.fill_between(minutes, 0, H2_soec, label='H2 SOEC', color=COLORS['soec'], alpha=0.5)
    ax.fill_between(minutes, H2_soec, H2_total, label='H2 PEM', color=COLORS['pem'], alpha=0.5)
    ax.plot(minutes, H2_total, color=COLORS['h2_total'], linestyle='--', label='Total H2')
    
    ax.set_title('Hydrogen Production Rate (kg/min)', fontsize=12)
    ax.set_ylabel('Production Rate (kg/min)')
    ax.set_xlabel('Time (Minutes)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def create_oxygen_figure(df: pd.DataFrame) -> Figure:
    """Create oxygen production chart."""
    fig = Figure(figsize=(10, 5), dpi=100)
    ax = fig.add_subplot(111)
    
    minutes = df['minute']
    O2_pem = df.get('O2_pem_kg', df['H2_pem'] * 8.0)
    O2_soec = df['H2_soec'] * 8.0
    O2_total = O2_soec + O2_pem
    
    ax.fill_between(minutes, 0, O2_soec, label='O2 SOEC', color=COLORS['soec'], alpha=0.5)
    ax.fill_between(minutes, O2_soec, O2_total, label='O2 PEM', color=COLORS['oxygen'], alpha=0.5)
    ax.plot(minutes, O2_total, color='black', linestyle='--', label='Total O2')
    
    ax.set_title('Oxygen Co-Production (kg/min)', fontsize=12)
    ax.set_ylabel('Production Rate (kg/min)')
    ax.set_xlabel('Time (Minutes)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def create_water_figure(df: pd.DataFrame) -> Figure:
    """Create water consumption chart."""
    fig = Figure(figsize=(10, 5), dpi=100)
    ax = fig.add_subplot(111)
    
    minutes = df['minute']
    water_soec = df['Steam_soec'] * 1.10
    water_pem = df['H2O_pem'] * 1.02
    total = water_soec + water_pem
    
    ax.fill_between(minutes, 0, water_soec, label='H2O SOEC', color=COLORS['soec'], alpha=0.5)
    ax.fill_between(minutes, water_soec, total, label='H2O PEM', color='brown', alpha=0.5)
    ax.plot(minutes, total, color=COLORS['water_total'], linestyle='--', label='Total H2O')
    
    ax.set_title('Water Consumption (Including Losses)', fontsize=12)
    ax.set_ylabel('Water Flow (kg/min)')
    ax.set_xlabel('Time (Minutes)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def create_energy_pie_figure(df: pd.DataFrame) -> Figure:
    """Create energy distribution pie chart."""
    fig = Figure(figsize=(8, 6), dpi=100)
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
    
    fig.tight_layout()
    return fig


def create_histogram_figure(df: pd.DataFrame) -> Figure:
    """Create price histogram."""
    fig = Figure(figsize=(10, 5), dpi=100)
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
    fig.tight_layout()
    return fig


def create_dispatch_curve_figure(df: pd.DataFrame) -> Figure:
    """Create dispatch curve scatter plot (P vs H2)."""
    fig = Figure(figsize=(10, 5), dpi=100)
    ax = fig.add_subplot(111)
    
    P_total = df['P_soec'] + df['P_pem']
    H2_total = df['H2_soec'] + df['H2_pem']
    
    ax.scatter(P_total, H2_total, alpha=0.5, c='blue', edgecolors='none', s=10)
    
    ax.set_title('Dispatch Curve: H2 Production vs Power', fontsize=12)
    ax.set_xlabel('Total Power Input (MW)')
    ax.set_ylabel('H2 Production (kg/min)')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ==============================================================================
# NEW CHARTS - Cumulative, Efficiency, Revenue, Power Balance
# ==============================================================================

def create_cumulative_h2_figure(df: pd.DataFrame) -> Figure:
    """Create cumulative hydrogen production chart."""
    fig = Figure(figsize=(10, 5), dpi=100)
    ax = fig.add_subplot(111)
    
    minutes = df['minute']
    H2_soec_cum = df['H2_soec'].cumsum()
    H2_pem_cum = df['H2_pem'].cumsum()
    H2_total_cum = H2_soec_cum + H2_pem_cum
    
    ax.fill_between(minutes, 0, H2_soec_cum, label='SOEC (Cumulative)', 
                    color=COLORS['soec'], alpha=0.6)
    ax.fill_between(minutes, H2_soec_cum, H2_total_cum, label='PEM (Cumulative)', 
                    color=COLORS['pem'], alpha=0.6)
    ax.plot(minutes, H2_total_cum, color='black', linestyle='--', 
            label=f'Total: {H2_total_cum.iloc[-1]:.1f} kg', linewidth=1.5)
    
    ax.set_title('Cumulative Hydrogen Production', fontsize=12)
    ax.set_xlabel('Time (Minutes)')
    ax.set_ylabel('Cumulative H2 (kg)')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def create_cumulative_energy_figure(df: pd.DataFrame) -> Figure:
    """Create cumulative energy consumption chart."""
    fig = Figure(figsize=(10, 5), dpi=100)
    ax = fig.add_subplot(111)
    
    minutes = df['minute']
    # Convert from MW to MWh (divide by 60 for minute data)
    E_soec_cum = (df['P_soec'] / 60.0).cumsum()
    E_pem_cum = (df['P_pem'] / 60.0).cumsum()
    E_sold_cum = (df['P_sold'] / 60.0).cumsum()
    E_total_cum = E_soec_cum + E_pem_cum
    
    ax.fill_between(minutes, 0, E_soec_cum, label='SOEC Energy', 
                    color=COLORS['soec'], alpha=0.6)
    ax.fill_between(minutes, E_soec_cum, E_total_cum, label='PEM Energy', 
                    color=COLORS['pem'], alpha=0.6)
    ax.plot(minutes, E_sold_cum, color=COLORS['sold'], linestyle='--', 
            label=f'Grid Sale: {E_sold_cum.iloc[-1]:.1f} MWh', linewidth=1.5)
    ax.plot(minutes, E_total_cum, color='black', linestyle='-', 
            label=f'Total Used: {E_total_cum.iloc[-1]:.1f} MWh', linewidth=1.5)
    
    ax.set_title('Cumulative Energy Consumption', fontsize=12)
    ax.set_xlabel('Time (Minutes)')
    ax.set_ylabel('Energy (MWh)')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def create_efficiency_curve_figure(df: pd.DataFrame) -> Figure:
    """Create system efficiency over time chart."""
    fig = Figure(figsize=(10, 5), dpi=100)
    ax = fig.add_subplot(111)
    
    minutes = df['minute']
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
    
    ax.plot(minutes, eff_smooth, color='green', linewidth=2, label='System Efficiency')
    ax.fill_between(minutes, 0, eff_smooth, color='green', alpha=0.1)
    
    avg_eff = eff_smooth[eff_smooth > 0].mean() if (eff_smooth > 0).any() else 0
    ax.axhline(y=avg_eff, color='darkgreen', linestyle='--', 
               label=f'Average: {avg_eff:.1f}%', alpha=0.8)
    
    ax.set_title('System Efficiency Over Time (LHV Basis)', fontsize=12)
    ax.set_xlabel('Time (Minutes)')
    ax.set_ylabel('Efficiency (%)')
    ax.set_ylim(0, min(100, max(eff_smooth) * 1.2) if max(eff_smooth) > 0 else 100)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def create_revenue_analysis_figure(df: pd.DataFrame) -> Figure:
    """Create revenue analysis chart showing grid sales revenue."""
    fig = Figure(figsize=(10, 5), dpi=100)
    ax = fig.add_subplot(111)
    
    minutes = df['minute']
    spot_price = df['Spot']
    P_sold = df['P_sold']
    
    # Revenue per minute = (P_sold MW) * (price EUR/MWh) / 60
    revenue_per_min = (P_sold * spot_price) / 60.0
    cumulative_revenue = revenue_per_min.cumsum()
    
    # Also calculate H2 value (opportunity cost)
    H2_total = df['H2_soec'] + df['H2_pem']
    H2_PRICE = 9.6  # EUR/kg
    h2_value_per_min = H2_total * H2_PRICE
    cumulative_h2_value = h2_value_per_min.cumsum()
    
    ax.plot(minutes, cumulative_revenue, color=COLORS['sold'], linewidth=2, 
            label=f'Grid Revenue: €{cumulative_revenue.iloc[-1]:.0f}')
    ax.plot(minutes, cumulative_h2_value, color=COLORS['h2_total'], linewidth=2, 
            linestyle='--', label=f'H2 Value: €{cumulative_h2_value.iloc[-1]:.0f}')
    
    ax.fill_between(minutes, 0, cumulative_revenue, color=COLORS['sold'], alpha=0.2)
    ax.fill_between(minutes, 0, cumulative_h2_value, color=COLORS['h2_total'], alpha=0.1)
    
    ax.set_title('Cumulative Revenue Analysis', fontsize=12)
    ax.set_xlabel('Time (Minutes)')
    ax.set_ylabel('Value (EUR)')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def create_power_balance_figure(df: pd.DataFrame) -> Figure:
    """Create detailed power balance chart."""
    fig = Figure(figsize=(10, 6), dpi=100)
    ax = fig.add_subplot(111)
    
    minutes = df['minute']
    P_soec = df['P_soec']
    P_pem = df['P_pem']
    P_sold = df['P_sold']
    P_offer = df['P_offer']
    
    # Stacked area for consumption
    ax.fill_between(minutes, 0, P_soec, label='SOEC', color=COLORS['soec'], alpha=0.7)
    ax.fill_between(minutes, P_soec, P_soec + P_pem, label='PEM', color=COLORS['pem'], alpha=0.7)
    ax.fill_between(minutes, P_soec + P_pem, P_soec + P_pem + P_sold, 
                    label='Grid Sale', color=COLORS['sold'], alpha=0.7)
    
    # Reference line for offered power
    ax.plot(minutes, P_offer, color='black', linewidth=1.5, linestyle='--', 
            label='Offered', alpha=0.8)
    
    # Add utilization percentage
    P_used = P_soec + P_pem
    utilization = np.where(P_offer > 0, (P_used / P_offer) * 100, 0)
    avg_util = utilization[utilization > 0].mean() if (utilization > 0).any() else 0
    
    ax.set_title(f'Power Balance (Avg Utilization: {avg_util:.1f}%)', fontsize=12)
    ax.set_xlabel('Time (Minutes)')
    ax.set_ylabel('Power (MW)')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ==============================================================================
# PHYSICS-BASED CHARTS - PEM Polarization, Degradation, Compressor T-s
# ==============================================================================

def create_polarization_figure(df: pd.DataFrame) -> Figure:
    """
    Create PEM polarization curve showing BOL, Current, and EOL states.
    Uses physics from constants_physics.py.
    """
    fig = Figure(figsize=(10, 6), dpi=100)
    ax = fig.add_subplot(111)
    
    # Physics constants (self-contained to avoid import issues)
    R = 8.314  # J/(mol·K)
    F = 96485.33  # C/mol
    T = 333.15  # K (60°C)
    P_op = 40.0e5  # Pa (40 bar)
    P_ref = 1.0e5  # Pa
    z = 2
    alpha = 0.5
    j0 = 1.0e-6  # A/cm²
    j_lim = 4.0  # A/cm²
    delta_mem = 100e-4  # cm
    sigma_base = 0.1  # S/cm
    j_nom = 2.91  # A/cm²
    
    # Degradation table (from constants_physics.py)
    YEARS_TABLE = np.array([1.0, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 9.0, 10.0])
    V_STACK_TABLE = np.array([171, 172, 176, 178, 178, 180, 181, 183, 184, 187, 190, 193, 197])
    N_cells = 85
    V_CELL_TABLE = V_STACK_TABLE / N_cells
    T_OP_H_TABLE = YEARS_TABLE * 8760
    
    # Current density range
    j_range = np.linspace(0.01, j_lim * 0.95, 200)
    
    def calculate_vcell(j, U_deg=0.0):
        """Calculate cell voltage for given current density."""
        # Reversible voltage (Nernst)
        U_rev = 1.229 - 0.9e-3 * (T - 298.15) + (R * T) / (z * F) * np.log((P_op / P_ref)**1.5)
        
        # Activation overpotential
        eta_act = (R * T) / (alpha * z * F) * np.log(np.maximum(j, 1e-10) / j0)
        
        # Ohmic overpotential
        eta_ohm = j * (delta_mem / sigma_base)
        
        # Concentration overpotential
        eta_conc = np.where(
            j >= j_lim * 0.99, 100.0,
            (R * T) / (z * F) * np.log(j_lim / np.maximum(j_lim - j, 1e-6))
        )
        
        return U_rev + eta_act + eta_ohm + eta_conc + U_deg
    
    # BOL voltage curve
    V_bol = calculate_vcell(j_range, U_deg=0.0)
    
    # EOL voltage curve (10 years)
    V_bol_nom = calculate_vcell(j_nom, U_deg=0.0)
    V_eol_ref = np.interp(87600, T_OP_H_TABLE, V_CELL_TABLE)  # 10 years
    U_deg_eol = max(0, V_eol_ref - V_bol_nom)
    V_eol = calculate_vcell(j_range, U_deg=U_deg_eol)
    
    # Current state (estimate from simulation time - use 2 years as example)
    t_sim_hours = len(df) / 60.0  # Approximate hours from data length
    t_op_h = min(t_sim_hours, 8760 * 2)  # Cap at 2 years for demo
    V_current_ref = np.interp(t_op_h, T_OP_H_TABLE, V_CELL_TABLE)
    U_deg_current = max(0, V_current_ref - V_bol_nom)
    V_current = calculate_vcell(j_range, U_deg=U_deg_current)
    
    # Plot curves
    ax.plot(j_range, V_bol, 'g--', linewidth=1.5, alpha=0.7, label='BOL (Year 0)')
    ax.plot(j_range, V_current, 'b-', linewidth=2, label=f'Current (~{t_op_h/8760:.1f} yr)')
    ax.plot(j_range, V_eol, 'r--', linewidth=1.5, alpha=0.7, label='EOL (Year 10)')
    
    # Nominal operating point
    ax.axvline(x=j_nom, color='black', linestyle=':', alpha=0.5, label=f'Nominal ({j_nom} A/cm²)')
    
    ax.set_title('PEM Polarization Curve: BOL → Current → EOL', fontsize=12)
    ax.set_xlabel('Current Density (A/cm²)')
    ax.set_ylabel('Cell Voltage (V)')
    ax.set_xlim(0, j_lim)
    ax.set_ylim(1.4, 2.4)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def create_degradation_figure(df: pd.DataFrame) -> Figure:
    """Create degradation projection chart showing voltage evolution over years."""
    fig = Figure(figsize=(10, 5), dpi=100)
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
    fig.tight_layout()
    return fig


def create_compressor_ts_figure(df: pd.DataFrame) -> Figure:
    """
    Create T-s diagram for hydrogen compression.
    Uses CoolProp for real thermodynamic properties with fallback to simplified model.
    """
    fig = Figure(figsize=(10, 6), dpi=100)
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
    fig.tight_layout()
    return fig


def create_module_power_figure(df: pd.DataFrame) -> Figure:
    """
    Create module power distribution chart for SOEC cluster.
    Uses real module data from simulation when available.
    """
    fig = Figure(figsize=(10, 6), dpi=100)
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
    fig.tight_layout()
    return fig


# Need to import plt for colors
import matplotlib.pyplot as plt


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
    'power_balance': {
        'name': 'Power Balance',
        'func': create_power_balance_figure,
        'description': 'Detailed power utilization breakdown'
    },
    # --- PHYSICS CHARTS ---
    'polarization': {
        'name': 'PEM Polarization Curve',
        'func': create_polarization_figure,
        'description': 'Voltage vs current density (BOL/Current/EOL)'
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
