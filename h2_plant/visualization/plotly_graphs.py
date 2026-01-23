"""
Plotly graph implementations for H2 Plant visualization.
"""

from typing import Dict, Any, List, Optional
import logging
import pandas as pd

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    class MockGo:
        Figure = Any
    go = MockGo()
    px = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


from h2_plant.visualization import utils

logger = logging.getLogger(__name__)

# =============================================================================
# PERFORMANCE: WebGL rendering for large datasets
# =============================================================================
WEBGL_THRESHOLD = 5000  # Use Scattergl if more than this many points


def get_scatter_type(n_points: int, force_webgl: bool = False):
    """
    Return the appropriate Scatter class based on data size.
    
    Uses go.Scattergl (WebGL) for large datasets (>5000 points) which provides
    GPU-accelerated rendering, supporting 100k+ points without browser crashes.
    
    Args:
        n_points: Number of data points
        force_webgl: If True, always use Scattergl
    
    Returns:
        go.Scattergl or go.Scatter class
    """
    if force_webgl or n_points > WEBGL_THRESHOLD:
        return go.Scattergl
    return go.Scatter


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


def _check_dependencies():
    """Check if required dependencies are available."""
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required for graph generation. Install with: pip install plotly")
    if not NUMPY_AVAILABLE:
        raise ImportError("NumPy is required. Install with: pip install numpy")


# =============================================================================
# INTERACTIVE CONTROLS: Style Toggle Utility
# =============================================================================

def _build_style_toggle_menu(
    trace_indices: List[int],
    stackgroup_name: str = 'one',
    x_position: float = 0.82,
    default_mode: str = 'lines'
) -> dict:
    """
    Generate Lines/Stacked toggle with targeted trace application.
    
    Args:
        trace_indices: List of trace indices to modify (others remain unchanged)
        stackgroup_name: Unique identifier (e.g., 'mixer_flow', 'heat_rejection')
        x_position: Horizontal placement (0.72 for dual menus, 0.82 for single)
        default_mode: 'lines' (active=0) or 'stacked' (active=1)
    
    Returns:
        dict: Plotly updatemenu configuration
    """
    if not trace_indices:
        return dict(type="dropdown", buttons=[])
        
    n = max(trace_indices) + 1
    
    # Lines mode: EXPLICIT reset for targeted traces ('' clears stackgroup, 'none' clears fill)
    stack_lines = [None] * n
    fill_lines = [None] * n
    width_lines = [None] * n
    
    for idx in trace_indices:
        stack_lines[idx] = ''  # Explicit empty string clears stackgroup
        fill_lines[idx] = 'none'  # Explicit 'none' clears fill
        width_lines[idx] = 1.5
    
    # Stacked mode: Set stackgroup and fill for targeted traces
    stack_stacked = [None] * n
    fill_stacked = [None] * n
    width_stacked = [None] * n
    
    for idx in trace_indices:
        stack_stacked[idx] = stackgroup_name
        fill_stacked[idx] = 'tonexty'
        width_stacked[idx] = 0.5
    
    return dict(
        type="dropdown", direction="down",
        x=x_position, y=1.15, xanchor="left",
        showactive=True,
        active=0 if default_mode == 'lines' else 1,
        buttons=[
            dict(label="Lines", method="restyle",
                 args=[{"stackgroup": stack_lines, "fill": fill_lines, 
                        "line.width": width_lines}, trace_indices]),
            dict(label="Stacked", method="restyle",
                 args=[{"stackgroup": stack_stacked, "fill": fill_stacked,
                        "line.width": width_stacked}, trace_indices]),
        ]
    )


# Subsystem color mapping for consistent coloring across related components
_SUBSYSTEM_COLORS = {
    'PEM': '#1f77b4',    # Blue
    'SOEC': '#ff7f0e',   # Orange
    'ATR': '#2ca02c',    # Green
    'HP': '#d62728',     # Red (High Pressure)
    'LP': '#9467bd',     # Purple (Low Pressure)
    'STORAGE': '#17becf', # Cyan
    'WATER': '#8c564b',  # Brown
    'O2': '#e377c2',     # Pink
    'BIOGAS': '#7f7f7f', # Gray
}

def _get_subsystem_color(comp_id: str) -> str:
    """
    Get color based on subsystem prefix in component ID.
    
    Examples:
        'PEM_H2_Chiller_1' -> Blue (PEM)
        'SOEC_Chiller_1' -> Orange (SOEC)
        'ATR_Syngas_Cooler' -> Green (ATR)
    
    Falls back to _enhanced_color if no subsystem match.
    """
    comp_upper = comp_id.upper()
    for subsystem, color in _SUBSYSTEM_COLORS.items():
        if comp_upper.startswith(subsystem) or f'_{subsystem}_' in comp_upper:
            return color
    # Fallback to hash-based color
    return _enhanced_color(comp_id)


@log_graph_errors
def plot_pem_production_timeline(df: pd.DataFrame, **kwargs) -> go.Figure:
    """
    Plot PEM H2 production rate over time.
    
    Args:
        df: DataFrame containing 'minute' and 'H2_pem' columns
        **kwargs: Additional plot customization options
    
    Note: H2_pem_kg is mass per timestep. We convert to kg/h for display.
    """
    _check_dependencies()
    
    from h2_plant.visualization.utils import downsample_dataframe, get_time_axis_hours, get_viz_config
    
    maxpoints = kwargs.get('maxpoints', get_viz_config('performance.max_points_default', 2000))
    df_plot = downsample_dataframe(df, max_points=maxpoints)
    
    if df_plot.empty:
        return _empty_figure("No DataFrame provided")

    hours = get_time_axis_hours(df_plot)
    
    # Flexible column resolution
    h2_col = next((c for c in ['H2_pem_kg', 'H2_pem', 'H2_pem_kg_h'] if c in df_plot.columns), None)
    
    if not h2_col:
        # Fallback to look for partial matches if needed, or return empty
        h2_col = utils.find_column(df_plot, 'PEM', 'h2_production_kg_h')
        
    if not h2_col:
         return _empty_figure("No PEM H2 data column found")
    
    # Get timestep for unit conversion
    dt_seconds = df.attrs.get('dt_seconds', 60.0)
    dt_h = dt_seconds / 3600.0
    
    # Convert from per-timestep (kg) to rate (kg/h)
    # If column ends with 'kg_h', it's already a rate
    production_raw = df_plot[h2_col].values
    if h2_col.endswith('_kg_h') or h2_col.endswith('kg_h'):
        production = production_raw  # Already kg/h
    else:
        production = production_raw / dt_h  # kg/timestep -> kg/h
    
    color = get_viz_config('styling.colors.pem', '#1f77b4')
    
    # PERFORMANCE: Use WebGL for large datasets
    ScatterType = get_scatter_type(len(production))
    
    fig = go.Figure()
    fig.add_trace(ScatterType(
        x=hours,
        y=production,
        mode='lines',
        name='PEM Production',
        line=dict(color=color, width=2),
        fill='tozeroy',
        fillcolor=f"rgba{tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.1,)}" if color.startswith('#') else color
    ))
    
    fig.update_layout(
        title=kwargs.get('title', 'PEM H2 Production Rate'),
        xaxis_title='Time (hours)',
        yaxis_title='H2 Production (kg/h)',
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig


@log_graph_errors
def plot_soec_production_timeline(df: pd.DataFrame, **kwargs) -> go.Figure:
    """Plot SOEC H2 production rate over time.
    
    Note: H2_soec_kg is mass per timestep. We convert to kg/h for display.
    """
    _check_dependencies()
    
    from h2_plant.visualization.utils import downsample_dataframe, get_time_axis_hours, get_viz_config
    
    maxpoints = kwargs.get('maxpoints', get_viz_config('performance.max_points_default', 2000))
    df_plot = downsample_dataframe(df, max_points=maxpoints)
    hours = get_time_axis_hours(df_plot)
    
    h2_col = next((c for c in ['H2_soec_kg', 'H2_soec', 'H2_soec_kg_h'] if c in df_plot.columns), None)
    if not h2_col:
        h2_col = utils.find_column(df_plot, 'SOEC', 'h2_production_kg_h')
        
    if not h2_col:
         return _empty_figure("No SOEC H2 data column found")
    
    # Get timestep for unit conversion
    dt_seconds = df.attrs.get('dt_seconds', 60.0)
    dt_h = dt_seconds / 3600.0
    
    # Convert from per-timestep (kg) to rate (kg/h)
    production_raw = df_plot[h2_col].values
    if h2_col.endswith('_kg_h') or h2_col.endswith('kg_h'):
        production = production_raw  # Already kg/h
    else:
        production = production_raw / dt_h  # kg/timestep -> kg/h
         
    color = get_viz_config('styling.colors.soec', '#ff7f0e')
    
    fig = go.Figure()
    fig.add_trace(get_scatter_type(len(hours))(
        x=hours,
        y=production,
        mode='lines',
        name='SOEC Production',
        line=dict(color=color, width=2),
        fill='tozeroy',
        fillcolor=f"rgba{tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.1,)}" if color.startswith('#') else color
    ))
    
    fig.update_layout(
        title=kwargs.get('title', 'SOEC H2 Production Rate'),
        xaxis_title='Time (hours)',
        yaxis_title='H2 Production (kg/h)',
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig

@log_graph_errors
def plot_total_production_stacked(df: pd.DataFrame, **kwargs) -> go.Figure:
    """Plot stacked area chart showing PEM + SOEC contributions.
    
    Note: H2_pem_kg/H2_soec_kg are mass per timestep. We convert to kg/h for display.
    """
    _check_dependencies()
    
    from h2_plant.visualization.utils import downsample_dataframe, get_time_axis_hours, get_viz_config
    
    maxpoints = kwargs.get('maxpoints', get_viz_config('performance.max_points_default', 2000))
    df_plot = downsample_dataframe(df, max_points=maxpoints)
    hours = get_time_axis_hours(df_plot)
    
    pem_col = next((c for c in ['H2_pem_kg', 'H2_pem'] if c in df_plot.columns), None)
    soec_col = next((c for c in ['H2_soec_kg', 'H2_soec'] if c in df_plot.columns), None)
    
    # Unit conversion
    dt_seconds = df.attrs.get('dt_seconds', 60.0)
    dt_h = dt_seconds / 3600.0
    
    pem_raw = df_plot[pem_col].values if pem_col else np.zeros(len(hours))
    soec_raw = df_plot[soec_col].values if soec_col else np.zeros(len(hours))
    
    # Convert per-timestep to rate (kg/h)
    pem_production = pem_raw / dt_h if pem_col and not pem_col.endswith('_kg_h') else pem_raw
    soec_production = soec_raw / dt_h if soec_col and not soec_col.endswith('_kg_h') else soec_raw
    
    color_pem = get_viz_config('styling.colors.pem', '#1f77b4')
    color_soec = get_viz_config('styling.colors.soec', '#ff7f0e')
    
    fig = go.Figure()
    
    fig.add_trace(get_scatter_type(len(hours))(
        x=hours,
        y=pem_production,
        mode='lines',
        name='PEM',
        stackgroup='one',
        line=dict(color=color_pem, width=0.5),
        fillcolor=f"rgba{tuple(int(color_pem.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.7,)}" if color_pem.startswith('#') else color_pem
    ))
    
    fig.add_trace(get_scatter_type(len(hours))(
        x=hours,
        y=soec_production,
        mode='lines',
        name='SOEC',
        stackgroup='one',
        line=dict(color=color_soec, width=0.5),
        fillcolor=f"rgba{tuple(int(color_soec.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.7,)}" if color_soec.startswith('#') else color_soec
    ))
    
    fig.update_layout(
        title=kwargs.get('title', 'Total H2 Production (PEM + SOEC)'),
        xaxis_title='Time (hours)',
        yaxis_title='H2 Production (kg/h)',
        template='plotly_white',
        hovermode='x unified',
        legend=dict(x=0.02, y=0.98)
    )
    
    return fig



@log_graph_errors
def plot_cumulative_production(df: pd.DataFrame, **kwargs) -> go.Figure:
    """
    Cumulative H2 Production from all sources (SOEC, PEM, ATR).
    Merged: Interactive Lines + Stacked toggle.
    """
    _check_dependencies()
    from h2_plant.visualization.utils import downsample_dataframe, get_time_axis_hours, get_viz_config, find_columns_by_type
    
    maxpoints = kwargs.get('maxpoints', get_viz_config('performance.max_points_default', 2000))
    df_plot = downsample_dataframe(df, max_points=maxpoints)
    hours = get_time_axis_hours(df_plot)
    
    # Identify Production Sources
    sources = {}
    known_prefixes = ['pem', 'soec', 'atr']
    
    # 1. Search for explicit per-source cumulative columns (e.g., Cumulative_H2_PEM_kg)
    for col in df_plot.columns:
        col_lower = col.lower()
        if 'cumulative' in col_lower and 'h2' in col_lower and 'rfnbo' not in col_lower:
            for prefix in known_prefixes:
                if prefix in col_lower:
                    sources[prefix.upper()] = df_plot[col]
                    break

    # 2. If no per-source cumulative columns, calculate from rate columns
    if not sources:
        dt_hours = np.mean(np.diff(hours)) if len(hours) > 1 else (df.attrs.get('dt_seconds', 60)/3600)
        
        for col in df_plot.columns:
            col_lower = col.lower()
            for prefix in known_prefixes:
                if f"h2_{prefix}" in col_lower and "cumulative" not in col_lower and "kg" in col_lower:
                    # Integrate rate to get cumulative
                    sources[prefix.upper()] = (df_plot[col] * dt_hours).cumsum()
                    break

    if not sources:
        return _empty_figure("No Cumulative H2 Data Found")

    fig = go.Figure()
    ScatterType = get_scatter_type(len(hours))
    
    trace_counter = 0
    stackable_indices = []
    
    for name, data in sources.items():
        fig.add_trace(ScatterType(
            x=hours,
            y=data,
            mode='lines',
            name=name,
            line=dict(width=1.5, color=_get_subsystem_color(name)),
            stackgroup=None,  # Default to Lines
            hovertemplate=f'<b>{name}</b><br>%{{y:,.0f}} kg<extra></extra>'
        ))
        stackable_indices.append(trace_counter)
        trace_counter += 1

    # Add the Toggle Menu
    style_menu = _build_style_toggle_menu(stackable_indices, 'cumulative_h2', x_position=0.82)

    fig.update_layout(
        updatemenus=[style_menu],
        title=kwargs.get('title', 'Cumulative H2 Production'),
        xaxis_title='Time (hours)',
        yaxis_title='Cumulative Production (kg)',
        template='plotly_white',
        hovermode='x unified',
        showlegend=True,
        legend=dict(x=0.02, y=0.98)
    )
    
    return fig


@log_graph_errors
def plot_pem_voltage_timeline(df: pd.DataFrame, **kwargs) -> go.Figure:
    """Plot PEM cell voltage over time."""
    _check_dependencies()

    from h2_plant.visualization.utils import downsample_dataframe, get_time_axis_hours, get_viz_config

    maxpoints = kwargs.get('maxpoints', get_viz_config('performance.max_points_default', 2000))
    df_plot = downsample_dataframe(df, max_points=maxpoints)
    hours = get_time_axis_hours(df_plot)
    
    # Find voltage column
    volt_col = next((c for c in ['pem_voltage', 'voltage_V', 'PEM_voltage'] if c in df_plot.columns), None)
    if not volt_col:
         volt_col = utils.find_column(df_plot, 'PEM', 'voltage')
         
    voltage = df_plot[volt_col].values if volt_col else np.array([])
    
    fig = go.Figure()
    if len(voltage) > 0:
        fig.add_trace(get_scatter_type(len(hours))(
            x=hours,
            y=voltage,
            mode='lines',
            name='Cell Voltage',
            line=dict(color='#d62728', width=2)
        ))
    
    # Add threshold lines
    fig.add_hline(y=2.0, line_dash='dash', line_color='orange', 
                  annotation_text='2.0V Nominal', annotation_position='right')
    fig.add_hline(y=2.4, line_dash='dash', line_color='red',
                  annotation_text='2.4V Max', annotation_position='right')
    
    fig.update_layout(
        title=kwargs.get('title', 'PEM Cell Voltage'),
        xaxis_title='Time (hours)',
        yaxis_title='Cell Voltage (V)',
        template='plotly_white',
        hovermode='x unified',
        yaxis=dict(range=[1.5, 2.5])
    )
    
    return fig


@log_graph_errors
def plot_pem_efficiency_timeline(df: pd.DataFrame, **kwargs) -> go.Figure:
    """
    Plot PEM system efficiency over time.
    
    FIX: If efficiency column is missing, calculate from H2 production and power.
    Efficiency (% LHV) = (H2_kg * 33.33 kWh/kg) / P_kW * 100
    """
    _check_dependencies()
    
    from h2_plant.visualization.utils import downsample_dataframe, get_time_axis_hours, get_viz_config

    maxpoints = kwargs.get('maxpoints', get_viz_config('performance.max_points_default', 2000))
    df_plot = downsample_dataframe(df, max_points=maxpoints)
    hours = get_time_axis_hours(df_plot)
    
    # Try to find existing efficiency column
    eff_col = next((c for c in ['pem_efficiency', 'efficiency_pct', 'PEM_efficiency'] if c in df_plot.columns), None)
    
    if eff_col:
        efficiency = df_plot[eff_col].values
    else:
        # FIX: Calculate efficiency from H2 production and power
        # Efficiency (LHV) = m_H2 * LHV_H2 / P_input
        # LHV_H2 = 33.33 kWh/kg (120 MJ/kg)
        LHV_H2_KWH_KG = 33.33
        
        h2_col = next((c for c in ['H2_pem_kg', 'H2_pem', 'pem_h2_production_kg'] if c in df_plot.columns), None)
        power_col = next((c for c in ['P_pem', 'pem_power_kw', 'pem_power_mw'] if c in df_plot.columns), None)
        
        if h2_col and power_col:
            h2 = df_plot[h2_col].values
            power = df_plot[power_col].values
            
            # Convert units if needed
            dt_h = df.attrs.get('dt_seconds', 60.0) / 3600.0
            
            # H2 is per timestep, convert to rate (kg/h)
            h2_rate = h2 / dt_h if dt_h > 0 else h2
            
            # Power: if MW, convert to kW
            if 'mw' in power_col.lower() or power.mean() < 10:
                power = power * 1000  # MW to kW
            
            # Efficiency = (H2 rate * LHV) / Power * 100
            with np.errstate(divide='ignore', invalid='ignore'):
                efficiency = np.where(power > 0, (h2_rate * LHV_H2_KWH_KG) / power * 100, 0)
                efficiency = np.clip(efficiency, 0, 100)  # Cap at 100%
        else:
            return _empty_figure("No PEM efficiency or H2/Power data available")
    
    fig = go.Figure()
    fig.add_trace(get_scatter_type(len(hours))(
        x=hours,
        y=efficiency,
        mode='lines',
        name='PEM System Efficiency',
        line=dict(color='#2ca02c', width=2),
        hovertemplate='Time: %{x:.1f}h<br>Efficiency: %{y:.1f}%<extra></extra>'
    ))
    
    # Add average line
    avg_eff = np.mean(efficiency[efficiency > 0]) if np.any(efficiency > 0) else 0
    fig.add_hline(
        y=avg_eff, 
        line_dash="dash", 
        line_color="gray",
        annotation_text=f"Avg: {avg_eff:.1f}%"
    )
    
    fig.update_layout(
        title=kwargs.get('title', f'PEM System Efficiency (Avg: {avg_eff:.1f}%)'),
        xaxis_title='Time (hours)',
        yaxis_title='Efficiency (% LHV)',
        template='plotly_white',
        hovermode='x unified',
        yaxis=dict(range=[0, 100])
    )
    
    return fig


@log_graph_errors
def plot_energy_price_timeline(df: pd.DataFrame, **kwargs) -> go.Figure:
    """Plot energy price over time."""
    _check_dependencies()
    
    from h2_plant.visualization.utils import downsample_dataframe, get_time_axis_hours, get_viz_config

    maxpoints = kwargs.get('maxpoints', get_viz_config('performance.max_points_default', 2000))
    df_plot = downsample_dataframe(df, max_points=maxpoints)
    hours = get_time_axis_hours(df_plot)
    
    # Try different potential column names for energy price
    # Usually it's 'spot_price' or 'energy_price_eur_kwh' (sim data might be MWh or kWh)
    # The original function expected 'energy_price_eur_kwh' and multiplied by 1000 to get MWh
    
    price_col = next((c for c in ['energy_price_eur_kwh', 'pricing_energy_price_eur_kwh'] if c in df_plot.columns), None)
    
    # If not found, check for spot_price (usually EUR/MWh)
    spot_col = next((c for c in ['spot_price', 'Spot'] if c in df_plot.columns), None)
    
    price_mwh = []
    if price_col:
        price_mwh = df_plot[price_col].values * 1000
    elif spot_col:
        price_mwh = df_plot[spot_col].values # Assuming Spot is already MWh
    else:
        price_mwh = np.zeros(len(hours))
    
    fig = go.Figure()
    fig.add_trace(get_scatter_type(len(hours))(
        x=hours,
        y=price_mwh,
        mode='lines',
        name='Energy Price',
        line=dict(color='#9467bd', width=2),
        fill='tozeroy',
        fillcolor='rgba(148, 103, 189, 0.1)'
    ))
    
    fig.update_layout(
        title=kwargs.get('title', 'Energy Price'),
        xaxis_title='Time (hours)',
        yaxis_title='Price (EUR/MWh)',
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig


@log_graph_errors
def plot_dispatch_strategy(df: pd.DataFrame, **kwargs) -> go.Figure:
    """Plot dispatch strategy as stacked area chart."""
    _check_dependencies()
    
    from h2_plant.visualization.utils import downsample_dataframe, get_time_axis_hours, get_viz_config

    maxpoints = kwargs.get('maxpoints', get_viz_config('performance.max_points_default', 2000))
    df_plot = downsample_dataframe(df, max_points=maxpoints)
    hours = get_time_axis_hours(df_plot)
    
    cols = {
        'pem': next((c for c in ['P_pem', 'pem_setpoint_mw', 'coordinator_pem_setpoint_mw'] if c in df_plot.columns), None),
        'soec': next((c for c in ['P_soec', 'P_soec_actual', 'soec_setpoint_mw', 'coordinator_soec_setpoint_mw'] if c in df_plot.columns), None),
        'sell': next((c for c in ['P_sold', 'sell_power_mw', 'coordinator_sell_power_mw'] if c in df_plot.columns), None)
    }
    
    # Auxiliary power
    aux_col = next((c for c in ['auxiliary_power_kw', 'P_bop_mw'] if c in df_plot.columns), None)
    
    pem_power = df_plot[cols['pem']].values if cols['pem'] else np.zeros(len(hours))
    soec_power = df_plot[cols['soec']].values if cols['soec'] else np.zeros(len(hours))
    sell_power = df_plot[cols['sell']].values if cols['sell'] else np.zeros(len(hours))
    
    if aux_col:
        if 'kw' in aux_col.lower():
            aux_power_mw = df_plot[aux_col].values / 1000.0
        else:
            aux_power_mw = df_plot[aux_col].values
    else:
        aux_power_mw = np.zeros(len(hours))
    
    fig = go.Figure()
    
    fig.add_trace(get_scatter_type(len(hours))(
        x=hours,
        y=pem_power,
        mode='lines',
        name='PEM',
        stackgroup='one',
        line=dict(color=get_viz_config('styling.colors.pem', '#1f77b4'), width=0.5),
        fillcolor='rgba(31, 119, 180, 0.7)'
    ))
    
    fig.add_trace(get_scatter_type(len(hours))(
        x=hours,
        y=soec_power,
        mode='lines',
        name='SOEC',
        stackgroup='one',
        line=dict(color=get_viz_config('styling.colors.soec', '#ff7f0e'), width=0.5),
        fillcolor='rgba(255, 127, 14, 0.7)'
    ))
    
    fig.add_trace(get_scatter_type(len(hours))(
        x=hours,
        y=sell_power,
        mode='lines',
        name='Grid Export',
        stackgroup='one',
        line=dict(color='#2ca02c', width=0.5),
        fillcolor='rgba(44, 160, 44, 0.7)'
    ))
    
    # Reordered: BOP on top of stack
    fig.add_trace(get_scatter_type(len(hours))(
        x=hours,
        y=aux_power_mw,
        mode='lines',
        name='Balance Of Plant (BOP)',
        stackgroup='one',
        line=dict(color='#9467bd', width=0.5),
        fillcolor='rgba(148, 103, 189, 0.7)'
    ))
    
    # P1 PARITY FIX: Add "Offered Power" trace (matches Matplotlib version)
    offer_col = next((c for c in ['P_offer', 'offered_power_mw'] if c in df_plot.columns), None)
    if offer_col:
        offer_power = df_plot[offer_col].values
        fig.add_trace(get_scatter_type(len(hours))(
            x=hours,
            y=offer_power,
            mode='lines',
            name='RFNBO Wind Power',
            line=dict(color='black', width=1.5, dash='dash')
        ))
    
    fig.update_layout(
        title=kwargs.get('title', 'Power Dispatch Strategy'),
        xaxis_title='Time (hours)',
        yaxis_title='Power (MW)',
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig


@log_graph_errors
def plot_power_breakdown_pie(df: pd.DataFrame, **kwargs) -> go.Figure:
    """
    Plot power consumption breakdown as pie chart.
    
    FIX: Uses FULL dataframe for energy calculation (no downsampling distortion).
    Also includes BOP, Auxiliary, and Cooling categories.
    """
    _check_dependencies()
    
    # CRITICAL FIX: Get dt from attrs or calculate from minute column
    # DO NOT downsample for energy calculations!
    dt_seconds = df.attrs.get('dt_seconds', 60.0)
    dt_h = dt_seconds / 3600.0  # Convert to hours
    
    # Helper to calculate energy in MWh from power columns
    def calc_energy_mwh(col_name, df):
        if col_name and col_name in df.columns:
            power = df[col_name].sum()
            # Detect unit: if column contains 'kw' or mean > 100, assume kW
            if 'kw' in col_name.lower() or power / len(df) > 100:
                return power * dt_h / 1000.0  # kW*h -> MWh
            else:
                return power * dt_h  # MW*h -> MWh
        return 0.0
    
    # Find power columns
    pem_col = next((c for c in ['P_pem', 'pem_power_kw', 'pem_power_mw'] if c in df.columns), None)
    soec_col = next((c for c in ['P_soec_actual', 'P_soec', 'soec_power_kw'] if c in df.columns), None)
    comp_col = next((c for c in ['compressor_power_kw', 'total_compressor_power_kw'] if c in df.columns), None)
    bop_col = next((c for c in ['P_bop_mw', 'bop_power_kw', 'auxiliary_power_kw'] if c in df.columns), None)
    cooling_col = next((c for c in ['cooling_power_kw', 'chiller_power_kw'] if c in df.columns), None)
    
    # Calculate energies
    pem_energy = calc_energy_mwh(pem_col, df)
    soec_energy = calc_energy_mwh(soec_col, df)
    comp_energy = calc_energy_mwh(comp_col, df)
    bop_energy = calc_energy_mwh(bop_col, df)
    cooling_energy = calc_energy_mwh(cooling_col, df)
    
    # Build chart data
    categories = [
        ('PEM Electrolysis', pem_energy, '#1f77b4'),
        ('SOEC Electrolysis', soec_energy, '#ff7f0e'),
        ('Compression', comp_energy, '#9467bd'),
        ('BOP/Auxiliary', bop_energy, '#2ca02c'),
        ('Cooling', cooling_energy, '#17becf'),
    ]
    
    # Filter out zero values
    plot_data = [(l, v, c) for l, v, c in categories if v > 0.01]  # Ignore negligible values
    if plot_data:
        plot_labels, plot_values, plot_colors = zip(*plot_data)
    else:
        return _empty_figure("No power consumption data available")
    
    fig = go.Figure(data=[go.Pie(
        labels=plot_labels,
        values=plot_values,
        marker=dict(colors=plot_colors),
        textinfo='label+percent',
        textposition='inside',
        hovertemplate='%{label}: %{value:.2f} MWh<extra></extra>'
    )])
    
    total_energy_mwh = sum(plot_values)
    
    fig.update_layout(
        title=kwargs.get('title', f'Total Energy Consumption Breakdown ({total_energy_mwh:.1f} MWh)'),
        legend=dict(title=f"Total: {total_energy_mwh:.2f} MWh"),
        template='plotly_white'
    )
    
    return fig


@log_graph_errors
def plot_soec_modules_timeline(df: pd.DataFrame, **kwargs) -> go.Figure:
    """Plot number of active SOEC modules over time."""
    _check_dependencies()
    
    from h2_plant.visualization.utils import downsample_dataframe, get_time_axis_hours, get_viz_config

    maxpoints = kwargs.get('maxpoints', get_viz_config('performance.max_points_default', 2000))
    df_plot = downsample_dataframe(df, max_points=maxpoints)
    hours = get_time_axis_hours(df_plot)
    
    mod_col = next((c for c in ['soec_active_modules', 'active_modules'] if c in df_plot.columns), None)
    if not mod_col:
        mod_col = utils.find_column(df_plot, 'SOEC', 'active_modules')
        
    active_modules = df_plot[mod_col].values if mod_col else np.zeros(len(hours))
    
    fig = go.Figure()
    fig.add_trace(get_scatter_type(len(hours))(
        x=hours,
        y=active_modules,
        mode='lines',
        name='Active Modules',
        line=dict(color='#e377c2', width=2, shape='hv'),
        fill='tozeroy',
        fillcolor='rgba(227, 119, 194, 0.3)'
    ))
    
    fig.update_layout(
        title=kwargs.get('title', 'SOEC Active Modules'),
        xaxis_title='Time (hours)',
        yaxis_title='Number of Active Modules',
        template='plotly_white',
        hovermode='x unified',
        yaxis=dict(range=[0, 8], dtick=1) # Assuming max 7-8 based on prev code
    )
    
    return fig


@log_graph_errors
def plot_tank_storage_timeline(df: pd.DataFrame, **kwargs) -> go.Figure:
    """Plot tank storage levels over time."""
    _check_dependencies()
    
    from h2_plant.visualization.utils import downsample_dataframe, get_time_axis_hours, get_viz_config, find_columns_by_type
    
    maxpoints = kwargs.get('maxpoints', get_viz_config('performance.max_points_default', 2000))
    df_plot = downsample_dataframe(df, max_points=maxpoints)
    hours = get_time_axis_hours(df_plot)
    
    # Try to find tank pressure or level columns
    tank_cols = find_columns_by_type(df_plot, 'Tank', 'pressure_bar')
    
    fig = go.Figure()
    
    if not tank_cols:
        return _empty_figure("No Tank data found")
    else:
        ScatterType = get_scatter_type(len(hours))
        for tank_id, col in tank_cols.items():
             fig.add_trace(ScatterType(
                x=hours,
                y=df_plot[col],
                mode='lines',
                name=f"{tank_id} Pressure"
            ))
    
    fig.update_layout(
        title=kwargs.get('title', 'Tank Storage Pressure'),
        xaxis_title='Time (hours)',
        yaxis_title='Pressure (bar)',
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig


@log_graph_errors
def plot_storage_fatigue_cycling_3d(df: pd.DataFrame, **kwargs) -> go.Figure:
    _check_dependencies()
    
    from h2_plant.visualization.utils import downsample_dataframe, get_time_axis_hours, get_viz_config, find_columns_by_type
    
    maxpoints = kwargs.get('maxpoints', get_viz_config('performance.max_points_default', 1000)) # Lower default for 3D
    df_plot = downsample_dataframe(df, max_points=maxpoints)
    hours = get_time_axis_hours(df_plot)
    
    tank_cols = find_columns_by_type(df_plot, 'Tank', 'pressure_bar')
    
    if not tank_cols:
        return _empty_figure("No Tank pressure data for 3D plot")
        
    tank_ids = sorted(list(tank_cols.keys()))
    pressures = []
    
    for tid in tank_ids:
        pressures.append(df_plot[tank_cols[tid]].values)
        
    # Convert to numpy for meshgrid
    pressures_np = np.array(pressures).T # Shape: (timesteps, n_tanks)
    
    # Create meshgrid
    # Y is tank index (0, 1, 2...)
    # X is time
    
    fig = go.Figure(data=[go.Surface(
        z=pressures_np.T, # Surface expects z as (y, x) or similar? 
        # API: z is 2D array. x and y are 1D arrays showing coordinates.
        # If z is (n_tanks, timesteps), then y should be length n_tanks, x length timesteps
        x=hours,
        y=np.arange(len(tank_ids)),
        colorscale='Viridis',
        colorbar_title='Pressure (bar)'
    )])
    
    fig.update_layout(
        title=kwargs.get('title', 'Storage Tank Fatigue Cycling (3D)'),
        scene=dict(
            xaxis_title='Time (hours)',
            yaxis_title='Tank ID',
            zaxis_title='Pressure (bar)',
             yaxis=dict(tickvals=list(range(len(tank_ids))), ticktext=tank_ids)
        ),
        template='plotly_white'
    )
    return fig


@log_graph_errors
def plot_ramp_rate_stress_distribution(df: pd.DataFrame, **kwargs) -> go.Figure:
    """
    Plot distribution of ramp rates (Stress Analysis).
    Violin plot of MW/min for SOEC (and PEM if available).
    """
    _check_dependencies()
    
    # Calculate ramp rates from power derivative if not explicitly stored?
    # Or expect 'ramp_rate' column.
    # Typically ramp rate is dP/dt. 
    # Let's check for ramp columns.
    
    cols = utils.find_columns_by_type(df, 'SOEC', 'ramp_rate')
    # If no explicit column, calculate from power
    soec_ramps = np.array([])
    
    if cols:
        soec_ramps = df[list(cols.values())[0]].values
    else:
        # Calculate
        power_col = next((c for c in ['P_soec', 'P_soec_actual'] if c in df.columns), None)
        if power_col:
             dt_min = df.attrs.get('dt_seconds', 60.0) / 60.0 # Time step in minutes
             power = df[power_col].values
             soec_ramps = np.diff(power) / dt_min # MW/min
    
    # Downsample for violin plot? Usually better to use all data for distribution unless huge
    if len(soec_ramps) > 10000:
        soec_ramps = np.random.choice(soec_ramps, 10000)
    
    fig = go.Figure()
    
    if len(soec_ramps) > 0:
        fig.add_trace(go.Violin(
            y=soec_ramps,
            name='SOEC Ramp Rates',
            box_visible=True,
            meanline_visible=True,
            line_color='#ff7f0e'
        ))
    
    fig.update_layout(
        title=kwargs.get('title', 'Ramp Rate Stress Distribution'),
        yaxis_title='Ramp Rate (MW/min)',
        template='plotly_white'
    )
    return fig


@log_graph_errors
def plot_wind_utilization_duration_curve(df: pd.DataFrame, **kwargs) -> go.Figure:
    """
    Plot wind utilization duration curve (Grid Integration).
    
    FIX: Correctly scales X-axis to total simulation hours.
    Shows Available Wind, Utilized Power, and Curtailment.
    """
    _check_dependencies()

    from h2_plant.visualization.utils import get_viz_config

    # Get total hours from attrs (for proper X-axis scaling)
    dt_seconds = df.attrs.get('dt_seconds', 60.0)
    
    # Calculate robust total_hours from 'minute' column if available (handles downsampling)
    if 'minute' in df.columns:
        # Duration = (MaxMinute - MinMinute + dt_minutes) / 60
        t_range_min = df['minute'].max() - df['minute'].min()
        total_hours = (t_range_min * 60.0 + dt_seconds) / 3600.0
    else:
        # Fallback (underestimates if downsampled)
        total_hours = len(df) * dt_seconds / 3600.0
    
    # Get wind capacity from config
    WIND_CAPACITY_MW = get_viz_config('plant_parameters.wind_capacity_mw', 20.0)
    
    # Find renewable power offer column (represents available wind)
    wind_col = next((c for c in ['P_offer', 'P_renewable_mw', 'wind_power_mw'] if c in df.columns), None)
    
    if not wind_col and 'wind_coefficient' in df.columns:
         # Fallback to coefficient if P_offer missing (legacy)
         wind_coeffs = df['wind_coefficient'].values
         wind_available = wind_coeffs * WIND_CAPACITY_MW
    elif wind_col:
         wind_available = df[wind_col].values
    else:
        return _empty_figure("No wind data (P_offer) found")
    
    # Calculate used power (PEM + SOEC)
    pem_col = next((c for c in ['P_pem', 'pem_power_mw', 'P_pem_mw'] if c in df.columns), None)
    soec_col = next((c for c in ['P_soec', 'soec_power_mw', 'P_soec_mw'] if c in df.columns), None)
    
    pem_power = df[pem_col].values if pem_col else np.zeros(len(wind_available))
    soec_power = df[soec_col].values if soec_col else np.zeros(len(wind_available))
    
    # Convert kW to MW if needed
    if pem_power.mean() > 100: pem_power /= 1000.0
    if soec_power.mean() > 100: soec_power /= 1000.0
    
    total_used = pem_power + soec_power
    
    # Calculate curtailment (wind available but not used)
    curtailment = np.maximum(0, wind_available - total_used)
    
    # Sort all curves descending for duration curve
    wind_sorted = np.sort(wind_available)[::-1]
    used_sorted = np.sort(total_used)[::-1]
    curtail_sorted = np.sort(curtailment)[::-1]
    
    # CRITICAL FIX: X-axis should be 0 to total_hours
    hours_axis = np.linspace(0, total_hours, len(wind_sorted))
    
    fig = go.Figure()
    
    # Available Wind (fill to zero)
    fig.add_trace(get_scatter_type(len(hours_axis))(
        x=hours_axis, 
        y=wind_sorted, 
        mode='lines', 
        name='Available Wind Power',
        fill='tozeroy',
        line=dict(color='#3498db', width=1),
        fillcolor='rgba(52, 152, 219, 0.3)'
    ))
    
    # Utilized Power (solid fill)
    fig.add_trace(get_scatter_type(len(hours_axis))(
        x=hours_axis, 
        y=used_sorted, 
        mode='lines', 
        name='Utilized Power',
        fill='tozeroy',
        line=dict(color='#2ecc71', width=2),
        fillcolor='rgba(46, 204, 113, 0.5)'
    ))
    
    # Curtailment (dashed line)
    fig.add_trace(get_scatter_type(len(hours_axis))(
        x=hours_axis,
        y=curtail_sorted,
        mode='lines',
        name='Curtailment',
        line=dict(color='#e74c3c', width=1.5, dash='dash')
    ))
    
    # Guaranteed Power Output (horizontal line)
    # Try multiple sources for guaranteed power value
    config = df.attrs.get('config', {})
    economics = df.attrs.get('economics', {})
    
    # Check multiple possible key names and config locations
    guaranteed_mw = (
        config.get('guaranteed_power_mw') or 
        config.get('guaranteed_mw') or
        economics.get('guaranteed_power_mw') or
        economics.get('guaranteed_mw') or
        get_viz_config('plant_parameters.guaranteed_power_mw', None) or
        get_viz_config('ppa_parameters.guaranteed_power_mw', 10.0)  # Default fallback
    )
    
    if guaranteed_mw is None:
        # Final fallback: look for column in df
        guar_col = next((c for c in ['guaranteed_mw', 'guaranteed_power_mw'] if c in df.columns), None)
        if guar_col:
            guaranteed_mw = df[guar_col].mean()
    
    if guaranteed_mw and guaranteed_mw > 0:
        fig.add_hline(
            y=guaranteed_mw,
            line=dict(color='#9b59b6', width=2, dash='dot'),
            annotation_text=f"Guaranteed ({guaranteed_mw:.1f} MW)",
            annotation_position="top right"
        )
    
    # Add utilization stats annotation
    total_available_mwh = wind_available.sum() * dt_seconds / 3600.0
    total_used_mwh = total_used.sum() * dt_seconds / 3600.0
    utilization_pct = (total_used_mwh / total_available_mwh * 100) if total_available_mwh > 0 else 0
    
    fig.update_layout(
        title=kwargs.get('title', f'Wind Utilization Duration Curve ({utilization_pct:.1f}% Utilization)'),
        xaxis_title='Hours at or Above Power Level (Sorted by Value, Not Time)',
        yaxis_title='Power (MW)',
        template='plotly_white',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    
    return fig


@log_graph_errors
def plot_grid_interaction_phase_portrait(df: pd.DataFrame, **kwargs) -> go.Figure:
    """
    Plot phase portrait of Grid Exchange vs Wind Power.
    X=Wind Power, Y=Grid Exchange (+Import/-Export)
    """
    _check_dependencies()
    
    from h2_plant.visualization.utils import downsample_dataframe, get_viz_config
    
    maxpoints = kwargs.get('maxpoints', get_viz_config('performance.max_points_default', 5000)) # scattergl or density needs points
    df_plot = downsample_dataframe(df, max_points=maxpoints)
    
    WIND_CAPACITY_MW = get_viz_config('plant_parameters.wind_capacity_mw', 20.0)
    
    wind_coeff_col = next((c for c in ['wind_coefficient', 'pricing_wind_coefficient'] if c in df_plot.columns), None)
    if not wind_coeff_col: return _empty_figure("No wind data")
    
    wind_coeffs = df_plot[wind_coeff_col].values
    wind_power = wind_coeffs * WIND_CAPACITY_MW
    
    grid_col = next((c for c in ['grid_exchange_mw', 'pricing_grid_exchange_mw', 'P_grid_exchange'] if c in df_plot.columns), None)
    if not grid_col: return _empty_figure("No grid exchange data")
    
    grid_exchange = df_plot[grid_col].values
    
    # Use Density Heatmap to show operational regimes
    fig = go.Figure(go.Histogram2d(
        x=wind_power,
        y=grid_exchange,
        nbinsx=50,
        nbinsy=50,
        colorscale='Viridis',
        colorbar=dict(title='Count')
    ))
    
    fig.update_layout(
        title=kwargs.get('title', 'Grid Interaction Phase Portrait (Density Heatmap)'),
        xaxis_title='Wind Power Available (MW)',
        yaxis_title='Grid Exchange (MW) [+Import / -Export]',
        template='plotly_white'
    )
    return fig


@log_graph_errors
def plot_lcoh_waterfall_breakdown(df: pd.DataFrame, **kwargs) -> go.Figure:
    """
    Plot LCOH breakdown waterfall chart (Economics).
    """
    _check_dependencies()
    
    # Uses values from Config or specific summary columns, usually not time-series.
    # The original implementation used hardcoded placeholders.
    # We should try to read from config or metrics.
    
    from h2_plant.visualization.utils import get_config_value
    
    # Placeholder strategy: read from kwargs or config, fall back to defaults
    # In real app, these might come from a "metrics" dict passed in kwargs or global result
    
    energy_cost = kwargs.get('energy_cost', 4.5)
    capex = kwargs.get('capex', 2.0)
    opex = kwargs.get('opex', 1.0)
    water = kwargs.get('water', 0.1)
    compression = kwargs.get('compression', 0.5)
    
    # Try to read from dataframe attributes if available (some pipelines attach results to df.attrs)
    metrics = df.attrs.get('metrics', {})
    
    # P0 FIX: Guard clause - return empty figure if LCOH metrics are not available
    # Prevents rendering a meaningless chart with placeholder defaults.
    if 'lcoh_breakdown' not in metrics and not any(k in kwargs for k in ['energy_cost', 'capex', 'opex']):
        return _empty_figure("LCOH metrics not available. Run economic analysis first.")
    
    if 'lcoh_breakdown' in metrics:
        b = metrics['lcoh_breakdown']
        energy_cost = b.get('energy', energy_cost)
        capex = b.get('capex', capex)
        opex = b.get('opex', opex)
        water = b.get('water', water)
        compression = b.get('compression', compression)
    
    fig = go.Figure(go.Waterfall(
        name = "LCOH Breakdown",
        orientation = "v",
        measure = ["relative", "relative", "relative", "relative", "relative", "total"],
        x = ["Energy", "CAPEX", "O&M", "Water", "Compression", "Total LCOH"],
        textposition = "outside",
        text = [f"{x:.2f}" for x in [energy_cost, capex, opex, water, compression, sum([energy_cost, capex, opex, water, compression])]],
        y = [energy_cost, capex, opex, water, compression, 0],
        connector = {"line": {"color": "rgb(63, 63, 63)"}},
    ))
    
    fig.update_layout(
        title=kwargs.get('title', 'Levelized Cost of Hydrogen (LCOH) Breakdown'),
        yaxis_title='Cost (EUR/kg H2)',
        template='plotly_white'
    )
    return fig


@log_graph_errors
def plot_pem_performance_surface(df: pd.DataFrame, **kwargs) -> go.Figure:
    _check_dependencies()
    
    from h2_plant.visualization.utils import downsample_dataframe, get_time_axis_hours, get_viz_config
    
    # This requires reconstructing the surface from scattered points
    # For simplicity, we'll use a 3D Scatter plot which is easier with unstructured data
    
    maxpoints = kwargs.get('maxpoints', get_viz_config('performance.max_points_default', 2000))
    df_plot = downsample_dataframe(df, max_points=maxpoints)
    hours = get_time_axis_hours(df_plot)
    
    # power = data['pem'].get('power_mw', [])    # New Y axis
    # production = data['pem'].get('h2_production_kg_h', []) # New Z axis
    
    prod_col = next((c for c in ['H2_pem', 'H2_pem_kg_h'] if c in df_plot.columns), None)
    pow_col = next((c for c in ['P_pem', 'P_pem_mw'] if c in df_plot.columns), None)
    
    if not prod_col or not pow_col: return _empty_figure("No PEM performance data")
    
    production = df_plot[prod_col].values
    power = df_plot[pow_col].values
    # If power in kW, convert?
    if power.mean() > 1000: power /= 1000
    
    # use_webgl = kwargs.get('use_webgl', False)
    
    fig = go.Figure(data=[go.Scatter3d(
        x=hours,
        y=production,
        z=power,
        mode='markers',
        marker=dict(
            size=3,
            color=production,
            colorscale='Viridis',
            opacity=0.8,
            colorbar=dict(title='H2 Prod (kg/h)')
        )
    )])
    
    fig.update_layout(
        title=kwargs.get('title', 'PEM Performance Surface (Time vs Power vs H2)'),
        scene=dict(
            xaxis_title='Time (hours)',
            yaxis_title='H2 Production (kg/h)',
            zaxis_title='Power (MW)'
        ),
        template='plotly_white'
    )
    return fig


@log_graph_errors
def plot_arbitrage_opportunity(df: pd.DataFrame, dpi: int = 100, **kwargs) -> go.Figure:
    """
    Plot Arbitrage Opportunity (Interactive).
    
    Visualizes the relationship between hydrogen production and wind availability:
    - Left Axis (Area): Total H2 Production (kg/min)
    - Right Axis (Line): Available Wind Power (MW)
    
    This shows the correlation between H2 production and wind resource availability.
    """
    _check_dependencies()
    
    from h2_plant.visualization.utils import downsample_dataframe, get_time_axis_hours, get_viz_config, get_config_value
    
    # Downsample using utils
    maxpoints = kwargs.get('maxpoints', get_viz_config('performance.max_points_default', 2000))
    df_plot = downsample_dataframe(df, max_points=maxpoints)
    
    # Extract Data
    hours = get_time_axis_hours(df_plot)
    
    spot_price = df_plot.get('Spot', df_plot.get('spot_price'))
    
    # H2 Production
    h2_soec = df_plot.get('H2_soec', df_plot.get('H2_soec_kg', pd.Series(0, index=df_plot.index)))
    h2_pem = df_plot.get('H2_pem', df_plot.get('H2_pem_kg', pd.Series(0, index=df_plot.index)))
    h2_total = h2_soec + h2_pem
    
    # Reference metrics
    # Try config in attrs, or get_viz_config
    ppa_price = get_config_value(df_plot, 'ppa_price_eur_mwh', 
                                get_viz_config('plant_parameters.ppa_price', 50.0))
        
    threshold = df_plot.get('spot_threshold_eur_mwh', pd.Series(np.nan, index=df_plot.index))
    
    # Create Dual-Axis Figure
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Trace 1: H2 Production (Left Axis, Area/Bar)
    h2_color = get_viz_config('styling.colors.h2total', 'rgba(46, 204, 113, 0.6)')
    fig.add_trace(
        get_scatter_type(len(hours))(
            x=hours, 
            y=h2_total, 
            name="H2 Production Rate (kg/min)",
            fill='tozeroy',
            line=dict(color=h2_color, width=1), 
            hovertemplate="Time: %{x:.1f}h<br>Rate: %{y:.2f} kg/min<extra></extra>"
        ),
        secondary_y=False
    )
    
    # Trace 2: Available Wind Potential (Right Axis, Line)
    # Use P_offer (wind power available) instead of spot price
    wind_col = next((c for c in ['P_offer', 'wind_power_mw', 'wind_available_mw'] if c in df_plot.columns), None)
    
    if wind_col:
        wind_power = df_plot[wind_col]
        wind_color = get_viz_config('styling.colors.wind', '#9b59b6')
        fig.add_trace(
            get_scatter_type(len(hours))(
                x=hours, 
                y=wind_power, 
                name="Available Wind Power (MW)",
                line=dict(color=wind_color, width=2),
                hovertemplate="Wind: %{y:.2f} MW<extra></extra>"
            ),
            secondary_y=True
        )
    
    # Trace 3: Threshold (Right Axis, Dashed Line)
    if isinstance(threshold, pd.Series) and threshold.mean() > 0:
        fig.add_trace(
            get_scatter_type(len(hours))(
                x=hours,
                y=threshold,
                name="Purchase Threshold",
                line=dict(color='#E74C3C', width=2, dash='dash'), # Red Dashed
                hovertemplate="Threshold: %{y:.2f} EUR/MWh<extra></extra>"
            ),
            secondary_y=True
        )
        
    # P1 PARITY FIX: Add H2 Equivalent Price threshold (matches Matplotlib version)
    # Formula: h2_equiv = (1000 / η_H2) * h2_price_eur_kg
    config = df.attrs.get('config', {})
    h2_price = config.get('h2_price_eur_kg')
    efficiency = config.get('soec_h2_kwh_kg', 37.5)  # Default SOEC efficiency
    
    if h2_price is not None and efficiency > 0:
        h2_equiv_price = (1000 / efficiency) * h2_price
        # Also show this in legend or just annotation? User asked about "dotted red line", 
        # which was PPA. If this was also invisible, maybe convert it too.
        # But H2 Breakeven is usually green dash.
        eq_series = pd.Series(h2_equiv_price, index=range(len(hours)))
        fig.add_trace(
            get_scatter_type(len(hours))(
                x=hours,
                y=eq_series,
                name=f"H2 Breakeven ({h2_equiv_price:.0f} EUR/MWh)",
                line=dict(color='green', width=2, dash='dash'),
                hovertemplate="Breakeven: %{y:.2f} EUR/MWh<extra></extra>"
            ),
            secondary_y=True
        )

    # Layout
    fig.update_layout(
        title=kwargs.get('title', 'H2 Production vs Wind Availability'),
        template='plotly_white',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    fig.update_yaxes(title_text='H2 Production Rate (kg/min)', secondary_y=False)
    fig.update_yaxes(title_text='Wind Power (MW)', secondary_y=True)
    fig.update_xaxes(title_text='Simulation Time (Hours)')
    
    return fig


# =============================================================================
# P2 REFACTORING: NEW PLOTLY TWIN IMPLEMENTATIONS
# =============================================================================

@log_graph_errors
def plot_effective_ppa(df: pd.DataFrame, **kwargs) -> go.Figure:
    """
    Plot Effective PPA Price over time (Interactive twin of create_effective_ppa_figure).
    
    Shows how the weighted average PPA price varies with wind availability.
    """
    _check_dependencies()
    
    from h2_plant.visualization.utils import downsample_dataframe, get_time_axis_hours, get_viz_config
    
    maxpoints = kwargs.get('maxpoints', get_viz_config('performance.max_points_default', 2000))
    df_plot = downsample_dataframe(df, max_points=maxpoints)
    hours = get_time_axis_hours(df_plot)
    
    # Primary column: ppa_price_effective_eur_mwh
    ppa_col = next((c for c in ['ppa_price_effective_eur_mwh', 'effective_ppa_price'] if c in df_plot.columns), None)
    spot_col = next((c for c in ['spot_price', 'Spot'] if c in df_plot.columns), None)
    
    if not ppa_col:
        return _empty_figure("No Effective PPA data found")
    
    ppa_price = df_plot[ppa_col].values
    spot_price = df_plot[spot_col].values if spot_col else None
    
    fig = go.Figure()
    
    # Effective PPA line
    fig.add_trace(get_scatter_type(len(hours))(
        x=hours,
        y=ppa_price,
        mode='lines',
        name='Effective PPA Price',
        line=dict(color='#e74c3c', width=2)
    ))
    
    # Spot price overlay
    if spot_price is not None:
        fig.add_trace(get_scatter_type(len(hours))(
            x=hours,
            y=spot_price,
            mode='lines',
            name='Spot Price',
            line=dict(color='#3498db', width=1.5, dash='dot'),
            opacity=0.7
        ))
    
    fig.update_layout(
        title=kwargs.get('title', 'Effective PPA Price vs Spot Price'),
        xaxis_title='Time (hours)',
        yaxis_title='Price (EUR/MWh)',
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig


@log_graph_errors
def plot_storage_apc(df: pd.DataFrame, **kwargs) -> go.Figure:
    """
    Plot Storage APC (Advanced Process Control) visualization (Interactive twin).
    
    Shows SOC, control zones, and action factor.
    """
    _check_dependencies()
    
    from h2_plant.visualization.utils import downsample_dataframe, get_time_axis_hours, get_viz_config
    from plotly.subplots import make_subplots
    
    maxpoints = kwargs.get('maxpoints', get_viz_config('performance.max_points_default', 2000))
    df_plot = downsample_dataframe(df, max_points=maxpoints)
    hours = get_time_axis_hours(df_plot)
    
    # Required columns
    soc_col = next((c for c in ['storage_soc', 'soc', 'state_of_charge'] if c in df_plot.columns), None)
    zone_col = next((c for c in ['storage_zone', 'control_zone'] if c in df_plot.columns), None)
    factor_col = next((c for c in ['storage_action_factor', 'action_factor'] if c in df_plot.columns), None)
    
    if not soc_col:
        return _empty_figure("No Storage SOC data found")
    
    soc = df_plot[soc_col].values * 100  # Convert to percentage
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # SOC line (left axis)
    fig.add_trace(
        get_scatter_type(len(hours))(
            x=hours,
            y=soc,
            mode='lines',
            name='State of Charge (%)',
            line=dict(color='#2ecc71', width=2),
            fill='tozeroy',
            fillcolor='rgba(46, 204, 113, 0.2)'
        ),
        secondary_y=False
    )
    
    # Zone thresholds as horizontal bands
    zone_colors = {
        'A': 'rgba(231, 76, 60, 0.2)',   # Red - Low
        'B': 'rgba(241, 196, 15, 0.2)',  # Yellow - Normal
        'C': 'rgba(46, 204, 113, 0.2)'   # Green - High
    }
    
    # Add zone threshold lines
    fig.add_hline(y=30, line_dash="dash", line_color="red", annotation_text="Zone A (30%)")
    fig.add_hline(y=60, line_dash="dash", line_color="orange", annotation_text="Zone B (60%)")
    fig.add_hline(y=90, line_dash="dash", line_color="green", annotation_text="Zone C (90%)")
    
    # Action factor (right axis) if available
    if factor_col:
        factor = df_plot[factor_col].values
        fig.add_trace(
            get_scatter_type(len(hours))(
                x=hours,
                y=factor,
                mode='lines',
                name='Action Factor',
                line=dict(color='#9b59b6', width=1.5, dash='dot')
            ),
            secondary_y=True
        )
    
    fig.update_layout(
        title=kwargs.get('title', 'Storage APC Control'),
        template='plotly_white',
        hovermode='x unified'
    )
    
    fig.update_yaxes(title_text='Hydrogen Storage SOC (%)', range=[0, 100], secondary_y=False)
    fig.update_yaxes(title_text='Action Factor', range=[0, 1.2], secondary_y=True)
    fig.update_xaxes(title_text='Time (hours)')
    
    return fig


@log_graph_errors
def plot_temporal_averages(df: pd.DataFrame, **kwargs) -> go.Figure:
    """
    Plot Temporal Averages (Hourly aggregated price, power, H2) - Interactive twin.
    
    Calculates efficiency dynamically by determining the Specific Production (kg/MWh)
    achieved by each system (Ratio of kg/h output to MW input), then deriving
    the % LHV efficiency. This eliminates hardcoded parameters.
    """
    _check_dependencies()
    
    from h2_plant.visualization.utils import get_viz_config
    from plotly.subplots import make_subplots
    
    # Aggregate to hourly
    df_hourly = df.copy()
    if 'minute' in df_hourly.columns:
        df_hourly['hour'] = df_hourly['minute'] // 60
    elif 'hour' not in df_hourly.columns:
        df_hourly['hour'] = df_hourly.index // 60
    
    # Group by hour
    agg_cols = {}
    
    # Price
    price_col = next((c for c in ['spot_price', 'Spot'] if c in df.columns), None)
    if price_col:
        agg_cols[price_col] = 'mean'
    
    # Power (MW)
    power_cols = [c for c in ['P_pem', 'P_soec', 'P_offer', 'P_sold', 'sell_power_mw'] if c in df.columns]
    for c in power_cols:
        agg_cols[c] = 'mean'
    
    # H2 Production
    # Use MEAN to get Average Rate (kg/h). 
    # (Total Mass in 1h = Average Rate kg/h * 1h)
    h2_cols = [c for c in ['H2_pem', 'H2_soec', 'H2_pem_kg', 'H2_soec_kg', 'H2_atr', 'H2_atr_kg'] if c in df.columns]
    for c in h2_cols:
        agg_cols[c] = 'mean'
    
    if not agg_cols:
        return _empty_figure("No aggregatable columns found")
    
    df_agg = df_hourly.groupby('hour').agg(agg_cols).reset_index()
    hours = df_agg['hour'].values
    
    # Identify specific columns for plotting
    p_soec_col = next((c for c in ['P_soec'] if c in df_agg.columns), None)
    p_pem_col = next((c for c in ['P_pem'] if c in df_agg.columns), None)
    p_wind_col = next((c for c in ['P_offer'] if c in df_agg.columns), None)
    p_grid_col = next((c for c in ['P_sold', 'sell_power_mw'] if c in df_agg.columns), None)
    
    # Calculate Total H2 Rate (Sum of all present H2 component rates)
    df_agg['Total_H2'] = 0.0
    agg_h2_cols = [c for c in h2_cols if c in df_agg.columns]
    for c in agg_h2_cols:
        df_agg['Total_H2'] += df_agg[c]

    # For efficiency calculation re-identification (keep specific cols for calc)
    h2_soec_col = next((c for c in ['H2_soec', 'H2_soec_kg'] if c in df_agg.columns), None)
    h2_pem_col = next((c for c in ['H2_pem', 'H2_pem_kg'] if c in df_agg.columns), None)
    
    # Calculate Efficiencies dynamically using Specific Production (kg/MWh)
    # LHV H2 = 33.33 kWh/kg = 0.03333 MWh/kg
    LHV_MWh_kg = 0.03333
    
    if p_soec_col and h2_soec_col:
        # Specific Production (kg/MWh) = H2 Rate (kg/h) / Power (MW)
        # H2 data is kg/min (mass per minute), so multiply by 60 to get kg/h
        with np.errstate(divide='ignore', invalid='ignore'):
            df_agg['Spec_Prod_SOEC'] = np.where(df_agg[p_soec_col] > 0.01, (df_agg[h2_soec_col] * 60) / df_agg[p_soec_col], 0)
            df_agg['Eff_soec'] = df_agg['Spec_Prod_SOEC'] * LHV_MWh_kg * 100
            df_agg['Eff_soec'] = df_agg['Eff_soec'].clip(0, 100)
    
    if p_pem_col and h2_pem_col:
        # H2 data is kg/min -> kg/h conversion
        with np.errstate(divide='ignore', invalid='ignore'):
            df_agg['Spec_Prod_PEM'] = np.where(df_agg[p_pem_col] > 0.01, (df_agg[h2_pem_col] * 60) / df_agg[p_pem_col], 0)
            df_agg['Eff_pem'] = df_agg['Spec_Prod_PEM'] * LHV_MWh_kg * 100
            df_agg['Eff_pem'] = df_agg['Eff_pem'].clip(0, 100)
    
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        subplot_titles=('Average Wind & Grid Power', 'Average Consumption', 
                                        'Total H2 Production', 'System Efficiency (Dynamic SEC)'),
                        vertical_spacing=0.08)
    
    # Row 1: Wind vs Grid (Double Bars)
    if p_wind_col:
        fig.add_trace(
            go.Bar(x=hours, y=df_agg[p_wind_col], name='Wind Power', marker_color='#3498db'),
            row=1, col=1
        )
    if p_grid_col:
        fig.add_trace(
            go.Bar(x=hours, y=df_agg[p_grid_col], name='Grid Export', marker_color='#f1c40f'),
            row=1, col=1
        )
    
    # Row 2: Consumption (SOEC/PEM)
    if p_soec_col:
        fig.add_trace(
            go.Bar(x=hours, y=df_agg[p_soec_col], name='SOEC Power', marker_color='#2ecc71'),
            row=2, col=1
        )
    if p_pem_col:
        fig.add_trace(
            go.Bar(x=hours, y=df_agg[p_pem_col], name='PEM Power', marker_color='#e74c3c'),
            row=2, col=1
        )
    
    # Row 3: H2 (Combined Total)
    if agg_h2_cols:
        fig.add_trace(
            go.Bar(x=hours, y=df_agg['Total_H2'], name='Total H2 Production', marker_color='#9b59b6'),
            row=3, col=1
        )
    
    # Row 4: Efficiencies (Dynamic load-dependent calculation)
    # The efficiency here reflects the ACTUAL observed efficiency from simulation,
    # which includes the part-load penalty from the spline-based SEC curve.
    if 'Eff_soec' in df_agg.columns:
        # Calculate SEC (inverse of Specific Production) for display
        df_agg['SEC_soec'] = np.where(
            df_agg['Spec_Prod_SOEC'] > 0.01, 
            1000.0 / df_agg['Spec_Prod_SOEC'],  # kWh/kg
            0
        )
        
        fig.add_trace(
            go.Scatter(
                x=hours, 
                y=df_agg['Eff_soec'], 
                name='SOEC Efficiency (Dynamic)', 
                mode='lines+markers', 
                line=dict(color='#2ecc71', width=2),
                hovertemplate=(
                    'Time: %{x}h<br>'
                    'Efficiency: %{y:.1f}% LHV<br>'
                    'SEC: %{customdata[0]:.1f} kWh/kg<br>'
                    'Yield: %{customdata[1]:.1f} kg/MWh<extra></extra>'
                ),
                customdata=np.column_stack([df_agg['SEC_soec'], df_agg['Spec_Prod_SOEC']])
            ),
            row=4, col=1
        )
        
        # Add reference line for BOL efficiency at 100% load (37.54 kWh/kg → 88.8% LHV)
        bol_eff_100pct = (1000.0 / 37.54) * 0.03333 * 100  # ~88.8%
        fig.add_hline(
            y=bol_eff_100pct, 
            line_dash="dash", 
            line_color="rgba(46, 204, 113, 0.5)",
            annotation_text=f"BOL @100% ({bol_eff_100pct:.0f}%)",
            annotation_position="bottom right",
            row=4, col=1
        )
        
    if 'Eff_pem' in df_agg.columns:
         fig.add_trace(
            go.Scatter(
                x=hours, 
                y=df_agg['Eff_pem'], 
                name='PEM Efficiency', 
                mode='lines+markers', 
                line=dict(color='#e74c3c', width=2),
                hovertemplate='Time: %{x}h<br>Eff: %{y:.1f}%<br>Yield: %{customdata:.2f} kg/MWh<extra></extra>',
                customdata=df_agg['Spec_Prod_PEM']
            ),
            row=4, col=1
        )
    
    fig.update_layout(
        title=kwargs.get('title', 'Hourly Temporal Averages'),
        template='plotly_white',
        height=800, # Increased height for 4 rows
        showlegend=True,
        barmode='group'
    )
    
    # Update axes with units
    fig.update_yaxes(title_text="Power (MW)", row=1, col=1)
    fig.update_yaxes(title_text="Power (MW)", row=2, col=1)
    fig.update_yaxes(title_text="H2 Rate (kg/h)", row=3, col=1)
    fig.update_yaxes(title_text="Efficiency (% LHV)", range=[0, 100], row=4, col=1)
    
    fig.update_xaxes(title_text='Hour of Day', row=4, col=1)
    
    return fig


@log_graph_errors
def plot_thermal_load_time_series(df: pd.DataFrame, **kwargs) -> go.Figure:
    """
    Total Thermal Load Time Series (Chillers + Dry Coolers + Intercoolers).
    Merged: Interactive Lines + Stacked toggle.
    """
    _check_dependencies()
    from h2_plant.visualization.utils import downsample_dataframe, get_time_axis_hours, get_viz_config, find_columns_by_type
    
    maxpoints = kwargs.get('maxpoints', get_viz_config('performance.max_points_default', 2000))
    df_plot = downsample_dataframe(df, max_points=maxpoints)
    hours = get_time_axis_hours(df_plot)
    
    # Collect all thermal loads
    loads = {}
    
    # Chillers (Cooling Load)
    chillers = find_columns_by_type(df_plot, 'Chiller', 'cooling_load_kw')
    for name, col in chillers.items():
        loads[f"{name} (Chiller)"] = df_plot[col]
        
    # Dry Coolers / Intercoolers (Heat Rejected / TQC)
    # Check 'tqc_duty_kw' first (Total Quality Control/Total Heat), else 'heat_rejected_kw'
    dcs = find_columns_by_type(df_plot, 'DryCooler', 'tqc_duty_kw')
    if not dcs:
        dcs = find_columns_by_type(df_plot, 'DryCooler', 'heat_rejected_kw')
    
    for name, col in dcs.items():
        loads[f"{name} (DryCooler)"] = df_plot[col]
        
    # Intercoolers
    ics = find_columns_by_type(df_plot, 'Intercooler', 'tqc_duty_kw')
    for name, col in ics.items():
        loads[f"{name} (Intercooler)"] = df_plot[col]

    if not loads:
        return _empty_figure("No Thermal Load Data Found")

    fig = go.Figure()
    ScatterType = get_scatter_type(len(hours))
    trace_counter = 0
    stackable_indices = []
    
    for name, series in loads.items():
        fig.add_trace(ScatterType(
            x=hours,
            y=series,
            mode='lines',
            name=name,
            line=dict(width=1.5, color=_enhanced_color(name)),
            stackgroup=None, # Default Lines
            hovertemplate='%{y:.2f} kW<extra></extra>'
        ))
        stackable_indices.append(trace_counter)
        trace_counter += 1

    # Add Toggle (using new utility)
    style_menu = _build_style_toggle_menu(stackable_indices, 'thermal_load', x_position=0.82)
    
    fig.update_layout(
        updatemenus=[style_menu],
        title=kwargs.get('title', 'System Thermal Load Breakdown'),
        xaxis_title='Time (hours)',
        yaxis_title='Thermal Load (kW)',
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig


@log_graph_errors
def plot_soec_module_heatmap(df: pd.DataFrame, **kwargs) -> go.Figure:
    """
    Plot SOEC Module Activity Heatmap (Interactive twin).
    
    Shows module power over time as a heatmap.
    """
    _check_dependencies()
    
    from h2_plant.visualization.utils import downsample_dataframe, get_time_axis_hours, get_viz_config
    
    maxpoints = kwargs.get('maxpoints', get_viz_config('performance.max_points_default', 500))
    df_plot = downsample_dataframe(df, max_points=maxpoints)
    hours = get_time_axis_hours(df_plot)
    
    # Find module columns
    # Assuming columns like 'SOEC_Module_1_power_mw', etc.
    # Pattern matching for 'Module_X'
    import re
    
    module_data = {}
    pattern = re.compile(r'SOEC_Module_(\d+)_power_mw')
    
    for col in df_plot.columns:
        match = pattern.search(col)
        if match:
            module_num = int(match.group(1))
            module_data[module_num] = df_plot[col].values
            
    if not module_data:
        return _empty_figure("No SOEC Module data found")
        
    # Create Matrix: Rows=Modules, Cols=Time
    # Sort by module number
    sorted_modules = sorted(module_data.keys())
    matrix = np.array([module_data[m] for m in sorted_modules])
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=hours,
        y=[f"Mod {m}" for m in sorted_modules],
        colorscale='Viridis',
        colorbar=dict(title='Power (MW)'),
        hovertemplate='Time: %{x}h<br>Module: %{y}<br>Power: %{z:.2f} MW<extra></extra>'
    ))
    
    fig.update_layout(
        title=kwargs.get('title', 'SOEC Module Activation'),
        xaxis_title='Time (hours)',
        yaxis_title='Module',
        template='plotly_white'
    )
    
    return fig


# =============================================================================
# CHILLER & DRYCOOLER GRAPHS (Plotly Twins)
# =============================================================================

@log_graph_errors
def plot_chiller_cooling_load(df: pd.DataFrame, **kwargs) -> go.Figure:
    """
    Plot Chiller cooling load over time (kW).
    Merged: Interactive Lines + Stacked toggle.
    """
    _check_dependencies()
    from h2_plant.visualization.utils import downsample_dataframe, get_time_axis_hours, get_viz_config, find_columns_by_type
    
    maxpoints = kwargs.get('maxpoints', get_viz_config('performance.max_points_default', 2000))
    df_plot = downsample_dataframe(df, max_points=maxpoints)
    hours = get_time_axis_hours(df_plot)
    
    cooling_data = find_columns_by_type(df_plot, 'Chiller', 'cooling_load_kw')
    
    if not cooling_data:
        return _empty_figure("No Chiller cooling load data found")
    
    fig = go.Figure()
    ScatterType = get_scatter_type(len(hours))
    trace_counter = 0
    stackable_indices = []
    
    for comp_id, col in cooling_data.items():
        fig.add_trace(ScatterType(
            x=hours,
            y=df_plot[col],
            mode='lines',
            name=comp_id,
            line=dict(width=1.5, color=_get_subsystem_color(comp_id)),
            stackgroup=None, # Default to Lines
            hovertemplate='%{y:.2f} kW<extra></extra>'
        ))
        stackable_indices.append(trace_counter)
        trace_counter += 1
    
    # Add Toggle
    style_menu = _build_style_toggle_menu(stackable_indices, 'chiller_cooling', x_position=0.82)
    
    fig.update_layout(
        updatemenus=[style_menu],
        title=kwargs.get('title', 'Chiller Cooling Load'),
        xaxis_title='Time (hours)',
        yaxis_title='Cooling Load (kW)',
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig


@log_graph_errors
def plot_chiller_power(df: pd.DataFrame, **kwargs) -> go.Figure:
    """
    Plot Chiller electrical power consumption over time (kW).
    Merged: Interactive Lines + Stacked toggle.
    """
    _check_dependencies()
    from h2_plant.visualization.utils import downsample_dataframe, get_time_axis_hours, get_viz_config, find_columns_by_type
    
    maxpoints = kwargs.get('maxpoints', get_viz_config('performance.max_points_default', 2000))
    df_plot = downsample_dataframe(df, max_points=maxpoints)
    hours = get_time_axis_hours(df_plot)
    
    elec_data = find_columns_by_type(df_plot, 'Chiller', 'electrical_power_kw')
    
    if not elec_data:
        return _empty_figure("No Chiller power data found")
    
    fig = go.Figure()
    ScatterType = get_scatter_type(len(hours)) # Continuing...
    trace_counter = 0
    stackable_indices = []
    
    for comp_id, col in elec_data.items():
        fig.add_trace(ScatterType(
            x=hours,
            y=df_plot[col],
            mode='lines',
            name=comp_id,
            line=dict(width=1.5, color=_get_subsystem_color(comp_id)),
            stackgroup=None, # Default to Lines
            hovertemplate='%{y:.2f} kW<extra></extra>'
        ))
        stackable_indices.append(trace_counter)
        trace_counter += 1
    
    # Add Toggle
    style_menu = _build_style_toggle_menu(stackable_indices, 'chiller_power', x_position=0.82)
    
    fig.update_layout(
        updatemenus=[style_menu],
        title=kwargs.get('title', 'Chiller Electrical Consumption'),
        xaxis_title='Time (hours)',
        yaxis_title='Electrical Power (kW)',
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig


@log_graph_errors
def plot_dry_cooler_performance(df: pd.DataFrame, **kwargs) -> go.Figure:
    """
    Dry Cooler Performance: Heat Rejection and Outlet Temperature.
    
    Features:
    - Heat Rejection (Stacked/Line Toggle) + Temperature (Lines)
    - Subsystem Selection Dropdown (All/PEM/SOEC/ATR/Storage)
    """
    _check_dependencies()
    from h2_plant.visualization.utils import downsample_dataframe, get_time_axis_hours, get_viz_config, find_columns_by_type
    from plotly.subplots import make_subplots
    
    maxpoints = kwargs.get('maxpoints', get_viz_config('performance.max_points_default', 2000))
    df_plot = downsample_dataframe(df, max_points=maxpoints)
    hours = get_time_axis_hours(df_plot)
    
    # Find Data
    heat_rejected = find_columns_by_type(df_plot, 'DryCooler', 'heat_rejected_kw')
    outlet_temp = find_columns_by_type(df_plot, 'DryCooler', 'outlet_temp_c')
    
    # Add Intercoolers
    ic_heat = find_columns_by_type(df_plot, 'Intercooler', 'heat_rejected_kw')
    ic_temp = find_columns_by_type(df_plot, 'Intercooler', 'outlet_temp_c')
    
    heat_rejected.update(ic_heat)
    outlet_temp.update(ic_temp)
    
    if not heat_rejected and not outlet_temp:
        return _empty_figure("No Dry Cooler/Intercooler data found")
    
    # FIX: Subsystem classification helper
    def classify_subsystem(comp_id: str) -> str:
        u = comp_id.upper()
        if 'PEM' in u: return 'PEM'
        if 'SOEC' in u: return 'SOEC'
        if 'ATR' in u: return 'ATR'
        if any(x in u for x in ['STORAGE', 'STORE', 'TANK', 'HP_', 'LP_', 'COMPRESSOR']):
            return 'Storage'
        return 'BOP'  # Balance of Plant
    
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Heat Rejection", "Outlet Temperature")
    )
    
    ScatterType = get_scatter_type(len(hours))
    
    # Track indices for toggle buttons and subsystem categorization
    heat_trace_indices = []
    trace_counter = 0
    trace_subsystems = []  # Subsystem for each trace

    # Row 1: Heat Rejected (These will be toggled)
    for comp_id, col in heat_rejected.items():
        subsys = classify_subsystem(comp_id)
        trace_subsystems.append(subsys)
        
        fig.add_trace(ScatterType(
            x=hours,
            y=df_plot[col],
            mode='lines',
            name=f"{comp_id} Heat",
            line=dict(width=1.5, color=_get_subsystem_color(comp_id)),
            stackgroup=None, # Default to Lines
            legendgroup=comp_id,
            hovertemplate=f'<b>{comp_id}</b><br>%{{y:.2f}} kW<extra></extra>'
        ), row=1, col=1)
        
        heat_trace_indices.append(trace_counter)
        trace_counter += 1
        
    # Row 2: Outlet Temperature (These remain lines)
    for comp_id, col in outlet_temp.items():
        subsys = classify_subsystem(comp_id)
        trace_subsystems.append(subsys)
        
        data_c = df_plot[col].values
        # Simple heuristic for K->C conversion
        if np.nanmean(data_c) > 200:
            data_c = data_c - 273.15
            
        fig.add_trace(ScatterType(
            x=hours,
            y=data_c,
            mode='lines',
            name=f"{comp_id} Temp",
            line=dict(width=1.5, dash='dot', color=_get_subsystem_color(comp_id)),
            legendgroup=comp_id,
            showlegend=False, 
            hovertemplate=f'<b>{comp_id}</b><br>%{{y:.1f}} deg C<extra></extra>'
        ), row=2, col=1)
        
        trace_counter += 1

    n_traces = trace_counter
    
    # Menu 1: Lines/Stacked Toggle (existing, targets ONLY heat traces)
    style_menu = _build_style_toggle_menu(heat_trace_indices, 'dry_cooler_heat', x_position=0.72)
    
    # FIX: Menu 2: Subsystem Selection Dropdown (NEW)
    present_subsystems = sorted(list(set(trace_subsystems)))
    
    # "All" button
    subsystem_buttons = [
        dict(label="All", method="update", args=[{"visible": [True] * n_traces}])
    ]
    
    # Per-subsystem buttons
    for subsys in present_subsystems:
        visibility = [ts == subsys for ts in trace_subsystems]
        subsystem_buttons.append(dict(
            label=subsys,
            method="update",
            args=[{"visible": visibility}]
        ))
    
    # Combine both menus
    fig.update_layout(
        updatemenus=[
            style_menu,  # Lines/Stacked at x=0.72
            dict(  # Subsystem filter at x=1.0
                type="dropdown",
                direction="down",
                x=1.0, y=1.15,
                xanchor="left",
                showactive=True,
                active=0,  # Default to "All"
                buttons=subsystem_buttons
            )
        ],
        title=kwargs.get('title', 'Dry Cooler & Intercooler Performance'),
        template='plotly_white',
        hovermode='x unified',
        height=700
    )
    
    fig.update_yaxes(title_text="Heat (kW)", row=1, col=1)
    fig.update_yaxes(title_text="Temp (deg C)", row=2, col=1)
    fig.update_xaxes(title_text="Time (hours)", row=2, col=1)
    
    return fig


# =============================================================================
# ENHANCED STACKED GRAPHS (with UX improvements)
# =============================================================================

# Shared color palette for enhanced graphs
_ENHANCED_PALETTE = [
    '#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
    '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52',
    '#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD',
    '#8C564B', '#E377C2', '#7F7F7F', '#BCBD22', '#17BECF',
    '#AEC7E8', '#FFBB78', '#98DF8A', '#C49C94'
]

def _enhanced_color(comp_id: str) -> str:
    """Deterministic color from component ID hash."""
    h = hash(comp_id) % len(_ENHANCED_PALETTE)
    return _ENHANCED_PALETTE[h]


@log_graph_errors
def plot_chiller_cooling_load_stacked(df: pd.DataFrame, **kwargs) -> go.Figure:
    """
    Enhanced stacked area chart of Chiller cooling load with UX improvements.
    
    Features:
    - Component-level colors (hash-based, stable)
    - Top N visibility (default: 8)
    - Show All/Hide All/Top N buttons
    - Sort by mean load
    """
    _check_dependencies()
    
    from h2_plant.visualization.utils import downsample_dataframe, get_time_axis_hours, get_viz_config, find_columns_by_type
    
    top_n = kwargs.get('top_n', 8)
    maxpoints = kwargs.get('maxpoints', get_viz_config('performance.max_points_default', 2000))
    df_plot = downsample_dataframe(df, max_points=maxpoints)
    hours = get_time_axis_hours(df_plot)
    
    cooling_data = find_columns_by_type(df_plot, 'Chiller', 'cooling_load_kw')
    
    if not cooling_data:
        return _empty_figure("No Chiller cooling load data found")
    
    # Calculate mean loads and sort
    comp_means = {cid: abs(df_plot[col].mean()) for cid, col in cooling_data.items()}
    sorted_comps = sorted(cooling_data.keys(), key=lambda x: -comp_means[x])
    top_n_ids = set(sorted_comps[:top_n])
    
    fig = go.Figure()
    ScatterType = get_scatter_type(len(hours))
    trace_ids = []
    
    for rank, comp_id in enumerate(sorted_comps):
        col = cooling_data[comp_id]
        trace_ids.append(comp_id)
        is_visible = comp_id in top_n_ids
        
        fig.add_trace(ScatterType(
            x=hours, y=df_plot[col], mode='lines', name=comp_id.replace('_', ' '),
            stackgroup='one', legendrank=rank,
            visible=True if is_visible else 'legendonly',
            line=dict(width=0.5, color=_enhanced_color(comp_id)),
            fillcolor=_enhanced_color(comp_id),
            hovertemplate=f"<b>{comp_id.replace('_', ' ')}</b><br>%{{y:.2f}} kW<extra></extra>"
        ))
    
    n = len(trace_ids)
    buttons = [
        dict(label="Show All", method="update", args=[{"visible": [True] * n}]),
        dict(label="Hide All", method="update", args=[{"visible": ['legendonly'] * n}]),
        dict(label=f"Top {top_n}", method="update", args=[{"visible": [cid in top_n_ids for cid in trace_ids]}])
    ]
    
    fig.update_layout(
        title=kwargs.get('title', f'Chiller Cooling Load - Stacked (Top {top_n} shown)'),
        xaxis_title='Time (hours)', yaxis_title='Cooling Load (kW)',
        template='plotly_white', hovermode='x unified',
        legend=dict(groupclick="toggleitem"),
        updatemenus=[dict(type="dropdown", direction="down", x=1.0, y=1.15, xanchor="left", showactive=True, buttons=buttons)]
    )
    return fig


@log_graph_errors
def plot_chiller_power_stacked(df: pd.DataFrame, **kwargs) -> go.Figure:
    """
    Enhanced stacked area chart of Chiller electrical power with UX improvements.
    """
    _check_dependencies()
    
    from h2_plant.visualization.utils import downsample_dataframe, get_time_axis_hours, get_viz_config, find_columns_by_type
    
    top_n = kwargs.get('top_n', 8)
    maxpoints = kwargs.get('maxpoints', get_viz_config('performance.max_points_default', 2000))
    df_plot = downsample_dataframe(df, max_points=maxpoints)
    hours = get_time_axis_hours(df_plot)
    
    elec_data = find_columns_by_type(df_plot, 'Chiller', 'electrical_power_kw')
    
    if not elec_data:
        return _empty_figure("No Chiller power data found")
    
    comp_means = {cid: abs(df_plot[col].mean()) for cid, col in elec_data.items()}
    sorted_comps = sorted(elec_data.keys(), key=lambda x: -comp_means[x])
    top_n_ids = set(sorted_comps[:top_n])
    
    fig = go.Figure()
    ScatterType = get_scatter_type(len(hours))
    trace_ids = []
    
    for rank, comp_id in enumerate(sorted_comps):
        col = elec_data[comp_id]
        trace_ids.append(comp_id)
        is_visible = comp_id in top_n_ids
        
        fig.add_trace(ScatterType(
            x=hours, y=df_plot[col], mode='lines', name=comp_id.replace('_', ' '),
            stackgroup='one', legendrank=rank,
            visible=True if is_visible else 'legendonly',
            line=dict(width=0.5, color=_enhanced_color(comp_id)),
            fillcolor=_enhanced_color(comp_id),
            hovertemplate=f"<b>{comp_id.replace('_', ' ')}</b><br>%{{y:.2f}} kW<extra></extra>"
        ))
    
    n = len(trace_ids)
    buttons = [
        dict(label="Show All", method="update", args=[{"visible": [True] * n}]),
        dict(label="Hide All", method="update", args=[{"visible": ['legendonly'] * n}]),
        dict(label=f"Top {top_n}", method="update", args=[{"visible": [cid in top_n_ids for cid in trace_ids]}])
    ]
    
    fig.update_layout(
        title=kwargs.get('title', f'Chiller Electrical Power - Stacked (Top {top_n} shown)'),
        xaxis_title='Time (hours)', yaxis_title='Electrical Power (kW)',
        template='plotly_white', hovermode='x unified',
        legend=dict(groupclick="toggleitem"),
        updatemenus=[dict(type="dropdown", direction="down", x=1.0, y=1.15, xanchor="left", showactive=True, buttons=buttons)]
    )
    return fig


@log_graph_errors
def plot_cumulative_production_stacked(df: pd.DataFrame, **kwargs) -> go.Figure:
    """
    Enhanced stacked area chart of Cumulative H2 Production with UX improvements.
    
    Stacks PEM, SOEC, ATR contributions to show composition.
    """
    _check_dependencies()
    
    from h2_plant.visualization.utils import downsample_dataframe, get_time_axis_hours, get_viz_config
    
    maxpoints = kwargs.get('maxpoints', get_viz_config('performance.max_points_default', 2000))
    df_plot = downsample_dataframe(df, max_points=maxpoints)
    hours = get_time_axis_hours(df_plot)
    
    dt_orig_sec = df.attrs.get('dt_seconds', 60.0)
    dt_orig_h = dt_orig_sec / 3600.0
    if len(hours) > 1:
        dt_new_h = np.mean(np.diff(hours))
        scale = dt_new_h / dt_orig_h if dt_orig_h > 0 else 1.0
    else:
        scale = 1.0
    
    # Build source data
    sources = {}
    color_map = {'PEM': '#1f77b4', 'SOEC': '#ff7f0e', 'ATR': '#9467bd'}
    
    pem_col = next((c for c in ['H2_pem', 'H2_pem_kg'] if c in df_plot.columns), None)
    if pem_col:
        sources['PEM'] = (df_plot[pem_col] * scale).cumsum().values
    
    soec_col = next((c for c in ['H2_soec', 'H2_soec_kg'] if c in df_plot.columns), None)
    if soec_col:
        sources['SOEC'] = (df_plot[soec_col] * scale).cumsum().values
    
    atr_col = next((c for c in ['H2_atr_kg', 'H2_atr'] if c in df_plot.columns), None)
    if atr_col:
        sources['ATR'] = (df_plot[atr_col] * scale).cumsum().values
    
    if not sources:
        return _empty_figure("No H2 production data found")
    
    fig = go.Figure()
    ScatterType = get_scatter_type(len(hours))
    
    for src, vals in sources.items():
        fig.add_trace(ScatterType(
            x=hours, y=vals, mode='lines', name=src,
            stackgroup='one',
            line=dict(width=0.5, color=color_map.get(src, '#7f7f7f')),
            fillcolor=color_map.get(src, '#7f7f7f'),
            hovertemplate=f"<b>{src}</b><br>%{{y:.1f}} kg<extra></extra>"
        ))
    
    n = len(sources)
    buttons = [
        dict(label="Show All", method="update", args=[{"visible": [True] * n}]),
        dict(label="Hide All", method="update", args=[{"visible": ['legendonly'] * n}])
    ]
    
    fig.update_layout(
        title=kwargs.get('title', 'Cumulative H2 Production - Stacked'),
        xaxis_title='Time (hours)', yaxis_title='Cumulative H2 (kg)',
        template='plotly_white', hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        updatemenus=[dict(type="dropdown", direction="down", x=1.0, y=1.15, xanchor="left", showactive=True, buttons=buttons)]
    )
    return fig


@log_graph_errors
def plot_dry_cooler_stacked(df: pd.DataFrame, **kwargs) -> go.Figure:
    """
    Enhanced stacked area chart of Dry Cooler heat rejection with UX improvements.
    """
    _check_dependencies()
    
    from h2_plant.visualization.utils import downsample_dataframe, get_time_axis_hours, get_viz_config, find_columns_by_type
    
    top_n = kwargs.get('top_n', 8)
    maxpoints = kwargs.get('maxpoints', get_viz_config('performance.max_points_default', 2000))
    df_plot = downsample_dataframe(df, max_points=maxpoints)
    hours = get_time_axis_hours(df_plot)
    
    # Find DryCoolers and Intercoolers
    heat_data = find_columns_by_type(df_plot, 'DryCooler', 'heat_rejected_kw')
    ic_heat = find_columns_by_type(df_plot, 'Intercooler', 'heat_rejected_kw')
    heat_data.update(ic_heat)
    
    if not heat_data:
        return _empty_figure("No Dry Cooler/Intercooler heat rejection data found")
    
    comp_means = {cid: abs(df_plot[col].mean()) for cid, col in heat_data.items()}
    sorted_comps = sorted(heat_data.keys(), key=lambda x: -comp_means[x])
    top_n_ids = set(sorted_comps[:top_n])
    
    fig = go.Figure()
    ScatterType = get_scatter_type(len(hours))
    trace_ids = []
    
    for rank, comp_id in enumerate(sorted_comps):
        col = heat_data[comp_id]
        trace_ids.append(comp_id)
        is_visible = comp_id in top_n_ids
        
        fig.add_trace(ScatterType(
            x=hours, y=df_plot[col], mode='lines', name=comp_id.replace('_', ' '),
            stackgroup='one', legendrank=rank,
            visible=True if is_visible else 'legendonly',
            line=dict(width=0.5, color=_enhanced_color(comp_id)),
            fillcolor=_enhanced_color(comp_id),
            hovertemplate=f"<b>{comp_id.replace('_', ' ')}</b><br>%{{y:.2f}} kW<extra></extra>"
        ))
    
    n = len(trace_ids)
    buttons = [
        dict(label="Show All", method="update", args=[{"visible": [True] * n}]),
        dict(label="Hide All", method="update", args=[{"visible": ['legendonly'] * n}]),
        dict(label=f"Top {top_n}", method="update", args=[{"visible": [cid in top_n_ids for cid in trace_ids]}])
    ]
    
    fig.update_layout(
        title=kwargs.get('title', f'Dry Cooler Heat Rejection - Stacked (Top {top_n} shown)'),
        xaxis_title='Time (hours)', yaxis_title='Heat Rejected (kW)',
        template='plotly_white', hovermode='x unified',
        legend=dict(groupclick="toggleitem"),
        updatemenus=[dict(type="dropdown", direction="down", x=1.0, y=1.15, xanchor="left", showactive=True, buttons=buttons)]
    )
    return fig


# =============================================================================
# THERMAL ANALYSIS GRAPHS (Q Breakdown & Time Series)
# =============================================================================

@log_graph_errors
def plot_q_breakdown(df: pd.DataFrame, **kwargs) -> go.Figure:
    """
    Thermal Load Breakdown (Q_dot) by Component and Phase (Average kW).
    
    Row 1: Cooling Summary (Horizontal Stacked Bar: DryCooler vs Chiller Units).
    Row 2+: Detailed Breakdown (Sensible vs Latent Heat).
    """
    _check_dependencies()
    from plotly.subplots import make_subplots
    import re
    
    # 1. Identify Components and Calculate Averages
    raw_data = {}
    
    # Helper to safe mean
    def get_mean(col):
        return df[col].mean() if col in df.columns else 0.0

    # Chillers
    chiller_cols = [c for c in df.columns if '_cooling_load_kw' in c]
    for col in chiller_cols:
        cid = col.replace('_cooling_load_kw', '')
        total = get_mean(col)
        sens = get_mean(f"{cid}_sensible_heat_kw")
        lat = get_mean(f"{cid}_latent_heat_kw")
        if sens == 0 and lat == 0: sens = total
        raw_data[cid] = {'Total': total, 'Sensible': sens, 'Latent': lat, 'Type': 'Chiller'}

    # Dry Coolers / Intercoolers (Treat Intercooler as DryCooler per request)
    dc_cols = [c for c in df.columns if '_heat_rejected_kw' in c]
    for col in dc_cols:
        cid = col.replace('_heat_rejected_kw', '')
        total = get_mean(col)
        lat = get_mean(f"{cid}_latent_heat_kw")
        sens = max(0.0, total - lat)
        raw_data[cid] = {'Total': total, 'Sensible': sens, 'Latent': lat, 'Type': 'DryCooler'}
        
    # Interchangers / TQC (Keep separate, maybe generic Heat Exchanger)
    tqc_cols = [c for c in df.columns if '_tqc_duty_kw' in c]
    for col in tqc_cols:
        cid = col.replace('_tqc_duty_kw', '')
        total = get_mean(col)
        lat = get_mean(f"{cid}_latent_heat_kw")
        sens = max(0.0, total - lat)
        raw_data[cid] = {'Total': total, 'Sensible': sens, 'Latent': lat, 'Type': 'Other'}

    # Electric Boilers (case-insensitive search)
    boiler_cols = [c for c in df.columns if '_power_input_kw' in c and 'boiler' in c.lower()]
    for col in boiler_cols:
        cid = col.replace('_power_input_kw', '')
        val = get_mean(col)
        raw_data[cid] = {'Total': -val, 'Sensible': -val, 'Latent': 0.0, 'Type': 'Boiler'}

    if not raw_data:
        return _empty_figure("No thermal load data found")

    # 2. Categorize for System Summary (Row 1)
    # We want 2 Horizontal Bars: "DryCooler Unit" and "Chiller Unit"
    # Stacked by System: PEM, SOEC, ATR, STORAGE
    
    # 2. Categorize for System Summary (Row 1)
    # We want 3 Horizontal Bars: "Boiler Unit" (Top), "DryCooler Unit", "Chiller Unit"
    # Stacked by System: PEM, SOEC, ATR, STORAGE
    
    summary_data = {
        'Boiler Unit':    {'PEM': 0, 'SOEC': 0, 'ATR': 0, 'STORAGE': 0},
        'DryCooler Unit': {'PEM': 0, 'SOEC': 0, 'ATR': 0, 'STORAGE': 0},
        'Chiller Unit':   {'PEM': 0, 'SOEC': 0, 'ATR': 0, 'STORAGE': 0}
    }
    
    for cid, vals in raw_data.items():
        comp_type = vals.get('Type')
        if comp_type not in ['DryCooler', 'Chiller', 'Boiler']: continue
        
        # Identify System
        lower_id = cid.lower()
        if 'soec' in lower_id: system = 'SOEC'
        elif 'pem' in lower_id: system = 'PEM'
        elif 'atr' in lower_id or 'biogas' in lower_id: system = 'ATR'
        elif any(x in lower_id for x in ['hp', 'lp', 'storage', 'production_cooler']): system = 'STORAGE'
        else: continue # Skip 'Other' for this specific summary
        
        target_unit = f"{comp_type} Unit"
        # Boilers are stored as negative (heat input), keep as negative for Summary (Inverted Bar)
        summary_data[target_unit][system] += vals['Total']

    # 3. Categorize for Detailed Vertical Plots (Row 2+)
    # User Request: Detailed breakdown by "Path": PEM H2, PEM O2, SOEC H2, SOEC O2, ATR, STORAGE
    
    detailed_groups = {
        'PEM H2': {}, 
        'PEM O2': {}, 
        'SOEC H2': {}, 
        'SOEC O2': {}, 
        'ATR': {},
        'STORAGE': {} # Fallback / Other
    }
    
    for cid, vals in raw_data.items():
        # Do NOT skip Boilers now
        
        lower_id = cid.lower()
        
        # Categorize by Path
        if 'pem' in lower_id:
            if 'h2' in lower_id: detailed_groups['PEM H2'][cid] = vals
            elif 'o2' in lower_id: detailed_groups['PEM O2'][cid] = vals
            else: detailed_groups['PEM H2'][cid] = vals # Fallback
                
        elif 'soec' in lower_id:
            if 'h2' in lower_id: detailed_groups['SOEC H2'][cid] = vals
            elif 'o2' in lower_id: detailed_groups['SOEC O2'][cid] = vals
            else: detailed_groups['SOEC H2'][cid] = vals
            
        elif 'atr' in lower_id or 'biogas' in lower_id:
            detailed_groups['ATR'][cid] = vals
            
        elif any(x in lower_id for x in ['hp', 'lp', 'storage', 'production_cooler']):
            detailed_groups['STORAGE'][cid] = vals
            
        else:
            if 'other' not in detailed_groups: detailed_groups['Other'] = {}
            detailed_groups['Other'][cid] = vals
            
    active_detailed = {k: v for k, v in detailed_groups.items() if v}

    # 4. Create Subplots
    n_rows = 1 + len(active_detailed)
    # Row titles
    # 4. Create Subplots
    n_rows = 1 + len(active_detailed)
    # Row titles
    row_titles = ['Thermal Load Overview (Heating [-] vs Cooling [+])'] + list(active_detailed.keys())
    
    fig = make_subplots(
        rows=n_rows, cols=1,
        subplot_titles=row_titles,
        vertical_spacing=0.08,
        shared_xaxes=False
    )
    
    # --- Row 1: Horizontal Summary ---
    # We want stacked bars. 
    # Y Categories: "Boiler Unit", "DryCooler Unit", "Chiller Unit"
    # Traces: One per System.
    # Color Logic: Same color palette for all bars (User Request)
    
    systems_order = ['PEM', 'SOEC', 'ATR', 'STORAGE']
    
    # Palettes
    sys_colors = {'PEM': '#3498db', 'SOEC': '#2ecc71', 'ATR': '#e67e22', 'STORAGE': '#9b59b6'}
    
    for sys_name in systems_order:
        y_cats = ['Boiler Unit', 'DryCooler Unit', 'Chiller Unit']
        x_vals = [summary_data[u][sys_name] for u in y_cats]
        
        if abs(sum(x_vals)) < 0.1: continue
        
        fig.add_trace(go.Bar(
            y=y_cats,
            x=x_vals,
            name=f"{sys_name}",
            orientation='h',
            marker_color=sys_colors.get(sys_name, '#bdc3c7'),
            hovertemplate=f'{sys_name}<br>%{{y}}: %{{x:.1f}} kW<extra></extra>',
            legendgroup='summary',
            showlegend=True
        ), row=1, col=1)

    # --- Row 2+: Detailed Breakdown ---
    colors_detail = {'Latent': '#aec7e8', 'Sensible': '#1f77b4'}
    color_boiler = '#d62728' # Red for Boilers
    
    for idx, (group_name, components) in enumerate(active_detailed.items(), start=2):
        sorted_comps = sorted(components.keys())
        sens_vals = [components[c]['Sensible'] for c in sorted_comps]
        lat_vals = [components[c]['Latent'] for c in sorted_comps]
    for idx, (group_name, components) in enumerate(active_detailed.items(), start=2):
        sorted_comps = sorted(components.keys())
        sens_vals = [components[c]['Sensible'] for c in sorted_comps]
        lat_vals = [components[c]['Latent'] for c in sorted_comps]
        
    for idx, (group_name, components) in enumerate(active_detailed.items(), start=2):
        sorted_comps = sorted(components.keys())
        sens_vals = [components[c]['Sensible'] for c in sorted_comps]
        lat_vals = [components[c]['Latent'] for c in sorted_comps]
        
        # --- Label Generation with Collision Handling ---
        # Goal: Rename 'Intercooler' -> 'drycooler' with SPACES ('ID drycooler n'). 
        # Collision Rule: If 'PEM drycooler 1' exists (native), rename legacy 'PEM_Intercooler_1' to 'PEM drycooler 2'.
        
        final_labels_map = {}
        used_labels = set()
        
        # Pass 1: Reserve names for non-legacy components (Native Drycoolers, Chillers, etc.)
        for c in sorted_comps:
            if 'Intercooler' not in c:
                # Clean native labels: 
                # 1. Replace underscores with spaces
                # 2. Normalize "Drycooler"/"DryCooler" to "drycooler" to match user preference and avoid case-dupes
                base_clean = c.replace('_', ' ')
                base_clean = re.sub(r'(?i)drycooler', 'drycooler', base_clean) # Case-insensitive replace
                
                # Check for INTERNAL collision in native set (e.g. "Drycooler 1" vs "drycooler 1")
                candidate = base_clean
                
                match = re.search(r'(.*) drycooler (\d+)$', candidate)
                if match:
                    prefix = match.group(1)
                    num = int(match.group(2))
                    while candidate in used_labels:
                        num += 1
                        candidate = f"{prefix} drycooler {num}"
                else:
                    counter = 2
                    base = candidate
                    while candidate in used_labels:
                        candidate = f"{base} {counter}"
                        counter += 1
                
                final_labels_map[c] = candidate
                used_labels.add(candidate)

        # Pass 2: Process High Priority Renames (Intercoolers)
        for c in sorted_comps:
            if 'Intercooler' in c:
                # Proposed base name: PEM_Intercooler_1 -> PEM drycooler 1
                # Replace 'Intercooler' with 'drycooler' AND underscores with spaces
                base_text = c.replace('Intercooler', 'drycooler').replace('_', ' ')
                
                candidate = base_text
                
                # Collision Handling
                # Extract the "ID drycooler " prefix and the number suffix
                # Regex looks for: (Any Prefix) drycooler (Number)
                match = re.search(r'(.*) drycooler (\d+)$', candidate)
                
                if match:
                    prefix = match.group(1) # e.g. "PEM"
                    num = int(match.group(2))
                    
                    # Increment until free
                    while candidate in used_labels:
                        num += 1
                        candidate = f"{prefix} drycooler {num}"
                
                else:
                    # Non-numbered fallback
                    counter = 2
                    base = candidate
                    while candidate in used_labels:
                        candidate = f"{base} {counter}"
                        counter += 1

                final_labels_map[c] = candidate
                used_labels.add(candidate)
        
        labels = [final_labels_map[c] for c in sorted_comps]
        
        # --- Value Coloring ---
        trace_sens_colors = []
        trace_lat_colors = []
        
        for c in sorted_comps:
            if components[c].get('Type') == 'Boiler':
                trace_sens_colors.append(color_boiler)
                trace_lat_colors.append(color_boiler)
            else:
                trace_sens_colors.append(colors_detail['Sensible'])
                trace_lat_colors.append(colors_detail['Latent'])

        # Latent
        fig.add_trace(go.Bar(
            x=labels, y=lat_vals, name='Latent',
            marker_color=trace_lat_colors,
            showlegend=(idx==2), legendgroup='detail',
            hovertemplate='%{x}<br>Latent: %{y:.1f} kW<extra></extra>'
        ), row=idx, col=1)
        
        # Sensible
        fig.add_trace(go.Bar(
            x=labels, y=sens_vals, name='Sensible',
            marker_color=trace_sens_colors,
            showlegend=(idx==2), legendgroup='detail',
            hovertemplate='%{x}<br>Sensible: %{y:.1f} kW<extra></extra>'
        ), row=idx, col=1)

    fig.update_layout(
        title=kwargs.get('title', 'Thermal Load Breakdown'),
        barmode='stack',
        height=400 + (300 * len(active_detailed)),
        template='plotly_white',
        legend=dict(groupclick="toggleitem")
    )
    
    # Update axes
    fig.update_xaxes(title_text="Total Load (kW)", row=1, col=1)
    for i in range(2, n_rows + 1):
        fig.update_yaxes(title_text="Load (kW)", row=i, col=1)

    return fig


@log_graph_errors
def plot_thermal_load_breakdown_time_series(df: pd.DataFrame, **kwargs) -> go.Figure:
    """
    Time-series stacked area chart of thermal loads.
    
    Visualizes total cooling demand over time, broken down by component
    (Chillers, Dry Coolers, Intercoolers).
    """
    _check_dependencies()
    
    from h2_plant.visualization.utils import downsample_dataframe, get_time_axis_hours, get_viz_config
    
    maxpoints = kwargs.get('maxpoints', get_viz_config('performance.max_points_default', 2000))
    df_plot = downsample_dataframe(df, max_points=maxpoints)
    hours = get_time_axis_hours(df_plot)
    
    # Collect Data Columns
    plot_data = {}
    
    # 1. Chillers (cooling_load_kw)
    chiller_cols = [c for c in df_plot.columns if '_cooling_load_kw' in c]
    for col in chiller_cols:
        name = col.replace('_cooling_load_kw', '').replace('_', ' ') + ' (Chiller)'
        plot_data[name] = df_plot[col]
        
    # 2. Dry Coolers / TQC (heat_rejected or tqc_duty)
    dc_cols = [c for c in df_plot.columns if '_heat_rejected_kw' in c or '_tqc_duty_kw' in c]
    for col in dc_cols:
        clean_col = col.replace('_heat_rejected_kw', '').replace('_tqc_duty_kw', '').replace('_', ' ')
        name = f"{clean_col} (DryCooler)"
        if name not in plot_data:
             plot_data[name] = df_plot[col]
             
    if not plot_data:
        return _empty_figure("No thermal load time-series data found")
        
    # Create Figure
    fig = go.Figure()
    
    # Use WebGL for performance if needed
    ScatterType = get_scatter_type(len(hours))
    
    # Add traces (Stacked Area)
    for name in sorted(plot_data.keys()):
        series = plot_data[name]
        if series.mean() > 0.1:
            fig.add_trace(ScatterType(
                x=hours,
                y=series,
                mode='lines',
                name=name,
                stackgroup='one',
                line=dict(width=0.5),
                hovertemplate='%{y:.1f} kW'
            ))
            
    fig.update_layout(
        title=kwargs.get('title', 'Thermal Load Profile (Cooling Demand)'),
        xaxis_title='Time (hours)',
        yaxis_title='Cooling Load (kW)',
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig


@log_graph_errors
def plot_thermal_load_breakdown_timeseries(df: pd.DataFrame, **kwargs) -> go.Figure:
    """
    Plots thermal load (kW) over time with enhanced UX features.
    
    Features:
    - Component-level colors (hash-based, stable across runs)
    - Hierarchical legend with group headers
    - Interactive dropdown: Show All/Hide All/Top N/Per-Subsystem
    - Lines/Stacked style toggle (compatible with visibility filters)
    - Default: Only top 8 contributors visible
    - Sorted by mean load within each subsystem
    - WebGL rendering for large datasets
    """
    _check_dependencies()
    
    # 1. Configuration and Data Prep
    max_points = kwargs.get('max_points', 10000)
    top_n = kwargs.get('top_n', 8)  # Number of traces visible by default
    df_plot = utils.downsample_dataframe(df, max_points=max_points)
    
    if df_plot.empty:
        return _empty_figure("No DataFrame provided")
        
    hours = utils.get_time_axis_hours(df_plot)
    
    # 2. Identify Thermal Load Columns & Calculate Mean Loads
    suffixes = ['cooling_load_kw', 'heat_rejected_kw', 'heat_removed_kw', 
                'tqc_duty_kw', 'dc_duty_kw', 'duty_kw', 'q_transferred_kw',
                'power_input_kw']  # Include boilers
    
    component_data = {}  # {comp_id: {'col': col_name, 'mean': mean_val}}
    
    for col in df_plot.columns:
        for suffix in suffixes:
            if suffix in col:
                comp_id = col.replace(f"_{suffix}", "").replace(suffix, "").strip("_")
                if comp_id:
                    mean_val = abs(df_plot[col].mean())  # Use absolute for ranking
                    component_data[comp_id] = {'col': col, 'mean': mean_val}
                break
    
    if not component_data:
        return _empty_figure("No thermal load data found")

    # 3. Classification & Sorting
    def classify_subsystem(name):
        u_name = name.upper()
        if "PEM" in u_name: return "PEM"
        if "SOEC" in u_name: return "SOEC"
        if "ATR" in u_name: return "ATR"
        if any(x in u_name for x in ["STORE", "STORAGE", "TANK", "COMPRESSOR", "HP_", "LP_"]): 
            return "Storage"
        return "Balance of Plant"
    
    # Sort by: (subsystem, -mean_load, name) so largest load first within subsystem
    sorted_comps = sorted(
        component_data.keys(), 
        key=lambda x: (classify_subsystem(x), -component_data[x]['mean'], x)
    )
    
    # 4. Component-Level Color Assignment (hash-based, stable)
    # Use Plotly's qualitative palette (24 distinct colors)
    QUALITATIVE_PALETTE = [
        '#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
        '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52',
        '#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD',
        '#8C564B', '#E377C2', '#7F7F7F', '#BCBD22', '#17BECF',
        '#AEC7E8', '#FFBB78', '#98DF8A', '#C49C94'
    ]
    
    def component_color(comp_id: str) -> str:
        """Deterministic color from component ID hash."""
        h = hash(comp_id) % len(QUALITATIVE_PALETTE)
        return QUALITATIVE_PALETTE[h]
    
    # 5. Determine Default Visibility (Top N by mean load)
    all_means = [(cid, component_data[cid]['mean']) for cid in sorted_comps]
    all_means_sorted = sorted(all_means, key=lambda x: -x[1])  # Descending by mean
    top_n_ids = set([cid for cid, _ in all_means_sorted[:top_n]])
    
    # 6. Build Figure
    fig = go.Figure()
    ScatterType = get_scatter_type(len(hours))
    
    trace_categories = []
    trace_ids = []  # For button logic
    
    for rank, comp_id in enumerate(sorted_comps):
        col_name = component_data[comp_id]['col']
        category = classify_subsystem(comp_id)
        trace_categories.append(category)
        trace_ids.append(comp_id)
        
        display_name = comp_id.replace('_', ' ')
        is_visible = comp_id in top_n_ids
        
        # FIX: Add stackgroup=None for toggle compatibility
        fig.add_trace(ScatterType(
            x=hours,
            y=df_plot[col_name],
            mode='lines',
            name=display_name,
            legendgroup=category,
            legendgrouptitle_text=category,
            legendrank=rank,  # Enforce ordering
            visible=True if is_visible else 'legendonly',
            line=dict(width=1.5, color=component_color(comp_id)),
            stackgroup=None,  # NEW: Required for toggle to work
            hovertemplate=f"<b>{display_name}</b><br>Load: %{{y:.2f}} kW<extra></extra>"
        ))

    # 7. Interactive Buttons (Dropdown with enhanced options)
    n_traces = len(trace_categories)
    all_trace_indices = list(range(n_traces))
    
    # Visibility filter buttons
    vis_buttons = [
        dict(
            label="Show All",
            method="update",
            args=[{"visible": [True] * n_traces}]
        ),
        dict(
            label="Hide All",
            method="update",
            args=[{"visible": ['legendonly'] * n_traces}]
        ),
        dict(
            label=f"Top {top_n}",
            method="update",
            args=[{"visible": [cid in top_n_ids for cid in trace_ids]}]
        )
    ]
    
    # Per-subsystem buttons
    present_categories = sorted(list(set(trace_categories)))
    for cat in present_categories:
        visibility = [t_cat == cat for t_cat in trace_categories]
        vis_buttons.append(dict(
            label=f"{cat} Only",
            method="update",
            args=[{"visible": visibility}]
        ))
    
    # FIX: Add Lines/Stacked style toggle (new dropdown)
    # Build restyle arrays for all traces
    stack_lines = ['' for _ in range(n_traces)]
    fill_lines = ['none' for _ in range(n_traces)]
    width_lines = [1.5 for _ in range(n_traces)]
    
    stack_stacked = ['thermal_load' for _ in range(n_traces)]
    fill_stacked = ['tonexty' for _ in range(n_traces)]
    width_stacked = [0.5 for _ in range(n_traces)]
    
    style_buttons = [
        dict(label="Lines", method="restyle",
             args=[{"stackgroup": stack_lines, "fill": fill_lines, 
                    "line.width": width_lines}, all_trace_indices]),
        dict(label="Stacked", method="restyle",
             args=[{"stackgroup": stack_stacked, "fill": fill_stacked,
                    "line.width": width_stacked}, all_trace_indices]),
    ]

    # 8. Layout with TWO independent dropdown menus
    fig.update_layout(
        title=kwargs.get('title', f"Thermal Load Time Series (Top {top_n} shown)"),
        xaxis_title="Time (hours)",
        yaxis_title="Thermal Load (kW)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            groupclick="toggleitem",
            tracegroupgap=10
        ),
        updatemenus=[
            # Menu 1: Visibility filters (existing)
            dict(
                type="dropdown", 
                direction="down",
                x=1.0, y=1.15,
                xanchor="left",
                showactive=True, 
                buttons=vis_buttons
            ),
            # Menu 2: Style toggle (new - Lines/Stacked)
            dict(
                type="dropdown",
                direction="down",
                x=0.82, y=1.15,
                xanchor="left",
                showactive=True,
                active=0,  # Default to Lines
                buttons=style_buttons
            )
        ]
    )
    
    return fig


@log_graph_errors
def plot_thermal_load_stacked_timeseries(df: pd.DataFrame, **kwargs) -> go.Figure:
    """
    Stacked area chart of thermal loads (kW) over time.
    
    Same features as plot_thermal_load_breakdown_timeseries but uses stacked areas
    to visualize total demand composition.
    
    Features:
    - Component-level colors (hash-based, stable across runs)
    - Hierarchical legend with group headers
    - Interactive dropdown: Show All/Hide All/Top N/Per-Subsystem
    - Default: Only top 8 contributors visible
    - Sorted by mean load within each subsystem
    - Stacked area visualization
    """
    _check_dependencies()
    
    # 1. Configuration and Data Prep
    max_points = kwargs.get('max_points', 10000)
    top_n = kwargs.get('top_n', 8)
    df_plot = utils.downsample_dataframe(df, max_points=max_points)
    
    if df_plot.empty:
        return _empty_figure("No DataFrame provided")
        
    hours = utils.get_time_axis_hours(df_plot)
    
    # 2. Identify Thermal Load Columns & Calculate Mean Loads
    suffixes = ['cooling_load_kw', 'heat_rejected_kw', 'heat_removed_kw', 
                'tqc_duty_kw', 'dc_duty_kw', 'duty_kw', 'q_transferred_kw',
                'power_input_kw']
    
    component_data = {}
    
    for col in df_plot.columns:
        for suffix in suffixes:
            if suffix in col:
                comp_id = col.replace(f"_{suffix}", "").replace(suffix, "").strip("_")
                if comp_id:
                    mean_val = abs(df_plot[col].mean())
                    component_data[comp_id] = {'col': col, 'mean': mean_val}
                break
    
    if not component_data:
        return _empty_figure("No thermal load data found")

    # 3. Classification & Sorting
    def classify_subsystem(name):
        u_name = name.upper()
        if "PEM" in u_name: return "PEM"
        if "SOEC" in u_name: return "SOEC"
        if "ATR" in u_name: return "ATR"
        if any(x in u_name for x in ["STORE", "STORAGE", "TANK", "COMPRESSOR", "HP_", "LP_"]): 
            return "Storage"
        return "Balance of Plant"
    
    # Sort by: (subsystem, -mean_load, name) so largest load first
    sorted_comps = sorted(
        component_data.keys(), 
        key=lambda x: (classify_subsystem(x), -component_data[x]['mean'], x)
    )
    
    # 4. Component-Level Color Assignment (hash-based, stable)
    QUALITATIVE_PALETTE = [
        '#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
        '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52',
        '#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD',
        '#8C564B', '#E377C2', '#7F7F7F', '#BCBD22', '#17BECF',
        '#AEC7E8', '#FFBB78', '#98DF8A', '#C49C94'
    ]
    
    def component_color(comp_id: str) -> str:
        h = hash(comp_id) % len(QUALITATIVE_PALETTE)
        return QUALITATIVE_PALETTE[h]
    
    # 5. Determine Default Visibility (Top N by mean load)
    all_means = [(cid, component_data[cid]['mean']) for cid in sorted_comps]
    all_means_sorted = sorted(all_means, key=lambda x: -x[1])
    top_n_ids = set([cid for cid, _ in all_means_sorted[:top_n]])
    
    # 6. Build Figure with Stacked Areas
    fig = go.Figure()
    ScatterType = get_scatter_type(len(hours))
    
    trace_categories = []
    trace_ids = []
    
    for rank, comp_id in enumerate(sorted_comps):
        col_name = component_data[comp_id]['col']
        category = classify_subsystem(comp_id)
        trace_categories.append(category)
        trace_ids.append(comp_id)
        
        display_name = comp_id.replace('_', ' ')
        is_visible = comp_id in top_n_ids
        color = component_color(comp_id)
        
        # Use absolute values for stacking (boilers have negative values)
        y_vals = np.abs(df_plot[col_name].values)
        
        fig.add_trace(ScatterType(
            x=hours,
            y=y_vals,
            mode='lines',
            name=display_name,
            legendgroup=category,
            legendgrouptitle_text=category,
            legendrank=rank,
            visible=True if is_visible else 'legendonly',
            stackgroup='one',  # Enable stacking
            line=dict(width=0.5, color=color),
            fillcolor=color,
            hovertemplate=f"<b>{display_name}</b><br>Load: %{{y:.2f}} kW<extra></extra>"
        ))

    # 7. Interactive Buttons
    n_traces = len(trace_categories)
    
    buttons = [
        dict(label="Show All", method="update", args=[{"visible": [True] * n_traces}]),
        dict(label="Hide All", method="update", args=[{"visible": ['legendonly'] * n_traces}]),
        dict(label=f"Top {top_n}", method="update", args=[{"visible": [cid in top_n_ids for cid in trace_ids]}])
    ]
    
    present_categories = sorted(list(set(trace_categories)))
    for cat in present_categories:
        visibility = [t_cat == cat for t_cat in trace_categories]
        buttons.append(dict(label=f"{cat} Only", method="update", args=[{"visible": visibility}]))

    # 8. Layout
    fig.update_layout(
        title=kwargs.get('title', f"Stacked Thermal Load Time Series (Top {top_n} shown)"),
        xaxis_title="Time (hours)",
        yaxis_title="Thermal Load (kW)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(groupclick="toggleitem", tracegroupgap=10),
        updatemenus=[
            dict(
                type="dropdown", direction="down",
                x=1.0, y=1.15, xanchor="left",
                showactive=True, buttons=buttons
            )
        ]
    )
    
    return fig


# =============================================================================
# SEPARATION EQUIPMENT GRAPHS (Coalescer, KOD, Mixer)
# =============================================================================

@log_graph_errors
def plot_coalescer_separation(df: pd.DataFrame, **kwargs) -> go.Figure:
    """
    Coalescer Performance: Pressure Drop and Liquid Drain rate.
    
    Features:
    - Component-level colors (hash-based)
    - Multi-panel layout (Pressure Drop, Drain Flow)
    - Show All/Hide All buttons
    - Lines/Stacked style toggle
    """
    _check_dependencies()
    from h2_plant.visualization.utils import downsample_dataframe, get_time_axis_hours, get_viz_config, find_columns_by_type
    from plotly.subplots import make_subplots
    
    maxpoints = kwargs.get('maxpoints', get_viz_config('performance.max_points_default', 2000))
    df_plot = downsample_dataframe(df, max_points=maxpoints)
    hours = get_time_axis_hours(df_plot)
    
    # Find Data
    delta_p_data = find_columns_by_type(df_plot, 'Coalescer', 'delta_p_bar')
    drain_data = find_columns_by_type(df_plot, 'Coalescer', 'drain_flow_kg_h')
    
    if not delta_p_data and not drain_data:
        return _empty_figure("No Coalescer data found")
        
    # Layout: 2 Rows
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Pressure Drop", "Liquid Removal")
    )
    
    ScatterType = get_scatter_type(len(hours))
    trace_ids = []
    dp_trace_idxs = []
    drain_trace_idxs = []
    trace_counter = 0  # NEW: Explicit trace counter
    
    # Panel 1: Pressure Drop (default: lines, no stack)
    for comp_id, col in delta_p_data.items():
        trace_ids.append(comp_id)
        fig.add_trace(ScatterType(
            x=hours, y=df_plot[col],
            mode='lines', name=f"{comp_id} dP",
            legendgroup=comp_id,
            line=dict(color=_enhanced_color(comp_id), width=1.5),
            hovertemplate=f"<b>{comp_id}</b><br>dP: %{{y:.4f}} bar<extra></extra>"
        ), row=1, col=1)
        dp_trace_idxs.append(trace_counter)
        trace_counter += 1
        
    # Panel 2: Drain Flow (default: lines, no stack)
    for comp_id, col in drain_data.items():
        if comp_id not in trace_ids:
            trace_ids.append(comp_id)
        fig.add_trace(ScatterType(
            x=hours, y=df_plot[col],
            mode='lines', name=f"{comp_id} Flow",
            legendgroup=comp_id, showlegend=False,
            line=dict(color=_enhanced_color(comp_id), width=1.5),
            hovertemplate=f"<b>{comp_id}</b><br>Flow: %{{y:.2f}} kg/h<extra></extra>"
        ), row=2, col=1)
        drain_trace_idxs.append(trace_counter)
        trace_counter += 1
    
    # Build style toggle arrays
    n = trace_counter
    all_idxs = list(range(n))
    
    # Lines mode: no stacking, no fill
    stack_lines = [None] * n
    fill_lines = [None] * n
    width_lines = [1.5] * n
    
    # Stacked mode: separate stackgroups per panel
    stack_stacked = [None] * n
    fill_stacked = [None] * n
    width_stacked = [0.5] * n
    
    for i in dp_trace_idxs:
        stack_stacked[i] = "coalescer_dp"
        fill_stacked[i] = "tonexty"
    
    for i in drain_trace_idxs:
        stack_stacked[i] = "coalescer_drain"
        fill_stacked[i] = "tonexty"
    
    # Visibility buttons
    vis_buttons = [
        dict(label="Show All", method="update", args=[{"visible": [True] * n}]),
        dict(label="Hide All", method="update", args=[{"visible": ['legendonly'] * n}])
    ]
    
    # Style toggle buttons
    style_buttons = [
        dict(
            label="Lines",
            method="restyle",
            args=[{"stackgroup": stack_lines, "fill": fill_lines, "line.width": width_lines}, all_idxs]
        ),
        dict(
            label="Stacked",
            method="restyle",
            args=[{"stackgroup": stack_stacked, "fill": fill_stacked, "line.width": width_stacked}, all_idxs]
        )
    ]
        
    fig.update_layout(
        title=kwargs.get('title', 'Coalescer Separation Performance'),
        template='plotly_white',
        hovermode='x unified',
        height=700,
        updatemenus=[
            dict(type="dropdown", direction="down", x=1.0, y=1.15, xanchor="left", showactive=True, buttons=vis_buttons),
            dict(type="dropdown", direction="down", x=0.82, y=1.15, xanchor="left", showactive=True, buttons=style_buttons)
        ]
    )
    
    fig.update_yaxes(title_text="Delta P (bar)", row=1, col=1)
    fig.update_yaxes(title_text="Drain (kg/h)", row=2, col=1)
    fig.update_xaxes(title_text="Time (hours)", row=2, col=1)
    
    return fig


@log_graph_errors
def plot_kod_separation(df: pd.DataFrame, **kwargs) -> go.Figure:
    """
    Knock-Out Drum (KOD) Water Removal Performance.
    
    Features:
    - Single panel showing water removal rate
    - Subsystem coloring (PEM/SOEC/ATR)
    - Lines/Stacked toggle
    - Show All/Hide All buttons
    """
    _check_dependencies()
    from h2_plant.visualization.utils import downsample_dataframe, get_time_axis_hours, get_viz_config
    
    maxpoints = kwargs.get('maxpoints', get_viz_config('performance.max_points_default', 2000))
    df_plot = downsample_dataframe(df, max_points=maxpoints)
    hours = get_time_axis_hours(df_plot)
    
    # Search for water removal columns from KOD components
    water_data = {}
    for col in df_plot.columns:
        col_lower = col.lower()
        # Must contain 'kod' and 'water_removed'
        if 'kod' in col_lower and 'water_removed' in col_lower:
            # Exclude vapor/steam/dissolved
            if not any(ex in col_lower for ex in ['vapor', 'steam', 'dissolved']):
                # Extract component ID by removing the metric suffix
                if '_water_removed_kg_h' in col:
                    comp_id = col.replace('_water_removed_kg_h', '')
                else:
                    comp_id = col.rsplit('_', 2)[0]  # Fallback
                water_data[comp_id] = col
    
    # Debug: If no data found, return helpful message
    if not water_data:
        kod_cols = [c for c in df_plot.columns if 'kod' in c.lower()]
        if kod_cols:
            msg = f"No KOD water removal data found. Available KOD columns: {kod_cols[:10]}"
        else:
            all_cols = list(df_plot.columns)[:20]
            msg = f"No KOD data found. Available columns: {all_cols}..."
        return _empty_figure(msg)
    
    ScatterType = get_scatter_type(len(hours))
    fig = go.Figure()
    trace_counter = 0
    
    # Add water removal traces
    for comp_id, col in water_data.items():
        # Handle potential NaN values
        y_data = np.nan_to_num(df_plot[col].values, nan=0.0)
        
        fig.add_trace(ScatterType(
            x=hours, y=y_data,
            mode='lines',
            stackgroup=None,  # Default to Lines for toggle
            fill=None,
            name=comp_id.replace('_', ' '),
            line=dict(color=_get_subsystem_color(comp_id), width=1.5),
            hovertemplate=f"<b>{comp_id.replace('_', ' ')}</b><br>Water: %{{y:.2f}} kg/h<extra></extra>"
        ))
        trace_counter += 1
    
    n = trace_counter
    all_idxs = list(range(n))
    
    # Visibility buttons
    vis_buttons = [
        dict(label="Show All", method="update", args=[{"visible": [True] * n}]),
        dict(label="Hide All", method="update", args=[{"visible": ['legendonly'] * n}])
    ]
    
    # Lines/Stacked toggle
    style_buttons = [
        dict(label="Lines", method="restyle",
             args=[{"stackgroup": ['' for _ in all_idxs], "fill": ['none' for _ in all_idxs], "line.width": [1.5 for _ in all_idxs]}, all_idxs]),
        dict(label="Stacked", method="restyle",
             args=[{"stackgroup": ['kod_water' for _ in all_idxs], "fill": ['tonexty' for _ in all_idxs], "line.width": [0.5 for _ in all_idxs]}, all_idxs]),
    ]
    
    fig.update_layout(
        title=kwargs.get('title', 'Knock-Out Drum (KOD) Water Removal'),
        template='plotly_white',
        hovermode='x unified',
        height=500,
        xaxis_title="Time (hours)",
        yaxis_title="Water Removal Rate (kg/h)",
        updatemenus=[
            dict(type="dropdown", direction="down", x=1.0, y=1.15, xanchor="left", showactive=True, buttons=vis_buttons),
            dict(type="dropdown", direction="down", x=0.82, y=1.15, xanchor="left", showactive=True, active=0, buttons=style_buttons)
        ]
    )
    
    return fig


@log_graph_errors
def plot_mixer_comparison(df: pd.DataFrame, **kwargs) -> go.Figure:
    """
    Drain/Mixer Comparison: Temperature, Pressure, Mass Flow.
    
    Features:
    - 3-panel layout with explicit Lines/Stacked toggle for Row 3 (Flow)
    - Subsystem coloring for flow traces
    """
    _check_dependencies()
    from h2_plant.visualization.utils import downsample_dataframe, get_time_axis_hours, get_viz_config, find_columns_by_type
    from plotly.subplots import make_subplots
    
    maxpoints = kwargs.get('maxpoints', get_viz_config('performance.max_points_default', 2000))
    df_plot = downsample_dataframe(df, max_points=maxpoints)
    hours = get_time_axis_hours(df_plot)
    
    # Search for mixer data
    def search_mixers(metric_suffixes):
        found = {}
        types = ['Mixer', 'DrainRecorder', 'Drain_Mixer', 'WaterMixer', 'Combiner']
        for comp_type in types:
            for suffix in metric_suffixes:
                cols = find_columns_by_type(df_plot, comp_type, suffix)
                for comp_id, col_name in cols.items():
                    if comp_type == 'Mixer':
                        if 'Drain' in comp_id or 'Combiner' in comp_id:
                            found[comp_id] = col_name
                    else:
                        found[comp_id] = col_name
        return found

    t_data = search_mixers(['temperature_k', 'outlet_temp_k', 'outlet_temperature_c', 'temp_c'])
    p_data = search_mixers(['pressure_pa', 'outlet_pressure_kpa', 'outlet_pressure_bar', 'pressure_bar'])
    m_data = search_mixers(['outlet_mass_kg_h', 'outlet_mass_flow_kg_h'])
    
    if not t_data and not m_data:
        return _empty_figure("No Mixer/Drain data found")

    # Layout: 3 Rows
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Outlet Temperature", "Operating Pressure", "Mass Flow Rate")
    )
    
    ScatterType = get_scatter_type(len(hours))
    trace_ids = []
    flow_trace_idxs = []
    trace_counter = 0  # Explicit counter for robustness
    
    # Panel 1: Temperature (K->deg C)
    for comp_id, col in t_data.items():
        trace_ids.append(comp_id)
        data = df_plot[col].values
        if np.nanmean(data) > 200:  # Kelvin
            data = data - 273.15
            
        fig.add_trace(ScatterType(
            x=hours, y=data,
            mode='lines', name=comp_id.replace('_', ' '),
            legendgroup=comp_id,
            line=dict(color=_get_subsystem_color(comp_id), width=1.5),
            hovertemplate=f"<b>{comp_id.replace('_', ' ')}</b><br>T: %{{y:.1f}} deg C<extra></extra>"
        ), row=1, col=1)
        trace_counter += 1

    # Panel 2: Pressure (Pa/kPa->bar)
    for comp_id, col in p_data.items():
        if comp_id not in trace_ids:
            trace_ids.append(comp_id)
        data = df_plot[col].values
        mean_val = np.nanmean(data)
        if mean_val > 50000:  # Pa
            data = data / 1e5
        elif mean_val > 50 and 'kpa' in col.lower():  # kPa
            data = data / 100.0
            
        fig.add_trace(ScatterType(
            x=hours, y=data,
            mode='lines', name=comp_id.replace('_', ' '),
            legendgroup=comp_id, showlegend=False,
            line=dict(color=_get_subsystem_color(comp_id), width=1.5),
            hovertemplate=f"<b>{comp_id.replace('_', ' ')}</b><br>P: %{{y:.2f}} bar<extra></extra>"
        ), row=2, col=1)
        trace_counter += 1

    # Panel 3: Mass Flow (default: Lines, with explicit stackgroup for toggle)
    # FIX: Sort by ascending mean flow for correct stacking (smaller at bottom)
    m_data_sorted = sorted(
        m_data.items(),
        key=lambda x: np.nanmean(df_plot[x[1]].values)
    )
    
    for comp_id, col in m_data_sorted:
        if comp_id not in trace_ids:
            trace_ids.append(comp_id)
        
        # Handle potential NaN values
        y_data = np.nan_to_num(df_plot[col].values, nan=0.0)
        
        fig.add_trace(ScatterType(
            x=hours, y=y_data,
            mode='lines',
            name=comp_id.replace('_', ' '),
            legendgroup=comp_id, showlegend=False,
            line=dict(color=_get_subsystem_color(comp_id), width=1.5),
            stackgroup=None,  # Default to Lines - this MUST be set for toggle to work
            fill=None,  # Explicit fill=None for lines mode
            hovertemplate=f"<b>{comp_id.replace('_', ' ')}</b><br>Flow: %{{y:.1f}} kg/h<extra></extra>"
        ), row=3, col=1)
        flow_trace_idxs.append(trace_counter)
        trace_counter += 1

    # Total trace count
    n = trace_counter
    
    # Visibility buttons
    vis_buttons = [
        dict(label="Show All", method="update", args=[{"visible": [True] * n}]),
        dict(label="Hide All", method="update", args=[{"visible": ['legendonly'] * n}])
    ]
    
    # FIX: Explicit Lines/Stacked toggle for Row 3 ONLY (not using utility function)
    # Build restyle arrays targeting ONLY flow traces
    # For "Lines": clear stackgroup and fill for flow traces
    # For "Stacked": set stackgroup and fill for flow traces
    
    # Arrays for ALL traces (None = no change)
    stack_lines = [None] * n
    fill_lines = [None] * n
    width_lines = [None] * n
    
    stack_stacked = [None] * n
    fill_stacked = [None] * n
    width_stacked = [None] * n
    
    # Configure ONLY the flow traces (Row 3)
    for idx in flow_trace_idxs:
        # Lines mode: explicitly clear
        stack_lines[idx] = ''  # Empty string clears stackgroup
        fill_lines[idx] = 'none'  # 'none' clears fill
        width_lines[idx] = 1.5
        
        # Stacked mode: set stackgroup and fill
        stack_stacked[idx] = 'mixer_flow'
        fill_stacked[idx] = 'tonexty'
        width_stacked[idx] = 0.5
    
    style_buttons = [
        dict(
            label="Lines", 
            method="restyle",
            args=[{
                "stackgroup": stack_lines, 
                "fill": fill_lines, 
                "line.width": width_lines
            }, list(range(n))]  # Target ALL traces (non-flow get None = no change)
        ),
        dict(
            label="Stacked (Flow)", 
            method="restyle",
            args=[{
                "stackgroup": stack_stacked, 
                "fill": fill_stacked,
                "line.width": width_stacked
            }, list(range(n))]  # Target ALL traces (non-flow get None = no change)
        ),
    ]

    fig.update_layout(
        title=kwargs.get('title', 'Drain/Mixer Comparison'),
        template='plotly_white',
        hovermode='x unified',
        height=900,
        updatemenus=[
            dict(type="dropdown", direction="down", x=1.0, y=1.15, xanchor="left", showactive=True, buttons=vis_buttons),
            dict(type="dropdown", direction="down", x=0.82, y=1.15, xanchor="left", showactive=True, active=0, buttons=style_buttons)
        ]
    )
    
    fig.update_yaxes(title_text="Temp (deg C)", row=1, col=1)
    fig.update_yaxes(title_text="Pressure (bar)", row=2, col=1)
    fig.update_yaxes(title_text="Flow (kg/h)", row=3, col=1)
    fig.update_xaxes(title_text="Time (hours)", row=3, col=1)
    
    return fig


def _empty_figure(text: str) -> go.Figure:
    """Helper to create an empty figure with a message."""
    fig = go.Figure()
    fig.add_annotation(
        text=text,
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False
    )
    return fig


# =============================================================================
# PROCESS TRAIN PROFILE (Interactive Multi-Panel)
# =============================================================================

# Train component definitions (ordered by process step)
_TRAIN_COMPONENTS = {
    'H2_SOEC': [
        'SOEC_Cluster',
        'SOEC_H2_Interchanger_1',
        'SOEC_H2_DryCooler_1',
        'SOEC_H2_KOD_1',
        'SOEC_H2_Chiller_1',
        'SOEC_H2_KOD_2',
        'SOEC_H2_Cyclone_1',
        'SOEC_H2_Compressor_S1',
        'SOEC_H2_Intercooler_1',
        'SOEC_H2_Compressor_S2',
        'SOEC_H2_Intercooler_2',
        'SOEC_H2_Cyclone_2',
        'SOEC_H2_Compressor_S3',
        'SOEC_H2_Intercooler_3',
        'SOEC_H2_Cyclone_3',
        'SOEC_H2_Compressor_S4',
        'SOEC_H2_Intercooler_4',
        'SOEC_H2_Cyclone_4',
        'SOEC_H2_Compressor_S5',
        'SOEC_H2_Intercooler_5',
        'SOEC_H2_Cyclone_5',
        'SOEC_H2_Compressor_S6',
        'SOEC_H2_Intercooler_6',
        'SOEC_H2_Deoxo_1',
        'SOEC_H2_Chiller_2',
        'SOEC_H2_Coalescer_2',
        'SOEC_H2_ElectricBoiler_PSA',
        'SOEC_H2_PSA_1',
    ],
    'O2_SOEC': [
        'SOEC_O2_Interchanger_1',
        'SOEC_O2_Drycooler_1',
        'SOEC_O2_compressor_1',
        'SOEC_O2_Drycooler_2',
        'SOEC_O2_compressor_2',
        'SOEC_O2_Drycooler_3',
        'SOEC_O2_compressor_3',
        'SOEC_O2_Drycooler_4',
        'SOEC_O2_compressor_4',
    ],
    'H2_PEM': [
        'PEM_Electrolyzer',
        'PEM_H2_KOD_1',
        'PEM_H2_DryCooler_1',
        'PEM_H2_Chiller_1',
        'PEM_H2_KOD_2',
        'PEM_H2_Coalescer_1',
        'PEM_H2_ElectricBoiler_1',
        'PEM_H2_Deoxo_1',
        'PEM_H2_Chiller_2',
        'PEM_H2_KOD_3',
        'PEM_H2_ElectricBoiler_2',
        'PEM_H2_PSA_1',
    ],
    'O2_PEM': [
        'PEM_Electrolyzer',
        'PEM_O2_KOD_1',
        'PEM_O2_Drycooler_1',
        'PEM_O2_Chiller_1',
        'PEM_O2_KOD_2',
        'PEM_O2_Coalescer_1',
    ],
}


@log_graph_errors
def plot_process_train_profile(df: pd.DataFrame, **kwargs) -> go.Figure:
    """
    Process Train Profile: Temperature, Pressure, and Composition.
    
    Interactive 3-panel graph showing time-averaged properties along each
    process train. Includes train selector dropdown for H2/O2 SOEC/PEM trains.
    
    Features:
    - 3 panels: Temperature (deg C), Pressure (bar), Composition (kg/h + impurity %)
    - Train selector dropdown (7 options)
    - Subsystem-based coloring
    - Hover templates with component details
    """
    _check_dependencies()
    from plotly.subplots import make_subplots
    
    # Extract profile data for all trains
    def extract_profile(components: list, stream_type: str = 'H2') -> dict:
        """Extract time-averaged T, P, Flow, Impurity for components."""
        data = {'components': [], 'temp': [], 'press': [], 'flow': [], 'impurity': [], 'h2o_ppm': []}
        
        # Molecular weights for mass-to-mole conversion (kg/kmol)
        MW_H2O = 18.015
        MW_H2 = 2.016
        MW_O2 = 32.0
        
        for cid in components:
            # 1. IDENTIFY COLUMN NAMES FIRST
            flow_col = next((c for c in df.columns if c in [
                f'{cid}_outlet_mass_flow_kg_h', f'{cid}_mass_flow_kg_h'
            ]), None)
            
            temp_col = next((c for c in df.columns if c in [
                f'{cid}_outlet_temp_c', f'{cid}_temperature_c', f'{cid}_temp_c', f'{cid}_T_c'
            ]), None)
            
            # Pressure
            press_col = next((c for c in df.columns if c in [
                f'{cid}_outlet_pressure_bar', f'{cid}_pressure_bar', f'{cid}_P_bar'
            ]), None)
            
            # 2. FILTER FOR ACTIVE FLOW (Issue 3: Representative Averaging)
            # Create a view of the dataframe where flow > cutoff
            if flow_col and flow_col in df.columns:
                df_active = df[df[flow_col] > 1e-6]
                if df_active.empty:
                    # No active flow, fall back to full DF but expect zeros
                    df_active = df 
            else:
                df_active = df
            
            # 3. CALCULATE AVERAGES FROM ACTIVE DATA
            flow_val = df[flow_col].mean() if flow_col else 0.0 # Flow always mean of TOTAL time (production)
            
            temp_val = 0.0
            if temp_col:
                temp_val = df_active[temp_col].mean()
            else:
                # Kelvin fallback
                temp_k_col = next((c for c in df.columns if c in [
                    f'{cid}_outlet_temp_k', f'{cid}_temperature_k', f'{cid}_temp_k'
                ]), None)
                if temp_k_col:
                    temp_val = df_active[temp_k_col].mean() - 273.15
            
            press_val = 0.0
            if press_col:
                press_val = df_active[press_col].mean()
            else:
                 # Pa fallback
                press_pa_col = next((c for c in df.columns if c in [
                    f'{cid}_outlet_pressure_pa', f'{cid}_pressure_pa'
                ]), None)
                if press_pa_col:
                    press_val = df_active[press_pa_col].mean() / 1e5
            
            # --- IMPURITY CALCULATIONS (Issue 2/4) ---
            impurity_val = np.nan
            
            if stream_type == 'H2':
                # O2 Impurity
                ppm_col = next((c for c in df.columns if c in [
                    f'{cid}_outlet_o2_ppm_mol', f'{cid}_outlet_O2_ppm_mol', f'{cid}_o2_ppm'
                ]), None)
                molf_col = next((c for c in df.columns if c in [
                    f'{cid}_outlet_O2_molf', f'{cid}_outlet_o2_molf', f'{cid}_y_O2', f'{cid}_y_o2'
                ]), None)
                
                if ppm_col:
                    impurity_val = df_active[ppm_col].mean()
                elif molf_col:
                    impurity_val = df_active[molf_col].mean() * 1e6
                else: 
                     # Mass fraction fallback (legacy)
                    mass_col = next((c for c in df.columns if c in [
                        f'{cid}_mass_fraction_o2', f'{cid}_w_o2'
                    ]), None)
                    if mass_col:
                        impurity_val = df_active[mass_col].mean() * (MW_H2/MW_O2) * 1e6 # Approx
            else:
                # H2 Impurity
                ppm_col = next((c for c in df.columns if c in [
                    f'{cid}_outlet_h2_ppm_mol', f'{cid}_outlet_H2_ppm_mol', f'{cid}_h2_ppm'
                ]), None)
                molf_col = next((c for c in df.columns if c in [
                    f'{cid}_outlet_H2_molf', f'{cid}_outlet_h2_molf', f'{cid}_y_H2', f'{cid}_y_h2'
                ]), None)
                
                if ppm_col:
                    impurity_val = df_active[ppm_col].mean()
                elif molf_col:
                    impurity_val = df_active[molf_col].mean() * 1e6

            # --- H2O PPM (Issue 2 Fix) ---
            h2o_ppm = 0.0
            
            # Priority: Direct Mole Fraction Column (New Standard)
            h2o_molf_col = next((c for c in df.columns if c in [
                f'{cid}_outlet_H2O_molf', f'{cid}_outlet_h2o_molf', f'{cid}_y_H2O'
            ]), None)
            
            if h2o_molf_col:
                # FIX: Filter strictly positive values to represent "Wet" intervals
                # This prevents zero-flow periods (where mole_frac=0) from dragging average down
                valid_vals = df_active[h2o_molf_col][df_active[h2o_molf_col] > 1e-9]
                if not valid_vals.empty:
                    h2o_ppm = valid_vals.mean() * 1e6
                else:
                    h2o_ppm = df_active[h2o_molf_col].mean() * 1e6
            else:
                # Fallback: Estimate from mass fraction (Simplified)
                h2o_mass_col = next((c for c in df.columns if c in [
                    f'{cid}_outlet_h2o_frac', f'{cid}_mass_fraction_h2o'
                ]), None)
                
                if h2o_mass_col:
                     w_h2o = df_active[h2o_mass_col].mean()
                     # If very small, use ppm approx
                     if w_h2o < 1.0:
                         MW_c = MW_H2 if stream_type == 'H2' else MW_O2
                         h2o_ppm = (w_h2o / MW_H2O) / ((1-w_h2o)/MW_c + w_h2o/MW_H2O) * 1e6
                     else:
                         h2o_ppm = 1e6

            # If values exist (even if flow is 0, we might have config data, but usually we skip)
            # Logic: If we have a valid column for T or Flow, we include point.
            if temp_col or flow_col:
                if np.isnan(impurity_val): impurity_val = 0.0
                
                data['components'].append(cid)
                data['temp'].append(temp_val)
                data['press'].append(press_val)
                data['flow'].append(flow_val)
                data['impurity'].append(impurity_val)
                data['h2o_ppm'].append(h2o_ppm)
        
        return data
    
    # Extract all train profiles
    profiles = {
        'H2_SOEC': extract_profile(_TRAIN_COMPONENTS['H2_SOEC'], 'H2'),
        'O2_SOEC': extract_profile(_TRAIN_COMPONENTS['O2_SOEC'], 'O2'),
        'H2_PEM': extract_profile(_TRAIN_COMPONENTS['H2_PEM'], 'H2'),
        'O2_PEM': extract_profile(_TRAIN_COMPONENTS['O2_PEM'], 'O2'),
    }
    
    # Check if any data exists
    total_components = sum(len(p['components']) for p in profiles.values())
    if total_components == 0:
        return _empty_figure("No process train data found. Check column naming.")
    
    # Create subplot layout
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=False,  # Each train has different components
        vertical_spacing=0.25,  # Significantly increased spacing
        subplot_titles=("Temperature Profile", "Pressure Profile", "Flow & Composition Profile"),
        specs=[[{}], [{}], [{"secondary_y": True}]]
    )
    
    # Track trace indices per train for visibility toggling
    train_trace_indices = {train: [] for train in profiles.keys()}
    trace_counter = 0
    
    # Color map for trains (Main properties: Temp, Press, Flow)
    train_colors = {
        'H2_SOEC': '#1f77b4',  # Blue
        'O2_SOEC': '#ff7f0e',  # Orange
        'H2_PEM': '#2ca02c',   # Green
        'O2_PEM': '#d62728',   # Red
    }
    
    # Distinct colors for Impurity traces
    impurity_colors = {
        'H2_SOEC': '#9467bd',  # Purple
        'O2_SOEC': '#8c564b',  # Brown
        'H2_PEM': '#e377c2',   # Pink
        'O2_PEM': '#7f7f7f',   # Gray
    }
    
    # Distinct colors for H2O traces
    h2o_colors = {
        'H2_SOEC': '#17becf',  # Cyan
        'O2_SOEC': '#bcbd22',  # Olive
        'H2_PEM': '#bcbd22',   # Lime/Olive
        'O2_PEM': '#17becf',   # Cyan
    }
    
    # Add traces for each train
    for train_id, data in profiles.items():
        if not data['components']:
            continue
            
        x_vals = list(range(len(data['components'])))
        color = train_colors[train_id]
        
        # Row 1: Temperature
        fig.add_trace(go.Scatter(
            x=data['components'],
            y=data['temp'],
            mode='lines+markers',
            name=f"{train_id} Temp",
            # legendgroup=train_id, # Removed to allow individual toggling
            line=dict(color=color, width=2),
            marker=dict(size=8),
            hovertemplate='<b>%{x}</b><br>Temp: %{y:.1f} deg C<extra></extra>'
        ), row=1, col=1)
        train_trace_indices[train_id].append(trace_counter)
        trace_counter += 1
        
        # Row 2: Pressure
        fig.add_trace(go.Scatter(
            x=data['components'],
            y=data['press'],
            mode='lines+markers',
            name=f"{train_id} Press",
            # legendgroup=train_id,
            showlegend=True, # Explicitly show independent legend item
            line=dict(color=color, width=2),
            marker=dict(size=8, symbol='square'),
            hovertemplate='<b>%{x}</b><br>Press: %{y:.2f} bar<extra></extra>'
        ), row=2, col=1)
        train_trace_indices[train_id].append(trace_counter)
        trace_counter += 1
        
        # Row 3: Flow (bars)
        fig.add_trace(go.Bar(
            x=data['components'],
            y=data['flow'],
            name=f"{train_id} Flow",
            # legendgroup=train_id,
            showlegend=True,
            marker_color=color,
            opacity=0.6,
            hovertemplate='<b>%{x}</b><br>Flow: %{y:.1f} kg/h<extra></extra>'
        ), row=3, col=1)
        train_trace_indices[train_id].append(trace_counter)
        trace_counter += 1
        
        # Row 3: Impurity (secondary y-axis line) - with legend
        impurity_label = 'O2 ppm' if 'H2' in train_id else 'H2 ppm'
        fig.add_trace(go.Scatter(
            x=data['components'],
            y=data['impurity'],
            mode='lines+markers',
            name=f"{train_id} {impurity_label}",
            # legendgroup=train_id,
            showlegend=True,  # Show in legend for panel 3
            line=dict(color=impurity_colors.get(train_id, 'purple'), width=2, dash='dot'),
            marker=dict(size=6, symbol='diamond'),
            hovertemplate=f'<b>%{{x}}</b><br>{impurity_label}: %{{y:.1f}}<extra></extra>'
        ), row=3, col=1, secondary_y=True)
        train_trace_indices[train_id].append(trace_counter)
        trace_counter += 1
        
        # Row 3: H2O molar ppm (secondary y-axis line) - with legend
        fig.add_trace(go.Scatter(
            x=data['components'],
            y=data['h2o_ppm'],
            mode='lines+markers',
            name=f"{train_id} H2O ppm",
            # legendgroup=train_id,
            showlegend=True,  # Show in legend for panel 3
            line=dict(color=h2o_colors.get(train_id, 'teal'), width=2, dash='dash'),
            marker=dict(size=6, symbol='triangle-up'),
            hovertemplate='<b>%{x}</b><br>H2O ppm: %{y:.1f}<extra></extra>'
        ), row=3, col=1, secondary_y=True)
        train_trace_indices[train_id].append(trace_counter)
        trace_counter += 1
    
    n_traces = trace_counter
    
    # Build visibility arrays for dropdown buttons
    def get_visibility(selected_trains: list) -> list:
        """Generate visibility array for selected trains."""
        vis = [False] * n_traces
        for train in selected_trains:
            for idx in train_trace_indices.get(train, []):
                vis[idx] = True
        return vis
    
    # Train selector dropdown (7 options)
    train_buttons = [
        dict(label="H2 SOEC", method="update", args=[{"visible": get_visibility(['H2_SOEC'])}]),
        dict(label="O2 SOEC", method="update", args=[{"visible": get_visibility(['O2_SOEC'])}]),
        dict(label="H2 PEM", method="update", args=[{"visible": get_visibility(['H2_PEM'])}]),
        dict(label="O2 PEM", method="update", args=[{"visible": get_visibility(['O2_PEM'])}]),
        dict(label="H2 All", method="update", args=[{"visible": get_visibility(['H2_SOEC', 'H2_PEM'])}]),
        dict(label="O2 All", method="update", args=[{"visible": get_visibility(['O2_SOEC', 'O2_PEM'])}]),
        dict(label="All", method="update", args=[{"visible": [True] * n_traces}]),
    ]
    
    fig.update_layout(
        title=kwargs.get('title', 'Process Train Profile'),
        template='plotly_white',
        height=1000,
        updatemenus=[
            dict(
                type="dropdown",
                direction="down",
                x=1.0, y=1.08,
                xanchor="left",
                showactive=True,
                active=6,  # Default to "All"
                buttons=train_buttons
            )
        ],
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=0.9
        )
    )
    
    # Update axes
    fig.update_yaxes(title_text="Temperature (deg C)", row=1, col=1)
    fig.update_yaxes(title_text="Pressure (bar)", row=2, col=1)
    fig.update_yaxes(title_text="Mass Flow (kg/h)", row=3, col=1)
    fig.update_yaxes(title_text="Impurity / H2O (ppm)", type="log", row=3, col=1, secondary_y=True)
    
    # Rotate x-axis labels for readability
    fig.update_xaxes(tickangle=45, row=1, col=1)
    fig.update_xaxes(tickangle=45, row=2, col=1)
    fig.update_xaxes(tickangle=45, title_text="Component", row=3, col=1)
    
    return fig
