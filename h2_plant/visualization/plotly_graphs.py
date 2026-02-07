"""
Plotly graph implementations for H2 Plant visualization.
"""

from typing import Dict, Any, List, Optional
import logging
import math
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
def plot_soec_module_degradation(df: pd.DataFrame, **kwargs) -> go.Figure:
    """
    Plot SOEC Per-Module Degradation Tracking.
    
    Shows Physical Module operating hours and efficiency (SEC) over time to
    visualize wear distribution and rotation effectiveness.
    
    Args:
        df: DataFrame containing 'soec_module_hours_{i}' and 'soec_module_eff_{i}' columns.
        **kwargs: Additional plot customization options.
    
    Returns:
        go.Figure: Dual-axis plot showing hours (left) and efficiency (right).
    """
    _check_dependencies()
    
    from h2_plant.visualization.utils import downsample_dataframe, get_time_axis_hours, get_viz_config
    
    maxpoints = kwargs.get('maxpoints', get_viz_config('performance.max_points_default', 2000))
    df_plot = downsample_dataframe(df, max_points=maxpoints)
    hours = get_time_axis_hours(df_plot)
    
    # Detect number of modules by scanning columns
    hours_cols = sorted([c for c in df_plot.columns if c.startswith('soec_module_hours_')])
    eff_cols = sorted([c for c in df_plot.columns if c.startswith('soec_module_eff_')])
    
    if not hours_cols:
        return _empty_figure("No SOEC module degradation data found")
    
    num_modules = len(hours_cols)
    
    # Create dual-axis figure
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Accumulated Operating Hours per Module", "Specific Energy Consumption per Module (SEC)")
    )
    
    # Color palette for modules
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    ScatterType = get_scatter_type(len(hours))
    
    # Row 1: Accumulated Hours (Physical wear indicator)
    for i, col in enumerate(hours_cols):
        module_num = col.split('_')[-1]
        color = colors[i % len(colors)]
        
        fig.add_trace(ScatterType(
            x=hours,
            y=df_plot[col].values,
            mode='lines',
            name=f'Module {module_num} Hours',
            line=dict(color=color, width=1.5),
            legendgroup=f"mod{module_num}",
            hovertemplate=f'<b>Module {module_num}</b><br>Hours: %{{y:.1f}}<extra></extra>'
        ), row=1, col=1)
    
    # Row 2: Efficiency (SEC kWh/kg) - Lower is better
    for i, col in enumerate(eff_cols):
        module_num = col.split('_')[-1]
        color = colors[i % len(colors)]
        
        fig.add_trace(ScatterType(
            x=hours,
            y=df_plot[col].values,
            mode='lines',
            name=f'Module {module_num} SEC',
            line=dict(color=color, width=1.5, dash='dot'),
            legendgroup=f"mod{module_num}",
            showlegend=False,
            hovertemplate=f'<b>Module {module_num}</b><br>SEC: %{{y:.2f}} kWh/kg<extra></extra>'
        ), row=2, col=1)
    
    # Add BOL reference line on SEC plot
    fig.add_hline(
        y=37.54, row=2, col=1,
        line_dash="dash", line_color="green",
        annotation_text="BOL (37.54 kWh/kg)",
        annotation_position="right"
    )
    
    fig.update_layout(
        title=kwargs.get('title', 'SOEC Physical Module Degradation Tracking'),
        template='plotly_white',
        hovermode='x unified',
        legend=dict(x=1.02, y=1.0, xanchor='left'),
        height=600
    )
    
    fig.update_xaxes(title_text="Time (hours)", row=2, col=1)
    fig.update_yaxes(title_text="Operating Hours", row=1, col=1)
    fig.update_yaxes(title_text="SEC (kWh/kg)", row=2, col=1)
    
    return fig


@log_graph_errors
def plot_total_production_stacked(df: pd.DataFrame, **kwargs) -> go.Figure:
    """Plot stacked area chart showing SOEC + ATR + PEM contributions.
    
    Note: H2_pem_kg/H2_soec_kg are mass per timestep. We convert to kg/h for display.
    """
    _check_dependencies()
    
    from h2_plant.visualization.utils import downsample_dataframe, get_time_axis_hours, get_viz_config
    
    maxpoints = kwargs.get('maxpoints', get_viz_config('performance.max_points_default', 2000))
    df_plot = downsample_dataframe(df, max_points=maxpoints)
    hours = get_time_axis_hours(df_plot)
    
    pem_col = next((c for c in ['H2_pem_kg', 'H2_pem'] if c in df_plot.columns), None)
    soec_col = next((c for c in ['H2_soec_kg', 'H2_soec'] if c in df_plot.columns), None)
    atr_col = next((c for c in ['H2_atr_kg', 'H2_atr'] if c in df_plot.columns), None)
    
    # Unit conversion
    dt_seconds = df.attrs.get('dt_seconds', 60.0)
    dt_h = dt_seconds / 3600.0
    
    pem_raw = df_plot[pem_col].values if pem_col else np.zeros(len(hours))
    soec_raw = df_plot[soec_col].values if soec_col else np.zeros(len(hours))
    atr_raw = df_plot[atr_col].values if atr_col else np.zeros(len(hours))
    
    # Convert per-timestep to rate (kg/h)
    pem_production = pem_raw / dt_h if pem_col and not pem_col.endswith('_kg_h') else pem_raw
    soec_production = soec_raw / dt_h if soec_col and not soec_col.endswith('_kg_h') else soec_raw
    atr_production = atr_raw / dt_h if atr_col and not atr_col.endswith('_kg_h') else atr_raw
    
    color_pem = get_viz_config('styling.colors.pem', '#1f77b4')
    color_soec = get_viz_config('styling.colors.soec', '#ff7f0e')
    color_atr = get_viz_config('styling.colors.atr', _SUBSYSTEM_COLORS.get('ATR', '#2ca02c'))
    
    fig = go.Figure()
    
    fig.add_trace(get_scatter_type(len(hours))(
        x=hours,
        y=soec_production,
        mode='lines',
        name='SOEC',
        stackgroup='one',
        line=dict(color=color_soec, width=0.5),
        fillcolor=f"rgba{tuple(int(color_soec.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.7,)}" if color_soec.startswith('#') else color_soec
    ))
    
    if atr_col:
        fig.add_trace(get_scatter_type(len(hours))(
            x=hours,
            y=atr_production,
            mode='lines',
            name='ATR',
            stackgroup='one',
            line=dict(color=color_atr, width=0.5),
            fillcolor=f"rgba{tuple(int(color_atr.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.7,)}" if color_atr.startswith('#') else color_atr
        ))
    
    fig.add_trace(get_scatter_type(len(hours))(
        x=hours,
        y=pem_production,
        mode='lines',
        name='PEM',
        stackgroup='one',
        line=dict(color=color_pem, width=0.5),
        fillcolor=f"rgba{tuple(int(color_pem.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.7,)}" if color_pem.startswith('#') else color_pem
    ))
    
    fig.update_layout(
        title=kwargs.get('title', 'Total H2 Production (SOEC + ATR + PEM)'),
        xaxis_title='Time (hours)',
        yaxis_title='H2 Production (kg/h)',
        template='plotly_white',
        hovermode='x unified',
        legend=dict(x=0.02, y=0.98)
    )
    
    return fig


@log_graph_errors
def plot_energy_price_daily_monthly(df: pd.DataFrame, **kwargs) -> go.Figure:
    """
    Plot daily and monthly average energy prices (EUR/MWh) in a single chart.
    """
    _check_dependencies()

    from h2_plant.visualization.utils import get_viz_config

    price_col = next((c for c in ['energy_price_eur_kwh', 'pricing_energy_price_eur_kwh'] if c in df.columns), None)
    spot_col = next((c for c in ['spot_price', 'Spot'] if c in df.columns), None)

    if not price_col and not spot_col:
        return _empty_figure("No price data found")

    if price_col:
        price_mwh = df[price_col].values * 1000.0
    else:
        price_mwh = df[spot_col].values

    if 'minute' in df.columns:
        minutes = df['minute'].values
    else:
        dt_seconds = df.attrs.get('dt_seconds', 60.0)
        minutes = np.arange(len(df)) * (dt_seconds / 60.0)

    df_tmp = pd.DataFrame({
        'minute': minutes,
        'price_mwh': price_mwh
    })

    df_tmp['day_group'] = (df_tmp['minute'] // 1440).astype(int)
    df_tmp['month_group'] = (df_tmp['minute'] // 43800).astype(int)

    daily = df_tmp.groupby('day_group', as_index=False)['price_mwh'].mean()
    monthly = df_tmp.groupby('month_group', as_index=False)['price_mwh'].mean()

    # X-axis in days
    daily_x = daily['day_group'].values.astype(float)
    monthly_x = monthly['month_group'].values.astype(float) * 30.4

    fig = go.Figure()

    fig.add_trace(get_scatter_type(len(daily_x))(
        x=daily_x,
        y=daily['price_mwh'].values,
        mode='lines',
        name='Daily Avg Price',
        line=dict(color=get_viz_config('styling.colors.price', '#9467bd'), width=2)
    ))

    fig.add_trace(get_scatter_type(len(monthly_x))(
        x=monthly_x,
        y=monthly['price_mwh'].values,
        mode='lines+markers',
        name='Monthly Avg Price',
        line=dict(color='#2c3e50', width=2, dash='dash'),
        marker=dict(size=6, color='#2c3e50')
    ))

    fig.update_layout(
        title=kwargs.get('title', 'Energy Price (Daily & Monthly Averages)'),
        xaxis_title='Time (days)',
        yaxis_title='Price (EUR/MWh)',
        template='plotly_white',
        hovermode='x unified'
    )

    return fig


@log_graph_errors
def plot_atr_oxygen_supply(df: pd.DataFrame, **kwargs) -> go.Figure:
    """
    Plot stacked oxygen supply to ATR (PEM vs External) with total inlet line.
    
    Legend shows total mass consumed from each source.
    """
    _check_dependencies()
    from h2_plant.visualization.utils import downsample_dataframe, get_time_axis_hours, get_viz_config, get_dt_hours, find_column
    
    maxpoints = kwargs.get('maxpoints', get_viz_config('performance.max_points_default', 2000))
    df_plot = downsample_dataframe(df, max_points=maxpoints)
    if df_plot.empty:
        return _empty_figure("No DataFrame provided")
    
    hours = get_time_axis_hours(df_plot)
    dt_h = get_dt_hours(df)
    
    def _hex_rgba(color: str, alpha: float = 0.7) -> str:
        if isinstance(color, str) and color.startswith('#') and len(color) == 7:
            rgb = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            return f"rgba{rgb + (alpha,)}"
        return color
    
    def _rate_from_col(frame: pd.DataFrame, col: Optional[str], factor: float = 1.0) -> Optional[np.ndarray]:
        if not col or col not in frame.columns:
            return None
        data = frame[col].values.astype(float) * factor
        col_lower = col.lower()
        if col_lower.endswith('_kg_h') or 'mass_flow' in col_lower or 'flow_kg_h' in col_lower:
            return data
        return data / dt_h if dt_h > 0 else data
    
    def _find_by_parts(frame: pd.DataFrame, parts: List[str]) -> Optional[str]:
        for c in frame.columns:
            lower = c.lower()
            if all(p in lower for p in parts):
                return c
        return None
    
    total_col = find_column(
        df_plot,
        'ATR_O2_Compressor',
        'outlet_mass_flow_kg_h',
        fallback_patterns=['outlet_mass_flow', 'outlet_mass_kg_h', 'outlet_flow_kg_h', 'mass_flow_kg_h']
    )
    if not total_col:
        total_col = _find_by_parts(df_plot, ['atr_o2_compressor', 'mass_flow'])
    
    pem_col = find_column(
        df_plot,
        'O2_Production_Mixer',
        'outlet_mass_flow_kg_h',
        fallback_patterns=['outlet_mass_flow', 'outlet_mass_kg_h', 'outlet_flow_kg_h', 'mass_flow_kg_h']
    )
    if not pem_col:
        pem_col = find_column(
            df_plot,
            'PEM_O2_ElectricBoiler',
            'outlet_mass_flow_kg_h',
            fallback_patterns=['outlet_mass_flow', 'outlet_mass_kg_h', 'outlet_flow_kg_h', 'mass_flow_kg_h']
        )
    if not pem_col:
        pem_col = find_column(
            df_plot,
            'PEM_O2_Valve',
            'outlet_mass_flow_kg_h',
            fallback_patterns=['outlet_mass_flow', 'outlet_mass_kg_h', 'outlet_flow_kg_h', 'mass_flow_kg_h']
        )
    
    pem_factor = 1.0
    if not pem_col:
        if 'O2_pem_kg' in df_plot.columns:
            pem_col = 'O2_pem_kg'
        elif 'H2_pem_kg' in df_plot.columns:
            pem_col = 'H2_pem_kg'
            pem_factor = 8.0
    
    ext_col = find_column(
        df_plot,
        'O2_Backup_Supply',
        'makeup_flow_kg_h',
        fallback_patterns=['makeup_flow', 'makeup_kg_h', 'makeup']
    )
    if not ext_col:
        ext_col = _find_by_parts(df_plot, ['o2_backup_supply', 'makeup'])
    
    def _build_rates(frame: pd.DataFrame) -> tuple:
        total_rate = _rate_from_col(frame, total_col)
        pem_rate = _rate_from_col(frame, pem_col, pem_factor)
        ext_rate = _rate_from_col(frame, ext_col)
        
        if total_rate is None:
            if pem_rate is not None and ext_rate is not None:
                total_rate = pem_rate + ext_rate
            elif pem_rate is not None:
                total_rate = pem_rate
            elif ext_rate is not None:
                total_rate = ext_rate
            else:
                return None, None, None
        
        if ext_rate is not None:
            ext_rate = np.minimum(ext_rate, total_rate)
            pem_rate = np.maximum(total_rate - ext_rate, 0.0)
        elif pem_rate is not None:
            pem_rate = np.minimum(pem_rate, total_rate)
            ext_rate = np.maximum(total_rate - pem_rate, 0.0)
        else:
            return total_rate, None, None
        
        return total_rate, pem_rate, ext_rate
    
    total_rate_plot, pem_rate_plot, ext_rate_plot = _build_rates(df_plot)
    if total_rate_plot is None:
        o2_cols = [c for c in df_plot.columns if 'o2' in c.lower() or 'oxygen' in c.lower()]
        msg = "No ATR oxygen flow data found."
        if o2_cols:
            msg += f" Available O2 columns: {o2_cols[:10]}"
        return _empty_figure(msg)
    
    if pem_rate_plot is None or ext_rate_plot is None:
        return _empty_figure("ATR oxygen supply split unavailable (missing PEM or external O2 columns).")
    
    total_rate_full, pem_rate_full, ext_rate_full = _build_rates(df)
    total_pem = float(np.nansum(pem_rate_full) * dt_h) if pem_rate_full is not None else 0.0
    total_ext = float(np.nansum(ext_rate_full) * dt_h) if ext_rate_full is not None else 0.0
    total_o2 = float(np.nansum(total_rate_full) * dt_h) if total_rate_full is not None else 0.0
    
    def _fmt_total(value: float) -> str:
        if np.isnan(value) or value < 0:
            value = 0.0
        return f"{value:,.0f} kg"
    
    pem_name = f"PEM O2 (total {_fmt_total(total_pem)})"
    ext_name = f"External O2 (total {_fmt_total(total_ext)})"
    total_name = f"ATR O2 Inlet (total {_fmt_total(total_o2)})"
    
    color_pem = get_viz_config('styling.colors.pem', '#1f77b4')
    color_ext = get_viz_config('styling.colors.o2', _SUBSYSTEM_COLORS.get('O2', '#e377c2'))
    
    fig = go.Figure()
    ScatterType = get_scatter_type(len(hours))
    
    fig.add_trace(ScatterType(
        x=hours,
        y=np.nan_to_num(pem_rate_plot, nan=0.0),
        mode='lines',
        name=pem_name,
        stackgroup='one',
        line=dict(color=color_pem, width=0.5),
        fillcolor=_hex_rgba(color_pem, 0.7),
        hovertemplate='PEM O2: %{y:.1f} kg/h<extra></extra>'
    ))
    
    fig.add_trace(ScatterType(
        x=hours,
        y=np.nan_to_num(ext_rate_plot, nan=0.0),
        mode='lines',
        name=ext_name,
        stackgroup='one',
        line=dict(color=color_ext, width=0.5),
        fillcolor=_hex_rgba(color_ext, 0.7),
        hovertemplate='External O2: %{y:.1f} kg/h<extra></extra>'
    ))
    
    fig.add_trace(ScatterType(
        x=hours,
        y=np.nan_to_num(total_rate_plot, nan=0.0),
        mode='lines',
        name=total_name,
        line=dict(color='#222222', width=1.5, dash='dash'),
        hovertemplate='ATR O2 Inlet: %{y:.1f} kg/h<extra></extra>'
    ))
    
    fig.update_layout(
        title=kwargs.get('title', 'ATR Oxygen Supply (PEM + External)'),
        xaxis_title='Time (hours)',
        yaxis_title='O2 Flow (kg/h)',
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
    from h2_plant.visualization.utils import downsample_dataframe, get_time_axis_hours, get_viz_config
    
    maxpoints = kwargs.get('maxpoints', get_viz_config('performance.max_points_default', 2000))
    df_plot = downsample_dataframe(df, max_points=maxpoints)
    hours = get_time_axis_hours(df_plot)
    
    # Identify Production Sources
    sources = {}
    known_prefixes = ['pem', 'soec', 'atr']
    
    # NEW LOGIC: Use pre-calculated cumulative metrics if available (OPTIMIZATION)
    # Check for direct cumulative columns first (accurate across chunks)
    for prefix in known_prefixes:
        col_cand = next((c for c in df_plot.columns if f"cumulative_h2_{prefix}_kg" in c.lower()), None)
        if col_cand:
            sources[prefix.upper()] = df_plot[col_cand].values
    
    # Fallback: Integration from rate (less accurate across chunks if history is split)
    if not sources:
        dt_hours = np.mean(np.diff(hours)) if len(hours) > 1 else (df.attrs.get('dt_seconds', 60)/3600)
        for col in df_plot.columns:
            col_lower = col.lower()
            for prefix in known_prefixes:
                if f"h2_{prefix}" in col_lower and "cumulative" not in col_lower and "kg" in col_lower:
                    if prefix.upper() not in sources:
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
def plot_rfnbo_compliance(df: pd.DataFrame, **kwargs) -> go.Figure:
    """
    Plot RFNBO vs Non-RFNBO H2 production with compliance threshold.
    """
    _check_dependencies()
    from h2_plant.visualization.utils import downsample_dataframe, get_time_axis_hours, get_viz_config

    maxpoints = kwargs.get('maxpoints', get_viz_config('performance.max_points_default', 2000))
    df_plot = downsample_dataframe(df, max_points=maxpoints)
    hours = get_time_axis_hours(df_plot)

    # Columns
    rfnbo_col = next((c for c in df_plot.columns if 'cumulative_h2_rfnbo_kg' in c.lower()), None)
    non_rfnbo_col = next((c for c in df_plot.columns if 'cumulative_h2_non_rfnbo_kg' in c.lower()), None)
    threshold_col = next((c for c in df_plot.columns if 'purchase_threshold' in c.lower() or 'spot_threshold' in c.lower()), None)

    if not rfnbo_col or not non_rfnbo_col:
        # Try finding non-cumulative rate columns and integrating
        rfnbo_rate = next((c for c in df_plot.columns if 'h2_rfnbo_kg' in c.lower() and 'cumulative' not in c.lower()), None)
        non_rfnbo_rate = next((c for c in df_plot.columns if 'h2_non_rfnbo_kg' in c.lower() and 'cumulative' not in c.lower()), None)
        
        if rfnbo_rate and non_rfnbo_rate:
             rfnbo_data = df_plot[rfnbo_rate].cumsum().values
             non_rfnbo_data = df_plot[non_rfnbo_rate].cumsum().values
        else:
            return _empty_figure("No RFNBO production data available")
    else:
        rfnbo_data = df_plot[rfnbo_col].values
        non_rfnbo_data = df_plot[non_rfnbo_col].values

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    ScatterType = get_scatter_type(len(hours))

    # 1. Stacked Area for Production
    fig.add_trace(ScatterType(
        x=hours, y=rfnbo_data,
        name='RFNBO H2',
        stackgroup='one',
        line=dict(width=0.5, color='#2ca02c'), # Green
        fillcolor='rgba(44, 160, 44, 0.6)'
    ), secondary_y=False)

    fig.add_trace(ScatterType(
        x=hours, y=non_rfnbo_data,
        name='Non-RFNBO H2',
        stackgroup='one',
        line=dict(width=0.5, color='#7f7f7f'), # Grey
        fillcolor='rgba(127, 127, 127, 0.6)'
    ), secondary_y=False)

    # 2. Threshold Price (Secondary Axis)
    if threshold_col:
        price_limit = df_plot[threshold_col].values
        fig.add_trace(ScatterType(
            x=hours, y=price_limit,
            name='Spot Price Threshold',
            mode='lines',
            line=dict(width=2, color='red', dash='dash'),
            hovertemplate='%{y:.2f} EUR/MWh'
        ), secondary_y=True)

    fig.update_layout(
        title=kwargs.get('title', 'RFNBO Compliance & Production'),
        template='plotly_white',
        hovermode='x unified',
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)')
    )
    
    fig.update_yaxes(title_text="Cumulative H2 (kg)", secondary_y=False)
    fig.update_yaxes(title_text="Price Threshold (EUR/MWh)", secondary_y=True, showgrid=False)

    return fig


@log_graph_errors
def plot_rfnbo_pie(df: pd.DataFrame, **kwargs) -> go.Figure:
    """
    Pie chart showing percentage of RFNBO vs Non-RFNBO hydrogen.
    """
    _check_dependencies()
    
    # Get total cumulative values from the last row
    rfnbo_col = next((c for c in df.columns if 'cumulative_h2_rfnbo_kg' in c.lower()), None)
    non_rfnbo_col = next((c for c in df.columns if 'cumulative_h2_non_rfnbo_kg' in c.lower()), None)
    
    total_rfnbo = 0.0
    total_non_rfnbo = 0.0
    
    if rfnbo_col and non_rfnbo_col:
        total_rfnbo = df[rfnbo_col].iloc[-1]
        total_non_rfnbo = df[non_rfnbo_col].iloc[-1]
    else:
        # Fallback to sum of rates
        rfnbo_rate = next((c for c in df.columns if 'h2_rfnbo_kg' in c.lower() and 'cumulative' not in c.lower()), None)
        non_rfnbo_rate = next((c for c in df.columns if 'h2_non_rfnbo_kg' in c.lower() and 'cumulative' not in c.lower()), None)
        
        if rfnbo_rate and non_rfnbo_rate:
            total_rfnbo = df[rfnbo_rate].sum()
            total_non_rfnbo = df[non_rfnbo_rate].sum()
        else:
            return _empty_figure("No RFNBO data for Pie Chart")
            
    total = total_rfnbo + total_non_rfnbo
    if total < 1e-3:
        return _empty_figure("No Hydrogen Produced")
        
    values = [total_rfnbo, total_non_rfnbo]
    labels = ['RFNBO Compliant', 'Non-Compliant']
    colors = ['#2ca02c', '#7f7f7f'] # Green, Grey
    
    fig = go.Figure(data=[go.Pie(
        labels=labels, 
        values=values, 
        hole=.4,
        marker=dict(colors=colors),
        textinfo='label+percent',
        hovertemplate='<b>%{label}</b><br>%{value:,.0f} kg<br>%{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title=kwargs.get('title', f"RFNBO Compliance Share (Total: {total:,.0f} kg)"),
        template='plotly_white'
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
def plot_atr_efficiency_timeline(df: pd.DataFrame, **kwargs) -> go.Figure:
    """
    Plot ATR efficiency over time (Chemical and Global).
    """
    _check_dependencies()
    
    from h2_plant.visualization.utils import downsample_dataframe, get_time_axis_hours, get_viz_config

    maxpoints = kwargs.get('maxpoints', get_viz_config('performance.max_points_default', 2000))
    df_plot = downsample_dataframe(df, max_points=maxpoints)
    hours = get_time_axis_hours(df_plot)
    
    # Find ATR efficiency columns
    chem_col = next((c for c in df_plot.columns if 'atr_efficiency_chemical' in c or 'atr_eff_chem' in c), None)
    glob_col = next((c for c in df_plot.columns if 'atr_efficiency_global' in c or 'atr_eff_global' in c), None)
    
    if not chem_col:
        return _empty_figure("No ATR efficiency data available")
        
    eff_chem = df_plot[chem_col].values * 100.0
    eff_glob = df_plot[glob_col].values * 100.0 if glob_col else np.zeros(len(hours))
    
    # Filter out potential crazy noise or zero-start artifacts if needed
    # But usually raw data is preferred for engineering debug.
    
    fig = go.Figure()
    
    # Chemical Efficiency
    fig.add_trace(get_scatter_type(len(hours))(
        x=hours,
        y=eff_chem,
        mode='lines',
        name='Chemical Efficiency (LHV)',
        line=dict(color='#2ca02c', width=2), # Green
        hovertemplate='Time: %{x:.1f}h<br>Chemical Eff: %{y:.1f}%<extra></extra>'
    ))
    
    # Global Efficiency (CHP)
    if glob_col and np.max(eff_glob) > 0.1:
        fig.add_trace(get_scatter_type(len(hours))(
            x=hours,
            y=eff_glob,
            mode='lines',
            name='Global Efficiency (CHP)',
            line=dict(color='#ff7f0e', width=2, dash='dash'), # Orange
            hovertemplate='Time: %{x:.1f}h<br>Global Eff: %{y:.1f}%<extra></extra>'
        ))

    # Add reference line (Mock target ~80% usually?)
    # Just grid is fine.
    
    fig.update_layout(
        title=kwargs.get('title', 'ATR Plant Efficiency'),
        xaxis_title='Time (hours)',
        yaxis_title='Efficiency (%)',
        template='plotly_white',
        hovermode='x unified',
        yaxis=dict(range=[0, 105])
    )
    
    return fig



@log_graph_errors
def plot_global_efficiency_timeline(df: pd.DataFrame, **kwargs) -> go.Figure:
    """
    Plot Integrated Plant Efficiency (Eq 5.42) over time.
    """
    _check_dependencies()
    
    from h2_plant.visualization.utils import downsample_dataframe, get_time_axis_hours, get_viz_config

    maxpoints = kwargs.get('maxpoints', get_viz_config('performance.max_points_default', 2000))
    df_plot = downsample_dataframe(df, max_points=maxpoints)
    hours = get_time_axis_hours(df_plot)
    
    # Find Efficiency Column
    eff_col = next((c for c in ['integrated_global_efficiency', 'global_efficiency'] if c in df_plot.columns), None)
    
    if not eff_col:
        return _empty_figure("No Global Efficiency data available")
        
    eff = df_plot[eff_col].values * 100.0
    
    # Filter out noise/zeros if needed (e.g. idle states)
    # eff = np.where(eff > 0.01, eff, np.nan) 
    
    fig = go.Figure()
    
    fig.add_trace(get_scatter_type(len(hours))(
        x=hours,
        y=eff,
        mode='lines',
        name='Integrated Plant Efficiency',
        line=dict(color='darkgreen', width=2),
        hovertemplate='Time: %{x:.1f}h<br>Eff: %{y:.1f}%<extra></extra>'
    ))
    
    # Add mean line for operating periods
    if np.any(eff > 0.1):
        mean_eff = np.mean(eff[eff > 0.1])
        fig.add_hline(
            y=mean_eff, 
            line_dash="dash", 
            line_color="green",
            annotation_text=f"Mean: {mean_eff:.1f}%"
        )
    
    fig.update_layout(
        title=kwargs.get('title', 'Integrated Plant Efficiency'),
        xaxis_title='Time (hours)',
        yaxis_title='Efficiency (% LHV)',
        template='plotly_white',
        hovermode='x unified',
        yaxis=dict(range=[0, 100])
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
        y=soec_power,
        mode='lines',
        name='SOEC',
        stackgroup='one',
        line=dict(color=get_viz_config('styling.colors.soec', '#ff7f0e'), width=0.5),
        fillcolor='rgba(255, 127, 14, 0.7)'
    ))

    fig.add_trace(get_scatter_type(len(hours))(
        x=hours,
        y=pem_power,
        mode='lines',
        name='PEM',
        stackgroup='one',
        line=dict(color=get_viz_config('styling.colors.pem', '#1f77b4'), width=0.5),
        fillcolor='rgba(31, 119, 180, 0.7)'
    ))
    
    # Reordered stack: SOEC -> PEM -> Grid Export -> BOP
    fig.add_trace(get_scatter_type(len(hours))(
        x=hours,
        y=sell_power,
        mode='lines',
        name='Grid Export',
        stackgroup='one',
        line=dict(color='#2ca02c', width=0.5),
        fillcolor='rgba(44, 160, 44, 0.7)'
    ))

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
    Donut chart showing Total Energy Consumption breakdown.
    Designed to be compact for academic thesis use.
    
    Calculates Total Energy (MWh) for each subsystem over the simulation period.
    """
    _check_dependencies()
    
    # -------------------------
    # 1) Category definitions
    # -------------------------
    storage_compression_comps = [
        'LP_Compressor_S1', 'LP_Intercooler_1',
        'HP_Compressor_S2', 'HP_Intercooler_2',
        'HP_Compressor_S3', 'HP_Intercooler_3',
        'HP_Compressor_S4', 'HP_Intercooler_4',
        'HP_Compressor_S5', 'Truck_Station'
    ]
    pem_comps = [
        'PEM_Electrolyzer', 'PEM_H2_KOD_1', 'PEM_H2_KOD_2', 'PEM_H2_Coalescer_1',
        'PEM_H2_ElectricBoiler_1', 'PEM_H2_Deoxo_1', 'PEM_H2_KOD_3',
        'PEM_H2_ElectricBoiler_2', 'PEM_H2_PSA_1', 'PEM_O2_KOD_1', 
        'PEM_O2_KOD_2', 'PEM_O2_Coalescer_1', 'PEM_O2_ElectricBoiler'
    ]
    soec_comps = [
        'SOEC_Cluster', 'SOEC_H2_Interchanger_1', 'SOEC_H2_KOD_1', 'SOEC_H2_Cyclone_1',
        'SOEC_H2_Compressor_S1', 'SOEC_H2_Compressor_S2', 'SOEC_H2_Cyclone_2',
        'SOEC_H2_Compressor_S3', 'SOEC_H2_Cyclone_3', 'SOEC_H2_Compressor_S4',
        'SOEC_H2_Cyclone_4', 'SOEC_H2_Compressor_S5', 'SOEC_H2_Cyclone_5',
        'SOEC_H2_Compressor_S6', 'SOEC_H2_Deoxo_1', 'SOEC_H2_Coalescer_2',
        'SOEC_H2_ElectricBoiler_PSA', 'SOEC_H2_PSA_1',
        'SOEC_O2_Interchanger_1', 'SOEC_O2_compressor_1', 'SOEC_O2_compressor_2',
        'SOEC_O2_compressor_3', 'SOEC_O2_compressor_4',
        'SOEC_Steam_Boiler', 'SOEC_Steam_Compressor_1', 'SOEC_Steam_Drycooler',
        'SOEC_Steam_Compressor_2', 'SOEC_Feed_Pump', 'SOEC_H2_Boiler'
    ]
    atr_comps = [
        'ATR_Plant', 'ATR_H01_Boiler', 'ATR_H02_Boiler', 'ATR_H04_Boiler',
        'ATR_O2_Compressor', 'ATR_PSA_1', 'ATR_H2_Compressor_1', 'ATR_H2_Compressor_2',
        'Biogas_Source', 'Biogas_Compressor_1', 'Biogas_Compressor_2', 'Biogas_Compressor_3'
    ]
    water_comps = [
        'Water_Source', 'Water_Purifier', 'UltraPure_Tank',
        'ATR_Feed_Pump', 'PEM_Water_Pump', 'SOEC_Drain_Pump',
        'SOEC_DRAIN_PUMP_1', 'SOEC_DRAIN_PUMP_2',
        'ATR_Drain_Pump_1', 'ATR_Drain_Pump_2'
    ]

    # Categories initialization (MWh)
    categories = {
        'Storage & Compression': 0.0,
        'PEM': 0.0,
        'SOEC': 0.0,
        'ATR': 0.0,
        'Water Treatment': 0.0,
        'Cooling': 0.0
    }

    # -------------------------
    # 2) Calculate Energy
    # -------------------------
    # Determine simulation duration in hours
    dt_seconds = df.attrs.get('dt_seconds', 60.0)
    total_duration_hours = len(df) * dt_seconds / 3600.0
    
    # If minute column exists, try to be more precise
    if 'minute' in df.columns and len(df) > 1:
        total_duration_hours = (df['minute'].max() - df['minute'].min() + (dt_seconds/60.0)) / 60.0

    def is_cooling(col_name: str) -> bool:
        lower = col_name.lower()
        keywords = ['drycooler', 'chiller', 'intercooler', 'coolingmanager']
        is_cooling_comp = any(k in lower for k in keywords)
        is_power = ('fan_power' in lower or 'electrical_power' in lower or
                    'power_kw' in lower or '_kw' in lower or lower.endswith('kw') or lower.endswith('mw'))
        return is_cooling_comp and is_power and 'duty' not in lower

    # Gather power columns
    cols = df.columns.tolist()
    power_cols = [c for c in cols if (c.lower().endswith('kw') or c.lower().endswith('mw') or c.startswith('P_'))
                  and 'price' not in c.lower()]

    for col in power_cols:
        # Get Mean Power in kW
        mean_kw = df[col].mean()
        if pd.isna(mean_kw): continue
        
        # Convert MW columns to kW
        if 'mw' in col.lower() or col in ['P_pem', 'P_soec_actual', 'P_offer', 'P_sold', 'P_bop_mw']:
            mean_kw *= 1000.0

        if abs(mean_kw) < 0.001: continue

        # Calculate Energy for this component (kWh) -> convert to MWh later
        energy_kwh = mean_kw * total_duration_hours

        col_lower = col.lower()
        if is_cooling(col):
            categories['Cooling'] += energy_kwh
        elif any(col.startswith(comp) for comp in storage_compression_comps):
            categories['Storage & Compression'] += energy_kwh
        elif any(col.startswith(comp) for comp in pem_comps) or col == 'P_pem':
            categories['PEM'] += energy_kwh
        elif any(col.startswith(comp) for comp in soec_comps) or col in ['P_soec_actual', 'P_soec']:
            categories['SOEC'] += energy_kwh
        elif any(col.startswith(comp) for comp in atr_comps):
            categories['ATR'] += energy_kwh
        elif any(col.startswith(comp) for comp in water_comps) or 'pump' in col_lower:
            categories['Water Treatment'] += energy_kwh

    # Convert kWh to MWh
    for k in categories:
        categories[k] /= 1000.0

    # -------------------------
    # 3) Plotting
    # -------------------------
    # Filter zeros
    labels = [k for k, v in categories.items() if v > 0.01]
    values = [categories[k] for k in labels]
    total_mwh = sum(values)
    
    if total_mwh == 0:
        return _empty_figure("No energy consumption data found")

    # Sort slices descending
    sorted_indices = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    labels = [labels[i] for i in sorted_indices]
    values = [values[i] for i in sorted_indices]

    # Colors
    color_map = {
        'Storage & Compression': '#1f77b4', # Blue
        'PEM': '#d62728',    # Red
        'SOEC': '#ff7f0e',   # Orange
        'ATR': '#2ca02c',    # Green
        'Water Treatment': '#17becf', # Cyan
        'Cooling': '#9467bd' # Purple
    }
    colors = [color_map.get(l, '#7f7f7f') for l in labels]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.6, # Donut style
        textinfo='label+percent',
        textposition='outside', # cleaner for thesis
        marker=dict(colors=colors, line=dict(color='#FFFFFF', width=2)),
        hovertemplate='<b>%{label}</b><br>Energy: %{value:,.2f} MWh<br>Share: %{percent}<extra></extra>'
    )])

    # Center text
    fig.add_annotation(
        text=f"<b>Total<br>{total_mwh:,.0f} MWh</b>",
        x=0.5, y=0.5,
        font=dict(size=14, family="Arial Black"),
        showarrow=False
    )

    fig.update_layout(
        title=dict(
            text=kwargs.get('title', "Plant Energy Consumption Breakdown"),
            x=0.5,
            xanchor='center'
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2, # Below chart
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=20, r=20, t=50, b=20),
        height=400, # Compact height
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
def plot_wind_power_production_timeline(df: pd.DataFrame, **kwargs) -> go.Figure:
    """
    Plot wind power production timeline (Grid Integration).
    
    Shows available, utilized, and curtailed wind power over time.
    """
    _check_dependencies()

    from h2_plant.visualization.utils import downsample_dataframe, get_time_axis_hours, get_viz_config

    maxpoints = kwargs.get('maxpoints', get_viz_config('performance.max_points_default', 2000))
    df_plot = downsample_dataframe(df, max_points=maxpoints)
    hours = get_time_axis_hours(df_plot)

    WIND_CAPACITY_MW = get_viz_config('plant_parameters.wind_capacity_mw', 20.0)

    wind_col = next((c for c in ['P_offer', 'P_renewable_mw', 'wind_power_mw'] if c in df_plot.columns), None)
    if not wind_col and 'wind_coefficient' in df_plot.columns:
        wind_available = df_plot['wind_coefficient'].values * WIND_CAPACITY_MW
    elif wind_col:
        wind_available = df_plot[wind_col].values
    else:
        return _empty_figure("No wind data (P_offer) found")

    pem_col = next((c for c in ['P_pem', 'pem_power_mw', 'P_pem_mw'] if c in df_plot.columns), None)
    soec_col = next((c for c in ['P_soec', 'soec_power_mw', 'P_soec_mw'] if c in df_plot.columns), None)

    pem_power = df_plot[pem_col].values if pem_col else np.zeros(len(wind_available))
    soec_power = df_plot[soec_col].values if soec_col else np.zeros(len(wind_available))

    if pem_power.mean() > 100: pem_power /= 1000.0
    if soec_power.mean() > 100: soec_power /= 1000.0

    total_used = pem_power + soec_power
    curtailment = np.maximum(0, wind_available - total_used)

    ScatterType = get_scatter_type(len(hours))
    fig = go.Figure()

    fig.add_trace(ScatterType(
        x=hours, y=wind_available,
        mode='lines', name='Available Wind Power',
        line=dict(color='#3498db', width=2),
        hovertemplate='Time: %{x:.1f}h<br>Available: %{y:.2f} MW<extra></extra>'
    ))

    fig.add_trace(ScatterType(
        x=hours, y=total_used,
        mode='lines', name='Utilized Wind Power',
        line=dict(color='#2ecc71', width=2),
        hovertemplate='Time: %{x:.1f}h<br>Utilized: %{y:.2f} MW<extra></extra>'
    ))

    fig.add_trace(ScatterType(
        x=hours, y=curtailment,
        mode='lines', name='Curtailment',
        line=dict(color='#e74c3c', width=1.5, dash='dash'),
        hovertemplate='Time: %{x:.1f}h<br>Curtailment: %{y:.2f} MW<extra></extra>'
    ))

    fig.update_layout(
        title=kwargs.get('title', 'Wind Power Production (Time Series)'),
        xaxis_title='Time (hours)',
        yaxis_title='Power (MW)',
        template='plotly_white',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )

    return fig


@log_graph_errors
def plot_wind_energy_cumulative(df: pd.DataFrame, **kwargs) -> go.Figure:
    """
    Plot cumulative wind energy (Grid Integration).
    
    Shows cumulative available, utilized, and curtailed wind energy.
    """
    _check_dependencies()

    from h2_plant.visualization.utils import downsample_dataframe, get_time_axis_hours, get_viz_config

    maxpoints = kwargs.get('maxpoints', get_viz_config('performance.max_points_default', 2000))
    df_plot = downsample_dataframe(df, max_points=maxpoints)
    hours = get_time_axis_hours(df_plot)

    WIND_CAPACITY_MW = get_viz_config('plant_parameters.wind_capacity_mw', 20.0)

    wind_col = next((c for c in ['P_offer', 'P_renewable_mw', 'wind_power_mw'] if c in df_plot.columns), None)
    if not wind_col and 'wind_coefficient' in df_plot.columns:
        wind_available = df_plot['wind_coefficient'].values * WIND_CAPACITY_MW
    elif wind_col:
        wind_available = df_plot[wind_col].values
    else:
        return _empty_figure("No wind data (P_offer) found")

    pem_col = next((c for c in ['P_pem', 'pem_power_mw', 'P_pem_mw'] if c in df_plot.columns), None)
    soec_col = next((c for c in ['P_soec', 'soec_power_mw', 'P_soec_mw'] if c in df_plot.columns), None)

    pem_power = df_plot[pem_col].values if pem_col else np.zeros(len(wind_available))
    soec_power = df_plot[soec_col].values if soec_col else np.zeros(len(wind_available))

    if pem_power.mean() > 100: pem_power /= 1000.0
    if soec_power.mean() > 100: soec_power /= 1000.0

    total_used = pem_power + soec_power
    curtailment = np.maximum(0, wind_available - total_used)

    if len(hours) > 1:
        dt_h = np.median(np.diff(hours))
    else:
        dt_h = df.attrs.get('dt_seconds', 60.0) / 3600.0
    if not np.isfinite(dt_h) or dt_h <= 0:
        dt_h = df.attrs.get('dt_seconds', 60.0) / 3600.0

    available_mwh = np.cumsum(wind_available * dt_h)
    used_mwh = np.cumsum(total_used * dt_h)
    curtailed_mwh = np.cumsum(curtailment * dt_h)

    ScatterType = get_scatter_type(len(hours))
    fig = go.Figure()

    fig.add_trace(ScatterType(
        x=hours, y=available_mwh,
        mode='lines', name='Available Wind Energy',
        line=dict(color='#3498db', width=2),
        hovertemplate='Time: %{x:.1f}h<br>Available: %{y:.1f} MWh<extra></extra>'
    ))

    fig.add_trace(ScatterType(
        x=hours, y=used_mwh,
        mode='lines', name='Utilized Wind Energy',
        line=dict(color='#2ecc71', width=2),
        hovertemplate='Time: %{x:.1f}h<br>Utilized: %{y:.1f} MWh<extra></extra>'
    ))

    fig.add_trace(ScatterType(
        x=hours, y=curtailed_mwh,
        mode='lines', name='Curtailed Wind Energy',
        line=dict(color='#e74c3c', width=1.5, dash='dash'),
        hovertemplate='Time: %{x:.1f}h<br>Curtailed: %{y:.1f} MWh<extra></extra>'
    ))

    fig.update_layout(
        title=kwargs.get('title', 'Cumulative Wind Energy'),
        xaxis_title='Time (hours)',
        yaxis_title='Energy (MWh)',
        template='plotly_white',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
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
def plot_cumulative_net_profit(df: pd.DataFrame, **kwargs) -> go.Figure:
    """
    Plot cumulative net profit over time after CAPEX and OPEX deduction.
    
    Net Profit = (Cumulative Revenue) - CAPEX - OPEX
    """
    _check_dependencies()
    
    from h2_plant.visualization.utils import (
        downsample_dataframe,
        get_time_axis_hours,
        get_viz_config,
        get_config_value,
        get_dt_hours
    )
    
    maxpoints = kwargs.get('maxpoints', get_viz_config('performance.max_points_default', 2000))
    df_plot = downsample_dataframe(df, max_points=maxpoints)
    
    if df_plot.empty:
        return _empty_figure("No data available")
    
    hours = get_time_axis_hours(df_plot)
    
    # H2 price (EUR/kg)
    h2_price = kwargs.get('h2_price_eur_kg')
    if h2_price is None:
        h2_price = get_config_value(df_plot, 'h2_price_eur_kg', None)
    if h2_price is None:
        h2_price = get_config_value(df_plot, 'h2_price_kg', None)
    if h2_price is None:
        h2_price = get_viz_config('plant_parameters.h2_price_eur_kg', 9.6)
    try:
        h2_price = float(h2_price)
    except (TypeError, ValueError):
        h2_price = 0.0
    
    def _is_rate_col(col_name: str) -> bool:
        col_lower = col_name.lower()
        return (
            col_lower.endswith('_kg_h') or
            'kg_h' in col_lower or
            'mass_flow_kg_h' in col_lower or
            'flow_kg_h' in col_lower
        )
    
    # --- Cumulative H2 Production ---
    cumulative_h2 = None
    cumulative_col = None
    for col in df_plot.columns:
        col_lower = col.lower()
        if col_lower in ['cumulative_h2_kg', 'cumulative_h2_total_kg', 'cumulative_h2_all_kg']:
            cumulative_col = col
            break
    if cumulative_col is None:
        for col in df_plot.columns:
            col_lower = col.lower()
            if col_lower.startswith('cumulative_h2') and 'kg' in col_lower and 'rfnbo' not in col_lower and 'non' not in col_lower:
                cumulative_col = col
                break
    
    if cumulative_col:
        cumulative_h2 = pd.to_numeric(df_plot[cumulative_col], errors='coerce').fillna(0).values
    else:
        # Fallback: integrate per-step mass from production columns
        dt_h = np.median(np.diff(hours)) if len(hours) > 1 else get_dt_hours(df_plot)
        if not np.isfinite(dt_h) or dt_h <= 0:
            dt_h = get_dt_hours(df_plot)
        
        col_map = {c.lower(): c for c in df_plot.columns}

        step_mass = np.zeros(len(df_plot))
        found = False

        # Priority 1: Purified H2 (PSA outlets)
        psa_cols = [
            c for c in df_plot.columns
            if 'psa' in c.lower() and 'h2' in c.lower() and
            ('mass_flow_kg_h' in c.lower() or 'flow_kg_h' in c.lower())
        ]
        if psa_cols:
            for col in psa_cols:
                data = pd.to_numeric(df_plot[col], errors='coerce').fillna(0).values
                step_mass += data * dt_h
            found = True

        # Priority 2: Total H2 column if PSA not available
        if not found:
            total_col = None
            for key in ['h2_kg', 'h2_total_kg', 'total_h2_kg']:
                if key in col_map:
                    total_col = col_map[key]
                    break
            if total_col:
                step_mass = pd.to_numeric(df_plot[total_col], errors='coerce').fillna(0).values
                if _is_rate_col(total_col):
                    step_mass = step_mass * dt_h
                found = True

        # Priority 3: Sum of source production if PSA/total not available
        if not found:
            source_keys = [
                'h2_pem_kg', 'h2_soec_kg', 'h2_atr_kg',
                'h2_pem', 'h2_soec', 'h2_atr',
                'h2_pem_kg_h', 'h2_soec_kg_h', 'h2_atr_kg_h'
            ]
            
            for key in source_keys:
                col = col_map.get(key)
                if col:
                    data = pd.to_numeric(df_plot[col], errors='coerce').fillna(0).values
                    if _is_rate_col(col):
                        data = data * dt_h
                    step_mass += data
                    found = True
        
        if not found:
            return _empty_figure("No purified H2 data found (PSA outlets missing).")
        
        cumulative_h2 = np.cumsum(step_mass)
    
    purification_yield = kwargs.get('purification_yield', 1.0)
    try:
        purification_yield = float(purification_yield)
    except (TypeError, ValueError):
        purification_yield = 1.0
    if purification_yield > 0 and purification_yield != 1.0:
        cumulative_h2 = cumulative_h2 * purification_yield

    cumulative_h2_value = cumulative_h2 * h2_price

    # --- Cumulative Electricity-Sale Revenue ---
    cumulative_grid_revenue = None
    grid_cumulative_col = next(
        (
            c for c in df_plot.columns
            if c.lower() in ['cumulative_grid_revenue_eur', 'cumulative_electricity_revenue_eur']
        ),
        None
    )
    if grid_cumulative_col:
        cumulative_grid_revenue = pd.to_numeric(df_plot[grid_cumulative_col], errors='coerce').fillna(0).values
    else:
        sold_col = next(
            (c for c in ['P_sold', 'sell_power_mw', 'coordinator_sell_power_mw'] if c in df_plot.columns),
            None
        )
        price_col = next((c for c in ['spot_price', 'Spot'] if c in df_plot.columns), None)
        if sold_col and price_col:
            # Use local timestep integration if cumulative revenue was not precomputed.
            if len(hours) > 1:
                dt_h_fallback = np.median(np.diff(hours))
            else:
                dt_h_fallback = get_dt_hours(df_plot)
            if not np.isfinite(dt_h_fallback) or dt_h_fallback <= 0:
                dt_h_fallback = get_dt_hours(df_plot)
            dt_h_steps = np.full(len(hours), dt_h_fallback, dtype=float)
            if len(hours) > 1:
                dt_h_steps[1:] = np.diff(hours)
                invalid_dt = ~np.isfinite(dt_h_steps) | (dt_h_steps <= 0)
                if invalid_dt.any():
                    dt_h_steps[invalid_dt] = dt_h_fallback
            sold_mw = pd.to_numeric(df_plot[sold_col], errors='coerce').fillna(0).values
            sold_mw = np.clip(sold_mw, a_min=0.0, a_max=None)
            spot_price = pd.to_numeric(df_plot[price_col], errors='coerce').fillna(0).values
            cumulative_grid_revenue = np.cumsum(sold_mw * spot_price * dt_h_steps)
        else:
            if sold_col is None:
                logger.warning("Electricity-sale power column missing. Electricity revenue set to zero.")
            if price_col is None:
                logger.warning("Electricity-sale price column missing. Electricity revenue set to zero.")
            cumulative_grid_revenue = np.zeros(len(df_plot), dtype=float)

    cumulative_total_revenue = cumulative_h2_value + cumulative_grid_revenue
    
    # --- CAPEX / OPEX Extraction ---
    metrics = df.attrs.get('metrics', {})
    config = df.attrs.get('config', {})
    metrics_lc = {str(k).lower(): v for k, v in metrics.items()}
    config_lc = {str(k).lower(): v for k, v in config.items()}
    kwargs_lc = {str(k).lower(): v for k, v in kwargs.items()}
    col_map = {c.lower(): c for c in df_plot.columns}
    
    def _resolve_cost(keys: List[str]) -> Optional[float]:
        # kwargs (case-insensitive)
        for key in keys:
            if key in kwargs_lc and kwargs_lc[key] is not None:
                return float(kwargs_lc[key])
        # df.attrs['metrics']
        for key in keys:
            if key in metrics_lc and metrics_lc[key] is not None:
                return float(metrics_lc[key])
        # df.attrs['config']
        for key in keys:
            if key in config_lc and config_lc[key] is not None:
                return float(config_lc[key])
        # direct column match
        for key in keys:
            col = col_map.get(key)
            if col:
                series = pd.to_numeric(df_plot[col], errors='coerce')
                if series.notna().any():
                    return float(series.dropna().iloc[-1])
        # substring match fallback
        for col_lower, col in col_map.items():
            if 'per_kg' in col_lower or 'perkg' in col_lower:
                continue
            if any(key in col_lower for key in keys):
                series = pd.to_numeric(df_plot[col], errors='coerce')
                if series.notna().any():
                    return float(series.dropna().iloc[-1])
        return None
    
    capex_keys = [
        'capex', 'capex_total', 'total_capex', 'total_c_bm',
        'total_installed_cost', 'fixed_capital_investment', 'fci'
    ]
    opex_keys = [
        'opex', 'opex_total', 'total_opex', 'opex_cost',
        'annual_opex', 'opex_annual', 'total_annual_opex'
    ]
    
    capex = _resolve_cost(capex_keys)
    opex = _resolve_cost(opex_keys)
    
    missing = []
    if capex is None:
        missing.append('CAPEX')
        capex = 0.0
    if opex is None:
        missing.append('OPEX')
        opex = 0.0
    
    if len(missing) == 2:
        return _empty_figure("CAPEX/OPEX values not available. Provide economics data (attrs or columns).")
    
    # --- Lifecycle-aware OPEX and yearly bars ---
    stack_threshold = kwargs.get('stack_power_threshold_mw', 0.01)
    pem_lifecycle_h = kwargs.get('pem_lifecycle_h')
    soec_lifecycle_h = kwargs.get('soec_lifecycle_h')
    pem_reserve_pct = kwargs.get('pem_reserve_pct')
    soec_reserve_pct = kwargs.get('soec_reserve_pct')

    try:
        stack_threshold = float(stack_threshold)
    except (TypeError, ValueError):
        stack_threshold = 0.01

    def _to_float(val: Optional[float]) -> Optional[float]:
        if val is None:
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    pem_lifecycle_h = _to_float(pem_lifecycle_h)
    soec_lifecycle_h = _to_float(soec_lifecycle_h)
    pem_reserve_pct = _to_float(pem_reserve_pct)
    soec_reserve_pct = _to_float(soec_reserve_pct)

    annual_reserve_pem = capex * pem_reserve_pct if pem_reserve_pct is not None else 0.0
    annual_reserve_soec = capex * soec_reserve_pct if soec_reserve_pct is not None else 0.0
    base_opex = opex - annual_reserve_pem - annual_reserve_soec
    if base_opex < 0:
        logger.warning("Annual OPEX is lower than reserve totals; base OPEX set to 0.")
        base_opex = 0.0

    if len(hours) > 1:
        dt_h = np.median(np.diff(hours))
    else:
        dt_h = get_dt_hours(df_plot)
    if not np.isfinite(dt_h) or dt_h <= 0:
        dt_h = get_dt_hours(df_plot)

    year_idx = np.floor(hours / 8760.0).astype(int)
    n_years = int(year_idx.max()) + 1 if len(year_idx) > 0 else 1
    years = np.arange(n_years)

    # Yearly revenues (delta cumulative values)
    year_series = pd.Series(year_idx)
    h2_value_series = pd.Series(cumulative_h2_value)
    grid_value_series = pd.Series(cumulative_grid_revenue)
    h2_last_by_year = h2_value_series.groupby(year_series).last().reindex(years, fill_value=0.0)
    grid_last_by_year = grid_value_series.groupby(year_series).last().reindex(years, fill_value=0.0)
    h2_revenue_per_year = h2_last_by_year.diff().fillna(h2_last_by_year).values
    grid_revenue_per_year = grid_last_by_year.diff().fillna(grid_last_by_year).values

    # Lifecycle event costs split by source (PEM/SOEC)
    event_costs_time_pem = np.zeros(len(hours))
    event_costs_time_soec = np.zeros(len(hours))
    event_costs_year_pem = np.zeros(n_years)
    event_costs_year_soec = np.zeros(n_years)

    def _apply_events(
        power_col: Optional[str],
        lifecycle_h: Optional[float],
        cost_per_event: float,
        event_costs_time_out: np.ndarray,
        event_costs_year_out: np.ndarray,
    ) -> None:
        if power_col is None or lifecycle_h is None or lifecycle_h <= 0 or cost_per_event <= 0:
            return
        power = pd.to_numeric(df_plot[power_col], errors='coerce').fillna(0).values
        active = power > stack_threshold
        cum_hours = np.cumsum(active.astype(float) * dt_h)
        if len(cum_hours) == 0:
            return
        events = np.floor(cum_hours / lifecycle_h).astype(int)
        delta = np.diff(events, prepend=0)
        if not np.any(delta > 0):
            return
        event_costs_time_out[:] += delta * cost_per_event
        for idx, count in enumerate(delta):
            if count > 0:
                year = year_idx[idx]
                if 0 <= year < n_years:
                    event_costs_year_out[year] += count * cost_per_event

    pem_col = next((c for c in ['P_pem', 'P_pem_mw'] if c in df_plot.columns), None)
    soec_col = next((c for c in ['P_soec_actual', 'P_soec', 'P_soec_mw'] if c in df_plot.columns), None)

    if pem_col is None:
        logger.warning("PEM power column missing; PEM lifecycle spikes skipped.")
    if soec_col is None:
        logger.warning("SOEC power column missing; SOEC lifecycle spikes skipped.")

    pem_event_cost = annual_reserve_pem * (pem_lifecycle_h / 8760.0) if pem_lifecycle_h else 0.0
    soec_event_cost = annual_reserve_soec * (soec_lifecycle_h / 8760.0) if soec_lifecycle_h else 0.0

    _apply_events(
        pem_col,
        pem_lifecycle_h,
        pem_event_cost,
        event_costs_time_pem,
        event_costs_year_pem,
    )
    _apply_events(
        soec_col,
        soec_lifecycle_h,
        soec_event_cost,
        event_costs_time_soec,
        event_costs_year_soec,
    )

    event_costs_time_total = event_costs_time_pem + event_costs_time_soec
    base_opex_per_year = np.full(n_years, base_opex, dtype=float)
    pem_spike_per_year = event_costs_year_pem
    soec_spike_per_year = event_costs_year_soec

    capex_per_year = np.zeros(n_years)
    if n_years > 0:
        capex_per_year[0] = capex

    cumulative_opex_time = base_opex * (hours / 8760.0) + np.cumsum(event_costs_time_total)
    net_profit = cumulative_total_revenue - capex - cumulative_opex_time

    # --- Plot ---
    profit_color = get_viz_config('styling.colors.profit', '#2c3e50')
    value_color = get_viz_config('styling.colors.h2total', '#2ecc71')

    ScatterType = get_scatter_type(len(hours))
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.08,
        row_heights=[0.65, 0.35],
        subplot_titles=(
            "Cumulative Net Profit (Time-Based OPEX)",
            "Annual Revenue Sources vs Cost Sources"
        )
    )

    # Net Profit (Primary)
    fig.add_trace(ScatterType(
        x=hours,
        y=net_profit,
        mode='lines',
        name='Net Profit (CAPEX + OPEX Deducted)',
        line=dict(color=profit_color, width=2),
        hovertemplate='Net Profit: %{y:,.0f}<extra></extra>'
    ), row=1, col=1)

    # Cumulative Total Revenue (Reference)
    fig.add_trace(ScatterType(
        x=hours,
        y=cumulative_total_revenue,
        mode='lines',
        name='Cumulative Total Revenue',
        line=dict(color=value_color, width=1.5, dash='dot'),
        hovertemplate='Total Revenue: %{y:,.0f}<extra></extra>'
    ), row=1, col=1)

    # Break-even line
    fig.add_hline(y=0, line=dict(color='#7f8c8d', width=1, dash='dash'), row=1, col=1)

    # Bottom bar panel
    x_years = years + 1
    fig.add_trace(go.Bar(
        x=x_years,
        y=h2_revenue_per_year,
        name='H2 Revenue',
        marker_color='#2ecc71'
    ), row=2, col=1)

    fig.add_trace(go.Bar(
        x=x_years,
        y=grid_revenue_per_year,
        name='Electricity Sale Revenue',
        marker_color='#3498db'
    ), row=2, col=1)

    fig.add_trace(go.Bar(
        x=x_years,
        y=-base_opex_per_year,
        name='OPEX Base',
        marker_color='#e74c3c'
    ), row=2, col=1)

    fig.add_trace(go.Bar(
        x=x_years,
        y=-pem_spike_per_year,
        name='PEM Replacement',
        marker_color='#d35400'
    ), row=2, col=1)

    fig.add_trace(go.Bar(
        x=x_years,
        y=-soec_spike_per_year,
        name='SOEC Replacement',
        marker_color='#c0392b'
    ), row=2, col=1)

    fig.add_trace(go.Bar(
        x=x_years,
        y=-capex_per_year,
        name='CAPEX (Year 1)',
        marker_color='#922b21'
    ), row=2, col=1)

    # Cost annotation
    capex_label = f"{capex:,.0f}" if 'CAPEX' not in missing else "n/a (0)"
    opex_label = f"{opex:,.0f}" if 'OPEX' not in missing else "n/a (0)"
    fig.add_annotation(
        text=f"CAPEX: {capex_label} | Annual OPEX (Total): {opex_label}",
        xref="paper", yref="paper",
        x=0.01, y=1.08,
        showarrow=False,
        align="left",
        font=dict(size=10, color="#555")
    )

    fig.update_layout(
        title=kwargs.get('title', 'Cumulative Net Profit (Revenue - CAPEX - OPEX)'),
        template='plotly_white',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        barmode='relative',
        height=720
    )

    fig.update_xaxes(title_text='Time (hours)', row=1, col=1)
    fig.update_yaxes(title_text='Value (EUR)', row=1, col=1)
    fig.update_xaxes(title_text='Year', row=2, col=1)
    fig.update_yaxes(title_text='Annual Value (EUR)', row=2, col=1)

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

    def _find_col(names):
        for name in names:
            for col in df_plot.columns:
                if col.lower() == name.lower():
                    return col
        return None

    p_offer_col = _find_col(['P_offer', 'p_offer'])
    p_sold_col = _find_col(['P_sold', 'P_sold_mw', 'p_sold'])
    sell_decision_col = _find_col(['sell_decision'])
    spot_purchased_col = _find_col(['spot_purchased_mw'])

    p_offer = pd.to_numeric(df_plot[p_offer_col], errors='coerce').fillna(0.0).values if p_offer_col else None
    p_sold = pd.to_numeric(df_plot[p_sold_col], errors='coerce').fillna(0.0).values if p_sold_col else None
    sell_decision = pd.to_numeric(df_plot[sell_decision_col], errors='coerce').fillna(0.0).values if sell_decision_col else None
    spot_purchased = pd.to_numeric(df_plot[spot_purchased_col], errors='coerce').fillna(0.0).values if spot_purchased_col else None

    tol = 1e-6

    # Dispatch-aware classification:
    #   Green: Plant at max capacity + selling excess (sell_decision=1, PEM active)
    #   Yellow: No surplus sold — plant consumes all power (sell_decision=0)
    #   Red: Selling because energy is more valuable than H2 (sell_decision=1, PEM off)
    p_pem_col = _find_col(['P_pem', 'p_pem'])
    p_pem_vals = pd.to_numeric(df_plot[p_pem_col], errors='coerce').fillna(0.0).values if p_pem_col else None

    green_mask = np.zeros(len(hours), dtype=bool)
    yellow_mask = np.zeros(len(hours), dtype=bool)
    red_mask = np.zeros(len(hours), dtype=bool)

    if sell_decision is not None and p_pem_vals is not None:
        sell_flag = sell_decision > 0
        green_mask = sell_flag & (p_pem_vals > tol)
        red_mask = sell_flag & (p_pem_vals <= tol)
        yellow_mask = ~sell_flag
    elif sell_decision is not None:
        sell_flag = sell_decision > 0
        green_mask = sell_flag
        yellow_mask = ~sell_flag
    elif p_sold is not None:
        green_mask = p_sold > tol
        yellow_mask = ~green_mask
    else:
        yellow_mask = np.ones(len(hours), dtype=bool)

    # Dilate sparse masks by 1 point so segments connect to adjacent lines
    for mask in [yellow_mask, red_mask]:
        if np.any(mask):
            dilated = mask.copy()
            dilated[1:] |= mask[:-1]    # extend right
            dilated[:-1] |= mask[1:]    # extend left
            mask[:] = dilated

    fig = go.Figure()

    # Effective PPA line with dispatch-aware coloring
    if np.any(green_mask):
        fig.add_trace(get_scatter_type(len(hours))(
            x=hours,
            y=np.where(green_mask, ppa_price, np.nan),
            mode='lines',
            name='Max Production + Selling Excess',
            line=dict(color='#2ecc71', width=2)
        ))

    if np.any(yellow_mask):
        fig.add_trace(get_scatter_type(len(hours))(
            x=hours,
            y=np.where(yellow_mask, ppa_price, np.nan),
            mode='lines+markers',
            name='Full Consumption (No Surplus)',
            line=dict(color="#fff344", width=3),
            marker=dict(size=6, color='#fff344')
        ))

    if np.any(red_mask):
        fig.add_trace(get_scatter_type(len(hours))(
            x=hours,
            y=np.where(red_mask, ppa_price, np.nan),
            mode='lines+markers',
            name='Selling (Energy > H2 Value)',
            line=dict(color='#e74c3c', width=3),
            marker=dict(size=6, color='#e74c3c')
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
def plot_renewable_grid_ppa_panels(df: pd.DataFrame, **kwargs) -> go.Figure:
    """
    Plot Renewable & Grid Energy + Effective PPA (3-Panel).
    
    Panels:
    1) Wind availability with guaranteed line
    2) Renewable sales vs Non-RFNBO grid purchases
    3) Effective PPA (with optional spot price overlay)
    """
    _check_dependencies()
    
    from h2_plant.visualization.utils import downsample_dataframe, get_time_axis_hours, get_viz_config
    from plotly.subplots import make_subplots
    
    maxpoints = kwargs.get('maxpoints', get_viz_config('performance.max_points_default', 2000))
    df_plot = downsample_dataframe(df, max_points=maxpoints)
    hours = get_time_axis_hours(df_plot)
    
    WIND_CAPACITY_MW = get_viz_config('plant_parameters.wind_capacity_mw', 20.0)
    
    wind_col = next((c for c in ['P_offer', 'P_renewable_mw', 'wind_power_mw'] if c in df_plot.columns), None)
    if not wind_col and 'wind_coefficient' in df_plot.columns:
        wind_available = df_plot['wind_coefficient'].values * WIND_CAPACITY_MW
    elif wind_col:
        wind_available = df_plot[wind_col].values
    else:
        return _empty_figure("No wind data (P_offer) found")
    
    # Guaranteed power line (try config, attrs, or column)
    config = df.attrs.get('config', {})
    economics = df.attrs.get('economics', {})
    guaranteed_mw = (
        config.get('guaranteed_power_mw') or
        config.get('guaranteed_mw') or
        economics.get('guaranteed_power_mw') or
        economics.get('guaranteed_mw') or
        get_viz_config('plant_parameters.guaranteed_power_mw', None) or
        get_viz_config('ppa_parameters.guaranteed_power_mw', None)
    )
    if guaranteed_mw is None:
        guar_col = next((c for c in ['guaranteed_mw', 'guaranteed_power_mw'] if c in df_plot.columns), None)
        if guar_col:
            guaranteed_mw = pd.to_numeric(df_plot[guar_col], errors='coerce').mean()
    
    # Renewable sales vs grid purchases
    sell_col = next((c for c in ['P_sold', 'sell_power_mw', 'coordinator_sell_power_mw'] if c in df_plot.columns), None)
    spot_purchased_col = next((c for c in ['spot_purchased_mw'] if c in df_plot.columns), None)
    
    renewable_sales = pd.to_numeric(df_plot[sell_col], errors='coerce').fillna(0.0).values if sell_col else None
    spot_purchased = pd.to_numeric(df_plot[spot_purchased_col], errors='coerce').fillna(0.0).values if spot_purchased_col else None
    
    # Effective PPA
    ppa_col = next((c for c in ['ppa_price_effective_eur_mwh', 'effective_ppa_price'] if c in df_plot.columns), None)
    spot_col = next((c for c in ['spot_price', 'Spot'] if c in df_plot.columns), None)
    
    ppa_price = df_plot[ppa_col].values if ppa_col else None
    spot_price = df_plot[spot_col].values if spot_col else None
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            'Wind Availability',
            'Renewable Sales vs Grid Purchases',
            'Effective PPA Price'
        )
    )
    
    ScatterType = get_scatter_type(len(hours))
    
    # Panel 1: Wind availability + guaranteed line
    fig.add_trace(ScatterType(
        x=hours, y=wind_available,
        mode='lines', name='Available Wind Power',
        line=dict(color='#3498db', width=2),
        hovertemplate='Time: %{x:.1f}h<br>Wind: %{y:.2f} MW<extra></extra>'
    ), row=1, col=1)
    
    if guaranteed_mw is not None and guaranteed_mw > 0:
        fig.add_hline(
            y=guaranteed_mw,
            line=dict(color='#9b59b6', width=2, dash='dot'),
            annotation_text=f"Guaranteed ({guaranteed_mw:.1f} MW)",
            annotation_position="top right",
            row=1, col=1
        )
    
    # Panel 2: Renewable sales vs Non-RFNBO grid purchases
    if renewable_sales is not None:
        fig.add_trace(ScatterType(
            x=hours, y=renewable_sales,
            mode='lines', name='Renewable Sales',
            line=dict(color='#2ecc71', width=2),
            hovertemplate='Time: %{x:.1f}h<br>Sales: %{y:.2f} MW<extra></extra>'
        ), row=2, col=1)
    
    if spot_purchased is not None:
        fig.add_trace(ScatterType(
            x=hours, y=spot_purchased,
            mode='lines', name='Grid Purchases (Non-RFNBO)',
            line=dict(color='#e74c3c', width=2, dash='dash'),
            hovertemplate='Time: %{x:.1f}h<br>Purchase: %{y:.2f} MW<extra></extra>'
        ), row=2, col=1)
    
    # Panel 3: Effective PPA (with spot overlay if available)
    if ppa_price is not None:
        fig.add_trace(ScatterType(
            x=hours, y=ppa_price,
            mode='lines', name='Effective PPA',
            line=dict(color='#1976D2', width=2),
            hovertemplate='Time: %{x:.1f}h<br>PPA: %{y:.2f} EUR/MWh<extra></extra>'
        ), row=3, col=1)
    
    if spot_price is not None:
        fig.add_trace(ScatterType(
            x=hours, y=spot_price,
            mode='lines', name='Spot Price',
            line=dict(color='#7f8c8d', width=1.5, dash='dot'),
            hovertemplate='Time: %{x:.1f}h<br>Spot: %{y:.2f} EUR/MWh<extra></extra>'
        ), row=3, col=1)
    
    fig.update_layout(
        title=kwargs.get('title', 'Renewable & Grid Energy + Effective PPA'),
        template='plotly_white',
        height=900,
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    fig.update_yaxes(title_text='Power (MW)', row=1, col=1)
    fig.update_yaxes(title_text='Power (MW)', row=2, col=1)
    fig.update_yaxes(title_text='Price (EUR/MWh)', row=3, col=1)
    fig.update_xaxes(title_text='Time (hours)', row=3, col=1)
    
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
        'SOEC_H2_Coalescer',
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
    'H2_Storage': [
        'H2_Production_Mixer',
        'H2_Production_Cooler',
        'LP_Compressor_S1',
        'LP_Intercooler_1',
        'LP_Storage_Tank',
        'HP_Compressor_S2',
        'HP_Intercooler_2',
        'HP_Compressor_S3',
        'HP_Intercooler_3',
        'HP_Compressor_S4',
        'HP_Intercooler_4',
        'HP_Compressor_S5',
        'Truck_Station_1'
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
            # Priority: Standardized Total Mass > Legacy Bulk + Entrained
            total_flow_col = next((c for c in df.columns if c in [
                f'{cid}_outlet_total_mass_flow_kg_h'
            ]), None)

            flow_col = next((c for c in df.columns if c in [
                f'{cid}_outlet_mass_flow_kg_h', f'{cid}_mass_flow_kg_h'
            ]), None)
            
            entrained_col = next((c for c in df.columns if c in [
                f'{cid}_outlet_entrained_mass_kg_h', f'{cid}_entrained_mass_kg_h'
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
            active_col = total_flow_col if total_flow_col else flow_col
            if active_col and active_col in df.columns:
                df_active = df[df[active_col] > 1e-6]
                if df_active.empty:
                    # No active flow, fall back to full DF but expect zeros
                    df_active = df 
            else:
                df_active = df
            
            # 3. CALCULATE AVERAGES FROM ACTIVE DATA
            if total_flow_col:
                flow_val = df_active[total_flow_col].mean()
            else:
                bulk_flow = df_active[flow_col].mean() if flow_col else 0.0
                entrained_flow = df_active[entrained_col].mean() if entrained_col else 0.0
                flow_val = bulk_flow + entrained_flow # Total Mass Flow
            
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

            # --- H2O PPM (Calculate from Total Moles: Bulk + Entrained) ---
            h2o_ppm = 0.0
            
            # 1. Try Specific Total Mole Fraction Column (First Priority - Rigorous)
            h2o_molf_col = next((c for c in df.columns if c in [
                f'{cid}_outlet_H2O_molf', f'{cid}_outlet_h2o_molf'
            ]), None)
            
            if h2o_molf_col:
                 h2o_ppm = df_active[h2o_molf_col].mean() * 1e6
                 
            # 2. Fallback: Reconstruct from Mass Fractions (Bulk) + Entrained Mass
            else:
                # Vapor Mole Fraction Column
                vap_molf_col = next((c for c in df.columns if c in [
                    f'{cid}_outlet_H2O_molf', f'{cid}_outlet_h2o_molf', f'{cid}_y_H2O'
                ]), None)
    
                # Liquid Mole Fraction Column
                liq_molf_col = next((c for c in df.columns if c in [
                    f'{cid}_outlet_H2O_liq_molf', f'{cid}_outlet_h2o_liq_molf', f'{cid}_x_H2O'
                ]), None)
                
                if vap_molf_col or liq_molf_col:
                    s_vap = df_active[vap_molf_col] if vap_molf_col else pd.Series(0.0, index=df_active.index)
                    s_liq = df_active[liq_molf_col] if liq_molf_col else pd.Series(0.0, index=df_active.index)
                    
                    # If we have explicit liquid fraction column, assume it captures the entrained part OR 
                    # if entrained_col exists, we might need to add it?
                    # Component fix 1 (Total Molf) avoids this ambiguity. 
                    # If we are here, we lack the Total Molf column.
                    
                    s_total = s_vap + s_liq
                    # ... [existing logic] ...
                    valid_vals = s_total[s_total > 1e-9]
                    if not valid_vals.empty:
                        h2o_ppm = valid_vals.mean() * 1e6
                    else:
                        h2o_ppm = s_total.mean() * 1e6
    
                else:
                    # Mass Fraction Fallback
                    h2o_mass_col = next((c for c in df.columns if c in [
                        f'{cid}_outlet_h2o_frac', f'{cid}_mass_fraction_h2o'
                    ]), None)
                    
                    if h2o_mass_col:
                         w_h2o_bulk = df_active[h2o_mass_col].mean()
                         m_bulk = bulk_flow
                         m_entrained = entrained_flow
                         
                         m_h2o_total = (m_bulk * w_h2o_bulk) + m_entrained
                         m_total = m_bulk + m_entrained
                         
                         # Simplified PPM calc (assume MW_bulk approx MW_target_gas for small impurities)
                         if m_total > 0:
                             w_h2o_total = m_h2o_total / m_total
                             if w_h2o_total < 1.0:
                                 MW_c = MW_H2 if stream_type == 'H2' else MW_O2
                                 h2o_ppm = (w_h2o_total / MW_H2O) / ((1-w_h2o_total)/MW_c + w_h2o_total/MW_H2O) * 1e6
                             else:
                                 h2o_ppm = 1e6
            
            # Legacy/Redundant block removal (lines 3738-3784 replaced by above logic)


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
        'H2_Storage': extract_profile(_TRAIN_COMPONENTS.get('H2_Storage', []), 'H2'),
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
        'H2_Storage': '#9467bd', # Purple
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
        impurity_label = 'O2 ppm (mol)' if 'H2' in train_id else 'H2 ppm (mol)'
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
            name=f"{train_id} H2O ppm (mol)",
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
        dict(label="H2 Storage", method="update", args=[{"visible": get_visibility(['H2_Storage'])}]),
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
    fig.update_yaxes(title_text="Impurity / H2O (ppm mol)", type="log", row=3, col=1, secondary_y=True)
    
    # Rotate x-axis labels for readability
    fig.update_xaxes(tickangle=45, row=1, col=1)
    fig.update_xaxes(tickangle=45, row=2, col=1)
    fig.update_xaxes(tickangle=45, title_text="Component", row=3, col=1)
    
    return fig

# =============================================================================
# PHYSICS-BASED ANALYSIS GRAPHS (PEM)
# =============================================================================

def _calculate_pem_physics_curves(t_op_h: float = 0.0):
    """
    Internal helper to generate V-j curves based on physics model.
    Matches logic from pem_operator.py using shared constants.
    """
    from h2_plant.config.constants_physics import PEMConstants
    from h2_plant.models import pem_physics as phys
    
    CONST = PEMConstants()
    
    # Range: 0.01 to 95% of limit
    j_lim = CONST.j_lim
    j_range = np.linspace(0.01, j_lim * 0.95, 200)
    T = CONST.T_default
    P = CONST.P_op_default
    
    # 1. Reversible Voltage (Nernst)
    U_rev_val = phys.calculate_Urev(T, P)
    U_rev = np.full_like(j_range, U_rev_val)
    
    # 2. Activation Overpotential
    # eta_act = (R * T) / (alpha * z * F) * np.log(j / j0)
    eta_act = (CONST.R * T) / (CONST.alpha * CONST.z * CONST.F) * np.log(np.maximum(j_range, 1e-10) / CONST.j0)
    
    # 3. Ohmic Overpotential
    # eta_ohm = j * (delta_mem / sigma)
    eta_ohm = j_range * (CONST.delta_mem / CONST.sigma_base)
    
    # 4. Concentration Overpotential
    limit_term = np.maximum(1e-6, j_lim - j_range)
    eta_conc = (CONST.R * T) / (CONST.z * CONST.F) * np.log(j_lim / limit_term)
    
    # 5. Degradation
    # Mirroring DetailedPEMElectrolyzer._calculate_U_deg logic:
    t_table = np.array(CONST.DEGRADATION_YEARS) * 8760.0
    v_stack_table = np.array(CONST.DEGRADATION_V_STACK)
    v_cell_table = v_stack_table / CONST.N_cell_per_stack
    
    # BOL Reference (at nominal j)
    # Ensure BOL reference uses same consistent calculation
    V_BOL_NOM = phys.calculate_Vcell_base(CONST.j_nom, T, P)
    
    # Interpolate for current t_op_h
    # Apply reasonable cap for interpolation (10 years)
    t_interp = min(t_op_h, t_table[-1])
    V_cell_degraded = np.interp(t_interp, t_table, v_cell_table)
    
    U_deg_val = np.maximum(0.0, V_cell_degraded - V_BOL_NOM)
    
    U_deg = np.full_like(j_range, U_deg_val)
    
    # Total
    V_total = U_rev + eta_act + eta_ohm + eta_conc + U_deg
    
    return j_range, U_rev, eta_act, eta_ohm, eta_conc, V_total, U_deg, CONST

@log_graph_errors
def plot_physics_polarization(df: pd.DataFrame, **kwargs) -> go.Figure:
    """
    Generates polarization curve comparing BOL, EOL, and Current State.
    Updated with horizontal top-aligned legend and validation title.
    """
    _check_dependencies()
    from h2_plant.visualization.utils import get_viz_config
    
    # Determine current operating hours from dataframe
    t_op_current = 0.0
    if 'minute' in df.columns:
        t_op_current = df['minute'].max() / 60.0
    
    # Calculate curves
    j, U_rev, _, _, _, V_bol, _, CONST = _calculate_pem_physics_curves(t_op_h=0)
    _, _, _, _, _, V_eol, _, _ = _calculate_pem_physics_curves(t_op_h=87600) # 10 years
    _, _, _, _, _, V_curr, _, _ = _calculate_pem_physics_curves(t_op_h=t_op_current)
    
    # Calculate Reference Curve (Piecewise Theoretical)
    x_ref = np.asarray(j)
    V_ref = np.empty_like(x_ref, dtype=float)
    mask = x_ref <= 0.346
    V_ref[mask] = 0.266142 * (x_ref[mask] ** 0.390646) + 1.453864
    V_ref[~mask] = 0.194938 * (x_ref[~mask] ** 0.849221) + 1.553147
    
    # Calculate Difference (Current - Reference)
    V_diff = V_curr - V_ref
    
    fig = go.Figure()
    
    # Reversible Voltage Area
    fig.add_trace(go.Scatter(
        x=j, y=U_rev,
        mode='lines',
        name='Reversible Voltage',
        line=dict(color='blue', width=0),
        fill='tozeroy',
        fillcolor='rgba(0, 0, 255, 0.05)',
        hoverinfo='skip'
    ))
    
    # BOL
    fig.add_trace(go.Scatter(
        x=j, y=V_bol,
        mode='lines',
        name='BOL',
        line=dict(color='green', width=2, dash='dash'),
        hovertemplate='BOL: %{y:.2f} V<extra></extra>'
    ))
    
    # EOL
    fig.add_trace(go.Scatter(
        x=j, y=V_eol,
        mode='lines',
        name='EOL (10y)',
        line=dict(color='red', width=2, dash='dash'),
        hovertemplate='EOL: %{y:.2f} V<extra></extra>'
    ))
    
    # Current
    label_curr = f'Sim ({t_op_current/8760:.1f}y)'
    fig.add_trace(go.Scatter(
        x=j, y=V_curr,
        mode='lines',
        name=label_curr,
        line=dict(color='blue', width=3),
        hovertemplate='Current: %{y:.2f} V<extra></extra>'
    ))
    
    # Reference (Theoretical)
    fig.add_trace(go.Scatter(
        x=j, y=V_ref,
        mode='lines',
        name='Theoretical Ref',
        line=dict(color='gray', width=2, dash='dot'),
        hovertemplate='Ref: %{y:.2f} V<extra></extra>'
    ))
    
    # Difference (Secondary Y)
    fig.add_trace(go.Scatter(
        x=j, y=V_diff,
        mode='lines',
        name='Δ (Sim-Ref)',
        yaxis='y2',
        line=dict(color='purple', width=1),
        hovertemplate='ΔV: %{y:.3f} V<extra></extra>'
    ))

    # Nominal Point
    fig.add_vline(x=CONST.j_nom, line_dash="solid", line_color="black", annotation_text="Nominal")
    
    fig.update_layout(
        title=kwargs.get('title', 'PEM Physics Validation: Polarization Curve'),
        xaxis_title='Current Density (A/cm²)',
        yaxis_title='Voltage (V)',
        yaxis2=dict(
            title='Δ Voltage (V)',
            overlaying='y',
            side='right',
            showgrid=False,
            zeroline=True,
            zerolinecolor='gray'
        ),
        template='plotly_white',
        hovermode='x unified',
        # Move legend to top, side-by-side
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        margin=dict(t=100) # Ensure title and legend don't overlap
    )
    
    return fig


@log_graph_errors
def plot_physics_efficiency(df: pd.DataFrame, **kwargs) -> go.Figure:
    """Generates SYSTEM efficiency curve vs Current Density."""
    _check_dependencies()
    
    # Calculate base curves
    j, _, _, _, _, V_total, _, CONST = _calculate_pem_physics_curves(t_op_h=0) # Use BOL for generalized curve
    
    # Power Calculations
    # I = j * Area
    Area = CONST.Area_Total # cm2
    I_total = j * Area # Amps
    P_stack_W = I_total * V_total
    
    # System Power
    # P_bop_fixo is absolute Watts. k_bop_var is fraction.
    P_sys_W = P_stack_W + CONST.P_bop_fixo + (CONST.k_bop_var * P_stack_W)
    
    # Hydrogen Energy Output (LHV)
    from h2_plant.models.pem_physics import calculate_eta_F
    eta_F = calculate_eta_F(j)
    
    # Molar flow mol/s = (I / z F) * eta_F
    mol_s = (I_total * eta_F) / (CONST.z * CONST.F)
    mass_s = mol_s * CONST.MH2
    energy_out_W = mass_s * (CONST.LHVH2_kWh_kg * 3.6e6) # kWh/kg -> J/kg -> W
    
    # Efficiency
    sys_eff = np.divide(energy_out_W, P_sys_W, out=np.zeros_like(P_sys_W), where=P_sys_W!=0) * 100.0
    
    # Stack-only Efficiency
    # Calculate roughly for comparison (using Voltage Efficiency concept)
    # Or rigorously: P_h2 / P_stack
    stack_eff = np.divide(energy_out_W, P_stack_W, out=np.zeros_like(P_stack_W), where=P_stack_W!=0) * 100.0
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=j, y=sys_eff,
        mode='lines',
        name='System Efficiency (Stack + BoP)',
        line=dict(color='green', width=3),
        hovertemplate='System: %{y:.1f}%<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=j, y=stack_eff,
        mode='lines',
        name='Stack Only Efficiency',
        line=dict(color='gray', width=2, dash='dot'),
        hovertemplate='Stack: %{y:.1f}%<extra></extra>'
    ))
    
    fig.add_vline(x=CONST.j_nom, line_dash="solid", line_color="black", annotation_text="Nominal")
    
    fig.update_layout(
        title=kwargs.get('title', 'PEM System Efficiency vs Current Density'),
        xaxis_title='Current Density (A/cm²)',
        yaxis_title='Efficiency (% LHV)',
        template='plotly_white',
        hovermode='x unified',
        yaxis=dict(range=[0, 90])
    )
    
    return fig


@log_graph_errors
def plot_physics_power_balance(df: pd.DataFrame, **kwargs) -> go.Figure:
    """Generates Power Balance: Stack vs BoP vs Total."""
    _check_dependencies()
    
    j, _, _, _, _, V_total, _, CONST = _calculate_pem_physics_curves(t_op_h=0)
    
    Area = CONST.Area_Total
    I_total = j * Area
    P_stack_W = I_total * V_total
    
    # BoP
    P_bop_var_W = CONST.k_bop_var * P_stack_W
    P_bop_fix_W = np.full_like(P_stack_W, CONST.P_bop_fixo)
    
    P_total_W = P_stack_W + P_bop_fix_W + P_bop_var_W
    
    # Convert to kW
    P_stack_kW = P_stack_W / 1000.0
    P_total_kW = P_total_W / 1000.0
    
    fig = go.Figure()
    
    # Stack Power (Filled Area)
    fig.add_trace(go.Scatter(
        x=j, y=P_stack_kW,
        mode='lines',
        name='Stack Power',
        stackgroup='one',
        line=dict(color='#1f77b4', width=0),
        fillcolor='rgba(31, 119, 180, 0.6)'
    ))
    
    # BoP Power (Stacked)
    # Calculated as remainder for visual stacking
    BoP_kW = (P_total_kW - P_stack_kW)
    
    fig.add_trace(go.Scatter(
        x=j, y=BoP_kW,
        mode='lines',
        name='BoP Losses',
        stackgroup='one',
        line=dict(color='gray', width=0),
        fillcolor='rgba(128, 128, 128, 0.4)'
    ))
    
    # Total Line Overlay
    fig.add_trace(go.Scatter(
        x=j, y=P_total_kW,
        mode='lines',
        name='Total System Power',
        line=dict(color='darkred', width=2),
        hovertemplate='Total: %{y:.0f} kW<extra></extra>'
    ))
    
    fig.add_vline(x=CONST.j_nom, line_dash="solid", line_color="black", annotation_text="Nominal")
    
    fig.update_layout(
        title=kwargs.get('title', 'PEM Power Balance: Stack vs BoP'),
        xaxis_title='Current Density (A/cm²)',
        yaxis_title='Power (kW)',
        template='plotly_white',
        hovermode='x unified',
         legend=dict(x=0.02, y=0.98)
    )
    
    return fig
@log_graph_errors
def plot_all_efficiencies(df: pd.DataFrame, **kwargs) -> go.Figure:
    """
    Plot all individual subsystem efficiencies and Plant Global Efficiency.
    
    FIXES:
    - PEM: Line continuity fixed (Idle periods show 0% instead of gaps).
    - ATR: Chemical Efficiency clamped to theoretical limit (40%) to remove dynamic artifacts.
    """
    _check_dependencies()
    from h2_plant.visualization.utils import downsample_dataframe, get_time_axis_hours, get_viz_config
    
    # 1. Prepare Data
    maxpoints = kwargs.get('maxpoints', get_viz_config('performance.max_points_default', 2000))
    df_plot = downsample_dataframe(df, max_points=maxpoints)
    hours = get_time_axis_hours(df_plot)
    dt_seconds = df.attrs.get('dt_seconds', 60.0)
    dt_hours = dt_seconds / 3600.0
    
    # Constants
    LHV_H2 = 0.03333    # MWh/kg
    LHV_CH4 = 0.0139    # MWh/kg (Pure Methane)
    BIOGAS_CH4_FRAC = 0.60 
    
    # LIMITS
    ATR_CHEM_LIMIT = 40.1  # Hard theoretical limit for Biogas-> H2 chemical conversion

    fig = go.Figure()

    # --- HELPERS ---
    def get_col_val(exact_name: str, fallback_pattern: str = None):
        if exact_name in df_plot.columns:
            return df_plot[exact_name].values, exact_name
        if fallback_pattern:
            col = next((c for c in df_plot.columns if fallback_pattern.lower() in c.lower()), None)
            if col: return df_plot[col].values, col
        return np.zeros(len(df_plot)), None

    def to_rate_kg_h(mass_array_kg):
        return np.divide(mass_array_kg, dt_hours, out=np.zeros_like(mass_array_kg), where=dt_hours>0)

    # --- 1. DATA GATHERING ---

    # A. Biogas / Methane Input
    bio_vals, bio_col = get_col_val('Biogas_Compressor_1_outlet_mass_flow_kg_h')
    if not bio_col: bio_vals, bio_col = get_col_val('Fm_bio_func')
    
    biogas_flow_kg_h = bio_vals 
    methane_flow_kg_h = biogas_flow_kg_h * BIOGAS_CH4_FRAC
    biogas_mw = methane_flow_kg_h * LHV_CH4

    # B. ATR H2 Output
    h2_atr_vals, h2_atr_col = get_col_val('ATR_PSA_1_outlet_mass_flow_kg_h')
    if not h2_atr_col: h2_atr_vals, h2_atr_col = get_col_val('PSA', 'ATR_PSA')
    h2_atr_rate = h2_atr_vals
    
    # C. Recovered Heat
    q_recovered_kw = np.zeros(len(df_plot))
    recovery_ids = ['H01', 'H02', 'H04', 'Syngas_Cooler', 'Boiler']
    explicit_found = False
    syngas_cooler_duty = np.zeros(len(df_plot))
    
    for tag in recovery_ids:
        cols = [c for c in df_plot.columns if tag in c]
        for col in cols:
            if any(x in col.lower() for x in ['duty', 'q_transferred', 'heat_flow']):
                if 'electric' in col.lower(): continue
                vals = np.abs(df_plot[col].values)
                q_recovered_kw += vals
                if 'Syngas_Cooler' in tag: syngas_cooler_duty += vals
                if any(x in tag for x in ['H01', 'H02', 'H04']): explicit_found = True

    if not explicit_found and np.mean(syngas_cooler_duty) > 1.0:
        q_recovered_kw += syngas_cooler_duty * 3.0 
    
    q_atr_mw = q_recovered_kw / 1000.0

    # D. Work Input
    atr_power_cols = [c for c in df_plot.columns if 'ATR' in c and ('power' in c or 'kw' in c) 
                      and 'q_' not in c and 'heat' not in c and 'duty' not in c
                      and 'Boiler' not in c and 'Cooler' not in c and 'Interchanger' not in c]
    atr_work_mw = np.zeros(len(df_plot))
    for c in atr_power_cols: atr_work_mw += (df_plot[c].values / 1000.0)

    # --- 2. PLOTTING ---

    # PEM
    if 'H2_pem_kg' in df_plot.columns:
        p_pem, _ = get_col_val('P_pem_actual', 'P_pem')
        h2_rate = to_rate_kg_h(df_plot['H2_pem_kg'].values)
        with np.errstate(divide='ignore', invalid='ignore'):
            eff = (h2_rate * LHV_H2) / p_pem * 100
            # FIX 1: Set idle periods (Power < 0.01 MW) to 0.0 instead of NaN to prevent gaps
            eff = np.where(p_pem > 0.01, eff, 0.0) 
            
        fig.add_trace(get_scatter_type(len(hours))(
            x=hours, y=np.clip(eff, 0, 100), mode='lines', name='PEM System',
            line=dict(color='#1f77b4', width=2),
            hovertemplate='Time: %{x:.1f}h<br>PEM: %{y:.1f}%<extra></extra>'
        ))

    # SOEC
    if 'H2_soec_kg' in df_plot.columns:
        p_soec, _ = get_col_val('P_soec_actual', 'P_soec')
        h2_rate = to_rate_kg_h(df_plot['H2_soec_kg'].values)
        with np.errstate(divide='ignore', invalid='ignore'):
            eff = (h2_rate * LHV_H2) / p_soec * 100
            # FIX 1: Apply same fix to SOEC for consistency
            eff = np.where(p_soec > 0.01, eff, 0.0)
            
        fig.add_trace(get_scatter_type(len(hours))(
            x=hours, y=np.clip(eff, 0, 110), mode='lines', name='SOEC System',
            line=dict(color='#ff7f0e', width=2),
            hovertemplate='Time: %{x:.1f}h<br>SOEC: %{y:.1f}%<extra></extra>'
        ))

    # ATR Chemical
    if np.max(biogas_mw) > 0.01:
        with np.errstate(divide='ignore', invalid='ignore'):
            atr_chem_eff = (h2_atr_rate * LHV_H2) / biogas_mw * 100
            # FIX 1: Ensure continuity at zero input
            atr_chem_eff = np.where(biogas_mw > 0.01, atr_chem_eff, 0.0)
            
            # FIX 2: Theoretical Limiter (Hard Clamp at 40%)
            atr_chem_eff = np.minimum(atr_chem_eff, ATR_CHEM_LIMIT)
            
        fig.add_trace(get_scatter_type(len(hours))(
            x=hours, y=np.clip(atr_chem_eff, 0, 100), mode='lines', name='ATR Chemical',
            line=dict(color='#2ca02c', width=2, dash='dot'),
            hovertemplate='Time: %{x:.1f}h<br>ATR Chem: %{y:.1f}%<extra></extra>'
        ))

    # ATR Global
    if np.max(biogas_mw) > 0.01:
        with np.errstate(divide='ignore', invalid='ignore'):
            useful_output = (h2_atr_rate * LHV_H2) + q_atr_mw
            total_input = biogas_mw + atr_work_mw
            
            atr_glob_eff = useful_output / total_input * 100
            atr_glob_eff = np.where(total_input > 0.1, atr_glob_eff, 0.0) # Continuity fix
            
            # Apply same logical limit to Global if desired, or let it float higher due to heat recovery
            # Usually global can exceed chemical, so we leave the clamp off or set it higher (e.g. 90%)
            
        fig.add_trace(get_scatter_type(len(hours))(
            x=hours, y=np.clip(atr_glob_eff, 0, 120), mode='lines', name='ATR Global',
            line=dict(color='#2ca02c', width=2), 
            hovertemplate='Time: %{x:.1f}h<br>ATR Global: %{y:.1f}%<br>(Est. Heat Recovery)<extra></extra>'
        ))

    # Plant Global
    p_soec, _ = get_col_val('P_soec_grid_mw', 'P_soec_actual')
    p_pem, _ = get_col_val('P_pem_grid_mw', 'P_pem_actual')
    p_stacks = p_soec + p_pem
    p_bop, _ = get_col_val('bop_grid_import_mw', 'P_bop_mw')
    
    p_grid_total = p_stacks + p_bop
    p_system_in = p_grid_total + biogas_mw
    
    h2_pem_gross = to_rate_kg_h(df_plot.get('H2_pem_kg', np.zeros(len(df_plot))).values)
    h2_soec_gross = to_rate_kg_h(df_plot.get('H2_soec_kg', np.zeros(len(df_plot))).values)
    total_h2_rate = h2_pem_gross + h2_soec_gross + h2_atr_rate

    with np.errstate(divide='ignore', invalid='ignore'):
        global_eff = (total_h2_rate * LHV_H2) / p_system_in * 100
        # FIX 1: Continuity
        global_eff = np.where(p_system_in > 0.01, global_eff, 0.0)

    fig.add_trace(get_scatter_type(len(hours))(
        x=hours, y=np.clip(global_eff, 0, 100), mode='lines', name='Plant Global',
        line=dict(color='black', width=3),
        hovertemplate='Time: %{x:.1f}h<br>Plant Global: %{y:.1f}%<extra></extra>'
    ))

    fig.update_layout(
        title=kwargs.get('title', 'Plant Efficiency Overview'),
        xaxis_title='Time (hours)',
        yaxis_title='Efficiency (% LHV)',
        template='plotly_white',
        hovermode='x unified',
        yaxis=dict(range=[0, 105]),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    return fig

@log_graph_errors
def plot_temporal_averages(df: pd.DataFrame, **kwargs) -> go.Figure:
    """
    Plot Temporal Averages (Hourly aggregated price, power, H2) - Interactive twin.
    
    Features:
    - Interactive Tabs (Hourly, Daily, Monthly, Yearly).
    - Capacity Factor (CF) legends for SOEC/PEM.
    - Dynamic Efficiency Calculation.
    """
    _check_dependencies()
    
    from h2_plant.visualization.utils import get_viz_config
    from plotly.subplots import make_subplots
    import pandas as pd
    import numpy as np
    
    # --- 1. DATA PREPARATION ---
    df_calc = df.copy()
    
    # Ensure 'minute' column exists for grouping
    if 'minute' not in df_calc.columns:
        if 'time' in df_calc.columns: 
             df_calc['minute'] = df_calc.index 
        else:
             df_calc['minute'] = df_calc.index * 60 

    # Identify Columns
    p_wind_col = next((c for c in ['P_offer'] if c in df_calc.columns), None)
    p_grid_col = next((c for c in ['P_sold', 'sell_power_mw'] if c in df_calc.columns), None)

    p_pem_col = next((c for c in ['P_pem', 'P_pem_actual', 'P_pem_mw', 'pem_power_mw'] if c in df_calc.columns), None)
    p_soec_col = next((c for c in ['P_soec', 'P_soec_actual', 'P_soec_mw', 'soec_power_mw'] if c in df_calc.columns), None)
    
    h2_cols = [c for c in ['H2_pem', 'H2_soec', 'H2_pem_kg', 'H2_soec_kg', 'H2_atr', 'H2_atr_kg'] if c in df_calc.columns]
    
    h2_soec_col = next((c for c in ['H2_soec', 'H2_soec_kg'] if c in df_calc.columns), None)
    h2_pem_col = next((c for c in ['H2_pem', 'H2_pem_kg'] if c in df_calc.columns), None)
    
    # --- 2. CALCULATE CAPACITY FACTOR ---
    pem_cf_str = ""
    if p_pem_col:
        # Get capacity from config or max observed
        cap_pem = get_viz_config('plant_parameters.pem_capacity_mw', df_calc[p_pem_col].max())
        if cap_pem > 0:
            cf = (df_calc[p_pem_col].mean() / cap_pem) * 100
            pem_cf_str = f" (CF: {cf:.1f}%)"

    soec_cf_str = ""
    if p_soec_col:
        cap_soec = get_viz_config('plant_parameters.soec_capacity_mw', df_calc[p_soec_col].max())
        if cap_soec > 0:
            cf = (df_calc[p_soec_col].mean() / cap_soec) * 100
            soec_cf_str = f" (CF: {cf:.1f}%)"

    # --- 3. AGGREGATION FUNCTION ---
    def get_aggregated_df(resolution_minutes):
        """Aggregate dataframe by resolution (min) and recalculate metrics."""
        df_res = df_calc.copy()
        df_res['group'] = df_res['minute'] // resolution_minutes
        
        # Define aggregation rules
        agg_rules = {}
        # Price (mean)
        price_col = next((c for c in ['spot_price', 'Spot'] if c in df_calc.columns), None)
        if price_col: agg_rules[price_col] = 'mean'
        
        # Power (mean)
        if p_pem_col: agg_rules[p_pem_col] = 'mean'
        if p_soec_col: agg_rules[p_soec_col] = 'mean'
        if p_wind_col: agg_rules[p_wind_col] = 'mean'
        if p_grid_col: agg_rules[p_grid_col] = 'mean'
        
        # H2 (mean rate kg/h)
        for c in h2_cols: agg_rules[c] = 'mean'
        
        # Add authoritative total columns to aggregation rules
        mixer_mass_col = 'h2_kg'
        mixer_rate_col = 'H2_Production_Mixer_outlet_mass_flow_kg_h'
        
        psa_cols = [
            'SOEC_H2_PSA_1_outlet_mass_flow_kg_h',
            'PEM_H2_PSA_1_outlet_mass_flow_kg_h', 
            'ATR_PSA_1_outlet_mass_flow_kg_h'
        ]
        
        if mixer_mass_col in df_calc.columns: agg_rules[mixer_mass_col] = 'mean'
        if mixer_rate_col in df_calc.columns: agg_rules[mixer_rate_col] = 'mean'
        
        for p_col in psa_cols:
             if p_col in df_calc.columns: agg_rules[p_col] = 'mean'
        
        if not agg_rules: return pd.DataFrame()
        
        df_grouped = df_res.groupby('group').agg(agg_rules).reset_index()
        
        # Recalculate Totals
        
        # Priority 1: Sum of PSA Outputs (True Purified Production)
        # Matches logic in scripts/plot_daily_h2_production.py
        soec_psa_col = 'SOEC_H2_PSA_1_outlet_mass_flow_kg_h'
        pem_psa_col = 'PEM_H2_PSA_1_outlet_mass_flow_kg_h'
        atr_psa_col = 'ATR_PSA_1_outlet_mass_flow_kg_h'
        psa_cols = [soec_psa_col, pem_psa_col, atr_psa_col]
        available_psa_cols = [c for c in psa_cols if c in df_grouped.columns]
        
        if available_psa_cols:
            df_grouped['Total_H2'] = df_grouped[available_psa_cols].sum(axis=1)
            
        # Priority 2: Mixer Outlet Rate (kg/h) - Direct reading (if PSA cols missing)
        elif mixer_rate_col in df_grouped.columns:
            df_grouped['Total_H2'] = df_grouped[mixer_rate_col]
            
        # Priority 3: Integrated Mass (kg/min) - Convert to Rate
        elif mixer_mass_col in df_grouped.columns:
            df_grouped['Total_H2'] = df_grouped[mixer_mass_col] * 60.0
            
        # Priority 4: Component Summation (Gross)
        else:
            df_grouped['Total_H2'] = 0.0
            current_h2_cols = [c for c in h2_cols if c in df_grouped.columns]
            for c in current_h2_cols:
                # Heuristic: If column name implies mass (ends in _kg), convert to rate
                if c.endswith('_kg'):
                    df_grouped['Total_H2'] += df_grouped[c] * 60.0
                else:
                    df_grouped['Total_H2'] += df_grouped[c]

        # Component Rates (kg/h) with PSA priority, then H2_*, then H2_*_kg
        def _component_rate(psa_col, rate_col, mass_col):
            if psa_col in df_grouped.columns:
                return df_grouped[psa_col]
            if rate_col in df_grouped.columns:
                return df_grouped[rate_col]
            if mass_col in df_grouped.columns:
                return df_grouped[mass_col] * 60.0
            return None

        soec_rate = _component_rate(soec_psa_col, 'H2_soec', 'H2_soec_kg')
        if soec_rate is not None:
            df_grouped['H2_SOEC_Rate'] = soec_rate

        pem_rate = _component_rate(pem_psa_col, 'H2_pem', 'H2_pem_kg')
        if pem_rate is not None:
            df_grouped['H2_PEM_Rate'] = pem_rate

        atr_rate = _component_rate(atr_psa_col, 'H2_atr', 'H2_atr_kg')
        if atr_rate is not None:
            df_grouped['H2_ATR_Rate'] = atr_rate
            
        # Recalculate Efficiencies (Dynamic)
        LHV_MWh_kg = 0.03333

        # SOEC Eff calc
        if p_soec_col and h2_soec_col and h2_soec_col in df_grouped.columns:
            with np.errstate(divide='ignore', invalid='ignore'):
                # Spec Prod = (kg/h) / MW
                # Assume h2_col is kg/min, so * 60 to get kg/h
                df_grouped['Spec_Prod_SOEC'] = np.where(df_grouped[p_soec_col] > 0.01, (df_grouped[h2_soec_col] * 60) / df_grouped[p_soec_col], 0)
                df_grouped['Eff_soec'] = df_grouped['Spec_Prod_SOEC'] * LHV_MWh_kg * 100
                df_grouped['Eff_soec'] = df_grouped['Eff_soec'].clip(0, 100)
                # SEC = 1000 / Spec Prod
                df_grouped['SEC_soec'] = np.where(df_grouped['Spec_Prod_SOEC'] > 0.01, 1000.0 / df_grouped['Spec_Prod_SOEC'], 0)

        # PEM Eff calc
        if p_pem_col and h2_pem_col and h2_pem_col in df_grouped.columns:
            with np.errstate(divide='ignore', invalid='ignore'):
                df_grouped['Spec_Prod_PEM'] = np.where(df_grouped[p_pem_col] > 0.01, (df_grouped[h2_pem_col] * 60) / df_grouped[p_pem_col], 0)
                df_grouped['Eff_pem'] = df_grouped['Spec_Prod_PEM'] * LHV_MWh_kg * 100
                df_grouped['Eff_pem'] = df_grouped['Eff_pem'].clip(0, 100)
        
        return df_grouped

    # Pre-calculate Aggregations
    df_h = get_aggregated_df(60)       # Hourly
    df_d = get_aggregated_df(1440)     # Daily
    df_m = get_aggregated_df(43800)    # Monthly (~30.4 days)
    df_y = get_aggregated_df(525600)   # Yearly

    # --- 4. PLOT CONFIGURATION ---
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True,
        subplot_titles=('Average Wind & Grid Power', 'Average Consumption', 
                        'Total H2 Production', 'System Efficiency (Dynamic SEC)'),
        vertical_spacing=0.08
    )
    
    def get_col(df_in, col): return df_in[col] if col in df_in.columns else np.array([])
    
    # Store trace configuration to map buttons later
    # Format: (column_name, is_custom_data_needed, custom_data_type)
    traces_config = [] 
    
    x_axis = df_h['group'].values
    
    # -- Row 1 --
    if p_wind_col:
        fig.add_trace(go.Bar(
            x=x_axis, y=get_col(df_h, p_wind_col), name='Wind Power', marker_color='#3498db',
            offsetgroup='r1_wind'
        ), row=1, col=1)
        traces_config.append((p_wind_col, False, None))
        
    if p_grid_col:
        fig.add_trace(go.Bar(
            x=x_axis, y=get_col(df_h, p_grid_col), name='Grid Export', marker_color='#f1c40f',
            offsetgroup='r1_grid'
        ), row=1, col=1)
        traces_config.append((p_grid_col, False, None))
    
    # -- Row 2 --
    if p_soec_col:
        fig.add_trace(go.Bar(
            x=x_axis, y=get_col(df_h, p_soec_col), name=f'SOEC Power{soec_cf_str}', marker_color='#2ecc71',
            offsetgroup='r2_soec'
        ), row=2, col=1)
        traces_config.append((p_soec_col, False, None))
        
    if p_pem_col:
        fig.add_trace(go.Bar(
            x=x_axis, y=get_col(df_h, p_pem_col), name=f'PEM Power{pem_cf_str}', marker_color='#e74c3c',
            offsetgroup='r2_pem'
        ), row=2, col=1)
        traces_config.append((p_pem_col, False, None))
    
    # -- Row 3 --
    if 'H2_SOEC_Rate' in df_h.columns:
        fig.add_trace(go.Bar(
            x=x_axis, y=df_h['H2_SOEC_Rate'], name='SOEC H2 Production', marker_color='#2ecc71',
            offsetgroup='h2_prod'
        ), row=3, col=1)
        traces_config.append(('H2_SOEC_Rate', False, None))

    if 'H2_PEM_Rate' in df_h.columns:
        fig.add_trace(go.Bar(
            x=x_axis, y=df_h['H2_PEM_Rate'], name='PEM H2 Production', marker_color='#e74c3c',
            offsetgroup='h2_prod'
        ), row=3, col=1)
        traces_config.append(('H2_PEM_Rate', False, None))

    if 'H2_ATR_Rate' in df_h.columns:
        fig.add_trace(go.Bar(
            x=x_axis, y=df_h['H2_ATR_Rate'], name='ATR H2 Production', marker_color='#9b59b6',
            offsetgroup='h2_prod'
        ), row=3, col=1)
        traces_config.append(('H2_ATR_Rate', False, None))
    
    # -- Row 4 --
    if 'Eff_soec' in df_h.columns:
        custom = np.column_stack([df_h['SEC_soec'], df_h['Spec_Prod_SOEC']]) if 'SEC_soec' in df_h else None
        fig.add_trace(
            go.Scatter(
                x=x_axis, y=df_h['Eff_soec'], name='SOEC Efficiency', 
                mode='lines+markers', line=dict(color='#2ecc71', width=2),
                customdata=custom,
                hovertemplate='Time: %{x}<br>Eff: %{y:.1f}%<br>SEC: %{customdata[0]:.1f} kWh/kg<br>Yield: %{customdata[1]:.1f} kg/MWh<extra></extra>'
            ), row=4, col=1
        )
        traces_config.append(('Eff_soec', True, 'soec'))
        
        # Reference Line (Static, not part of traces_config for updates)
        bol_eff_100pct = (1000.0 / 37.54) * 0.03333 * 100
        fig.add_hline(y=bol_eff_100pct, line_dash="dash", line_color="rgba(46, 204, 113, 0.5)",
                      annotation_text=f"BOL @100% ({bol_eff_100pct:.0f}%)", annotation_position="bottom right", row=4, col=1)
        
    if 'Eff_pem' in df_h.columns:
        custom_pem = df_h['Spec_Prod_PEM'] if 'Spec_Prod_PEM' in df_h else None
        fig.add_trace(
            go.Scatter(
                x=x_axis, y=df_h['Eff_pem'], name='PEM Efficiency', 
                mode='lines+markers', line=dict(color='#e74c3c', width=2),
                customdata=custom_pem,
                hovertemplate='Time: %{x}<br>Eff: %{y:.1f}%<br>Yield: %{customdata:.2f} kg/MWh<extra></extra>'
            ), row=4, col=1
        )
        traces_config.append(('Eff_pem', True, 'pem'))

    # --- 5. BUILD INTERACTIVE BUTTONS (CORRECTED STRUCTURE) ---
    def build_button_args(df_target):
        """Generates the properly structured args for Plotly RESTYLE method."""
        new_x = df_target['group'].values
        
        # 1. Create Dictionary of updates
        # Keys are properties to update ('x', 'y', 'customdata')
        # Values are LISTS containing the new data for EACH trace index passed in arg[1]
        updates = {
            'x': [],
            'y': [],
            'customdata': []
        }
        
        for col_name, needs_custom, custom_type in traces_config:
            # Append X data (same for all, but needed per trace)
            updates['x'].append(new_x)
            
            # Append Y data
            updates['y'].append(get_col(df_target, col_name))
            
            # Append CustomData
            if needs_custom:
                if custom_type == 'soec' and 'SEC_soec' in df_target:
                    updates['customdata'].append(np.column_stack([df_target['SEC_soec'], df_target['Spec_Prod_SOEC']]))
                elif custom_type == 'pem' and 'Spec_Prod_PEM' in df_target:
                    updates['customdata'].append(df_target['Spec_Prod_PEM'])
                else:
                    updates['customdata'].append(None)
            else:
                updates['customdata'].append(None)
                
        # 2. Define Trace Indices to apply these updates to
        # We assume traces are added in the exact order of traces_config
        # Note: We must skip the static HLine traces, but since they are added via add_hline (shapes) 
        # or separate traces not tracked in config, we use range(len(traces_config))
        # Be careful if static traces were added via add_trace before the config loop finished.
        # In this code, add_hline does NOT add a trace to the .data array in the same way.
        trace_indices = list(range(len(traces_config)))
        
        # RESTYLE signature: [changes_dict, trace_indices]
        return [updates, trace_indices]

    buttons = [
        dict(label="Hourly", method="restyle", args=build_button_args(df_h)),
        dict(label="Daily", method="restyle", args=build_button_args(df_d)),
        dict(label="Monthly", method="restyle", args=build_button_args(df_m)),
        dict(label="Yearly", method="restyle", args=build_button_args(df_y)),
    ]

    # --- 6. FINAL LAYOUT UPDATE ---
    fig.update_layout(
        title=dict(text=kwargs.get('title', 'Temporal Averages Overview'), y=0.98),
        template='plotly_white',
        height=900,
        showlegend=True,
        barmode='relative',
        # Margin adjusted to accommodate buttons
        margin=dict(t=120, b=50, l=60, r=60), 
        updatemenus=[dict(
            type="buttons",
            direction="right",
            # Position: Centered horizontally, slightly above the chart area
            x=0.5, y=1.06,
            xanchor='center', yanchor='bottom',
            buttons=buttons,
            pad={"r": 5, "t": 10},
            showactive=True,
            bgcolor="#f8f9fa",
            bordercolor="#dee2e6",
            borderwidth=1
        )]
    )
    
    fig.update_yaxes(title_text="Power (MW)", row=1, col=1)
    fig.update_yaxes(title_text="Power (MW)", row=2, col=1)
    fig.update_yaxes(title_text="H2 Rate (kg/h)", row=3, col=1)
    fig.update_yaxes(title_text="Efficiency (% LHV)", range=[0, 100], row=4, col=1)
    fig.update_xaxes(title_text='Time Group', row=4, col=1)
    
    return fig


@log_graph_errors
def plot_power_vs_ppa(df: pd.DataFrame, **kwargs) -> go.Figure:
    """
    Plot Power Offer vs Effective PPA Price with view switching (Scatter vs Time Series).
    """
    _check_dependencies()
    
    from h2_plant.visualization.utils import downsample_dataframe, get_time_axis_hours, get_viz_config
    
    maxpoints = kwargs.get('maxpoints', get_viz_config('performance.max_points_default', 2000))
    df_plot = downsample_dataframe(df, max_points=maxpoints)
    hours = get_time_axis_hours(df_plot)
    
    p_offer_col = 'P_offer'
    ppa_col = 'ppa_price_effective_eur_mwh'
    
    if p_offer_col not in df_plot.columns or ppa_col not in df_plot.columns:
        return _empty_figure("No power offer or PPA data found")

    def _get_dispatch_order_mask(df_local: pd.DataFrame):
        # Priority 1: explicit sell decision flag
        sell_cols = [c for c in df_local.columns if 'sell_decision' in c.lower()]
        if sell_cols:
            series = pd.to_numeric(df_local[sell_cols[0]], errors='coerce').fillna(0.0)
            return series > 0, "Dispatch Order (Sell)"

        # Priority 2: sold power > 0
        sold_cols = [c for c in df_local.columns if c.lower() in ['p_sold', 'p_sold_mw'] or 'p_sold' in c.lower()]
        if sold_cols:
            series = pd.to_numeric(df_local[sold_cols[0]], errors='coerce').fillna(0.0)
            return series > 0, "Dispatch Order (Grid Sell)"

        # Priority 3: spot purchase (economic dispatch)
        spot_cols = [c for c in df_local.columns if 'spot_purchased_mw' in c.lower()]
        if spot_cols:
            series = pd.to_numeric(df_local[spot_cols[0]], errors='coerce').fillna(0.0)
            return series > 0, "Dispatch Order (Spot Purchase)"

        return None, None

    dispatch_mask, dispatch_label = _get_dispatch_order_mask(df_plot)
    has_dispatch = dispatch_mask is not None and np.any(dispatch_mask)
    
    fig = go.Figure()
    
    # View A: Scatter Plot (P_offer vs PPA) - Default Visible
    fig.add_trace(go.Scatter(
        x=df_plot[p_offer_col],
        y=df_plot[ppa_col],
        mode='markers',
        name='Correlation',
        marker=dict(opacity=0.6, size=5),
        visible=True 
    ))

    # Shipment markers (Scatter View)
    if has_dispatch:
        fig.add_trace(go.Scatter(
            x=df_plot[p_offer_col].values[dispatch_mask],
            y=df_plot[ppa_col].values[dispatch_mask],
            mode='markers',
            name=dispatch_label,
            marker=dict(symbol='x', size=9, color='#f39c12', line=dict(width=1, color='#d35400')),
            visible=True
        ))
    
    # View B: Time Series (Dual Axis) - Default Hidden
    fig.add_trace(go.Scatter(
        x=hours, y=df_plot[p_offer_col],
        mode='lines', name='Power Offer',
        line=dict(color='blue'),
        visible=False
    ))
    
    fig.add_trace(go.Scatter(
        x=hours, y=df_plot[ppa_col],
        mode='lines', name='PPA Price',
        line=dict(color='red'),
        yaxis='y2',
        visible=False
    ))

    # Shipment markers (Time Series View)
    if has_dispatch:
        fig.add_trace(go.Scatter(
            x=hours[dispatch_mask],
            y=df_plot[ppa_col].values[dispatch_mask],
            mode='markers',
            name=dispatch_label,
            marker=dict(symbol='x', size=9, color='#f39c12', line=dict(width=1, color='#d35400')),
            yaxis='y2',
            visible=False
        ))

    if has_dispatch:
        scatter_visible = [True, True, False, False, False]
        timeseries_visible = [False, False, True, True, True]
    else:
        scatter_visible = [True, False, False]
        timeseries_visible = [False, True, True]
    
    # Dropdown for switching views
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=[
                    dict(label='Scatter View (Correlation)',
                         method='update',
                         args=[{'visible': scatter_visible},
                               {'title': 'Power Offer vs PPA Price (Scatter)',
                                'xaxis': {'title': 'Power Offer (MW)'},
                                'yaxis': {'title': 'PPA Price (EUR/MWh)'},
                                'yaxis2': {'visible': False}}]),
                    dict(label='Time Series View',
                         method='update',
                         args=[{'visible': timeseries_visible},
                               {'title': 'Power Offer and PPA Price Over Time',
                                'xaxis': {'title': 'Time (hours)'},
                                'yaxis': {'title': 'Power Offer (MW)'},
                                'yaxis2': {'title': 'PPA Price (EUR/MWh)', 'overlaying': 'y', 'side': 'right', 'visible': True}}])
                ],
                direction="down",
                showactive=True,
                x=0.0, xanchor="left", y=1.1, yanchor="top"
            )
        ],
        title='Power Offer vs PPA Price (Scatter)',
        xaxis_title='Power Offer (MW)',
        yaxis_title='PPA Price (EUR/MWh)',
        yaxis2=dict(title='PPA Price (EUR/MWh)', overlaying='y', side='right', visible=False),
        template='plotly_white'
    )
    
    return fig

@log_graph_errors
def plot_storage_apc_enhanced(df: pd.DataFrame, **kwargs) -> go.Figure:
    """
    Enhanced Storage APC Plot.
    
    Row 1: Storage SOC (Right Axis) vs APC Action Factor (Left Axis).
    Row 2: H2 Production Rate vs Demand Rate (Mass Balance).
    Row 3: Accumulated Purified H2 (Tank Inlets or PSA Outputs).
    """
    _check_dependencies()
    from plotly.subplots import make_subplots
    from h2_plant.visualization.utils import downsample_dataframe, get_time_axis_hours, get_viz_config

    maxpoints = kwargs.get('maxpoints', get_viz_config('performance.max_points_default', 2000))
    df_plot = downsample_dataframe(df, max_points=maxpoints)
    hours = get_time_axis_hours(df_plot)

    # --- Data Retrieval ---
    # 1. Storage Metrics
    soc_col = next((c for c in ['storage_soc', 'soc', 'state_of_charge'] if c in df_plot.columns), None)
    factor_col = next((c for c in ['storage_action_factor', 'action_factor'] if c in df_plot.columns), None)
    
    if not soc_col:
        return _empty_figure("No Storage SOC data found for Enhanced APC plot")

    soc = df_plot[soc_col].values * 100
    factor = df_plot[factor_col].values if factor_col else np.zeros(len(hours))

    # 2. Flow Metrics (Production vs Demand)
    # Calculate Total Production Rate (kg/h)
    dt_seconds = df.attrs.get('dt_seconds', 60.0)
    dt_h = dt_seconds / 3600.0
    
    prod_rate = np.zeros(len(hours))
    psa_cols = [
        'SOEC_H2_PSA_1_outlet_mass_flow_kg_h',
        'PEM_H2_PSA_1_outlet_mass_flow_kg_h', 
        'ATR_PSA_1_outlet_mass_flow_kg_h'
    ]
    
    for col in psa_cols:
        if col in df_plot.columns:
            prod_rate += df_plot[col].values

    # Find Demand Rate
    # Look for specific discharge station columns or generic demand signals
    demand_col = next((c for c in df_plot.columns if 'truck_demand_kg_h' in c or 'total_demand_signal' in c), None)
    
    if demand_col:
        demand_rate = df_plot[demand_col].values
    else:
        # Fallback: Check for generic 'demand' column
        generic_dem = next((c for c in ['demand_kg_h', 'h2_demand_kg_h'] if c in df_plot.columns), None)
        demand_rate = df_plot[generic_dem].values if generic_dem else np.zeros(len(hours))

    # 3. Accumulated Purified H2 (Tank Inlets or PSA Outputs)
    dt_h = np.median(np.diff(hours)) if len(hours) > 1 else dt_h
    if not np.isfinite(dt_h) or dt_h <= 0:
        dt_h = dt_seconds / 3600.0

    tank_inlet_cols = [
        c for c in df_plot.columns
        if ('tank' in c.lower() or 'storage' in c.lower())
        and 'inlet' in c.lower()
        and ('kg_h' in c.lower() or 'mass_flow_kg_h' in c.lower() or 'flow_kg_h' in c.lower())
    ]
    psa_outlet_cols = [
        c for c in df_plot.columns
        if 'psa' in c.lower() and 'h2' in c.lower() and 'outlet' in c.lower()
        and ('kg_h' in c.lower() or 'mass_flow_kg_h' in c.lower() or 'flow_kg_h' in c.lower())
    ]

    purified_rate = np.zeros(len(hours))
    purified_source = None

    if tank_inlet_cols:
        for col in tank_inlet_cols:
            purified_rate += df_plot[col].values
        purified_source = "Tank Inlets"
    elif psa_outlet_cols:
        for col in psa_outlet_cols:
            purified_rate += df_plot[col].values
        purified_source = "PSA Outlets"
    else:
        purified_source = "No purified H2 flow data"

    purified_cumulative = np.cumsum(purified_rate * dt_h)

    # --- Plot Generation ---
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "Control Status (SOC & Action Factor)",
            "Mass Balance (Production vs Demand)",
            "Accumulated Purified H2 Balance"
        ),
        specs=[[{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": False}]]
    )

    ScatterType = get_scatter_type(len(hours))

    # === ROW 1: Control Status ===
    
    # Trace 1: Action Factor (Left Axis - Primary Control Output)
    fig.add_trace(ScatterType(
        x=hours, y=factor,
        mode='lines', name='Action Factor',
        line=dict(color='#9b59b6', width=1.5, dash='dot'), # Purple
        hovertemplate='Factor: %{y:.2f}<extra></extra>'
    ), row=1, col=1, secondary_y=False)

    # Trace 2: SOC (Right Axis - System State)
    fig.add_trace(ScatterType(
        x=hours, y=soc,
        mode='lines', name='State of Charge (%)',
        line=dict(color='#2ecc71', width=2), # Green
        fill='tozeroy',
        fillcolor='rgba(46, 204, 113, 0.1)',
        hovertemplate='SOC: %{y:.1f}%<extra></extra>'
    ), row=1, col=1, secondary_y=True)

    # Zone Thresholds (on SOC axis)
    for y_zone, color, label in [(30, 'red', 'Zone A'), (60, 'orange', 'Zone B'), (90, 'green', 'Zone C')]:
        fig.add_hline(y=y_zone, line_dash="dash", line_color=color, opacity=0.5, row=1, col=1, secondary_y=True)

    # === ROW 2: Mass Balance ===

    # Trace 3: Total Production
    fig.add_trace(ScatterType(
        x=hours, y=prod_rate,
        mode='lines', name='Total Production',
        line=dict(color='#1f77b4', width=2), # Blue
        hovertemplate='Prod: %{y:.1f} kg/h<extra></extra>'
    ), row=2, col=1)

    # Trace 4: Demand
    fig.add_trace(ScatterType(
        x=hours, y=demand_rate,
        mode='lines', name='H2 Demand',
        line=dict(color='#d62728', width=2, dash='dash'), # Red
        hovertemplate='Demand: %{y:.1f} kg/h<extra></extra>'
    ), row=2, col=1)

    # Optional: Net Flow (Prod - Demand) fill
    # This visually indicates Filling (Blue) vs Draining (Red)
    # Note: Requires a zero line for reference
    fig.add_trace(ScatterType(
        x=hours, y=prod_rate - demand_rate,
        mode='lines', name='Net Flow',
        line=dict(width=0),
        showlegend=False,
        fill='tozeroy',
        fillcolor='rgba(128, 128, 128, 0.1)',
        hoverinfo='skip'
    ), row=2, col=1)

    # === ROW 3: Accumulated Purified H2 ===
    fig.add_trace(ScatterType(
        x=hours, y=purified_cumulative,
        mode='lines', name=f'Purified H2 Accumulated ({purified_source})',
        line=dict(color='#34495e', width=2),
        hovertemplate='Accumulated H2: %{y:,.0f} kg<extra></extra>'
    ), row=3, col=1)

    # --- Layout Updates ---
    fig.update_layout(
        title=kwargs.get('title', 'Storage APC Dynamics & Mass Balance'),
        template='plotly_white',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=900
    )

    # Axis Labels
    fig.update_yaxes(title_text="Action Factor (0-1)", range=[-0.05, 1.1], row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="SOC (%)", range=[0, 105], row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Mass Flow (kg/h)", row=2, col=1)
    fig.update_yaxes(title_text="Accumulated H2 (kg)", row=3, col=1)
    fig.update_xaxes(title_text="Time (hours)", row=3, col=1)

    return fig

@log_graph_errors
def plot_temporal_sums(df: pd.DataFrame, **kwargs) -> go.Figure:
    """
    Plot Temporal Sums (Hourly, Daily, Monthly, Yearly totals) - Interactive.
    
    Shows cumulative Energy (MWh) and Mass (kg) instead of averages.
    Rows:
    1. Energy Availability (Wind vs Grid Export) [MWh]
    2. Energy Consumption (PEM vs SOEC) [MWh]
    3. Total H2 Production [kg]
    """
    _check_dependencies()
    from plotly.subplots import make_subplots
    import pandas as pd
    import numpy as np

    # --- 1. DATA PREPARATION ---
    df_calc = df.copy()

    # Ensure 'minute' column exists
    if 'minute' not in df_calc.columns:
        if 'time' in df_calc.columns:
            df_calc['minute'] = df_calc.index
        else:
            df_calc['minute'] = df_calc.index * 60

    # Determine Timestep (dt in hours) for Integration (MW -> MWh)
    # If minute column is monotonic, calc diff. Otherwise assume 1 min default or use config.
    dt_seconds = df.attrs.get('dt_seconds', 60.0)
    if len(df_calc) > 1:
        dt_minutes = np.diff(df_calc['minute'], append=df_calc['minute'].iloc[-1] + 1)
        # Handle potential downsampling gaps or resets by median
        dt_h = np.median(dt_minutes) / 60.0
    else:
        dt_h = dt_seconds / 3600.0
        
    # --- 2. CALCULATE ENERGY (MWh) & TOTAL MASS (kg) ---
    
    # Identify Power Columns (MW)
    p_wind_col = next((c for c in ['P_offer'] if c in df_calc.columns), None)
    p_grid_col = next((c for c in ['P_sold', 'sell_power_mw'] if c in df_calc.columns), None)
    p_pem_col = next((c for c in ['P_pem', 'P_pem_actual', 'P_pem_mw', 'pem_power_mw'] if c in df_calc.columns), None)
    p_soec_col = next((c for c in ['P_soec', 'P_soec_actual', 'P_soec_mw', 'soec_power_mw'] if c in df_calc.columns), None)

    # Calculate Energy (MWh) = Power (MW) * dt (h)
    if p_wind_col: df_calc['Energy_Wind'] = df_calc[p_wind_col] * dt_h
    if p_grid_col: df_calc['Energy_Grid'] = df_calc[p_grid_col] * dt_h
    if p_pem_col:  df_calc['Energy_PEM'] = df_calc[p_pem_col] * dt_h
    if p_soec_col: df_calc['Energy_SOEC'] = df_calc[p_soec_col] * dt_h

    # Identify H2 Columns (PSA Outlet Flow Rates in kg/h)
    # User requested specific PSA tags:
    soec_psa_col = 'SOEC_H2_PSA_1_outlet_mass_flow_kg_h'
    pem_psa_col = 'PEM_H2_PSA_1_outlet_mass_flow_kg_h'
    atr_psa_col = 'ATR_PSA_1_outlet_mass_flow_kg_h'
    psa_cols = [soec_psa_col, pem_psa_col, atr_psa_col]

    # Consolidate H2 Mass (kg)
    # Rates are in kg/h. To get Mass (kg), multiply by dt_h.
    # For H2_*_kg fallback, assume kg/min and convert to kg/h via *60 first.
    def _component_rate(psa_col, rate_col, mass_col):
        if psa_col in df_calc.columns:
            return df_calc[psa_col]
        if rate_col in df_calc.columns:
            return df_calc[rate_col]
        if mass_col in df_calc.columns:
            return df_calc[mass_col] * 60.0
        return None

    df_calc['Mass_H2_Total'] = 0.0

    soec_rate = _component_rate(soec_psa_col, 'H2_soec', 'H2_soec_kg')
    if soec_rate is not None:
        df_calc['Mass_H2_SOEC'] = soec_rate * dt_h
        df_calc['Mass_H2_Total'] += df_calc['Mass_H2_SOEC']

    pem_rate = _component_rate(pem_psa_col, 'H2_pem', 'H2_pem_kg')
    if pem_rate is not None:
        df_calc['Mass_H2_PEM'] = pem_rate * dt_h
        df_calc['Mass_H2_Total'] += df_calc['Mass_H2_PEM']

    atr_rate = _component_rate(atr_psa_col, 'H2_atr', 'H2_atr_kg')
    if atr_rate is not None:
        df_calc['Mass_H2_ATR'] = atr_rate * dt_h
        df_calc['Mass_H2_Total'] += df_calc['Mass_H2_ATR']

    # --- 3. AGGREGATION FUNCTION ---
    def get_summed_df(resolution_minutes):
        """Aggregate dataframe by resolution (min) and SUM metrics."""
        df_res = df_calc.copy()
        df_res['group'] = df_res['minute'] // resolution_minutes
        
        # Define Summation Rules
        agg_rules = {}
        
        # Energy (MWh)
        if 'Energy_Wind' in df_res: agg_rules['Energy_Wind'] = 'sum'
        if 'Energy_Grid' in df_res: agg_rules['Energy_Grid'] = 'sum'
        if 'Energy_PEM' in df_res:  agg_rules['Energy_PEM'] = 'sum'
        if 'Energy_SOEC' in df_res: agg_rules['Energy_SOEC'] = 'sum'
        
        # H2 Mass (kg)
        if 'Mass_H2_SOEC' in df_res: agg_rules['Mass_H2_SOEC'] = 'sum'
        if 'Mass_H2_PEM' in df_res: agg_rules['Mass_H2_PEM'] = 'sum'
        if 'Mass_H2_ATR' in df_res: agg_rules['Mass_H2_ATR'] = 'sum'
        agg_rules['Mass_H2_Total'] = 'sum'
        
        if not agg_rules: return pd.DataFrame()
        
        df_grouped = df_res.groupby('group').agg(agg_rules).reset_index()
        
        # Add time label (approximate)
        # For labeling, we can convert group index back to hours/days
        if resolution_minutes == 60:
            df_grouped['Time_Label'] = df_grouped['group']  # Hours
        elif resolution_minutes == 1440:
            df_grouped['Time_Label'] = df_grouped['group']  # Days
        elif resolution_minutes >= 40000:
             df_grouped['Time_Label'] = df_grouped['group'] # Months
             
        return df_grouped

    # Pre-calculate Aggregations
    df_h = get_summed_df(60)       # Hourly
    df_d = get_summed_df(1440)     # Daily
    df_m = get_summed_df(43800)    # Monthly (~30.4 days)
    df_y = get_summed_df(525600)   # Yearly

    # --- 4. PLOT CONFIGURATION ---
    # 3 Rows (Efficiency removed)
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True,
        subplot_titles=('Total Energy Availability (Wind vs Export)', 
                        'Total Energy Consumption', 
                        'Total H2 Production'),
        vertical_spacing=0.1
    )
    
    def get_col(df_in, col): return df_in[col] if col in df_in.columns else np.array([])
    
    # Store trace configuration
    # Format: (column_name, color, name, row_idx, offsetgroup)
    traces_config = []
    
    # Define Traces logic
    # Row 1: Energy Availability
    if 'Energy_Wind' in df_calc:
        traces_config.append(('Energy_Wind', '#3498db', 'Wind Energy Available', 1, 'r1_wind'))
    if 'Energy_Grid' in df_calc:
        traces_config.append(('Energy_Grid', '#f1c40f', 'Grid Export', 1, 'r1_grid'))
        
    # Row 2: Consumption
    if 'Energy_SOEC' in df_calc:
        traces_config.append(('Energy_SOEC', '#2ecc71', 'SOEC Consumption', 2, 'r2_soec'))
    if 'Energy_PEM' in df_calc:
        traces_config.append(('Energy_PEM', '#e74c3c', 'PEM Consumption', 2, 'r2_pem'))
        
    # Row 3: H2 Production
    if 'Mass_H2_SOEC' in df_calc:
        traces_config.append(('Mass_H2_SOEC', '#2ecc71', 'SOEC H2 Mass', 3, 'h2_prod'))
    if 'Mass_H2_PEM' in df_calc:
        traces_config.append(('Mass_H2_PEM', '#e74c3c', 'PEM H2 Mass', 3, 'h2_prod'))
    if 'Mass_H2_ATR' in df_calc:
        traces_config.append(('Mass_H2_ATR', '#9b59b6', 'ATR H2 Mass', 3, 'h2_prod'))

    # Initial Plot (Hourly)
    x_axis = df_h['group'].values
    
    for col, color, name, row, offsetgroup in traces_config:
        fig.add_trace(
            go.Bar(
                x=x_axis, 
                y=get_col(df_h, col), 
                name=name, 
                marker_color=color,
                offsetgroup=offsetgroup
            ), row=row, col=1
        )

    # --- 5. BUILD INTERACTIVE BUTTONS ---
    def build_button_args(df_target):
        new_x = df_target['group'].values
        updates = {'x': [], 'y': []}
        
        for col, _, _, _, _ in traces_config:
            updates['x'].append(new_x)
            updates['y'].append(get_col(df_target, col))
            
        return [updates, list(range(len(traces_config)))]

    buttons = [
        dict(label="Hourly", method="restyle", args=build_button_args(df_h)),
        dict(label="Daily", method="restyle", args=build_button_args(df_d)),
        dict(label="Monthly", method="restyle", args=build_button_args(df_m)),
        dict(label="Yearly", method="restyle", args=build_button_args(df_y)),
    ]

    # --- 6. LAYOUT ---
    fig.update_layout(
        title=dict(text=kwargs.get('title', 'Temporal Sums (Energy & Mass)'), y=0.98),
        template='plotly_white',
        height=800,  # Slightly shorter since 1 row removed
        showlegend=True,
        barmode='relative',
        margin=dict(t=120, b=50, l=60, r=60), 
        updatemenus=[dict(
            type="buttons",
            direction="right",
            x=0.5, y=1.08,
            xanchor='center', yanchor='bottom',
            buttons=buttons,
            pad={"r": 5, "t": 10},
            showactive=True,
            bgcolor="#f8f9fa",
            bordercolor="#dee2e6",
            borderwidth=1
        )]
    )
    
    fig.update_yaxes(title_text="Energy (MWh)", row=1, col=1)
    fig.update_yaxes(title_text="Energy (MWh)", row=2, col=1)
    fig.update_yaxes(title_text="Mass (kg)", row=3, col=1)
    fig.update_xaxes(title_text='Time Group Index', row=3, col=1)
    
    return fig
