"""
Plotly graph implementations for H2 Plant visualization.
"""

from typing import Dict, Any, List, Optional
import logging

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


def _check_dependencies():
    """Check if required dependencies are available."""
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required for graph generation. Install with: pip install plotly")
    if not NUMPY_AVAILABLE:
        raise ImportError("NumPy is required. Install with: pip install numpy")


@log_graph_errors
def plot_pem_production_timeline(data: Dict[str, Any], **kwargs) -> go.Figure:
    """
    Plot PEM H2 production rate over time.
    
    Args:
        data: Dictionary containing 'timestamps' and 'pem' data
        **kwargs: Additional plot customization options
    
    Returns:
        Plotly Figure object
    """
    _check_dependencies()
    
    timestamps = data.get('timestamps', [])
    production = data['pem'].get('h2_production_kg_h', [])
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=production,
        mode='lines',
        name='PEM Production',
        line=dict(color='#1f77b4', width=2),
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.1)'
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
def plot_soec_production_timeline(data: Dict[str, Any], **kwargs) -> go.Figure:
    """Plot SOEC H2 production rate over time."""
    _check_dependencies()
    
    timestamps = data.get('timestamps', [])
    production = data['soec'].get('h2_production_kg_h', [])
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=production,
        mode='lines',
        name='SOEC Production',
        line=dict(color='#ff7f0e', width=2),
        fill='tozeroy',
        fillcolor='rgba(255, 127, 14, 0.1)'
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
def plot_total_production_stacked(data: Dict[str, Any], **kwargs) -> go.Figure:
    """Plot stacked area chart showing PEM + SOEC contributions."""
    _check_dependencies()
    
    timestamps = data.get('timestamps', [])
    pem_production = data['pem'].get('h2_production_kg_h', [])
    soec_production = data['soec'].get('h2_production_kg_h', [])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=pem_production,
        mode='lines',
        name='PEM',
        stackgroup='one',
        line=dict(color='#1f77b4', width=0.5),
        fillcolor='rgba(31, 119, 180, 0.7)'
    ))
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=soec_production,
        mode='lines',
        name='SOEC',
        stackgroup='one',
        line=dict(color='#ff7f0e', width=0.5),
        fillcolor='rgba(255, 127, 14, 0.7)'
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
def plot_cumulative_production(data: Dict[str, Any], **kwargs) -> go.Figure:
    """Plot cumulative H2 production from both systems."""
    _check_dependencies()
    
    timestamps = data.get('timestamps', [])
    pem_cumulative = data['pem'].get('cumulative_h2_kg', [])
    soec_cumulative = data['soec'].get('cumulative_h2_kg', [])
    
    # Calculate total
    total_cumulative = [p + s for p, s in zip(pem_cumulative, soec_cumulative)]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=pem_cumulative,
        mode='lines',
        name='PEM Cumulative',
        line=dict(color='#1f77b4', width=2, dash='dot')
    ))
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=soec_cumulative,
        mode='lines',
        name='SOEC Cumulative',
        line=dict(color='#ff7f0e', width=2, dash='dot')
    ))
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=total_cumulative,
        mode='lines',
        name='Total',
        line=dict(color='#2ca02c', width=3)
    ))
    
    fig.update_layout(
        title=kwargs.get('title', 'Cumulative H2 Production'),
        xaxis_title='Time (hours)',
        yaxis_title='Cumulative H2 (kg)',
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig


@log_graph_errors
def plot_pem_voltage_timeline(data: Dict[str, Any], **kwargs) -> go.Figure:
    """Plot PEM cell voltage over time."""
    _check_dependencies()
    
    timestamps = data.get('timestamps', [])
    voltage = data['pem'].get('voltage', [])
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps,
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
def plot_pem_efficiency_timeline(data: Dict[str, Any], **kwargs) -> go.Figure:
    """Plot PEM system efficiency over time."""
    _check_dependencies()
    
    timestamps = data.get('timestamps', [])
    efficiency = data['pem'].get('efficiency', [])
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=efficiency,
        mode='lines',
        name='System Efficiency',
        line=dict(color='#2ca02c', width=2)
    ))
    
    fig.update_layout(
        title=kwargs.get('title', 'PEM System Efficiency'),
        xaxis_title='Time (hours)',
        yaxis_title='Efficiency (% LHV)',
        template='plotly_white',
        hovermode='x unified',
        yaxis=dict(range=[0, 100])
    )
    
    return fig


@log_graph_errors
def plot_energy_price_timeline(data: Dict[str, Any], **kwargs) -> go.Figure:
    """Plot energy price over time."""
    _check_dependencies()
    
    timestamps = data.get('timestamps', [])
    price = data['pricing'].get('energy_price_eur_kwh', [])
    
    # Convert to €/MWh for better readability
    price_mwh = [p * 1000 for p in price]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps,
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
        yaxis_title='Price (€/MWh)',
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig


@log_graph_errors
def plot_dispatch_strategy(data: Dict[str, Any], **kwargs) -> go.Figure:
    """Plot dispatch strategy as stacked area chart."""
    _check_dependencies()
    
    timestamps = data.get('timestamps', [])
    pem_power = data['coordinator'].get('pem_setpoint_mw', [])
    soec_power = data['coordinator'].get('soec_setpoint_mw', [])
    sell_power = data['coordinator'].get('sell_power_mw', [])
    
    # Get auxiliary power (convert kW to MW for consistency)
    aux_power_kw = data.get('auxiliary_power_kw', [])
    aux_power_mw = [p / 1000.0 for p in aux_power_kw] if aux_power_kw else [0] * len(timestamps)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=pem_power,
        mode='lines',
        name='PEM',
        stackgroup='one',
        line=dict(color='#1f77b4', width=0.5),
        fillcolor='rgba(31, 119, 180, 0.7)'
    ))
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=soec_power,
        mode='lines',
        name='SOEC',
        stackgroup='one',
        line=dict(color='#ff7f0e', width=0.5),
        fillcolor='rgba(255, 127, 14, 0.7)'
    ))
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=aux_power_mw,
        mode='lines',
        name='Auxiliary',
        stackgroup='one',
        line=dict(color='#9467bd', width=0.5),
        fillcolor='rgba(148, 103, 189, 0.7)'
    ))
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=sell_power,
        mode='lines',
        name='Grid Export',
        stackgroup='one',
        line=dict(color='#2ca02c', width=0.5),
        fillcolor='rgba(44, 160, 44, 0.7)'
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
def plot_power_breakdown_pie(data: Dict[str, Any], **kwargs) -> go.Figure:
    """Plot power consumption breakdown as pie chart."""
    _check_dependencies()
    
    pem_energy = data['pem'].get('cumulative_energy_kwh', [0])[-1] if data['pem'].get('cumulative_energy_kwh') else 0
    soec_energy = data['soec'].get('cumulative_energy_kwh', [0])[-1] if data['soec'].get('cumulative_energy_kwh') else 0
    
    fig = go.Figure(data=[go.Pie(
        labels=['PEM', 'SOEC'],
        values=[pem_energy, soec_energy],
        marker=dict(colors=['#1f77b4', '#ff7f0e']),
        textinfo='label+percent',
        textposition='inside'
    )])
    
    fig.update_layout(
        title=kwargs.get('title', 'Total Energy Consumption Breakdown'),
        template='plotly_white'
    )
    
    return fig


@log_graph_errors
def plot_soec_modules_timeline(data: Dict[str, Any], **kwargs) -> go.Figure:
    """Plot number of active SOEC modules over time."""
    _check_dependencies()
    
    timestamps = data.get('timestamps', [])
    active_modules = data['soec'].get('active_modules', [])
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps,
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
        yaxis=dict(range=[0, 7], dtick=1)
    )
    
    return fig


@log_graph_errors
def plot_tank_storage_timeline(data: Dict[str, Any], **kwargs) -> go.Figure:
    """Plot tank storage levels over time (placeholder)."""
    _check_dependencies()
    
    # TODO: Implement tank visualization with 2D arrays
    fig = go.Figure()
    fig.add_annotation(
        text="Tank visualization not yet implemented",
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False
    )
    
    fig.update_layout(
        title=kwargs.get('title', 'Tank Storage Timeline (Not Implemented)'),
        template='plotly_white'
    )
    
    return fig


def plot_storage_fatigue_cycling_3d(data: Dict[str, Any], **kwargs) -> go.Figure:
    """
    Plot 3D surface of tank pressures over time (Reliability Analysis).
    X=Time, Y=Tank ID, Z=Pressure
    """
    _check_dependencies()
    
    timestamps = data.get('timestamps', [])
    pressures = data['tanks'].get('hp_pressures', []) # 2D array [timestep][tank_id]
    
    if not pressures or not timestamps:
        return go.Figure()
        
    # Convert to numpy for easier handling
    pressures_np = np.array(pressures) # Shape: (timesteps, n_tanks)
    n_tanks = pressures_np.shape[1]
    tank_ids = list(range(n_tanks))
    
    # Create meshgrid
    X, Y = np.meshgrid(timestamps, tank_ids)
    Z = pressures_np.T # Transpose to match (n_tanks, timesteps)
    
    fig = go.Figure(data=[go.Surface(
        x=X, y=Y, z=Z,
        colorscale='Viridis',
        colorbar_title='Pressure (bar)'
    )])
    
    fig.update_layout(
        title=kwargs.get('title', 'Storage Tank Fatigue Cycling (3D)'),
        scene=dict(
            xaxis_title='Time (hours)',
            yaxis_title='Tank ID',
            zaxis_title='Pressure (bar)'
        ),
        template='plotly_white'
    )
    return fig


@log_graph_errors
def plot_ramp_rate_stress_distribution(data: Dict[str, Any], **kwargs) -> go.Figure:
    """
    Plot distribution of ramp rates (Stress Analysis).
    Violin plot of MW/min for SOEC (and PEM if available).
    """
    _check_dependencies()
    
    soec_ramps = data['soec'].get('ramp_rates', [])
    
    fig = go.Figure()
    
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
def plot_wind_utilization_duration_curve(data: Dict[str, Any], **kwargs) -> go.Figure:
    """
    Plot wind utilization duration curve (Grid Integration).
    Sorted hours 0-8760 vs Power.
    """
    _check_dependencies()
    
    # Calculate available wind power (assuming 15MW capacity for now, or derive from data)
    # We have wind_coefficient (0-1). Let's assume nominal capacity is sum of electrolyzers approx 15MW
    # Or better, just plot coefficient if capacity unknown.
    # Let's assume 20MW wind farm for this scale.
    WIND_CAPACITY_MW = 20.0 
    
    wind_coeffs = np.array(data['pricing'].get('wind_coefficient', []))
    wind_available = wind_coeffs * WIND_CAPACITY_MW
    
    # Calculate used power (PEM + SOEC)
    pem_power = np.array(data['pem'].get('power_mw', []))
    soec_power = np.array(data['soec'].get('power_mw', []))
    total_used = pem_power + soec_power
    
    # Sort descending
    wind_sorted = np.sort(wind_available)[::-1]
    used_sorted = np.sort(total_used)[::-1]
    
    hours = np.arange(len(wind_sorted))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hours, y=wind_sorted, mode='lines', name='Available Wind Power', fill='tozeroy'))
    fig.add_trace(go.Scatter(x=hours, y=used_sorted, mode='lines', name='Utilized Power', fill='tozeroy'))
    
    fig.update_layout(
        title=kwargs.get('title', 'Wind Utilization Duration Curve'),
        xaxis_title='Hours (sorted)',
        yaxis_title='Power (MW)',
        template='plotly_white'
    )
    return fig


@log_graph_errors
def plot_grid_interaction_phase_portrait(data: Dict[str, Any], **kwargs) -> go.Figure:
    """
    Plot phase portrait of Grid Exchange vs Wind Power.
    X=Wind Power, Y=Grid Exchange (+Import/-Export)
    """
    _check_dependencies()
    
    WIND_CAPACITY_MW = 20.0
    wind_coeffs = np.array(data['pricing'].get('wind_coefficient', []))
    wind_power = wind_coeffs * WIND_CAPACITY_MW
    
    grid_exchange = np.array(data['pricing'].get('grid_exchange_mw', []))
    
    # Use Density Heatmap to show operational regimes
    fig = go.Figure(go.Histogram2d(
        x=wind_power,
        y=grid_exchange,
        nbinsx=50,
        nbinsy=50,
        colorscale='Viridis',
        colorbar=dict(title='Hours of Operation')
    ))
    
    fig.update_layout(
        title=kwargs.get('title', 'Grid Interaction Phase Portrait (Density Heatmap)'),
        xaxis_title='Wind Power Available (MW)',
        yaxis_title='Grid Exchange (MW) [+Import / -Export]',
        template='plotly_white'
    )
    return fig


@log_graph_errors
def plot_lcoh_waterfall_breakdown(data: Dict[str, Any], **kwargs) -> go.Figure:
    """
    Plot LCOH breakdown waterfall chart (Economics).
    """
    _check_dependencies()
    
    # Placeholder values if not fully tracked yet
    # In real impl, sum these from 'economics' category
    energy_cost = 4.5 # €/kg
    capex = 2.0
    opex = 1.0
    water = 0.1
    compression = 0.5
    
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
        yaxis_title='Cost (€/kg H2)',
        template='plotly_white'
    )
    return fig


@log_graph_errors
def plot_pem_performance_surface(data: Dict[str, Any], **kwargs) -> go.Figure:
    """
    3D Surface: Current Density vs Time vs Voltage.
    """
    _check_dependencies()
    
    # This requires reconstructing the surface from scattered points
    # For simplicity, we'll use a 3D Scatter plot which is easier with unstructured data
    
    timestamps = data.get('timestamps', [])
    # voltage = data['pem'].get('voltage', []) # Old Z axis
    power = data['pem'].get('power_mw', [])    # New Y axis
    production = data['pem'].get('h2_production_kg_h', []) # New Z axis
    
    use_webgl = kwargs.get('use_webgl', False)
    
    fig = go.Figure(data=[go.Scatter3d(
        x=timestamps,
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
            yaxis_title='Power (MW)',
            zaxis_title='H2 Production (kg/h)'
        ),
        template='plotly_white'
    )
    return fig
