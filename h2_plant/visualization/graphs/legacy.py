"""
Legacy Graph Wrapper Module.

Wraps static_graphs.py functions with Orchestrator-compatible signatures.
This enables the GraphOrchestrator to call legacy Matplotlib functions
using the standardized handler interface.

Handler Signature:
    handler(df: pd.DataFrame, component_ids: list, title: str, config: dict) -> Figure

Legacy Signature:
    create_*_figure(df: pd.DataFrame, dpi: int = 100) -> Figure
"""
import pandas as pd
from matplotlib.figure import Figure
from typing import List, Dict, Any, Optional, Callable
import logging

logger = logging.getLogger(__name__)


def _wrap_legacy(func: Callable, default_title: str = "Graph") -> Callable:
    """
    Wrap a legacy static_graphs function with Orchestrator signature.
    
    Args:
        func: The legacy function from static_graphs.py.
        default_title: Fallback title if none provided.
        
    Returns:
        Wrapped function with Orchestrator-compatible signature.
    """
    def wrapper(df: pd.DataFrame, component_ids: List[str], 
                title: str, config: Dict[str, Any]) -> Optional[Figure]:
        try:
            # Legacy functions only need df, ignore component_ids and config
            fig = func(df)
            
            # Override title if the figure was created and has axes
            if fig is not None and fig.axes:
                fig.axes[0].set_title(title or default_title)
            
            return fig
        except Exception as e:
            logger.error(f"Legacy graph '{default_title}' failed: {e}")
            return None
    
    wrapper.__name__ = f"wrapped_{func.__name__}"
    wrapper.__doc__ = f"Orchestrator wrapper for {func.__name__}"
    return wrapper


# Import legacy functions
try:
    from h2_plant.visualization import static_graphs as sg
    
    # ===========================================================================
    # WRAPPED LEGACY HANDLERS
    # All follow the signature: (df, component_ids, title, config) -> Figure
    # ===========================================================================
    
    # Core Dispatch/Economics
    plot_dispatch = _wrap_legacy(sg.create_dispatch_figure, "Power Dispatch")
    plot_arbitrage = _wrap_legacy(sg.create_arbitrage_figure, "Arbitrage Scatter")
    plot_h2_production = _wrap_legacy(sg.create_h2_production_figure, "H2 Production")
    plot_oxygen = _wrap_legacy(sg.create_oxygen_figure, "Oxygen Production")
    plot_water = _wrap_legacy(sg.create_water_figure, "Water Consumption")
    plot_energy_pie = _wrap_legacy(sg.create_energy_pie_figure, "Energy Distribution")
    plot_histogram = _wrap_legacy(sg.create_histogram_figure, "Price Histogram")
    plot_dispatch_curve = _wrap_legacy(sg.create_dispatch_curve_figure, "Dispatch Curve")
    plot_cumulative_h2 = _wrap_legacy(sg.create_cumulative_h2_figure, "Cumulative H2")
    plot_cumulative_energy = _wrap_legacy(sg.create_cumulative_energy_figure, "Cumulative Energy")
    plot_efficiency = _wrap_legacy(sg.create_efficiency_curve_figure, "Efficiency Curve")
    plot_revenue = _wrap_legacy(sg.create_revenue_analysis_figure, "Revenue Analysis")
    plot_temporal = _wrap_legacy(sg.create_temporal_averages_figure, "Temporal Averages")
    
    # Monthly/Performance
    plot_monthly_performance = _wrap_legacy(sg.create_monthly_performance_figure, "Monthly Performance")
    plot_monthly_efficiency = _wrap_legacy(sg.create_monthly_efficiency_figure, "Monthly Efficiency")
    plot_monthly_cf = _wrap_legacy(sg.create_monthly_capacity_factor_figure, "Monthly Capacity Factor")
    
    # SOEC Operations
    plot_soec_heatmap = _wrap_legacy(sg.create_soec_module_heatmap_figure, "SOEC Module Heatmap")
    plot_soec_power_stacked = _wrap_legacy(sg.create_soec_module_power_stacked_figure, "SOEC Module Power")
    plot_soec_wear = _wrap_legacy(sg.create_soec_module_wear_figure, "SOEC Module Wear")
    
    # Thermal & Separation
    plot_water_removal = _wrap_legacy(sg.create_water_removal_total_figure, "Water Removal")
    plot_drains_discarded = _wrap_legacy(sg.create_drains_discarded_figure, "Drains Discarded")
    plot_chiller = _wrap_legacy(sg.create_chiller_cooling_figure, "Chiller Cooling")
    plot_coalescer = _wrap_legacy(sg.create_coalescer_separation_figure, "Coalescer")
    plot_kod = _wrap_legacy(sg.create_kod_separation_figure, "Knock-Out Drum")
    plot_dry_cooler = _wrap_legacy(sg.create_dry_cooler_figure, "Dry Cooler")
    
    # Energy & Analysis
    plot_energy_flows = _wrap_legacy(sg.create_energy_flows_figure, "Energy Flows")
    plot_q_breakdown = _wrap_legacy(sg.create_q_breakdown_figure, "Thermal Load Breakdown")
    plot_plant_balance = _wrap_legacy(sg.create_plant_balance_schematic, "Plant Balance")
    plot_mixer_comparison = _wrap_legacy(sg.create_mixer_comparison_figure, "Mixer Comparison")
    plot_individual_drains = _wrap_legacy(sg.create_individual_drains_figure, "Individual Drains")
    plot_dissolved_gas = _wrap_legacy(sg.create_dissolved_gas_figure, "Dissolved Gas Concentration")
    plot_drain_concentration = _wrap_legacy(sg.create_drain_concentration_figure, "Drain Concentration")
    
    # Crossover Impurities - SPECIAL HANDLER
    # Now generates single-stream graph based on stream_type config
    def _wrap_crossover():
        """Special wrapper for crossover impurities that passes stream_type and components."""
        def wrapper(df: pd.DataFrame, component_ids: List[str], 
                    title: str, config: Dict[str, Any]) -> Optional[Figure]:
            try:
                # Get stream_type (H2 or O2) and components from config
                stream_type = config.get('stream_type', 'H2')
                components = config.get('components', component_ids)
                
                fig = sg.create_crossover_impurities_figure(
                    df, 
                    stream_type=stream_type,
                    components=components
                )
                
                if fig is not None and fig.axes:
                    fig.axes[0].set_title(title)
                
                return fig
            except Exception as e:
                logger.error(f"Stream impurity graph failed: {e}")
                return None
        
        wrapper.__name__ = "wrapped_create_crossover_impurities_figure"
        return wrapper
    
    plot_crossover = _wrap_crossover()
    plot_drain_line = _wrap_legacy(sg.create_drain_line_properties_figure, "Drain Line Properties")
    plot_deoxo = _wrap_legacy(sg.create_deoxo_profile_figure, "Deoxo Profile")
    plot_drain_mixer = _wrap_legacy(sg.create_drain_mixer_figure, "Drain Mixer Balance")
    plot_drain_scheme = _wrap_legacy(sg.create_drain_scheme_schematic, "Drain Scheme")
    plot_energy_flow = _wrap_legacy(sg.create_energy_flow_figure, "Energy Flow")
    plot_process_scheme = _wrap_legacy(sg.create_process_scheme_schematic, "Process Scheme")
    plot_recirculation = _wrap_legacy(sg.create_recirculation_comparison_figure, "Recirculation")
    plot_entrained_liquid = _wrap_legacy(sg.create_entrained_liquid_figure, "Entrained Liquid")
    
    # Flow Tracking
    plot_water_vapor = _wrap_legacy(sg.create_water_vapor_tracking_figure, "Water Vapor")
    plot_total_mass_flow = _wrap_legacy(sg.create_total_mass_flow_figure, "Total Mass Flow")
    
    # Stacked Properties (Process Train) - SPECIAL HANDLERS
    # These need to pass component list to build metadata
    def _wrap_stacked_properties(func, train_tag: str):
        """Special wrapper for stacked properties that builds metadata from config."""
        def wrapper(df: pd.DataFrame, component_ids: List[str], 
                    title: str, config: Dict[str, Any]) -> Optional[Figure]:
            try:
                # Build metadata dict with system_group tags from components list
                metadata = {}
                for i, comp_id in enumerate(component_ids):
                    metadata[comp_id] = {
                        'system_group': train_tag,
                        'process_step': i
                    }
                
                # Call the underlying function with metadata
                fig = func(df, metadata=metadata)
                
                if fig is not None and fig.axes:
                    fig.axes[0].set_title(title)
                
                return fig
            except Exception as e:
                logger.error(f"Stacked properties '{title}' failed: {e}")
                return None
        
        wrapper.__name__ = f"wrapped_{func.__name__}"
        return wrapper
    
    plot_h2_stacked = _wrap_stacked_properties(sg.create_h2_stacked_properties, "H2_Train")
    plot_o2_stacked = _wrap_stacked_properties(sg.create_o2_stacked_properties, "O2_Train")
    
    LEGACY_AVAILABLE = True
    
except ImportError as e:
    logger.warning(f"Could not import static_graphs: {e}")
    LEGACY_AVAILABLE = False


# Registry of all wrapped legacy handlers for Orchestrator registration
LEGACY_HANDLERS = {
    # Core Dispatch/Economics
    'legacy_dispatch': plot_dispatch,
    'legacy_arbitrage': plot_arbitrage,
    'legacy_h2_production': plot_h2_production,
    'legacy_oxygen': plot_oxygen,
    'legacy_water': plot_water,
    'legacy_energy_pie': plot_energy_pie,
    'legacy_histogram': plot_histogram,
    'legacy_dispatch_curve': plot_dispatch_curve,
    'legacy_cumulative_h2': plot_cumulative_h2,
    'legacy_cumulative_energy': plot_cumulative_energy,
    'legacy_efficiency': plot_efficiency,
    'legacy_revenue': plot_revenue,
    'legacy_temporal': plot_temporal,
    
    # Monthly/Performance
    'legacy_monthly_performance': plot_monthly_performance,
    'legacy_monthly_efficiency': plot_monthly_efficiency,
    'legacy_monthly_cf': plot_monthly_cf,
    
    # SOEC Operations  
    'legacy_soec_heatmap': plot_soec_heatmap,
    'legacy_soec_power_stacked': plot_soec_power_stacked,
    'legacy_soec_wear': plot_soec_wear,
    
    # Thermal & Separation
    'legacy_water_removal': plot_water_removal,
    'legacy_drains_discarded': plot_drains_discarded,
    'legacy_chiller': plot_chiller,
    'legacy_coalescer': plot_coalescer,
    'legacy_kod': plot_kod,
    'legacy_dry_cooler': plot_dry_cooler,
    
    # Energy & Analysis
    'legacy_energy_flows': plot_energy_flows,
    'legacy_q_breakdown': plot_q_breakdown,
    'legacy_plant_balance': plot_plant_balance,
    'legacy_mixer_comparison': plot_mixer_comparison,
    'legacy_individual_drains': plot_individual_drains,
    'legacy_dissolved_gas': plot_dissolved_gas,
    'legacy_drain_concentration': plot_drain_concentration,
    'legacy_crossover': plot_crossover,  # Backward compat
    'legacy_h2_stream_impurity': plot_crossover,  # H2 stream (O2 impurity)
    'legacy_o2_stream_impurity': plot_crossover,  # O2 stream (H2 impurity)
    'legacy_drain_line': plot_drain_line,
    'legacy_deoxo': plot_deoxo,
    'legacy_drain_mixer': plot_drain_mixer,
    'legacy_drain_scheme': plot_drain_scheme,
    'legacy_energy_flow': plot_energy_flow,
    'legacy_process_scheme': plot_process_scheme,
    'legacy_recirculation': plot_recirculation,
    'legacy_entrained_liquid': plot_entrained_liquid,
    
    # Flow Tracking
    'legacy_water_vapor': plot_water_vapor,
    'legacy_total_mass_flow': plot_total_mass_flow,
    
    # Stacked Properties
    'legacy_h2_stacked': plot_h2_stacked,
    'legacy_o2_stacked': plot_o2_stacked,
} if LEGACY_AVAILABLE else {}
