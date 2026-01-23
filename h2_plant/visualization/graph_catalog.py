"""
GraphCatalog: Registry of available visualization graphs with metadata.

This module provides the authoritative source of truth for:
1. Graph metadata (title, description, priority, category)
2. Column requirements for memory-efficient data loading
3. Enable/disable state for YAML-driven configuration
"""

from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# COLUMN REQUIREMENTS (Migrated from run_integrated_simulation.py)
# =============================================================================
# These patterns are used by UnifiedGraphExecutor to load only required columns,
# reducing memory usage by 80-90% for year-long simulations.
#
# Pattern syntax:
#   - Exact column names: 'minute', 'P_offer'
#   - Glob patterns: '*_outlet_temp*', 'soec_module_*'
#
# Note: 'minute' is always included by the executor.

CORE_COLUMNS = [
    'minute', 'P_offer', 'P_soec_actual', 'P_pem', 'P_sold', 
    'spot_price', 'h2_kg', 'time', 'hour'
]

COLUMN_REQUIREMENTS: Dict[str, List[str]] = {
    # =========================================================================
    # CORE DISPATCH & ECONOMICS
    # =========================================================================
    'dispatch': CORE_COLUMNS + ['P_bop_mw'],
    'dispatch_strategy_stacked': CORE_COLUMNS + ['P_bop_mw'],
    'arbitrage': CORE_COLUMNS,
    'arbitrage_scatter': CORE_COLUMNS,
    'effective_ppa': CORE_COLUMNS + ['ppa_price_effective_eur_mwh'],
    
    # =========================================================================
    # H2 PRODUCTION
    # =========================================================================
    'h2_production': CORE_COLUMNS + ['H2_soec_kg', 'H2_pem_kg', 'H2_atr_kg'],
    'total_h2_production_stacked': CORE_COLUMNS + ['H2_soec_kg', 'H2_pem_kg', 'H2_atr_kg'],
    'cumulative_h2': CORE_COLUMNS + ['H2_soec_kg', 'H2_pem_kg', 'H2_atr_kg', 'cumulative_h2_kg'],
    'cumulative_h2_production': CORE_COLUMNS + [
        'H2_soec_kg', 'H2_pem_kg', 'H2_atr_kg', 'cumulative_h2_kg',
        'h2_rfnbo_kg', 'h2_non_rfnbo_kg', 'cumulative_h2_rfnbo_kg', 'cumulative_h2_non_rfnbo_kg',
        '*_PSA_*_outlet_mass_flow_kg_h'
    ],
    
    # =========================================================================
    # O2 PRODUCTION
    # =========================================================================
    'oxygen_production': CORE_COLUMNS + ['O2_pem_kg'],
    'oxygen_production_stacked': CORE_COLUMNS + ['O2_pem_kg', '*_O2_*'],
    
    # =========================================================================
    # WATER CONSUMPTION
    # =========================================================================
    'water_consumption': CORE_COLUMNS + ['steam_soec_kg', 'H2O_soec_out_kg', 'H2O_pem_kg'],
    'water_consumption_stacked': CORE_COLUMNS + ['steam_soec_kg', 'H2O_soec_out_kg', 'H2O_pem_kg'],
    'water_tank_inventory': ['minute', '*UltraPure_Tank*', '*mass_kg*', '*control_zone*'],
    
    # =========================================================================
    # ECONOMICS & ENERGY
    # =========================================================================
    'energy_pie': CORE_COLUMNS + ['compressor_power_kw'],
    'power_consumption_breakdown_pie': CORE_COLUMNS + ['compressor_power_kw', '*_cooling_load_kw'],
    'price_histogram': ['spot_price', 'minute'],
    'dispatch_curve': CORE_COLUMNS,
    'dispatch_curve_scatter': CORE_COLUMNS,
    'efficiency_curve': CORE_COLUMNS + ['H2_soec_kg', 'H2_pem_kg'],
    'revenue_analysis': CORE_COLUMNS + ['cumulative_h2_kg'],
    'temporal_averages': CORE_COLUMNS + ['H2_soec_kg', 'H2_pem_kg'],
    'cumulative_energy': CORE_COLUMNS,
    
    # =========================================================================
    # RFNBO COMPLIANCE
    # =========================================================================
    'rfnbo_compliance_stacked': CORE_COLUMNS + ['h2_rfnbo_kg', 'h2_non_rfnbo_kg', 'purchase_threshold_eur_mwh'],
    'rfnbo_spot_analysis': CORE_COLUMNS + ['purchase_threshold_eur_mwh'],
    'rfnbo_pie': ['h2_rfnbo_kg', 'h2_non_rfnbo_kg', 'minute'],
    'cumulative_rfnbo': CORE_COLUMNS + ['cumulative_h2_rfnbo_kg', 'cumulative_h2_non_rfnbo_kg'],
    
    # =========================================================================
    # SOEC MODULES
    # =========================================================================
    'soec_module_heatmap': ['minute', 'soec_module_*', 'P_soec_actual', 'soec_active_modules'],
    'soec_module_power_stacked': ['minute', 'soec_module_*', 'P_soec_actual'],
    'soec_module_wear_stats': ['minute', 'soec_module_*'],
    
    # =========================================================================
    # MONTHLY/PERFORMANCE
    # =========================================================================
    'monthly_performance': CORE_COLUMNS + ['H2_soec_kg', 'H2_pem_kg'],
    'monthly_efficiency': CORE_COLUMNS + ['H2_soec_kg', 'H2_pem_kg'],
    'monthly_capacity_factor': CORE_COLUMNS + ['H2_soec_kg', 'H2_pem_kg'],
    
    # =========================================================================
    # THERMAL & SEPARATION (Pattern-based)
    # =========================================================================
    'water_removal_total': ['minute', '*_liquid_removed_kg*', '*_water_removed*'],
    'drains_discarded': ['minute', '*_liquid_removed_kg*', '*_drain*'],
    'chiller_cooling': ['minute', '*Chiller*', '*_cooling*', '*_duty*'],
    'coalescer_separation': ['minute', '*Coalescer*'],
    'kod_separation': ['minute', '*KOD*', '*_liquid_removed*'],
    'dry_cooler_performance': ['minute', '*DryCooler*', '*Drycooler*', '*_cooling*'],
    
    # =========================================================================
    # ENERGY/SCHEMATIC
    # =========================================================================
    'energy_flows': CORE_COLUMNS + ['compressor_power_kw', '*_power_kw'],
    'energy_flow': CORE_COLUMNS + ['*_power_kw', 'compressor_power_kw'],
    'q_breakdown': ['minute', '*_cooling*', '*_duty*', '*_heat*'],
    'thermal_load_breakdown_time_series': ['minute', '*_cooling*', '*_duty*', '*Chiller*', '*Intercooler*', '*Boiler*_power_input*'],
    'plant_balance': CORE_COLUMNS + ['H2_soec_kg', 'H2_pem_kg'],
    'mixer_comparison': ['minute', '*Mixer*'],
    
    # =========================================================================
    # SEPARATION ANALYSIS
    # =========================================================================
    'individual_drains': ['minute', '*_drain*', '*_liquid*'],
    'dissolved_gas_concentration': ['minute', '*_dissolved*', '*KOD*', '*Mixer*'],
    'dissolved_gas_efficiency': ['minute', '*_dissolved*', '*Mixer*'],
    'crossover_impurities': ['minute', '*_o2_*', '*_h2_*', '*_impurity*', '*_outlet_*'],
    'drain_line_properties': ['minute', '*Drain*', '*_outlet_temp*', '*_outlet_pressure*'],
    'deoxo_profile': ['minute', '*Deoxo*'],
    'drain_mixer_balance': ['minute', '*Drain_Mixer*', '*_mass_flow*'],
    'drain_scheme': ['minute', '*Drain*', '*_mass_flow*'],
    'process_scheme': CORE_COLUMNS + ['H2_soec_kg', 'H2_pem_kg'],
    'drain_line_concentration': ['minute', '*Drain_Mixer*', '*_dissolved*'],
    'recirculation_comparison': ['minute', '*recirc*', '*recirculation*'],
    'entrained_liquid_flow': ['minute', '*_entrained*', '*_liquid*'],
    
    # =========================================================================
    # STACKED PROPERTIES (Temperature/Pressure Profiles)
    # =========================================================================
    'h2_stacked_properties': ['minute', '*SOEC_H2_*_outlet_*', '*PEM_H2_*_outlet_*', 'SOEC_Cluster_*'],
    'o2_stacked_properties': ['minute', '*O2_*_outlet_*'],
    'process_train_profile': ['minute', '*_outlet_temp*', '*_outlet_pressure*', '*_outlet_mass_flow*'],
    
    # =========================================================================
    # FLOW TRACKING
    # =========================================================================
    'water_vapor_tracking': ['minute', '*_h2o_*', '*_vapor*', '*_moisture*'],
    'total_mass_flow': ['minute', '*_mass_flow*', '*_flow_kg*'],
    
    # =========================================================================
    # STORAGE
    # =========================================================================
    'storage_levels': ['minute', 'tank_*', '*Tank*_level*', '*Tank*_pressure*'],
    'compressor_power': ['minute', '*Compressor*_power*', 'compressor_power_kw'],
    'storage_apc': ['minute', 'storage_soc', 'storage_zone', 'storage_action_factor', '*soc*', '*zone*'],
    'storage_inventory': ['minute', '*inventory_kg*', '*Tank*'],
    
    # =========================================================================
    # PLOTLY GRAPHS (use same patterns, exec will adapt to dict format)
    # =========================================================================
    'pem_h2_production_over_time': CORE_COLUMNS + ['H2_pem_kg'],
    'soec_h2_production_over_time': CORE_COLUMNS + ['H2_soec_kg'],
    'pem_cell_voltage_over_time': CORE_COLUMNS,
    'pem_efficiency_over_time': CORE_COLUMNS,
    'energy_price_over_time': CORE_COLUMNS,
    'soec_active_modules_over_time': ['minute', 'soec_active_modules', 'soec_module_*'],
}


def get_columns_for_graph(graph_id: str) -> List[str]:
    """
    Get column requirements for a specific graph.
    
    Returns list of column names/patterns. Returns ['minute'] if graph not found
    (allows graceful fallback with minimal data).
    """
    return COLUMN_REQUIREMENTS.get(graph_id, ['minute'])


class GraphPriority(Enum):
    """Priority levels for graph generation."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class GraphLibrary(Enum):
    """Supported visualization libraries."""
    PLOTLY = "plotly"
    SEABORN = "seaborn"
    MATPLOTLIB = "matplotlib"


@dataclass
class GraphMetadata:
    """
    Metadata for a visualization graph.
    
    Attributes:
        data_required: Column patterns needed. If ['history'] is passed,
                       it will be auto-resolved using get_columns_for_graph().
    """
    graph_id: str
    title: str
    description: str
    function: Callable
    library: GraphLibrary
    data_required: List[str]
    priority: GraphPriority
    category: str
    enabled: bool = True
    kwargs: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}
        
        # Auto-resolve ['history'] placeholder to actual column requirements
        if self.data_required == ['history']:
            resolved = get_columns_for_graph(self.graph_id)
            if resolved != ['minute']:  # Found in COLUMN_REQUIREMENTS
                self.data_required = resolved
            # else: keep ['history'] for legacy fallback (loads all columns)


class GraphCatalog:
    """
    Registry and manager for all available visualization graphs.
    
    Provides:
    - Graph metadata and discovery
    - Enable/disable functionality per graph
    - Category-based filtering
    - Priority-based sorting
    """
    
    def __init__(self):
        """Initialize the graph catalog."""
        self._registry: Dict[str, GraphMetadata] = {}
        self._enabled_graphs: set = set()
        self._load_default_registry()
        logger.info(f"GraphCatalog initialized with {len(self._registry)} graphs")
    
    def register(self, metadata: GraphMetadata) -> None:
        """
        Register a new graph in the catalog.
        
        Args:
            metadata: Graph metadata including function and requirements
        """
        self._registry[metadata.graph_id] = metadata
        if metadata.enabled:
            self._enabled_graphs.add(metadata.graph_id)
        logger.debug(f"Registered graph: {metadata.graph_id}")
    
    def enable(self, graph_id: str) -> None:
        """
        Enable a specific graph for generation.
        
        Args:
            graph_id: ID of the graph to enable
        """
        if graph_id in self._registry:
            self._registry[graph_id].enabled = True
            self._enabled_graphs.add(graph_id)
            logger.info(f"Enabled graph: {graph_id}")
        else:
            logger.warning(f"Cannot enable unknown graph: {graph_id}")
    
    def disable(self, graph_id: str) -> None:
        """
        Disable a specific graph from generation.
        
        Args:
            graph_id: ID of the graph to disable
        """
        if graph_id in self._registry:
            self._registry[graph_id].enabled = False
            self._enabled_graphs.discard(graph_id)
            logger.info(f"Disabled graph: {graph_id}")
        else:
            logger.warning(f"Cannot disable unknown graph: {graph_id}")
    
    def enable_category(self, category: str) -> None:
        """Enable all graphs in a category."""
        count = 0
        for graph_id, metadata in self._registry.items():
            if metadata.category == category:
                self.enable(graph_id)
                count += 1
        logger.info(f"Enabled {count} graphs in category: {category}")
    
    def disable_category(self, category: str) -> None:
        """Disable all graphs in a category."""
        count = 0
        for graph_id, metadata in self._registry.items():
            if metadata.category == category:
                self.disable(graph_id)
                count += 1
        logger.info(f"Disabled {count} graphs in category: {category}")
    
    def enable_all(self) -> None:
        """Enable all registered graphs."""
        for graph_id in self._registry:
            self.enable(graph_id)
    
    def disable_all(self) -> None:
        """Disable all registered graphs."""
        self._enabled_graphs.clear()
        for metadata in self._registry.values():
            metadata.enabled = False
        
    def get(self, graph_id: str) -> Optional[GraphMetadata]:
        """Get metadata for a specific graph."""
        return self._registry.get(graph_id)
        
    def __contains__(self, graph_id: str) -> bool:
        """Check if graph exists in registry."""
        return graph_id in self._registry

    def get_enabled(self) -> List[GraphMetadata]:
        """
        Get list of all enabled graph metadata.
        
        Returns:
            List of GraphMetadata objects for enabled graphs, sorted by priority.
        """
        enabled = [self._registry[gid] for gid in self._enabled_graphs if gid in self._registry]
        
        # Sort by priority (CRITICAL -> LOW)
        priority_order = {
            GraphPriority.CRITICAL: 0,
            GraphPriority.HIGH: 1,
            GraphPriority.MEDIUM: 2,
            GraphPriority.LOW: 3
        }
        
        return sorted(enabled, key=lambda m: priority_order.get(m.priority, 99))
    
    def get_by_category(self, category: str) -> List[GraphMetadata]:
        """Get all graphs in a specific category."""
        return [
            metadata for metadata in self._registry.values()
            if metadata.category == category
        ]
    
    def get_by_priority(self, priority: GraphPriority) -> List[GraphMetadata]:
        """Get all graphs with a specific priority."""
        return [
            metadata for metadata in self._registry.values()
            if metadata.priority == priority
        ]
    
    def list_categories(self) -> List[str]:
        """List all unique categories."""
        return list(set(m.category for m in self._registry.values()))
    
    def list_enabled(self) -> List[str]:
        """List IDs of all enabled graphs."""
        return list(self._enabled_graphs)
    
    def summary(self) -> Dict[str, Any]:
        """Return summary of the catalog."""
        return {
            'total_graphs': len(self._registry),
            'enabled_graphs': len(self._enabled_graphs),
            'categories': self.list_categories(),
            'by_priority': {
                'CRITICAL': len(self.get_by_priority(GraphPriority.CRITICAL)),
                'HIGH': len(self.get_by_priority(GraphPriority.HIGH)),
                'MEDIUM': len(self.get_by_priority(GraphPriority.MEDIUM)),
                'LOW': len(self.get_by_priority(GraphPriority.LOW))
            }
        }
    
    def _load_default_registry(self) -> None:
        """
        Load default graph registry.
        
        Matplotlib graphs use column patterns from COLUMN_REQUIREMENTS.
        Plotly graphs are registered but DISABLED by default (require dict data format).
        """
        from h2_plant.visualization import plotly_graphs as pg
        from h2_plant.visualization import static_graphs as sg
        
        # =====================================================================
        # PLOTLY GRAPHS (Disabled by default - require structured dict data)
        # =====================================================================
        # These will be enabled post-refactor when Plotly data adapter is ready.
        
        # =====================================================================
        # PLOTLY EXPLICIT GRAPHS (Refactored to accept DataFrame)
        # =====================================================================
        
        # 1. Dispatch Strategy (Twin for legacy 'dispatch')
        self.register(GraphMetadata(
            graph_id='dispatch_plotly',
            title='Power Dispatch Strategy (Interactive)',
            description='Stacked power consumption and grid sales',
            function=pg.plot_dispatch_strategy,
            library=GraphLibrary.PLOTLY,
            data_required=['P_pem', 'P_soec', 'P_sold', 'spot_price'],
            priority=GraphPriority.HIGH,
            category='economics',
            enabled=True  # P2 FIX: Enabled for dual-mode
        ))

        # 2. Cumulative Production (Twin for 'cumulative_h2')
        self.register(GraphMetadata(
            graph_id='cumulative_h2_plotly',
            title='Cumulative H2 Production (Interactive)',
            description='Cumulative hydrogen production from both systems',
            function=pg.plot_cumulative_production,
            library=GraphLibrary.PLOTLY,
            # Use ACTUAL columns from parquet:
            # - cumulative_h2_kg (total), cumulative_h2_rfnbo_kg, cumulative_h2_non_rfnbo_kg
            # - H2_pem, H2_soec, H2_atr_kg (rates for per-source breakdown)
            data_required=['cumulative_h2_kg', 'cumulative_h2_rfnbo_kg', 'cumulative_h2_non_rfnbo_kg',
                          'H2_pem', 'H2_soec', 'H2_atr_kg'],
            priority=GraphPriority.HIGH,
            category='production',
            enabled=True  # P2 FIX: Enabled for dual-mode
        ))

        # 3. Energy Pie (Twin for 'energy_pie')
        self.register(GraphMetadata(
            graph_id='energy_pie_plotly',
            title='Power Consumption Breakdown (Interactive)',
            description='Pie chart showing power distribution',
            function=pg.plot_power_breakdown_pie,
            library=GraphLibrary.PLOTLY,
            data_required=['P_pem', 'P_soec', 'compressor_power_kw'],
            priority=GraphPriority.MEDIUM,
            category='economics',
            enabled=True  # P2 FIX: Enabled for dual-mode
        ))
        
        # 4. Arbitrage (Twin for 'arbitrage')
        self.register(GraphMetadata(
            graph_id='arbitrage_plotly',
            title='Arbitrage Opportunity (Interactive)',
            description='Interactive dual-axis chart of H2 production vs Prices',
            function=pg.plot_arbitrage_opportunity,
            library=GraphLibrary.PLOTLY,
            data_required=['spot_price', 'ppa_price', 'H2_soec', 'H2_pem'],
            priority=GraphPriority.HIGH,
            category='economics',
            enabled=True  # P2 FIX: Enabled for dual-mode
        ))

        # 5. PEM Efficiency (Twin for 'legacy_efficiency' - partial)
        self.register(GraphMetadata(
            graph_id='efficiency_curve_plotly',
            title='PEM Efficiency (Interactive)',
            description='PEM system efficiency (%) over time',
            function=pg.plot_pem_efficiency_timeline,
            library=GraphLibrary.PLOTLY,
            data_required=['pem_efficiency'],
            priority=GraphPriority.HIGH,
            category='performance',
            enabled=True  # P2 FIX: Enabled for dual-mode
        ))

        # 6. Wind Duration (New/Twin)
        self.register(GraphMetadata(
            graph_id='wind_duration_plotly',
            title='Wind Utilization Duration Curve',
            description='Duration curve of wind power usage',
            function=pg.plot_wind_utilization_duration_curve,
            library=GraphLibrary.PLOTLY,
            data_required=['wind_coefficient', 'P_pem', 'P_soec'],
            priority=GraphPriority.HIGH,
            category='grid_integration',
            enabled=True  # P2 FIX: Enabled for dual-mode
        ))
        
        # P2 FIX: Missing Plotly twins (identified as unreachable by audit)
        # These now mirror their Matplotlib counterparts and will be auto-enabled by dual-mode logic.
        
        # 7. Storage APC (Twin for 'storage_apc')
        self.register(GraphMetadata(
            graph_id='storage_apc_plotly',
            title='Storage APC Control (Interactive)',
            description='State of Charge, control zones, and power modulation factor',
            function=pg.plot_storage_apc,  # P2 FIX: Now points to dedicated implementation
            library=GraphLibrary.PLOTLY,
            data_required=['storage_soc', 'storage_zone', 'storage_action_factor'],
            priority=GraphPriority.HIGH,
            category='storage',
            enabled=True
        ))
        
        # 8. Temporal Averages (Twin for 'temporal_averages')
        self.register(GraphMetadata(
            graph_id='temporal_averages_plotly',
            title='Temporal Averages (Interactive)',
            description='Hourly aggregated price, power, H2 data',
            function=pg.plot_temporal_averages,  # P2 FIX: Now points to dedicated implementation
            library=GraphLibrary.PLOTLY,
            data_required=['spot_price', 'P_pem', 'P_soec', 'H2_pem_kg', 'H2_soec_kg'],
            priority=GraphPriority.MEDIUM,
            category='legacy',
            enabled=True
        ))
        
        # 9. Effective PPA (Twin for 'effective_ppa')
        self.register(GraphMetadata(
            graph_id='effective_ppa_plotly',
            title='Effective PPA Price (Interactive)',
            description='Weighted average PPA price over time',
            function=pg.plot_effective_ppa,  # P2 FIX: Now points to dedicated implementation
            library=GraphLibrary.PLOTLY,
            data_required=['ppa_price_effective_eur_mwh', 'spot_price'],
            priority=GraphPriority.HIGH,
            category='economics',
            enabled=True
        ))
        
        # Keep existing explicit IDs for direct access if needed
        self.register(GraphMetadata(
            graph_id='pem_h2_production_over_time',
            title='PEM H2 Production Rate',
            description='PEM hydrogen production rate (kg/h) over time',
            function=pg.plot_pem_production_timeline,
            library=GraphLibrary.PLOTLY,
            data_required=['H2_pem_kg'],
            priority=GraphPriority.CRITICAL,
            category='production',
            enabled=False 
        ))
        
        self.register(GraphMetadata(
            graph_id='soec_h2_production_over_time',
            title='SOEC H2 Production Rate',
            description='SOEC hydrogen production rate (kg/h) over time',
            function=pg.plot_soec_production_timeline,
            library=GraphLibrary.PLOTLY,
            data_required=['soec.h2_production_kg_h', 'timestamps'],
            priority=GraphPriority.CRITICAL,
            category='production',
            enabled=False  # Plotly disabled
        ))
        
        self.register(GraphMetadata(
            graph_id='total_h2_production_stacked',
            title='Total H2 Production (Stacked)',
            description='Stacked area chart showing PEM + SOEC contributions',
            function=pg.plot_total_production_stacked,
            library=GraphLibrary.PLOTLY,
            data_required=['pem.h2_production_kg_h', 'soec.h2_production_kg_h', 'timestamps'],
            priority=GraphPriority.CRITICAL,
            category='production',
            enabled=False  # Plotly disabled
        ))
        
        self.register(GraphMetadata(
            graph_id='cumulative_h2_production',
            title='Cumulative H2 Production',
            description='Cumulative hydrogen production from both systems',
            function=pg.plot_cumulative_production,
            library=GraphLibrary.PLOTLY,
            data_required=['pem.cumulative_h2_kg', 'soec.cumulative_h2_kg', 'timestamps'],
            priority=GraphPriority.HIGH,
            category='production',
            enabled=False  # Plotly disabled
        ))
        
        # Voltage & Efficiency
        self.register(GraphMetadata(
            graph_id='pem_cell_voltage_over_time',
            title='PEM Cell Voltage',
            description='PEM cell voltage (V) over time',
            function=pg.plot_pem_voltage_timeline,
            library=GraphLibrary.PLOTLY,
            data_required=['pem.voltage', 'timestamps'],
            priority=GraphPriority.CRITICAL,
            category='performance',
            enabled=False  # Plotly disabled
        ))
        
        self.register(GraphMetadata(
            graph_id='pem_efficiency_over_time',
            title='PEM System Efficiency',
            description='PEM system efficiency (%) over time',
            function=pg.plot_pem_efficiency_timeline,
            library=GraphLibrary.PLOTLY,
            data_required=['pem.efficiency', 'timestamps'],
            priority=GraphPriority.HIGH,
            category='performance',
            enabled=False  # Plotly disabled
        ))
        
        # Energy Economics
        self.register(GraphMetadata(
            graph_id='energy_price_over_time',
            title='Energy Price Timeline',
            description='Electricity price (€/kWh) over time',
            function=pg.plot_energy_price_timeline,
            library=GraphLibrary.PLOTLY,
            data_required=['pricing.energy_price_eur_kwh', 'timestamps'],
            priority=GraphPriority.HIGH,
            category='economics',
            enabled=False  # Plotly disabled
        ))
        
        self.register(GraphMetadata(
            graph_id='dispatch_strategy_stacked',
            title='Dispatch Strategy',
            description='Stacked chart: power to PEM, SOEC, and grid',
            function=pg.plot_dispatch_strategy,
            library=GraphLibrary.PLOTLY,
            data_required=['coordinator.pem_setpoint_mw', 'coordinator.soec_setpoint_mw', 'coordinator.sell_power_mw', 'timestamps'],
            priority=GraphPriority.CRITICAL,
            category='economics',
            enabled=False  # Plotly disabled
        ))
        
        self.register(GraphMetadata(
            graph_id='power_consumption_breakdown_pie',
            title='Power Consumption Breakdown',
            description='Pie chart showing power distribution',
            function=pg.plot_power_breakdown_pie,
            library=GraphLibrary.PLOTLY,
            data_required=['pem.cumulative_energy_kwh', 'soec.cumulative_energy_kwh', 'compression.total_energy_kwh'],
            priority=GraphPriority.MEDIUM,
            category='economics',
            enabled=False  # Plotly disabled
        ))
        
        # SOEC Operations
        self.register(GraphMetadata(
            graph_id='soec_active_modules_over_time',
            title='SOEC Active Modules',
            description='Number of active SOEC modules over time',
            function=pg.plot_soec_modules_timeline,
            library=GraphLibrary.PLOTLY,
            data_required=['soec.active_modules', 'timestamps'],
            priority=GraphPriority.HIGH,
            category='soec_ops',
            enabled=False  # Plotly disabled
        ))
        
        # Reliability & Stress Analysis
        self.register(GraphMetadata(
            graph_id='storage_fatigue_cycling_3d',
            title='Storage Tank Fatigue Cycling (3D)',
            description='3D surface of tank pressures over time',
            function=pg.plot_storage_fatigue_cycling_3d,
            library=GraphLibrary.PLOTLY,
            data_required=['tanks.hp_pressures', 'timestamps'],
            priority=GraphPriority.CRITICAL,
            category='reliability',
            enabled=False  # Plotly disabled
        ))
        self.register(GraphMetadata(
            graph_id='ramp_rate_stress_distribution',
            title='Ramp Rate Stress Distribution',
            description='Violin plot of component ramp rates',
            function=pg.plot_ramp_rate_stress_distribution,
            library=GraphLibrary.PLOTLY,
            data_required=['soec.ramp_rates'],
            priority=GraphPriority.HIGH,
            category='reliability',
            enabled=False  # Plotly disabled
        ))
        
        # Grid & Renewables Integration
        self.register(GraphMetadata(
            graph_id='wind_utilization_duration_curve',
            title='Wind Utilization Duration Curve',
            description='Duration curve of wind power usage',
            function=pg.plot_wind_utilization_duration_curve,
            library=GraphLibrary.PLOTLY,
            # Use P_offer (actual renewable power) instead of wind_coefficient
            # Need 'minute' to calculate true duration if downsampled
            data_required=['P_offer', 'pem_power_mw', 'soec_power_mw', 'minute'],
            priority=GraphPriority.HIGH,
            category='grid_integration',
            enabled=False  # Plotly disabled
        ))
        self.register(GraphMetadata(
            graph_id='grid_interaction_phase_portrait',
            title='Grid Interaction Phase Portrait',
            description='Phase portrait of Grid Exchange vs Wind Power',
            function=pg.plot_grid_interaction_phase_portrait,
            library=GraphLibrary.PLOTLY,
            data_required=['pricing.wind_coefficient', 'pricing.grid_exchange_mw'],
            priority=GraphPriority.HIGH,
            category='grid_integration',
            enabled=False  # Plotly disabled
        ))
        
        # Economic Deep Dive
        self.register(GraphMetadata(
            graph_id='lcoh_waterfall_breakdown',
            title='LCOH Breakdown',
            description='Waterfall chart of LCOH components',
            function=pg.plot_lcoh_waterfall_breakdown,
            library=GraphLibrary.PLOTLY,
            data_required=['economics.lcoh_cumulative'], # Placeholder requirement
            priority=GraphPriority.CRITICAL,
            category='economics',
            enabled=False  # Plotly disabled
        ))
        
        # Advanced Multi-Dimensional
        self.register(GraphMetadata(
            graph_id='pem_performance_surface',
            title='PEM Performance Surface',
            description='3D Surface of PEM performance',
            function=pg.plot_pem_performance_surface,
            library=GraphLibrary.PLOTLY,
            data_required=['pem.h2_production_kg_h', 'pem.power_mw', 'timestamps'],
            priority=GraphPriority.MEDIUM,
            category='advanced',
            enabled=False  # Plotly disabled
        ))

        # Storage (Placeholder)
        self.register(GraphMetadata(
            graph_id='tank_storage_timeline',
            title='Tank Storage Levels',
            description='Storage levels over time',
            function=pg.plot_tank_storage_timeline,
            library=GraphLibrary.PLOTLY,
            data_required=['tanks.total_stored', 'timestamps'],
            priority=GraphPriority.HIGH,
            category='storage',
            enabled=False
        ))

        self.register(GraphMetadata(
            graph_id='arbitrage_opportunity_interactive',
            title='Arbitrage Opportunity (Interactive)',
            description='Interactive dual-axis chart of H2 production vs Prices',
            function=pg.plot_arbitrage_opportunity,
            library=GraphLibrary.PLOTLY,
            data_required=['minute', '*Spot*', '*price*', 'H2_soec*', 'H2_pem*'],
            priority=GraphPriority.HIGH,
            category='economics',
            enabled=True
        ))

        # --- MATPLOTLIB GRAPHS (Merged from plotter.py) ---
        self.register(GraphMetadata(
            graph_id='dispatch',
            title='Power Dispatch',
            description='Stacked power consumption and grid sales',
            function=sg.create_dispatch_figure,
            library=GraphLibrary.MATPLOTLIB,
            data_required=['history'],
            priority=GraphPriority.HIGH,
            category='legacy',
            enabled=True
        ))
        self.register(GraphMetadata(
            graph_id='arbitrage',
            title='Price Scenario',
            description='Spot prices, PPA, and H2 breakeven',
            function=sg.create_arbitrage_figure,
            library=GraphLibrary.MATPLOTLIB,
            data_required=['history'],
            priority=GraphPriority.MEDIUM,
            category='legacy',
            enabled=True
        ))
        self.register(GraphMetadata(
            graph_id='h2_production',
            title='H2 Production Rate',
            description='Hydrogen production rates by source',
            function=sg.create_h2_production_figure,
            library=GraphLibrary.MATPLOTLIB,
            data_required=['history'],
            priority=GraphPriority.MEDIUM,
            category='legacy',
            enabled=True
        ))
        self.register(GraphMetadata(
            graph_id='oxygen_production',
            title='O2 Production',
            description='Oxygen co-production rates',
            function=sg.create_oxygen_figure,
            library=GraphLibrary.MATPLOTLIB,
            data_required=['history'],
            priority=GraphPriority.LOW,
            category='legacy',
            enabled=True
        ))
        self.register(GraphMetadata(
            graph_id='water_consumption',
            title='Water Consumption',
            description='Total water usage including losses',
            function=sg.create_water_figure,
            library=GraphLibrary.MATPLOTLIB,
            data_required=['history'],
            priority=GraphPriority.MEDIUM,
            category='legacy',
            enabled=True
        ))
        self.register(GraphMetadata(
            graph_id='energy_pie',
            title='Energy Distribution',
            description='Donut chart of energy breakdown',
            function=sg.create_energy_pie_figure,
            library=GraphLibrary.MATPLOTLIB,
            data_required=['history'],
            priority=GraphPriority.LOW,
            category='legacy',
            enabled=True
        ))
        self.register(GraphMetadata(
            graph_id='price_histogram',
            title='Price Histogram',
            description='Distribution of spot prices',
            function=sg.create_histogram_figure,
            library=GraphLibrary.MATPLOTLIB,
            data_required=['history'],
            priority=GraphPriority.LOW,
            category='legacy',
            enabled=True
        ))
        self.register(GraphMetadata(
            graph_id='dispatch_curve',
            title='Dispatch Curve',
            description='H2 output vs power input scatter',
            function=sg.create_dispatch_curve_figure,
            library=GraphLibrary.MATPLOTLIB,
            data_required=['history'],
            priority=GraphPriority.LOW,
            category='legacy',
            enabled=True
        ))
        self.register(GraphMetadata(
            graph_id='cumulative_h2',
            title='Cumulative H2',
            description='Total hydrogen produced over time',
            function=sg.create_cumulative_h2_figure,
            library=GraphLibrary.MATPLOTLIB,
            data_required=['history'],
            priority=GraphPriority.HIGH,
            category='legacy',
            enabled=True
        ))
        self.register(GraphMetadata(
            graph_id='cumulative_energy',
            title='Cumulative Energy',
            description='Total energy consumed and sold',
            function=sg.create_cumulative_energy_figure,
            library=GraphLibrary.MATPLOTLIB,
            data_required=['history'],
            priority=GraphPriority.HIGH,
            category='legacy',
            enabled=True
        ))
        self.register(GraphMetadata(
            graph_id='efficiency_curve',
            title='Efficiency Curve',
            description='System efficiency (LHV) over time',
            function=sg.create_efficiency_curve_figure,
            library=GraphLibrary.MATPLOTLIB,
            data_required=['history'],
            priority=GraphPriority.HIGH,
            category='legacy',
            enabled=True
        ))
        self.register(GraphMetadata(
            graph_id='revenue_analysis',
            title='Revenue Analysis',
            description='Grid revenue vs H2 value comparison',
            function=sg.create_revenue_analysis_figure,
            library=GraphLibrary.MATPLOTLIB,
            data_required=['history'],
            priority=GraphPriority.MEDIUM,
            category='legacy',
            enabled=True
        ))
        self.register(GraphMetadata(
            graph_id='temporal_averages',
            title='Temporal Averages',
            description='Hourly aggregated price, power, H2 data',
            function=sg.create_temporal_averages_figure,
            library=GraphLibrary.MATPLOTLIB,
            data_required=['history'],
            priority=GraphPriority.MEDIUM,
            category='legacy',
            enabled=True
        ))
        
        # Storage APC (Advanced Process Control)
        self.register(GraphMetadata(
            graph_id='storage_apc',
            title='Storage APC Control',
            description='State of Charge, control zones, and power modulation factor',
            function=sg.create_storage_apc_figure,
            library=GraphLibrary.MATPLOTLIB,
            data_required=['storage_soc', 'storage_zone', 'storage_action_factor'],
            priority=GraphPriority.HIGH,
            category='storage',
            enabled=True
        ))
        
        # Effective PPA Pricing (Dual Contract/Variable)
        self.register(GraphMetadata(
            graph_id='effective_ppa',
            title='Effective PPA Price',
            description='Weighted average PPA price over time (contract + variable)',
            function=sg.create_effective_ppa_figure,
            library=GraphLibrary.MATPLOTLIB,
            data_required=['ppa_price_effective_eur_mwh'],
            priority=GraphPriority.HIGH,
            category='economics',
            enabled=True
        ))
        
        # Thermal & Separation (Static)
        self.register(GraphMetadata(
            graph_id='legacy_chiller',
            title='Chiller Cooling Load',
            description='Chiller cooling load over time (kW)',
            function=sg.create_chiller_cooling_load_figure,
            library=GraphLibrary.MATPLOTLIB,
            data_required=['*Chiller*', '*cooling_load*'],
            priority=GraphPriority.MEDIUM,
            category='thermal',
            enabled=True
        ))
        
        self.register(GraphMetadata(
            graph_id='legacy_chiller_power',
            title='Chiller Electrical Power',
            description='Chiller electrical power consumption over time (kW)',
            function=sg.create_chiller_power_figure,
            library=GraphLibrary.MATPLOTLIB,
            data_required=['*Chiller*', '*electrical_power*'],
            priority=GraphPriority.MEDIUM,
            category='thermal',
            enabled=True
        ))
        
        # Plotly Twins: Chiller & DryCooler
        self.register(GraphMetadata(
            graph_id='chiller_cooling_plotly',
            title='Chiller Cooling Load (Interactive)',
            description='Interactive chiller cooling load over time (kW)',
            function=pg.plot_chiller_cooling_load,
            library=GraphLibrary.PLOTLY,
            data_required=['*Chiller*cooling_load*'],
            priority=GraphPriority.MEDIUM,
            category='thermal',
            enabled=True
        ))
        
        self.register(GraphMetadata(
            graph_id='chiller_power_plotly',
            title='Chiller Electrical Power (Interactive)',
            description='Interactive chiller electrical power consumption (kW)',
            function=pg.plot_chiller_power,
            library=GraphLibrary.PLOTLY,
            data_required=['*Chiller*electrical_power*'],
            priority=GraphPriority.MEDIUM,
            category='thermal',
            enabled=True
        ))
        
        self.register(GraphMetadata(
            graph_id='dry_cooler_performance_plotly',
            title='Dry Cooler Performance (Interactive)',
            description='Interactive dry cooler/intercooler heat rejection and outlet temperature',
            function=pg.plot_dry_cooler_performance,
            library=GraphLibrary.PLOTLY,
            data_required=['*Cooler*heat_rejected*', '*Intercooler*heat_rejected*', '*Cooler*outlet_temp*', '*Intercooler*outlet_temp*'],
            priority=GraphPriority.MEDIUM,
            category='thermal',
            enabled=True
        ))
        
        # Q Breakdown (Average Bar Chart)
        self.register(GraphMetadata(
            graph_id='q_breakdown_plotly',
            title='Thermal Load Breakdown (Avg)',
            description='Average thermal load by subsystem and heat type',
            function=pg.plot_q_breakdown,
            library=GraphLibrary.PLOTLY,
            data_required=['minute', '*_cooling*', '*_duty*', '*_heat*', '*_power_input*'],
            priority=GraphPriority.MEDIUM,
            category='thermal',
            enabled=True
        ))

        # Thermal Load Time Series (Hierarchical Legend + Dropdown)
        self.register(GraphMetadata(
            graph_id='thermal_load_breakdown_time_series_plotly',
            title='Thermal Load Time Series',
            description='Dynamic cooling/heating load profile with hierarchical legend and subsystem filters',
            function=pg.plot_thermal_load_breakdown_timeseries,
            library=GraphLibrary.PLOTLY,
            data_required=['minute', '*_cooling*', '*_duty*', '*_heat_rejected*', '*Boiler*_power_input*'],
            priority=GraphPriority.MEDIUM,
            category='thermal',
            enabled=True
        ))

        # Stacked Thermal Load Time Series
        # DISABLED: Now merged into plot_thermal_load_time_series with toggle
        self.register(GraphMetadata(
            graph_id='thermal_load_stacked_time_series_plotly',
            title='Stacked Thermal Load Time Series',
            description='Stacked area chart showing thermal load composition over time',
            function=pg.plot_thermal_load_stacked_timeseries,
            library=GraphLibrary.PLOTLY,
            data_required=['minute', '*_cooling*', '*_duty*', '*_heat_rejected*', '*Boiler*_power_input*'],
            priority=GraphPriority.MEDIUM,
            category='thermal',
            enabled=False  # Merged with toggle
        ))

        # Enhanced Stacked Graphs with UX improvements
        # DISABLED: Now merged into plot_chiller_cooling_load with toggle
        self.register(GraphMetadata(
            graph_id='chiller_cooling_load_stacked_plotly',
            title='Chiller Cooling Load - Stacked',
            description='Stacked area chart of chiller cooling loads with enhanced UX',
            function=pg.plot_chiller_cooling_load_stacked,
            library=GraphLibrary.PLOTLY,
            data_required=['minute', '*Chiller*_cooling_load_kw'],
            priority=GraphPriority.MEDIUM,
            category='thermal',
            enabled=False  # Merged with toggle
        ))

        # DISABLED: Now merged into plot_chiller_power with toggle
        self.register(GraphMetadata(
            graph_id='chiller_power_stacked_plotly',
            title='Chiller Electrical Power - Stacked',
            description='Stacked area chart of chiller electrical power with enhanced UX',
            function=pg.plot_chiller_power_stacked,
            library=GraphLibrary.PLOTLY,
            data_required=['minute', '*Chiller*_electrical_power_kw'],
            priority=GraphPriority.MEDIUM,
            category='thermal',
            enabled=False  # Merged with toggle
        ))

        # DISABLED: Now merged into plot_cumulative_production with toggle
        self.register(GraphMetadata(
            graph_id='cumulative_h2_stacked_plotly',
            title='Cumulative H2 Production - Stacked',
            description='Stacked area chart showing H2 production composition by source',
            function=pg.plot_cumulative_production_stacked,
            library=GraphLibrary.PLOTLY,
            data_required=['minute', 'H2_pem_kg', 'H2_soec_kg'],
            priority=GraphPriority.HIGH,
            category='production',
            enabled=False  # Merged with toggle
        ))

        # DISABLED: Now merged into plot_dry_cooler_performance with toggle
        self.register(GraphMetadata(
            graph_id='dry_cooler_stacked_plotly',
            title='Dry Cooler Heat Rejection - Stacked',
            description='Stacked area chart of dry cooler heat rejection with enhanced UX',
            function=pg.plot_dry_cooler_stacked,
            library=GraphLibrary.PLOTLY,
            data_required=['minute', '*DryCooler*_heat_rejected_kw', '*Intercooler*_heat_rejected_kw'],
            priority=GraphPriority.MEDIUM,
            category='thermal',
            enabled=False  # Merged with toggle
        ))
        
        self.register(GraphMetadata(
            graph_id='coalescer_separation',
            title='Coalescer Separation',
            description='Pressure drop and liquid removal',
            function=sg.create_coalescer_separation_figure,
            library=GraphLibrary.MATPLOTLIB,
            data_required=['history'],
            priority=GraphPriority.MEDIUM,
            category='separation',
            enabled=True
        ))
        
        # Coalescer Plotly Twin
        self.register(GraphMetadata(
            graph_id='coalescer_separation_plotly',
            title='Coalescer Separation (Interactive)',
            description='Plotly: Pressure drop and liquid removal with enhanced UX',
            function=pg.plot_coalescer_separation,
            library=GraphLibrary.PLOTLY,
            data_required=['minute', '*Coalescer*_delta_p_bar', '*Coalescer*_drain_flow_kg_h'],
            priority=GraphPriority.MEDIUM,
            category='separation',
            enabled=True
        ))
        
        self.register(GraphMetadata(
            graph_id='kod_separation',
            title='Knock-Out Drum',
            description='Gas density, velocity, and liquid drainage',
            function=sg.create_kod_separation_figure,
            library=GraphLibrary.MATPLOTLIB,
            data_required=['history'],
            priority=GraphPriority.MEDIUM,
            category='separation',
            enabled=True
        ))
        
        # KOD Plotly Twin
        self.register(GraphMetadata(
            graph_id='kod_separation_plotly',
            title='Knock-Out Drum (Interactive)',
            description='Plotly: Gas density, velocity, and liquid drainage with enhanced UX',
            function=pg.plot_kod_separation,
            library=GraphLibrary.PLOTLY,
            data_required=['minute', '*KOD*_rho_g', '*KOD*_v_real', '*KOD*_water_removed_kg_h'],
            priority=GraphPriority.MEDIUM,
            category='separation',
            enabled=True
        ))
        
        # Mixer Comparison Plotly (no Matplotlib twin listed here, but adding it)
        self.register(GraphMetadata(
            graph_id='mixer_comparison_plotly',
            title='Drain/Mixer Comparison (Interactive)',
            description='Plotly: Temperature, pressure, and flow comparison with enhanced UX',
            function=pg.plot_mixer_comparison,
            library=GraphLibrary.PLOTLY,
            data_required=['minute', '*Mixer*', '*DrainRecorder*', '*Combiner*'],
            priority=GraphPriority.MEDIUM,
            category='separation',
            enabled=True
        ))
        
        # Process Train Profile Plotly
        self.register(GraphMetadata(
            graph_id='process_train_profile_plotly',
            title='Process Train Profile (Interactive)',
            description='Plotly: Temperature, Pressure, and Composition profiles along H2/O2 trains with selector',
            function=pg.plot_process_train_profile,
            library=GraphLibrary.PLOTLY,
            data_required=['history'],
            priority=GraphPriority.HIGH,
            category='profile',
            enabled=True
        ))
        self.register(GraphMetadata(
            graph_id='dry_cooler_performance',
            title='Dry Cooler Performance',
            description='Heat rejection, fan power, and outlet temperature',
            function=sg.create_dry_cooler_figure,
            library=GraphLibrary.MATPLOTLIB,
            data_required=['history'],
            priority=GraphPriority.MEDIUM,
            category='thermal',
            enabled=True
        ))
        self.register(GraphMetadata(
            graph_id='water_removal_total',
            title='Total Water Removal',
            description='Total liquid water removed by each separation component',
            function=sg.create_water_removal_total_figure,
            library=GraphLibrary.MATPLOTLIB,
            data_required=['history'],
            priority=GraphPriority.HIGH,
            category='separation',
            enabled=True
        ))
        self.register(GraphMetadata(
            graph_id='drains_discarded',
            title='Discarded Drains Overview',
            description='Multi-panel overview of discarded drain properties (Mass, T, P)',
            function=sg.create_drains_discarded_figure,
            library=GraphLibrary.MATPLOTLIB,
            data_required=['history'],
            priority=GraphPriority.HIGH,
            category='separation',
            enabled=True
        ))
        self.register(GraphMetadata(
            graph_id='individual_drains',
            title='Individual Drain Properties',
            description='Mass Flow, T, P for each drain - plot_drenos_individuais',
            function=sg.create_individual_drains_figure,
            library=GraphLibrary.MATPLOTLIB,
            data_required=['history'],
            priority=GraphPriority.HIGH,
            category='separation',
            enabled=True
        ))
        self.register(GraphMetadata(
            graph_id='dissolved_gas_concentration',
            title='Dissolved Gas Concentration',
            description='PPM (mg/kg) of dissolved gas in drains - plot_concentracao_dreno',
            function=sg.create_dissolved_gas_figure,
            library=GraphLibrary.MATPLOTLIB,
            data_required=['history'],
            priority=GraphPriority.MEDIUM,
            category='separation',
            enabled=True
        ))
        self.register(GraphMetadata(
            graph_id='dissolved_gas_efficiency',
            title='Dissolved Gas Removal Efficiency',
            description='IN vs OUT concentration with removal % - plot_concentracao_dreno (bar chart)',
            function=sg.create_drain_concentration_figure,
            library=GraphLibrary.MATPLOTLIB,
            data_required=['history'],
            priority=GraphPriority.HIGH,
            category='separation',
            enabled=True
        ))

        self.register(GraphMetadata(
            graph_id='crossover_impurities',
            title='Crossover Impurities',
            description='O2 in H2 / H2 in O2 tracking (ppm molar) - plot_impurezas_crossover',
            function=sg.create_crossover_impurities_figure,
            library=GraphLibrary.MATPLOTLIB,
            data_required=['history'],
            priority=GraphPriority.MEDIUM,
            category='quality',
            enabled=True
        ))
        
        # Profile Plot (New - Process Train)
        self.register(GraphMetadata(
            graph_id='process_train_profile',
            title='Process Train Profile',
            description='Temperature, Entropy, Enthalpy profile along the train',
            function=sg.create_process_train_profile_figure,
            library=GraphLibrary.MATPLOTLIB,
            data_required=['profile_data'],
            priority=GraphPriority.CRITICAL,
            category='profile',
            enabled=True
        ))
        self.register(GraphMetadata(
            graph_id='energy_flows',
            title='Energy Flows & Consumption',
            description='Heat (Q) and Work (W) by component - plot_fluxos_energia',
            function=sg.create_energy_flows_figure,
            library=GraphLibrary.MATPLOTLIB,
            data_required=['history'],
            priority=GraphPriority.HIGH,
            category='energy',
            enabled=True
        ))
        self.register(GraphMetadata(
            graph_id='plant_balance',
            title='Plant Balance Schematic',
            description='Control volume diagram showing mass/energy balance - replaces plot_esquema_planta_completa',
            function=sg.create_plant_balance_schematic,
            library=GraphLibrary.MATPLOTLIB,
            data_required=['history'],
            priority=GraphPriority.CRITICAL,
            category='summary',
            enabled=True
        ))
        self.register(GraphMetadata(
            graph_id='q_breakdown',
            title='Thermal Load Breakdown',
            description='Cooling load by component - plot_q_breakdown',
            function=sg.create_thermal_load_breakdown_figure,
            library=GraphLibrary.MATPLOTLIB,
            data_required=['history'],
            priority=GraphPriority.MEDIUM,
            category='thermal',
            enabled=True
        ))
        self.register(GraphMetadata(
            graph_id='mixer_comparison',
            title='Drain Mixer Comparison',
            description='Mixer T/P/Flow properties - plot_drenos_mixer',
            function=sg.create_mixer_comparison_figure,
            library=GraphLibrary.MATPLOTLIB,
            data_required=['history'],
            priority=GraphPriority.MEDIUM,
            category='separation',
            enabled=True
        ))
        self.register(GraphMetadata(
            graph_id='deoxo_profile',
            title='Deoxo Reactor Profile',
            description='Temperature and Conversion along reactor length - plot_deoxo_perfil',
            function=sg.create_deoxo_profile_figure,
            library=GraphLibrary.MATPLOTLIB,
            data_required=['deoxo_profiles'], # Custom marker
            priority=GraphPriority.MEDIUM,
            category='separation',
            enabled=True
        ))
        self.register(GraphMetadata(
            graph_id='drain_line_properties',
            title='Drain Line Properties',
            description='Mixed drain stream properties (Mass, T, P) - plot_propriedades_linha_dreno',
            function=sg.create_drain_line_properties_figure,
            library=GraphLibrary.MATPLOTLIB,
            data_required=['history'],
            priority=GraphPriority.MEDIUM,
            category='separation',
            enabled=True
        ))
        # NOTE: drain_line_concentration removed - duplicate of dissolved_gas_efficiency
        self.register(GraphMetadata(
            graph_id='recirculation_comparison',
            title='Recirculation Comparison',
            description='Water recovery vs recirculation properties - plot_recirculacao_mixer',
            function=sg.create_recirculation_comparison_figure,
            library=GraphLibrary.MATPLOTLIB,
            data_required=['history'],
            priority=GraphPriority.MEDIUM,
            category='separation',
            enabled=True
        ))
        self.register(GraphMetadata(
            graph_id='entrained_liquid_flow',
            title='Entrained Liquid Flow',
            description='Liquid carryover in gas streams - plot_vazao_liquida_acompanhante',
            function=sg.create_entrained_liquid_figure,
            library=GraphLibrary.MATPLOTLIB,
            data_required=['history'],
            priority=GraphPriority.MEDIUM,
            category='separation',
            enabled=True
        ))
        self.register(GraphMetadata(
            graph_id='water_vapor_tracking',
            title='Water Vapor Tracking',
            description='Water vapor flow with PPM labels - plot_vazao_agua_separada',
            function=sg.create_water_vapor_tracking_figure,
            library=GraphLibrary.MATPLOTLIB,
            data_required=['history'],
            priority=GraphPriority.HIGH,
            category='separation',
            enabled=True
        ))
        self.register(GraphMetadata(
            graph_id='total_mass_flow',
            title='Total Mass Flow Comparison',
            description='Gas + Vapor + Liquid mass flow - plot_vazao_massica_total_e_removida',
            function=sg.create_total_mass_flow_figure,
            library=GraphLibrary.MATPLOTLIB,
            data_required=['history'],
            priority=GraphPriority.MEDIUM,
            category='flow',
            enabled=True
        ))

        # ========================================================================
        # ORPHANED FUNCTIONS (Previously unregistered) - Added 2025-12
        # ========================================================================
        
        # Monthly Performance (combined figure)
        self.register(GraphMetadata(
            graph_id='monthly_performance',
            title='Monthly Performance Overview',
            description='Combined: Efficiency + Capacity Factor + SOEC Heatmap',
            function=sg.create_monthly_performance_figure,
            library=GraphLibrary.MATPLOTLIB,
            data_required=['history'],
            priority=GraphPriority.MEDIUM,
            category='performance',
            enabled=True
        ))
        
        # Monthly Efficiency (standalone)
        self.register(GraphMetadata(
            graph_id='monthly_efficiency',
            title='Monthly Efficiency',
            description='System efficiency aggregated by month',
            function=sg.create_monthly_efficiency_figure,
            library=GraphLibrary.MATPLOTLIB,
            data_required=['history'],
            priority=GraphPriority.MEDIUM,
            category='performance',
            enabled=True
        ))
        
        # Monthly Capacity Factor (standalone)
        self.register(GraphMetadata(
            graph_id='monthly_capacity_factor',
            title='Monthly Capacity Factor',
            description='System capacity factor aggregated by month',
            function=sg.create_monthly_capacity_factor_figure,
            library=GraphLibrary.MATPLOTLIB,
            data_required=['history'],
            priority=GraphPriority.MEDIUM,
            category='performance',
            enabled=True
        ))
        
        # SOEC Module Heatmap
        self.register(GraphMetadata(
            graph_id='soec_module_heatmap',
            title='SOEC Module Activity Heatmap',
            description='Heatmap of module power over time',
            function=sg.create_soec_module_heatmap_figure,
            library=GraphLibrary.MATPLOTLIB,
            data_required=['history'],
            priority=GraphPriority.HIGH,
            category='soec_ops',
            enabled=True
        ))
        
        # SOEC Module Power Stacked
        self.register(GraphMetadata(
            graph_id='soec_module_power_stacked',
            title='SOEC Module Power Stacked',
            description='Stacked area chart of individual module power',
            function=sg.create_soec_module_power_stacked_figure,
            library=GraphLibrary.MATPLOTLIB,
            data_required=['history'],
            priority=GraphPriority.HIGH,
            category='soec_ops',
            enabled=True
        ))
        
        # SOEC Module Wear Statistics
        self.register(GraphMetadata(
            graph_id='soec_module_wear_stats',
            title='SOEC Module Wear Statistics',
            description='Runtime hours and start/stop cycles per module',
            function=sg.create_soec_module_wear_figure,
            library=GraphLibrary.MATPLOTLIB,
            data_required=['history'],
            priority=GraphPriority.HIGH,
            category='soec_ops',
            enabled=True
        ))
        
        # Drain Scheme Schematic
        self.register(GraphMetadata(
            graph_id='drain_scheme',
            title='Drain System Schematic',
            description='Static diagram of drain system topology',
            function=sg.create_drain_scheme_schematic,
            library=GraphLibrary.MATPLOTLIB,
            data_required=['history'],
            priority=GraphPriority.LOW,
            category='schematic',
            enabled=True
        ))
        
        # Energy Flow Figure
        self.register(GraphMetadata(
            graph_id='energy_flow',
            title='Energy Flow Breakdown',
            description='Thermal (Q) and Electrical (W) power by component',
            function=sg.create_energy_flow_figure,
            library=GraphLibrary.MATPLOTLIB,
            data_required=['history'],
            priority=GraphPriority.MEDIUM,
            category='energy',
            enabled=True
        ))
        
        # Process Scheme Schematic
        self.register(GraphMetadata(
            graph_id='process_scheme',
            title='Process Scheme Schematic',
            description='Dynamic PFD with energy/mass annotations',
            function=sg.create_process_scheme_schematic,
            library=GraphLibrary.MATPLOTLIB,
            data_required=['history'],
            priority=GraphPriority.LOW,
            category='schematic',
            enabled=True
        ))
        
        # Drain Mixer Figure
        self.register(GraphMetadata(
            graph_id='drain_mixer_balance',
            title='Drain Mixer Balance',
            description='Mass and energy balance for drain mixers',
            function=sg.create_drain_mixer_figure,
            library=GraphLibrary.MATPLOTLIB,
            data_required=['history'],
            priority=GraphPriority.MEDIUM,
            category='separation',
            enabled=True
        ))
        
        # Q Breakdown Figure (verify not duplicate)
        self.register(GraphMetadata(
            graph_id='thermal_load_breakdown_time_series',
            title='Thermal Load Time Series',
            description='Cooling load by component over time',
            function=sg.create_q_breakdown_figure,
            library=GraphLibrary.MATPLOTLIB,
            data_required=['history'],
            priority=GraphPriority.MEDIUM,
            category='thermal',
            enabled=True
        ))

        
        logger.info(f"Loaded {len(self._registry)} default graphs")


# Global registry instance
GRAPH_REGISTRY = GraphCatalog()
