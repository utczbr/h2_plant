"""
GraphCatalog: Registry of available visualization graphs with metadata.
"""

from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


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
    """Metadata for a visualization graph."""
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
        """Load default graph registry with placeholder functions."""
        # Import placeholder functions (will be defined in plotly_graphs.py)
        from h2_plant.visualization import plotly_graphs as pg
        from h2_plant.visualization import static_graphs as sg
        
        # --- PLOTLY GRAPHS ---
        # Production & Performance
        self.register(GraphMetadata(
            graph_id='pem_h2_production_over_time',
            title='PEM H2 Production Rate',
            description='PEM hydrogen production rate (kg/h) over time',
            function=pg.plot_pem_production_timeline,
            library=GraphLibrary.PLOTLY,
            data_required=['pem.h2_production_kg_h', 'timestamps'],
            priority=GraphPriority.CRITICAL,
            category='production',
            enabled=True
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
            enabled=True
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
            enabled=True
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
            enabled=True
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
            enabled=True
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
            enabled=True
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
            enabled=True
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
            enabled=True
        ))
        
        self.register(GraphMetadata(
            graph_id='power_consumption_breakdown_pie',
            title='Power Consumption Breakdown',
            description='Pie chart showing power distribution',
            function=pg.plot_power_breakdown_pie,
            library=GraphLibrary.PLOTLY,
            data_required=['pem.cumulative_energy_kwh', 'soec.cumulative_energy_kwh'],
            priority=GraphPriority.MEDIUM,
            category='economics',
            enabled=True
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
            enabled=True
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
            enabled=True
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
            enabled=True
        ))
        
        # Grid & Renewables Integration
        self.register(GraphMetadata(
            graph_id='wind_utilization_duration_curve',
            title='Wind Utilization Duration Curve',
            description='Duration curve of wind power usage',
            function=pg.plot_wind_utilization_duration_curve,
            library=GraphLibrary.PLOTLY,
            data_required=['pricing.wind_coefficient', 'pem.power_mw', 'soec.power_mw'],
            priority=GraphPriority.HIGH,
            category='grid_integration',
            enabled=True
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
            enabled=True
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
            enabled=True
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
            enabled=True
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
        
        # Thermal & Separation (Static)
        self.register(GraphMetadata(
            graph_id='chiller_cooling',
            title='Chiller Cooling Load',
            description='Cooling load and electrical consumption',
            function=sg.create_chiller_cooling_figure,
            library=GraphLibrary.MATPLOTLIB,
            data_required=['history'],
            priority=GraphPriority.MEDIUM,
            category='thermal',
            enabled=True
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
        # Energy & Thermal Analysis (From Plots.csv)
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
        self.register(GraphMetadata(
            graph_id='drain_line_concentration',
            title='Drain Line Concentration',
            description='Dissolved gas PPM in drain line - plot_concentracao_linha_dreno',
            function=sg.create_drain_concentration_figure,
            library=GraphLibrary.MATPLOTLIB,
            data_required=['history'],
            priority=GraphPriority.MEDIUM,
            category='separation',
            enabled=True
        ))
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

        
        logger.info(f"Loaded {len(self._registry)} default graphs")


# Global registry instance
GRAPH_REGISTRY = GraphCatalog()
