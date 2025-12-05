"""
Example script demonstrating visualization system usage.
"""

from h2_plant.visualization import MetricsCollector, GraphGenerator, GRAPH_REGISTRY
from h2_plant.core.component_registry import ComponentRegistry
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_metrics_collection_during_simulation():
    """
    Example: Integrate metrics collection into simulation loop.
    """
    # Initialize metrics collector
    collector = MetricsCollector()
    
    # Generate mock data for verification
    import numpy as np
    
    steps = 100  # Generate 100 hours of data
    
    logger.info(f"Generating mock data for {steps} steps...")
    
    for t in range(steps):
        # Create mock component states
        mock_states = {
            'pem_electrolyzer_detailed': {
                'h2_production_kg_h': 50.0 + 10 * np.sin(t/10),
                'cell_voltage_v': 1.8 + 0.1 * np.random.random(),
                'system_efficiency_percent': 65.0,
                'power_consumption_mw': 2.5 + 0.5 * np.sin(t/10),
                'cumulative_h2_kg': 50 * t,
                'cumulative_energy_kwh': 2500 * t
            },
            'soec_cluster': {
                'h2_production_kg_h': 100.0 + 20 * np.cos(t/10),
                'active_modules': int(3 + 2 * np.sin(t/20)),
                'power_consumption_mw': 4.0 + 1.0 * np.cos(t/10),
                'cumulative_h2_kg': 100 * t,
                'cumulative_energy_kwh': 4000 * t
            },
            'hp_tanks': {
                'masses': [500.0 + 10 * t for _ in range(5)],
                'states': [1.0] * 5,
                'pressures': [200.0 + 50 * np.sin(t/10 + i) for i in range(5)] # Mock pressures
            },
            'environment_manager': {
                'energy_price_eur_kwh': 0.15 + 0.05 * np.sin(t/5),
                'wind_power_coefficient': 0.5 + 0.4 * np.sin(t/8),
                'air_density_kg_m3': 1.225
            },
            'dual_path_coordinator': {
                'pem_setpoint_mw': 2.5,
                'soec_setpoint_mw': 4.0,
                'sell_power_mw': 1.0 if t % 24 < 12 else 0.0
            },
            'demand_scheduler': {
                'current_demand_kg_h': 120.0,
                'cumulative_demand_kg': 120 * t
            }
        }
        
        collector.collect_step(float(t), mock_states)
    
    logger.info(f"Collected {collector.summary()['total_steps']} timesteps")
    return collector


def example_configure_graphs():
    """
    Example: Configure which graphs to generate.
    """
    # Option 1: Enable/disable specific graphs
    GRAPH_REGISTRY.disable('tank_storage_timeline')  # Not implemented yet
    GRAPH_REGISTRY.enable('pem_cell_voltage_over_time')
    
    # Option 2: Enable/disable entire categories
    GRAPH_REGISTRY.disable_category('storage')  # Disable all storage graphs
    GRAPH_REGISTRY.enable_category('production')  # Enable all production graphs
    
    # Enable advanced categories
    GRAPH_REGISTRY.enable_category('reliability')
    GRAPH_REGISTRY.enable_category('grid_integration')
    GRAPH_REGISTRY.enable_category('economics')
    GRAPH_REGISTRY.enable_category('advanced')
    
    # Option 3: Disable all and enable only specific ones
    # GRAPH_REGISTRY.disable_all()
    # GRAPH_REGISTRY.enable('total_h2_production_stacked')
    
    logger.info(f"Enabled graphs: {GRAPH_REGISTRY.list_enabled()}")


def example_generate_graphs(collector: MetricsCollector):
    """
    Example: Generate graphs from collected data.
    """
    # Initialize graph generator
    generator = GraphGenerator(collector)
    
    # Configure performance settings
    generator.use_webgl = True
    generator.max_workers = 4
    
    # Option 1: Generate a specific graph
    fig = generator.generate('pem_cell_voltage_over_time')
    # if fig: fig.show()
    
    # Option 2: Generate all enabled graphs (parallel)
    all_figs = generator.generate_all_enabled(parallel=True)
    logger.info(f"Generated {len(all_figs)} graphs")
    
    return generator


def example_export_graphs(generator: GraphGenerator, output_dir: str):
    """
    Example: Export generated graphs to files.
    """
    # Export a specific graph
    generator.export(
        'total_h2_production_stacked',
        f'{output_dir}/production_stacked.html',
        format='html'
    )
    
    # Export all generated graphs
    count = generator.export_all(
        output_dir=output_dir,
        format='html'
    )
    logger.info(f"Exported {count} graphs to {output_dir}")


def example_create_dashboard(generator: GraphGenerator, output_path: str):
    """
    Example: Create an HTML dashboard with multiple graphs.
    """
    # Specify graphs to include (or None for all enabled)
    dashboard_graphs = [
        'total_h2_production_stacked',
        'dispatch_strategy_stacked',
        'pem_cell_voltage_over_time',
        'soec_active_modules_over_time',
        'energy_price_over_time',
        'cumulative_h2_production',
        'storage_fatigue_cycling_3d', # New
        'ramp_rate_stress_distribution', # New
        'wind_utilization_duration_curve', # New
        'grid_interaction_phase_portrait', # New
        'lcoh_waterfall_breakdown', # New
        'pem_performance_surface' # New
    ]
    
    html = generator.create_dashboard(
        graph_ids=dashboard_graphs,
        title="H2 Plant Advanced Dashboard"
    )
    
    # Save dashboard
    with open(output_path, 'w') as f:
        f.write(html)
    
    logger.info(f"Dashboard saved to {output_path}")


def example_full_workflow():
    """
    Example: Complete workflow from data collection to dashboard.
    """
    # Step 1: Configure graphs
    example_configure_graphs()
    
    # Step 2: Simulate data collection (normally done during simulation)
    collector = example_metrics_collection_during_simulation()
    
    # Step 3: Generate graphs
    generator = example_generate_graphs(collector)
    
    # Step 4: Export to files
    example_export_graphs(generator, 'simulation_output/example/graphs')
    
    # Step 5: Create dashboard
    example_create_dashboard(generator, 'simulation_output/example/dashboard.html')
    
    logger.info("Visualization workflow complete!")


if __name__ == '__main__':
    # Run the full example
    example_full_workflow()
    
    # Or run individual examples:
    # example_configure_graphs()
    # collector = example_metrics_collection_during_simulation()
    # generator = example_generate_graphs(collector)
    # example_export_graphs(generator, 'output/graphs')
