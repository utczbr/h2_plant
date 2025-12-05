import json
import logging
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from h2_plant.visualization.metrics_collector import MetricsCollector
from h2_plant.visualization.graph_generator import GraphGenerator
from h2_plant.visualization.graph_catalog import GRAPH_REGISTRY

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_graphs():
    output_dir = Path("simulation_output_weekly")
    raw_metrics_path = output_dir / "metrics" / "raw_metrics.json"
    graphs_dir = output_dir / "graphs"
    graphs_dir.mkdir(exist_ok=True)
    
    if not raw_metrics_path.exists():
        logger.error(f"Raw metrics file not found: {raw_metrics_path}")
        return

    logger.info(f"Loading raw metrics from {raw_metrics_path}...")
    with open(raw_metrics_path, 'r') as f:
        raw_data = json.load(f)
    
    # Reconstruct MetricsCollector
    collector = MetricsCollector()
    
    # Populate timeseries from component_metrics
    # MetricsCollector expects:
    # collector.timeseries['pem']['h2_production_kg_h'] = [...]
    # raw_data['component_metrics']['pem_electrolyzer_detailed']['h2_production_kg_h'] = [...]
    
    # 1. Base timeseries (timestamps, etc)
    if 'timeseries' in raw_data:
        if 'hour' in raw_data['timeseries']:
            collector.timeseries['timestamps'] = raw_data['timeseries']['hour']
        # Map other base metrics if needed
        if 'energy_price_mwh' in raw_data['timeseries']:
             # Convert MWh to kWh for pricing
             collector.timeseries['pricing']['energy_price_eur_kwh'] = [p/1000.0 for p in raw_data['timeseries']['energy_price_mwh']]

    # 2. Map Component Metrics
    cm = raw_data.get('component_metrics', {})
    
    # PEM
    if 'pem_electrolyzer_detailed' in cm:
        pem_data = cm['pem_electrolyzer_detailed']
        collector.timeseries['pem']['h2_production_kg_h'] = pem_data.get('h2_output_kg', []) # Note: output vs production key mismatch?
        # Check if key is h2_output_kg or h2_production_kg_h in component state
        # In detailed PEM, get_state returns 'h2_output_kg' (per step) but graph expects rate?
        # Actually graph expects 'h2_production_kg_h'. 
        # Let's check what detailed PEM returns. 
        # It returns 'h2_production_kg_h' AND 'h2_output_kg'.
        if 'h2_production_kg_h' in pem_data:
             collector.timeseries['pem']['h2_production_kg_h'] = pem_data['h2_production_kg_h']
        
        collector.timeseries['pem']['voltage'] = pem_data.get('cell_voltage_v', [])
        collector.timeseries['pem']['efficiency'] = pem_data.get('system_efficiency_percent', [])
        collector.timeseries['pem']['power_mw'] = pem_data.get('power_consumption_mw', [])
        collector.timeseries['pem']['cumulative_h2_kg'] = pem_data.get('cumulative_h2_kg', [])
        collector.timeseries['pem']['cumulative_energy_kwh'] = pem_data.get('cumulative_energy_kwh', [])

    # SOEC
    if 'soec_cluster' in cm:
        soec_data = cm['soec_cluster']
        collector.timeseries['soec']['h2_production_kg_h'] = soec_data.get('h2_production_kg_h', [])
        collector.timeseries['soec']['active_modules'] = soec_data.get('active_modules', [])
        collector.timeseries['soec']['power_mw'] = soec_data.get('power_consumption_mw', [])
        collector.timeseries['soec']['cumulative_h2_kg'] = soec_data.get('cumulative_h2_kg', [])
        collector.timeseries['soec']['cumulative_energy_kwh'] = soec_data.get('cumulative_energy_kwh', [])
        
        # Calculate ramp rates if missing
        power = collector.timeseries['soec']['power_mw']
        if power and len(power) > 1:
            ramps = [0.0] + [(power[i] - power[i-1])/60.0 for i in range(1, len(power))]
            collector.timeseries['soec']['ramp_rates'] = ramps

    # Tanks
    if 'hp_tanks' in cm:
        hp_data = cm['hp_tanks']
        collector.timeseries['tanks']['hp_pressures'] = hp_data.get('pressures', [])
        # Total stored
        if 'total_mass_kg' in hp_data:
             collector.timeseries['tanks']['total_stored'] = hp_data['total_mass_kg']

    # Coordinator
    if 'dual_path_coordinator' in cm:
        coord_data = cm['dual_path_coordinator']
        collector.timeseries['coordinator']['pem_setpoint_mw'] = coord_data.get('pem_setpoint_mw', [])
        collector.timeseries['coordinator']['soec_setpoint_mw'] = coord_data.get('soec_setpoint_mw', [])
        collector.timeseries['coordinator']['sell_power_mw'] = coord_data.get('sell_power_mw', [])

    # Environment / Pricing
    if 'environment_manager' in cm:
        env_data = cm['environment_manager']
        collector.timeseries['pricing']['wind_coefficient'] = env_data.get('wind_power_coefficient', [])
        # Grid exchange
        if 'grid_exchange_mw' in env_data:
             collector.timeseries['pricing']['grid_exchange_mw'] = env_data['grid_exchange_mw']
        elif 'dual_path_coordinator' in cm:
             # Fallback: estimate from sell power
             sell = collector.timeseries['coordinator']['sell_power_mw']
             collector.timeseries['pricing']['grid_exchange_mw'] = [-s for s in sell] if sell else []
    
    # Initialize GraphGenerator
    generator = GraphGenerator(metrics_collector=collector, catalog=GRAPH_REGISTRY)
    
    # Enable all graphs
    GRAPH_REGISTRY.enable_all()
    
    # Generate all enabled graphs
    logger.info("Generating advanced graphs...")
    generated = generator.generate_all_enabled(parallel=False) # Use serial for debugging if needed
    
    # Export graphs
    logger.info(f"Exporting {len(generated)} graphs to {graphs_dir}...")
    generator.export_all(graphs_dir, format='html')
    
    logger.info("Advanced graph generation complete.")

if __name__ == "__main__":
    generate_graphs()
