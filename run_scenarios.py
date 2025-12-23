"""
Scenario Runner using Unified SimulationEngine Architecture.

Usage:
    # Run a specific topology:
    python run_scenarios.py scenarios/topologies/topology_h2_compression_test.yaml
    
    # Run all topologies in scenarios/topologies/:
    python run_scenarios.py

Migrated from legacy Orchestrator to align with GUI codepath.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob
import yaml

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from h2_plant.config.loader import ConfigLoader
from h2_plant.core.graph_builder import PlantGraphBuilder
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.simulation.engine import SimulationEngine
from h2_plant.control.engine_dispatch import HybridArbitrageEngineStrategy
from h2_plant.data.price_loader import EnergyPriceLoader
from h2_plant.reporting.report_generator import ReportGenerator

import logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ScenarioRunner")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Molecular weights (kg/mol)
MW_SPECIES = {
    'H2': 2.016e-3,
    'O2': 32.0e-3,
    'H2O': 18.015e-3,
    'H2O_liq': 18.015e-3,
    'N2': 28.0e-3,
    'CO2': 44.0e-3,
    'CH4': 16.0e-3,
}

def mass_frac_to_mole_frac(composition: dict) -> dict:
    """
    Convert mass fractions to mole fractions.
    
    Args:
        composition: Dictionary of {species: mass_fraction}
        
    Returns:
        Dictionary of {species: mole_fraction}
    """
    # Calculate relative moles: n_i = x_i / M_i
    n_species = {}
    n_total = 0.0
    for species, mass_frac in composition.items():
        mw = MW_SPECIES.get(species, 28.0e-3)  # Default to N2 MW for unknowns
        if mass_frac > 0 and mw > 0:
            n = mass_frac / mw
            n_species[species] = n
            n_total += n
    
    # Convert to mole fractions
    if n_total > 0:
        return {s: n / n_total for s, n in n_species.items()}
    return composition.copy()


def load_enabled_graphs() -> list:
    """
    Load visualization configuration and return list of enabled graph IDs.
    Defaults to all graphs if config file is missing or invalid.
    """
    config_path = os.path.join(BASE_DIR, "scenarios/visualization_config.yaml")
    if not os.path.exists(config_path):
        return None # Return None to trigger default behavior (all graphs)
        
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        viz_config = config.get('visualization', {})
        graphs_config = viz_config.get('graphs', {})
        
        # In this config structure, we only have explicit enables/disables.
        # We need to collect all keys where value is True.
        # Note: The GUI plotter registry might have more keys than the config.
        # If we return a list, generate_all_graphs_to_files will ONLY generate those.
        
        enabled_graphs = [k for k, v in graphs_config.items() if v is True]
        
        # If no graphs are explicitly enabled, we might return empty list.
        # But wait, config has 'categories'.
        # For now, let's respect explicit 'graphs' keys.
        
        return enabled_graphs
    except Exception as e:
        print(f"Error loading visualization config: {e}")
        return None


def discover_topologies() -> list:
    """
    Discover all topology YAML files in scenarios/topologies/.
    
    Returns:
        list: List of dicts with 'name' and 'topology' keys.
    """
    topologies_dir = os.path.join(BASE_DIR, "scenarios/topologies")
    topology_files = glob(os.path.join(topologies_dir, "*.yaml")) + \
                     glob(os.path.join(topologies_dir, "*.yml"))
    
    scenarios = []
    for topo_path in sorted(topology_files):
        name = Path(topo_path).stem
        scenarios.append({
            "name": name,
            "topology": topo_path
        })
    
    return scenarios


def run_scenario(scenario: dict) -> dict:
    """
    Run a single scenario using SimulationEngine.
    
    Args:
        scenario (dict): Scenario definition with 'name' and 'topology' keys.
        
    Returns:
        dict: Summary metrics for the scenario.
    """
    name = scenario["name"]
    topology_file = scenario["topology"]
    
    print(f"\n=== Running Scenario: {name} ===")
    print(f"Topology: {topology_file}")
    
    # 1. Load Configuration with specified topology
    scenarios_dir = os.path.join(BASE_DIR, "scenarios")
    loader = ConfigLoader(scenarios_dir)
    context = loader.load_context(topology_file=topology_file)
    
    hours = context.simulation.duration_hours
    
    # 2. Build Component Graph
    builder = PlantGraphBuilder(context)
    components = builder.build()
    
    # 3. Create and Populate Registry
    registry = ComponentRegistry()
    for cid, comp in components.items():
        registry.register(cid, comp)
    
    # 4. Load Dispatch Data
    data_loader = EnergyPriceLoader(scenarios_dir)
    prices, wind = data_loader.load_data(
        context.simulation.energy_price_file,
        context.simulation.wind_data_file,
        hours,
        context.simulation.timestep_hours
    )
    total_steps = len(prices)
    
    # 5. Create Dispatch Strategy
    dispatch_strategy = HybridArbitrageEngineStrategy()

    # 6. Create and Configure SimulationEngine
    output_path = Path(BASE_DIR) / f"simulation_output/{name}"
    
    # Import ConnectionConfig for proper object construction
    from h2_plant.config.plant_config import ConnectionConfig, IndexedConnectionConfig
    
    # Derive connections from topology nodes as proper ConnectionConfig objects
    topology_connections = []
    indexed_connections = []
    if context.topology and context.topology.nodes:
        for node in context.topology.nodes:
            for conn in node.connections:
                # Create proper ConnectionConfig object
                topology_connections.append(ConnectionConfig(
                    source_id=node.id,
                    source_port=conn.source_port,
                    target_id=conn.target_name,
                    target_port=conn.target_port,
                    resource_type=conn.resource_type
                ))
    
    engine = SimulationEngine(
        registry=registry,
        config=context.simulation,
        output_dir=output_path,
        topology=topology_connections,
        indexed_topology=indexed_connections,
        dispatch_strategy=dispatch_strategy
    )

    # 7. Set up history collection and run the simulation
    component_history_records = []
    def collect_component_states(hour: float):
        """Callback to collect detailed state from all components at each step."""
        # The engine runs at 1-minute steps.
        step_idx = int(round(hour * 60))
        record = {'minute': step_idx}
        for comp_id, comp in registry.list_components():
             if hasattr(comp, 'get_state'):
                try:
                    state = comp.get_state()
                    if state: # Ensure state is not None
                        for k, v in state.items():
                            record[f"{comp_id}_{k}"] = v
                except Exception:
                    pass # Ignore components that fail get_state
        component_history_records.append(record)

    engine.set_dispatch_data(prices, wind)
    engine.initialize()
    engine.initialize_dispatch_strategy(context=context, total_steps=total_steps)
    engine.set_callbacks(post_step=collect_component_states)

    print(f"Running simulation for {hours} hours ({total_steps} steps)...")
    engine.run()
    print("Simulation finished.")

    # 8. Merge dispatch history with detailed component history
    dispatch_history_df = pd.DataFrame(dispatch_strategy.get_history())
    component_history_df = pd.DataFrame(component_history_records)

    # Merge dataframes to create a comprehensive history
    if not component_history_df.empty:
        # Ensure 'minute' column is of the same type for merging
        dispatch_history_df['minute'] = dispatch_history_df['minute'].astype(int)
        component_history_df['minute'] = component_history_df['minute'].astype(int)
        history_df = pd.merge(dispatch_history_df, component_history_df, on='minute', how='outer')
    else:
        history_df = dispatch_history_df

    history_dict = history_df.to_dict(orient='list')

    from h2_plant.visualization.graph_catalog import GRAPH_REGISTRY, GraphLibrary
    from h2_plant.visualization import static_graphs
    from h2_plant.visualization.dashboard_generator import DashboardGenerator
    
    # Get topology order for summary table generation
    topo_order = [node.id for node in context.topology.nodes] if context.topology and context.topology.nodes else []

    # 9. Print Enhanced Stream Summary Table
    from h2_plant.reporting.stream_table import print_stream_summary_table
    
    # Build connection map: source_id -> list of target_ids
    connection_map = {}
    if context.topology and context.topology.nodes:
        for node in context.topology.nodes:
            targets = [conn.target_name for conn in node.connections]
            connection_map[node.id] = targets
    
    profile_data = print_stream_summary_table(components, topo_order, connection_map)

    # 10. Save History and Generate Graphs
    output_dir = os.path.join(BASE_DIR, f"simulation_output/{name}")
    graphs_dir = os.path.join(output_dir, "graphs")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(graphs_dir, exist_ok=True)
    
    history_path = os.path.join(output_dir, "simulation_history.csv")
    history_df.to_csv(history_path, index=False)
    print(f"Saved history to {history_path}")

    # Generate Graphs via Catalog
    print("Generating graphs...")
    try:
        # 1. Apply Configuration
        enabled_graphs_config = load_enabled_graphs()
        if enabled_graphs_config is not None:
             print(f"  Configuration loaded: Enabling {len(enabled_graphs_config)} graphs from config.")
             GRAPH_REGISTRY.disable_all()
             for gid in enabled_graphs_config:
                 GRAPH_REGISTRY.enable(gid)
        else:
             print("  No visualization config found. Using default enabled graphs.")

        # 2. Prepare Data Contexts
        norm_history_df = static_graphs.normalize_history(history_dict)
        profile_df_obj = pd.DataFrame(profile_data)
        profile_df_obj.attrs['scenario_name'] = name

        # 3. Iterate and Generate
        enabled_metadata = GRAPH_REGISTRY.get_enabled()
        print(f"  Generating {len(enabled_metadata)} active graphs...")
        
        for metadata in enabled_metadata:
             try:
                 if metadata.graph_id == 'process_train_profile':
                      data = profile_df_obj
                 elif metadata.graph_id == 'deoxo_profile':
                      deoxo_comp = next((c for cid, c in components.items() if 'deoxo' in cid.lower() or 'deoxo' in c.__class__.__name__.lower()), None)
                      data = deoxo_comp.get_last_profiles() if deoxo_comp and hasattr(deoxo_comp, 'get_last_profiles') else {'L': [], 'T': [], 'X': []}
                 elif metadata.graph_id == 'crossover_impurities':
                      # Build component state data for crossover impurities
                      # Collect ppm values from component get_state()
                      crossover_data = {}
                      for cid, comp in components.items():
                          try:
                              state = comp.get_state()
                              for key, val in state.items():
                                  if 'ppm' in key.lower() and ('o2' in key.lower() or 'h2' in key.lower()):
                                      col_name = f"{cid}_{key}"
                                      crossover_data[col_name] = [val]  # Single row with final value
                          except Exception:
                              pass
                      if crossover_data:
                          data = pd.DataFrame(crossover_data)
                          data['minute'] = [0]  # Dummy minute column
                      else:
                          data = norm_history_df
                 elif metadata.library == GraphLibrary.MATPLOTLIB:
                      data = norm_history_df
                 elif metadata.library == GraphLibrary.PLOTLY:
                      data = history_dict
                 else:
                      continue

                 fig = metadata.function(data)
                 
                 if fig:
                     if metadata.library == GraphLibrary.MATPLOTLIB:
                          out_path = os.path.join(graphs_dir, f"{metadata.graph_id}.png")
                          fig.savefig(out_path, dpi=100)
                     elif metadata.library == GraphLibrary.PLOTLY:
                          out_path = os.path.join(graphs_dir, f"{metadata.graph_id}.html")
                          fig.write_html(out_path, include_plotlyjs='cdn')
             except Exception as g_err:
                 print(f"  Failed to generate {metadata.graph_id}: {g_err}")

    except Exception as e:
        print(f"Error in graph generation workflow: {e}")
        import traceback
        traceback.print_exc()

    # Return summary metrics
    hist = history_dict
    dt = context.simulation.timestep_hours
    
    total_h2 = np.sum(hist.get('h2_kg', [0]))
    soec_h2 = np.sum(hist.get('H2_soec_kg', [0]))
    pem_h2 = np.sum(hist.get('H2_pem_kg', [0]))
    
    e_offered = np.sum(hist.get('P_offer', [0])) * dt
    e_sold = np.sum(hist.get('P_sold', [0])) * dt
    
    e_consumed_soec = np.sum(hist.get('P_soec_actual', [0])) * dt
    e_consumed_pem = np.sum(hist.get('P_pem', [0])) * dt
    e_consumed_total = e_consumed_soec + e_consumed_pem
    
    efficiency = (e_consumed_total * 1000) / total_h2 if total_h2 > 0 else 0
    
    return {
        "Scenario": name,
        "H2 Produced (kg)": total_h2,
        "SOEC H2 (kg)": soec_h2,
        "PEM H2 (kg)": pem_h2,
        "Energy Offered (MWh)": e_offered,
        "Energy Sold (MWh)": e_sold,
        "Efficiency (kWh/kg)": efficiency
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run H2 plant simulation scenarios.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a specific topology:
  python run_scenarios.py scenarios/topologies/topology_h2_compression_test.yaml
  
  # Run all topologies in scenarios/topologies/:
  python run_scenarios.py
  
  # Run O2 treatment train:
  python run_scenarios.py scenarios/topologies/o2_treatment_train.yaml
"""
    )
    parser.add_argument(
        'topology', 
        nargs='?', 
        default=None,
        help='Path to specific topology YAML file. If omitted, runs all topologies in scenarios/topologies/'
    )
    
    args = parser.parse_args()
    
    # Determine scenarios to run
    if args.topology:
        # Single topology specified
        topology_path = os.path.abspath(args.topology)
        if not os.path.exists(topology_path):
            print(f"Error: Topology file not found: {topology_path}")
            sys.exit(1)
        
        scenarios = [{
            "name": Path(topology_path).stem,
            "topology": topology_path
        }]
    else:
        # Discover all topologies
        scenarios = discover_topologies()
        if not scenarios:
            print("No topology files found in scenarios/topologies/")
            sys.exit(1)
        print(f"Discovered {len(scenarios)} topology file(s):")
        for s in scenarios:
            print(f"  - {s['name']}")
    
    # Run scenarios
    results = []
    for scenario in scenarios:
        try:
            res = run_scenario(scenario)
            results.append(res)
        except Exception as e:
            print(f"Error running scenario {scenario['name']}: {e}")
            import traceback
            traceback.print_exc()

    # Print Comparison Table
    if results:
        print("\n" + "=" * 160)
        print(f"{'Scenario':<30} | {'H2 Total (kg)':<15} | {'SOEC H2 (kg)':<14} | {'PEM H2 (kg)':<13} | {'E Offered (MWh)':<16} | {'E Sold (MWh)':<14} | {'Eff (kWh/kg)':<14}")
        print("-" * 160)
        for r in results:
            print(f"{r['Scenario']:<30} | {r['H2 Produced (kg)']:>15.2f} | {r['SOEC H2 (kg)']:>14.2f} | {r['PEM H2 (kg)']:>13.2f} | {r['Energy Offered (MWh)']:>16.2f} | {r['Energy Sold (MWh)']:>14.2f} | {r['Efficiency (kWh/kg)']:>14.2f}")
        print("=" * 160 + "\n")


if __name__ == "__main__":
    main()
