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
    
    # 5. Create Dispatch Strategy (for stream summary printing)
    dispatch_strategy = HybridArbitrageEngineStrategy()
    
    # 7. Create SimulationEngine
    engine_config = {
        'simulation': {
            'timestep_hours': context.simulation.timestep_hours,
            'duration_hours': hours,
            'start_hour': context.simulation.start_hour,
        }
    }
    
    class MinimalPlant:
        """Adapter to provide plant interface for SimulationEngine."""
        def __init__(self, components, registry):
            self.components = components
            self.registry = registry
            self.graph_builder = type('obj', (object,), {
                'components': components,
                'get_component': lambda cid: components.get(cid)
            })()
    
    plant = MinimalPlant(components, registry)
    engine = SimulationEngine(plant, engine_config)
    
    # NOTE: We are not using the engine's dispatch now since we have specific
    # purification train logic. For simplicity, run manual timesteps.
    
    from h2_plant.visualization.graph_catalog import GRAPH_REGISTRY, GraphLibrary
    from h2_plant.visualization import static_graphs
    from h2_plant.visualization.dashboard_generator import DashboardGenerator
    
    # 8. Manual simulation with stream propagation
    dt = context.simulation.timestep_hours
    num_steps = int(hours / dt)
    
    # Initialize all components
    registry.initialize_all(dt)
    
    # History collection
    history_records = []
    
    # Get topology order
    topo_order = [node.id for node in context.topology.nodes] if context.topology.nodes else []
    
    # Run one full timestep to generate outputs
    # LIMIT REMOVED: Run full duration or at least reasonable amount if hours is small
    # If hours=168 (1 week), num_steps is huge (10k). Let's respect variable.
    steps_to_run = num_steps 
    
    print(f"Running simulation for {steps_to_run} steps ({hours} hours)...")
    
    for step in range(steps_to_run):
        t = step * dt
        
        # Log progress every 10%
        if step % max(1, steps_to_run // 10) == 0:
             print(f"  Step {step}/{steps_to_run} ({step/steps_to_run*100:.0f}%)")
        
        record = {'minute': step, 'timestamp_hours': t}
        
        # Step each component in topology order
        for comp_id in topo_order:
            comp = components.get(comp_id)
            if comp:
                comp.step(t)
                
                # Collect state for history
                # Assuming component has some state or we extract from streams?
                # For graph plots, we need specific keys like 'water_removed_kg_h'
                # Attempt to extract state dict
                if hasattr(comp, 'get_state'):
                    state = comp.get_state()
                    # Prefix keys with component name for flat history structure
                    for k, v in state.items():
                        record[f"{comp_id}_{k}"] = v
                
                # Propagate outputs to connected inputs
                if context.topology.nodes:
                    node = next((n for n in context.topology.nodes if n.id == comp_id), None)
                    if node:
                        for conn in node.connections:
                            target_comp = components.get(conn.target_name)
                            if target_comp:
                                try:
                                    output = comp.get_output(conn.source_port)
                                    if output is not None:
                                        target_comp.receive_input(
                                            conn.target_port, output, conn.resource_type
                                        )
                                except Exception as e:
                                    print(f"Error propagating {comp_id} -> {target_comp.component_id}: {e}")
        
        history_records.append(record)
    
    # Compile History
    import pandas as pd
    history_df = pd.DataFrame(history_records)
    history_dict = history_df.to_dict(orient='list') # Plotter expects dict of lists
    
    # 9. Print Stream Summary Table
    # ... (Keep existing summary table code, omitted for brevity in search replacement if possible, but I need to replace block)
    # Re-inserting check logic below...
    primary_gas = 'H2'  # Default
    for comp_id in topo_order:
        comp = components.get(comp_id)
        if comp:
            for port in ['outlet', 'h2_out', 'o2_out', 'fluid_out', 'gas_outlet']:
                try:
                    stream = comp.get_output(port)
                    if stream and stream.composition:
                        h2_frac = stream.composition.get('H2', 0)
                        o2_frac = stream.composition.get('O2', 0)
                        if o2_frac > h2_frac:
                            primary_gas = 'O2'
                        break
                except:
                    pass
            else:
                 continue
            break
            
    # Set column headers based on primary gas
    if primary_gas == 'O2':
        col3_header = "O2 Purity"
        col5_header = "H2"
    else:
        col3_header = "H2 Purity"
        col5_header = "O2"
    
    print(f"\n### Stream Summary Table (Topology Order) - {primary_gas} Train")
    print("-" * 88)
    print(f"{'Component':<18} | {'T_out':>10} | {'P_out':>12} | {col3_header:>10} | {'H2O':>12} | {'Mass Flow':>10} | {col5_header:>10}")
    print("-" * 88)
    
    profile_data = []

    for comp_id in topo_order:
        comp = components.get(comp_id)
        if not comp:
            continue
        
        # Try to get output stream
        stream = None
        for port in ['outlet', 'h2_out', 'o2_out', 'fluid_out', 'gas_outlet', 'purified_gas_out']:
            try:
                stream = comp.get_output(port)
                if stream is not None:
                    break
            except Exception as e:
                # print(f"Debug table: {comp_id} port {port} error: {e}")
                pass
        
        if stream is None:
            continue
        
        T_c = stream.temperature_k - 273.15
        P_bar = stream.pressure_pa / 1e5
        h2_frac = stream.composition.get('H2', 0.0)
        h2o_frac = stream.composition.get('H2O', 0.0) + stream.composition.get('H2O_liq', 0.0)
        o2_frac = stream.composition.get('O2', 0.0)
        
        # Collect for plotting
        try:
             s_val = stream.specific_entropy_j_kgK / 1000.0
        except:
             s_val = 0.0
             
        profile_data.append({
            'Component': comp_id,
            'T_c': T_c,
            'P_bar': P_bar,
            'H_kj_kg': stream.specific_enthalpy_j_kg / 1000.0,
            'S_kj_kgK': s_val,
            'MassFrac_H2': h2_frac,
            'MassFrac_O2': o2_frac,
            'MassFrac_H2O': h2o_frac
        })

        # Format H2O display
        if h2o_frac >= 0.01:
            h2o_str = f"{h2o_frac*100:.2f}%"
        elif h2o_frac > 0:
            h2o_str = f"{h2o_frac*1e6:.0f} ppm"
        else:
            h2o_str = "0 ppm"
        
        # Determine purity and impurity based on primary gas
        if primary_gas == 'O2':
            purity_frac = o2_frac
            impurity_frac = h2_frac
        else:
            purity_frac = h2_frac
            impurity_frac = o2_frac
        
        # Format impurity display
        if impurity_frac >= 0.01:
            impurity_str = f"{impurity_frac*100:.2f}%"
        elif impurity_frac > 0:
            impurity_str = f"{impurity_frac*1e6:.0f} ppm"
        else:
            impurity_str = "0 ppm"
        
        print(f"{comp_id:<18} | {T_c:>8.1f}°C | {P_bar:>10.2f} bar | {purity_frac*100:>9.2f}% | {h2o_str:>12} | {stream.mass_flow_kg_h:>8.2f} | {impurity_str:>10}")
    
    print("-" * 88)

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
        # Matplotlib graphs expect normalized DataFrame
        norm_history_df = static_graphs.normalize_history(history_dict)
        
        # Profile graphs expect specific DF with metadata
        profile_df_obj = pd.DataFrame(profile_data)
        profile_df_obj.attrs['scenario_name'] = name

        # 3. Iterate and Generate
        enabled_metadata = GRAPH_REGISTRY.get_enabled()
        print(f"  Generating {len(enabled_metadata)} active graphs...")
        
        for metadata in enabled_metadata:
             try:
                 # Determine Data Source
                 if metadata.graph_id == 'process_train_profile':
                      data = profile_df_obj
                 elif metadata.graph_id == 'deoxo_profile':
                      # Find Deoxo component
                      # Assume Deoxo component class name or ID pattern
                      # Try to find by ID containing 'Deoxo' or 'deoxo'
                      deoxo_comp = None
                      for cid, c in components.items():
                          if 'deoxo' in cid.lower() or 'deoxo' in c.__class__.__name__.lower():
                              deoxo_comp = c
                              break
                              
                      if deoxo_comp and hasattr(deoxo_comp, 'get_last_profiles'):
                          data = deoxo_comp.get_last_profiles()
                      else:
                          # Fallback empty
                          data = {'L': [], 'T': [], 'X': []}
                 elif metadata.library == GraphLibrary.MATPLOTLIB:
                      data = norm_history_df
                 elif metadata.library == GraphLibrary.PLOTLY:
                      data = history_dict
                 else:
                      continue

                 # Generate Figure
                 fig = metadata.function(data)
                 
                 # Save Figure
                 if metadata.library == GraphLibrary.MATPLOTLIB:
                      if fig:
                           # Matplotlib
                           out_path = os.path.join(graphs_dir, f"{metadata.graph_id}.png")
                           fig.savefig(out_path, dpi=100)
                           # Explicitly close to free memory, though Figure() avoids pyplot state
                           pass 
                 elif metadata.library == GraphLibrary.PLOTLY:
                      # Plotly (HTML)
                      out_path = os.path.join(graphs_dir, f"{metadata.graph_id}.html")
                      try:
                          fig.write_html(out_path)
                      except Exception as e:
                          pass # Plotly write might fail if no kaleido/etc, ignore for now

             except Exception as g_err:
                 print(f"  Failed to generate {metadata.graph_id}: {g_err}")

        # 4. Generate Dashboard
        # ... (Dashboard logic remains similar, simplified here)
        metrics = {
            'total_production_kg': 0, 'total_demand_kg': 0, 'total_cost': 0, 'average_cost_per_kg': 0
        }
        results_wrapper = {
            'metrics': metrics,
            'dashboard_data': {'timeseries': {'hour': history_df['timestamp_hours'].tolist()}}
        }
        # dash_gen = DashboardGenerator(Path(output_dir))
        # dash_gen.generate(results_wrapper)
        # print(f"Dashboard generated.")
        
    except Exception as e:
        print(f"Error in graph generation workflow: {e}")
        import traceback
        traceback.print_exc()

    # Return summary metrics (simplified for purification train)
    return {
        "Scenario": name,
        "H2 Produced (kg)": 0.0,
        "SOEC H2 (kg)": 0.0,
        "PEM H2 (kg)": 0.0,
        "Energy Offered (MWh)": 0.0,
        "Energy Sold (MWh)": 0.0,
        "Efficiency (kWh/kg)": 0.0
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
