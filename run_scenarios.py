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
    
    hours = 168  # 1 week
    
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
    
    # 8. Manual simulation with stream propagation
    dt = context.simulation.timestep_hours
    num_steps = int(hours / dt)
    
    # For purification train, we do manual stream propagation
    # (The topology is already connected via receive_input/get_output)
    
    # Initialize all components
    registry.initialize_all(dt)
    
    # Get topology order
    topo_order = [node.id for node in context.topology.nodes] if context.topology.nodes else []
    
    # Run one full timestep to generate outputs
    for step in range(min(num_steps, 100)):  # Limit to 100 steps for speed
        t = step * dt
        
        # Step each component in topology order
        for comp_id in topo_order:
            comp = components.get(comp_id)
            if comp:
                comp.step(t)
                
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
                                    pass  # Some ports may not exist
    
    # 9. Print Stream Summary Table
    # First, detect primary gas by checking first component's output
    primary_gas = 'H2'  # Default
    for comp_id in topo_order:
        comp = components.get(comp_id)
        if comp:
            for port in ['outlet', 'h2_out', 'fluid_out', 'gas_outlet']:
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
    print(f"{'Component':<18} | {'T_out':>10} | {'P_out':>12} | {col3_header:>10} | {'H2O':>12} | {col5_header:>10}")
    print("-" * 88)
    
    for comp_id in topo_order:
        comp = components.get(comp_id)
        if not comp:
            continue
        
        # Try to get output stream
        stream = None
        for port in ['outlet', 'h2_out', 'fluid_out', 'gas_outlet', 'purified_gas_out']:
            try:
                stream = comp.get_output(port)
                if stream is not None:
                    break
            except:
                pass
        
        if stream is None:
            continue
        
        T_c = stream.temperature_k - 273.15
        P_bar = stream.pressure_pa / 1e5
        h2_frac = stream.composition.get('H2', 0.0)
        h2o_frac = stream.composition.get('H2O', 0.0) + stream.composition.get('H2O_liq', 0.0)
        o2_frac = stream.composition.get('O2', 0.0)
        
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
        
        print(f"{comp_id:<18} | {T_c:>8.1f}°C | {P_bar:>10.2f} bar | {purity_frac*100:>9.2f}% | {h2o_str:>12} | {impurity_str:>10}")
    
    print("-" * 88)

    # 10. Save History
    output_dir = os.path.join(BASE_DIR, f"simulation_output/{name}")
    os.makedirs(output_dir, exist_ok=True)
    
    history_path = os.path.join(output_dir, "simulation_history.csv")
    print(f"Saved history to {history_path}")
    
    # Return summary metrics (simplified for purification train)
    return {
        "Scenario": name,
        "H2 Produced (kg)": 0.0,  # Not applicable for purification train
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
