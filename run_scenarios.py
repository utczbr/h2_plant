import os
import shutil
import sys
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from h2_plant.orchestrator import Orchestrator
from h2_plant.reporting.report_generator import ReportGenerator


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SCENARIOS = [
    # {"name": "1_PEM_Only", "topology": os.path.join(BASE_DIR, "scenarios/topologies/topology_pem_only.yaml")},
    # {"name": "2_SOEC_Only", "topology": os.path.join(BASE_DIR, "scenarios/topologies/topology_soec_only.yaml")},
    {"name": "3_Hybrid_Simple", "topology": os.path.join(BASE_DIR, "scenarios/topologies/topology_hybrid_simple.yaml")},
    {"name": "4_Hybrid_Cascade", "topology": os.path.join(BASE_DIR, "scenarios/topologies/topology_hybrid_cascade.yaml")},
]

def run_scenario(scenario):
    name = scenario["name"]
    topology_file = scenario["topology"]
    
    print(f"\n=== Running Scenario: {name} ===")
    
    # 1. Update Topology
    # 1. Update Topology
    target_topology = os.path.join(BASE_DIR, "scenarios/plant_topology.yaml")
    shutil.copy(topology_file, target_topology)
    print(f"Updated topology from {topology_file}")
    
    # DEBUG: Verify Topology File Content
    import hashlib
    with open(target_topology, 'rb') as f:
        content = f.read()
        print(f"[DEBUG] Topology Size: {len(content)} bytes")
        print(f"[DEBUG] Topology MD5: {hashlib.md5(content).hexdigest()}")
        print(f"[DEBUG] First 5 lines:\n{content.decode('utf-8').splitlines()[:5]}")

    # 2. Run Simulation (1 Week = 168 Hours)
    orch = Orchestrator(os.path.join(BASE_DIR, 'scenarios'))
    history = orch.run_simulation(hours=168)
    
    # 3. Save Results
    output_dir = os.path.join(BASE_DIR, f"simulation_output/{name}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save History CSV
    df = pd.DataFrame(history)
    csv_path = os.path.join(output_dir, "simulation_history.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved history to {csv_path}")
    
    # 4. Generate Reports
    print("Generating reports...")
    # Load visualization config (ensure it's enabled)
    import yaml
    # Load visualization config (ensure it's enabled)
    import yaml
    with open(os.path.join(BASE_DIR, 'scenarios/visualization_config.yaml'), 'r') as f:
        viz_config = yaml.safe_load(f)
    
    # Disable some complex charts for simple scenarios if needed, but keeping all is fine
    # ReportGenerator handles missing keys gracefully (hopefully)
    
    reporter = ReportGenerator(output_dir, viz_config)
    reporter.generate_all(history)

    # --- DEBUG: Inspect BoP Columns ---
    print(f"\n[DEBUG] History Keys: {list(history.keys())}")
    for key in history.keys():
        if 'power_kw' in key:
            vals = history[key]
            total_kw = np.sum(vals)
            avg_kw = np.mean(vals) if len(vals) > 0 else 0
            print(f"[DEBUG] {key}: Sum={total_kw:.2f} kW, Avg={avg_kw:.2f} kW, Max={np.max(vals) if len(vals) > 0 else 0:.2f} kW")
    # ----------------------------------

    
    # 5. Return Summary Metrics
    # 5. Return Summary Metrics
    total_h2 = np.sum(history.get('h2_kg', []))
    
    # Calculate Total Energy (Electrolyzers + BoP)
    # Identify all power columns (usually end in _power_kw or P_...)
    p_electrolyzers = np.sum(history.get('P_soec_actual', [])) + np.sum(history.get('P_pem', []))
    
    # Identify BoP columns dynamically
    bop_power_kw_sum = 0.0
    for key in history.keys():
        if key.endswith('_power_kw'):
            bop_power_kw_sum += np.sum(history[key])
            
    # Total Energy MWh = (Electrolyzer MW + BoP kW/1000) * hours
    # History is in minutes (divide by 60)
    # P_soec/P_pem are in MW. BoP is in kW.
    
    total_energy_mwh = (p_electrolyzers + (bop_power_kw_sum / 1000.0)) / 60.0
    
    # Water
    h2o_soec = np.sum(history.get('steam_soec_kg', [])) # Input
    h2o_pem = np.sum(history.get('H2O_pem_kg', []))
    total_water = h2o_soec + h2o_pem
    
    return {
        "Scenario": name,
        "H2 Produced (kg)": total_h2,
        "Energy (MWh)": total_energy_mwh,
        "BoP Energy (MWh)": bop_power_kw_sum / 60.0 / 1000.0,
        "Sold Energy (MWh)": np.sum(history.get('P_sold', [])) / 60.0,
        "Water Input (kg)": total_water,
        "Efficiency (kWh/kg)": (total_energy_mwh * 1000) / total_h2 if total_h2 > 0 else 0
    }

if __name__ == "__main__":
    results = []
    
    # Backup original topology
    # Backup original topology
    original_topo = os.path.join(BASE_DIR, "scenarios/plant_topology.yaml")
    if os.path.exists(original_topo):
        shutil.copy(original_topo, original_topo + ".bak")
    
    try:
        for scenario in SCENARIOS:
            try:
                res = run_scenario(scenario)
                results.append(res)
            except Exception as e:
                print(f"Error running scenario {scenario['name']}: {e}")
                import traceback
                traceback.print_exc()
                
    finally:
        # Restore original topology
        # Restore original topology
        if os.path.exists(original_topo + ".bak"):
            shutil.copy(original_topo + ".bak", original_topo)
            os.remove(original_topo + ".bak")
            print("\nRestored original topology.")

    # Print Comparison Table
    print("\n" + "="*140)
    print(f"{'Scenario':<20} | {'H2 (kg)':<15} | {'Total Consumed (MWh)':<22} | {'BoP Energy (MWh)':<18} | {'Sold Energy (MWh)':<18} | {'Eff (kWh/kg)':<15}")
    print("-" * 140)
    for r in results:
        print(f"{r['Scenario']:<20} | {r['H2 Produced (kg)']:15.2f} | {r['Energy (MWh)']:22.2f} | {r['BoP Energy (MWh)']:18.2f} | {r['Sold Energy (MWh)']:18.2f} | {r['Efficiency (kWh/kg)']:15.2f}")
    print("="*140 + "\n")
