
import os
import sys
import logging
import numpy as np

# Add project root to path
sys.path.append(os.getcwd())

from h2_plant.orchestrator import Orchestrator

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("VERIFY")
logger.setLevel(logging.INFO)

def verify_dry_cooler_connections():
    print("\nStarting DryCooler Connection Verification...")
    
    # Path to scenarios (we found it in root)
    scenarios_dir = os.path.abspath("scenarios")
    if not os.path.exists(scenarios_dir):
        print(f"Error: Scenarios directory not found at {scenarios_dir}")
        return

    # Initialize Orchestrator
    try:
        orchestrator = Orchestrator(scenarios_dir)
        orchestrator.initialize_components()
    except Exception as e:
        print(f"Failed to initialize Orchestrator: {e}")
        return

    # Find SOEC_H2_DryCooler_1
    target_id = "SOEC_H2_DryCooler_1"
    dry_cooler = orchestrator.components.get(target_id)
    
    if not dry_cooler:
        print(f"Error: Component {target_id} not found in registry.")
        print("Available DryCoolers:")
        for cid, cmp in orchestrator.components.items():
            if "DryCooler" in cid or "DryCooler" in cmp.__class__.__name__:
                print(f" - {cid}")
        return

    print(f"\nTarget Component: {target_id}")
    print(f"Class: {dry_cooler.__class__.__name__}")
    print(f"Cooling Manager: {dry_cooler.cooling_manager}")
    if dry_cooler.cooling_manager:
        print(f"Cooling Manager ID: {dry_cooler.cooling_manager.component_id}")
    else:
        print("WARNING: No Cooling Manager found (Local Mode).")

    # Run 1 step (1 minute)
    print("\nRunning 1 simulation step (1 hr)...")
    # We need to ensure SOEC produces something. Wind/Power needed.
    # The default data loader loads from CSV. We rely on the first hours having some power.
    # Usually Orchestrator handles this.
    
    try:
        # Run manually to inspect state mid-step if needed, or just let it run.
        # run_simulation returns history.
        history = orchestrator.run_simulation(hours=1)
    except Exception as e:
        print(f"Simulation run failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # post-run inspection
    print(f"\nState after 1 hour:")
    
    # Check inputs
    inlet = dry_cooler.inlet_stream
    if inlet:
        print(f"Inlet Stream: Mass Flow={inlet.mass_flow_kg_h:.4f} kg/h, Temp={inlet.temperature_k:.2f} K")
        print(f"Inlet Composition: {inlet.composition}")
    else:
        print("Inlet Stream: None (No flow received!)")
        
    # Check outputs/state
    state = dry_cooler.get_state()
    print(f"Thermal Duty: {state['tqc_duty_kw']:.4f} kW")
    print(f"Fan Power: {state['fan_power_kw']:.4f} kW")
    print(f"Cooling Manager Load: {dry_cooler.cooling_manager.glycol_duty_kw:.4f} kW" if dry_cooler.cooling_manager else "CM Load: N/A")

    # Check upstream Interchanger
    upstream_id = "SOEC_H2_Interchanger_1"
    interchanger = orchestrator.components.get(upstream_id)
    if interchanger:
        print(f"\nUpstream {upstream_id}:")
        # Interchanger doesn't expose inlet explicitly in get_state but we can check attributes if public
        # Using get_output('hot_out')
        hot_out = interchanger.get_output('hot_out')
        if hot_out:
            print(f"Hot Out Stream: Mass Flow={hot_out.mass_flow_kg_h:.4f} kg/h, Temp={hot_out.temperature_k:.2f} K")
        else:
            print("Hot Out Stream: None")
            
    # Check SOEC
    soec_id = "SOEC_Cluster" # Defined in topology lines 143
    # Mapped ID might be different if mapped in orchestrator
    # Check components keys
    soec = None
    if "SOEC_Cluster" in orchestrator.components:
        soec = orchestrator.components["SOEC_Cluster"]
    elif "soec_cluster" in orchestrator.components:
        soec = orchestrator.components["soec_cluster"]
    
    if soec:
        print(f"\nSOEC Cluster:")
        print(f"Power: {soec.get_state().get('power_mw', 0.0):.4f} MW")
        # Check output
        out = soec.get_output('h2_out')
        if out:
             print(f"H2 Out: Mass Flow={out.mass_flow_kg_h:.4f} kg/h")
        else:
             print("H2 Out: None")
    else:
        print("\nSOEC Cluster not found in components.")

if __name__ == "__main__":
    verify_dry_cooler_connections()
