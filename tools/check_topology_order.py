
import yaml
import sys
import os

def check_topology_order(topology_path):
    print(f"Checking topology order in: {topology_path}")
    
    with open(topology_path, 'r') as f:
        data = yaml.safe_load(f)
        
    nodes = data.get('nodes', [])
    node_map = {}
    
    # 1. Map IDs to Steps
    print("--- Loading Node Steps ---")
    for node in nodes:
        node_id = node.get('id')
        step = node.get('params', {}).get('process_step')
        
        if step is None and node.get('type') == 'CoolingManager': # Global utilities often 0
             step = 0
             
        if node_id:
            node_map[node_id] = {'step': step, 'type': node.get('type')}
            
    # 2. Check Connections
    print("\n--- Verifying Connections (Source Step < Target Step) ---")
    violations = []
    loops = []
    
    for node in nodes:
        source_id = node.get('id')
        source_step = node_map.get(source_id, {}).get('step')
        
        if source_step is None:
            continue
            
        connections = node.get('connections', [])
        for conn in connections:
            target_id = conn.get('target_name')
            target_node = node_map.get(target_id)
            
            if not target_node:
                print(f"WARNING: Target '{target_id}' not found for Source '{source_id}'")
                continue
                
            target_step = target_node.get('step')
            
            if target_step is None:
                print(f"WARNING: Target '{target_id}' has no process_step")
                continue
            
            # CHECK LOGIC
            if source_step >= target_step:
                # Potential Violation
                # Check if it's a feedback loop (e.g. Signal, or Drain Pump recycling)
                is_feedback = False
                
                # Heuristic: Drain/Recycle loops usually go from higher step to lower step
                msg = f"{source_id} (Step {source_step}) -> {target_id} (Step {target_step})"
                
                if "drain" in source_id.lower() or "pump" in source_id.lower() or "signal" in conn.get('resource_type', '').lower():
                    loops.append(msg + " [Likely Feedback Loop]")
                else:
                    violations.append(msg + " [VIOLATION]")

    print("\n=== REPORT ===")
    if violations:
        print(f"\nCRITICAL ORDER VIOLATIONS ({len(violations)}):")
        for v in violations:
            print(f"  X  {v}")
    else:
        print("\nNo Critical Forward-Path Violations found.")
        
    if loops:
        print(f"\nDetected Feedback Loops ({len(loops)}):")
        for l in loops:
            print(f"  O  {l}")

if __name__ == "__main__":
    check_topology_order("scenarios/plant_topology.yaml")
