
import yaml

# Read topology
with open('scenarios/plant_topology.yaml', 'r') as f:
    topo = yaml.safe_load(f)

nodes = topo.get('nodes', [])

total_cap_kw = 0.0
count = 0

print("--- Cooling Capacity Inventory ---")
for node in nodes:
    if node['type'] == 'DryCooler':
        # Default defaults to 100 kW if not specified
        cap = node.get('params', {}).get('design_capacity_kw', 
              node.get('params', {}).get('cooling_capacity_kw', 100.0))
        
        # Check if active/enabled (some might be placeholders, but we sum installed)
        total_cap_kw += cap
        count += 1
        # print(f"{node['id']}: {cap} kW")

print(f"Total Dry Coolers: {count}")
print(f"Total Installed Capacity: {total_cap_kw:.2f} kW ({total_cap_kw/1000:.2f} MW)")

# Calculate required airflow for this capacity
# Assumption: Delta T = 15 K (Standard Design: Air in 25C, Air out 40C)
# Cp_air = 1.005 kJ/kgK
# Q = m * Cp * dT => m = Q / (Cp * dT)
delta_t_design = 15.0
cp_air = 1.005
required_airflow = total_cap_kw / (cp_air * delta_t_design)

print(f"--- Sizing Recommendation ---")
print(f"Target Delta T: {delta_t_design} K")
print(f"Required Design Airflow: {required_airflow:.2f} kg/s")
print(f"Current Central Airflow: 5000.00 kg/s")
