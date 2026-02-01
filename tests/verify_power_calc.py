
from h2_plant.core.constants import DryCoolerIndirectConstants as DCC

# Inputs from Topology (scenarios/plant_topology.yaml)
# dc_air_flow_kg_s: 1000.0 (Optimized) 
m_dot_air = 1000.0

# Constants
rho = DCC.RHO_AIR_KG_M3
dp = DCC.DP_AIR_DESIGN_PA
eta = DCC.ETA_FAN

# Calculation
vol_flow = m_dot_air / rho
power_watts = (vol_flow * dp) / eta
power_kw = power_watts / 1000.0

print(f"--- Manual Central Power Calculation ---")
print(f"Air Mass Flow: {m_dot_air} kg/s")
print(f"Density: {rho} kg/m3")
print(f"Vol Flow: {vol_flow:.2f} m3/s")
print(f"Pressure Drop: {dp} Pa")
print(f"Fan Efficiency: {eta}")
print(f"Calculated Power: {power_kw:.2f} kW")
