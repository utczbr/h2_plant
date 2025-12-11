import numpy as np
from scipy.interpolate import interp1d
import os

# Data extracted from main_pem_simulator_S.py
DEGRADATION_TABLE_YEARS = np.array([1.0, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 9.0, 10.0])
DEGRADATION_TABLE_V_STACK = np.array([1290, 1300, 1325, 1340, 1345, 1355, 1365, 1380, 1390, 1410, 1435, 1460, 1490])
N_cell_per_stack = 85
V_CELL_BOL_NOM = 1.290 # Approximate BOL, will be calculated properly in the main code but needed for offset here?
# Actually, the original code calculates U_deg = V_cell_degraded - V_CELL_BOL_NOM
# We should store the raw V_cell_degraded curve or the U_deg curve.
# The plan says: "Output: U_deg (degradation voltage, V)"
# But V_CELL_BOL_NOM depends on j_nom, T, P_op.
# It's safer to store the V_stack/V_cell vs Time curve, and let the component calculate U_deg dynamically if needed,
# OR calculate U_deg assuming the reference conditions are constant (which they are).

# Let's calculate V_cell vs Time
T_OP_H_TABLE = DEGRADATION_TABLE_YEARS * 8760.0 
V_CELL_TABLE = DEGRADATION_TABLE_V_STACK / N_cell_per_stack

# Create interpolator
v_cell_degraded_interpolator = interp1d(
    T_OP_H_TABLE, 
    V_CELL_TABLE, 
    kind='linear', 
    fill_value=(V_CELL_TABLE[0], V_CELL_TABLE[-1]), 
    bounds_error=False
)

# Generate a dense LUT for faster lookup without scipy at runtime if desired, 
# or just save the key points. The plan suggested saving the interpolated array.
# "Storage: h2_plant/data/lut_pem_degradation_interp.npy"

# Let's generate points every 100 hours up to 100,000 hours (approx 11 years)
max_hours = 100000
hours = np.arange(0, max_hours, 100)
v_cell_values = v_cell_degraded_interpolator(hours)

# We also need the BOL reference to calculate delta.
# In the original code: V_CELL_BOL_NOM = calculate_Vcell_base(j_nom, T, P_op)
# We don't have j_nom here easily without importing physics.
# But we can see DEGRADATION_TABLE_V_STACK[0] is at year 1.
# Wait, the table starts at year 1. What about year 0?
# The original code uses fill_value=(V_CELL_TABLE[0], ...) which means for t < 1 year, it uses Year 1 value?
# Let's check the original code:
# DEGRADATION_TABLE_YEARS = np.array([1.0, ...])
# fill_value=(V_CELL_TABLE[0], V_CELL_TABLE[-1])
# So for t=0, it returns V_CELL_TABLE[0].
# This implies no degradation difference between year 0 and year 1? Or rather, the table starts at year 1.
# Actually, usually BOL is t=0.
# If the table starts at year 1, then for t < 8760, it returns value at 8760.
# This seems to be the behavior of the original script. I will replicate it.

data = np.column_stack((hours, v_cell_values))

# Save to .npy
from pathlib import Path
output_path = str(Path(__file__).parent / "lut_pem_degradation.npy")
np.save(output_path, data)

print(f"LUT saved to {output_path}")
print(f"Shape: {data.shape}")
print(f"First 5 rows:\n{data[:5]}")
