import numpy as np
import pickle
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from h2_plant.models.pem_physics import calculate_Vcell, PEMConstants
from h2_plant.config.constants_physics import PEMConstants as PEMConstantsConfig

# Initialize constants
CONST = PEMConstantsConfig()

# Load degradation LUT
deg_path = "/home/stuart/Documentos/Planta Hidrogenio/h2_plant/data/lut_pem_degradation.npy"
deg_data = np.load(deg_path)
from scipy.interpolate import interp1d
deg_interpolator = interp1d(deg_data[:, 0], deg_data[:, 1], kind='linear', fill_value="extrapolate")

def get_u_deg(t_op_h):
    # U_deg = V_cell_degraded(t) - V_cell_BOL
    # We need V_cell_BOL.
    # In the original script: V_CELL_BOL_NOM = calculate_Vcell_base(j_nom, T, P_op)
    # Here we can calculate it using our physics model.
    # But wait, the degradation table stores V_stack (or V_cell).
    # So V_cell(t) = V_cell_from_table(t) + (V_cell_physics(j) - V_cell_physics(j_nom))?
    # No, the original model was:
    # U_deg = V_cell_degraded_from_table(t) - V_CELL_BOL_NOM
    # V_cell(j, t) = V_base(j) + U_deg(t)
    
    # Let's calculate V_CELL_BOL_NOM
    # j_nom = 2.91 (from original script)
    j_nom = 2.91
    T = 333.15
    P_op = 40.0e5
    
    # We need to use the physics model to get V_base(j_nom)
    # But wait, calculate_Vcell in pem_physics.py calls calculate_Vcell_base + U_deg.
    # So we can use calculate_Vcell_base directly.
    from h2_plant.models.pem_physics import calculate_Vcell_base
    
    V_cell_bol_nom = calculate_Vcell_base(j_nom, T, P_op)
    
    v_cell_degraded = deg_interpolator(t_op_h)
    u_deg = np.maximum(0.0, v_cell_degraded - V_cell_bol_nom)
    return u_deg

# Define ranges
j_op_min = 0.001
j_op_max = 4.0
j_points = 50
j_grid = np.linspace(j_op_min, j_op_max, j_points)

t_op_h_max = 39420 # 4.5 years
t_points = 100
t_grid = np.linspace(0, t_op_h_max, t_points)

# Generate LUT
# Shape: [j_idx, t_idx]
v_cell_lut = np.zeros((j_points, t_points))

T = 333.15
P_op = 40.0e5

print("Generating PEM V_cell LUT...")
for i, j in enumerate(j_grid):
    for k, t in enumerate(t_grid):
        u_deg = get_u_deg(t)
        v_cell_lut[i, k] = calculate_Vcell(j, T, P_op, u_deg)

# Save to .npz
output_path = "/home/stuart/Documentos/Planta Hidrogenio/h2_plant/data/lut_pem_vcell.npz"
np.savez(output_path, v_cell=v_cell_lut, j_op=j_grid, t_op_h=t_grid)

print(f"LUT saved to {output_path}")
print(f"Shape: {v_cell_lut.shape}")
