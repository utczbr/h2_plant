"""
Script to Calibrate Coalescer K_PERDA for 0.1500 bar Pressure Drop
"""
import sys
import time
sys.path.insert(0, '/home/stuart/Documentos/Planta Hidrogenio')

from h2_plant.components.separation.coalescer import Coalescer
from h2_plant.core.stream import Stream
from h2_plant.core.constants import CoalescerConstants

# Target
TARGET_DP_BAR = 0.1500
TOLERANCE = 0.0005

# Input Conditions (Legacy Nominal)
Q_M = 80.46  # kg/h
P_IN_BAR = 35.73
T_IN_K = 277.15
BAR_TO_PA = 1e5

def test_k(k_value):
    # Hijack the constant
    CoalescerConstants.K_PERDA = k_value
    
    coal = Coalescer(gas_type='H2')
    coal.initialize(dt=1/60, registry=None)
    
    inlet = Stream(
        mass_flow_kg_h=Q_M,
        temperature_k=T_IN_K,
        pressure_pa=P_IN_BAR * BAR_TO_PA,
        composition={'H2': 1.0}
    )
    
    coal.receive_input('inlet', inlet)
    return coal.current_delta_p_bar

print(f"Calibrating K_PERDA for Target DeltaP = {TARGET_DP_BAR} bar...")

# Initial Guess
k_current = 1.7e10
step = 1e8
iteration = 0

while True:
    dp = test_k(k_current)
    error = dp - TARGET_DP_BAR
    
    print(f"Iter {iteration}: K={k_current:.4e} -> dP={dp:.6f} bar (Err={error:.6f})")
    
    if abs(error) <= TOLERANCE:
        print("\n[CONVERGED]")
        print(f"Optimal K_PERDA = {k_current:.6e}")
        break
        
    # Simple proportional adjustment
    # dP is linear with K (Darcy law approx in this regime)
    # K_new = K_old * (Target / Current)
    k_current = k_current * (TARGET_DP_BAR / dp)
    iteration += 1
    
    if iteration > 20:
        print("Max iterations reached.")
        break

