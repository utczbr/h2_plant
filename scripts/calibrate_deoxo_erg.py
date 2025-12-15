import sys
import os
import numpy as np

# Adjust path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from h2_plant.components.purification.deoxo_reactor import DeoxoReactor
from h2_plant.core.stream import Stream
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.constants import DeoxoConstants, GasConstants

def calibrate_deoxo():
    print("\n=== DEOXO CALIBRATION SCRIPT ===")
    
    # Target Values from Legacy (Independent Test)
    TARGET_DP_BAR = 0.05
    TARGET_DELTA_T = 1.6050 # C
    
    # Test Conditions
    m_dot_h = 92.0
    P_Pa = 39.0 * 1e5
    T_K = 293.15
    y_O2 = 200e-6
    
    # Setup Component
    deoxo = DeoxoReactor('calib')
    registry = ComponentRegistry()
    deoxo.initialize(1/60, registry)
    
    # Setup Stream
    y_H2 = 1.0 - y_O2
    MW_mix = 2.016e-3 * y_H2 + 32e-3 * y_O2
    x_O2 = (y_O2 * 32e-3) / MW_mix
    x_H2 = 1.0 - x_O2
    
    stream = Stream(m_dot_h, T_K, P_Pa, {'H2': x_H2, 'O2': x_O2})
    deoxo.input_stream = stream # Direct inject for util access
    
    # 1. Calibrate Pressure Drop (Ergun Design Point)
    # Current constant: dp_design = 0.0019
    # We want 0.05 at this flow.
    # Current logic: u_curr = ...; ratio = u/u_design; dp = dp_design * ratio^1.5
    # If we update dp_design such that output is 0.05.
    # Let's run step to see u_curr.
    
    # We can invoke internal logic:
    molar_flow = (m_dot_h/3600)/MW_mix
    rho_gas = (P_Pa * MW_mix) / (8.314 * T_K)
    vol_flow = (m_dot_h/3600) / rho_gas
    u_curr = vol_flow / DeoxoConstants.AREA_REACTOR_M2
    print(f"Current Velocity u: {u_curr:.4f} m/s")
    
    # We want DP = 0.05 when u = u_curr.
    # We can just set u_design = u_curr (0.09 m/s?) and dp_design = 0.05.
    # This defines the design point AS the nominal operation.
    print(f"Proposed U_DESIGN: {u_curr:.6f}")
    print(f"Proposed DP_DESIGN: {TARGET_DP_BAR:.6f}")
    
    # 2. Calibrate Cp for Delta T Parity
    # Current Delta T with Cp=29.5 is 1.6438.
    # Q = m_dot * Cp * dT.
    # Since Q is fixed (same reaction extent checked previously), Cp * dT = Const.
    # Cp_new * dT_target = Cp_old * dT_old.
    # Cp_new = Cp_old * (dT_old / dT_target).
    # Wait.
    # If we want LOWER Delta T (1.6050), we need HIGHER Cp (more thermal mass).
    # Cp_new = 29.5 * (1.6438 / 1.6050).
    
    current_dT = 1.6438
    cp_new = 29.5 * (current_dT / TARGET_DELTA_T)
    
    print(f"Current DeltaT: {current_dT}")
    print(f"Target DeltaT: {TARGET_DELTA_T}")
    print(f"Proposed CP_MIX: {cp_new:.4f}")
    
    return u_curr, TARGET_DP_BAR, cp_new

if __name__ == "__main__":
    calibrate_deoxo()
