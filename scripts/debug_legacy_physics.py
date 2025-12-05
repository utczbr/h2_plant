import sys
from pathlib import Path
import numpy as np

# Setup path to import pem_operator
PROJECT_ROOT = Path(__file__).parent.parent
LEGACY_DIR = PROJECT_ROOT / 'h2_plant' / 'legacy' / 'pem_soec_reference'
sys.path.insert(0, str(LEGACY_DIR))

import pem_operator

def debug_physics():
    print("--- Debugging Legacy PEM Physics ---")
    
    # Initialize state
    state = pem_operator.initialize_pem_simulation()
    
    # Target Power: 5 MW
    P_target_kW = 5000.0
    
    print(f"Target Power: {P_target_kW} kW")
    
    # Run step
    m_H2, m_O2, m_H2O, state = pem_operator.run_pem_step(P_target_kW, state)
    
    print(f"Output H2: {m_H2:.4f} kg/min")
    print(f"Output H2 (Hourly): {m_H2 * 60:.4f} kg/h")
    
    # We can't easily inspect internal vars of run_pem_step without modifying it.
    # But we can call the internal functions if they are available.
    
    # Re-calculate j using the same logic as run_pem_step
    P_target_W = P_target_kW * 1000.0
    
    # From pem_operator.py
    j_nom = pem_operator.j_nom
    P_nominal = pem_operator.P_nominal_sistema_W
    j_guess = j_nom * (P_target_W / P_nominal)
    print(f"j_guess: {j_guess:.4f}")
    
    from scipy.optimize import fsolve
    
    def func_to_solve(j_g):
        return pem_operator.P_input_system_from_j(j_g, pem_operator.T, pem_operator.P_op, state.t_op_h) - P_target_W
        
    j_sol, infodict, ier, msg = fsolve(func_to_solve, j_guess, full_output=True, xtol=1e-4)
    
    print(f"fsolve ier: {ier} (1=Success)")
    print(f"fsolve msg: {msg}")
    print(f"j_sol: {j_sol[0]:.4f}")
    
    j_final = j_sol[0] if ier == 1 else j_guess
    print(f"j_final used: {j_final:.4f}")
    
    # Calculate V_cell
    V_cell = pem_operator.calculate_Vcell(j_final, pem_operator.T, pem_operator.P_op, state.t_op_h)
    print(f"V_cell: {V_cell:.4f} V")
    
    # Calculate I_total
    I_total = j_final * pem_operator.Area_Total
    print(f"I_total: {I_total:.2f} A")
    
    # Calculate Power
    P_calc = I_total * V_cell
    print(f"P_stack (I*V): {P_calc/1e6:.4f} MW")
    
    # Calculate Efficiency
    # 33.3 kWh/kg -> 1 kg = 33.3 kWh = 120 MJ
    # Power = 5 MW. H2 = 137 kg/h = 0.038 kg/s.
    # Energy in H2 = 0.038 * 120 = 4.56 MW.
    # Eff = 4.56 / 5.0 = 91%.
    
    h2_per_hour = m_H2 * 60
    energy_in_h2_mw = h2_per_hour * 33.33 / 1000.0
    eff = energy_in_h2_mw / (P_target_kW / 1000.0)
    print(f"Efficiency (LHV): {eff*100:.2f}%")

if __name__ == "__main__":
    debug_physics()
