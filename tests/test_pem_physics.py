import pytest
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from h2_plant.models.pem_physics import calculate_Urev, calculate_Vcell_base, calculate_Vcell, calculate_eta_F
from h2_plant.config.constants_physics import PEMConstants

CONST = PEMConstants()

def test_pem_urev_calculation():
    T = 333.15  # 60°C
    P_op = 40e5  # 40 bar
    U_rev = calculate_Urev(T, P_op)
    # Expected value around 1.27V at 40 bar
    assert 1.25 < U_rev < 1.30, f"Expected ~1.27V, got {U_rev}"

def test_pem_vcell_base():
    j_op = 2.91  # Nominal
    T, P_op = 333.15, 40e5
    V_cell = calculate_Vcell_base(j_op, T, P_op)
    # Expected value around 1.8V (typical for PEM at nominal)
    assert 1.5 < V_cell < 2.2, f"Expected ~1.8V, got {V_cell}"

def test_pem_eta_f():
    j_op = 2.91
    eta = calculate_eta_F(j_op)
    assert 0.9 < eta < 1.0, f"Expected high efficiency, got {eta}"
    
    j_low = 0.01
    eta_low = calculate_eta_F(j_low)
    assert eta_low < eta, "Efficiency should drop at very low current density due to shunt currents"

if __name__ == "__main__":
    test_pem_urev_calculation()
    test_pem_vcell_base()
    test_pem_eta_f()
    print("All physics tests passed!")
