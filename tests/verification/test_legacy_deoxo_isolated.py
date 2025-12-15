import sys
import os
import numpy as np

# Adjust path to find modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from h2_plant.legacy.NEW.PEM.modulos.modelo_deoxo import modelar_deoxo

def test_legacy_deoxo():
    print("\n=== LEGACY MODELO_DEOXO ISOLATED TEST ===")
    
    # Inputs
    m_dot_kg_h = 92.0
    m_dot_kg_s = m_dot_kg_h / 3600.0
    P_bar = 39.0
    T_C = 20.0
    
    # Legacy expects Mole Fractions (y)
    # 200 ppm O2 by Mass approx 75-100 ppm by Mole?
    # H2 (2g/mol), O2 (32g/mol).
    # x_O2 = 0.0002.
    # n_O2 ~ 0.0002 / 32 ~ 6.25e-6
    # n_H2 ~ 0.9998 / 2 ~ 0.5
    # y_O2 = 6.25e-6 / 0.5 ~ 1.25e-5 (12.5 ppm)
    # BUT comparison test used 0.0002 as the number passed into "y_o2_in" for Legacy?
    # Let's check compare_deoxo.py... it didn't run legacy. 
    # I will use y_O2 = 0.0002 (200 ppm mole) to be rigorous about inputs, 
    # OR should I match the mass fraction used in the New component?
    # I will TEST with y_O2 = 200e-6 (200 ppm mole) for clarity.
    
    y_O2 = 200e-6
    y_H2O = 0.0
    L_m = 1.334 # Matched length
    
    print(f"Inputs: Flow={m_dot_kg_h} kg/h, P={P_bar} bar, T={T_C} C")
    print(f"Composition (Mole): y_O2={y_O2:.2e}, y_H2O={y_H2O}")
    print(f"Length: {L_m} m")
    
    # Run
    results = modelar_deoxo(m_dot_kg_s, P_bar, T_C, y_H2O, y_O2, L_m)
    
    # Report
    print("\n--- Outputs ---")
    print(f"T_out: {results['T_C']:.4f} C")
    print(f"P_out: {results['P_bar']:.4f} bar")
    print(f"y_O2_out: {results['y_O2_out']:.2e}")
    print(f"Conversion X_O2: {results['X_O2']*100:.4f} %")
    print(f"T_max: {results['T_max_calc']:.4f} C")
    
    # Heat Balance Check
    # Q = m_dot * Cp * deltaT
    # Q_gen = n_O2_react * DeltaH
    delta_T = results['T_C'] - T_C
    print(f"Delta T: {delta_T:.4f} C")

if __name__ == "__main__":
    test_legacy_deoxo()
