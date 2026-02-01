from h2_plant.core.stream import Stream
from h2_plant.core.constants import GasConstants, StandardConditions

s = Stream(1.0, 300.0, 101325.0, {'H2': 1.0})
print(f"T: {s.temperature_k}")
print(f"T_ref: {StandardConditions.TEMPERATURE_K}")

data = GasConstants.SPECIES_DATA['H2']
coeffs = data['cp_coeffs']
print(f"Coeffs: {coeffs}")

A, B, C, D, E = coeffs
def integral_cp(t):
    val = (A * t + 
            B * t**2 / 2 + 
            C * t**3 / 3 + 
            D * t**4 / 4 - 
            E / t)
    print(f"Integral at {t}: {val}")
    return val

i_t = integral_cp(300.0)
i_ref = integral_cp(298.15)
diff = i_t - i_ref
print(f"Diff (J/mol): {diff}")

mw = data['molecular_weight']
h_j_kg = diff * 1000.0 / mw
print(f"H (J/kg): {h_j_kg}")

print(f"Stream H: {s.specific_enthalpy_j_kg}")
