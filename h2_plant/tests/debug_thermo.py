
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))
from h2_plant.optimization.lut_manager import LUTManager
import CoolProp.CoolProp as CP

def debug_thermo():
    print("Debug Thermo H2")
    
    # Direct CoolProp
    P1 = 30e5
    T1 = 300.0
    P2 = 5e5
    
    h1 = CP.PropsSI('H', 'P', P1, 'T', T1, 'H2')
    h2_sameT = CP.PropsSI('H', 'P', P2, 'T', T1, 'H2')
    
    print(f"State 1 (30 bar, 300K): H = {h1:.2f} J/kg")
    print(f"State 2 (5 bar, 300K):  H = {h2_sameT:.2f} J/kg")
    
    diff = h1 - h2_sameT
    print(f"Excess Enthalpy in State 1: {diff:.2f} J/kg")
    
    # Cp approx 14300 J/kgK
    cp = CP.PropsSI('C', 'P', P2, 'T', T1, 'H2')
    print(f"Cp at State 2: {cp:.2f} J/kgK")
    
    dT_est = diff / cp
    print(f"Estimated T rise: {dT_est:.2f} K")
    
    # Verify my solver logic manually
    # Find T such that H(5 bar, T) = h1
    # Using CoolProp
    T_final_CP = CP.PropsSI('T', 'P', P2, 'H', h1, 'H2')
    print(f"CoolProp calculated T_final: {T_final_CP:.2f} K")
    
    # Test LUT Manager
    lut = LUTManager()
    lut.initialize()
    
    h1_lut = lut.lookup('H2', 'H', P1, T1)
    print(f"LUT H1: {h1_lut:.2f}")
    
    h2_lut_t1 = lut.lookup('H2', 'H', P2, T1)
    print(f"LUT H2 (at T1): {h2_lut_t1:.2f}")
    
if __name__ == "__main__":
    debug_thermo()
