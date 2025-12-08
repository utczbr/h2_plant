
import pytest
import numpy as np
from h2_plant.core.constants import DeoxoConstants, GasConstants
from h2_plant.optimization import numba_ops

class TestDeoxoSolver:
    
    def test_design_conditions(self):
        """
        Verify solver reproduces 99.99% conversion for design inputs.
        Source: dimensionando_deoxer.pdf
        """
        # Design Inputs
        # 0.02235 kg/s, 2% O2
        m_dot_total = 0.02235 # kg/s
        y_o2 = 0.02
        y_h2 = 0.98
        
        MW_H2 = 2.016e-3
        MW_O2 = 32.00e-3
        MW_MIX = y_o2 * MW_O2 + y_h2 * MW_H2 # approx 0.0026 kg/mol
        
        n_dot_total = m_dot_total / MW_MIX # approx 8.5 mol/s
        
        T_in = 4.0 + 273.15 # 277.15 K
        P_in = 39.55e5 # 39.55 bar
        
        conversion, t_out, t_peak = numba_ops.solve_deoxo_pfr_step(
            L_total=DeoxoConstants.L_REACTOR_M,
            steps=100,
            T_in=T_in,
            P_in_pa=P_in,
            molar_flow_total=n_dot_total,
            y_o2_in=y_o2,
            k0=DeoxoConstants.K0_VOL_S1,
            Ea=DeoxoConstants.EA_J_MOL,
            R=GasConstants.R_UNIVERSAL_J_PER_MOL_K,
            delta_H=DeoxoConstants.DELTA_H_RXN_J_MOL_O2,
            U_a=DeoxoConstants.U_A_W_M3_K,
            T_jacket=DeoxoConstants.T_JACKET_K,
            Area=DeoxoConstants.AREA_REACTOR_M2,
            Cp_mix=DeoxoConstants.CP_MIX_AVG_J_MOL_K
        )
        
        print(f"Conversion: {conversion*100:.4f}%")
        print(f"T_out: {t_out:.2f} K")
        print(f"T_peak: {t_peak:.2f} K")
        
        # Validation
        assert conversion > 0.999, "Conversion should be > 99.9%"
        assert t_peak < (190.0 + 273.15), "Peak temp exceeded 190C design limit"
        assert t_out > (100.0 + 273.15), "Outlet temp should be elevated"

if __name__ == "__main__":
    t = TestDeoxoSolver()
    t.test_design_conditions()
