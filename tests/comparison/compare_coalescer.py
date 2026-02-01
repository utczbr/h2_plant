"""
Comparison test: New Coalescer component vs Reference CoalescerModel.py

Uses exact same parameters from reference model to validate fidelity.
"""
import math
import sys
sys.path.insert(0, '/home/stuart/Documentos/Planta Hidrogenio')

from h2_plant.components.separation.coalescer import Coalescer
from h2_plant.core.stream import Stream
from h2_plant.core.constants import CoalescerConstants, ConversionFactors

# ============================================================================
# REFERENCE MODEL CONSTANTS (from CoalescerModel.py)
# ============================================================================
BAR_TO_PA = 1e5
PA_TO_BAR = 1e-5
M_H2 = 0.002016
M_O2 = 0.031998
M_H2O = 0.018015
R_J_K_mol = 8.31446
CELSIUS_TO_KELVIN = 273.15

P_NOMINAL_BAR = 39.70
T_NOMINAL_C = 4.00
VARIACAO_PERCENTUAL = 0.10

VISCOSIDADE_H2_REF = 9.0e-6
T_REF_K = 303.15
K_PERDA_EMPIRICO = 0.5e6
ETA_LIQ_REMOCAO = 0.9999
CARGA_LIQUIDA_RESIDUAL_MGM3 = 100.0

# Flow data from reference
FLUXOS = {
    'H2': {
        'M_gas': M_H2,
        'Q_M_nominal_kg_h': 0.02235 * 3600,  # 80.46 kg/h
        'D_shell_dim': 0.32,
        'L_elem_m': 1.00
    },
    'O2': {
        'M_gas': M_O2,
        'Q_M_nominal_kg_h': 0.19647 * 3600,  # 707.29 kg/h
        'D_shell_dim': 0.32,
        'L_elem_m': 1.00
    }
}

def reference_model(gas_name: str, Q_M_OP: float, M_gas: float, D_shell: float, L_elem: float):
    """
    Reference implementation from CoalescerModel.py
    """
    # Worst case conditions
    P_min_bar = P_NOMINAL_BAR * (1 - VARIACAO_PERCENTUAL)  # 35.73 bar
    T_nominal_K = T_NOMINAL_C + CELSIUS_TO_KELVIN
    T_max_op_K = T_nominal_K * (1 + VARIACAO_PERCENTUAL)  # ~304.87 K
    
    P_total_Pa = P_min_bar * BAR_TO_PA
    
    # Saturation pressure estimate
    P_sat_bar = 0.0469
    
    # Mixture properties
    y_H2O = P_sat_bar / P_min_bar
    y_gas = 1.0 - y_H2O
    M_avg = (y_gas * M_gas) + (y_H2O * M_H2O)
    rho_mix = (P_total_Pa * M_avg) / (R_J_K_mol * T_max_op_K)
    
    # Viscosity
    mu_g = VISCOSIDADE_H2_REF * (T_max_op_K / T_REF_K) ** 0.7
    
    # Volumetric flow
    Q_V_OP_total = Q_M_OP / rho_mix / 3600  # m³/s
    Q_V_OP_total_m3_h = Q_M_OP / rho_mix  # m³/h
    
    # Pressure drop
    A_shell = (math.pi / 4) * (D_shell ** 2)
    U_superficial = Q_V_OP_total / A_shell
    Delta_P_Pa_limpa = K_PERDA_EMPIRICO * mu_g * L_elem * U_superficial
    Delta_P_bar_limpa = Delta_P_Pa_limpa * PA_TO_BAR
    
    # Power
    Potencia_gasta_W = Q_V_OP_total * Delta_P_Pa_limpa
    
    # Liquid removal
    Q_M_liq_in_kg_h = (CARGA_LIQUIDA_RESIDUAL_MGM3 * Q_V_OP_total_m3_h) / 1e6
    Q_M_liq_out_kg_h = Q_M_liq_in_kg_h * (1.0 - ETA_LIQ_REMOCAO)
    C_liq_out_mg_m3 = (Q_M_liq_out_kg_h * 1e6) / Q_V_OP_total_m3_h
    
    return {
        'P_min_bar': P_min_bar,
        'T_max_K': T_max_op_K,
        'rho_mix': rho_mix,
        'mu_g': mu_g,
        'U_sup_m_s': U_superficial,
        'Delta_P_bar': Delta_P_bar_limpa,
        'Power_W': Potencia_gasta_W,
        'C_liq_out_mg_m3': C_liq_out_mg_m3,
        'Q_V_m3_h': Q_V_OP_total_m3_h
    }


def new_component(gas_name: str, Q_M_OP: float, M_gas: float, D_shell: float, L_elem: float):
    """
    New Coalescer component implementation
    """
    # Use same worst-case conditions
    P_min_bar = P_NOMINAL_BAR * (1 - VARIACAO_PERCENTUAL)
    T_nominal_K = T_NOMINAL_C + CELSIUS_TO_KELVIN
    T_max_op_K = T_nominal_K * (1 + VARIACAO_PERCENTUAL)
    
    P_total_Pa = P_min_bar * BAR_TO_PA
    
    # Create coalescer
    coalescer = Coalescer(d_shell=D_shell, l_elem=L_elem, gas_type=gas_name)
    coalescer.initialize(dt=1.0, registry=None)
    
    # Calculate inlet composition matching reference model
    # Reference includes H2O vapor in density via saturation pressure
    P_sat_bar = 0.0469
    y_H2O_vapor = P_sat_bar / P_min_bar  # Mole fraction
    
    # Convert mole fraction to mass fraction for Stream
    # M_gas is already available from function argument
    # M_H2O is a global constant
    y_gas = 1.0 - y_H2O_vapor
    
    # Mass fractions
    w_H2O_vapor = (y_H2O_vapor * M_H2O) / (y_gas * M_gas + y_H2O_vapor * M_H2O)
    w_gas = 1.0 - w_H2O_vapor
    
    # Liquid aerosol (from C_liq = 100 mg/m³)
    # First estimate density for Q_V
    M_avg = (y_gas * M_gas + y_H2O_vapor * M_H2O)
    rho_approx = (P_total_Pa * M_avg) / (R_J_K_mol * T_max_op_K)
    Q_V_m3_h = Q_M_OP / rho_approx
    liq_in_kg_h = (CARGA_LIQUIDA_RESIDUAL_MGM3 * Q_V_m3_h) / 1e6
    liq_frac = liq_in_kg_h / Q_M_OP if Q_M_OP > 0 else 0
    
    # Composition for Stream - include H2O vapor for correct density!
    gas_comp = gas_name if gas_name in ['H2', 'O2'] else 'H2'
    inlet = Stream(
        mass_flow_kg_h=Q_M_OP,
        temperature_k=T_max_op_K,
        pressure_pa=P_total_Pa,
        composition={
            gas_comp: w_gas * (1 - liq_frac),
            'H2O': w_H2O_vapor * (1 - liq_frac),  # H2O vapor for density
            'H2O_liq': liq_frac  # Aerosol for separation
        }
    )
    
    # Process
    coalescer.receive_input('inlet', inlet)
    outlet = coalescer.get_output('outlet')
    drain = coalescer.get_output('drain')
    
    # Calculate output concentration
    rho_out = outlet.density_kg_m3 if outlet.density_kg_m3 > 0 else rho_mix_approx
    Q_V_out_m3_h = outlet.mass_flow_kg_h / rho_out if rho_out > 0 else 0
    liq_out_kg_h = outlet.mass_flow_kg_h * outlet.composition.get('H2O_liq', 0)
    C_liq_out = (liq_out_kg_h * 1e6) / Q_V_out_m3_h if Q_V_out_m3_h > 0 else 0
    
    return {
        'Delta_P_bar': coalescer.current_delta_p_bar,
        'Power_W': coalescer.current_power_loss_w,
        'C_liq_out_mg_m3': C_liq_out,
        'drain_kg_h': drain.mass_flow_kg_h,
        'rho_stream': inlet.density_kg_m3
    }


def compare():
    """Run comparison for H2 and O2"""
    print("=" * 70)
    print("COALESCER VALIDATION: Reference vs New Component")
    print("=" * 70)
    
    for gas_name, data in FLUXOS.items():
        Q_M = data['Q_M_nominal_kg_h']
        M_gas = data['M_gas']
        D_shell = data['D_shell_dim']
        L_elem = data['L_elem_m']
        
        ref = reference_model(gas_name, Q_M, M_gas, D_shell, L_elem)
        new = new_component(gas_name, Q_M, M_gas, D_shell, L_elem)
        
        print(f"\n--- {gas_name} Stream (Q_M = {Q_M:.2f} kg/h) ---")
        print(f"{'Parameter':<25} {'Reference':<15} {'New Component':<15} {'Diff %':<10}")
        print("-" * 65)
        
        # Pressure drop
        diff_dp = abs(ref['Delta_P_bar'] - new['Delta_P_bar']) / ref['Delta_P_bar'] * 100 if ref['Delta_P_bar'] > 0 else 0
        print(f"{'Pressure Drop (bar)':<25} {ref['Delta_P_bar']:<15.6f} {new['Delta_P_bar']:<15.6f} {diff_dp:<10.2f}")
        
        # Power
        diff_pwr = abs(ref['Power_W'] - new['Power_W']) / ref['Power_W'] * 100 if ref['Power_W'] > 0 else 0
        print(f"{'Power Loss (W)':<25} {ref['Power_W']:<15.4f} {new['Power_W']:<15.4f} {diff_pwr:<10.2f}")
        
        # Liquid output concentration
        diff_cliq = abs(ref['C_liq_out_mg_m3'] - new['C_liq_out_mg_m3']) / ref['C_liq_out_mg_m3'] * 100 if ref['C_liq_out_mg_m3'] > 0 else 0
        print(f"{'C_liq_out (mg/m³)':<25} {ref['C_liq_out_mg_m3']:<15.4f} {new['C_liq_out_mg_m3']:<15.4f} {diff_cliq:<10.2f}")
        
        # PASS/FAIL
        max_diff = max(diff_dp, diff_pwr, diff_cliq)
        status = "✅ PASS" if max_diff < 5.0 else "⚠️  CHECK"
        print(f"\nMax deviation: {max_diff:.2f}% — {status}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    compare()
