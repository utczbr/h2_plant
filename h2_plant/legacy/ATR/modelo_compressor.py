import numpy as np
import CoolProp.CoolProp as CP
from scipy.interpolate import RegularGridInterpolator

# Geração da Tabela (Mantida para velocidade de cálculo)
P_vec = np.linspace(0.5e5, 25e5, 50)
T_vec = np.linspace(270.0, 900.0, 60)
H_mat = np.zeros((len(P_vec), len(T_vec)))
S_mat = np.zeros((len(P_vec), len(T_vec)))

for i, p in enumerate(P_vec):
    for j, t in enumerate(T_vec):
        H_mat[i, j] = CP.PropsSI('H', 'P', p, 'T', t, 'Nitrogen')
        S_mat[i, j] = CP.PropsSI('S', 'P', p, 'T', t, 'Nitrogen')

interp_H = RegularGridInterpolator((P_vec, T_vec), H_mat, bounds_error=False, fill_value=None)
interp_S = RegularGridInterpolator((P_vec, T_vec), S_mat, bounds_error=False, fill_value=None)

def modelo_compressor_ideal(T_in_C, P_in_Pa, P_out_Pa, m_dot_kg_s, Eta_is=0.75, Eta_m=0.95, Eta_el=0.93):
    try:
        if P_out_Pa <= P_in_Pa:
            return {'success': True, 'T_out_C': T_in_C, 'P_out_bar': P_out_Pa/1e5, 'W_dot_comp_W': 0.0, 'W_dot_elec_W': 0.0}
        
        T1_K = T_in_C + 273.15
        h1 = interp_H([P_in_Pa, T1_K])[0]
        s1 = interp_S([P_in_Pa, T1_K])[0]
        h2s = CP.PropsSI('H', 'S', s1, 'P', P_out_Pa, 'Nitrogen')
        
        dh_real = (h2s - h1) / Eta_is
        h2_real = h1 + dh_real
        T2_K = CP.PropsSI('T', 'H', h2_real, 'P', P_out_Pa, 'Nitrogen')
        
        W_dot_comp_W = (m_dot_kg_s * dh_real) / Eta_m
        return {
            'success': True,
            'T_out_C': T2_K - 273.15,
            'P_out_bar': P_out_Pa / 1e5,
            'W_dot_comp_W': W_dot_comp_W,
            'W_dot_elec_W': W_dot_comp_W / Eta_el
        }
    except:
        return {'success': False, 'T_out_C': T_in_C, 'P_out_bar': P_out_Pa/1e5, 'W_dot_comp_W': 0.0, 'W_dot_elec_W': 0.0}