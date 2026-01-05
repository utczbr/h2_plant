# modelo_mixer.py
import CoolProp.CoolProp as CP
import numpy as np

def mixer_model(input_streams, P_out_bar):
    """
    Calcula as propriedades de saída de um misturador de água (Balanço de Massa e Energia).
    O input_streams é uma lista de dicionários: [{'m_dot': m, 'T': T_C, 'P': P_bar}, ...]
    """
    
    fluid = 'Water'
    total_energy_in = 0.0 # kJ/s
    total_mass_in = 0.0   # kg/s
    
    for stream in input_streams:
        m_dot_i = stream['m_dot'] 
        T_i_C = stream['T']       
        P_i_bar = stream['P']     
        
        T_i_K = T_i_C + 273.15 
        P_i_Pa = P_i_bar * 1e5 
        
        try:
            # H em J/kg -> convertido para kJ/kg
            h_i_kJ_kg = CP.PropsSI('H', 'T', T_i_K, 'P', P_i_Pa, fluid) / 1000.0 
            total_mass_in += m_dot_i
            energy_in_i = m_dot_i * h_i_kJ_kg
            total_energy_in += energy_in_i
            
        except Exception as e:
            print(f"Erro termodinâmico no mixer: {e}")
            return None

    if total_mass_in <= 0:
        return {
            'm_dot_out_kg_s': 0.0,
            'T_out_C': 0.0,
            'h_out_kJ_kg': 0.0,
            'm_dot_out_kg_h': 0.0
        }
        
    h_out_kJ_kg = total_energy_in / total_mass_in
    h_out_J_kg = h_out_kJ_kg * 1000.0 
    P_out_Pa = P_out_bar * 1e5   
    
    try:
        T_out_K = CP.PropsSI('T', 'H', h_out_J_kg, 'P', P_out_Pa, fluid)
        T_out_C = T_out_K - 273.15 
    except Exception:
        T_out_C = np.nan
        
    return {
        'm_dot_out_kg_s': total_mass_in,
        'T_out_C': T_out_C,
        'h_out_kJ_kg': h_out_kJ_kg,
        'm_dot_out_kg_h': total_mass_in * 3600.0
    }