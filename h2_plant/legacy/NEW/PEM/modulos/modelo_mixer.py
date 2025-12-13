# Mixer.py
import CoolProp.CoolProp as CP
import numpy as np

def mixer_model(input_streams, P_out_kPa):
    """
    Calcula as propriedades de saída de um misturador de água (Balanço de Massa e Energia).
    O input_streams é uma lista de dicionários: [{'m_dot': m, 'T': T_C, 'P': P_kPa}, ...]
    """
    
    fluid = 'Water'
    total_energy_in = 0.0 # kJ/s
    total_mass_in = 0.0   # kg/s
    detailed_input_data = [] 
    
    for i, stream in enumerate(input_streams):
        m_dot_i = stream['m_dot'] 
        T_i_C = stream['T']       
        P_i_kPa = stream['P']     
        
        T_i_K = T_i_C + 273.15 
        P_i_Pa = P_i_kPa * 1000 
        
        try:
            h_i_kJ_kg = CP.PropsSI('H', 'T', T_i_K, 'P', P_i_Pa, fluid) / 1000 
            total_mass_in += m_dot_i
            energy_in_i = m_dot_i * h_i_kJ_kg
            total_energy_in += energy_in_i
            
            detailed_input_data.append({
                'Stream': i + 1, 'm_dot (kg/s)': m_dot_i, 'T (°C)': T_i_C, 
                'P (kPa)': P_i_kPa, 'h (kJ/kg)': h_i_kJ_kg
            })
            
        except ValueError:
            # Em caso de erro termodinâmico (ex: água em estado inválido), retorna erro.
            return None, None

    num_input_streams = len(input_streams)
    output_stream_number = num_input_streams + 1
    m_dot_out = total_mass_in
    
    if m_dot_out <= 0: return detailed_input_data, None
        
    h_out_kJ_kg = total_energy_in / m_dot_out
    h_out_J_kg = h_out_kJ_kg * 1000 
    P_out_Pa = P_out_kPa * 1000   
    
    try:
        T_out_K = CP.PropsSI('T', 'H', h_out_J_kg, 'P', P_out_Pa, fluid)
        T_out_C = T_out_K - 273.15 
    except ValueError:
        T_out_C = np.nan
        
    output_results = {
        f'Vazão Mássica de Saída (kg/s) (m_dot_{output_stream_number})': m_dot_out,
        f'Entalpia Específica de Saída (kJ/kg) (h_{output_stream_number})': h_out_kJ_kg,
        f'Pressão de Saída (kPa) (P_{output_stream_number})': P_out_kPa,
        f'Temperatura de Saída (°C) (T_{output_stream_number})': T_out_C
    }
    
    return detailed_input_data, output_results