# Mixer.py
import CoolProp.CoolProp as CP
import numpy as np

def mixer_model(input_streams, P_out_kPa):
    """
    Calcula as propriedades de saída de um misturador de água (Balanço de Massa e Energia).
    O input_streams é uma lista de dicionários: [{'m_dot': m, 'T': T_C, 'P': P_kPa, 'H': H_J_kg}, ...]
    """
    
    fluid = 'Water'
    total_energy_in = 0.0 # J/s
    total_mass_in = 0.0   # kg/s
    detailed_input_data = [] 
    
    for i, stream in enumerate(input_streams):
        m_dot_i = stream['m_dot'] 
        T_i_C = stream['T']       
        P_i_kPa = stream['P']     
        h_i_J_kg = stream['H']    
        
        total_mass_in += m_dot_i
        energy_in_i = m_dot_i * h_i_J_kg # H está em J/kg
        total_energy_in += energy_in_i
        
        detailed_input_data.append({
            'Stream': i + 1, 'm_dot (kg/s)': m_dot_i, 'T (°C)': T_i_C, 
            'P (kPa)': P_i_kPa, 'h (kJ/kg)': h_i_J_kg / 1000.0
        })

    num_input_streams = len(input_streams)
    output_stream_number = num_input_streams + 1
    m_dot_out = total_mass_in
    
    if m_dot_out <= 0: return detailed_input_data, None
        
    h_out_J_kg = total_energy_in / m_dot_out
    h_out_kJ_kg = h_out_J_kg / 1000.0 
    P_out_Pa = P_out_kPa * 1000   
    
    # NÃO HÁ FALLBACK: Confia-se no CoolProp
    T_out_K = CP.PropsSI('T', 'H', h_out_J_kg, 'P', P_out_Pa, fluid)
    T_out_C = T_out_K - 273.15 
        
    output_results = {
        f'Vazão Mássica de Saída (kg/s) (m_dot_{output_stream_number})': m_dot_out,
        f'Entalpia Específica de Saída (kJ/kg) (h_{output_stream_number})': h_out_kJ_kg,
        f'Pressão de Saída (kPa) (P_{output_stream_number})': P_out_kPa,
        f'Temperatura de Saída (°C) (T_{output_stream_number})': T_out_C
    }
    
    return detailed_input_data, output_results