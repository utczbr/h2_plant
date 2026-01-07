# modelo_trocador_hex.py
import numpy as np

def modelar_hex_recuperacao(m_dot_gas, cp_mix, T_in_gas, T_w_in, m_dot_w):
    """
    Modelo de HEX baseado em dimensionamento fixo (U e Area).
    Utiliza o método epsilon-NTU para calcular a troca térmica real.
    """
    # --- Parâmetros de Projeto (Dimensionamento Nominal) ---
    U = 0.100  # kW/m2.K (equivalente a 100 W/m2.K)
    AREA = 152.0  # m2 (calculada para o caso nominal)
    cp_agua = 4.186  # kJ/kg.K
    T_gas_min = 40.0  # Limite de segurança para evitar condensação excessiva/corrosão

    # 1. Capacidades caloríficas (C = m_dot * cp)
    C_gas = m_dot_gas * cp_mix
    C_agua = m_dot_w * cp_agua
    
    C_min = min(C_gas, C_agua)
    C_max = max(C_gas, C_agua)
    C_r = C_min / C_max
    
    # 2. NTU (Number of Transfer Units)
    ntu = (U * AREA) / C_min
    
    # 3. Efetividade (epsilon) - Assumindo Contra-corrente
    if C_r < 1.0:
        epsilon = (1 - np.exp(-ntu * (1 - C_r))) / (1 - C_r * np.exp(-ntu * (1 - C_r)))
    else:
        # Caso especial onde C_r = 1
        epsilon = ntu / (1 + ntu)
        
    # 4. Troca Térmica Máxima e Real
    q_max = C_min * (T_in_gas - T_w_in)
    q_real = epsilon * q_max
    
    # 5. Cálculo das Temperaturas de Saída
    t_out_gas_calc = T_in_gas - (q_real / C_gas)
    
    # Aplicação do limite de segurança T_gas_min (40°C)
    if t_out_gas_calc < T_gas_min:
        t_out_gas = T_gas_min
        q_real = C_gas * (T_in_gas - t_out_gas)
        t_out_w = T_w_in + (q_real / C_agua)
    else:
        t_out_gas = t_out_gas_calc
        t_out_w = T_w_in + (q_real / C_agua)
        
    return {
        "Q_hex_kW": q_real,
        "T_out_gas": t_out_gas,
        "T_out_w": t_out_w,
        "T_in_w": T_w_in,
        "m_dot_w": m_dot_w,
        "epsilon": epsilon,
        "UA_calc": U * AREA
    }