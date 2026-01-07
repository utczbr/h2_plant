import numpy as np

# --- PARÂMETROS DE DIMENSIONAMENTO RECOMENDADOS ---
# Redimensionado para UA_nominal = 100 W/K (Fim do superdimensionamento)
T_AMB = 30.0  
CP_GAS = 1700.0  

# Constante de performance baseada na Lei das Afinidades
# Referência: 300W para 250 W/K. K = 300 / 250^3
K_PERF = 300.0 / (250.0**3)

def simular_dry_cooler_com_setpoint(T_gas_in, T_target, m_dot_gas, W_min=5.0):
    """
    Calcula o consumo real acoplado à troca térmica.
    Sem travas arbitrárias, refletindo o dimensionamento para 100 W/K.
    """
    C_gas = max(0.01, m_dot_gas * CP_GAS)
    
    # 1. Efetividade requerida para atingir o setpoint
    # eff = (T_in - T_out) / (T_in - T_amb)
    eff_req = (T_gas_in - T_target) / max(0.1, T_gas_in - T_AMB)
    eff_req = min(0.95, max(0.01, eff_req))
    
    # 2. UA necessário baseado no método NTU (C_min = C_gas)
    NTU_req = -np.log(1 - eff_req)
    UA_req = NTU_req * C_gas
    
    # 3. Potência do Ventilador (W = K * UA^3)
    # Para 72 kg/h (UA ~ 75), resultará em aprox. 8-12W
    W_req = K_PERF * (UA_req**3)
    W_fan_final = max(W_min, W_req)
    
    # 4. Recálculo da temperatura de saída real (Acoplamento Físico)
    UA_real = (W_fan_final / K_PERF)**(1/3)
    NTU_real = UA_real / C_gas
    eff_real = 1 - np.exp(-NTU_real)
    T_out_real = T_gas_in - eff_real * (T_gas_in - T_AMB)
    
    return T_out_real, W_fan_final