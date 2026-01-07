# modelo_boiler.py
import pandas as pd

def simular_boiler_fluxo_continuo(m_dot_gas, cp_gas, T_in, T_out_target, eficiencia=0.99):
    """
    Simula o aquecimento de um fluxo de gás em linha (Boiler Elétrico / Aquecedor).
    Q = m_dot * cp * delta_T
    """
    delta_T = T_out_target - T_in
    
    if delta_T <= 0:
        return {
            'W_dot_kW': 0.0,
            'T_out_C': T_in,
            'Q_termico_kW': 0.0
        }
    
    # Q = m * cp * dT
    # m_dot em kg/s, cp em kJ/kg.K -> Q em kW
    q_termico = m_dot_gas * cp_gas * delta_T
    potencia_eletrica = q_termico / eficiencia
    
    return {
        'W_dot_kW': potencia_eletrica,
        'T_out_C': T_out_target,
        'Q_termico_kW': q_termico
    }

def simular_boiler_eletrico(massa_agua_kg, temp_inicial_celsius, temp_final_celsius, potencia_watts, eficiencia=0.99, preco_kwh_brl=0.85):
    # Mantida a função original para compatibilidade, embora não usada no fluxo contínuo
    Cp_agua = 4186 
    delta_T = temp_final_celsius - temp_inicial_celsius
    if delta_T <= 0:
        return 0, 0, 0, pd.DataFrame({"Resultado": ["Temperatura final deve ser maior que a inicial."]})

    Q_necessaria_joules = massa_agua_kg * Cp_agua * delta_T
    E_eletrica_joules = Q_necessaria_joules / eficiencia
    tempo_segundos = E_eletrica_joules / potencia_watts
    E_eletrica_kwh = E_eletrica_joules / 3.6e6
    custo_total_brl = E_eletrica_kwh * preco_kwh_brl

    dados = {
        "Parâmetro": ["Massa", "Temp. Inicial", "Temp. Final", "Potência", "Consumo", "Custo"],
        "Valor": [f"{massa_agua_kg}kg", f"{temp_inicial_celsius}°C", f"{temp_final_celsius}°C", 
                  f"{potencia_watts}W", f"{E_eletrica_kwh:.4f}kWh", f"R$ {custo_total_brl:.2f}"]
    }
    return tempo_segundos, E_eletrica_kwh, custo_total_brl, pd.DataFrame(dados)