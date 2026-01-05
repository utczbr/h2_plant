# modelo_boiler.py
import pandas as pd
import CoolProp.CoolProp as CP # Adicionado CoolProp para a função de fluxo contínuo

# 🛑 REMOVIDA A FUNÇÃO aquecer_boiler_modular (A SIMPLIFICADA)

def simular_boiler_eletrico(massa_agua_kg, temp_inicial_celsius, temp_final_celsius, potencia_watts, eficiencia=0.99, preco_kwh_brl=0.85):
    """
    Simula o aquecimento de água por um boiler elétrico (Regime Batch).
    ... (Docstring e lógica original de simulação em regime batch)
    """
    # 1. Constantes
    Cp_agua = 4186 # J/kg°C

    # 2. Cálculo da Variação de Temperatura
    delta_T = temp_final_celsius - temp_inicial_celsius

    # Se a temperatura final for menor ou igual à inicial, não há aquecimento.
    if delta_T <= 0:
        return 0, 0, 0, pd.DataFrame({
            "Massa de Água (kg)": [massa_agua_kg],
            "Temp. Inicial (°C)": [temp_inicial_celsius],
            "Temp. Final (°C)": [temp_final_celsius],
            "Resultado": ["Temperatura final deve ser maior que a inicial."]
        })

    # 3. Energia Térmica Necessária (Q)
    # Q = m * Cp * ΔT (Joules)
    Q_necessaria_joules = massa_agua_kg * Cp_agua * delta_T

    # 4. Energia Elétrica Total Requerida
    E_eletrica_joules = Q_necessaria_joules / eficiencia

    # 5. Tempo de Aquecimento
    tempo_segundos = E_eletrica_joules / potencia_watts
    tempo_minutos = tempo_segundos / 60
    tempo_horas = tempo_segundos / 3600

    # 6. Conversão de Energia e Cálculo de Custo
    E_eletrica_kwh = E_eletrica_joules / 3.6e6
    custo_total_brl = E_eletrica_kwh * preco_kwh_brl

    # 7. Organização da Saída em Tabela (DataFrame)
    dados = {
        "Parâmetro": [
            "Massa de Água",
            "Temp. Inicial",
            "Temp. Final",
            "Eficiência do Boiler",
            "Potência do Boiler",
            "Energia Térmica Necessária (Q)",
            "Energia Elétrica Requerida (E_eletrica)",
            "Tempo de Aquecimento",
            "Consumo de Energia (E_eletrica)",
            "Custo Total da Eletricidade",
            "Preço por kWh"
        ],
        "Valor": [
            f"{massa_agua_kg:.2f} kg",
            f"{temp_inicial_celsius:.2f} °C",
            f"{temp_final_celsius:.2f} °C",
            f"{eficiencia*100:.1f} %",
            f"{potencia_watts:.0f} W",
            f"{Q_necessaria_joules:.2f} J",
            f"{E_eletrica_joules:.2f} J",
            f"{tempo_segundos:.2f} s ({tempo_minutos:.2f} min)",
            f"{E_eletrica_kwh:.4f} kWh",
            f"R$ {custo_total_brl:.2f}",
            f"R$ {preco_kwh_brl:.2f}"
        ]
    }
    tabela_resultados = pd.DataFrame(dados)

    return tempo_segundos, E_eletrica_kwh, custo_brl, tabela_resultados

# --- Exemplo de Uso (Mantido para compatibilidade com o arquivo original) ---
MASSA = 5.0      
T_INICIAL = 20.0 
T_FINAL = 80.0   
POTENCIA = 3000  
EFICIENCIA = 0.99 
PRECO_KWH = 0.90 

tempo_s, energia_kwh, custo_brl, resultados_tabela = simular_boiler_eletrico(
    MASSA,
    T_INICIAL,
    T_FINAL,
    POTENCIA,
    EFICIENCIA,
    PRECO_KWH
)

if tempo_s > 0:
    print("## 📊 Resultados da Simulação do Boiler Elétrico")
    print(resultados_tabela.to_string(index=False, col_space=20))
else:
    print(resultados_tabela["Resultado"].iloc[0])