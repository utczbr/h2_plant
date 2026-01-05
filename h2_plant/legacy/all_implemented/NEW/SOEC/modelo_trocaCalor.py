import numpy as np

# --- Constantes e Suposições do Modelo ---
# As vazões e Cp's podem ser alterados para simular diferentes cenários.
# Cp's são aproximados para as faixas de temperatura mencionadas.

# Lado Frio (Água do Dreno)
T_c_e = 20.0  # [°C] Temperatura de entrada do fluido frio
T_c_s_alvo = 100.0  # [°C] Temperatura de saída ALVO do fluido frio
m_dot_c = 1.0  # [kg/s] Vazão mássica do fluido frio (EXEMPLO)
Cp_c = 4.183  # [kJ/(kg·K)] Calor específico da água líquida (média a 60°C)

# Lado Quente (Mistura H2/H2O)
T_h_e = 150.0  # [°C] Temperatura de entrada do fluido quente (Vapor Superaquecido)
m_dot_h = 0.5  # [kg/s] Vazão mássica do fluido quente (EXEMPLO)
Cp_h = 6.0  # [kJ/(kg·K)] Calor específico da mistura H2/H2O (Valor ESTIMADO)

# --- Cálculo Principal ---

# 1. Calor (Sensível) Absorvido pelo Fluido Frio (Q)
# (Delta T em °C é igual ao Delta T em K)
Delta_T_c = T_c_s_alvo - T_c_e

# Q [kW] = m_dot [kg/s] * Cp [kJ/(kg·K)] * Delta_T [K]
Q = m_dot_c * Cp_c * Delta_T_c

# 2. Variação de Temperatura no Fluido Quente (Delta_T_h)
# O calor cedido pelo fluido quente é igual ao calor absorvido pelo fluido frio.
# Delta_T_h = Q / (m_dot_h * Cp_h)

Delta_T_h = Q / (m_dot_h * Cp_h)

# 3. Temperatura de Saída do Fluido Quente (T_h_s)
# T_h_s = T_h_e - Delta_T_h

T_h_s = T_h_e - Delta_T_h

# --- Verificação de Viabilidade (Termodinâmica) ---
# A temperatura de saída do fluido quente (T_h_s) NUNCA pode ser menor
# que a temperatura de entrada do fluido frio (T_c_e) em um trocador de calor ideal.
# Para contrafluxo, T_h_s deve ser sempre > T_c_e.

viabilidade_termica = T_h_s >= T_c_e

# --- Organização da Saída no Terminal (Requerido pelo usuário) ---
print("="*60)
print("             RELATÓRIO DE BALANÇO DE ENERGIA")
print("="*60)

print("\n--- Parâmetros de Entrada ---")
print(f"Vazão Frio (m_dot_c): {m_dot_c: .2f} kg/s")
print(f"Temperatura Frio (T_c_e -> T_c_s): {T_c_e:.1f}°C -> {T_c_s_alvo:.1f}°C")
print(f"Vazão Quente (m_dot_h): {m_dot_h:.2f} kg/s")
print(f"Temperatura Quente (T_h_e): {T_h_e:.1f}°C")
print(f"Cp Frio (Cp_c): {Cp_c:.3f} kJ/(kg·K)")
print(f"Cp Quente (Cp_h): {Cp_h:.3f} kJ/(kg·K) (Estimado)")

print("\n--- Resultados do Balanço de Energia ---")
print(f"1. Taxa de Calor Requerida (Q): {Q:.2f} kW")
print(f"2. Queda de Temperatura do Fluido Quente (ΔT_h): {Delta_T_h:.2f} °C")
print(f"3. Temperatura de Saída do Fluido Quente (T_h_s): {T_h_s:.2f} °C")

print("\n--- Conclusão da Análise Termodinâmica ---")
if viabilidade_termica:
    print(f"VIÁVEL: T_h_s ({T_h_s:.2f}°C) é maior que T_c_e ({T_c_e:.1f}°C).")
else:
    print(f"INVIÁVEL: T_h_s ({T_h_s:.2f}°C) é menor que T_c_e ({T_c_e:.1f}°C). O trocador de calor não atinge o objetivo com estas vazões/Cp's.")

print("="*60)