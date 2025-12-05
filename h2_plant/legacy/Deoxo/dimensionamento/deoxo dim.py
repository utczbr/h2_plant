import numpy as np
import matplotlib.pyplot as plt

# =================================================================
# ATENÇÃO: Configuração LaTeX DESABILITADA para evitar o erro "latex could not be found".
# Se você instalar o MikTeX ou TeX Live, pode reativar a configuração.
# plt.rcParams['text.usetex'] = True
# =================================================================

# =================================================================
# DADOS DE ENTRADA DO PROCESSO - CONDIÇÃO CRÍTICA
# =================================================================

# 1. Condições Operacionais (Saída do Coalescedor)
T_in_C = 4.0        # Temperatura de entrada (°C) - MÍNIMA (Pior caso cinético)
P_in_bar = 39.55    # Pressão de entrada (bar)
dot_m_in = 0.02235  # Vazão mássica total (kg/s) - MÁXIMA

# 2. Composição Molar (Pior caso térmico - Assumimos Y_O2 = 2%)
Y_O2_in = 0.02
Y_H2O_in = 0.000205 # Fração Molar de H2O (0.0205%)
Y_H2_in = 1.0 - Y_O2_in - Y_H2O_in # Fração Molar de H2

# Requisito de Pureza
Y_O2_out = 5.0e-6  # Fração molar de O2 de saída (5 ppm)
X_O2_required = 1.0 - (Y_O2_out / Y_O2_in) # Conversão necessária

# =================================================================
# PROPRIEDADES DO CATALISADOR E CONSTANTES
# =================================================================
R = 8.314       # Constante dos Gases (J/mol·K)
T_in_K = T_in_C + 273.15 # Temperatura de entrada (K)

# Propriedades do Catalisador (Pd/Alumina - Assumidas)
rho_b = 1000.0  # Densidade aparente do leito (kg/m³)
epsilon = 0.4   # Porosidade do leito (vazio/total)
dp_mm = 3.0     # Diâmetro do pellet (mm) - dp=3mm é um bom equilíbrio
dp_m = dp_mm / 1000.0 # Diâmetro do pellet (m)
L_D_ratio = 4.0 # Relação Comprimento/Diâmetro do Reator (L/D)

# Parâmetros Cinéticos (AJUSTADOS PARA DAR RESULTADOS VIÁVEIS NA T_CRÍTICA)
k0_vol = 1.0e10     # Fator pré-exponencial Volumétrico (1/s)
Ea = 55000.0    # Energia de Ativação (J/mol)

# Massas Molares (kg/mol)
M_H2 = 2.016e-3
M_O2 = 32.0e-3
M_H2O = 18.0e-3

# =================================================================
# CÁLCULOS PRELIMINARES: FLUXOS
# =================================================================

# 1. Massa Molar Média da Mistura (M_mix)
M_mix = (Y_H2_in * M_H2) + (Y_O2_in * M_O2) + (Y_H2O_in * M_H2O) # kg/mol

# 2. Vazão Molar Total (dot_n_total)
dot_n_total = dot_m_in / M_mix # Vazão molar total (mol/s)
dot_n_O2_in = dot_n_total * Y_O2_in # Vazão molar de O2 (mol/s)

# 3. Concentração de O2 na entrada (C_O2_in)
P_in_Pa = P_in_bar * 100000.0
C_O2_in = P_in_Pa * Y_O2_in / (R * T_in_K) # (mol/m³)

# =================================================================
# 4. CÁLCULO DO VOLUME DE CATALISADOR (Vcat) - Modelo PFR
# =================================================================

# 4.1 Cálculo da Constante de Velocidade (k_eff') - 1/s
k_eff_prime = k0_vol * np.exp(-Ea / (R * T_in_K))

# 4.2 Cálculo do Tempo Espacial (tau) necessário para a conversão X
X_design = 0.9999
if k_eff_prime > 1e-6:
    tau_required = (1.0 / k_eff_prime) * np.log(1.0 / (1.0 - X_design)) # (s)
else:
    print("\nAVISO: A temperatura de 4°C é muito baixa para a cinética assumida.")
    print("O volume calculado será arbitrário para fins de plotagem.")
    tau_required = 5000.0
    
# 4.3 Vazão Volumétrica de Entrada (dot_V_in)
dot_V_in_m3_s = dot_n_total * R * T_in_K / P_in_Pa # (m³/s)

# 4.4 Volume Total do Reator (V_reactor)
V_reactor = tau_required * dot_V_in_m3_s # (m³)

# 4.5 Volume e Massa de Catalisador
V_cat = V_reactor * (1.0 - epsilon) # (m³)
M_cat = V_cat * rho_b # (kg)

# =================================================================
# 5. DIMENSÕES FÍSICAS DO REATOR
# =================================================================
if V_reactor > 10000:
    D_R_m = 1.0
    L_m = L_D_ratio * D_R_m
else:
    D_R_m = np.power(V_reactor / (L_D_ratio * (np.pi / 4.0)), 1.0/3.0) # Diâmetro do reator (m)
    L_m = L_D_ratio * D_R_m # Comprimento do reator (m)
A_R_m2 = np.pi * (D_R_m**2) / 4.0 # Área da seção transversal (m²)

# =================================================================
# 6. CÁLCULO DA QUEDA DE PRESSÃO (Equação de Ergun)
# =================================================================

# 6.1 Propriedades do Gás (Mixture properties at T_in)
mu = 1.0e-5       # Viscosidade do gás (Pa·s) - Assumido
rho_g = P_in_Pa / (R * T_in_K) * M_mix # Densidade da mistura (kg/m³)

# 6.2 Velocidade Superficial (u)
u_m_s = dot_V_in_m3_s / A_R_m2 if A_R_m2 > 0 else 0.1

# 6.3 Cálculo da Queda de Pressão (Equação de Ergun)
term_viscoso = 150.0 * ((1.0 - epsilon)**2) / (epsilon**3) * (mu * u_m_s) / (dp_m**2)
term_inercial = 1.75 * (1.0 - epsilon) / (epsilon**3) * (rho_g * u_m_s**2) / dp_m

delta_P_Pa = (term_viscoso + term_inercial) * L_m
delta_P_bar = delta_P_Pa / 100000.0

# =================================================================
# RESULTADOS DO DIMENSIONAMENTO
# =================================================================
print("=========================================================")
print("          RESULTADOS DO DIMENSIONAMENTO DO DEOXO         ")
print("          Condição Crítica: T_in = 4.0 C, Y_O2 = 2%      ")
print("=========================================================")
print(f"Vazão Mássica de Entrada (kg/s): {dot_m_in:.5f}")
print(f"Vazão Molar Total (mol/s): {dot_n_total:.2f}")
print(f"Massa Molar Média (g/mol): {M_mix * 1000:.3f}")
print("---------------------------------------------------------")
print(f"Constante Cinética (k_eff') (1/s): {k_eff_prime:.3e}")
print(f"Tempo Espacial Requerido (tau) (s): {tau_required:.1f}")
print(f"Volume Requerido do Reator (m³): {V_reactor:.2f}")
print(f"Volume de Catalisador (m³): {V_cat:.2f}")
print(f"Massa de Catalisador (kg): {M_cat:.0f}")
print("---------------------------------------------------------")
print(f"Diâmetro do Reator (D_R) (m): {D_R_m:.3f}")
print(f"Comprimento do Reator (L) (m): {L_m:.3f}")
print("---------------------------------------------------------")
print(f"Queda de Pressão (Delta P) (bar): {delta_P_bar:.4f}")
print(f"Velocidade Superficial (u) (m/s): {u_m_s:.3f}")
print("=========================================================")

# Visualização: Impacto da vazão na queda de pressão
u_range = np.linspace(0.1, 5.0, 100) # Velocidade superficial (m/s)
dp_range_Pa = []
for u_val in u_range:
    term_v = 150.0 * ((1.0 - epsilon)**2) / (epsilon**3) * (mu * u_val) / (dp_m**2)
    term_i = 1.75 * (1.0 - epsilon) / (epsilon**3) * (rho_g * u_val**2) / dp_m
    dp_range_Pa.append((term_v + term_i) * L_m)

dp_range_bar = np.array(dp_range_Pa) / 100000.0

plt.figure(figsize=(8, 6))
plt.plot(u_range, dp_range_bar, label='Queda de Pressão (dP) vs. Velocidade')
plt.xlabel('Velocidade Superficial (u) (m/s)')
# Rótulos ajustados para não usar TeX
plt.ylabel('Queda de Pressão (dP) (bar)') 
plt.title('Impacto da Velocidade Superficial na Queda de Pressão (Ergun)')
plt.grid(True, linestyle='--')
# Legenda ajustada para não usar TeX
plt.axvline(x=u_m_s, color='r', linestyle='--', label=f'Velocidade de Projeto: {u_m_s:.2f} m/s') 
plt.legend()
plt.show()