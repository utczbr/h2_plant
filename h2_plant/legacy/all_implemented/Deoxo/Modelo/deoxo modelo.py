import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# =================================================================
# ATENÇÃO: Configuração LaTeX DESABILITADA para evitar o erro "latex could not be found".
# Removido: plt.rcParams['text.usetex'] = True
# =================================================================

# =================================================================
# DADOS DE ENTRADA DO PROCESSO - CONDICAO CRITICA
# (Valores de entrada e resultados do dimensionamento anterior)
# =================================================================
# DADOS DE ENTRADA DO PROCESSO
T_in_C = 4.0      # Temperatura de entrada (C) - MÍNIMA (Pior caso cinético)
P_in_bar = 39.55  # Pressão de entrada (bar)
dot_n_total = 8.53  # Vazão molar total (mol/s) - Resultado do dimensionamento
Y_O2_in = 0.02    # Fração molar de O2 na entrada (2% - Pior caso térmico)
Y_H2O_in = 0.000205 # Fração molar de H2O na entrada

# RESULTADOS DO DIMENSIONAMENTO
# k_eff_prime_in = 0.4303 # Não utilizado diretamente, mas sim os parâmetros (k0_vol, Ea)
V_reactor = 0.11        # Volume total do reator (m³)
L_m = 1.294             # Comprimento do Reator (m)
D_R_m = 0.324           # Diâmetro do Reator (m)
A_R_m2 = np.pi * (D_R_m**2) / 4.0 # Área da seção transversal (m²)

# Propriedades do Catalisador
R = 8.314       # Constante dos Gases (J/mol·K)

# Parâmetros Cinéticos (AJUSTADOS - Utilizados para modelar a dependência com T)
k0_vol = 1.0e10     # Fator pré-exponencial Volumétrico (1/s)
Ea = 55000.0    # Energia de Ativação (J/mol)

# PARÂMETROS TERMODINÂMICOS
Delta_H_rxn = -242000.0 # Entalpia de Reação (J/mol de O2 consumido) - Exotérmica

# Capacidade Calorífica Molar Média (J/mol·K)
# Cp_mix = Σ Yi * Cpi
Cp_mix = (Y_O2_in * 29.0) + ((1 - Y_O2_in - Y_H2O_in) * 29.0) + (Y_H2O_in * 36.0) # J/mol·K

# PARÂMETROS DE CONTROLE TÉRMICO (CAMISA DE RESFRIAMENTO)
T_jacket_C = 120.0 # Temperatura da água de resfriamento na camisa (C)
U_a = 5000.0     # Coeficiente de transferência de calor por volume de reator (W/m³·K)

# Constantes de Temperatura
T_in_K = T_in_C + 273.15
T_jacket_K = T_jacket_C + 273.15

# Conversão alvo para pureza (5 ppm de O2)
Y_O2_out_target = 5.0e-6
conversion_target = 1.0 - (Y_O2_out_target / Y_O2_in)

# =================================================================
# DEFINIÇÃO DAS EQUAÇÕES DIFERENCIAIS (Balanço de Massa e Energia PFR)
# =================================================================
# Variáveis: P[0] = X (Conversão de O2), P[1] = T (Temperatura em K)

def pfr_ode(L, P):
    """
    Sistema de Equações Diferenciais Ordinárias (ODEs) para o PFR.
    L: Posição axial (m)
    P[0]: X (Conversão de O2)
    P[1]: T (Temperatura em K)
    """
    X = P[0]
    T = P[1]
    
    T_K = T
    P_Pa = P_in_bar * 100000.0 
    
    # 1. Checa a Condição de Parada e Conversão Máxima
    current_X = max(0.0, min(1.0, X)) # Garante que X esteja entre 0 e 1
    
    if current_X >= conversion_target or current_X >= 1.0:
         r_O2 = 0.0 # Zera a taxa de reação se a pureza for atingida ou X=1
         dX_dL = 0.0
    else:
        # 1.1 Taxa de Reação (r_O2): mol de O2 / m³ de reator / s
        k_eff_prime = k0_vol * np.exp(-Ea / (R * T_K))
        
        # Concentração de O2
        Y_O2 = Y_O2_in * (1.0 - current_X)
        C_O2 = P_Pa * Y_O2 / (R * T_K)
        
        r_O2 = k_eff_prime * C_O2 # (mol O2 / m³ reator / s)

        # 1.2 Balanço de Massa (dX/dL)
        F_O2_in = dot_n_total * Y_O2_in
        dX_dL = (A_R_m2 / F_O2_in) * r_O2
        
    # 2. Balanço de Energia (dT/dL)
    
    heat_generated = (-Delta_H_rxn) * r_O2 # W/m³
    heat_removed = U_a * (T_K - T_jacket_K) # W/m³
    
    # Evita que a temperatura caia abaixo de T_jacket se não houver reação
    if T_K < T_jacket_K and heat_generated < 1e-6:
        heat_removed = 0.0
    
    dT_dL = (A_R_m2 / (dot_n_total * Cp_mix)) * (heat_generated - heat_removed)
    
    return [dX_dL, dT_dL]

# =================================================================
# SOLUÇÃO DO MODELO
# =================================================================
# Condições Iniciais: X=0, T=T_in
X0 = 0.0
T0 = T_in_K

# Vetor de posições axiais (0 a L)
L_span = np.linspace(0, L_m, 100)

# Solução do sistema de ODEs
sol = solve_ivp(pfr_ode, [0, L_m], [X0, T0], t_eval=L_span)

X_profile = sol.y[0]
T_profile_K = sol.y[1]
T_profile_C = T_profile_K - 273.15

# Fixa a conversão máxima em 1.0 para o output
X_profile = np.clip(X_profile, 0.0, 1.0) 

# =================================================================
# CÁLCULO DA QUEDA DE TEMPERATURA ADIABÁTICA MÁXIMA
# =================================================================
X_max = 1.0
Delta_T_ad_K = (X_max * Y_O2_in * np.abs(Delta_H_rxn)) / Cp_mix

# =================================================================
# RESULTADOS DA MODELAGEM E GRÁFICOS
# =================================================================
print("=========================================================")
print("          RESULTADOS DA MODELAGEM TERMODINÂMICA          ")
print(f"          Condição de Entrada: T_in = {T_in_C} C, Y_O2 = {Y_O2_in * 100}%")
print("=========================================================")
print(f"Temperatura Máxima Adiabática Estimada (∆T_ad) (K): {Delta_T_ad_K:.1f} K")
print(f"Temperatura Máxima Alcançada no Reator (C): {np.max(T_profile_C):.1f} C")
print(f"Temperatura de Saída (T_out) (C): {T_profile_C[-1]:.1f} C")
# O erro SyntaxWarning é resolvido com a remoção da contra-barra
print(f"Conversão Final de O2 (%): {X_profile[-1] * 100:.6f}%") 
print("---------------------------------------------------------")
print(f"Temperatura da Camisa de Resfriamento (T_jacket) (C): {T_jacket_C:.1f} C")
print(f"Coeficiente de Transferência Volumétrico (Ua) (W/m³.K): {U_a:.0f}")
print("=========================================================")

# Gráfico do Perfil de Temperatura e Conversão
fig, ax1 = plt.subplots(figsize=(10, 6))

# Perfil de Conversão
# Rótulos ajustados para não usar TeX
ax1.set_xlabel('Comprimento do Reator (L) (m)')
ax1.set_ylabel('Conversão de O2 (X)', color='tab:blue')
ax1.plot(sol.t, X_profile, color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.set_ylim(0, 1.1)

# Perfil de Temperatura
ax2 = ax1.twinx()
# Rótulos ajustados para não usar TeX
ax2.set_ylabel('Temperatura (T) (C)', color='tab:red') 
ax2.plot(sol.t, T_profile_C, color='tab:red')

# Linhas de referência de temperatura
# Legendas ajustadas para não usar TeX
ax2.axhline(T_in_C, color='k', linestyle=':', label=f'T_in = {T_in_C:.0f} C')
ax2.axhline(np.max(T_profile_C), color='r', linestyle='--', label=f'T_max = {np.max(T_profile_C):.1f} C')
ax2.axhline(T_jacket_C, color='g', linestyle='-.', label=f'T_jacket = {T_jacket_C:.0f} C')
ax2.tick_params(axis='y', labelcolor='tab:red')

plt.title('Perfil de Temperatura e Conversão no Reator Deoxo (Com Resfriamento)')
fig.tight_layout()
plt.legend(loc='lower right')
plt.grid(True, linestyle='--')
plt.show()