import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from numpy.polynomial import polynomial as P

# ==============================================================================
# 1. DEFINIÇÃO DA FÍSICA (Comum aos dois arquivos)
# ==============================================================================

# Constantes
F = 96485.33
R = 8.314
T = 333.15      # 60°C
P_op = 40.0e5   # 40 bar
P_ref = 1.0e5
z = 2
MH2 = 2.016e-3

# Geometria e Parâmetros
N_stacks = 35
N_cell_per_stack = 645
A_cell = 300
Area_Total = N_stacks * N_cell_per_stack * A_cell
j_nom = 2.91
j_lim = 4.0

# Parâmetros Eletroquímicos
delta_mem = 100 * 1e-4
sigma_base = 0.1
j0 = 1.0e-6
alpha = 0.5
floss = 0.02

# BoP
P_nominal_sistema_kW = 5000
P_nominal_sistema_W = P_nominal_sistema_kW * 1000
P_bop_fixo = 0.025 * P_nominal_sistema_W
k_bop_var = 0.04

def calculate_Vcell(j):
    # Simplificação: Degradação = 0 para a comparação de métodos
    U_rev = 1.229 - 0.9e-3 * (T - 298.15) + (R * T) / (z * F) * np.log((P_op / P_ref)**1.5)
    eta_act = (R * T) / (alpha * z * F) * np.log(np.maximum(j, 1e-10) / j0)
    eta_ohm = j * (delta_mem / sigma_base)
    eta_conc = np.where(j >= j_lim, 100.0, (R * T) / (z * F) * np.log(j_lim / (j_lim - np.maximum(j, 1e-10))))
    return U_rev + eta_act + eta_ohm + eta_conc

def calculate_P_input(j):
    I_total = j * Area_Total
    V_cell = calculate_Vcell(j)
    P_stack = I_total * V_cell
    P_BoP = P_bop_fixo + k_bop_var * P_stack
    return P_stack + P_BoP

def calculate_H2_production(j):
    eta_F = np.maximum(j, 1e-6)**2 / (np.maximum(j, 1e-6)**2 + floss)
    I_total = j * Area_Total
    return (I_total * eta_F * MH2) / (z * F) * 60.0 # kg/min

# ==============================================================================
# 2. PREPARAÇÃO DO "SIMULADOR" (ABORDAGEM PIECEWISE / POR PARTES)
# ==============================================================================
print("--- [PREPARAÇÃO] Gerando Polinômio por Partes (Piecewise) ---")

# 1. Encontrar j máximo (igual ao original)
def find_j_for_power(target_power):
    j_guess = target_power / (Area_Total * 2.0)
    for _ in range(10): # Aumentei iterações levemente para garantir precisão
        P_calc = calculate_P_input(j_guess)
        error = target_power - P_calc
        j_guess += error / (Area_Total * 2.0)
    return j_guess

P_max_training = P_nominal_sistema_W * 1.2
j_max_training = find_j_for_power(P_max_training)

# 2. Gerar pontos mais densos (500 pontos garante melhor captura da curva)
j_train_raw = np.linspace(0.001, j_max_training, 500)
P_train_raw = np.array([calculate_P_input(j) for j in j_train_raw])

# 3. FILTRAGEM
valid_indices = P_train_raw > (P_bop_fixo * 1.01)
j_train = j_train_raw[valid_indices]
P_train = P_train_raw[valid_indices]

# 4. DEFINIÇÃO DAS REGIÕES (SPLIT)
# O "cotovelo" da curva geralmente está entre 15% e 25% da potência nominal.
P_split = P_nominal_sistema_W * 0.25 

mask_low = P_train <= P_split
mask_high = P_train > P_split

# 5. AJUSTE DOS POLINÔMIOS
# Região Baixa: Grau 5 para capturar a não-linearidade inicial
poly_coeffs_low = np.polyfit(P_train[mask_low], j_train[mask_low], 5)
model_low = np.poly1d(poly_coeffs_low)

# Região Alta: Grau 4 (ou até 3) é suficiente pois é quase linear
poly_coeffs_high = np.polyfit(P_train[mask_high], j_train[mask_high], 4)
model_high = np.poly1d(poly_coeffs_high)

print(f"   > Polinômio Low treinado com {np.sum(mask_low)} pontos.")
print(f"   > Polinômio High treinado com {np.sum(mask_high)} pontos.")

# 6. FUNÇÃO MODELO UNIFICADA
# Substitui o objeto np.poly1d simples por uma função que decide qual usar
def polynomial_model(P_input):
    # Se for um valor único (escalar)
    if np.ndim(P_input) == 0:
        if P_input <= P_split:
            return model_low(P_input)
        else:
            return model_high(P_input)
    
    # Se for um array (vetorizado para velocidade)
    else:
        return np.where(P_input <= P_split, model_low(P_input), model_high(P_input))

# ==============================================================================
# 3. COMPARAÇÃO DE DESEMPENHO
# ==============================================================================

# Cenário de Teste: 1000 pontos de operação aleatórios (1 dia simulado minuto a minuto seria 1440)
N_SAMPLES = 1000
P_targets_W = np.random.uniform(P_bop_fixo * 1.1, P_nominal_sistema_W, N_SAMPLES)

print(f"\n--- [TESTE] Rodando {N_SAMPLES} simulações ---")

# --- MÉTODO A: OPERATOR (Solver Numérico / White-Box) ---
start_time_op = time.time()
results_operator_H2 = []

for P_target in P_targets_W:
    # Função de erro para o solver
    def func_err(j):
        return calculate_P_input(j) - P_target
    
    # Solver (fsolve)
    j_guess = j_nom * (P_target / P_nominal_sistema_W)
    j_sol = fsolve(func_err, j_guess, xtol=1e-4)[0]
    
    results_operator_H2.append(calculate_H2_production(j_sol))

time_operator = time.time() - start_time_op


# --- MÉTODO B: SIMULATOR (Polinomial) ---
start_time_sim = time.time()
results_simulator_H2 = []

for P_target in P_targets_W:
    # Cálculo Direto via Polinômio
    j_poly = polynomial_model(P_target)
    
    # Simples clamp para segurança
    j_poly = np.clip(j_poly, 0, j_lim)
    
    results_simulator_H2.append(calculate_H2_production(j_poly))

time_sim = time.time() - start_time_sim

# ==============================================================================
# 4. ANÁLISE DOS RESULTADOS
# ==============================================================================

results_operator_H2 = np.array(results_operator_H2)
results_simulator_H2 = np.array(results_simulator_H2)

# Erro relativo (%)
errors = np.abs((results_simulator_H2 - results_operator_H2) / results_operator_H2) * 100
max_error = np.max(errors)
avg_error = np.mean(errors)

print("\n" + "="*50)
print("RESULTADOS DA COMPARAÇÃO")
print("="*50)
print(f"1. TEMPO DE EXECUÇÃO ({N_SAMPLES} passos):")
print(f"   - Operator (Solver):   {time_operator:.4f} s")
print(f"   - Simulator (Poly):    {time_sim:.4f} s")
print(f"   > Fator de Speedup:    {time_operator/time_sim:.1f}x mais rápido")

print(f"\n2. PRECISÃO (H2 Produzido):")
print(f"   - Erro Médio:          {avg_error:.6f} %")
print(f"   - Erro Máximo:         {max_error:.6f} %")

# Plotagem
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.title(f'Comparação de Produção H2 (Amostra de 100 pontos)')
plt.plot(results_operator_H2[:100], 'k-', linewidth=2, label='Operator (Exato/Solver)')
plt.plot(results_simulator_H2[:100], 'r--', label='Simulator (Aprox/Poly)')
plt.ylabel('H2 Produzido (kg/min)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.title('Erro Relativo do Método Polinomial')
plt.plot(errors, 'b.', markersize=2)
plt.xlabel('Iteração')
plt.ylabel('Erro (%)')
plt.grid(True, alpha=0.3)
plt.tight_layout()

print("\nGerando gráfico de erro...")
plt.show()