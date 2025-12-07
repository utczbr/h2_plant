import numpy as np
import pickle 
from numpy.polynomial import polynomial as P 
import warnings
from scipy.interpolate import interp1d

# Ignorar warnings de otimização (RankWarning) se houverem
warnings.filterwarnings("ignore") 

# ==========================================================
# 1. CONSTANTES FUNDAMENTAIS GLOBAIS
# ==========================================================
F = 96485.33    # Constante de Faraday
R = 8.314       
P_ref = 1.0e5   
z = 2           
MH2 = 2.016e-3  

# --- 2. CONFIGURAÇÃO E PARÂMETROS ---

P_nominal_sistema_kW = 5000 
P_nominal_sistema_W = P_nominal_sistema_kW * 1000
T = 333.15      
P_op = 40.0e5   

# --- 3. PARÂMETROS DO MODELO MECANICISTA ---

# A. Geometria 
N_stacks = 35           
N_cell_per_stack = 85   
A_cell = 300            
Area_Total = N_stacks * N_cell_per_stack * A_cell
j_nom = 2.91            

# B. Simulação
H_MES = 730.0           
H_SIM_YEARS = 4.5       
H_SIM_TOTAL_PRECALC = int(H_SIM_YEARS * 8760) 

# C. Físicos da Pilha
delta_mem = 100 * 1e-4  
sigma_base = 0.1        
j0 = 1.0e-6             
alpha = 0.5             
j_lim = 4.0             

# D. BoP e Limites
P_bop_fixo = 0.025 * P_nominal_sistema_W
floss = 0.02

# --- E. PARÂMETROS DE DEGRADAÇÃO ---
DEGRADATION_TABLE_YEARS = np.array([1.0, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 9.0, 10.0])
#DEGRADATION_TABLE_V_STACK = np.array([1290, 1300, 1325, 1340, 1345, 1355, 1365, 1380, 1390, 1410, 1435, 1460, 1490])
# Scaled for 85 cells/stack (5MW config) instead of 645 cells/stack (35MW original)
DEGRADATION_TABLE_V_STACK = np.array([171, 172, 176, 178, 178, 180, 181, 183, 184, 187, 190, 193, 197])
T_OP_H_TABLE = DEGRADATION_TABLE_YEARS * 8760.0 
V_CELL_TABLE = DEGRADATION_TABLE_V_STACK / N_cell_per_stack

# --- 4. FUNÇÕES DE SUPORTE ---

def calculate_Urev(T, P_op):
    U_rev_T = 1.229 - 0.9e-3 * (T - 298.15)
    pressure_ratio = P_op / P_ref
    Nernst_correction = (R * T) / (z * F) * np.log(pressure_ratio**1.5)
    return U_rev_T + Nernst_correction

def calculate_Vcell_base(j, T, P_op):
    U_rev = calculate_Urev(T, P_op)
    eta_act = (R * T) / (alpha * z * F) * np.log(np.maximum(j, 1e-10) / j0)
    eta_ohm = j * (delta_mem / sigma_base)
    eta_conc = np.where(j >= j_lim, 100.0, (R * T) / (z * F) * np.log(j_lim / (j_lim - np.maximum(j, 1e-10))))
    return U_rev + eta_act + eta_ohm + eta_conc
    
V_CELL_BOL_NOM = calculate_Vcell_base(j_nom, T, P_op)

v_cell_degraded_interpolator = interp1d(
    T_OP_H_TABLE, V_CELL_TABLE, kind='linear', 
    fill_value=(V_CELL_TABLE[0], V_CELL_TABLE[-1]), bounds_error=False
)

def calculate_U_deg_from_table(t_op_h):
    V_cell_degraded = v_cell_degraded_interpolator(t_op_h)
    return np.maximum(0.0, V_cell_degraded - V_CELL_BOL_NOM) 

def calculate_Vcell(j, T, P_op, t_op_h):
    U_rev = calculate_Urev(T, P_op)
    eta_act = (R * T) / (alpha * z * F) * np.log(np.maximum(j, 1e-10) / j0)
    eta_ohm = j * (delta_mem / sigma_base)
    eta_conc = np.where(j >= j_lim, 100.0, (R * T) / (z * F) * np.log(j_lim / (j_lim - np.maximum(j, 1e-10))))
    U_deg = calculate_U_deg_from_table(t_op_h) 
    return U_rev + eta_act + eta_ohm + eta_conc + U_deg

def P_input_system(j, T, P_op, t_op_h):
    I_total = j * Area_Total
    V_cell = calculate_Vcell(j, T, P_op, t_op_h)
    P_stack = I_total * V_cell
    P_BoP = P_bop_fixo + 0.04 * P_stack
    return P_stack + P_BoP

# ==========================================================
# 5. EXECUÇÃO DO PRÉ-CÁLCULO (PIECEWISE)
# ==========================================================

print(f"Iniciando Pré-cálculo Piecewise de {H_SIM_YEARS} Anos...")

num_meses = int(H_SIM_TOTAL_PRECALC / H_MES)
piecewise_models_list = []  # Lista para armazenar dicionários

# Aumentei para 2000 pontos para garantir precisão na região baixa
j_char_points = 2000 
j_char_range = np.linspace(0.001 * j_nom, 1.3 * j_nom, j_char_points)

# Definição do Ponto de Corte (Split)
# 20% da Potência Nominal é um ponto seguro para separar a não-linearidade
P_SPLIT_VALUE = 0.20 * P_nominal_sistema_W 

for mes in range(num_meses):
    t_op_h_mes = mes * H_MES
    
    # 1. Gerar curva real para este mês (Degradação inclusa)
    P_input_char_W = np.array([P_input_system(j, T, P_op, t_op_h_mes) for j in j_char_range])
    
    # 2. Filtrar dados inválidos (abaixo do consumo do BoP)
    valid_indices = P_input_char_W > (P_bop_fixo * 1.001)
    P_valid = P_input_char_W[valid_indices]
    j_valid = j_char_range[valid_indices]

    # 3. Separar as regiões (Low vs High)
    mask_low = P_valid <= P_SPLIT_VALUE
    mask_high = P_valid > P_SPLIT_VALUE

    # 4. Ajustar Polinômios Independentes
    # Região Baixa: Grau 5 para capturar a curva logarítmica
    # Região Alta: Grau 4 para a parte quase linear
    if np.sum(mask_low) > 5: # Segurança para não fitar sem pontos
        coeffs_low = np.polyfit(P_valid[mask_low], j_valid[mask_low], 5)
        poly_low = np.poly1d(coeffs_low)
    else:
        # Fallback raro se não houver pontos na baixa (improvável com 600 pts)
        poly_low = np.poly1d([0]) 

    if np.sum(mask_high) > 4:
        coeffs_high = np.polyfit(P_valid[mask_high], j_valid[mask_high], 4)
        poly_high = np.poly1d(coeffs_high)
    else:
        poly_high = np.poly1d([0])

    # 5. Armazenar como Dicionário
    model_dict = {
        'mes': mes,
        't_op_h': t_op_h_mes,
        'split_point': P_SPLIT_VALUE,
        'poly_low': poly_low,
        'poly_high': poly_high
    }
    
    piecewise_models_list.append(model_dict)
    
    if mes % 12 == 0:
        U_deg = calculate_U_deg_from_table(t_op_h_mes)
        print(f"  > Mês {mes}: Degradação {U_deg:.4f}V | Split @ {P_SPLIT_VALUE/1e6:.2f}MW")

# --- 6. SALVAR DADOS ---

file_name = "degradation_polynomials.pkl"
try:
    with open(file_name, 'wb') as f:
        pickle.dump(piecewise_models_list, f)
    print(f"\n✅ Pré-cálculo PIECEWISE concluído! {len(piecewise_models_list)} modelos salvos.")
except Exception as e:
    print(f"\n❌ Erro ao salvar: {e}")