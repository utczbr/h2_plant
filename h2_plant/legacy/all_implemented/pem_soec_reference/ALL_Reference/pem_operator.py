import numpy as np
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
import warnings
import pickle
import os

# Filtrar avisos de runtime para manter o console limpo
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==========================================================
# 1. CONSTANTES FUNDAMENTAIS E PARÂMETROS DO SISTEMA
# ==========================================================

F = 96485.33     # Faraday Constant (C/mol)
R = 8.314        # Gas Constant (J/(mol K))
P_ref = 1.0e5    # Pressão de referência (Pa)
z = 2            # Elétrons por molécula de H2
MH2 = 2.016e-3   # Massa Molar H2 (kg/mol)
MO2 = 31.998e-3  # Massa Molar O2 (kg/mol)
MH2O = 18.015e-3 # Massa Molar H2O (kg/mol)

# --- Configuração do Stack PEM ---
N_stacks = 35
N_cell_per_stack = 85
A_cell = 300     # cm²
Area_Total = N_stacks * N_cell_per_stack * A_cell # cm²
j_nom = 2.91     # A/cm²

# --- Variáveis Operacionais ---
T = 333.15       # 60°C em Kelvin
P_op = 40.0e5    # 40 bar em Pa

# --- Parâmetros de Física e Degradação ---
delta_mem = 100 * 1e-4 # cm
sigma_base = 0.1       # S/cm
j0 = 1.0e-6            # A/cm²
alpha = 0.5
j_lim = 4.0            # A/cm²

# --- Balance of Plant (BoP) ---
floss = 0.02
P_nominal_sistema_kW = 5000 # 5 MW
P_nominal_sistema_W = P_nominal_sistema_kW * 1000
P_bop_fixo = 0.025 * P_nominal_sistema_W
k_bop_var = 0.04

# --- Configuração de Simulação Temporal ---
H_MES = 730.0  # Horas em um mês operacional (para indexar o polinômio)

# ==========================================================
# 2. CARREGAMENTO DE DADOS DE DEGRADAÇÃO (PKL OU HARDCODED)
# ==========================================================

# Variável global para armazenar os polinômios
POLYNOMIAL_LIST = []
USE_POLYNOMIALS = False

# Tenta carregar o arquivo gerado pelo pre_calculator_pem_S.py
pkl_filename = "degradation_polynomials.pkl"

if os.path.exists(pkl_filename):
    try:
        with open(pkl_filename, 'rb') as f:
            POLYNOMIAL_LIST = pickle.load(f)
        USE_POLYNOMIALS = True
        print(f"   [PEM Operator] SUCESSO: Carregados {len(POLYNOMIAL_LIST)} polinômios de degradação de '{pkl_filename}'.")
    except Exception as e:
        print(f"   [PEM Operator] AVISO: Falha ao ler '{pkl_filename}'. Usando método analítico (mais lento). Erro: {e}")
        USE_POLYNOMIALS = False
else:
    print(f"   [PEM Operator] INFO: Arquivo '{pkl_filename}' não encontrado. Usando física White-Box padrão (Fallback).")
    USE_POLYNOMIALS = False


# --- Tabelas Hardcoded (Fallback caso o PKL falhe) ---
DEGRADATION_TABLE_YEARS = np.array([1.0, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 9.0, 10.0])
#DEGRADATION_TABLE_V_STACK = np.array([1290, 1300, 1325, 1340, 1345, 1355, 1365, 1380, 1390, 1410, 1435, 1460, 1490])
DEGRADATION_TABLE_V_STACK = np.array([171, 172, 176, 178, 178, 180, 181, 183, 184, 187, 190, 193, 197])
T_OP_H_TABLE = DEGRADATION_TABLE_YEARS * 8760.0
V_CELL_TABLE = DEGRADATION_TABLE_V_STACK / N_cell_per_stack

v_cell_degraded_interpolator = interp1d(
    T_OP_H_TABLE,
    V_CELL_TABLE,
    kind='linear',
    fill_value=(V_CELL_TABLE[0], V_CELL_TABLE[-1]),
    bounds_error=False
)

# ==========================================================
# 3. FUNÇÕES FÍSICAS (CORE PHYSICS - Usadas no Fallback ou Cálculos Auxiliares)
# ==========================================================

def calculate_Urev(T, P_op):
    """Tensão reversível (Nernst)."""
    U_rev_T = 1.229 - 0.9e-3 * (T - 298.15)
    pressure_ratio = P_op / P_ref
    Nernst_correction = (R * T) / (z * F) * np.log(pressure_ratio**1.5)
    return U_rev_T + Nernst_correction

def calculate_Vcell_base(j, T, P_op):
    """Calcula Vcell BOL (Início da vida)."""
    U_rev = calculate_Urev(T, P_op)
    eta_act = (R * T) / (alpha * z * F) * np.log(np.maximum(j, 1e-10) / j0)
    eta_ohm = j * (delta_mem / sigma_base)
    eta_conc = np.where(j >= j_lim, 100.0, (R * T) / (z * F) * np.log(j_lim / (j_lim - np.maximum(j, 1e-10))))
    return U_rev + eta_act + eta_ohm + eta_conc

V_CELL_BOL_NOM = calculate_Vcell_base(j_nom, T, P_op)

def calculate_U_deg_from_table(t_op_h):
    """Calcula a sobretensão de degradação (método fallback)."""
    V_cell_degraded = v_cell_degraded_interpolator(t_op_h)
    U_deg = np.maximum(0.0, V_cell_degraded - V_CELL_BOL_NOM)
    return U_deg

def calculate_Vcell(j, T, P_op, t_op_h):
    """Tensão da célula considerando física + degradação (método fallback)."""
    base_V = calculate_Vcell_base(j, T, P_op)
    U_deg = calculate_U_deg_from_table(t_op_h)
    return base_V + U_deg

def calculate_eta_F(j):
    """Eficiência de Faraday (perdas por crossover em baixa corrente)."""
    return np.maximum(j, 1e-6)**2 / (np.maximum(j, 1e-6)**2 + floss)

def P_input_system_from_j(j, T, P_op, t_op_h):
    """Calcula a potência de entrada (W) dado um j (A/cm²)."""
    I_total = j * Area_Total
    V_cell = calculate_Vcell(j, T, P_op, t_op_h)
    P_stack = I_total * V_cell
    P_BoP = P_bop_fixo + k_bop_var * P_stack
    return P_stack + P_BoP

# ==========================================================
# 4. CLASSE DE ESTADO E INTERFACE PARA O MANAGER
# ==========================================================

class PemState:
    """Classe para armazenar o estado da simulação PEM."""
    def __init__(self):
        self.t_op_h = 0.0  # Tempo de operação acumulado em horas
        self.H_EOL = T_OP_H_TABLE[-1] # Tempo de fim de vida estimado

def initialize_pem_simulation():
    """Inicializa e retorna o objeto de estado do PEM."""
    mode = "Polynomial (Fast)" if USE_POLYNOMIALS else "White-Box Solver (Detailed)"
    print(f"   [PEM Operator] Initialized in mode: {mode}")
    return PemState()

def run_pem_step(P_target_kW, current_state: PemState):
    """
    Executa um passo de tempo (1 minuto) do eletrolisador PEM.
    """
    P_target_W = P_target_kW * 1000.0
    dt_h = 1.0 / 60.0  # Passo de 1 minuto em horas
    
    # Se a potência for muito baixa (abaixo do mínimo do BoP ou zero), desliga
    if P_target_W < P_bop_fixo:
        return 0.0, 0.0, 0.0, current_state

    j_op = 0.0

    # -----------------------------------------------------------
    # MÉTODO A: POLINÔMIOS (Rápido - Se arquivo .pkl foi carregado)
    # -----------------------------------------------------------
    if USE_POLYNOMIALS:
        # Determina em qual mês de operação estamos
        month_index = int(current_state.t_op_h / H_MES)
        
        # Garante que o índice não exceda a lista disponível
        if month_index >= len(POLYNOMIAL_LIST):
            month_index = len(POLYNOMIAL_LIST) - 1
            
        # Carrega o objeto (pode ser um Polinômio único antigo ou um Dicionário novo)
        poly_object = POLYNOMIAL_LIST[month_index]
        
        # --- Verificação de Tipo ---
        if isinstance(poly_object, dict):
            # Lógica Piecewise (Novo Sistema)
            split_point = poly_object['split_point']
            poly_low = poly_object['poly_low']
            poly_high = poly_object['poly_high']
            
            if P_target_W <= split_point:
                j_op = poly_low(P_target_W)
            else:
                j_op = poly_high(P_target_W)
        else:
            # Lógica Legada (Caso carregue um pkl antigo com apenas 1 polinômio)
            j_op = poly_object(P_target_W)
        
    # -----------------------------------------------------------
    # MÉTODO B: SOLVER FÍSICO (Lento - Fallback / Alta precisão)
    # -----------------------------------------------------------
    else:
        # Encontrar a densidade de corrente (j) via fsolve
        def func_to_solve(j_guess):
            return P_input_system_from_j(j_guess, T, P_op, current_state.t_op_h) - P_target_W
        
        j_guess = j_nom * (P_target_W / P_nominal_sistema_W)
        j_sol, infodict, ier, msg = fsolve(func_to_solve, j_guess, full_output=True, xtol=1e-4)
        
        if ier != 1:
            j_op = j_guess
        else:
            j_op = j_sol[0]
            
    # -----------------------------------------------------------
    # PÓS-CÁLCULO E LIMITES
    # -----------------------------------------------------------
    
    # Clampar j dentro dos limites físicos
    j_op = np.clip(j_op, 0.0, j_lim)
    
    # Calcular fluxos com o j encontrado
    eta_F = calculate_eta_F(j_op)
    I_total = j_op * Area_Total
    
    # Fluxos instantâneos em kg/s
    m_H2_dot = (I_total * eta_F * MH2) / (z * F)
    m_O2_dot = (I_total * eta_F * (MO2 / 2.0)) / (z * F)
    m_H2O_dot = (I_total * eta_F * MH2O) / (z * F)
    
    # Converter para massa total no passo de 1 minuto (kg/min)
    m_H2_kg_step = m_H2_dot * 60.0
    m_O2_kg_step = m_O2_dot * 60.0
    m_H2O_kg_step = m_H2O_dot * 60.0
    
    # Atualizar o estado (envelhecimento)
    current_state.t_op_h += dt_h
    
    return m_H2_kg_step, m_O2_kg_step, m_H2O_kg_step, current_state